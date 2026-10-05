"""Differential tests for the vectorized (SIMD) pixel kernels.

Several inverse-transform kernels are widened to `simd_width_of[uint32]()` with
a scalar tail.  These tests drive each of them through `mw_kernel_run` and
compare against a scalar reference transcribed from upstream's C, at lengths
that straddle the lane count so both the vector body and the tail are covered,
and with poison either side of the run so a kernel that writes outside
`[0, n)` is caught.

The reference runs on Python ints and masks to 32 bits explicitly, which is what
the C `uint32_t` arithmetic does.
"""

from __future__ import annotations

import numpy as np
import pytest

from mojowebp._lib import lib

M32 = 0xFFFFFFFF

# mw_kernel_run kinds, mirroring src/mojowebp_capi.mojo
COPY, GREEN, CV_RGBA, CV_RGB, CV_BGR, PRED, XCFORM = range(7)

# Lengths that straddle the lane count: below it, exactly at it, and just past,
# so every case exercises the vector body, the tail, or both.
LENGTHS = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 200]

POISON = 0xAAAAAAAA
SENTINEL = 0xDEADBEEF
PAD = 8


# --------------------------------------------------------------- reference


def add_pixels(a: int, b: int) -> int:
    """VP8LAddPixels: per-component add, mod 256."""
    ag = ((a & 0xFF00FF00) + (b & 0xFF00FF00)) & M32
    rb = ((a & 0x00FF00FF) + (b & 0x00FF00FF)) & M32
    return (ag & 0xFF00FF00) | (rb & 0x00FF00FF)


def average2(a0: int, a1: int) -> int:
    return ((((a0 ^ a1) & 0xFEFEFEFE) >> 1) + (a0 & a1)) & M32


def average3(a0: int, a1: int, a2: int) -> int:
    return average2(average2(a0, a2), a1)


def average4(a0: int, a1: int, a2: int, a3: int) -> int:
    return average2(average2(a0, a1), average2(a2, a3))


def clip255(a: int) -> int:
    v = a & M32
    if v < 256:
        return v
    return ((M32 ^ v) & M32) >> 24


def add_sub_component_full(a: int, b: int, c: int) -> int:
    return clip255(a + b - c)


def clamped_add_subtract_full(c0: int, c1: int, c2: int) -> int:
    a = add_sub_component_full(c0 >> 24, c1 >> 24, c2 >> 24)
    r = add_sub_component_full((c0 >> 16) & 0xFF, (c1 >> 16) & 0xFF,
                               (c2 >> 16) & 0xFF)
    g = add_sub_component_full((c0 >> 8) & 0xFF, (c1 >> 8) & 0xFF,
                               (c2 >> 8) & 0xFF)
    b = add_sub_component_full(c0 & 0xFF, c1 & 0xFF, c2 & 0xFF)
    return (a << 24) | (r << 16) | (g << 8) | b


def add_sub_component_half(a: int, b: int) -> int:
    # Mojo's `//` is floor division, which is what this port uses here (see the
    # note in src/webplib/lossless.mojo's AddSubtractComponentHalf).
    return clip255(a + (a - b) // 2)


def clamped_add_subtract_half(c0: int, c1: int, c2: int) -> int:
    ave = average2(c0, c1)
    a = add_sub_component_half(ave >> 24, c2 >> 24)
    r = add_sub_component_half((ave >> 16) & 0xFF, (c2 >> 16) & 0xFF)
    g = add_sub_component_half((ave >> 8) & 0xFF, (c2 >> 8) & 0xFF)
    b = add_sub_component_half(ave & 0xFF, c2 & 0xFF)
    return (a << 24) | (r << 16) | (g << 8) | b


def sub3(a: int, b: int, c: int) -> int:
    return abs(b - c) - abs(a - c)


def select(a: int, b: int, c: int) -> int:
    s = (sub3(a >> 24, b >> 24, c >> 24)
         + sub3((a >> 16) & 0xFF, (b >> 16) & 0xFF, (c >> 16) & 0xFF)
         + sub3((a >> 8) & 0xFF, (b >> 8) & 0xFF, (c >> 8) & 0xFF)
         + sub3(a & 0xFF, b & 0xFF, c & 0xFF))
    return a if s <= 0 else b


ARGB_BLACK = 0xFF000000


def predictor(mode: int, left: int, top_left: int, top: int,
              top_right: int) -> int:
    return [ARGB_BLACK, left, top, top_right, top_left,
            average3(left, top, top_right),
            average2(left, top_left), average2(left, top),
            average2(top_left, top), average2(top, top_right),
            average4(left, top_left, top, top_right),
            select(top, left, top_left),
            clamped_add_subtract_full(left, top, top_left),
            clamped_add_subtract_half(left, top, top_left)][mode]


def ref_predictor(mode: int, ip: list, up: list, op: list, n: int,
                  start: int, up_off: int) -> None:
    """Fills `op[start : start+n]` in place.

    `op` is indexed in buffer coordinates; `up_off` is the index in `up` that
    the kernel's `upper` pointer names, so up[up_off - 1] is the top-left
    neighbour of the first pixel, exactly as upstream's `top[-1]` is.
    """
    for i in range(n):
        x = start + i
        j = up_off + i
        op[x] = add_pixels(ip[i],
                           predictor(mode, op[x - 1], up[j - 1], up[j],
                                     up[j + 1]))


def ref_green(src: list) -> int:
    argb = src
    green = (argb >> 8) & 0xFF
    rb = argb & 0x00FF00FF
    rb = (rb + (green << 16) + green) & 0x00FF00FF
    return (argb & 0xFF00FF00) | rb


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("mode", range(14))
@pytest.mark.parametrize("n", LENGTHS)
def test_predictor_modes_match_scalar_reference(mode, n):
    """Every predictor mode, vector body and tail, must equal the C reference."""
    rng = np.random.default_rng(mode * 1000 + n)
    inp = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    # `up` needs one element before the row (top_left) and one after (top_right).
    up = np.full(PAD + n + PAD + 16, 0xCCCCCCCC, dtype=np.uint32)
    up[PAD:PAD + n + 16] = rng.integers(0, 1 << 32, n + 16,
                                       dtype=np.uint64).astype(np.uint32)
    out = np.full(n + PAD + 16, POISON, dtype=np.uint32)
    out[PAD - 1] = SENTINEL  # the "left" pixel of pixel 0

    lib().mw_kernel_run(PRED, inp.ctypes.data, up.ctypes.data + PAD * 4,
                       out.ctypes.data + PAD * 4, n, mode)

    want = [POISON] * (n + PAD + 16)
    want[PAD - 1] = SENTINEL
    ref_predictor(mode, inp.tolist(), up.tolist(), want, n, PAD, PAD)

    got = out.tolist()
    assert got[PAD - 1:PAD + n] == want[PAD - 1:PAD + n], (
        f"mode {mode}, n={n}: first difference at "
        f"{next(i for i, (g, w) in enumerate(zip(got[PAD - 1:PAD + n], want[PAD - 1:PAD + n])) if g != w)}")
    # nothing outside [PAD, PAD + n) may be written
    assert got[:PAD - 1] == [POISON] * (PAD - 1)
    assert got[PAD + n:] == [POISON] * 16


@pytest.mark.parametrize("n", LENGTHS)
def test_subtract_green_matches_scalar_reference(n):
    """VP8LAddGreenToBlueAndRed, vector body and tail."""
    src = np.random.default_rng(n).integers(0, 1 << 32, n,
                                            dtype=np.uint64).astype(np.uint32)
    src_buf = np.zeros(n + 16, dtype=np.uint32)
    src_buf[:n] = src
    dst = np.zeros(n + 16, dtype=np.uint32)

    lib().mw_kernel_run(GREEN, src_buf.ctypes.data, dst.ctypes.data, 0, n, 0)

    assert dst[:n].tolist() == [ref_green(int(v)) for v in src]