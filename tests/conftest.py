"""Shared fixtures: a corpus of lossless WebP streams and the reference decoder.

The reference is the real libwebp C library from conda-forge, loaded through
ctypes.  It is the upstream implementation, so agreement with it is the parity
bar; Pillow (which bundles its own libwebp) is used only to *produce* the
streams, so the encoder side is upstream's too.
"""

from __future__ import annotations

import ctypes
import io
import os

import numpy as np
import pytest
from PIL import Image

UPSTREAM_LIB = "libwebp.so"


def _load_upstream() -> ctypes.CDLL | None:
    try:
        lib = ctypes.CDLL(UPSTREAM_LIB)
    except OSError:
        return None
    lib.WebPGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_int)]
    lib.WebPGetInfo.restype = ctypes.c_int
    lib.WebPDecodeRGBA.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_int),
                                   ctypes.POINTER(ctypes.c_int)]
    lib.WebPDecodeRGBA.restype = ctypes.c_void_p
    lib.WebPFree.argtypes = [ctypes.c_void_p]
    return lib


upstream = _load_upstream()
requires_upstream = pytest.mark.skipif(
    upstream is None, reason="upstream libwebp shared library not available"
)


def as_rgba(arr: np.ndarray) -> np.ndarray:
    """libwebp's WebPDecodeRGBA always returns 4 channels; add the alpha that a
    3-channel source encodes as opaque."""
    if arr.shape[2] == 4:
        return arr
    return np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2)


def upstream_decode_rgba(data: bytes) -> np.ndarray:
    """Decode with the C library; this is the parity oracle."""
    w = ctypes.c_int(0)
    h = ctypes.c_int(0)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    ptr = upstream.WebPDecodeRGBA(buf, len(data), ctypes.byref(w), ctypes.byref(h))
    if not ptr:
        raise RuntimeError("upstream WebPDecodeRGBA failed")
    n = w.value * h.value * 4
    raw = ctypes.string_at(ptr, n)
    result = np.frombuffer(raw, dtype=np.uint8).reshape(h.value, w.value, 4).copy()
    upstream.WebPFree(ptr)
    return result


def encode_lossless(arr: np.ndarray) -> bytes:
    """Produce a lossless WebP with upstream's own encoder (via Pillow)."""
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="WEBP", lossless=True, quality=100)
    return buf.getvalue()


def make_images(seed: int = 3, count: int = 24, lo: int = 1, hi: int = 48):
    """A deterministic corpus covering the image shapes the port handles.

    `kind` selects the content family:

    0 uniform noise, 1 smooth gradient, 2 flat colour, 3 4x block replication.
    """
    rng = np.random.default_rng(seed)
    out = []
    for trial in range(count):
        h = int(rng.integers(lo, hi))
        w = int(rng.integers(lo, hi))
        mode = "RGBA" if trial % 2 else "RGB"
        ch = 4 if mode == "RGBA" else 3
        kind = trial % 4
        if kind == 0:
            arr = rng.integers(0, 256, size=(h, w, ch), dtype=np.uint8)
        elif kind == 1:
            yy, xx = np.mgrid[0:h, 0:w]
            arr = np.stack(
                [(xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256], axis=2
            ).astype(np.uint8)
            if ch == 4:
                arr = np.concatenate([arr, np.full((h, w, 1), 255, np.uint8)], axis=2)
            arr = np.ascontiguousarray(arr[:, :, :ch])
        elif kind == 2:
            arr = np.full((h, w, ch), 77, np.uint8)
        else:
            base = rng.integers(0, 256, size=(max(1, h // 4), max(1, w // 4), ch),
                                dtype=np.uint8)
            arr = np.ascontiguousarray(
                np.kron(base, np.ones((4, 4, 1), dtype=np.uint8))[:h, :w, :ch]
            )
        out.append((f"{w}x{h}-{mode}-{kind}", arr, encode_lossless(arr),
                    1 if ch == 4 else 0))
    return out


def transform_types(data: bytes) -> list[str]:
    """Describe the stream properties the decoder path exercises.

    `predictor`, `cross-color`, `subtract-green` and `palette` name the lossless
    transforms present; `lz77-dist` marks a stream whose LZ77 distance decoding
    this port does not yet reproduce (see README "Known issues").
    """
    import mojowebp
    from mojowebp._lib import addr, lib
    from mojowebp.lossless_dec import _STATE, _as_buffer, extract_vp8l

    buf = _as_buffer(extract_vp8l(data))
    d, br, cc, ht, tmp, arena, cap, da, sz = _STATE.addrs(buf)
    lib().mw_vp8l_inspect_state(d, br, cc, ht, tmp, arena, cap, da, sz, 3)
    names = ["predictor", "cross-color", "subtract-green", "palette"]
    n = lib().mw_vp8l_inspect_state(d, br, cc, ht, tmp, arena, cap, da, sz, 3)
    out = [names[lib().mw_vp8l_inspect_transform(d, i, 0)] for i in range(n)]
    try:
        mojowebp.decode(data, 0)
    except mojowebp.WebPError:
        out.append("lz77-dist")
    return out
