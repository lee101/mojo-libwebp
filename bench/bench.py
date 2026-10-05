"""mojo-libwebp against upstream libwebp on lossless decode.

    pixi run bench

The pixi task holds a machine-wide flock so a concurrent job cannot distort
the numbers.  Upstream is loaded from conda-forge's libwebp through ctypes, so
both sides run on the same machine, in the same process, on the same bytes.
"""

from __future__ import annotations

import ctypes
import io
import math
import os
import platform
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojowebp  # noqa: E402


def load_upstream() -> ctypes.CDLL:
    lib = ctypes.CDLL("libwebp.so")
    lib.WebPDecodeRGBA.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.WebPDecodeRGBA.restype = ctypes.c_void_p
    lib.WebPFree.argtypes = [ctypes.c_void_p]
    return lib


def encode_lossless(arr: np.ndarray, quality: int = 100) -> bytes:
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="WEBP", lossless=True,
                                    quality=quality)
    return buf.getvalue()


def encode_lossless_method(arr: np.ndarray, method: int) -> bytes:
    """Lossless encode at a chosen effort.

    `method=0` is what keeps the predictor cases decodable by this port: at the
    default effort the encoder also picks the cross-colour transform, and this
    port does not reproduce that path (README "Known issues"), so those streams
    are reported as unsupported instead of being timed.
    """
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="WEBP", lossless=True,
                                    quality=100, method=method)
    return buf.getvalue()


def timeit(fn, repeat: int = 7) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def noise(h, w, ch, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, ch), dtype=np.uint8)


def gradient(h, w, ch):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256],
                axis=2).astype(np.uint8)
    if ch == 4:
        a = np.concatenate([a, np.full((h, w, 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(a[:, :, :ch])


def photo(h, w, ch, seed=4):
    """A smooth, photo-like image: big runs, which is what LZ77 is for."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, ch), dtype=np.uint8)
    a = np.kron(small, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w, :ch]
    return np.ascontiguousarray(a)


def smooth(h, w, ch, seed=7):
    """Low-frequency content with a little noise, so the encoder picks the
    predictor transform.  The steep `gradient` above encodes to a stream this
    port rejects (README "Known issues"), so it cannot time the predictor path;
    this one encodes to a predictor stream the port decodes bit-exactly."""
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(seed)
    a = np.stack([(xx // 3) % 256, (yy // 5) % 256, ((xx + yy) // 7) % 256],
                axis=2).astype(np.int32)
    a = np.clip(a + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)
    if ch == 4:
        a = np.concatenate([a, np.full((h, w, 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(a[:, :, :ch])


def upstream_decode_rgba(data: bytes) -> np.ndarray:
    """Decode with the C library and return an (h, w, 4) array.

    WebPDecodeRGBA returns a heap pointer that the caller frees with WebPFree;
    it is copied out here so both decoders hand back comparable arrays.
    """
    w = ctypes.c_int(0)
    h = ctypes.c_int(0)
    ptr = UP.WebPDecodeRGBA(AD(data), len(data), ctypes.byref(w), ctypes.byref(h))
    if not ptr:
        raise RuntimeError("upstream WebPDecodeRGBA failed")
    n = w.value * h.value * 4
    out = np.frombuffer(ctypes.string_at(ptr, n), dtype=np.uint8)
    out = out.reshape(h.value, w.value, 4).copy()
    UP.WebPFree(ptr)
    return out


@case("decode RGBA  128x128  noise")
def _():
    a = noise(128, 128, 4, 1)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  256x256  noise")
def _():
    a = noise(256, 256, 4, 2)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGB   256x256  noise")
def _():
    a = noise(256, 256, 3, 3)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGB(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  256x256  gradient")
def _():
    a = gradient(256, 256, 4)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  512x512 gradient")
def _():
    a = gradient(512, 512, 4)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  512x512  photo-like")
def _():
    a = photo(512, 512, 4)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGB   512x512  photo-like")
def _():
    a = photo(512, 512, 3)
    d = encode_lossless(a)
    return (lambda: mojowebp.WebPDecodeRGB(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  512x512  predictor")
def _():
    a = smooth(512, 512, 4)
    d = encode_lossless_method(a, 0)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGB   512x512  predictor")
def _():
    a = smooth(512, 512, 3)
    d = encode_lossless_method(a, 0)
    return (lambda: mojowebp.WebPDecodeRGB(d),
            lambda: upstream_decode_rgba(d), d)


@case("decode RGBA  256x256  predictor")
def _():
    a = smooth(256, 256, 4)
    d = encode_lossless_method(a, 0)
    return (lambda: mojowebp.WebPDecodeRGBA(d),
            lambda: upstream_decode_rgba(d), d)


def machine() -> str:
    model = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {os.cpu_count()} threads, {platform.system()} {platform.release()}"


UNSUPPORTED = ("lz77-dist",)


def supported(data: bytes) -> bool:
    """Streams using a path this port does not decode yet are reported as
    unsupported rather than timed; see README "Known issues"."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    from conftest import transform_types
    return not any(t in UNSUPPORTED for t in transform_types(data))


def _run_isolated(ours, theirs):
    """Time both decoders in a forked child; returns (None, None) on a crash.

    A decode that returns early (for example by reporting a bitstream error)
    would look artificially fast, so the two outputs are compared first and a
    mismatch is reported instead of a timing.
    """
    import multiprocessing as mp

    def child(pipe):
        try:
            a_out = ours()
            b_out = theirs()
            # Upstream is always asked for RGBA; a 3-channel case is compared
            # against the same channels of our RGBA output.
            if a_out.shape[2] == 3:
                b_out = b_out[:, :, :3]
            if a_out.shape != b_out.shape or not np.array_equal(a_out, b_out):
                pipe.send(("mismatch", None))
                pipe.close()
                return
            a = timeit(ours)
            b = timeit(theirs)
        except Exception as exc:
            pipe.send(("error", f"{type(exc).__name__}: {exc}"))
            pipe.close()
            return
        pipe.send((a, b))
        pipe.close()

    ctx = mp.get_context("fork")
    parent, child_pipe = ctx.Pipe()
    proc = ctx.Process(target=child, args=(child_pipe,))
    proc.start()
    if parent.poll(60):
        result = parent.recv()
    else:
        proc.kill()
        result = (None, None)
    proc.join(10)
    return result


def main() -> None:
    global UP, AD
    UP = load_upstream()
    AD = lambda d: (ctypes.c_uint8 * len(d)).from_buffer_copy(d)  # noqa: E731
    print(f"machine: {machine()}")
    print(f"{'case':<38}{'mojo-libwebp':>16}{'libwebp C':>16}{'ratio':>9}")
    print("-" * 79)
    for name, build in CASES:
        ours, theirs, data = build()
        if not supported(data):
            print(f"{name:<38}{'unsupported':>32}")
            continue
        # Each case runs in a forked child: the decoder still has memory-safety
        # bugs (README "Known issues") and a crash must not take the run down.
        ours_a, theirs_a = _run_isolated(ours, theirs)
        if ours_a == "mismatch":
            print(f"{name:<38}{'output != upstream':>32}")
            continue
        if ours_a == "error":
            print(f"{name:<38}{str(theirs_a)[:40]}")
            continue
        if ours_a is None:
            print(f"{name:<38}{'decoder crashed':>32}")
            continue
        a, b = ours_a, theirs_a
        flag = "faster" if a < b else "slower"
        print(f"{name:<38}{a * 1e3:>13.1f}ms{b * 1e3:>13.1f}ms{b / a:>8.2f}x  {flag}")
    print()
    print("ratio = libwebp C time / mojo-libwebp time; >1 means Mojo is faster.")


if __name__ == "__main__":
    main()
