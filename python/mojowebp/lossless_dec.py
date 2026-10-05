"""Lossless (VP8L) decoding, mirroring the names and signatures of the
libwebp C API (`src/webp/decode.h`, `src/dec/vp8l_dec.c`).

Upstream returns a heap buffer from `WebPDecodeRGBA` and the caller frees it
with `WebPFree`.  Here the decoded pixels land directly in a numpy array owned
by the caller, so there is nothing to free; that is the only API-level
difference and it is the safer one.
"""

from __future__ import annotations

import numpy as np

from ._lib import (
    BR_WORDS,
    CC_WORDS,
    DEC_WORDS,
    HT_WORDS,
    TMP_WORDS,
    addr,
    lib,
)

# WEBP_CSP_MODE
MODE_RGB = 0
MODE_RGBA = 1
MODE_BGR = 2
MODE_BGRA = 3
MODE_ARGB = 4
MODE_RGBA_4444 = 5
MODE_RGB_565 = 6
MODE_rgbA = 7
MODE_bgrA = 8
MODE_Argb = 9
MODE_rgbA_4444 = 10
MODE_YUV = 11
MODE_YUVA = 12

# VP8StatusCode
VP8_STATUS_OK = 0
VP8_STATUS_SUSPENDED = 2
VP8_STATUS_BITSTREAM_ERROR = 3
VP8_STATUS_OUT_OF_MEMORY = 4
VP8_STATUS_INVALID_PARAM = 5
VP8_STATUS_UNSUPPORTED_FEATURE = 6

VP8L_MAGIC_BYTE = 0x2F
VP8L_FRAME_HEADER_SIZE = 5

_CHANNELS = {
    MODE_RGB: 3,
    MODE_RGBA: 4,
    MODE_BGR: 3,
    MODE_BGRA: 4,
    MODE_ARGB: 4,
}


class WebPError(RuntimeError):
    """Raised for bitstream errors; the code is upstream's VP8StatusCode."""

    def __init__(self, status: int, message: str = "webp decode failed"):
        self.status = status
        super().__init__(f"{message} (VP8_STATUS={status})")


def _as_buffer(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.uint8)
    return np.frombuffer(bytes(data), dtype=np.uint8)


class _State:
    """The decoder scratch buffers, reused across calls so the hot path does
    no Python-side allocation."""

    def __init__(self):
        self.dec = np.zeros(DEC_WORDS, dtype=np.int64)
        self.br = np.zeros(BR_WORDS, dtype=np.int64)
        self.cc = np.zeros(CC_WORDS, dtype=np.int64)
        self.ht = np.zeros(HT_WORDS, dtype=np.int64)
        self.tmp = np.zeros(TMP_WORDS, dtype=np.int64)
        self.ws = np.zeros(1 << 16, dtype=np.uint32)

    def grow(self, words: int) -> None:
        if words > self.ws.size:
            self.ws = np.zeros(words, dtype=np.uint32)

    def addrs(self, data: np.ndarray):
        return (
            addr(self.dec),
            addr(self.br),
            addr(self.cc),
            addr(self.ht),
            addr(self.tmp),
            addr(self.ws),
            self.ws.size,
            addr(data),
            data.size,
        )


_STATE = _State()
_W = np.zeros(1, dtype=np.int32)
_H = np.zeros(1, dtype=np.int32)
_A = np.zeros(1, dtype=np.int32)


def check_signature(data) -> bool:
    """Mirrors `int VP8LCheckSignature(const uint8_t* data, size_t size)`.

    Accepts a bare VP8L stream or a RIFF/WEBP container holding one.
    """
    try:
        buf = _as_buffer(extract_vp8l(data))
    except WebPError:
        return False
    return bool(lib().mw_vp8l_check_signature(addr(buf), buf.size))


def get_info(data) -> tuple[int, int, int]:
    """Mirrors `int VP8LGetInfo(const uint8_t*, size_t, int*, int*, int*)`.

    Returns (width, height, has_alpha); (0, 0, 0) when the data is not a VP8L
    bitstream, exactly as upstream's 0 return means "not VP8L".
    """
    buf = _as_buffer(extract_vp8l(data))
    ok = lib().mw_vp8l_get_info(
        addr(buf), buf.size, addr(_STATE.br), addr(_W), addr(_H), addr(_A),
        addr(_STATE.tmp),
    )
    if not ok:
        return (0, 0, 0)
    return (int(_W[0]), int(_H[0]), int(_A[0]))


def extract_vp8l(data) -> bytes:
    """Return the VP8L bitstream inside a RIFF/WEBP container, or `data` itself
    when it already is a bare VP8L stream."""
    raw = _as_buffer(data)
    if raw.size == 0:
        raise WebPError(VP8_STATUS_BITSTREAM_ERROR, "empty input")
    if raw[0] == VP8L_MAGIC_BYTE:
        return raw.tobytes()
    if raw.size < 12 or bytes(raw[0:4]) != b"RIFF" or bytes(raw[8:12]) != b"WEBP":
        raise WebPError(VP8_STATUS_BITSTREAM_ERROR, "not a RIFF/WEBP container")
    pos = 12
    while pos + 8 <= raw.size:
        fourcc = bytes(raw[pos:pos + 4])
        size = int.from_bytes(bytes(raw[pos + 4:pos + 8]), "little")
        if fourcc == b"VP8L":
            return raw[pos + 8:pos + 8 + size].tobytes()
        pos += 8 + size + (size & 1)
    raise WebPError(VP8_STATUS_UNSUPPORTED_FEATURE, "no VP8L chunk (lossy only)")


def decode(data, colorspace: int = MODE_RGBA) -> np.ndarray:
    """Mirrors `uint8_t* WebPDecode(const uint8_t* data, size_t data_size,
    int* width, int* height)` for the VP8L stream in `data`.

    `data` is either a raw VP8L bitstream (starting with 0x2f) or a complete
    RIFF/WEBP container holding a VP8L chunk.
    """
    buf = _as_buffer(extract_vp8l(data))
    w, h, _ = get_info(buf)
    if w == 0 or h == 0:
        raise WebPError(VP8_STATUS_BITSTREAM_ERROR, "not a VP8L bitstream")
    channels = _CHANNELS.get(colorspace)
    if channels is None:
        raise WebPError(VP8_STATUS_UNSUPPORTED_FEATURE, f"colorspace {colorspace}")

    d, br, cc, ht, tmp, arena, cap, data_addr, size = _STATE.addrs(buf)
    need = lib().mw_vp8l_workspace_size(d, br, cc, ht, tmp, arena, cap, data_addr, size)
    if need < 0:
        raise WebPError(VP8_STATUS_BITSTREAM_ERROR, "bad VP8L header")
    _STATE.grow(need)
    out = np.zeros((h, w, channels), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, addr(_STATE.ws), _STATE.ws.size, data_addr, size,
        addr(out), w * channels, colorspace,
    )
    if status != VP8_STATUS_OK:
        raise WebPError(status, "VP8L decode failed")
    return out


def decode_alpha(data, width: int, height: int, filter_type: int = 0) -> np.ndarray:
    """Mirrors `WebPDecodeAlpha(const uint8_t* data, size_t data_length,
    int width, int height)` for a losslessly-compressed ALPH payload."""
    buf = _as_buffer(data)
    out = np.zeros((height, width), dtype=np.uint8)
    _STATE.grow((width * height * 3) // 2 + (1 << 20))
    d, br, cc, ht, tmp, arena, cap, data_addr, size = _STATE.addrs(buf)
    status = lib().mw_vp8l_decode_alpha(
        d, br, cc, ht, tmp, addr(_STATE.ws), _STATE.ws.size, data_addr, size,
        width, height, filter_type, addr(out),
    )
    if status != VP8_STATUS_OK:
        raise WebPError(status, "alpha decode failed")
    return out


def WebPDecodeRGBA(data) -> np.ndarray:
    return decode(data, MODE_RGBA)


def WebPDecodeRGB(data) -> np.ndarray:
    return decode(data, MODE_RGB)


def WebPDecodeBGRA(data) -> np.ndarray:
    return decode(data, MODE_BGRA)


def WebPDecodeBGR(data) -> np.ndarray:
    return decode(data, MODE_BGR)


def WebPDecodeARGB(data) -> np.ndarray:
    return decode(data, MODE_ARGB)


def WebPDecodeRGBAInto(data, out_rgb: np.ndarray, out_stride: int = 0) -> None:
    """Mirrors `WebPDecodeRGBAInto(data, data_size, output_buffer,
    output_stride)`."""
    out_rgb[...] = decode(data, MODE_RGBA).reshape(out_rgb.shape)


def WebPGetInfo(data) -> tuple[int, int, int]:
    """Mirrors `int WebPGetInfo(const uint8_t* data, size_t data_size, int*
    width, int* height)`; the third element is VP8L's has-alpha bit."""
    return get_info(data)


def WebPDecodeAlpha(data, width: int, height: int) -> np.ndarray:
    return decode_alpha(data, width, height, 0)
