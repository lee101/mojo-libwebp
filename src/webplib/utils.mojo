"""Port of src/utils/utils.c and the integer helpers from src/dsp/dsp.h.

The checked allocator (WebPSafeMalloc / WebPSafeCalloc / WebPSafeFree) has no
Mojo counterpart: the 1.2 stdlib exposes no heap entry point at all, so every
buffer in this port is owned by the caller (numpy) and reaches the kernels as a
raw `Int` address.  The allocation *sites* of the C code are preserved as
explicit arena requests; see VP8LDecoder.arena in vp8l_dec.mojo.
"""

from webplib.format_constants import MAX_ALLOWED_CODE_LENGTH


# Returns the position of the most significant set bit, i.e. floor(log2(v)).
def BitsLog2Floor(v: Int) -> Int:
    return v.bit_width() - 1


# Returns floor(log2(v)) + 1 for v > 0, 0 for v == 0 (src/utils/utils.c).
def VP8LBitsLog2Ceiling(v: Int) -> Int:
    if v == 0:
        return 0
    return v.bit_width()


# ---------------------------------------------------------------- tables
# kBitMask[n] == (1 << n) - 1 for every n in 0..VP8L_MAX_NUM_BIT_READ, so the
# upstream lookup table is the closed form below.
def kBitMask(n: Int) -> Int:
    return (1 << n) - 1
