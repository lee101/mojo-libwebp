"""Port of src/utils/bit_reader_utils.c (the VP8LBitReader half).

The boolean VP8BitReader is not ported: it belongs to the VP8 lossy decoder,
which is out of scope for this port.
"""

from webplib.format_constants import (
    VP8L_LBITS,
    VP8L_MAX_NUM_BIT_READ,
    VP8L_WBITS,
)
from webplib.utils import kBitMask

comptime VP8L_LOG8_WBITS: Int = 4  # log2(VP8L_WBITS / 8)


struct VP8LBitReader:
    var val: UInt64     # pre-fetched bits
    var buf: Int        # input byte buffer (address)
    var len: Int        # buffer length
    var pos: Int        # position in the buffer
    var bit_pos: Int    # current position in the bits, <= 8 * VP8L_LBITS
    var eos: Int        # end-Of-Stream condition


def VP8LInitBitReader(br: Int, start: Int, length: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    var n = length
    if n > 8:
        n = 8
    var value = UInt64(0)
    var i = 0
    while i < n:
        value |= UInt64(Pointer[UInt8, AnyOrigin[mut=True]](
            unsafe_from_address=start).unsafe_load(i)) << UInt64(8 * i)
        i += 1
    b[].val = value
    b[].len = length
    b[].pos = n
    b[].bit_pos = 0
    b[].eos = 0
    b[].buf = start


def VP8LBitReaderSetBuffer(br: Int, buf: Int, len: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    b[].buf = buf
    b[].len = len
    # 'pos' > 'len' should be considered a param error.
    b[].eos = 0
    if b[].pos > b[].len or VP8LIsEndOfStream(br):
        b[].eos = 1


def VP8LSetEndOfStream(br: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    b[].eos = 1
    b[].bit_pos = 0  # To avoid undefined behaviour with shifts.


# Return the prefetched bits, so they can be looked up.
def VP8LPrefetchBits(br: Int) -> UInt64:
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    return b[].val >> UInt64(b[].bit_pos & (VP8L_LBITS - 1))


# Returns true if there was an attempt at reading bit past the end of the
# buffer. Doesn't set br.eos flag.
def VP8LIsEndOfStream(br: Int) -> Int:
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    if b[].eos:
        return 1
    if b[].pos == b[].len and b[].bit_pos > VP8L_LBITS:
        return 1
    return 0


# For jumping over a number of bits in the bit stream when accessed with
# VP8LPrefetchBits and VP8LFillBitWindow.
# This function does *not* set br.eos, since it's speed-critical.
def VP8LSetBitPos(br: Int, val: Int):
    Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)[].bit_pos = val


# If not at EOS, reload up to VP8L_LBITS byte-by-byte.
def ShiftBytes(br: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    while b[].bit_pos >= 8 and b[].pos < b[].len:
        b[].val >>= 8
        b[].val |= UInt64(
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=b[].buf
            ).unsafe_load(b[].pos)
        ) << UInt64(VP8L_LBITS - 8)
        b[].pos += 1
        b[].bit_pos -= 8
    if VP8LIsEndOfStream(br):
        VP8LSetEndOfStream(br)


# Advances the read buffer by 4 bytes to make room for reading next 32 bits.
def VP8LDoFillBitWindow(br: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    if b[].pos + 8 < b[].len:
        b[].val >>= UInt64(VP8L_WBITS)
        b[].bit_pos -= VP8L_WBITS
        var src = Pointer[UInt8, AnyOrigin[mut=True]](
            unsafe_from_address=b[].buf + b[].pos
        )
        var word = UInt32(src.unsafe_load(0))
        word |= UInt32(src.unsafe_load(1)) << 8
        word |= UInt32(src.unsafe_load(2)) << 16
        word |= UInt32(src.unsafe_load(3)) << 24
        b[].val |= UInt64(word) << UInt64(VP8L_LBITS - VP8L_WBITS)
        b[].pos += VP8L_LOG8_WBITS
        return
    ShiftBytes(br)  # Slow path.


def VP8LFillBitWindow(br: Int):
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    if b[].bit_pos >= VP8L_WBITS:
        VP8LDoFillBitWindow(br)


# Reads the specified number of bits from read buffer.
# Flags an error in case end_of_stream or n_bits is more than the allowed limit
# of VP8L_MAX_NUM_BIT_READ (inclusive).
# Flags 'eos' if this read attempt is going to cross the read buffer.
def VP8LReadBits(br: Int, n_bits: Int) -> Int:
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    if not b[].eos and n_bits <= VP8L_MAX_NUM_BIT_READ:
        var value = Int(VP8LPrefetchBits(br) & UInt64(kBitMask(n_bits)))
        b[].bit_pos = b[].bit_pos + n_bits
        ShiftBytes(br)
        return value
    VP8LSetEndOfStream(br)
    return 0
