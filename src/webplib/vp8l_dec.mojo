"""Port of src/dec/vp8l_dec.c.

The whole decoder state lives in caller-owned buffers (numpy arrays from
python/mojowebp/_lib.py) reached as `Int` addresses, because Mojo exposes no
heap allocator and `@export` cannot take a parametric pointer.  The state is
split exactly along upstream's own struct boundaries:

    VP8LDecoder  (this file)   -> one Int-only record
    VP8LBitReader              -> src/utils/bit_reader_utils.c
    VP8LTransform[4]           -> src/dec/vp8li_dec.h
    VP8LColorCache             -> src/utils/color_cache_utils.h
    HuffmanTables              -> src/utils/huffman_utils.h
    HTreeGroup[] + payload     -> the shared arena

The arena replaces WebPSafeMalloc.  It is a monotone bump allocator whose high
water mark is measured by a header pass (mojo_vp8l_dec_workspace_size) and
never shrinks: upstream's free/deallocate sites all release memory that is
dead by the end of the call, so keeping it only makes the arena an upper bound.
"""

from std.utils import StaticTuple

from webplib.bit_reader import (
    VP8LBitReader,
    VP8LFillBitWindow,
    VP8LInitBitReader,
    VP8LIsEndOfStream,
    VP8LPrefetchBits,
    VP8LReadBits,
    VP8LSetBitPos,
)
from webplib.filters import (
    WEBP_FILTER_GRADIENT,
    WEBP_FILTER_HORIZONTAL,
    WEBP_FILTER_NONE,
    WebPUnfilter,
)
from webplib.color_cache import (
    VP8LColorCache,
    VP8LColorCacheInit,
    VP8LColorCacheInsert,
    VP8LColorCacheLookup,
)
from webplib.format_constants import (
    ARENA_HEADER_WORDS,
    ARGB_BLACK,
    CODE_LENGTH_CODES,
    COLOR_INDEXING_TRANSFORM,
    CROSS_COLOR_TRANSFORM,
    HUFFMAN_CODES_PER_META_CODE,
    HUFFMAN_PACKED_BITS,
    HUFFMAN_PACKED_TABLE_SIZE,
    HUFFMAN_TABLE_BITS,
    HUFFMAN_TABLE_MASK,
    LENGTHS_TABLE_BITS,
    LENGTHS_TABLE_MASK,
    MAX_CACHE_BITS,
    MIN_HUFFMAN_BITS,
    MIN_TRANSFORM_BITS,
    MODE_BGRA,
    MODE_RGB,
    NUM_DISTANCE_CODES,
    NUM_HUFFMAN_BITS,
    NUM_LENGTH_CODES,
    NUM_LITERAL_CODES,
    NUM_TRANSFORM_BITS,
    NUM_TRANSFORMS,
    PREDICTOR_TRANSFORM,
    SUBTRACT_GREEN_TRANSFORM,
    VP8L_FRAME_HEADER_SIZE,
    VP8L_IMAGE_SIZE_BITS,
    VP8L_MAGIC_BYTE,
    VP8L_VERSION,
    VP8L_VERSION_BITS,
)
from webplib.huffman import (
    HuffmanTables,
    HuffmanTablesSegment,
    VP8LBuildHuffmanTable,
    VP8LHuffmanTablesAllocate,
    VP8LHuffmanTablesUseRegion,
    hc_bits,
    hc_value,
)
from webplib.lossless import (
    VP8LColorIndexInverseTransformAlpha,
    VP8LInverseTransform,
    VP8LSubSampleSize,
    VP8LTransform,
    VP8LConvertFromBGRA,
)

comptime T120 = StaticTuple[Int, 120]
comptime T19 = StaticTuple[Int, 19]
comptime T12 = StaticTuple[Int, 12]
comptime T5 = StaticTuple[Int, 5]
comptime T3 = StaticTuple[Int, 3]
comptime T2 = StaticTuple[Int, 2]

# HuffIndex
comptime GREEN: Int = 0
comptime RED: Int = 1
comptime BLUE: Int = 2
comptime ALPHA: Int = 3
comptime DIST: Int = 4

comptime kCodeLengthLiterals: Int = 16
comptime kCodeLengthRepeatCode: Int = 16
comptime kCodeLengthExtraBits = T3(2, 3, 7)
comptime kCodeLengthRepeatOffsets = T3(3, 3, 11)

comptime NUM_CODE_LENGTH_CODES: Int = 19
comptime kCodeLengthCodeOrder = T19(
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
)

comptime CODE_TO_PLANE_CODES: Int = 120
comptime kCodeToPlane = T120(
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
    0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
    0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
    0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
    0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
    0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
    0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
    0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
    0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
    0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
    0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
    0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70
)

# Memory needed for lookup tables of one Huffman tree group. Red, blue, alpha
# and distance alphabets are constant (256 for red, blue and alpha, 40 for
# distance) and lookup table sizes for them in worst case are 630 and 410
# respectively.
comptime FIXED_TABLE_SIZE: Int = 630 * 3 + 410
comptime kTableSize = T12(
    FIXED_TABLE_SIZE + 654,
    FIXED_TABLE_SIZE + 656,
    FIXED_TABLE_SIZE + 658,
    FIXED_TABLE_SIZE + 662,
    FIXED_TABLE_SIZE + 670,
    FIXED_TABLE_SIZE + 686,
    FIXED_TABLE_SIZE + 718,
    FIXED_TABLE_SIZE + 782,
    FIXED_TABLE_SIZE + 912,
    FIXED_TABLE_SIZE + 1168,
    FIXED_TABLE_SIZE + 1680,
    FIXED_TABLE_SIZE + 2704
)
comptime MAX_TABLE_SIZE: Int = FIXED_TABLE_SIZE + 2704

comptime kAlphabetSize = T5(
    NUM_LITERAL_CODES + NUM_LENGTH_CODES,
    NUM_LITERAL_CODES,
    NUM_LITERAL_CODES,
    NUM_LITERAL_CODES,
    NUM_DISTANCE_CODES
)
comptime kLiteralMap = T5(0, 1, 1, 1, 0)

comptime NUM_ARGB_CACHE_ROWS: Int = 16
comptime BITS_SPECIAL_MARKER: Int = 0x100
comptime PACKED_NON_LITERAL_CODE: Int = 0

# Reserved arena words (words, not bytes).  Everything is sized for the worst
# case so that the header pass and the decode pass allocate identically.
comptime RES_SCRATCH_TABLE: Int = MAX_TABLE_SIZE
comptime RES_SCRATCH_SORTED: Int = 2560
comptime RES_SCRATCH_CL: Int = 2560
comptime RES_SCRATCH_CCL: Int = 32
# sizeof(HuffmanTables) in bytes: a HuffmanTablesSegment (4 words) plus
# curr_segment.  The table payload starts here.
comptime HUFFMAN_TABLES_HEADER_BYTES: Int = 40
comptime RES_LENGTH_TABLES: Int = HUFFMAN_TABLES_HEADER_BYTES // 4 + (
    1 << LENGTHS_TABLE_BITS
)
comptime RES_TRANSFORMS: Int = 4 * 10  # 4 * sizeof(VP8LTransform) / 4
comptime RES_FIXED: Int = (
    ARENA_HEADER_WORDS
    + RES_SCRATCH_TABLE
    + RES_SCRATCH_SORTED
    + RES_SCRATCH_CL
    + RES_SCRATCH_CCL
    + RES_LENGTH_TABLES
    + RES_TRANSFORMS
)


struct HTreeGroup:
    var htree0: Int
    var htree1: Int
    var htree2: Int
    var htree3: Int
    var htree4: Int
    var is_trivial_literal: Int
    var literal_arb: UInt32
    var is_trivial_code: Int
    var use_packed_table: Int
    var packed_table: Int


struct VP8LDecoder:
    # VP8LDecoder
    var status: Int
    var state: Int
    var width: Int
    var height: Int
    var next_transform: Int
    var transforms_seen: Int
    var last_row: Int
    var last_pixel: Int
    var last_out_row: Int
    var br_addr: Int
    var transforms_addr: Int
    var huffman_tables_addr: Int
    var color_cache_addr: Int
    # VP8LMetadata
    var color_cache_size: Int
    var huffman_image: Int
    var huffman_xsize: Int
    var huffman_subsample_bits: Int
    var huffman_mask: Int
    var num_htree_groups: Int
    var htree_groups: Int  # arena word offset
    # buffers
    var pixels: Int
    var argb_cache: Int
    var arena: Int  # 4-word header: [cursor, capacity, pad, pad]
    var scratch_table: Int
    var scratch_sorted: Int
    var scratch_cl: Int
    var scratch_ccl: Int
    var scratch_length_tables: Int
    var tmp_scratch: Int
    var decoded_data: Int
    # output
    var io_width: Int
    var io_height: Int
    var io_crop_bottom: Int
    var colorspace: Int
    var out: Int
    var out_stride: Int
    # alpha plane decoding
    var use_8b_decode: Int
    var alpha_filter: Int
    var prev_line: Int
    var alpha_output: Int


# -----------------------------------------------------------------------------
# arena helpers


def ArenaInit(arena: Int, cap_words: Int):
    # The first RES_FIXED words are the fixed reservations (bit-reader scratch
    # tables, sorted[]/code-length scratch, transform array); the bump cursor
    # starts after them.
    var a = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=arena
    )
    a.unsafe_offset(0)[] = UInt32(RES_FIXED)
    a.unsafe_offset(1)[] = UInt32(cap_words)
    a.unsafe_offset(2)[] = UInt32(0)
    a.unsafe_offset(3)[] = UInt32(0)


def ArenaUsed(arena: Int) -> Int:
    return Int(Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=arena
    ).unsafe_offset(0)[])


# Reserves 'words' words and returns their address, or 0 when the arena is full.
# The 4-word header (cursor, capacity, pad, pad) sits in front of the payload, so
# the usable capacity is cap - ARENA_HEADER_WORDS words: comparing `used` against
# the raw `cap` would let the last ARENA_HEADER_WORDS words be handed out and
# write past the end of the buffer.
def ArenaAlloc(arena: Int, words: Int) -> Int:
    var a = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=arena
    )
    var used = Int(a.unsafe_offset(0)[])
    if used + words + ARENA_HEADER_WORDS > Int(a.unsafe_offset(1)[]):
        return 0
    a.unsafe_offset(0)[] = UInt32(used + words)
    return arena + 16 + used * 4


def ArenaZero(arena: Int, addr: Int, words: Int):
    var p = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=addr)
    var i = 0
    while i < words:
        p.unsafe_store(i, UInt32(0))
        i += 1


def TransformPtr(dec: Int, i: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    return d[].transforms_addr + i * 40


def GroupPtr(dec: Int, i: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    return d[].arena + 16 + d[].htree_groups * 4 + i * 80


def VP8LSetError(dec: Int, error: Int) -> Int:
    # The oldest error reported takes precedence over the new one.
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if d[].status == 0 or d[].status == 2:
        d[].status = error
    return 0


# -----------------------------------------------------------------------------
#  Five Huffman codes are used at each meta code


def VP8LCheckSignature(data: Int, size: Int) -> Int:
    var p = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=data)
    if size < VP8L_FRAME_HEADER_SIZE:
        return 0
    if p.unsafe_load(0) != UInt8(VP8L_MAGIC_BYTE):
        return 0
    if (p.unsafe_load(4) >> 5) != 0:  # version
        return 0
    return 1


def ReadImageInfo(br: Int, width: Int, height: Int, has_alpha: Int) -> Int:
    var w = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=width)
    var h = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=height)
    var a = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=has_alpha)
    if VP8LReadBits(br, 8) != VP8L_MAGIC_BYTE:
        return 0
    w[0] = VP8LReadBits(br, VP8L_IMAGE_SIZE_BITS) + 1
    h[0] = VP8LReadBits(br, VP8L_IMAGE_SIZE_BITS) + 1
    a[0] = VP8LReadBits(br, 1)
    if VP8LReadBits(br, VP8L_VERSION_BITS) != VP8L_VERSION:
        return 0
    return 1 - VP8LIsEndOfStream(br)


# 'br' is a caller-owned VP8LBitReader, standing in for the C local.
def VP8LGetInfo(
    data: Int,
    data_size: Int,
    br: Int,
    width: Int,
    height: Int,
    has_alpha: Int,
    tmp: Int,
) -> Int:
    if data_size < VP8L_FRAME_HEADER_SIZE:
        return 0  # not enough data
    if VP8LCheckSignature(data, data_size) == 0:
        return 0  # bad signature
    var w = 0
    var h = 0
    var a = 0
    VP8LInitBitReader(br, data, data_size)
    if ReadImageInfo(br, tmp, tmp + 8, tmp + 16) == 0:
        return 0
    w = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=tmp)[0]
    h = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=tmp + 8)[0]
    a = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=tmp + 16)[0]
    if width != 0:
        Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=width)[0] = w
    if height != 0:
        Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=height)[0] = h
    if has_alpha != 0:
        Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=has_alpha)[0] = a
    return 1


# -----------------------------------------------------------------------------

def GetCopyDistance(distance_symbol: Int, br: Int) -> Int:
    if distance_symbol < 4:
        return distance_symbol + 1
    var extra_bits = (distance_symbol - 2) >> 1
    var offset = (2 + (distance_symbol & 1)) << extra_bits
    return offset + VP8LReadBits(br, extra_bits) + 1


def GetCopyLength(length_symbol: Int, br: Int) -> Int:
    # Length and distance prefixes are encoded the same way.
    return GetCopyDistance(length_symbol, br)


def PlaneCodeToDistance(xsize: Int, plane_code: Int) -> Int:
    if plane_code > CODE_TO_PLANE_CODES:
        return plane_code - CODE_TO_PLANE_CODES
    var dist_code = kCodeToPlane[plane_code - 1]
    var yoffset = dist_code >> 4
    var xoffset = 8 - (dist_code & 0xf)
    var dist = yoffset * xsize + xoffset
    # dist<1 can happen if xsize is very small
    return dist if dist >= 1 else 1


#------------------------------------------------------------------------------
# Decodes the next Huffman code from bit-stream.
# VP8LFillBitWindow(br) needs to be called at minimum every second call
# to ReadSymbol, in order to pre-fetch enough bits.
def ReadSymbol(table: Int, br: Int) -> Int:
    var t = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=table)
    var val = VP8LPrefetchBits(br)
    var idx = Int(val & UInt64(HUFFMAN_TABLE_MASK))
    var entry = t.unsafe_load(idx)
    var nbits = hc_bits(entry) - HUFFMAN_TABLE_BITS
    if nbits > 0:
        VP8LSetBitPos(br, Pointer[VP8LBitReader, AnyOrigin[mut=True]](
            unsafe_from_address=br
        )[].bit_pos + HUFFMAN_TABLE_BITS)
        val = VP8LPrefetchBits(br)
        idx = idx + hc_value(entry) + Int(val & UInt64((1 << nbits) - 1))
        entry = t.unsafe_load(idx)
    VP8LSetBitPos(
        br,
        Pointer[VP8LBitReader, AnyOrigin[mut=True]](
            unsafe_from_address=br
        )[].bit_pos + hc_bits(entry),
    )
    return hc_value(entry)


# Reads packed symbol depending on GREEN channel
def ReadPackedSymbols(group: Int, br: Int, dst: Int) -> Int:
    var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](unsafe_from_address=group)
    var val = Int(VP8LPrefetchBits(br) & UInt64(HUFFMAN_PACKED_TABLE_SIZE - 1))
    var pt = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].packed_table + val * 8
    )
    var code_bits = Int(pt.unsafe_load(0))
    var code_value = pt.unsafe_load(1)
    if code_bits < BITS_SPECIAL_MARKER:
        VP8LSetBitPos(br, Pointer[VP8LBitReader, AnyOrigin[mut=True]](
            unsafe_from_address=br
        )[].bit_pos + code_bits)
        Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=dst).unsafe_store(
            0, code_value
        )
        return PACKED_NON_LITERAL_CODE
    VP8LSetBitPos(
        br,
        Pointer[VP8LBitReader, AnyOrigin[mut=True]](
            unsafe_from_address=br
        )[].bit_pos + code_bits - BITS_SPECIAL_MARKER,
    )
    return Int(code_value)


def AccumulateHCode(
    hcode: UInt32,
    shift: Int,
    pt: Pointer[UInt32, AnyOrigin[mut=True]],
    slot: Int,
) -> Int:
    pt.unsafe_store(slot, UInt32(Int(pt.unsafe_load(slot)) + hc_bits(hcode)))
    pt.unsafe_store(
        slot + 1,
        pt.unsafe_load(slot + 1)
        | (((hcode >> 8) & 0xffff) << UInt32(shift)),
    )
    return hc_bits(hcode)


def BuildPackedTable(htree_group: Int):
    var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](
        unsafe_from_address=htree_group
    )
    var green = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].htree0
    )
    var red = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].htree1
    )
    var blue = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].htree2
    )
    var alpha = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].htree3
    )
    var pt = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=g[].packed_table
    )
    var code = 0
    while code < HUFFMAN_PACKED_TABLE_SIZE:
        var bits = code
        var hcode = green.unsafe_load(bits)
        if hc_value(hcode) >= NUM_LITERAL_CODES:
            pt.unsafe_store(
                code * 2, UInt32(hc_bits(hcode) + BITS_SPECIAL_MARKER)
            )
            pt.unsafe_store(code * 2 + 1, UInt32(hc_value(hcode)))
        else:
            pt.unsafe_store(code * 2, UInt32(0))
            pt.unsafe_store(code * 2 + 1, UInt32(0))
            bits >>= AccumulateHCode(hcode, 8, pt, code * 2)
            bits >>= AccumulateHCode(red.unsafe_load(bits), 16, pt, code * 2)
            bits >>= AccumulateHCode(blue.unsafe_load(bits), 0, pt, code * 2)
            bits >>= AccumulateHCode(alpha.unsafe_load(bits), 24, pt, code * 2)
        code += 1


def ReadHuffmanCodeLengths(
    dec: Int,
    br: Int,
    code_length_code_lengths: Int,
    num_symbols: Int,
    code_lengths: Int,
) -> Int:
    var ok = 0
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var clcl = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=code_length_code_lengths
    )
    var cl = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=code_lengths
    )
    var sorted = d[].scratch_sorted
    # Upstream declares `HuffmanTables tables;` on the stack here; the decoder
    # owns one fixed region for it instead.
    # The payload starts one HuffmanTables header past the header itself:
    # HuffmanTables is a HuffmanTablesSegment (4 words) plus curr_segment, so
    # 5 words = 40 bytes.  Using a larger offset would place the payload inside
    # the region the table is about to fill.
    VP8LHuffmanTablesUseRegion(
        d[].scratch_length_tables,
        1 << LENGTHS_TABLE_BITS,
        d[].scratch_length_tables + HUFFMAN_TABLES_HEADER_BYTES,
    )
    if VP8LBuildHuffmanTable(
        d[].scratch_length_tables,
        LENGTHS_TABLE_BITS,
        code_length_code_lengths,
        NUM_CODE_LENGTH_CODES,
        sorted,
        d[].arena,
    ) == 0:
        VP8LSetError(dec, 3)  # VP8_STATUS_BITSTREAM_ERROR
        return 0

    var prev_code_len = 8  # DEFAULT_CODE_LENGTH
    var max_symbol: Int
    if VP8LReadBits(br, 1):  # use length
        var length_nbits = 2 + 2 * VP8LReadBits(br, 3)
        max_symbol = 2 + VP8LReadBits(br, length_nbits)
        if max_symbol > num_symbols:
            VP8LSetError(dec, 3)
            return 0
    else:
        max_symbol = num_symbols

    var symbol = 0
    # Upstream reads `tables.curr_segment->start[...]`: the *start* of the current
    # segment, not its write cursor.  Reading curr_table would start one table
    # past it.  curr_segment itself matters because VP8LBuildHuffmanTable may
    # have grown the chain above and filled a fresh arena segment instead.
    var lseg = Pointer[HuffmanTablesSegment, AnyOrigin[mut=True]](
        unsafe_from_address=Pointer[HuffmanTables, AnyOrigin[mut=True]](
            unsafe_from_address=d[].scratch_length_tables
        )[].curr_segment
    )
    var tbl = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=lseg[].start
    )
    while symbol < num_symbols:
        if max_symbol == 0:
            break
        max_symbol -= 1
        VP8LFillBitWindow(br)
        var p = tbl.unsafe_load(Int(VP8LPrefetchBits(br)) & LENGTHS_TABLE_MASK)
        VP8LSetBitPos(br, Pointer[VP8LBitReader, AnyOrigin[mut=True]](
            unsafe_from_address=br
        )[].bit_pos + hc_bits(p))
        var code_len = hc_value(p)
        if code_len < kCodeLengthLiterals:
            cl.unsafe_store(symbol, UInt32(code_len))
            symbol += 1
            if code_len != 0:
                prev_code_len = code_len
        else:
            var use_prev = code_len == kCodeLengthRepeatCode
            var slot = code_len - kCodeLengthLiterals
            var extra_bits = kCodeLengthExtraBits[slot]
            var repeat_offset = kCodeLengthRepeatOffsets[slot]
            var repeat = VP8LReadBits(br, extra_bits) + repeat_offset
            if symbol + repeat > num_symbols:
                VP8LSetError(dec, 3)
                return 0
            var length = prev_code_len if use_prev else 0
            while repeat > 0:
                cl.unsafe_store(symbol, UInt32(length))
                symbol += 1
                repeat -= 1
    ok = 1
    _ = ok
    _ = clcl
    return 1


# 'code_lengths' is pre-allocated temporary buffer, used for creating Huffman
# tree.  Passing build == 0 mirrors upstream's NULL table (validate only).
def ReadHuffmanCode(
    alphabet_size: Int, dec: Int, br: Int, code_lengths: Int, build: Int
) -> Int:
    var size = 0
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var cl = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=code_lengths
    )
    var simple_code = VP8LReadBits(br, 1)

    ArenaZero(d[].arena, code_lengths, alphabet_size)

    if simple_code:  # Read symbols, codes & code lengths directly.
        var num_symbols = VP8LReadBits(br, 1) + 1
        var first_symbol_len_code = VP8LReadBits(br, 1)
        # The first code is either 1 bit or 8 bit code.
        var symbol = VP8LReadBits(br, 1 if first_symbol_len_code == 0 else 8)
        cl.unsafe_store(symbol, UInt32(1))
        # The second code (if present), is always 8 bits long.
        if num_symbols == 2:
            symbol = VP8LReadBits(br, 8)
            cl.unsafe_store(symbol, UInt32(1))
    else:  # Decode Huffman-coded code lengths.
        var code_length_code_lengths = d[].scratch_ccl
        var ccl = Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=code_length_code_lengths
        )
        var i = 0
        while i < NUM_CODE_LENGTH_CODES:
            ccl.unsafe_store(i, UInt32(0))
            i += 1
        var num_codes = VP8LReadBits(br, 4) + 4
        i = 0
        while i < num_codes:
            ccl.unsafe_store(
                kCodeLengthCodeOrder[i], UInt32(VP8LReadBits(br, 3))
            )
            i += 1
        if ReadHuffmanCodeLengths(
            dec, br, code_length_code_lengths, alphabet_size, code_lengths
        ) == 0:
            return 0

    var ok = 1 - VP8LIsEndOfStream(br)
    if ok:
        if build:
            size = VP8LBuildHuffmanTable(
                d[].huffman_tables_addr,
                HUFFMAN_TABLE_BITS,
                code_lengths,
                alphabet_size,
                d[].scratch_sorted,
                d[].arena,
            )
        else:
            size = 1
    if ok == 0 or size == 0:
        VP8LSetError(dec, 3)  # VP8_STATUS_BITSTREAM_ERROR
        return 0
    return size


def ReadHuffmanCodesHelper(
    color_cache_bits: Int,
    num_htree_groups: Int,
    num_htree_groups_max: Int,
    mapping: Int,
    dec: Int,
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var i = 0
    var j = 0
    var ok = 0
    var max_alphabet_size = (
        kAlphabetSize[0] + ((1 << color_cache_bits) if color_cache_bits > 0 else 0)
    )
    var code_lengths = d[].scratch_cl

    if (mapping == 0 and num_htree_groups != num_htree_groups_max) or (
        num_htree_groups > num_htree_groups_max
    ):
        VP8LSetError(dec, 4)  # VP8_STATUS_OUT_OF_MEMORY
        return 0

    var table_size = kTableSize[color_cache_bits]
    if VP8LHuffmanTablesAllocate(
        num_htree_groups * table_size, d[].huffman_tables_addr, d[].arena
    ) == 0:
        VP8LSetError(dec, 4)
        return 0

    # The group metadata and the packed tables also come from the arena.
    var groups_words = num_htree_groups * 20
    var groups_addr = ArenaAlloc(d[].arena, groups_words)
    if groups_addr == 0:
        VP8LSetError(dec, 4)
        return 0
    d[].htree_groups = ArenaUsed(d[].arena) - groups_words
    d[].num_htree_groups = num_htree_groups
    var packed_words = num_htree_groups * HUFFMAN_PACKED_TABLE_SIZE * 2
    var packed_addr = ArenaAlloc(d[].arena, packed_words)
    if packed_addr == 0:
        VP8LSetError(dec, 4)
        return 0
    var packed_off = ArenaUsed(d[].arena) - packed_words

    var ht = Pointer[HuffmanTables, AnyOrigin[mut=True]](
        unsafe_from_address=d[].huffman_tables_addr
    )
    while i < num_htree_groups_max:
        # If the index "i" is unused in the Huffman image, just make sure the
        # coefficients are valid but do not store them.
        if mapping != 0 and Pointer[Int32, AnyOrigin[mut=True]](
            unsafe_from_address=mapping
        ).unsafe_load(i) == -1:
            j = 0
            while j < HUFFMAN_CODES_PER_META_CODE:
                var alphabet_size = kAlphabetSize[j]
                if j == 0 and color_cache_bits > 0:
                    alphabet_size += (1 << color_cache_bits)
                # Passing build=0 so that nothing gets filled.
                if ReadHuffmanCode(alphabet_size, dec, d[].br_addr, code_lengths, 0) == 0:
                    return 0
                j += 1
        else:
            var gidx = i
            if mapping != 0:
                gidx = Int(
                    Pointer[Int32, AnyOrigin[mut=True]](
                        unsafe_from_address=mapping
                    ).unsafe_load(i)
                )
            var htree_group = GroupPtr(dec, gidx)
            var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](
                unsafe_from_address=htree_group
            )
            g[].packed_table = packed_addr + (gidx * HUFFMAN_PACKED_TABLE_SIZE * 2) * 4
            var size = 0
            var total_size = 0
            var is_trivial_literal = 1
            var max_bits = 0
            j = 0
            while j < HUFFMAN_CODES_PER_META_CODE:
                var alphabet_size = kAlphabetSize[j]
                if j == 0 and color_cache_bits > 0:
                    alphabet_size += (1 << color_cache_bits)
                size = ReadHuffmanCode(
                    alphabet_size, dec, d[].br_addr, code_lengths, 1
                )
                if size == 0:
                    return 0
                # Reading the Huffman codes can grow the segment chain and move
                # `curr_segment`, so `htrees[j]` is read from the segment that is
                # current *after* the build, as upstream does.
                var seg = Pointer[HuffmanTablesSegment, AnyOrigin[mut=True]](
                    unsafe_from_address=ht[].curr_segment
                )
                var htree = seg[].curr_table
                if j == GREEN:
                    g[].htree0 = htree
                elif j == RED:
                    g[].htree1 = htree
                elif j == BLUE:
                    g[].htree2 = htree
                elif j == ALPHA:
                    g[].htree3 = htree
                else:
                    g[].htree4 = htree
                var first = Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=htree
                ).unsafe_load(0)
                if is_trivial_literal and kLiteralMap[j] == 1:
                    is_trivial_literal = 1 if hc_bits(first) == 0 else 0
                total_size += hc_bits(first)
                seg[].curr_table = htree + size * 4
                if j <= ALPHA:
                    var clp = Pointer[UInt32, AnyOrigin[mut=True]](
                        unsafe_from_address=code_lengths
                    )
                    var local_max_bits = Int(clp.unsafe_load(0))
                    var k = 1
                    while k < alphabet_size:
                        if Int(clp.unsafe_load(k)) > local_max_bits:
                            local_max_bits = Int(clp.unsafe_load(k))
                        k += 1
                    max_bits += local_max_bits
                j += 1
            g[].is_trivial_literal = is_trivial_literal
            g[].is_trivial_code = 0
            if is_trivial_literal:
                var red_v = Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=g[].htree1
                ).unsafe_load(0)
                var blue_v = Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=g[].htree2
                ).unsafe_load(0)
                var alpha_v = Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=g[].htree3
                ).unsafe_load(0)
                var red = hc_value(red_v)
                var blue = hc_value(blue_v)
                var alpha = hc_value(alpha_v)
                g[].literal_arb = (UInt32(alpha) << 24) | (UInt32(red) << 16) | UInt32(blue)
                var green_v = Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=g[].htree0
                ).unsafe_load(0)
                if total_size == 0 and hc_value(green_v) < NUM_LITERAL_CODES:
                    g[].is_trivial_code = 1
                    g[].literal_arb = g[].literal_arb | (UInt32(hc_value(green_v)) << 8)
            g[].use_packed_table = (
                1 if (g[].is_trivial_code == 0 and max_bits < HUFFMAN_PACKED_BITS) else 0
            )
            if g[].use_packed_table:
                BuildPackedTable(htree_group)
        i += 1
    ok = 1
    _ = max_alphabet_size
    _ = packed_off
    return ok


def ReadHuffmanCodes(
    dec: Int, xsize: Int, ysize: Int, color_cache_bits: Int, allow_recursion: Int
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var br = d[].br_addr
    var huffman_image = 0
    var num_htree_groups = 1
    var num_htree_groups_max = 1
    var mapping = 0
    var ok = 0

    if allow_recursion and VP8LReadBits(br, 1):
        # use meta Huffman codes.
        var huffman_precision = MIN_HUFFMAN_BITS + VP8LReadBits(br, NUM_HUFFMAN_BITS)
        var huffman_xsize = VP8LSubSampleSize(xsize, huffman_precision)
        var huffman_ysize = VP8LSubSampleSize(ysize, huffman_precision)
        var huffman_pixs = huffman_xsize * huffman_ysize
        if DecodeImageStream(huffman_xsize, huffman_ysize, 0, dec) == 0:
            return 0
        huffman_image = d[].decoded_data
        d[].huffman_subsample_bits = huffman_precision
        d[].huffman_xsize = huffman_xsize
        var hp = Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=huffman_image
        )
        var i = 0
        while i < huffman_pixs:
            # The huffman data is stored in red and green bytes.
            var group = Int((hp.unsafe_load(i) >> 8) & 0xffff)
            hp.unsafe_store(i, UInt32(group))
            if group >= num_htree_groups_max:
                num_htree_groups_max = group + 1
            i += 1
        # Check the validity of num_htree_groups_max. If it seems too big, use
        # a smaller value for later.
        if num_htree_groups_max > 1000 or num_htree_groups_max > xsize * ysize:
            # Create a mapping from the used indices to the minimal set of
            # used values [0, num_htree_groups)
            var msize = num_htree_groups_max
            var msize_bytes = msize * 4
            if msize_bytes > 65536 * 4:
                VP8LSetError(dec, 4)
                return 0
            mapping = ArenaAlloc(d[].arena, msize)
            if mapping == 0:
                VP8LSetError(dec, 4)
                return 0
            var mp = Pointer[Int32, AnyOrigin[mut=True]](
                unsafe_from_address=mapping
            )
            var k = 0
            while k < msize:
                mp.unsafe_store(k, -1)
                k += 1
            num_htree_groups = 0
            i = 0
            while i < huffman_pixs:
                # Get the current mapping for the group and remap the image.
                var mapped_group = Int(mp.unsafe_load(Int(hp.unsafe_load(i))))
                if mapped_group == -1:
                    mapped_group = num_htree_groups
                    num_htree_groups += 1
                    mp.unsafe_store(Int(hp.unsafe_load(i)), Int32(mapped_group))
                hp.unsafe_store(i, UInt32(mapped_group & 0xffff))
                i += 1
        else:
            num_htree_groups = num_htree_groups_max

    if VP8LIsEndOfStream(br):
        return 0

    if ReadHuffmanCodesHelper(
        color_cache_bits, num_htree_groups, num_htree_groups_max, mapping, dec
    ) == 0:
        return 0
    ok = 1

    # All OK. Finalize pointers.
    d[].huffman_image = huffman_image
    d[].num_htree_groups = num_htree_groups
    _ = ok
    return 1


#------------------------------------------------------------------------------

def GetMetaIndex(image: Int, xsize: Int, bits: Int, x: Int, y: Int) -> Int:
    if bits == 0:
        return 0
    return Int(Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=image
    ).unsafe_load(xsize * (y >> bits) + (x >> bits)))


def GetHtreeGroupForPos(dec: Int, x: Int, y: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var meta_index = GetMetaIndex(
        d[].huffman_image, d[].huffman_xsize, d[].huffman_subsample_bits, x, y
    )
    return GroupPtr(dec, meta_index)


def ApplyInverseTransforms(dec: Int, start_row: Int, num_rows: Int, rows: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var n = d[].next_transform
    var cache_pixs = d[].width * num_rows
    var end_row = start_row + num_rows
    var rows_in = rows
    var rows_out = d[].argb_cache

    # Inverse transforms.
    while n > 0:
        n -= 1
        var transform = TransformPtr(dec, n)
        VP8LInverseTransform(transform, start_row, end_row, rows_in, rows_out)
        rows_in = rows_out
    if rows_in != rows_out:
        # No transform called, hence just copy.
        var sp = Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=rows_in
        )
        var dp = Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=rows_out
        )
        var i = 0
        while i < cache_pixs:
            dp.unsafe_store(i, sp.unsafe_load(i))
            i += 1


# Emit rows without any scaling.
def EmitRows(
    colorspace: Int, row_in: Int, in_stride: Int, mb_w: Int, mb_h: Int, outp: Int, out_stride: Int
) -> Int:
    var lines = mb_h
    var row_out = outp
    var row_in_addr = row_in
    while lines > 0:
        _ = VP8LConvertFromBGRA(row_in_addr, mb_w, colorspace, row_out)
        row_in_addr += in_stride
        row_out += out_stride
        lines -= 1
    return mb_h  # Num rows out == num rows in.


# Processes (transforms & color-converts) the rows decoded after the last call.
def ProcessRows(dec: Int, row: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var rows = d[].pixels + d[].width * d[].last_row * 4
    var num_rows = row - d[].last_row

    if num_rows > 0:  # Emit output.
        var rows_data = d[].argb_cache
        var in_stride = d[].io_width * 4  # in unit of RGBA
        ApplyInverseTransforms(dec, d[].last_row, num_rows, rows)
        var rgba = d[].out + d[].last_out_row * d[].out_stride
        var num_rows_out = EmitRows(
            d[].colorspace,
            rows_data,
            in_stride,
            d[].io_width,
            num_rows,
            rgba,
            d[].out_stride,
        )
        # Update 'last_out_row'.
        d[].last_out_row += num_rows_out

    # Update 'last_row'.
    d[].last_row = row


# Row-processing for the special case when alpha data contains only one
# transform (color indexing), and trivial non-green literals.
def Is8bOptimizable(dec: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if d[].color_cache_size > 0:
        return 0
    # When the Huffman tree contains only one symbol, we can skip the
    # call to ReadSymbol() for red/blue/alpha channels.
    var i = 0
    while i < d[].num_htree_groups:
        var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](
            unsafe_from_address=GroupPtr(dec, i)
        )
        if hc_bits(
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=g[].htree1
            ).unsafe_load(0)
        ) > 0:
            return 0
        if hc_bits(
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=g[].htree2
            ).unsafe_load(0)
        ) > 0:
            return 0
        if hc_bits(
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=g[].htree3
            ).unsafe_load(0)
        ) > 0:
            return 0
        i += 1
    return 1


#------------------------------------------------------------------------------
# Helper functions for fast pattern copy (8b and 32b)


# copy pattern of 1 or 2 uint32_t's
def CopySmallPattern32b(src: Int, dst: Int, length: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=dst)
    var i = 0
    # No explicit SIMD body here: an explicit `load[width=W]` / `store` version
    # was measured and was no faster, because this loop is already a pure
    # contiguous copy that the backend widens on its own.  See README.
    while i < length:
        dp.unsafe_store(i, sp.unsafe_load(i))
        i += 1


def CopyBlock32b(dst: Int, dist: Int, length: Int):
    var src = dst - dist * 4
    if dist >= length:  # no overlap
        CopySmallPattern32b(src, dst, length)
    else:
        # Overlapping: the run is copied in order, which is what makes a short
        # distance repeat correctly.  Both arms are the same copy, so they share
        # one call.
        CopySmallPattern32b(src, dst, length)


def CopyBlock8b(dst: Int, dist: Int, length: Int):
    var src = dst - dist
    if dist >= length:  # no overlap -> use memcpy()
        var i = 0
        while i < length:
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=dst
            ).unsafe_store(
                i,
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=src
                ).unsafe_load(i),
            )
            i += 1
    else:
        var i = 0
        while i < length:
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=dst
            ).unsafe_store(
                i,
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=src
                ).unsafe_load(i),
            )
            i += 1


# `process_mode` stands in for upstream's `process_func` pointer:
#   0 = NULL (no row processing; the transform images)
#   1 = ProcessRows   (the RGBA output path)
#   2 = ExtractAlphaRows (the alpha-plane path, used by VP8LDecodeAlphaImageStream)
# Dispatches a row callback the way upstream's process_func pointer does.
def _ProcessRow(dec: Int, row: Int, process_mode: Int):
    if process_mode == 2:
        ExtractAlphaRows(dec, row)
    else:
        ProcessRows(dec, row)


def DecodeImageData(
    dec: Int,
    data: Int,
    width: Int,
    height: Int,
    last_row: Int,
    process_mode: Int,
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var br = d[].br_addr
    var row = d[].last_pixel // width
    var col = d[].last_pixel % width
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=data)
    var src = d[].last_pixel
    var last_cached = src
    var src_end = width * height  # End of data
    var src_last = width * last_row  # Last pixel to decode
    var len_code_limit = NUM_LITERAL_CODES + NUM_LENGTH_CODES
    var color_cache_limit = len_code_limit + d[].color_cache_size
    var color_cache = d[].color_cache_addr
    var use_cache = 1 if d[].color_cache_size > 0 else 0
    var mask = d[].huffman_mask
    var bp = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)
    var htree_group = (
        GetHtreeGroupForPos(dec, col, row) if src < src_last else 0
    )
    var hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
        unsafe_from_address=htree_group
    )
    # DIVERGENCE (performance, not semantics): upstream re-reads
    # htree0..htree4 and the trivial/packed flags from the HTreeGroup on every
    # literal.  Those fields change only when the loop crosses a meta-Huffman
    # tile, so they are mirrored into locals here and refreshed at exactly the
    # points upstream refreshes them.  Same values, same order of reads.
    var t0 = hg[].htree0
    var t1 = hg[].htree1
    var t2 = hg[].htree2
    var t3 = hg[].htree3
    var t4 = hg[].htree4
    var trivial_code = hg[].is_trivial_code
    var trivial_literal = hg[].is_trivial_literal
    var literal_arb = hg[].literal_arb
    var use_packed = hg[].use_packed_table

    while src < src_last:
        var code = 0
        # Only update when changing tile.
        if (col & mask) == 0:
            htree_group = GetHtreeGroupForPos(dec, col, row)
            hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
                unsafe_from_address=htree_group
            )
            t0 = hg[].htree0
            t1 = hg[].htree1
            t2 = hg[].htree2
            t3 = hg[].htree3
            t4 = hg[].htree4
            trivial_code = hg[].is_trivial_code
            trivial_literal = hg[].is_trivial_literal
            literal_arb = hg[].literal_arb
            use_packed = hg[].use_packed_table
        if trivial_code:
            dp.unsafe_store(src, literal_arb)
            src += 1
            col += 1
            if col >= width:
                col = 0
                row += 1
                if process_mode != 0:
                    if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                        _ProcessRow(dec, row, process_mode)
            continue
        VP8LFillBitWindow(br)
        var advance_by_one = 0
        if use_packed:
            code = ReadPackedSymbols(htree_group, br, data + src * 4)
            if VP8LIsEndOfStream(br):
                break
            if code == PACKED_NON_LITERAL_CODE:
                advance_by_one = 1
        else:
            code = ReadSymbol(t0, br)
        if advance_by_one == 0:
            if VP8LIsEndOfStream(br):
                break
            if code < NUM_LITERAL_CODES:  # Literal
                if trivial_literal:
                    dp.unsafe_store(
                        src, literal_arb | (UInt32(code) << 8)
                    )
                else:
                    var red = ReadSymbol(t1, br)
                    VP8LFillBitWindow(br)
                    var blue = ReadSymbol(t2, br)
                    var alpha = ReadSymbol(t3, br)
                    if VP8LIsEndOfStream(br):
                        break
                    dp.unsafe_store(
                        src,
                        (UInt32(alpha) << 24)
                        | (UInt32(red) << 16)
                        | (UInt32(code) << 8)
                        | UInt32(blue),
                    )
                src += 1
                col += 1
                if col >= width:
                    col = 0
                    row += 1
                    if process_mode != 0:
                        if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                            _ProcessRow(dec, row, process_mode)
                    if use_cache:
                        while last_cached < src:
                            VP8LColorCacheInsert(
                                color_cache, dp.unsafe_load(last_cached)
                            )
                            last_cached += 1
            elif code < len_code_limit:  # Backward reference
                var length_sym = code - NUM_LITERAL_CODES
                var length = GetCopyLength(length_sym, br)
                var dist_symbol = ReadSymbol(t4, br)
                VP8LFillBitWindow(br)
                var dist_code = GetCopyDistance(dist_symbol, br)
                var dist = PlaneCodeToDistance(width, dist_code)

                if VP8LIsEndOfStream(br):
                    break
                if src < dist or src_end - src < length:
                    VP8LSetError(dec, 3)
                    return 0
                CopyBlock32b(data + src * 4, dist, length)
                src += length
                col += length
                while col >= width:
                    col -= width
                    row += 1
                    if process_mode != 0:
                        if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                            _ProcessRow(dec, row, process_mode)
                if col & mask:
                    htree_group = GetHtreeGroupForPos(dec, col, row)
                    hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
                        unsafe_from_address=htree_group
                    )
                    t0 = hg[].htree0
                    t1 = hg[].htree1
                    t2 = hg[].htree2
                    t3 = hg[].htree3
                    t4 = hg[].htree4
                    trivial_code = hg[].is_trivial_code
                    trivial_literal = hg[].is_trivial_literal
                    literal_arb = hg[].literal_arb
                    use_packed = hg[].use_packed_table
                if use_cache:
                    while last_cached < src:
                        VP8LColorCacheInsert(
                            color_cache, dp.unsafe_load(last_cached)
                        )
                        last_cached += 1
            elif code < color_cache_limit:  # Color cache
                var key = code - len_code_limit
                while last_cached < src:
                    VP8LColorCacheInsert(
                        color_cache, dp.unsafe_load(last_cached)
                    )
                    last_cached += 1
                dp.unsafe_store(src, VP8LColorCacheLookup(color_cache, key))
                src += 1
                col += 1
                if col >= width:
                    col = 0
                    row += 1
                    if process_mode != 0:
                        if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                            _ProcessRow(dec, row, process_mode)
            else:  # Not reached
                VP8LSetError(dec, 3)
                return 0

    # br->eos = VP8LIsEndOfStream(br);
    # In incremental decoding:
    # br->eos && src < src_last: if 'br' reached the end of the buffer and
    # 'src_last' has not been reached yet, there is not enough data. 'dec' has to
    # be reset until there is more data.
    # !br->eos && src < src_last: this cannot happen as either the buffer is
    # fully read, either enough has been read to reach 'src_last'.
    # src >= src_last: 'src_last' is reached, all is fine. 'src' can actually go
    # beyond 'src_last' in case the image is cropped and an LZ77 goes further.
    # The buffer might have been enough or there is some left. 'br->eos' does
    # not matter.
    #
    # DIVERGENCE: this port has no incremental mode, so `dec->incremental` is
    # always 0 and upstream's first arm is unreachable.  The second arm keeps
    # `!br->eos || src >= src_last`: reaching `src_last` ends the scan cleanly
    # even when the reader is at end-of-stream.  `process_mode` stands in for
    # upstream's non-NULL `process_func`.
    bp[].eos = VP8LIsEndOfStream(br)
    if src >= src_last or not VP8LIsEndOfStream(br):
        # Process the remaining rows corresponding to last row-block.
        if process_mode != 0:
            _ProcessRow(dec, row if row > last_row else last_row, process_mode)
        d[].status = 0  # VP8_STATUS_OK
        d[].last_pixel = src  # end-of-scan marker
        return 1
    # if not incremental, and we are past the end of buffer (eos=1), then this
    # is a real bitstream error.
    VP8LSetError(dec, 3)
    return 0


# -----------------------------------------------------------------------------
# VP8LTransform


# For security reason, we need to remap the color map to span
# the total possible bundled values, and not just the num_colors.
def ExpandColorMap(num_colors: Int, dec: Int, tidx: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=TransformPtr(dec, tidx)
    )
    var final_num_colors = 1 << (8 >> t[].bits)
    var new_color_map = ArenaAlloc(d[].arena, final_num_colors)
    if new_color_map == 0:
        return 0
    var data = Pointer[UInt8, AnyOrigin[mut=True]](
        unsafe_from_address=t[].data
    )
    var new_data = Pointer[UInt8, AnyOrigin[mut=True]](
        unsafe_from_address=new_color_map
    )
    Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=new_color_map
    ).unsafe_store(0, Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=t[].data
    ).unsafe_load(0))
    var i = 4
    while i < 4 * num_colors:
        # Equivalent to VP8LAddPixels(), on a byte-basis.
        new_data.unsafe_store(
            i, UInt8((Int(data.unsafe_load(i)) + Int(new_data.unsafe_load(i - 4))) & 0xff)
        )
        i += 1
    while i < 4 * final_num_colors:
        new_data.unsafe_store(i, UInt8(0))  # black tail.
        i += 1
    t[].data = new_color_map
    return 1


def ReadTransform(dec: Int, xsize: Int, ysize: Int) -> Tuple[Int, Int]:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var br = d[].br_addr
    var transform = TransformPtr(dec, d[].next_transform)
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var type_ = VP8LReadBits(br, 2)

    # Each transform type can only be present once in the stream.
    if (d[].transforms_seen & (1 << type_)) != 0:
        return (0, 0)  # Already there, let's not accept the second same.
    d[].transforms_seen = d[].transforms_seen | (1 << type_)

    t[].type = type_
    t[].xsize = xsize
    t[].ysize = ysize
    t[].data = 0
    t[].bits = 0
    d[].next_transform += 1

    var out_xsize = xsize
    if type_ == PREDICTOR_TRANSFORM or type_ == CROSS_COLOR_TRANSFORM:
        t[].bits = MIN_TRANSFORM_BITS + VP8LReadBits(br, NUM_TRANSFORM_BITS)
        var data = 0
        if DecodeImageStream(
            VP8LSubSampleSize(t[].xsize, t[].bits),
            VP8LSubSampleSize(t[].ysize, t[].bits),
            0,
            dec,
        ) == 0:
            return (0, 0)
        data = d[].decoded_data
        t[].data = data
    elif type_ == COLOR_INDEXING_TRANSFORM:
        var num_colors = VP8LReadBits(br, 8) + 1
        var bits = 0
        if num_colors > 16:
            bits = 0
        elif num_colors > 4:
            bits = 1
        elif num_colors > 2:
            bits = 2
        else:
            bits = 3
        out_xsize = VP8LSubSampleSize(t[].xsize, bits)
        t[].bits = bits
        var data = 0
        if DecodeImageStream(num_colors, 1, 0, dec) == 0:
            return (0, 0)
        data = d[].decoded_data
        t[].data = data
        if ExpandColorMap(num_colors, dec, d[].next_transform - 1) == 0:
            VP8LSetError(dec, 4)
            return (0, 0)
    elif type_ == SUBTRACT_GREEN_TRANSFORM:
        pass

    return (1, out_xsize)


def InitMetadata(dec: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    d[].color_cache_size = 0
    d[].huffman_image = 0
    d[].huffman_xsize = 0
    d[].huffman_subsample_bits = 0
    d[].huffman_mask = 0
    d[].num_htree_groups = 0
    d[].htree_groups = 0


def ClearMetadata(dec: Int):
    # Upstream frees the metadata here; the arena is a monotone bump, so only
    # the reachable metadata is reset.
    InitMetadata(dec)


def UpdateDecoder(dec: Int, width: Int, height: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var num_bits = d[].huffman_subsample_bits
    d[].width = width
    d[].height = height
    d[].huffman_xsize = VP8LSubSampleSize(width, num_bits)
    d[].huffman_mask = -1 if num_bits == 0 else (1 << num_bits) - 1


# Returns 1 on success, 0 on failure.  Upstream's `uint32_t** decoded_data`
# out-parameter becomes the VP8LDecoder.decoded_data field, because Mojo cannot
# take the address of a local.
def DecodeImageStream(
    xsize: Int, ysize: Int, is_level0: Int, dec: Int
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var br = d[].br_addr
    var ok = 1
    var transform_xsize = xsize
    var transform_ysize = ysize
    var data = 0
    var color_cache_bits = 0
    d[].decoded_data = 0
    # Read the transforms (may recurse).
    if is_level0:
        while ok and VP8LReadBits(br, 1):
            var r = ReadTransform(dec, transform_xsize, transform_ysize)
            ok = r[0]
            if ok:
                transform_xsize = r[1]

    # Color cache
    if ok and VP8LReadBits(br, 1):
        color_cache_bits = VP8LReadBits(br, 4)
        ok = 1 if (color_cache_bits >= 1 and color_cache_bits <= MAX_CACHE_BITS) else 0
        if ok == 0:
            VP8LSetError(dec, 3)
            return 0

    # Read the Huffman codes (may recurse).
    if ok:
        ok = ReadHuffmanCodes(dec, transform_xsize, transform_ysize, color_cache_bits, is_level0)
    if ok == 0:
        VP8LSetError(dec, 3)
        return 0

    # Finish setting up the color-cache
    if color_cache_bits > 0:
        d[].color_cache_size = 1 << color_cache_bits
        var colors = ArenaAlloc(d[].arena, 1 << color_cache_bits)
        if colors == 0:
            VP8LSetError(dec, 4)
            return 0
        # WebPSafeCalloc zero-fills the cache; the arena is reused, so do it here.
        ArenaZero(d[].arena, colors, 1 << color_cache_bits)
        if VP8LColorCacheInit(d[].color_cache_addr, color_cache_bits, colors) == 0:
            VP8LSetError(dec, 4)
            return 0
    else:
        d[].color_cache_size = 0

    UpdateDecoder(dec, transform_xsize, transform_ysize)

    if is_level0:  # level 0 complete
        d[].state = 2
        return 1

    var total_size = transform_xsize * transform_ysize
    data = ArenaAlloc(d[].arena, total_size)
    if data == 0:
        VP8LSetError(dec, 4)
        return 0

    # Use the Huffman trees to decode the LZ77 encoded data.
    ok = DecodeImageData(dec, data, transform_xsize, transform_ysize, transform_ysize, 0)
    ok = ok and (1 - VP8LIsEndOfStream(br))

    if ok == 0:
        return 0
    d[].decoded_data = data
    d[].last_pixel = 0  # Reset for future DECODE_DATA_FUNC() calls.
    if not is_level0:
        ClearMetadata(dec)  # Clean up temporary data behind.
    return 1


#------------------------------------------------------------------------------
# Allocate internal buffers dec->pixels and dec->argb_cache.
def AllocateInternalBuffers32b(dec: Int, final_width: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var num_pixels = d[].width * d[].height
    # Scratch buffer corresponding to top-prediction row for transforming the
    # first row in the row-blocks. Not needed for paletted alpha.
    var cache_top_pixels = final_width
    # Scratch buffer for temporary BGRA storage. Not needed for paletted alpha.
    var cache_pixels = final_width * NUM_ARGB_CACHE_ROWS
    var total_num_pixels = num_pixels + cache_top_pixels + cache_pixels
    var base = ArenaAlloc(d[].arena, total_num_pixels)
    if base == 0:
        return 0
    d[].pixels = base
    d[].argb_cache = base + (num_pixels + cache_top_pixels) * 4
    return 1


def AllocateInternalBuffers8b(dec: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var total_num_pixels = d[].width * d[].height
    var base = ArenaAlloc(d[].arena, total_num_pixels)
    if base == 0:
        return 0
    d[].pixels = base
    d[].argb_cache = 0
    return 1


#------------------------------------------------------------------------------


def VP8LDecodeHeader(dec: Int, data: Int, data_size: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var width = 0
    var height = 0
    var has_alpha = 0
    d[].status = 0
    VP8LInitBitReader(d[].br_addr, data, data_size)
    if ReadImageInfo(
        d[].br_addr, d[].tmp_scratch, d[].tmp_scratch + 8, d[].tmp_scratch + 16
    ) == 0:
        VP8LSetError(dec, 3)
        return 0
    width = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=d[].tmp_scratch)[0]
    height = Pointer[Int, AnyOrigin[mut=True]](
        unsafe_from_address=d[].tmp_scratch + 8
    )[0]
    has_alpha = Pointer[Int, AnyOrigin[mut=True]](
        unsafe_from_address=d[].tmp_scratch + 16
    )[0]
    d[].state = 1
    d[].io_width = width
    d[].io_height = height
    d[].io_crop_bottom = height
    if DecodeImageStream(width, height, 1, dec) == 0:
        return 0
    return 1


def VP8LDecodeImage(dec: Int) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if d[].state != 3:
        d[].state = 3
        if AllocateInternalBuffers32b(dec, d[].io_width) == 0:
            return 0
    d[].last_row = 0
    d[].last_pixel = 0
    d[].last_out_row = 0
    if DecodeImageData(dec, d[].pixels, d[].width, d[].height, d[].io_crop_bottom, 1) == 0:
        return 0
    return 1


#------------------------------------------------------------------------------
# Alpha plane decoding (src/dec/alpha_dec.c on top of the VP8L decoder).


def AlphaApplyFilter(dec: Int, first_row: Int, last_row: Int, outp: Int, stride: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if d[].alpha_filter != WEBP_FILTER_NONE:
        var y = first_row
        var prev_line = d[].prev_line
        var out_off = outp
        while y < last_row:
            WebPUnfilter(d[].alpha_filter, prev_line, out_off, out_off, stride)
            prev_line = out_off
            out_off += stride
            y += 1
        d[].prev_line = prev_line


def ExtractAlphaRows(dec: Int, last_row: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var cur_row = d[].last_row
    var num_rows = last_row - cur_row
    var in_addr = d[].pixels + d[].width * cur_row * 4

    while num_rows > 0:
        var num_rows_to_process = num_rows
        if num_rows > NUM_ARGB_CACHE_ROWS:
            num_rows_to_process = NUM_ARGB_CACHE_ROWS
        var width = d[].io_width  # the final width (!= dec->width)
        var cache_pixs = width * num_rows_to_process
        var dst = d[].alpha_output + width * cur_row
        var src = d[].argb_cache
        ApplyInverseTransforms(dec, cur_row, num_rows_to_process, in_addr)
        # Extract alpha (which is stored in the green plane).
        var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=dst)
        var k = 0
        while k < cache_pixs:
            dp.unsafe_store(k, UInt8((sp.unsafe_load(k) >> 8) & 0xff))
            k += 1
        AlphaApplyFilter(dec, cur_row, cur_row + num_rows_to_process, dst, width)
        num_rows -= num_rows_to_process
        in_addr += num_rows_to_process * d[].width * 4
        cur_row += num_rows_to_process
    d[].last_row = last_row
    d[].last_out_row = last_row


def ExtractPalettedAlphaRows(dec: Int, last_row: Int):
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    # For vertical and gradient filtering, we need to decode the part above the
    # crop_top row, in order to have the correct spatial predictors.
    var top_row = d[].last_row
    if (
        d[].alpha_filter == WEBP_FILTER_NONE
        or d[].alpha_filter == WEBP_FILTER_HORIZONTAL
    ):
        top_row = 0
    var first_row = top_row
    if d[].last_row < top_row:
        first_row = d[].last_row
    if last_row > first_row:
        # Special method for paletted alpha data. We only process the cropped
        # area.
        var width = d[].io_width
        var outp = d[].alpha_output + width * first_row
        var inp = d[].pixels + d[].width * first_row * 4
        VP8LColorIndexInverseTransformAlpha(
            TransformPtr(dec, 0), first_row, last_row, inp, outp
        )
        AlphaApplyFilter(dec, first_row, last_row, outp, width)
    d[].last_row = last_row
    d[].last_out_row = last_row


def DecodeAlphaData(dec: Int, data: Int, width: Int, height: Int, last_row: Int) -> Int:
    var ok = 1
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var br = d[].br_addr
    var row = d[].last_pixel // width
    var col = d[].last_pixel % width
    var pos = d[].last_pixel  # current position
    var end = width * height  # End of data
    var last = width * last_row  # Last pixel to decode
    var len_code_limit = NUM_LITERAL_CODES + NUM_LENGTH_CODES
    var mask = d[].huffman_mask
    var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=data)
    var htree_group = (
        GetHtreeGroupForPos(dec, col, row) if pos < last else 0
    )
    var hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
        unsafe_from_address=htree_group
    )
    var bp = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=br)

    while bp[].eos == 0 and pos < last:
        var code = 0
        # Only update when changing tile.
        if (col & mask) == 0:
            htree_group = GetHtreeGroupForPos(dec, col, row)
            hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
                unsafe_from_address=htree_group
            )
        VP8LFillBitWindow(br)
        code = ReadSymbol(hg[].htree0, br)
        if code < NUM_LITERAL_CODES:  # Literal
            dp.unsafe_store(pos, UInt8(code))
            pos += 1
            col += 1
            if col >= width:
                col = 0
                row += 1
                if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                    ExtractPalettedAlphaRows(dec, row)
        elif code < len_code_limit:  # Backward reference
            var length_sym = code - NUM_LITERAL_CODES
            var length = GetCopyLength(length_sym, br)
            var dist_symbol = ReadSymbol(hg[].htree4, br)
            VP8LFillBitWindow(br)
            var dist_code = GetCopyDistance(dist_symbol, br)
            var dist = PlaneCodeToDistance(width, dist_code)
            if pos >= dist and end - pos >= length:
                CopyBlock8b(data + pos, dist, length)
            else:
                ok = 0
                break
            pos += length
            col += length
            while col >= width:
                col -= width
                row += 1
                if row <= last_row and (row % NUM_ARGB_CACHE_ROWS == 0):
                    ExtractPalettedAlphaRows(dec, row)
            if pos < last and (col & mask):
                htree_group = GetHtreeGroupForPos(dec, col, row)
                hg = Pointer[HTreeGroup, AnyOrigin[mut=True]](
                    unsafe_from_address=htree_group
                )
        else:  # Not reached
            ok = 0
            break
        bp[].eos = VP8LIsEndOfStream(br)
    # Process the remaining rows corresponding to last row-block.
    ExtractPalettedAlphaRows(dec, row if row > last_row else last_row)

    bp[].eos = VP8LIsEndOfStream(br)
    if ok == 0 or (bp[].eos and pos < end):
        VP8LSetError(dec, 3)
        return 0
    d[].last_pixel = pos
    return 1


def VP8LDecodeAlphaImageStream(
    dec: Int,
    data: Int,
    data_size: Int,
    width: Int,
    height: Int,
    filter: Int,
    out_ptr: Int,
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    d[].width = width
    d[].height = height
    d[].io_width = width
    d[].io_height = height
    d[].io_crop_bottom = height
    d[].alpha_filter = filter
    d[].alpha_output = out_ptr
    d[].prev_line = 0
    d[].status = 0
    d[].last_row = 0
    d[].last_out_row = 0
    d[].last_pixel = 0

    VP8LInitBitReader(d[].br_addr, data, data_size)
    if DecodeImageStream(width, height, 1, dec) == 0:
        return 0

    # Special case: if alpha data uses only the color indexing transform and
    # doesn't use color cache (a frequent case), we will use DecodeAlphaData()
    # method that only needs allocation of 1 byte per pixel.
    var t0 = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=TransformPtr(dec, 0)
    )
    if (
        d[].next_transform == 1
        and t0[].type == COLOR_INDEXING_TRANSFORM
        and Is8bOptimizable(dec)
    ):
        d[].use_8b_decode = 1
        if AllocateInternalBuffers8b(dec) == 0:
            return 0
    else:
        d[].use_8b_decode = 0
        if AllocateInternalBuffers32b(dec, width) == 0:
            return 0

    if d[].last_row >= height:
        return 1  # done

    if d[].use_8b_decode:
        return DecodeAlphaData(dec, d[].pixels, d[].width, d[].height, height)
    if DecodeImageData(dec, d[].pixels, d[].width, d[].height, height, 2) == 0:
        return 0
    return 1


#------------------------------------------------------------------------------
# State initialisation: the C equivalent is VP8LNew() + WebPSafeCalloc, with
# every WebPSafeMalloc replaced by an arena region reserved up front.


def VP8LDecoderInit(
    dec: Int,
    br: Int,
    transforms: Int,
    color_cache: Int,
    huffman_tables: Int,
    arena: Int,
    arena_words: Int,
    tmp_scratch: Int,
) -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    d[].status = 0
    d[].state = 0
    d[].width = 0
    d[].height = 0
    d[].next_transform = 0
    d[].transforms_seen = 0
    d[].last_row = 0
    d[].last_pixel = 0
    d[].last_out_row = 0
    d[].br_addr = br
    d[].transforms_addr = transforms
    d[].color_cache_addr = color_cache
    d[].huffman_tables_addr = huffman_tables
    d[].color_cache_size = 0
    d[].huffman_image = 0
    d[].huffman_xsize = 0
    d[].huffman_subsample_bits = 0
    d[].huffman_mask = 0
    d[].num_htree_groups = 0
    d[].htree_groups = 0
    d[].pixels = 0
    d[].argb_cache = 0
    d[].arena = arena
    d[].scratch_table = arena + ARENA_HEADER_WORDS * 4
    d[].scratch_sorted = d[].scratch_table + RES_SCRATCH_TABLE * 4
    d[].scratch_cl = d[].scratch_sorted + RES_SCRATCH_SORTED * 4
    d[].scratch_ccl = d[].scratch_cl + RES_SCRATCH_CL * 4
    d[].scratch_length_tables = d[].scratch_ccl + RES_SCRATCH_CCL * 4
    d[].transforms_addr = d[].scratch_length_tables + RES_LENGTH_TABLES * 4
    d[].io_width = 0
    d[].io_height = 0
    d[].io_crop_bottom = 0
    d[].colorspace = 0
    d[].out = 0
    d[].out_stride = 0
    d[].use_8b_decode = 0
    d[].alpha_filter = 0
    d[].prev_line = 0
    d[].alpha_output = 0
    d[].tmp_scratch = tmp_scratch
    d[].decoded_data = 0
    ArenaInit(arena, arena_words)
    # Zero the transforms and the huffman tables header.
    ArenaZero(arena, d[].transforms_addr, RES_TRANSFORMS)
    return 1
