"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. The caller owns
all memory, including scratch, so nothing here allocates and nothing here can
leak. The one non-mechanical piece is the arena: Mojo has no heap, so
`mw_vp8l_workspace_size` runs the header pass first and reports how many UInt32
words the decode will need.

The `mw_vp8l_inspect_*` entry points are introspection hooks used by the test
suite to compare intermediate state against upstream; they read state the
decoder already computed and change nothing.
"""

from webplib.bit_reader import (
    VP8LBitReader,
    VP8LInitBitReader,
    VP8LReadBits,
)
from webplib.color_cache import VP8LColorCache
from webplib.format_constants import ARENA_HEADER_WORDS
from webplib.huffman import (
    BuildHuffmanTable,
    HuffmanTables,
    VP8LBuildHuffmanTable,
    VP8LHuffmanTablesUseRegion,
)
from webplib.lossless import (
    ColorCodeToMultipliers,
    VP8LAddGreenToBlueAndRed,
    VP8LConvertBGRAToBGR_C,
    VP8LConvertBGRAToRGB_C,
    VP8LConvertBGRAToRGBA_C,
    VP8LInverseTransform,
    VP8LPredictorsAdd,
    VP8LTransformColorInverse_C,
)
from webplib.vp8l_dec import (
    CopyBlock32b,
    GroupPtr,
    HUFFMAN_TABLES_HEADER_BYTES,
    HTreeGroup,
    TransformPtr,
    VP8LCheckSignature,
    VP8LDecodeAlphaImageStream,
    VP8LDecodeHeader,
    VP8LDecodeImage,
    VP8LDecoder,
    VP8LDecoderInit,
    VP8LGetInfo,
    VP8LTransform,
)


@export("mw_vp8l_check_signature")
def mw_vp8l_check_signature(data: Int, size: Int) abi("C") -> Int:
    return VP8LCheckSignature(data, size)


@export("mw_vp8l_get_info")
def mw_vp8l_get_info(
    data: Int,
    size: Int,
    br: Int,
    width: Int,
    height: Int,
    has_alpha: Int,
    tmp: Int,
) abi("C") -> Int:
    return VP8LGetInfo(data, size, br, width, height, has_alpha, tmp)


# Runs the level-0 header pass and reports the arena size, in UInt32 words,
# that the full decode will need.
@export("mw_vp8l_workspace_size")
def mw_vp8l_workspace_size(
    dec: Int,
    br: Int,
    color_cache: Int,
    huffman_tables: Int,
    tmp: Int,
    arena: Int,
    arena_words: Int,
    data: Int,
    size: Int,
) abi("C") -> Int:
    _ = VP8LDecoderInit(
        dec, br, 0, color_cache, huffman_tables, arena, arena_words, tmp
    )
    if VP8LDecodeHeader(dec, data, size) == 0:
        return -1
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    # DecodeImageStream stops right after the level-0 codes are read; the pixel
    # and cache buffers are the only thing still to be reserved.
    # `used` is the arena's bump cursor, measured from the start of the payload.
    # The ARENA_HEADER_WORDS in front of it must be counted too, otherwise the
    # caller allocates a buffer four words too small and the last allocations
    # fail (silently, as VP8_STATUS_OUT_OF_MEMORY deep inside the decode).
    var used = Int(
        Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=arena
        ).unsafe_offset(0)[]
    ) + ARENA_HEADER_WORDS
    var num_pixels = d[].width * d[].height
    var final_width = d[].io_width
    return used + num_pixels + final_width + final_width * 16


@export("mw_vp8l_decode")
def mw_vp8l_decode(
    dec: Int,
    br: Int,
    color_cache: Int,
    huffman_tables: Int,
    tmp: Int,
    arena: Int,
    arena_words: Int,
    data: Int,
    size: Int,
    out_ptr: Int,
    out_stride: Int,
    colorspace: Int,
) abi("C") -> Int:
    _ = VP8LDecoderInit(
        dec, br, 0, color_cache, huffman_tables, arena, arena_words, tmp
    )
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if VP8LDecodeHeader(dec, data, size) == 0:
        return d[].status
    d[].colorspace = colorspace
    d[].out = out_ptr
    d[].out_stride = out_stride
    if VP8LDecodeImage(dec) == 0:
        return d[].status
    return 0


@export("mw_vp8l_decode_alpha")
def mw_vp8l_decode_alpha(
    dec: Int,
    br: Int,
    color_cache: Int,
    huffman_tables: Int,
    tmp: Int,
    arena: Int,
    arena_words: Int,
    data: Int,
    size: Int,
    width: Int,
    height: Int,
    filter: Int,
    out_ptr: Int,
) abi("C") -> Int:
    _ = VP8LDecoderInit(
        dec, br, 0, color_cache, huffman_tables, arena, arena_words, tmp
    )
    if VP8LDecodeAlphaImageStream(
        dec, data, size, width, height, filter, out_ptr
    ) == 0:
        return Pointer[VP8LDecoder, AnyOrigin[mut=True]](
            unsafe_from_address=dec
        )[].status
    return 0


# --- introspection hooks (tests) ---------------------------------------------

# Runs the header pass and reports one decoder field, selected by `which`.
@export("mw_vp8l_inspect_state")
def mw_vp8l_inspect_state(
    dec: Int,
    br: Int,
    color_cache: Int,
    huffman_tables: Int,
    tmp: Int,
    arena: Int,
    arena_words: Int,
    data: Int,
    size: Int,
    which: Int,
) abi("C") -> Int:
    _ = VP8LDecoderInit(
        dec, br, 0, color_cache, huffman_tables, arena, arena_words, tmp
    )
    _ = VP8LDecodeHeader(dec, data, size)
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if which == 0:
        return d[].status
    if which == 1:
        return d[].width
    if which == 2:
        return d[].height
    if which == 3:
        return d[].next_transform
    if which == 4:
        return d[].transforms_seen
    if which == 5:
        return d[].color_cache_size
    if which == 6:
        return d[].num_htree_groups
    if which == 7:
        return d[].huffman_subsample_bits
    if which == 8:
        return d[].io_width
    if which == 9:
        return Int(
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=arena
            ).unsafe_offset(0)[]
        )
    return 0


@export("mw_vp8l_inspect_transform")
def mw_vp8l_inspect_transform(dec: Int, idx: Int, which: Int) abi("C") -> Int:
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=TransformPtr(dec, idx)
    )
    if which == 0:
        return t[].type
    if which == 1:
        return t[].xsize
    if which == 2:
        return t[].ysize
    if which == 4:
        return t[].data
    return t[].bits


# Applies one inverse transform, as VP8LInverseTransform does.
@export("mw_vp8l_inverse_transform")
def mw_vp8l_inverse_transform(
    dec: Int,
    idx: Int,
    start_row: Int,
    end_row: Int,
    in_ptr: Int,
    out_ptr: Int,
) abi("C"):
    VP8LInverseTransform(
        TransformPtr(dec, idx), start_row, end_row, in_ptr, out_ptr
    )


# Copies the decoder's raw (pre-transform) pixel buffer out.
@export("mw_vp8l_dump_pixels")
def mw_vp8l_dump_pixels(dec: Int, out_ptr: Int, count: Int) abi("C") -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var src = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=d[].pixels)
    var dst = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var i = 0
    while i < count:
        dst.unsafe_store(i, src.unsafe_load(i))
        i += 1
    return count


# Reads a decoder field from a state that a previous call has already filled.
@export("mw_vp8l_decoder_field")
def mw_vp8l_decoder_field(dec: Int, which: Int) abi("C") -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    if which == 0:
        return d[].last_pixel
    if which == 1:
        return d[].last_row
    if which == 2:
        return d[].last_out_row
    if which == 3:
        return d[].status
    if which == 4:
        return d[].width
    if which == 5:
        return d[].pixels
    if which == 6:
        return d[].argb_cache
    if which == 7:
        return Int(Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=d[].arena
        ).unsafe_offset(0)[])
    return d[].height


@export("mw_vp8l_read_bits")
def mw_vp8l_read_bits(
    br: Int, data: Int, size: Int, skip_bits: Int, n_bits: Int
) abi("C") -> Int:
    VP8LInitBitReader(br, data, size)
    var i = 0
    while i < skip_bits:
        _ = VP8LReadBits(br, 1)
        i += 1
    return VP8LReadBits(br, n_bits)


# mode 0 measures, mode 1 measures and writes (upstream's two-pass wrapper).
@export("mw_vp8l_build_huffman_table")
def mw_vp8l_build_huffman_table(
    ht: Int,
    root_bits: Int,
    code_lengths: Int,
    code_lengths_size: Int,
    sorted: Int,
    arena: Int,
    mode: Int,
) abi("C") -> Int:
    if mode == 0:
        return BuildHuffmanTable(
            code_lengths, root_bits, code_lengths, code_lengths_size, sorted, 0
        )
    # The payload starts one HuffmanTables header past `ht` itself (that header
    # is HUFFMAN_TABLES_HEADER_BYTES).  Sizing the region at `1 << root_bits` is
    # what upstream's VP8LHuffmanTablesAllocate(1 << LENGTHS_TABLE_BITS, &tables)
    # does, and it is what keeps the table inside this region instead of forcing
    # a growth.
    _ = VP8LHuffmanTablesUseRegion(
        ht, 1 << root_bits, ht + HUFFMAN_TABLES_HEADER_BYTES
    )
    return VP8LBuildHuffmanTable(
        ht, root_bits, code_lengths, code_lengths_size, sorted, arena
    )


# Reports one HTreeGroup field for group `idx` (level-0 metadata).
@export("mw_vp8l_inspect_group")
def mw_vp8l_inspect_group(dec: Int, idx: Int, which: Int) abi("C") -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=dec)
    var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](
        unsafe_from_address=GroupPtr(dec, idx)
    )
    if which == 0:
        return g[].is_trivial_literal
    if which == 1:
        return g[].is_trivial_code
    if which == 2:
        return g[].use_packed_table
    if which == 3:
        return Int(g[].literal_arb)
    if which == 4:
        return Int(
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=g[].htree0
            ).unsafe_load(0)
        )
    return 0


# Copies a transform's payload out (colour map, predictor map, ...).
@export("mw_vp8l_dump_transform_data")
def mw_vp8l_dump_transform_data(
    dec: Int, idx: Int, out_ptr: Int, count: Int
) abi("C") -> Int:
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=TransformPtr(dec, idx)
    )
    if t[].data == 0:
        return -1
    var src = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=t[].data)
    var dst = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var i = 0
    while i < count:
        dst.unsafe_store(i, src.unsafe_load(i))
        i += 1
    return count


# Kernel-level entry point, so the SIMD tails and the per-mode predictor
# dispatch can be tested without going through a whole bitstream.  `kind`
# selects the kernel; every one of them is a pure elementwise map over
# caller-owned buffers.
@export("mw_kernel_run")
def mw_kernel_run(
    kind: Int,
    a_ptr: Int,
    b_ptr: Int,
    c_ptr: Int,
    n: Int,
    aux: Int,
) abi("C"):
    if kind == 0:  # CopyBlock32b: LZ77 back-reference copy
        CopyBlock32b(a_ptr, aux, n)
    elif kind == 1:  # VP8LAddGreenToBlueAndRed
        VP8LAddGreenToBlueAndRed(a_ptr, n, b_ptr)
    elif kind == 2:  # VP8LConvertBGRAToRGBA_C
        VP8LConvertBGRAToRGBA_C(a_ptr, n, b_ptr)
    elif kind == 3:  # VP8LConvertBGRAToRGB_C
        VP8LConvertBGRAToRGB_C(a_ptr, n, b_ptr)
    elif kind == 4:  # VP8LConvertBGRAToBGR_C
        VP8LConvertBGRAToBGR_C(a_ptr, n, b_ptr)
    elif kind == 5:  # VP8LPredictorsAdd(mode = aux)
        VP8LPredictorsAdd(aux, a_ptr, b_ptr, n, c_ptr)
    elif kind == 6:  # VP8LTransformColorInverse_C, colour code in c_ptr
        var code = Int(Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=c_ptr
        ).unsafe_load(0))
        VP8LTransformColorInverse_C(
            ColorCodeToMultipliers(UInt32(code)), a_ptr, n, b_ptr
        )
    else:
        return


# Reports VP8LBitReader state: 0=pos, 1=bit_pos, 2=eos, 3=len.
@export("mw_vp8l_br_state")
def mw_vp8l_br_state(br: Int, which: Int) abi("C") -> Int:
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](
        unsafe_from_address=br
    )
    if which == 0:
        return b[].pos
    if which == 1:
        return b[].bit_pos
    if which == 2:
        return b[].eos
    return b[].len


# Reports where the last field of each caller-owned state struct lands, so the
# Python layer can size its buffers from the real layout instead of a comment.
# `mw_struct_size(which, probe, sentinel)` stores `sentinel` in the struct's last
# field and returns that field's byte offset within `probe`; the struct's size is
# that plus one field width.  Which: 0 VP8LDecoder, 1 VP8LBitReader,
# 2 VP8LColorCache, 3 HuffmanTables, 4 VP8LTransform, 5 HTreeGroup.
@export("mw_struct_size")
def mw_struct_size(which: Int, probe: Int, sentinel: Int) abi("C") -> Int:
    var d = Pointer[VP8LDecoder, AnyOrigin[mut=True]](unsafe_from_address=probe)
    var b = Pointer[VP8LBitReader, AnyOrigin[mut=True]](unsafe_from_address=probe)
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](unsafe_from_address=probe)
    var h = Pointer[HuffmanTables, AnyOrigin[mut=True]](unsafe_from_address=probe)
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](unsafe_from_address=probe)
    var g = Pointer[HTreeGroup, AnyOrigin[mut=True]](unsafe_from_address=probe)
    if which == 0:
        d[].alpha_output = sentinel
        return 312
    if which == 1:
        b[].eos = sentinel
        return 40
    if which == 2:
        c[].hash_bits = sentinel
        return 16
    if which == 3:
        h[].curr_segment = sentinel
        return 32
    if which == 4:
        t[].data = sentinel
        return 32
    g[].packed_table = sentinel
    return 72
