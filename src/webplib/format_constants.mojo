"""Port of src/webp/format_constants.h.

Every #define upstream becomes a `comptime` constant of the same name, so the
kernels below read like the C they came from.
"""

comptime VP8L_SIGNATURE_SIZE: Int = 1
comptime VP8L_MAGIC_BYTE: Int = 0x2f
comptime VP8L_IMAGE_SIZE_BITS: Int = 14
comptime VP8L_VERSION_BITS: Int = 3
comptime VP8L_VERSION: Int = 0
comptime VP8L_FRAME_HEADER_SIZE: Int = 5

comptime MAX_PALETTE_SIZE: Int = 256
comptime MAX_CACHE_BITS: Int = 11
comptime HUFFMAN_CODES_PER_META_CODE: Int = 5
comptime ARGB_BLACK: UInt32 = 0xff000000

comptime DEFAULT_CODE_LENGTH: Int = 8
comptime MAX_ALLOWED_CODE_LENGTH: Int = 15

comptime NUM_LITERAL_CODES: Int = 256
comptime NUM_LENGTH_CODES: Int = 24
comptime NUM_DISTANCE_CODES: Int = 40
comptime CODE_LENGTH_CODES: Int = 19

comptime MIN_HUFFMAN_BITS: Int = 2
comptime NUM_HUFFMAN_BITS: Int = 3

comptime MIN_TRANSFORM_BITS: Int = 2
comptime NUM_TRANSFORM_BITS: Int = 3

comptime TRANSFORM_PRESENT: Int = 1
comptime NUM_TRANSFORMS: Int = 4

# VP8LImageTransformType
comptime PREDICTOR_TRANSFORM: Int = 0
comptime CROSS_COLOR_TRANSFORM: Int = 1
comptime SUBTRACT_GREEN_TRANSFORM: Int = 2
comptime COLOR_INDEXING_TRANSFORM: Int = 3

# VP8LReadBits limits
comptime VP8L_MAX_NUM_BIT_READ: Int = 24
comptime VP8L_LBITS: Int = 64
comptime VP8L_WBITS: Int = 32

# Huffman tables
comptime HUFFMAN_TABLE_BITS: Int = 8
comptime HUFFMAN_TABLE_MASK: Int = (1 << HUFFMAN_TABLE_BITS) - 1
comptime LENGTHS_TABLE_BITS: Int = 7
comptime LENGTHS_TABLE_MASK: Int = (1 << LENGTHS_TABLE_BITS) - 1
comptime HUFFMAN_PACKED_BITS: Int = 6
comptime HUFFMAN_PACKED_TABLE_SIZE: Int = 1 << HUFFMAN_PACKED_BITS

# The decoder scratch arena is a caller-owned bump allocator. Its first words are
# the header [cursor, capacity, pad, pad]; payload starts at word
# ARENA_HEADER_WORDS.  This port has no heap, so every WebPSafeMalloc in the C
# decoder becomes a reservation in this one buffer.
comptime ARENA_HEADER_WORDS: Int = 4

# WebPRescaler / mode limits used by the export step
comptime MODE_RGB: Int = 0
comptime MODE_RGBA: Int = 1
comptime MODE_BGR: Int = 2
comptime MODE_BGRA: Int = 3
comptime MODE_ARGB: Int = 4
comptime MODE_RGBA_4444: Int = 5
comptime MODE_RGB_565: Int = 6
comptime MODE_rgbA: Int = 7
comptime MODE_bgrA: Int = 8
comptime MODE_Argb: Int = 9
comptime MODE_rgbA_4444: Int = 10
comptime MODE_YUV: Int = 11
comptime MODE_YUVA: Int = 12
