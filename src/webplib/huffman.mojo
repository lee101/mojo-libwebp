"""Port of the decoder half of src/utils/huffman_utils.c.

Only what the VP8L decoder calls is here: the HuffmanCode accessors, the tables
segment chain, and BuildHuffmanTable / VP8LBuildHuffmanTable.  The encoder-side
half of upstream's file (VP8LCreateHuffmanTree, GenerateOptimalTree,
SortHuffmanTrees, OptimizeHuffmanForRle and the RLE token coders) is NOT ported:
nothing in the decoder path reaches it, so carrying it would be untested code.

Divergences forced by the FFI (all also listed in the README):
  * HuffmanCode is a 4-byte C struct.  The arena stores it as one UInt32 word
    laid out exactly as the little-endian struct does: bits in byte 0, value in
    bytes 1..2.  hc_bits / hc_value / hc_pack are the only accessors.
  * HuffmanTables segments come from the decoder arena instead of
    WebPSafeMalloc; the "current table" cursor and the segment-chain growth rule
    are unchanged.
  * The `sorted[]` scratch buffer of VP8LBuildHuffmanTable comes from the arena
    instead of the C stack / heap.
"""

from std.utils import StaticTuple

from webplib.format_constants import (
    ARENA_HEADER_WORDS,
    MAX_ALLOWED_CODE_LENGTH,
)

comptime T16 = StaticTuple[Int, 16]

# HuffmanTablesSegment is four Int fields: 4 words = 32 bytes (measured, not
# assumed -- tests/test_units.py pins it through mw_struct_size).  Its record
# is reserved from the arena alongside the payload it points at, so the two
# must not overlap.
comptime HUFFMAN_TABLES_SEGMENT_WORDS: Int = 4
comptime HUFFMAN_TABLES_SEGMENT_BYTES: Int = HUFFMAN_TABLES_SEGMENT_WORDS * 8


# ---------------------------------------------------------------- HuffmanCode
def hc_pack(bits: Int, value: Int) -> UInt32:
    return UInt32(bits & 0xff) | (UInt32(value & 0xffff) << 8)


def hc_bits(entry: UInt32) -> Int:
    return Int(entry & 0xff)


def hc_value(entry: UInt32) -> Int:
    return Int((entry >> 8) & 0xffff)


# ------------------------------------------------------------- tables chain
struct HuffmanTablesSegment:
    var start: Int       # address of the segment
    var curr_table: Int  # address we are writing into
    var size: Int        # number of HuffmanCode entries
    var next: Int        # address of the next segment record, 0 if none


struct HuffmanTables:
    var root: HuffmanTablesSegment
    var curr_segment: Int  # address of the segment record


struct HuffmanTreeToken:
    var code: Int
    var extra_bits: Int


struct HuffmanTree:
    var total_count: Int
    var value: Int
    var pool_index_left: Int
    var pool_index_right: Int


struct HuffmanTreeCode:
    var num_symbols: Int
    var codes: Int        # address of num_symbols UInt32 entries
    var code_lengths: Int  # address of num_symbols UInt8-as-UInt32 entries


# Allocates a HuffmanTables with 'size' contiguous HuffmanCodes. Returns 0 on
# memory allocation error, 1 otherwise.  'arena' is the address of the shared
# 4-word arena header: [cursor, capacity, pad, pad]; payloads start at +16.
def VP8LHuffmanTablesAllocate(size: Int, huffman_tables: Int, arena: Int) -> Int:
    var ht = Pointer[HuffmanTables, AnyOrigin[mut=True]](
        unsafe_from_address=huffman_tables
    )
    var a = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=arena
    )
    var used = Int(a.unsafe_offset(0)[])
    var cap = Int(a.unsafe_offset(1)[])
    # The payload starts ARENA_HEADER_WORDS in, so the usable capacity is
    # cap - ARENA_HEADER_WORDS.  Comparing against the raw cap would hand out
    # the last ARENA_HEADER_WORDS words and write past the end of the buffer.
    if used + size + ARENA_HEADER_WORDS > cap:
        return 0
    a.unsafe_offset(0)[] = UInt32(used + size)
    # Have 'segment' point to the first segment for now, 'root'.
    return VP8LHuffmanTablesUseRegion(
        huffman_tables, size, arena + 16 + used * 4
    )


# Points 'huffman_tables' at an already-reserved region.  This is how
# ReadHuffmanCodeLengths gets the C code's stack-allocated local table.
def VP8LHuffmanTablesUseRegion(huffman_tables: Int, size: Int, start: Int) -> Int:
    var ht = Pointer[HuffmanTables, AnyOrigin[mut=True]](
        unsafe_from_address=huffman_tables
    )
    ht[].root.start = start
    ht[].root.curr_table = start
    ht[].root.size = size
    ht[].root.next = 0
    ht[].curr_segment = huffman_tables
    return 1


def VP8LHuffmanTablesDeallocate(huffman_tables: Int):
    var ht = Pointer[HuffmanTables, AnyOrigin[mut=True]](
        unsafe_from_address=huffman_tables
    )
    ht[].curr_segment = 0


# Returns reverse(reverse(key, len) + 1, len), where reverse(key, len) is the
# bit-wise reversal of the len least significant bits of key.
def GetNextKey(key: Int, len: Int) -> Int:
    var step = 1 << (len - 1)
    while key & step:
        step >>= 1
    if step == 0:
        return key
    return (key & (step - 1)) + step


# Stores code in table[0], table[step], table[2*step], ..., table[end].
# Assumes that end is an integer multiple of step.
def ReplicateValue(table: Int, step: Int, end_in: Int, code: UInt32):
    var t = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=table)
    var end = end_in
    while end > 0:
        end -= step
        t.unsafe_store(end, code)


# Returns the table width of the next 2nd level table. count is the histogram
# of bit lengths for the remaining symbols, len is the code length of the next
# processed symbol
def NextTableBitSize(count: T16, len_in: Int, root_bits: Int) -> Int:
    var left = 1 << (len_in - root_bits)
    var len = len_in
    while len < MAX_ALLOWED_CODE_LENGTH:
        left -= count[len]
        if left <= 0:
            break
        len += 1
        left <<= 1
    return len - root_bits


# sorted[code_lengths_size] is a pre-allocated array for sorting symbols by
# code length.  When 'write' is 0 nothing is stored: that is the measuring pass
# upstream performs with root_table == NULL.
def BuildHuffmanTable(
    root_table: Int,
    root_bits: Int,
    code_lengths: Int,
    code_lengths_size: Int,
    sorted: Int,
    write: Int,
) -> Int:
    var table = root_table  # next available space in table
    var total_size = 1 << root_bits  # total size root table + 2nd level table
    var symbol: Int
    # number of codes of each length:
    var count = T16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # offsets in sorted table for each length:
    var offset = T16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var cl = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=code_lengths
    )
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=sorted)
    var root = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=root_table)

    # Build histogram of code lengths.
    symbol = 0
    while symbol < code_lengths_size:
        var l = Int(cl.unsafe_load(symbol))
        if l > MAX_ALLOWED_CODE_LENGTH:
            return 0
        count[l] = count[l] + 1
        symbol += 1

    # Error, all code lengths are zeros.
    if count[0] == code_lengths_size:
        return 0

    # Generate offsets into sorted symbol table by code length.
    offset[1] = 0
    var len = 1
    while len < MAX_ALLOWED_CODE_LENGTH:
        if count[len] > (1 << len):
            return 0
        offset[len + 1] = offset[len] + count[len]
        len += 1

    # Sort symbols by length, by symbol order within each length.
    symbol = 0
    while symbol < code_lengths_size:
        var l = Int(cl.unsafe_load(symbol))
        if l > 0:
            if offset[l] >= code_lengths_size:
                return 0
            if write:
                sp.unsafe_store(offset[l], UInt32(symbol))
                offset[l] = offset[l] + 1
            else:
                offset[l] = offset[l] + 1
        symbol += 1

    # Special case code with only one value.
    if offset[MAX_ALLOWED_CODE_LENGTH] == 1:
        if write:
            ReplicateValue(table, 1, total_size, hc_pack(0, Int(sp.unsafe_load(0))))
        return total_size

    var step: Int
    var low = 0xffffffff  # low bits for current root entry
    var mask = total_size - 1  # mask for low bits
    var key = 0  # reversed prefix code
    var num_nodes = 1  # number of Huffman tree nodes
    var num_open = 1  # number of open branches in current tree level
    var table_bits = root_bits  # key length of current table
    var table_size = 1 << table_bits  # size of current table
    symbol = 0
    # Fill in root table.
    len = 1
    step = 2
    while len <= root_bits:
        num_open <<= 1
        num_nodes += num_open
        num_open -= count[len]
        if num_open < 0:
            return 0
        while count[len] > 0:
            count[len] = count[len] - 1
            if write:
                var code = hc_pack(len, Int(sp.unsafe_load(symbol)))
                ReplicateValue(table + key * 4, step, table_size, code)
            symbol += 1
            key = GetNextKey(key, len)
        len += 1
        step <<= 1

    # Fill in 2nd level tables and add pointers to root table.
    len = root_bits + 1
    step = 2
    while len <= MAX_ALLOWED_CODE_LENGTH:
        num_open <<= 1
        num_nodes += num_open
        num_open -= count[len]
        if num_open < 0:
            return 0
        while count[len] > 0:
            count[len] = count[len] - 1
            if (key & mask) != low:
                if write:
                    table += table_size * 4
                table_bits = NextTableBitSize(count, len, root_bits)
                table_size = 1 << table_bits
                total_size += table_size
                low = key & mask
                if write:
                    root.unsafe_store(
                        low,
                        hc_pack(
                            table_bits + root_bits,
                            (table - root_table) // 4 - low,
                        ),
                    )
            if write:
                var code2 = hc_pack(
                    len - root_bits, Int(sp.unsafe_load(symbol))
                )
                ReplicateValue(
                    table + (key >> root_bits) * 4, step, table_size, code2
                )
            symbol += 1
            key = GetNextKey(key, len)
        len += 1
        step <<= 1

    # Check if tree is full.
    if num_nodes != 2 * offset[MAX_ALLOWED_CODE_LENGTH] - 1:
        return 0

    return total_size


# Builds a Huffman lookup table assuming code lengths are in symbol order.
# Returns built table size or 0 in case of error (invalid tree or memory error).
def VP8LBuildHuffmanTable(
    huffman_tables: Int,
    root_bits: Int,
    code_lengths: Int,
    code_lengths_size: Int,
    sorted: Int,
    arena: Int,
) -> Int:
    var total_size = BuildHuffmanTable(
        code_lengths, root_bits, code_lengths, code_lengths_size, sorted, 0
    )
    if total_size == 0:
        return total_size

    var ht = Pointer[HuffmanTables, AnyOrigin[mut=True]](
        unsafe_from_address=huffman_tables
    )
    var seg = Pointer[HuffmanTablesSegment, AnyOrigin[mut=True]](
        unsafe_from_address=ht[].curr_segment
    )
    # 'curr_table' and 'start' are byte addresses while 'size' counts HuffmanCode
    # entries, so upstream's element comparison is scaled by 4 on this side.
    # The comparison is `>=`, as upstream has it: a table that exactly fills the
    # segment relocates.  That is safe because the caller reads the table back
    # through `curr_segment->start`, which follows the relocation.
    if seg[].curr_table + total_size * 4 >= seg[].start + seg[].size * 4:
        # If 'huffman_tables' does not have enough memory, allocate a new
        # segment.  The available part is left unused because we need a
        # contiguous buffer.
        var a = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=arena)
        var used = Int(a.unsafe_offset(0)[])
        var cap = Int(a.unsafe_offset(1)[])
        var segment_size = seg[].size
        var next_size = segment_size
        if total_size > segment_size:
            next_size = total_size
        # The segment record (HUFFMAN_TABLES_SEGMENT_WORDS) is written at the
        # cursor before the payload, so it has to be reserved too, not just the
        # payload.  The ARENA_HEADER_WORDS term is the arena's own header, which
        # sits in front of the payload.
        if used + next_size + HUFFMAN_TABLES_SEGMENT_WORDS + ARENA_HEADER_WORDS > cap:
            return 0
        # The record occupies the cursor's first HUFFMAN_TABLES_SEGMENT_WORDS
        # words and the payload starts immediately after it.  Both come out of the
        # same bump, so the payload base is next_record + SEGMENT_WORDS * 4 and
        # the cursor has to end up past both.  Reading the cursor back after
        # advancing it past the record would place the payload over the record's
        # own `size` and `next` fields.
        var next_record = arena + 16 + used * 4
        var next_payload = next_record + HUFFMAN_TABLES_SEGMENT_BYTES
        a.unsafe_offset(0)[] = UInt32(
            used + HUFFMAN_TABLES_SEGMENT_WORDS + next_size
        )
        var nxt = Pointer[HuffmanTablesSegment, AnyOrigin[mut=True]](
            unsafe_from_address=next_record
        )
        nxt[].start = next_payload
        nxt[].curr_table = next_payload
        nxt[].size = next_size
        nxt[].next = 0
        seg[].next = next_record
        ht[].curr_segment = next_record
        seg = nxt

    var written = BuildHuffmanTable(
        seg[].curr_table, root_bits, code_lengths, code_lengths_size, sorted, 1
    )
    return written
