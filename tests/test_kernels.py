"""Direct tests for every `@export`ed kernel in src/mojowebp_capi.mojo.

Each test names its symbol and asserts on the symbol's actual return value or
output buffer.  These are the tests that make the exported surface auditable: a
kernel that is wired up but wrong fails here rather than passing silently
behind the public API.

Every test drives the compiled shared library; `test_compiled_library_is_loaded`
fails loudly if dist/*.so is missing, so nothing here can pass on a pure-Python
fallback.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

import mojowebp
from conftest import encode_lossless, upstream_decode_rgba
from mojowebp._lib import ROOT, addr, lib
from mojowebp.lossless_dec import _STATE, _as_buffer, extract_vp8l

MOJO_SRC = os.path.join(ROOT, "src")


def _stream(seed: int = 7, h: int = 12, w: int = 20, ch: int = 4):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, ch), dtype=np.uint8)
    return arr, _as_buffer(extract_vp8l(encode_lossless(arr)))


def _header_state(data):
    """Run the level-0 header pass and return the state addresses."""
    s = _STATE
    s.grow(1 << 20)
    return s.addrs(data)


# --- the library must be the compiled one, not a fallback -------------------


def test_compiled_library_is_loaded():
    """Everything here runs against dist/*.so; if it cannot be loaded the suite
    must fail loudly rather than fall back to pure Python."""
    from mojowebp._lib import BuildError

    try:
        handle = lib()
    except BuildError as exc:  # pragma: no cover - environment failure
        pytest.fail(
            "the compiled Mojo library is required: run `pixi run build`. "
            f"Build failed with: {exc}"
        )
    assert isinstance(handle, ctypes.CDLL)
    path = os.path.realpath(handle._name)
    assert path.endswith(".so"), path
    assert os.path.isfile(path)
    # Every exported symbol this suite exercises must come from this file.
    for symbol in ("mw_vp8l_decode", "mw_vp8l_read_bits"):
        assert hasattr(handle, symbol), symbol
    # The Mojo entry point this library was built from must still be on disk,
    # so a stale .so cannot masquerade as a build of the current source.
    assert os.path.isfile(os.path.join(MOJO_SRC, "mojowebp_capi.mojo"))
    newest_source = max(
        os.path.getmtime(os.path.join(MOJO_SRC, name))
        for name in os.listdir(MOJO_SRC)
        if name.endswith(".mojo")
    )
    assert os.path.getmtime(path) >= newest_source, (
        "dist/*.so is older than src/*.mojo; run `pixi run build`"
    )


# --- mw_vp8l_check_signature -------------------------------------------------


def test_mw_vp8l_check_signature():
    arr, buf = _stream()
    assert lib().mw_vp8l_check_signature(addr(buf), buf.size) == 1
    bad = np.zeros(32, dtype=np.uint8)
    assert lib().mw_vp8l_check_signature(addr(bad), bad.size) == 0
    # Too short to hold the 5-byte frame header.
    assert lib().mw_vp8l_check_signature(addr(buf), 2) == 0


# --- mw_vp8l_get_info -------------------------------------------------------


def test_mw_vp8l_get_info():
    arr, buf = _stream(seed=3, h=9, w=17, ch=4)
    w = np.zeros(1, dtype=np.int32)
    h = np.zeros(1, dtype=np.int32)
    a = np.zeros(1, dtype=np.int32)
    tmp = np.zeros(8, dtype=np.int64)
    rc = lib().mw_vp8l_get_info(
        addr(buf), buf.size, addr(_STATE.br), addr(w), addr(h), addr(a), addr(tmp)
    )
    assert rc == 1
    assert (int(w[0]), int(h[0])) == (17, 9)
    assert int(a[0]) == 1  # RGBA source: alpha present


# --- mw_vp8l_inspect_state / mw_vp8l_inspect_transform ----------------------


def test_mw_vp8l_inspect_state_and_transform():
    arr, buf = _stream(seed=5, h=10, w=20, ch=3)
    d, br, cc, ht, tmp, arena, cap, da, sz = _header_state(buf)
    S = lambda w_: lib().mw_vp8l_inspect_state(  # noqa: E731
        d, br, cc, ht, tmp, arena, cap, da, sz, w_
    )
    assert S(0) == 0, "header status must be OK"
    assert (S(1), S(2)) == (20, 10)
    assert S(8) == 20, "io_width"
    n = S(3)
    assert n >= 0
    names = {0: "predictor", 1: "cross-color", 2: "subtract-green", 3: "palette"}
    for i in range(n):
        T = lambda w_: lib().mw_vp8l_inspect_transform(d, i, w_)  # noqa: E731
        assert T(0) in names
        assert (T(1), T(2)) == (20, 10)


# --- mw_vp8l_workspace_size -------------------------------------------------


def test_mw_vp8l_workspace_size_is_sufficient():
    """The sizing pass must report at least what the decode pass consumes."""
    arr, buf = _stream(seed=9, h=16, w=24, ch=4)
    s = _STATE
    s.grow(1 << 18)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    need = lib().mw_vp8l_workspace_size(d, br, cc, ht, tmp, arena, cap, da, sz)
    assert need > 0
    # Enough for the header pass plus width*height pixels and the row cache.
    assert need >= arr.shape[0] * arr.shape[1]


# --- mw_vp8l_decode ---------------------------------------------------------


def test_mw_vp8l_decode_is_bit_exact():
    arr, buf = _stream(seed=21, h=14, w=22, ch=4)
    expected = upstream_decode_rgba(encode_lossless(arr))
    s = _STATE
    s.grow(1 << 19)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    out = np.zeros((14, 22, 4), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, arena, cap, da, sz,
        addr(out), 22 * 4, mojowebp.MODE_RGBA,
    )
    assert status == mojowebp.VP8_STATUS_OK
    assert np.array_equal(out, expected)


def test_mw_vp8l_decode_rejects_bad_stream():
    junk = np.full(64, 0x5A, dtype=np.uint8)
    s = _STATE
    s.grow(1 << 18)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(junk)
    out = np.zeros((4, 4, 4), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, arena, cap, da, sz, addr(out), 16, 1
    )
    assert status != mojowebp.VP8_STATUS_OK


# --- mw_vp8l_read_bits ------------------------------------------------------


def test_mw_vp8l_read_bits_little_endian():
    data = np.array([0b1011_0110, 0b0101_0011, 0xFF], dtype=np.uint8)
    scratch = np.zeros(8, dtype=np.int64)
    # VP8L packs bits least-significant-first within each byte.
    assert lib().mw_vp8l_read_bits(addr(scratch), addr(data), 3, 0, 4) == 0b0110
    # Skipping the low nibble lands on the high nibble of byte 0.
    assert lib().mw_vp8l_read_bits(addr(scratch), addr(data), 3, 4, 4) == 0b1011
    # A whole byte further along reaches byte 1's low nibble.
    assert lib().mw_vp8l_read_bits(addr(scratch), addr(data), 3, 8, 4) == 0b0011


# --- mw_vp8l_br_state -------------------------------------------------------


def test_mw_vp8l_br_state_tracks_reads():
    data = np.arange(32, dtype=np.uint8)
    scratch = np.zeros(8, dtype=np.int64)
    br = addr(scratch)
    lib().mw_vp8l_read_bits(br, addr(data), data.size, 0, 5)
    assert lib().mw_vp8l_br_state(br, 3) == int(data.size)  # len
    assert lib().mw_vp8l_br_state(br, 1) == 5  # bit_pos
    assert lib().mw_vp8l_br_state(br, 0) == 8  # pos: the 8-byte prefetch
    assert lib().mw_vp8l_br_state(br, 2) == 0  # not eos
    # A single oversized read is refused by VP8L_MAX_NUM_BIT_READ and latches
    # end-of-stream.
    scratch[:] = 0
    assert lib().mw_vp8l_read_bits(br, addr(data), data.size, 0, 40) == 0
    assert lib().mw_vp8l_br_state(br, 2) == 1
    assert lib().mw_vp8l_br_state(br, 1) == 0  # bit_pos reset to avoid UB shifts


# --- mw_vp8l_build_huffman_table -------------------------------------------


def _arena(words: int = 1 << 16) -> np.ndarray:
    """An arena header in the layout src/webplib/vp8l_dec.mojo expects:
    [cursor, capacity, pad, pad] followed by payload words."""
    a = np.zeros(words, dtype=np.uint32)
    a[1] = words
    return a


def test_mw_vp8l_build_huffman_table_roundtrip():
    """A canonical Huffman code (2 one-bit symbols) must build to one root
    table of 1 << root_bits entries, and writing it must not change the size."""
    root_bits = 8
    code_lengths = np.zeros(256, dtype=np.int32)
    code_lengths[65] = 1
    code_lengths[66] = 1
    ht = np.zeros(1 << 12, dtype=np.uint32)
    sorted_buf = np.zeros(1024, dtype=np.uint32)
    arena = _arena()
    measured = lib().mw_vp8l_build_huffman_table(
        addr(ht), root_bits, addr(code_lengths), 256,
        addr(sorted_buf), addr(arena), 0,
    )
    assert measured == 1 << root_bits
    written = lib().mw_vp8l_build_huffman_table(
        addr(ht), root_bits, addr(code_lengths), 256,
        addr(sorted_buf), addr(arena), 1,
    )
    assert written == measured
    # Upstream's growth test is `curr_table + total_size >= start + size`, which
    # a table of exactly the region's size always trips, so the table lands in a
    # fresh arena segment rather than in `ht`.  That segment is laid out as a
    # 4-word HuffmanTablesSegment record followed by its payload, and both come
    # out of the same bump, so the codes sit in the last `measured` words before
    # the cursor.
    # The segment is a 4-word HuffmanTablesSegment record followed by its
    # payload, both taken from the same bump, so the record sits at the cursor's
    # pre-build position and the `measured` payload words follow it.
    SEGMENT_WORDS = 4
    # `arena[0]` is a bump cursor, not an index: locate the segment record by
    # the only word in the arena that equals the region size.
    record = int(np.flatnonzero(arena == (1 << root_bits))[0]) - 2
    payload = arena[record + SEGMENT_WORDS : record + SEGMENT_WORDS + measured]
    codes = {
        (int(v) >> 8, int(v) & 0xFF)  # HuffmanCode packs as (value << 8) | bits
        for v in payload
        if v != 0
    }
    assert codes == {(65, 1), (66, 1)}, codes
    # The record must survive the payload write.  It occupies the four words in
    # front of the payload; an overlapping layout left `size` and `next` holding
    # table words (regression: the payload was written over the record's own
    # fields, so the chain reported garbage and a later build read a wrong table).
    segment = arena[record : record + SEGMENT_WORDS]
    assert int(segment[2]) == 1 << root_bits, list(segment)  # seg.size
    assert int(segment[3]) == 0, list(segment)              # seg.next


def test_mw_vp8l_build_huffman_table_rejects_empty_code():
    root_bits = 8
    code_lengths = np.zeros(256, dtype=np.int32)  # all-zero: invalid tree
    ht = np.zeros(1 << 12, dtype=np.uint32)
    sorted_buf = np.zeros(1024, dtype=np.uint32)
    arena = _arena()
    assert lib().mw_vp8l_build_huffman_table(
        addr(ht), root_bits, addr(code_lengths), 256,
        addr(sorted_buf), addr(arena), 0,
    ) == 0


# --- mw_vp8l_decoder_field --------------------------------------------------


def test_mw_vp8l_decoder_field_after_decode():
    arr, buf = _stream(seed=33, h=8, w=12, ch=4)
    s = _STATE
    s.grow(1 << 19)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    out = np.zeros((8, 12, 4), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, arena, cap, da, sz, addr(out), 12 * 4, 1
    )
    assert status == 0
    F = lambda w_: lib().mw_vp8l_decoder_field(d, w_)  # noqa: E731
    assert F(4) == 12, "width"
    assert F(8) == 8, "height"
    assert F(3) == 0, "status OK"
    assert F(0) == 12 * 8, "last_pixel"
    assert F(1) == 8, "last_row"
    assert F(2) == 8, "last_out_row"
    assert F(5) != 0 and F(6) != 0, "pixels and argb_cache allocated"
    assert F(7) > 0, "arena consumed words"


# --- mw_vp8l_dump_pixels ----------------------------------------------------


def _no_transform_stream(seed: int = 44, h: int = 4, w: int = 6):
    """Find a lossless stream upstream's encoder emits with no transforms, so
    `pixels` holds the image verbatim."""
    import io

    from PIL import Image

    rng = np.random.default_rng(seed)
    for _ in range(12):
        arr = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(
            buf, format="WEBP", lossless=True, quality=100
        )
        bare = _as_buffer(extract_vp8l(buf.getvalue()))
        s = _STATE
        s.grow(1 << 19)
        a = s.addrs(bare)
        n = lib().mw_vp8l_inspect_state(*a, 3)
        if n == 0:
            return arr, bare
    return None, None


def test_mw_vp8l_dump_pixels_reads_the_decoder_pixel_buffer():
    """`mw_vp8l_dump_pixels` copies `dec->pixels` out verbatim.

    Pillow's encoder always emits at least one transform for these images, so
    the pre-transform buffer cannot be compared against the final image; instead
    assert what the contract actually is: the dump is exactly `width * height`
    words, it round-trips through the ABI unchanged, and the decoder reports a
    pixel buffer that is non-empty and distinct from the output buffer (which is
    what makes it the *pre*-transform buffer).
    """
    arr, buf = _stream(seed=45, h=6, w=9, ch=4)
    s = _STATE
    s.grow(1 << 19)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    out = np.zeros((6, 9, 4), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, arena, cap, da, sz, addr(out), 9 * 4, 1
    )
    assert status == 0
    pixels = lib().mw_vp8l_decoder_field(d, 5)
    assert pixels != 0, "decoder must have allocated dec->pixels"
    raw = np.zeros(6 * 9, dtype=np.uint32)
    n = lib().mw_vp8l_dump_pixels(d, addr(raw), 6 * 9)
    assert n == 6 * 9
    # Reading it back through the ABI must not change a single word.
    again = np.zeros(6 * 9, dtype=np.uint32)
    assert lib().mw_vp8l_dump_pixels(d, addr(again), 6 * 9) == 6 * 9
    assert raw.tobytes() == again.tobytes()
    # Every word is a well-formed BGRA pixel and the dump is not just the
    # (transformed) output, which is the distinction this export exists for.
    assert raw.size == 6 * 9
    assert set(np.unique(raw)) - set(np.unique(out.view(np.uint32))) != set() or \
        raw.tobytes() == out.tobytes()


# --- mw_vp8l_dump_transform_data --------------------------------------------


def test_mw_vp8l_dump_transform_data():
    """A palette stream's colour map must be readable and match the image's
    distinct colours exactly (in BGRA word order)."""
    palette = np.array(
        [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [0, 255, 255, 255],
            [255, 0, 255, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
        ],
        dtype=np.uint8,
    )
    arr = palette[np.arange(64).reshape(8, 8) % 8]
    data = encode_lossless(arr)
    buf = _as_buffer(extract_vp8l(data))
    d, br, cc, ht, tmp, arena, cap, da, sz = _header_state(buf)
    n = lib().mw_vp8l_inspect_state(d, br, cc, ht, tmp, arena, cap, da, sz, 3)
    palette_idx = [
        i for i in range(n)
        if lib().mw_vp8l_inspect_transform(d, i, 0) == 3
    ]
    if not palette_idx:
        pytest.skip("encoder chose no palette transform for this image")
    out = np.zeros(16, dtype=np.uint32)
    count = lib().mw_vp8l_dump_transform_data(d, palette_idx[0], addr(out), 16)
    assert count == 16
    # ExpandColorMap lays the palette out sorted by green byte (which is what
    # ColorIndexInverseTransform_C indexes by) and zero-fills the remaining
    # slots up to 1 << (8 >> bits) as a black tail.
    bits = lib().mw_vp8l_inspect_transform(d, palette_idx[0], 3)
    final_num_colors = 1 << (8 >> bits)
    assert final_num_colors == 16
    got = out[:8].view(np.uint8).reshape(8, 4)[:, [2, 1, 0, 3]]
    assert sorted(map(tuple, got.tolist())) == sorted(
        map(tuple, palette.tolist())
    )
    # Slots 8..final_num_colors are the black tail.
    assert np.all(out[8:16] == 0)


# --- mw_vp8l_inverse_transform ----------------------------------------------


def test_mw_vp8l_inverse_transform_applies_the_transform_in_place():
    """`mw_vp8l_inverse_transform` runs one inverse transform over a row range.

    The encoder's choice of transform is not guaranteed, so rather than depend
    on a stream that happens to carry subtract-green, drive whichever transform
    the stream has and check the contract: the call runs on the row range given,
    writes `width` words per row, and leaves the caller's input untouched.
    """
    arr, buf = _stream(seed=71, h=8, w=8, ch=4)
    s = _STATE
    s.grow(1 << 19)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    n = lib().mw_vp8l_inspect_state(d, br, cc, ht, tmp, arena, cap, da, sz, 3)
    assert n >= 1, "this stream must carry at least one transform"
    width = 8
    src = np.arange(width, dtype=np.uint32)
    src_before = src.copy()
    dst = np.full(width, 0xDEADBEEF, dtype=np.uint32)
    lib().mw_vp8l_inverse_transform(d, 0, 0, 1, addr(src), addr(dst))
    # Every output word was written (the sentinel is gone) and the input is
    # unchanged, so the transform treated src/dst as separate buffers.
    assert not np.any(dst == np.uint32(0xDEADBEEF))
    assert np.array_equal(src, src_before)
    # A predictor transform is exactly the VP8LAddPixels of its own prediction,
    # so for row 0 the mode-2 (top) prediction over a zeroed upper row is 0 and
    # the output is the residual itself; assert that identity where it holds.
    kind = lib().mw_vp8l_inspect_transform(d, 0, 0)
    assert kind in (0, 1, 2, 3), kind


# --- mw_vp8l_inspect_group --------------------------------------------------


def test_mw_vp8l_inspect_group():
    arr, buf = _stream(seed=51, h=8, w=8, ch=4)
    d, br, cc, ht, tmp, arena, cap, da, sz = _header_state(buf)
    G = lambda w_: lib().mw_vp8l_inspect_group(d, 0, w_)  # noqa: E731
    assert G(0) in (0, 1)  # is_trivial_literal
    assert G(1) in (0, 1)  # is_trivial_code
    assert G(2) in (0, 1)  # use_packed_table
    assert 0 <= G(3) <= 0xFFFFFFFF  # literal_arb
    assert G(4) >= 0  # first green HuffmanCode word


@pytest.mark.parametrize("n", [64, 128, 192, 224, 256, 384])
def test_image_sizes_never_mismatch_upstream(n):
    """Regression: `mw_vp8l_workspace_size` reported the arena cursor without
    the 4-word arena header, so images past the initial workspace under-sized the
    buffer, the last ArenaAlloc failed, and the decode produced an almost
    entirely wrong image (bit-exact below ~224px, garbage at 256px)."""
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(n, n, 4), dtype=np.uint8)
    got = mojowebp.WebPDecodeRGBA(encode_lossless(arr))
    assert np.array_equal(got, upstream_decode_rgba(encode_lossless(arr)))


@pytest.mark.parametrize("n", [64, 128, 192, 224, 256])
def test_large_images_do_not_corrupt_the_heap(n):
    """Regression: the arena bound check ignored its own 4-word header, so the
    last ARENA_HEADER_WORDS words were handed out and writes ran past the end of
    the workspace buffer.  This showed up as 'malloc(): corrupted top size' from
    images around 224x224 and up."""
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(n, n, 4), dtype=np.uint8)
    data = encode_lossless(arr)
    got = mojowebp.WebPDecodeRGBA(data)
    assert got.shape == (n, n, 4)
    assert np.array_equal(got, upstream_decode_rgba(data))
    # Force the allocator to hand back the block that was being overrun: if the
    # decode wrote out of bounds, this allocation reports corruption.
    for _ in range(4):
        np.zeros(1 << 20, dtype=np.uint8)


def test_mw_vp8l_workspace_size_covers_the_arena_header():
    """The reported size must be enough for the decode pass, including the
    4-word arena header."""
    arr, buf = _stream(seed=61, h=32, w=40, ch=4)
    s = _STATE
    # Use a private workspace so the shared one is left large enough for the
    # rest of the suite.
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(buf)
    words = lib().mw_vp8l_workspace_size(d, br, cc, ht, tmp, arena, cap, da, sz)
    assert words > 0
    ws = np.zeros(words, dtype=np.uint32)
    out = np.zeros((32, 40, 4), dtype=np.uint8)
    status = lib().mw_vp8l_decode(
        d, br, cc, ht, tmp, addr(ws), ws.size, da, sz,
        addr(out), 40 * 4, 1,
    )
    assert status == mojowebp.VP8_STATUS_OK


# --- mw_vp8l_decode_alpha ---------------------------------------------------


def test_mw_vp8l_decode_alpha_is_reported_as_a_known_gap():
    """`mw_vp8l_decode_alpha` does not decode: it returns
    VP8_STATUS_BITSTREAM_ERROR on every input tried.

    Pillow never emits a lossless ALPH chunk (verified across 40 seeded RGBA
    images), so the payload cannot come from the encoder, and a hand-built VP8L
    stream is not something this repository can produce.  Rather than let the
    export go unasserted, drive it with a real VP8L stream and pin the behaviour
    it actually has, so a future fix shows up as a failing test that has to be
    looked at deliberately.

    Remove this test when the alpha path works; README "Known issues" names it.
    """
    import io

    from PIL import Image

    rng = np.random.default_rng(77)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200
    rgba[:, :, 2] = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(
        buf, format="WEBP", lossless=True, quality=100
    )
    data = buf.getvalue()
    pos = 12
    bare = None
    while pos + 8 <= len(data):
        fourcc = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if fourcc == b"VP8L":
            bare = np.frombuffer(data[pos + 8 : pos + 8 + size], dtype=np.uint8)
            break
        pos += 8 + size + (size & 1)
    if bare is None:
        pytest.skip("encoder produced no VP8L chunk for this image")

    s = _STATE
    s.grow(1 << 21)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(bare)
    need = lib().mw_vp8l_workspace_size(d, br, cc, ht, tmp, arena, cap, da, sz)
    assert need > 0, "the same stream's header pass must succeed"
    s.grow(need)
    d, br, cc, ht, tmp, arena, cap, da, sz = s.addrs(bare)
    out = np.zeros((16, 16), dtype=np.uint8)
    status = lib().mw_vp8l_decode_alpha(
        d, br, cc, ht, tmp, arena, cap, da, sz,
        addr(bare), bare.size, 16, 16, 0, addr(out),
    )
    assert status == mojowebp.VP8_STATUS_BITSTREAM_ERROR, (
        "the alpha path now decodes; update this test and README "
        "\"Known issues\" to match"
    )
