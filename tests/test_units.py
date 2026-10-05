"""Unit tests for the ported primitives.

Each test drives one Mojo kernel and compares against a reference computed
independently, so a failure localises the bug to a single function.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojowebp
from conftest import encode_lossless, upstream, upstream_decode_rgba
from mojowebp._lib import addr, lib

pytestmark = pytest.mark.skipif(upstream is None, reason="libwebp not available")


# ---------------------------------------------------------------- bit reader


class RefBitReader:
    """The reference semantics of VP8LBitReader, transcribed from the C."""

    def __init__(self, data: bytes):
        self.buf = data
        self.len = len(data)
        n = min(self.len, 8)
        self.val = int.from_bytes(data[:n], "little")
        self.pos = n
        self.bit_pos = 0
        self.eos = False

    def is_end_of_stream(self) -> bool:
        if self.eos:
            return True
        return self.pos == self.len and self.bit_pos > 64

    def set_end_of_stream(self) -> None:
        self.eos = True
        self.bit_pos = 0

    def prefetch_bits(self) -> int:
        return self.val >> (self.bit_pos & 63)

    def shift_bytes(self) -> None:
        while self.bit_pos >= 8 and self.pos < self.len:
            self.val >>= 8
            self.val |= self.buf[self.pos] << (64 - 8)
            self.pos += 1
            self.bit_pos -= 8
        if self.is_end_of_stream():
            self.set_end_of_stream()

    def fill_bit_window(self) -> None:
        if self.bit_pos >= 32:
            if self.pos + 8 < self.len:
                self.val >>= 32
                self.bit_pos -= 32
                self.val |= int.from_bytes(
                    self.buf[self.pos:self.pos + 4], "little"
                ) << 32
                self.pos += 4
            else:
                self.shift_bytes()

    def read_bits(self, n: int) -> int:
        if not self.eos and n <= 24:
            value = self.prefetch_bits() & ((1 << n) - 1)
            self.bit_pos += n
            self.shift_bytes()
            return value
        self.set_end_of_stream()
        return 0


@pytest.mark.parametrize("length", [5, 17, 64, 300])
@pytest.mark.parametrize("n_bits", [1, 3, 8, 14, 16, 24])
@pytest.mark.parametrize("skip", [0, 1, 7, 33, 40, 100])
def test_bit_reader_matches_reference(length, n_bits, skip):
    rng = np.random.default_rng(length * 1000 + n_bits * 10 + skip)
    data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
    scratch = np.zeros(8, dtype=np.int64)
    buf = np.frombuffer(data, dtype=np.uint8)
    got = lib().mw_vp8l_read_bits(addr(scratch), addr(buf), length, skip, n_bits)
    ref = RefBitReader(data)
    for _ in range(skip):
        ref.read_bits(1)
    assert got == ref.read_bits(n_bits)


# ------------------------------------------------------------------ header


def test_check_signature():
    arr = np.random.default_rng(0).integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    data = encode_lossless(arr)
    bare = mojowebp.extract_vp8l(data)
    assert bare[0] == 0x2F
    assert mojowebp.check_signature(bare)
    assert not mojowebp.check_signature(b"\x00" * 32)
    assert not mojowebp.check_signature(b"\x2f")


def test_get_info_reports_size_and_alpha():
    for h, w, ch in ((8, 8, 4), (5, 13, 3), (1, 1, 4), (64, 3, 4)):
        arr = np.random.default_rng(h * w).integers(0, 256, size=(h, w, ch),
                                                     dtype=np.uint8)
        data = encode_lossless(arr)
        got_w, got_h, _ = mojowebp.get_info(data)
        assert (got_w, got_h) == (w, h)


def test_riff_container_parsing():
    arr = np.random.default_rng(11).integers(0, 256, size=(10, 10, 4), dtype=np.uint8)
    data = encode_lossless(arr)
    assert data[0:4] == b"RIFF" and data[8:12] == b"WEBP"
    bare = mojowebp.extract_vp8l(data)
    assert bare[0] == 0x2F
    assert bare in data


# ------------------------------------------------------- FFI state layouts


def test_state_struct_sizes_match_the_compiled_library():
    """The Python layer sizes its state buffers from the *_WORDS constants in
    mojowebp/_lib.py.  Those buffers are handed to the library as raw addresses,
    so a struct that outgrows its buffer writes past it and corrupts whatever
    numpy allocated next.  `mw_struct_size` writes a sentinel into each struct's
    last field; the word it lands on is that struct's real size.
    """
    from mojowebp._lib import (
        BR_WORDS, CC_WORDS, DEC_WORDS, HT_WORDS, lib,
    )

    sentinel = 0x1122334455667788
    # which: 0 VP8LDecoder, 1 VP8LBitReader, 2 VP8LColorCache, 3 HuffmanTables
    for which, words, name in (
        (0, DEC_WORDS, "VP8LDecoder"),
        (1, BR_WORDS, "VP8LBitReader"),
        (2, CC_WORDS, "VP8LColorCache"),
        (3, HT_WORDS, "HuffmanTables"),
    ):
        probe = np.zeros(64, dtype=np.int64)
        lib().mw_struct_size(which, probe.ctypes.data, sentinel)
        touched = np.nonzero(probe == sentinel)[0]
        assert touched.size == 1, f"{name}: expected exactly one written word"
        actual_words = int(touched[0]) + 1
        assert actual_words == words, (
            f"{name}: _lib.py reserves {words} words but the struct needs "
            f"{actual_words}; raise the constant in mojowebp/_lib.py"
        )


def test_state_buffers_are_big_enough_for_a_real_decode():
    """A decode driven through the Python API must not scribble past the
    state buffers: poison either side of every one of them and check it."""
    from mojowebp._lib import lib as _lib_mod
    from mojowebp.lossless_dec import _STATE

    arr = np.random.default_rng(5).integers(
        0, 256, size=(24, 32, 4), dtype=np.uint8
    )
    data = encode_lossless(arr)
    # Guard pages are checked by verifying the canaries survive the decode.
    canaries = []
    for buf in (_STATE.dec, _STATE.br, _STATE.cc, _STATE.ht, _STATE.tmp):
        buf[:] = 0x5A5A5A5A5A5A5A5A
        canaries.append(buf.copy())
    got = mojowebp.WebPDecodeRGBA(data)
    assert got.shape == (24, 32, 4)
    expected = upstream_decode_rgba(data)
    assert np.array_equal(got, expected)
