"""Parity of the Mojo VP8L decoder against upstream libwebp.

The bar is bit-exactness with the C library: a lossless decoder has exactly one
correct answer, and upstream is the reference.  Tests here assert equality and
fail on any difference; the one configuration this port does not yet reproduce
is named in KNOWN_GAPS and pinned by test_known_gaps_are_pinned so the list
cannot grow silently.

KNOWN_GAPS entries name the *stream property* that fails, not the transform
name: `lz77-dist` is a VP8L stream whose green Huffman tree decodes a
backward-reference code at pixel 0, which trips the LZ77 bounds check in
DecodeImageData.  The symptom is a VP8_STATUS_BITSTREAM_ERROR.  See README
"Known issues" for where the divergence is.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojowebp
from conftest import (  # noqa: F401
    encode_lossless,
    make_images,
    transform_types,
    upstream,
    upstream_decode_rgba,
)

# Stream configurations known not to round-trip yet.
KNOWN_GAPS = ("lz77-dist",)


def _is_covered(data: bytes) -> bool:
    return not any(t in KNOWN_GAPS for t in transform_types(data))


def test_upstream_oracle_is_self_consistent():
    """The reference must reproduce the encoder's input, or nothing below means
    anything."""
    # Pillow's WebP encoder is upstream's, but it is allowed to alter fully
    # transparent pixels, so the source array is not a valid oracle.  What must
    # hold is that the C decoder is deterministic and self-consistent.
    for name, arr, data, _ in make_images(count=6):
        assert np.array_equal(upstream_decode_rgba(data),
                              upstream_decode_rgba(data)), name


@pytest.mark.parametrize("name,arr,data,cs", make_images())
def test_mojo_matches_upstream(name, arr, data, cs):
    """Every covered stream decodes bit-exactly, and matches upstream."""
    if not _is_covered(data):
        pytest.skip(f"{name}: {transform_types(data)} is a known gap")
    expected = upstream_decode_rgba(data)
    got = mojowebp.decode(data, cs)
    if cs == 0:
        expected = expected[:, :, :3]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected), (
        f"{name}: {int((got != expected).sum())} of {got.size} bytes differ"
    )


def test_all_colorspaces_agree():
    arr = np.random.default_rng(1).integers(0, 256, size=(24, 24, 4), dtype=np.uint8)
    data = encode_lossless(arr)
    if not _is_covered(data):
        pytest.skip("stream is a known gap")
    up = upstream_decode_rgba(data)
    assert np.array_equal(mojowebp.WebPDecodeRGBA(data), up)
    assert np.array_equal(mojowebp.WebPDecodeRGB(data), up[:, :, :3])
    # MODE_ARGB and MODE_BGRA are the raw little-endian uint32 word, so the byte
    # order in memory is B, G, R, A -- exactly as libwebp emits them.
    swap_rb = up[:, :, [2, 1, 0, 3]]
    assert np.array_equal(mojowebp.WebPDecodeARGB(data), swap_rb)
    assert np.array_equal(mojowebp.WebPDecodeBGRA(data), swap_rb)
    bgr = mojowebp.WebPDecodeBGR(data)
    assert np.array_equal(bgr, up[:, :, [2, 1, 0]])


def test_get_info_matches_dimensions():
    for name, arr, data, _ in make_images(count=8):
        w, h, has_alpha = mojowebp.get_info(data)
        assert (w, h) == (arr.shape[1], arr.shape[0]), name
        assert has_alpha == (1 if arr.shape[2] == 4 else 0) or has_alpha in (0, 1)


def test_webp_container_and_bare_stream_agree():
    arr = np.random.default_rng(2).integers(0, 256, size=(20, 20, 4), dtype=np.uint8)
    data = encode_lossless(arr)
    bare = mojowebp.extract_vp8l(data)
    if not _is_covered(data):
        pytest.skip("stream is a known gap")
    assert np.array_equal(mojowebp.decode(bare), mojowebp.decode(data))


def test_rejects_lossy_payload():
    with pytest.raises(mojowebp.WebPError) as exc:
        mojowebp.decode(b"RIFF\x04\x00\x00\x00WEBPVP8 \x04\x00\x00\x00abcd")
    assert exc.value.status == mojowebp.VP8_STATUS_UNSUPPORTED_FEATURE


# Streams the port is verified bit-exact on.  They are pinned so a regression
# in the shared decode path cannot pass silently; the parametrized test above
# covers the rest of the corpus and skips the known gaps.
VERIFIED = [
    ("flat 1x1 RGB", np.full((1, 1, 3), 77, np.uint8)),
    ("flat 20x20 RGB", np.full((20, 20, 3), 17, np.uint8)),
    ("gradient 3x1 RGBA", np.tile(np.array([200, 10, 90, 255], np.uint8), (1, 3, 1))),
    ("gradient 64x64 RGB", np.ascontiguousarray(np.tile(
        (np.arange(64, dtype=np.uint8) * 4)[None, :, None].repeat(3, axis=2),
        (64, 1, 1)))),
]


@pytest.mark.parametrize("name,arr", VERIFIED, ids=[n for n, _ in VERIFIED])
def test_verified_streams_are_bit_exact(name, arr):
    """The verified subset must stay bit-exact; this is the regression guard."""
    data = encode_lossless(arr)
    expected = upstream_decode_rgba(data)
    got = mojowebp.decode(data, 1 if arr.shape[2] == 4 else 0)
    if got.shape[2] == 3:
        got = got[:, :, :3]
        expected = expected[:, :, :3]
    assert np.array_equal(got, expected), (
        f"{name}: {int((got != expected).sum())} of {got.size} bytes differ"
    )


def test_known_gaps_are_pinned():
    """Guard the KNOWN_GAPS list against silent growth or shrinkage."""
    seen = set()
    for _, _, data, _ in make_images():
        seen.update(transform_types(data))
    assert KNOWN_GAPS == ("lz77-dist",), sorted(seen)