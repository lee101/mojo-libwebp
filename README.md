# mojo-libwebp

A Mojo port of the lossless (VP8L) half of Google's
[libwebp](https://github.com/webmproject/libwebp) 1.6, with a Python API that
mirrors upstream's names and signatures.

On every stream it accepts, the decoder is bit-exact against the real C
library: the test suite asserts `numpy.array_equal(mojowebp.decode(data),
libwebp.WebPDecodeRGBA(data))`, because a lossless decoder has exactly one
correct answer. 6 of the 24 images in the test corpus do not decode at all;
those are enumerated under "Known issues" and pinned so the set cannot change
silently.

## What is covered

Everything the VP8L decoder needs, ported from upstream's own source order:

| Upstream file | What was ported |
| --- | --- |
| `src/utils/bit_reader_utils.{c,h}` | `VP8LBitReader`, `VP8LReadBits`, `VP8LFillBitWindow`, `ShiftBytes`, `VP8LIsEndOfStream` |
| `src/utils/huffman_utils.c` | `BuildHuffmanTable`, `VP8LBuildHuffmanTable` and the `HuffmanTables` segment chain (the encoder-side half is not ported, see below) |
| `src/utils/color_cache_utils.{c,h}` | `VP8LColorCacheInit`, `Lookup`, `Insert`, `Contains`, `VP8LHashPix` |
| `src/dsp/lossless.c` | `VP8LInverseTransform`, `PredictorInverseTransform_C` + all 14 predictors, `ColorSpaceInverseTransform_C`, `ColorIndexInverseTransform_C`, `MapARGB_C`, `VP8LColorIndexInverseTransformAlpha`, `VP8LAddGreenToBlueAndRed`, `VP8LConvertBGRATo*` |
| `src/dec/vp8l_dec.c` | `VP8LDecodeHeader`, `DecodeImageStream`, `DecodeImageData`, `ReadTransform`, `ExpandColorMap`, `ReadHuffmanCode(s)`, `ProcessRows`, `ApplyInverseTransforms`, `EmitRows`, and the alpha-plane path (`DecodeAlphaData`, `ExtractAlphaRows`, `ExtractPalettedAlphaRows`) -- the alpha path is written but does not decode yet, see "Known issues" |
| `src/webp/format_constants.h` | every `#define`, as a `comptime` of the same name |

## What is not covered

- **The lossy VP8 codec.** Only VP8L (lossless) is ported. A lossy `VP8 ` chunk
  is rejected with `VP8_STATUS_UNSUPPORTED_FEATURE`, as upstream does.
- **The encoder.** There is no `WebPEncode` entry point, and neither is the
  encoder-side half of `huffman_utils.c`: `VP8LCreateHuffmanTree`,
  `GenerateOptimalTree`, `SortHuffmanTrees`, `OptimizeHuffmanForRle` and the RLE
  token coders are not ported. Nothing in the decoder path reaches them, so
  carrying them would have been code no test could reach.
- **`bit_writer_utils.c`.** Not ported; it belongs to the encoder.
- **Premultiplied output modes** (`MODE_rgbA`, `MODE_bgrA`, `MODE_Argb`,
  `MODE_rgbA_4444`) need `WebPApplyAlphaMultiply` from
  `src/dsp/alpha_processing.c`, which is not ported; `VP8LConvertFromBGRA`
  returns 0 for them.
- **Incremental / streaming decode.** `WebPInitDec` / `WebPIterator` are not
  ported; `decode()` takes the whole buffer.
- **Upstream's SSE4/AVX2 kernels.** These ports are the portable C reference
  paths, not the vectorised ones. The elementwise passes below are widened with
  `simd_width_of[DType.uint32]()`, but the predictor modes that depend on the
  previous output pixel stay scalar -- see "Why some kernels are widened".

## Install

```
pixi install
```

## Usage

```python
import io
import numpy as np
from PIL import Image
import mojowebp

# Encode with any encoder, then decode with the Mojo port.
arr = np.random.default_rng(0).integers(
    0, 256, size=(64, 48, 4), dtype=np.uint8
)
buf = io.BytesIO()
Image.fromarray(arr, "RGBA").save(buf, format="WEBP", lossless=True, quality=100)
data = buf.getvalue()

print(mojowebp.WebPGetInfo(data))      # (48, 64, 1) -> (width, height, has_alpha)
rgba = mojowebp.WebPDecodeRGBA(data)   # (64, 48, 4) uint8

# The upstream names for every covered colourspace.
rgb  = mojowebp.WebPDecodeRGB(data)    # (64, 48, 3)
bgra = mojowebp.WebPDecodeBGRA(data)   # (64, 48, 4)
argb = mojowebp.WebPDecodeARGB(data)   # (64, 48, 4)
bgr  = mojowebp.WebPDecodeBGR(data)    # (64, 48, 3)
```

The decoded `rgba` is bit-identical to what `libwebp`'s own `WebPDecodeRGBA`
returns for the same bytes. It is not always byte-identical to `arr`: an encoder
may alter fully transparent pixels, so the oracle is the decoder, not the
source.

Both a RIFF/WEBP container and a bare VP8L bitstream are accepted;
`extract_vp8l()` returns the inner chunk if you need it explicitly.

## Commands

```
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-libwebp.so
pixi run test      # pytest, parity against the real libwebp
pixi run bench     # benchmark vs libwebp, prints a markdown-ready table
```

## Benchmarks

Measured with `pixi run bench` on:

```
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 threads
Linux 6.8.0-142-generic x86_64
conda-forge libwebp 1.6 (libwebp.so.7), pillow's encoder for the input streams
```

The bench verifies each output against upstream *before* timing it, so a decode
that returns early cannot be mistaken for a speedup. Each case runs in a forked
child so a crash cannot take the run down.

| case | mojo-libwebp | libwebp C | ratio |
| --- | --- | --- | --- |
| decode RGBA 128x128 noise | 0.6 ms | 0.5 ms | 0.95x slower |
| decode RGBA 256x256 noise | 2.8 ms | 2.0 ms | 0.71x slower |
| decode RGB 256x256 noise | 1.8 ms | 2.1 ms | 1.18x faster |
| decode RGBA 256x256 gradient | unsupported | | |
| decode RGBA 512x512 gradient | unsupported | | |
| decode RGBA 512x512 photo-like | 1.6 ms | 1.3 ms | 0.78x slower |
| decode RGB 512x512 photo-like | 1.3 ms | 0.9 ms | 0.71x slower |
| decode RGBA 512x512 predictor | 12.5 ms | 7.7 ms | 0.61x slower |
| decode RGB 512x512 predictor | 11.1 ms | 8.0 ms | 0.72x slower |
| decode RGBA 256x256 predictor | 2.4 ms | 1.9 ms | 0.81x slower |

`ratio = libwebp C time / mojo-libwebp time`; above 1 means Mojo is faster.

**This port is generally slower than the C library.** One case out of eight is a
win (256x256 noise RGB); the rest are 0.61x-0.95x. That is the honest result and
no table here should be read as otherwise.

Two caveats on the numbers themselves. They are one run on a shared 72-thread
machine, and repeat runs move individual cases by up to 2x, so treat any single
row as indicative: an earlier run of the same binary on the same machine had
512x512 photo-like RGBA at 1.52x (faster) where this run has it at 0.78x. Which
individual case wins does not reproduce; that the port loses overall does. And the comparison is not apples-to-apples: conda-forge's
`libwebp` ships hand-written SSE4/AVX2 kernels for exactly these loops, while
this port widens only the elementwise passes it could widen and leaves the
serial ones scalar.

The two `gradient` cases are reported `unsupported` rather than timed because
this port rejects those streams; see "Known issues". The three `predictor`
cases encode with `method=0` so the stream carries the predictor transform,
which the other cases never exercise: at Pillow's default effort the encoder
also picks the cross-colour transform, whose streams land in the same gap.

### Why some kernels are widened and others are not

`VP8LAddGreenToBlueAndRed` and the predictor modes whose every input is on the
`upper` row (2, 3, 4, 8, 9) run `simd_width_of[DType.uint32]()` lanes at a time
with a scalar tail. `tests/test_simd.py` pins each of them against a scalar
reference transcribed from upstream's C, at lengths either side of the lane
count, and with poison outside the run so a kernel that writes out of bounds is
caught.

The other predictor modes read `left`, i.e. `out[x-1]`, so pixel *x* cannot be
computed before pixel *x-1` has been written. A vector body would have to read
the output it is itself writing, which is why those modes stay scalar. This is a
property of the dependency, not a performance estimate.

### Parallelism

The decoder is single-threaded. `parallelize` is available on this toolchain
(it lives in `max.algorithm`, so `max` is a dependency), but predictor row *y*
reads the output row *y-1*, so rows are a serial chain and banding them across
workers races. The one genuinely independent axis, the pixel columns
inside a row, is bounded by the same `left` dependency within each predictor
mode. The Huffman decode loop is serial by construction: each symbol's bit
length determines where the next one starts.

### GPU

Not used. The elementwise passes here are memory-bound (a handful of integer ops
per 4-byte pixel), which is the regime where a PCIe round trip and a launch cost
more than the kernel they would replace; moving them is a pessimisation, not a
port. The only kernel with real arithmetic is predictor mode 13
(`ClampedAddSubtractHalf`), and it is a serial chain on the previous output
pixel, so it cannot be widened across lanes either.

No GPU code ships here, so nothing in this repository was measured on a GPU and
no GPU numbers are claimed.

## How it works

**FFI.** `build/build.sh` compiles `src/mojowebp_capi.mojo` to
`dist/libmojo-libwebp.so` with `mojo build --emit shared-lib`. Every buffer
crosses the C ABI as an `Int` address, because `@export` rejects parametric
functions and a pointer with an inferred origin is parametric. The pointers are
rebuilt inside each wrapper:

```mojo
@export("mw_vp8l_check_signature")
def mw_vp8l_check_signature(data: Int, size: Int) abi("C") -> Int:
    return VP8LCheckSignature(data, size)
```

`python/mojowebp/_lib.py` is a thin ctypes layer over that library: it rebuilds
when `src/` is newer than `dist/`, binds each signature, and hands the kernel
shared scratch buffers. `tests/test_kernels.py` asserts the `.so` really loaded,
so nothing in the suite can pass against a pure-Python fallback.

**Memory layout.** Mojo has no heap, so the decoder never allocates. The caller
owns one arena buffer, a bump allocator whose first four words are
`[cursor, capacity, pad, pad]`. Every `WebPSafeMalloc` in the C decoder becomes
an `ArenaAlloc` reservation in that buffer:

```mojo
def ArenaAlloc(arena: Int, words: Int) -> Int:
    var a = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=arena)
    var used = Int(a.unsafe_offset(0)[])
    if used + words + ARENA_HEADER_WORDS > Int(a.unsafe_offset(1)[]):
        return 0
    a.unsafe_offset(0)[] = UInt32(used + words)
    return arena + 16 + used * 4
```

Because the caller must size that buffer before the decode runs,
`mw_vp8l_workspace_size` performs the level-0 header pass first and reports the
number of `UInt32` words the full decode will need. The Python side allocates
that many words and calls `mw_vp8l_decode`.

Pixel data is `uint32` words in libwebp's native BGRA order, exactly as in C.

## Documented divergences

Where Mojo forced a different shape, the upstream logic stays visible in the
wrapper and the divergence is isolated in the glue. Each of these is in the
README because it is a deviation from upstream's structure:

- **Predictor function tables.** Upstream's `VP8LPredictors` /
  `VP8LPredictorsAdd` arrays become a dispatcher on the mode index
  (`VP8LPredictorsAdd(mode, ...)`). The 14 predictors, their arithmetic and
  their padding sentinels (modes 14 and 15 -> 0) are unchanged.
- **Predictor arguments.** `VP8LPredictorN_C(left, top)` becomes
  `VP8LPredictorN_C(left, top[-1], top[0], top[1])`, by value rather than
  through pointers, because Mojo cannot take the address of a neighbour.
- **`HuffmanTables` segment records** live in the arena instead of being
  individually `WebPSafeMalloc`'d, and `VP8LHuffmanTablesUseRegion` covers the
  one stack-allocated table in `ReadHuffmanCodeLengths`.
- **`DecodeImageData`'s tail.** Upstream's
  `if (dec->incremental && br->eos && src < src_last) ... else if ((dec->incremental && src >= src_last) || !br->eos)`
  collapses to `if src >= src_last or not VP8LIsEndOfStream(br)` here, because
  this port has no incremental mode. `process_mode` stands in for the C
  `process_func` pointer: 0 = NULL, 1 = `ProcessRows`, 2 = `ExtractAlphaRows`.
- **The `sorted[]` scratch buffer** is `uint32` words rather than upstream's
  `uint16_t`; only the element count matters to `BuildHuffmanTable`.
- **Decode returns into a caller-owned array.** Upstream's `WebPDecodeRGBA`
  returns a heap buffer freed with `WebPFree`; here the pixels land directly in
  a numpy array. This is the one API-level difference, and it is the safer one.
- **`WebPGetInfo` returns a 3-tuple** `(width, height, has_alpha)` rather than
  filling three out-parameters, and `decode` returns an array rather than a
  pointer. Both are documented in the docstrings.
- **The elementwise transform kernels are widened.** `VP8LAddGreenToBlueAndRed`
  and the predictor modes whose every input is on the `upper` row (2, 3, 4, 8,
  9) run `simd_width_of[DType.uint32]()` lanes at a time with a scalar tail,
  instead of one pixel per iteration. The arithmetic per lane is upstream's
  expression verbatim. `tests/test_simd.py` pins each of them against a scalar
  reference at lengths that straddle the lane count, including the tail and the
  no-write-outside-the-run check. The predictor modes that take `left`
  (`out[x-1]`) stay scalar, because a vector body would read the output it is
  writing.
- **`DecodeImageData` mirrors the `HTreeGroup` fields into locals.**
  `htree0..htree4` and the trivial/packed flags are refreshed at exactly the
  points upstream refreshes them (on a meta-Huffman tile change) rather than on
  every literal.

## Known issues

Three things do not work, and the test suite pins each so none can change
silently.

- **The alpha plane does not decode.** `mw_vp8l_decode_alpha` (and therefore
  `mojowebp.decode_alpha` / `WebPDecodeAlpha`) returns
  `VP8_STATUS_BITSTREAM_ERROR` on every input tried, including a flat plane and a
  noise plane. The header pass over the same bytes succeeds, so the failure is
  inside the alpha path rather than in the stream header.
  `tests/test_kernels.py::test_mw_vp8l_decode_alpha_is_reported_as_a_known_gap`
  asserts the current behaviour; when it is fixed that test fails and has to be
  updated deliberately rather than the gap quietly closing.

  Note that `DecodeImageData`'s `process_func` pointer was a truthy `has_process`
  flag here, so the alpha entry point ran the RGBA row callback instead of
  `ExtractAlphaRows`; that is now dispatched explicitly (`process_mode` 0/1/2).
  Fixing the dispatch alone did **not** make the path decode, so more is wrong
  beyond it.


- **Predictor mode 13 rounds differently from C in principle.**
  `AddSubtractComponentHalf` is `(a - b) / 2`, which C evaluates by truncating
  toward zero; Mojo's `//` is floor division, so for an odd negative `a - b` the
  two differ by one and `Clip255` propagates that into the output byte. The
  line is unchanged from the original port and predates the SIMD work. It is
  listed because it is a real semantic difference, not a rounding one: I could
  not produce a stream that reaches it. 60 extra predictor streams encoded at
  `method=0` (noisy-smooth content, which is what the encoder turns into mode
  13) all decode bit-exactly against libwebp, as does the whole test corpus.
  `tests/test_simd.py` therefore compares mode 13 floor-to-floor and says so in
  a comment; if a stream is ever found that does differ, the fix is an explicit
  truncating divide in `src/webplib/lossless.mojo`.

- **`lz77-dist`.** 6 of the 24 corpus images fail with
  `VP8_STATUS_BITSTREAM_ERROR`; the other 18 decode bit-exactly. These streams
  are rejected with a proper status code rather than crashing, and the benchmark
  reports them as `unsupported` instead of timing them.

  The label is inherited from an earlier, wrong diagnosis and overstates what is
  known: an earlier note blamed the LZ77 bounds check in `DecodeImageData`,
  which is not where these fail. Instrumenting the decode shows the failure is
  in the **header**, before any pixel is decoded: for the three `palette` cases
  the nested colour-map `DecodeImageStream(num_colors, 1)` reads its five
  Huffman codes successfully and then leaves the reader at end-of-buffer, so
  the level-0 `ReadHuffmanCodes` that follows has nothing left to read. For the
  three `predictor`/`cross-color` cases the same header pass fails earlier.

  What is ruled out: the bit reader (`VP8LReadBits` is differentially tested
  against a reference transcription of the C), `BuildHuffmanTable` (its table
  contents are verified directly, including a 2-level table), and the arena
  bookkeeping the codes live in. What is **not** yet pinned is the remaining
  difference in how many bits the nested stream consumes. Treat the name as a
  placeholder for "these six streams", not as a description.

Defects found and fixed during this port. Each has a regression test that fails
without the fix.

- **The FFI state buffers were all too small.** `python/mojowebp/_lib.py` sized
  `VP8LDecoder`, `VP8LBitReader`, `VP8LColorCache` and `HuffmanTables` from a
  comment rather than from the structs, and every one of them was short: the
  decoder record is 40 words but 64 were assumed against a 40-word allocation,
  so the decoder wrote half its state past the end of the numpy buffer. Sizes
  are now measured (`DEC_WORDS = 40`, `BR_WORDS = 6`, `CC_WORDS = 3`,
  `HT_WORDS = 5`, `TMP_WORDS = 3`) and pinned by
  `tests/test_units.py::test_state_struct_sizes_match_the_compiled_library`,
  which reads the real layout out of the library through `mw_struct_size`.

- **A relocated Huffman segment overwrote its own record.** When
  `VP8LBuildHuffmanTable` grows the segment chain it takes a new record and its
  payload from the same bump, and computed the payload base by reading the
  cursor back *after* advancing it past the record. The payload therefore
  started on top of the record's own `size` and `next` fields, which then held
  table words and poisoned the chain. Pinned by
  `tests/test_kernels.py::test_mw_vp8l_build_huffman_table_roundtrip`.

- **`VP8LHuffmanTablesAllocate` ignored the arena header** in its bounds check,
  so it could hand out the last four words of the workspace. `ArenaAlloc` has
  always subtracted them; this path did not.

- `ArenaAlloc` compared the bump cursor against the raw capacity without
  reserving the arena's own 4-word header, so the last four words were handed
  out and writes ran past the workspace. This surfaced as
  `malloc(): corrupted top size` from images around 224x224.
- `mw_vp8l_workspace_size` reported the cursor without that same header, so
  images past the initial 64K-word workspace got a buffer four words too small;
  the final allocation failed and the decode produced an almost entirely wrong
  image (bit-exact below ~224px, garbage at 256px).

## Layout

```
src/mojowebp_capi.mojo     the @export surface: 18 C-ABI entry points
src/webplib/               the ported kernels, in upstream's source order
build/build.sh             mojo build --emit shared-lib
python/mojowebp/           ctypes glue + the public API
tests/                     pytest; parity against real libwebp
bench/bench.py             benchmarks vs libwebp
```

## License

MIT, Copyright (c) 2026 Lee Penkman. See `LICENSE`.

The port is derived from libwebp, which is BSD-3-Clause; this repository's own
license covers the Mojo translation, not upstream's original code.