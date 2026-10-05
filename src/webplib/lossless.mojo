"""Port of src/dsp/lossless.c and the shared inlines of src/dsp/lossless_common.h.

Divergences forced by Mojo (all listed in the README):
  * The `VP8LPredictors` / `VP8LPredictorsAdd` function-pointer tables of
    VP8LDspInit become a dispatcher on the mode index, with the same 14
    predictors and the same padding sentinels (modes 14 and 15 -> 0).
  * Predictors take the four neighbour pixels by value instead of through
    `left` / `top` pointers: VP8LPredictorN_C(left, top[-1], top[0], top[1]).
    The arithmetic and the branch structure of every predictor are unchanged.
  * `struct HuffmanCode`-style packing is not needed here; VP8LTransform holds
    plain Int fields.
"""

from std.sys import simd_width_of

from webplib.format_constants import (
    ARGB_BLACK,
    COLOR_INDEXING_TRANSFORM,
    CROSS_COLOR_TRANSFORM,
    MODE_ARGB,
    MODE_BGR,
    MODE_BGRA,
    MODE_RGB,
    MODE_RGB_565,
    MODE_RGBA,
    MODE_RGBA_4444,
    PREDICTOR_TRANSFORM,
    SUBTRACT_GREEN_TRANSFORM,
)

# Lane count for the pixel-vector kernels.  uint32 is what the pixel buffers
# hold, so the elementwise passes below run W pixels at a time.
comptime W: Int = simd_width_of[DType.uint32]()


# Transforms are declared in dec/vp8li_dec.h upstream and forward-declared in
# dsp/lossless.h; they are defined here so both sides can use them.
struct VP8LTransform:
    var type: Int
    var xsize: Int
    var ysize: Int
    var bits: Int
    var data: Int  # address of the transform payload


struct VP8LMultipliers(ImplicitlyCopyable, Movable):
    # Note: the members are uint8_t upstream, so that any negative values are
    # automatically converted to "mod 256" values.
    var green_to_red: Int
    var green_to_blue: Int
    var red_to_blue: Int

    def __init__(out self, green_to_red: Int, green_to_blue: Int, red_to_blue: Int):
        self.green_to_red = green_to_red
        self.green_to_blue = green_to_blue
        self.red_to_blue = red_to_blue


#------------------------------------------------------------------------------
# src/dsp/lossless_common.h


# Computes sampled size of 'size' when sampling using 'sampling bits'.
def VP8LSubSampleSize(size: Int, sampling_bits: Int) -> Int:
    return (size + (1 << sampling_bits) - 1) >> sampling_bits


def VP8GetARGBIndex(idx: UInt32) -> Int:
    return Int((idx >> 8) & 0xff)


def VP8GetAlphaIndex(idx: UInt8) -> Int:
    return Int(idx)


def VP8GetARGBValue(val: UInt32) -> UInt32:
    return val


def VP8GetAlphaValue(val: UInt32) -> UInt8:
    return UInt8((val >> 8) & 0xff)


# Sum of each component, mod 256.
def VP8LAddPixels(a: UInt32, b: UInt32) -> UInt32:
    var alpha_and_green = (a & 0xff00ff00) + (b & 0xff00ff00)
    var red_and_blue = (a & 0x00ff00ff) + (b & 0x00ff00ff)
    return (alpha_and_green & 0xff00ff00) | (red_and_blue & 0x00ff00ff)


# Vector form of VP8LAddPixels: same two mask/add pairs, lane by lane.
def _AddPixelsV(a: SIMD[DType.uint32, W], b: SIMD[DType.uint32, W]) -> SIMD[
    DType.uint32, W
]:
    var alpha_and_green = (a & SIMD[DType.uint32, W](0xff00ff00)) + (
        b & SIMD[DType.uint32, W](0xff00ff00)
    )
    var red_and_blue = (a & SIMD[DType.uint32, W](0x00ff00ff)) + (
        b & SIMD[DType.uint32, W](0x00ff00ff)
    )
    return (alpha_and_green & SIMD[DType.uint32, W](0xff00ff00)) | (
        red_and_blue & SIMD[DType.uint32, W](0x00ff00ff)
    )


# Difference of each component, mod 256.
def VP8LSubPixels(a: UInt32, b: UInt32) -> UInt32:
    var alpha_and_green = (
        UInt32(0x00ff00ff) + (a & 0xff00ff00) - (b & 0xff00ff00)
    )
    var red_and_blue = (
        UInt32(0xff00ff00) + (a & 0x00ff00ff) - (b & 0x00ff00ff)
    )
    return (alpha_and_green & 0xff00ff00) | (red_and_blue & 0x00ff00ff)


# Splitting of distance and length codes into prefixes and extra bits.  The
# prefixes are encoded with an entropy code while the extra bits are stored
# just as normal bits.  (The 512-entry kPrefixEncodeCode lookup upstream is
# generated from exactly this function.)
def VP8LPrefixEncodeBitsNoLUT(
    distance: Int, code: Int, extra_bits: Int
) -> Tuple[Int, Int]:
    var d = distance - 1
    var highest_bit = d.bit_width() - 1
    var second_highest_bit = (d >> (highest_bit - 1)) & 1
    return (2 * highest_bit + second_highest_bit, highest_bit - 1)


def VP8LPrefixEncodeNoLUT(
    distance: Int, code: Int, extra_bits: Int, extra_bits_value: Int
) -> Tuple[Int, Int, Int]:
    var d = distance - 1
    var highest_bit = d.bit_width() - 1
    var second_highest_bit = (d >> (highest_bit - 1)) & 1
    return (
        2 * highest_bit + second_highest_bit,
        highest_bit - 1,
        d & ((1 << (highest_bit - 1)) - 1),
    )


#------------------------------------------------------------------------------
# Image transforms.


def Average2(a0: UInt32, a1: UInt32) -> UInt32:
    return (((a0 ^ a1) & 0xfefefefe) >> 1) + (a0 & a1)


def Average3(a0: UInt32, a1: UInt32, a2: UInt32) -> UInt32:
    return Average2(Average2(a0, a2), a1)


def Average4(a0: UInt32, a1: UInt32, a2: UInt32, a3: UInt32) -> UInt32:
    return Average2(Average2(a0, a1), Average2(a2, a3))


# Vector forms of Average2 / Average4, for the predictor passes that take every
# input from the `upper` row and so can run W pixels at a time.
def _Average2V(a0: SIMD[DType.uint32, W], a1: SIMD[DType.uint32, W]) -> SIMD[
    DType.uint32, W
]:
    var m = SIMD[DType.uint32, W](0xfefefefe)
    return (((a0 ^ a1) & m) >> 1) + (a0 & a1)


def _Average4V(
    a0: SIMD[DType.uint32, W],
    a1: SIMD[DType.uint32, W],
    a2: SIMD[DType.uint32, W],
    a3: SIMD[DType.uint32, W],
) -> SIMD[DType.uint32, W]:
    return _Average2V(_Average2V(a0, a1), _Average2V(a2, a3))


def Clip255(a: Int) -> Int:
    var v = a & 0xffffffff
    if v < 256:
        return v
    # return 0, when a is a negative integer.
    # return 255, when a is positive.
    return ((0xffffffff ^ v) & 0xffffffff) >> 24


def AddSubtractComponentFull(a: Int, b: Int, c: Int) -> Int:
    return Clip255(a + b - c)


def ClampedAddSubtractFull(c0: UInt32, c1: UInt32, c2: UInt32) -> UInt32:
    var a = AddSubtractComponentFull(Int(c0 >> 24), Int(c1 >> 24), Int(c2 >> 24))
    var r = AddSubtractComponentFull(
        Int((c0 >> 16) & 0xff), Int((c1 >> 16) & 0xff), Int((c2 >> 16) & 0xff)
    )
    var g = AddSubtractComponentFull(
        Int((c0 >> 8) & 0xff), Int((c1 >> 8) & 0xff), Int((c2 >> 8) & 0xff)
    )
    var b = AddSubtractComponentFull(
        Int(c0 & 0xff), Int(c1 & 0xff), Int(c2 & 0xff)
    )
    return (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)


def AddSubtractComponentHalf(a: Int, b: Int) -> Int:
    return Clip255(a + (a - b) // 2)


def ClampedAddSubtractHalf(c0: UInt32, c1: UInt32, c2: UInt32) -> UInt32:
    var ave = Average2(c0, c1)
    var a = AddSubtractComponentHalf(Int(ave >> 24), Int(c2 >> 24))
    var r = AddSubtractComponentHalf(
        Int((ave >> 16) & 0xff), Int((c2 >> 16) & 0xff)
    )
    var g = AddSubtractComponentHalf(
        Int((ave >> 8) & 0xff), Int((c2 >> 8) & 0xff)
    )
    var b = AddSubtractComponentHalf(Int(ave & 0xff), Int(c2 & 0xff))
    return (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)


def Sub3(a: Int, b: Int, c: Int) -> Int:
    var pb = b - c
    var pa = a - c
    var apb = pa if pa >= 0 else -pa
    var bpb = pb if pb >= 0 else -pb
    return bpb - apb


def Select(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
    var pa_minus_pb = (
        Sub3(Int(a >> 24), Int(b >> 24), Int(c >> 24))
        + Sub3(Int((a >> 16) & 0xff), Int((b >> 16) & 0xff), Int((c >> 16) & 0xff))
        + Sub3(Int((a >> 8) & 0xff), Int((b >> 8) & 0xff), Int((c >> 8) & 0xff))
        + Sub3(Int(a & 0xff), Int(b & 0xff), Int(c & 0xff))
    )
    if pa_minus_pb <= 0:
        return a
    return b


#------------------------------------------------------------------------------
# Predictors

def VP8LPredictor0_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return ARGB_BLACK


def VP8LPredictor1_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return left


def VP8LPredictor2_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return top


def VP8LPredictor3_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return top_right


def VP8LPredictor4_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return top_left


def VP8LPredictor5_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average3(left, top, top_right)


def VP8LPredictor6_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average2(left, top_left)


def VP8LPredictor7_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average2(left, top)


def VP8LPredictor8_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average2(top_left, top)


def VP8LPredictor9_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average2(top, top_right)


def VP8LPredictor10_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Average4(left, top_left, top, top_right)


def VP8LPredictor11_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return Select(top, left, top_left)


def VP8LPredictor12_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return ClampedAddSubtractFull(left, top, top_left)


def VP8LPredictor13_C(left: UInt32, top_left: UInt32, top: UInt32, top_right: UInt32) -> UInt32:
    return ClampedAddSubtractHalf(left, top, top_left)


# The predictor is added to the output pixel (which is therefore considered as
# a residual) to get the final prediction.  'upper' points at the top row
# element for the first pixel, i.e. upper[-1] is the top-left neighbour.


def PredictorAdd0_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), ARGB_BLACK))
        x += 1


def PredictorAdd1_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var left = op.unsafe_load(-1)
    var x = 0
    while x < num_pixels:
        left = VP8LAddPixels(ip.unsafe_load(x), left)
        op.unsafe_store(x, left)
        x += 1


# GENERATE_PREDICTOR_ADD(VP8LPredictorN_C, PredictorAddN_C)
# Modes 2 (T), 3 (TR) and 4 (TL) take their prediction from the `upper` row
# only: none of them reads `left`, so a whole run of pixels is independent and
# the run can be done a vector at a time.  `PredictorAdd2_C` is the special
# case where the prediction is exactly `upper[x]`, so it is a plain vector
# add; 3 and 4 are shifted vector loads.
#
# DIVERGENCE (performance, not semantics): same predictions and same
# VP8LAddPixels, only the loop is widened.  The scalar tail covers a run length
# that is not a multiple of the lane count.


def PredictorAdd2_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x + W <= num_pixels:
        op.store(x, _AddPixelsV(ip.load[width=W](x), up.load[width=W](x)))
        x += W
    while x < num_pixels:
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), up.unsafe_load(x)))
        x += 1


def PredictorAdd3_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x + W <= num_pixels:
        op.store(
            x, _AddPixelsV(ip.load[width=W](x), up.load[width=W](x + 1))
        )
        x += W
    while x < num_pixels:
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), up.unsafe_load(x + 1)))
        x += 1


def PredictorAdd4_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x + W <= num_pixels:
        op.store(
            x, _AddPixelsV(ip.load[width=W](x), up.load[width=W](x - 1))
        )
        x += W
    while x < num_pixels:
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), up.unsafe_load(x - 1)))
        x += 1


def PredictorAdd5_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor5_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


# Modes 5-13 take `left`, i.e. out[x-1], as an input, so pixel x cannot be
# computed until pixel x-1 has been written.  Those four stay scalar: a vector
# body would have to read the output it is itself writing, which is exactly the
# bug `tests/test_kernels.py::test_predictor_modes_match_scalar_reference`
# pins.  Only the modes whose every input is on the `upper` row (2, 3, 4, 8, 9,
# 10) are widened.


def PredictorAdd6_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor6_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


def PredictorAdd7_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor7_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


def PredictorAdd8_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    # Mode 8 is Average2(top_left, top): both inputs are on the `upper` row, so
    # the run is fully independent and can be widened.
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x + W <= num_pixels:
        op.store(
            x,
            _AddPixelsV(
                ip.load[width=W](x),
                _Average2V(up.load[width=W](x - 1), up.load[width=W](x)),
            ),
        )
        x += W
    while x < num_pixels:
        op.unsafe_store(
            x,
            VP8LAddPixels(
                ip.unsafe_load(x), Average2(up.unsafe_load(x - 1), up.unsafe_load(x))
            ),
        )
        x += 1


def PredictorAdd9_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    # Mode 9 is Average2(top, top_right): both inputs are on the `upper` row.
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x + W <= num_pixels:
        op.store(
            x,
            _AddPixelsV(
                ip.load[width=W](x),
                _Average2V(up.load[width=W](x), up.load[width=W](x + 1)),
            ),
        )
        x += W
    while x < num_pixels:
        op.unsafe_store(
            x,
            VP8LAddPixels(
                ip.unsafe_load(x), Average2(up.unsafe_load(x), up.unsafe_load(x + 1))
            ),
        )
        x += 1


def PredictorAdd10_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    # Mode 10 is Average4(left, top_left, top, top_right) and `left` is
    # out[x-1], so this one is serial like modes 5-7 and 11-13.  It also reads
    # top_right = up[x+1] only, so it never needs up[x+2].
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor10_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


def PredictorAdd11_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor11_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


def PredictorAdd12_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor12_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


def PredictorAdd13_C(in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var up = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=upper)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var x = 0
    while x < num_pixels:
        var pred = VP8LPredictor13_C(
            op.unsafe_load(x - 1),
            up.unsafe_load(x - 1),
            up.unsafe_load(x),
            up.unsafe_load(x + 1),
        )
        op.unsafe_store(x, VP8LAddPixels(ip.unsafe_load(x), pred))
        x += 1


# VP8LPredictorsAdd[16]: the C dispatch table, kept as a branch so the padding
# sentinels at 14 and 15 still resolve to PredictorAdd0_C.
def VP8LPredictorsAdd(mode: Int, in_ptr: Int, upper: Int, num_pixels: Int, out_ptr: Int):
    if mode == 1:
        PredictorAdd1_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 2:
        PredictorAdd2_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 3:
        PredictorAdd3_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 4:
        PredictorAdd4_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 5:
        PredictorAdd5_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 6:
        PredictorAdd6_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 7:
        PredictorAdd7_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 8:
        PredictorAdd8_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 9:
        PredictorAdd9_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 10:
        PredictorAdd10_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 11:
        PredictorAdd11_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 12:
        PredictorAdd12_C(in_ptr, upper, num_pixels, out_ptr)
    elif mode == 13:
        PredictorAdd13_C(in_ptr, upper, num_pixels, out_ptr)
    else:
        PredictorAdd0_C(in_ptr, upper, num_pixels, out_ptr)


#------------------------------------------------------------------------------
# Inverse prediction.
def PredictorInverseTransform_C(
    transform: Int, y_start: Int, y_end: Int, in_ptr: Int, out_ptr: Int
):
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var width = t[].xsize
    var ip = in_ptr
    var op = out_ptr
    var y = y_start
    if y_start == 0:  # First Row follows the L (mode=1) mode.
        PredictorAdd0_C(ip, 0, 1, op)
        PredictorAdd1_C(ip + 4, 0, width - 1, op + 4)
        ip += width * 4
        op += width * 4
        y = 1

    var tile_width = 1 << t[].bits
    var mask = tile_width - 1
    var tiles_per_row = VP8LSubSampleSize(width, t[].bits)
    var pred_mode_base = t[].data + (y >> t[].bits) * tiles_per_row * 4

    # DIVERGENCE (performance, not semantics): the row body is factored into
    # _PredictorRow / PredictorRows purely to keep the loop readable; the rows
    # stay serial because each reads the output row above it.
    PredictorRows(
        y, y_end, ip, op, width, tile_width, mask, tiles_per_row, pred_mode_base
    )


# One row of the predictor transform, factored out so the row loop stays
# readable.  `ip` / `op` are byte addresses of this row's residual and output;
# `up` is the row above; `modes` is this row's slice of the predictor-mode map.
def _PredictorRow(
    ip: Int, op: Int, up: Int, width: Int, tile_width: Int, mask: Int, modes: Int
):
    var m_p = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=modes)
    # First pixel of the row follows the T (mode=2) mode.
    PredictorAdd2_C(ip, up, 1, op)
    var x = 1
    # `m_p` is walked once per tile, exactly as upstream's
    # `const uint32_t* pred_mode_src` cursor is: it advances by one word each
    # time the predictor mode changes, not once per pixel.
    var tile = 0
    while x < width:
        var mode = (Int(m_p.unsafe_load(tile)) >> 8) & 0xf
        tile += 1
        var x_end = (x & ~mask) + tile_width
        if x_end > width:
            x_end = width
        VP8LPredictorsAdd(
            mode,
            ip + x * 4,
            up + x * 4,
            x_end - x,
            op + x * 4,
        )
        x = x_end


def PredictorRows(
    first: Int,
    last: Int,
    in_ptr: Int,
    out_ptr: Int,
    width: Int,
    tile_width: Int,
    mask: Int,
    tiles_per_row: Int,
    modes_base: Int,
):
    # The mode map advances one slice every `tile_width` rows, so the slice for
    # row y is selected by `y >> bits`, exactly as upstream's
    # `if ((y & mask) == 0) pred_mode_base += tiles_per_row` does.
    #
    # NOT parallelised: row y reads the output row y-1, so rows are a serial
    # chain and `parallelize` over row bands would race.  See README
    # "Parallelism".
    var y = first
    # `in_ptr` / `out_ptr` already point at row `first`: the caller advanced
    # them past the L-mode prefix row when y_start == 0.
    var ip = in_ptr
    var op = out_ptr
    var cur = modes_base
    while y < last:
        _PredictorRow(ip, op, op - width * 4, width, tile_width, mask, cur)
        ip += width * 4
        op += width * 4
        y += 1
        if (y & mask) == 0:
            cur += tiles_per_row * 4


# Add green to blue and red channels (i.e. perform the inverse transform of
# 'subtract green').
def VP8LAddGreenToBlueAndRed(src: Int, num_pixels: Int, dst: Int):
    # DIVERGENCE (performance, not semantics): upstream's expression verbatim,
    # run W pixels at a time.  Every lane does exactly what the scalar body
    # does to one pixel; the scalar tail covers the remainder.
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=dst)
    var m_ff = SIMD[DType.uint32, W](0xff)
    var m_hi = SIMD[DType.uint32, W](0xff00ff00)
    var m_lo = SIMD[DType.uint32, W](0x00ff00ff)
    var i = 0
    while i + W <= num_pixels:
        var argb = sp.load[width=W](i)
        var green = (argb >> 8) & m_ff
        var red_blue = argb & m_lo
        red_blue += (green << 16) | green
        red_blue &= m_lo
        dp.store(i, (argb & m_hi) | red_blue)
        i += W
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var green = (argb >> 8) & 0xff
        var red_blue = argb & 0x00ff00ff
        red_blue += (green << 16) | green
        red_blue &= 0x00ff00ff
        dp.unsafe_store(i, (argb & 0xff00ff00) | red_blue)
        i += 1


def ColorTransformDelta(color_pred: Int, color: Int) -> Int:
    return (color_pred * color) >> 5


def U32ToS8(v: Int) -> Int:
    var x = v & 0xff
    if x >= 128:
        return x - 256
    return x


def ColorCodeToMultipliers(color_code: UInt32) -> VP8LMultipliers:
    var m = VP8LMultipliers(0, 0, 0)
    m.green_to_red = Int((color_code >> 0) & 0xff)
    m.green_to_blue = Int((color_code >> 8) & 0xff)
    m.red_to_blue = Int((color_code >> 16) & 0xff)
    return m


def VP8LTransformColorInverse_C(
    m: VP8LMultipliers, src: Int, num_pixels: Int, dst: Int
):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=dst)
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var green = U32ToS8(Int(argb >> 8))
        var red = argb >> 16
        var new_red = Int(red) & 0xff
        var new_blue = Int(argb & 0xff)
        new_red += ColorTransformDelta(U32ToS8(m.green_to_red), green)
        new_red &= 0xff
        new_blue += ColorTransformDelta(U32ToS8(m.green_to_blue), green)
        new_blue += ColorTransformDelta(U32ToS8(m.red_to_blue), new_red)
        new_blue &= 0xff
        dp.unsafe_store(i, (argb & 0xff00ff00) | (UInt32(new_red) << 16) | UInt32(new_blue))
        i += 1


# Color space inverse transform.
def ColorSpaceInverseTransform_C(
    transform: Int, y_start: Int, y_end: Int, src: Int, dst: Int
):
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var width = t[].xsize
    var tile_width = 1 << t[].bits
    var mask = tile_width - 1
    var safe_width = width & ~mask
    var remaining_width = width - safe_width
    var tiles_per_row = VP8LSubSampleSize(width, t[].bits)
    var y = y_start
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=dst)
    var pred_row = t[].data + (y >> t[].bits) * tiles_per_row * 4
    var sp_off = Int(sp)
    var dp_off = Int(dp)
    while y < y_end:
        # DIVERGENCE (performance, not semantics): upstream walks a
        # `const uint32_t* pred` cursor; here the tile loop uses a pointer
        # cursor, and the per-tile call is given addresses as before because
        # that is its signature.  Same tiles, same multipliers, same order.
        var pred = Pointer[UInt32, AnyOrigin[mut=True]](
            unsafe_from_address=pred_row
        )
        var src_safe_end = sp_off + safe_width * 4
        var src_end = sp_off + width * 4
        while sp_off < src_safe_end:
            var m = ColorCodeToMultipliers(pred.unsafe_load(0))
            pred += 1
            VP8LTransformColorInverse_C(m, sp_off, tile_width, dp_off)
            sp_off += tile_width * 4
            dp_off += tile_width * 4
        if sp_off < src_end:  # Left-overs using C-version.
            var m = ColorCodeToMultipliers(pred.unsafe_load(0))
            pred += 1
            VP8LTransformColorInverse_C(m, sp_off, remaining_width, dp_off)
            sp_off += remaining_width * 4
            dp_off += remaining_width * 4
        y += 1
        if (y & mask) == 0:
            pred_row += tiles_per_row * 4


# Separate out pixels packed together using pixel-bundling.
def ColorIndexInverseTransform_C(
    transform: Int, y_start: Int, y_end: Int, src: Int, dst: Int
):
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var bits_per_pixel = 8 >> t[].bits
    var width = t[].xsize
    var color_map = Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=t[].data
    )
    if bits_per_pixel < 8:
        var pixels_per_byte = 1 << t[].bits
        var count_mask = pixels_per_byte - 1
        var bit_mask = UInt32((1 << bits_per_pixel) - 1)
        var y = y_start
        var si = src
        var di = dst
        while y < y_end:
            var packed_pixels = UInt32(0)
            var x = 0
            while x < width:
                # We need to load fresh 'packed_pixels' once every
                # 'pixels_per_byte' increments of x. Fortunately,
                # pixels_per_byte is a power of 2, so can just use a mask for
                # that, instead of decrementing a counter.
                if (x & count_mask) == 0:
                    packed_pixels = (
                        Pointer[UInt32, AnyOrigin[mut=True]](
                            unsafe_from_address=si
                        ).unsafe_load(0)
                        >> 8
                    ) & 0xff
                    si += 4
                Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=di
                ).unsafe_store(
                    0, color_map.unsafe_load(Int(packed_pixels & bit_mask))
                )
                di += 4
                packed_pixels >>= UInt32(bits_per_pixel)
                x += 1
            y += 1
    else:
        MapARGB_C(src, t[].data, dst, y_start, y_end, width)


def MapARGB_C(
    src: Int, color_map: Int, dst: Int, y_start: Int, y_end: Int, width: Int
):
    var cm = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=color_map)
    var y = y_start
    var si = src
    var di = dst
    while y < y_end:
        var x = 0
        while x < width:
            var idx = VP8GetARGBIndex(
                Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=si
                ).unsafe_load(0)
            )
            Pointer[UInt32, AnyOrigin[mut=True]](
                unsafe_from_address=di
            ).unsafe_store(0, cm.unsafe_load(idx))
            si += 4
            di += 4
            x += 1
        y += 1


# Similar to the static method ColorIndexInverseTransform() that is part of
# lossless.c, but used only for alpha decoding. It takes UInt8 (rather than
# UInt32) arguments for 'src' and 'dst'.
def VP8LColorIndexInverseTransformAlpha(
    transform: Int, y_start: Int, y_end: Int, src: Int, dst: Int
):
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var bits_per_pixel = 8 >> t[].bits
    var width = t[].xsize
    var cm = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=t[].data)
    if bits_per_pixel < 8:
        var pixels_per_byte = 1 << t[].bits
        var count_mask = pixels_per_byte - 1
        var bit_mask = (1 << bits_per_pixel) - 1
        var y = y_start
        var si = src
        var di = dst
        while y < y_end:
            var packed_pixels = 0
            var x = 0
            while x < width:
                if (x & count_mask) == 0:
                    packed_pixels = Int(
                        Pointer[UInt8, AnyOrigin[mut=True]](
                            unsafe_from_address=si
                        ).unsafe_load(0)
                    )
                    si += 1
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=di
                ).unsafe_store(
                    0,
                    VP8GetAlphaValue(
                        cm.unsafe_load(packed_pixels & bit_mask)
                    ),
                )
                di += 1
                packed_pixels >>= bits_per_pixel
                x += 1
            y += 1
    else:
        MapAlpha_C(src, t[].data, dst, y_start, y_end, width)


def MapAlpha_C(
    src: Int, color_map: Int, dst: Int, y_start: Int, y_end: Int, width: Int
):
    var cm = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=color_map)
    var y = y_start
    var si = src
    var di = dst
    while y < y_end:
        var x = 0
        while x < width:
            var idx = VP8GetAlphaIndex(
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=si
                ).unsafe_load(0)
            )
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=di
            ).unsafe_store(0, VP8GetAlphaValue(cm.unsafe_load(idx)))
            si += 1
            di += 1
            x += 1
        y += 1


def VP8LInverseTransform(
    transform: Int, row_start: Int, row_end: Int, in_ptr: Int, out_ptr: Int
):
    var t = Pointer[VP8LTransform, AnyOrigin[mut=True]](
        unsafe_from_address=transform
    )
    var width = t[].xsize
    var ip = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var op = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    if t[].type == SUBTRACT_GREEN_TRANSFORM:
        VP8LAddGreenToBlueAndRed(Int(ip), (row_end - row_start) * width, Int(op))
    elif t[].type == PREDICTOR_TRANSFORM:
        PredictorInverseTransform_C(transform, row_start, row_end, Int(ip), Int(op))
        if row_end != t[].ysize:
            # The last predicted row in this iteration will be the top-pred row
            # for the first row in next iteration.
            var n = width
            var src = Int(op) + (row_end - row_start - 1) * width * 4
            var dst = Int(op) - width * 4
            var k = 0
            while k < n:
                Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=dst
                ).unsafe_store(k, Pointer[UInt32, AnyOrigin[mut=True]](
                    unsafe_from_address=src
                ).unsafe_load(k))
                k += 1
    elif t[].type == CROSS_COLOR_TRANSFORM:
        ColorSpaceInverseTransform_C(transform, row_start, row_end, Int(ip), Int(op))
    elif t[].type == COLOR_INDEXING_TRANSFORM:
        if Int(ip) == Int(op) and t[].bits > 0:
            # Move packed pixels to the end of unpacked region, so that unpacking
            # can occur seamlessly.
            var out_stride = (row_end - row_start) * width
            var in_stride = (row_end - row_start) * VP8LSubSampleSize(
                t[].xsize, t[].bits
            )
            var src = Int(op) + (out_stride - in_stride) * 4
            var n = in_stride
            var d = n - 1
            while d >= 0:
                op.unsafe_store(
                    out_stride - in_stride + d,
                    op.unsafe_load(d),
                )
                d -= 1
            ColorIndexInverseTransform_C(transform, row_start, row_end, src, Int(op))
        else:
            ColorIndexInverseTransform_C(transform, row_start, row_end, Int(ip), Int(op))


#------------------------------------------------------------------------------
# Color space conversion.

def VP8LConvertBGRAToRGB_C(src: Int, num_pixels: Int, dst: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var di = dst
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=di)
        dp.unsafe_store(0, UInt8((argb >> 16) & 0xff))
        dp.unsafe_store(1, UInt8((argb >> 8) & 0xff))
        dp.unsafe_store(2, UInt8(argb & 0xff))
        di += 3
        i += 1


def VP8LConvertBGRAToRGBA_C(src: Int, num_pixels: Int, dst: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var di = dst
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=di)
        dp.unsafe_store(0, UInt8((argb >> 16) & 0xff))
        dp.unsafe_store(1, UInt8((argb >> 8) & 0xff))
        dp.unsafe_store(2, UInt8(argb & 0xff))
        dp.unsafe_store(3, UInt8((argb >> 24) & 0xff))
        di += 4
        i += 1


def VP8LConvertBGRAToRGBA4444_C(src: Int, num_pixels: Int, dst: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var di = dst
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var rg = UInt8(((argb >> 16) & 0xf0) | ((argb >> 12) & 0xf))
        var ba = UInt8(((argb >> 0) & 0xf0) | ((argb >> 28) & 0xf))
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=di)
        dp.unsafe_store(0, rg)
        dp.unsafe_store(1, ba)
        di += 2
        i += 1


def VP8LConvertBGRAToRGB565_C(src: Int, num_pixels: Int, dst: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var di = dst
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var rg = UInt8(((argb >> 16) & 0xf8) | ((argb >> 13) & 0x7))
        var gb = UInt8(((argb >> 5) & 0xe0) | ((argb >> 3) & 0x1f))
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=di)
        dp.unsafe_store(0, rg)
        dp.unsafe_store(1, gb)
        di += 2
        i += 1


def VP8LConvertBGRAToBGR_C(src: Int, num_pixels: Int, dst: Int):
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=src)
    var di = dst
    var i = 0
    while i < num_pixels:
        var argb = sp.unsafe_load(i)
        var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=di)
        dp.unsafe_store(0, UInt8(argb & 0xff))
        dp.unsafe_store(1, UInt8((argb >> 8) & 0xff))
        dp.unsafe_store(2, UInt8((argb >> 16) & 0xff))
        di += 3
        i += 1


# Converts from BGRA to other color spaces.  Premultiplied modes (rgbA, bgrA,
# Argb, rgbA_4444) need WebPApplyAlphaMultiply from dsp/alpha_processing.c,
# which is not part of this port; they are reported as unsupported.
def VP8LConvertFromBGRA(
    in_data: Int, num_pixels: Int, out_colorspace: Int, rgba: Int
) -> Int:
    if out_colorspace == MODE_RGB:
        VP8LConvertBGRAToRGB_C(in_data, num_pixels, rgba)
    elif out_colorspace == MODE_RGBA:
        VP8LConvertBGRAToRGBA_C(in_data, num_pixels, rgba)
    elif out_colorspace == MODE_BGR:
        VP8LConvertBGRAToBGR_C(in_data, num_pixels, rgba)
    elif out_colorspace == MODE_BGRA:
        # CopyOrSwap(in_data, num_pixels, rgba, 1): a plain copy on a
        # little-endian host, so the bytes land as B, G, R, A.
        var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_data)
        var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=rgba)
        var i = 0
        while i < num_pixels:
            dp.unsafe_store(i, sp.unsafe_load(i))
            i += 1
    elif out_colorspace == MODE_ARGB:
        # CopyOrSwap(in_data, num_pixels, rgba, 0): also a plain copy here.
        var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=in_data)
        var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=rgba)
        var i = 0
        while i < num_pixels:
            dp.unsafe_store(i, sp.unsafe_load(i))
            i += 1
    elif out_colorspace == MODE_RGBA_4444:
        VP8LConvertBGRAToRGBA4444_C(in_data, num_pixels, rgba)
    elif out_colorspace == MODE_RGB_565:
        VP8LConvertBGRAToRGB565_C(in_data, num_pixels, rgba)
    else:
        return 0
    return 1
