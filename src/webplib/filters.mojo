"""Port of src/dsp/filters.c (the WEBP_FILTER_LAST family).

Divergence: the WebPFilters / WebPUnfilters function-pointer tables become a
dispatcher on the filter type; the filter ids and every arithmetic step are
unchanged.
"""

# WebPFilterType
comptime WEBP_FILTER_NONE: Int = 0
comptime WEBP_FILTER_HORIZONTAL: Int = 1
comptime WEBP_FILTER_VERTICAL: Int = 2
comptime WEBP_FILTER_GRADIENT: Int = 3
comptime WEBP_FILTER_LAST: Int = 4


def PredictLine_C(src: Int, pred: Int, dst: Int, length: Int):
    var sp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=src)
    var pp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=pred)
    var dp = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=dst)
    var i = 0
    while i < length:
        dp.unsafe_store(i, UInt8(Int(sp.unsafe_load(i)) - Int(pp.unsafe_load(i))))
        i += 1


#------------------------------------------------------------------------------
# Horizontal filter.

def DoHorizontalFilter_C(inp: Int, width: Int, height: Int, stride: Int, outp: Int):
    var preds = inp
    var in_off = inp
    var out_off = outp
    var op = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=outp)
    var ip = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=inp)

    # Leftmost pixel is the same as input for topmost scanline.
    op.unsafe_store(0, ip.unsafe_load(0))
    PredictLine_C(in_off + 1, preds, out_off + 1, width - 1)
    preds += stride
    in_off += stride
    out_off += stride

    # Filter line-by-line.
    var row = 1
    while row < height:
        # Leftmost pixel is predicted from above.
        PredictLine_C(in_off, preds - stride, out_off, 1)
        PredictLine_C(in_off + 1, preds, out_off + 1, width - 1)
        preds += stride
        in_off += stride
        out_off += stride
        row += 1


#------------------------------------------------------------------------------
# Vertical filter.

def DoVerticalFilter_C(inp: Int, width: Int, height: Int, stride: Int, outp: Int):
    var preds = inp
    var in_off = inp
    var out_off = outp
    var op = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=outp)
    var ip = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=inp)

    # Very first top-left pixel is copied.
    op.unsafe_store(0, ip.unsafe_load(0))
    # Rest of top scan-line is left-predicted.
    PredictLine_C(in_off + 1, preds, out_off + 1, width - 1)
    in_off += stride
    out_off += stride

    # Filter line-by-line.
    var row = 1
    while row < height:
        PredictLine_C(in_off, preds, out_off, width)
        preds += stride
        in_off += stride
        out_off += stride
        row += 1


#------------------------------------------------------------------------------
# Gradient filter.

def GradientPredictor_C(a: Int, b: Int, c: Int) -> Int:
    var g = a + b - c
    if (g & ~0xff) == 0:
        return g
    if g < 0:
        return 0
    return 255  # clip to 8bit


def DoGradientFilter_C(inp: Int, width: Int, height: Int, stride: Int, outp: Int):
    var preds = inp
    var in_off = inp
    var out_off = outp
    var op = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=outp)
    var ip = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=inp)

    # left prediction for top scan-line
    op.unsafe_store(0, ip.unsafe_load(0))
    PredictLine_C(in_off + 1, preds, out_off + 1, width - 1)
    preds += stride
    in_off += stride
    out_off += stride

    # Filter line-by-line.
    var row = 1
    while row < height:
        var w = 0
        # leftmost pixel: predict from above.
        PredictLine_C(in_off, preds - stride, out_off, 1)
        while w < width:
            var pred = GradientPredictor_C(
                Int(Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=preds
                ).unsafe_load(w - 1)),
                Int(Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=preds
                ).unsafe_load(w - stride)),
                Int(Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=preds
                ).unsafe_load(w - stride - 1)),
            )
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=out_off
            ).unsafe_store(
                w,
                UInt8(
                    Int(
                        Pointer[UInt8, AnyOrigin[mut=True]](
                            unsafe_from_address=in_off
                        ).unsafe_load(w)
                    ) - pred
                ),
            )
            w += 1
        preds += stride
        in_off += stride
        out_off += stride
        row += 1


def HorizontalFilter_C(data: Int, width: Int, height: Int, stride: Int, filtered_data: Int):
    DoHorizontalFilter_C(data, width, height, stride, filtered_data)


def VerticalFilter_C(data: Int, width: Int, height: Int, stride: Int, filtered_data: Int):
    DoVerticalFilter_C(data, width, height, stride, filtered_data)


def GradientFilter_C(data: Int, width: Int, height: Int, stride: Int, filtered_data: Int):
    DoGradientFilter_C(data, width, height, stride, filtered_data)


#------------------------------------------------------------------------------

def NoneUnfilter_C(prev: Int, inp: Int, outp: Int, width: Int):
    if outp != inp:
        var i = 0
        while i < width:
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=outp
            ).unsafe_store(
                i,
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=inp
                ).unsafe_load(i),
            )
            i += 1


def HorizontalUnfilter_C(prev: Int, inp: Int, outp: Int, width: Int):
    var pred = 0
    if prev != 0:
        pred = Int(
            Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=prev).unsafe_load(0)
        )
    var i = 0
    while i < width:
        var v = UInt8((pred + Int(
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=inp
            ).unsafe_load(i)
        )) & 0xff)
        Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=outp).unsafe_store(
            i, v
        )
        pred = Int(v)
        i += 1


def VerticalUnfilter_C(prev: Int, inp: Int, outp: Int, width: Int):
    if prev == 0:
        HorizontalUnfilter_C(0, inp, outp, width)
    else:
        var i = 0
        while i < width:
            Pointer[UInt8, AnyOrigin[mut=True]](
                unsafe_from_address=outp
            ).unsafe_store(
                i,
                UInt8(
                    (
                        Int(
                            Pointer[UInt8, AnyOrigin[mut=True]](
                                unsafe_from_address=prev
                            ).unsafe_load(i)
                        )
                        + Int(
                            Pointer[UInt8, AnyOrigin[mut=True]](
                                unsafe_from_address=inp
                            ).unsafe_load(i)
                        )
                    ) & 0xff
                ),
            )
            i += 1


def GradientUnfilter_C(prev: Int, inp: Int, outp: Int, width: Int):
    if prev == 0:
        HorizontalUnfilter_C(0, inp, outp, width)
    else:
        var top = Int(
            Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=prev).unsafe_load(0)
        )
        var top_left = top
        var left = top
        var i = 0
        while i < width:
            top = Int(
                Pointer[UInt8, AnyOrigin[mut=True]](
                    unsafe_from_address=prev
                ).unsafe_load(i)
            )  # need to read this first, in case prev==out
            left = (
                Int(
                    Pointer[UInt8, AnyOrigin[mut=True]](
                        unsafe_from_address=inp
                    ).unsafe_load(i)
                )
                + GradientPredictor_C(left, top, top_left)
            ) & 0xff
            top_left = top
            Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=outp).unsafe_store(
                i, UInt8(left)
            )
            i += 1


# WebPUnfilters[WEBP_FILTER_LAST]
def WebPUnfilter(filter: Int, prev: Int, inp: Int, outp: Int, width: Int):
    if filter == WEBP_FILTER_HORIZONTAL:
        HorizontalUnfilter_C(prev, inp, outp, width)
    elif filter == WEBP_FILTER_VERTICAL:
        VerticalUnfilter_C(prev, inp, outp, width)
    elif filter == WEBP_FILTER_GRADIENT:
        GradientUnfilter_C(prev, inp, outp, width)
    else:
        NoneUnfilter_C(prev, inp, outp, width)


# WebPFilters[WEBP_FILTER_LAST]
def WebPFilter(filter: Int, data: Int, width: Int, height: Int, stride: Int, out_ptr: Int):
    if filter == WEBP_FILTER_HORIZONTAL:
        HorizontalFilter_C(data, width, height, stride, out_ptr)
    elif filter == WEBP_FILTER_VERTICAL:
        VerticalFilter_C(data, width, height, stride, out_ptr)
    elif filter == WEBP_FILTER_GRADIENT:
        GradientFilter_C(data, width, height, stride, out_ptr)
