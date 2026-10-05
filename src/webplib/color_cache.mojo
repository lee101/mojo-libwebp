"""Port of src/utils/color_cache_utils.c and src/utils/color_cache_utils.h.

Divergence: upstream's VP8LColorCacheInit calls WebPSafeCalloc for the entry
array.  Mojo has no heap, so the caller (VP8LDecoder.alloc) hands in a
zero-filled arena region; see the note in vp8l_dec.mojo.  `colors` is therefore
an absolute address rather than a `uint32_t*`.
"""

comptime kHashMul: UInt32 = 0x1e35a7bd


struct VP8LColorCache:
    var colors: Int   # address of the entries
    var hash_shift: Int
    var hash_bits: Int


# Initializes the color cache with 'hash_bits' bits for the keys.  'colors' is
# the address of a zero-filled (1 << hash_bits) entry region.
def VP8LColorCacheInit(color_cache: Int, hash_bits: Int, colors: Int) -> Int:
    var cc = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=color_cache
    )
    cc[].colors = colors
    cc[].hash_shift = 32 - hash_bits
    cc[].hash_bits = hash_bits
    return 1


def VP8LColorCacheClear(color_cache: Int):
    var cc = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=color_cache
    )
    cc[].colors = 0
    cc[].hash_bits = 0


def VP8LColorCacheCopy(src: Int, dst: Int):
    var s = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=src
    )
    var d = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=dst
    )
    d[].hash_bits = s[].hash_bits
    d[].hash_shift = s[].hash_shift
    d[].colors = s[].colors
    var n = 1 << s[].hash_bits
    var sp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=s[].colors)
    var dp = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=d[].colors)
    var i = 0
    while i < n:
        dp.unsafe_store(i, sp.unsafe_load(i))
        i += 1


def VP8LColorCacheLookup(cc: Int, key: Int) -> UInt32:
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=cc
    )
    return Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=c[].colors
    ).unsafe_load(key)


def VP8LHashPix(argb: UInt32, shift: Int) -> Int:
    return Int((argb * kHashMul) >> UInt32(shift))


def VP8LColorCacheSet(cc: Int, key: Int, argb: UInt32):
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=cc
    )
    Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=c[].colors
    ).unsafe_store(key, argb)


def VP8LColorCacheInsert(cc: Int, argb: UInt32):
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=cc
    )
    Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=c[].colors
    ).unsafe_store(VP8LHashPix(argb, c[].hash_shift), argb)


def VP8LColorCacheGetIndex(cc: Int, argb: UInt32) -> Int:
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=cc
    )
    return VP8LHashPix(argb, c[].hash_shift)


# Return the key if cc contains argb, and -1 otherwise.
def VP8LColorCacheContains(cc: Int, argb: UInt32) -> Int:
    var c = Pointer[VP8LColorCache, AnyOrigin[mut=True]](
        unsafe_from_address=cc
    )
    var key = VP8LHashPix(argb, c[].hash_shift)
    if Pointer[UInt32, AnyOrigin[mut=True]](
        unsafe_from_address=c[].colors
    ).unsafe_load(key) == argb:
        return key
    return -1
