"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOLIBWEBP_LIB") or os.path.join(
    ROOT, "dist", "libmojo-libwebp.so"
)
ENTRYPOINT = os.path.join(SRC, "mojowebp_capi.mojo")

# Every decoder state struct is a record of machine words laid out in a
# caller-owned buffer.  The sizes below are the real Mojo layouts, measured
# from the structs themselves; tests/test_units.py pins each of them against the
# compiled library through `mw_struct_size`, so they cannot drift.
DEC_WORDS = 40      # VP8LDecoder     -- 40 Int fields
BR_WORDS = 6        # VP8LBitReader   -- 6 fields (val is UInt64)
CC_WORDS = 3        # VP8LColorCache  -- 3 Int fields
HT_WORDS = 5        # HuffmanTables   -- HuffmanTablesSegment + curr_segment
TMP_WORDS = 3       # ReadImageInfo out-params (width, height, has_alpha)

I = ctypes.c_int64

_SIGNATURES = {
    "mw_vp8l_check_signature": ([I, I], I),
    "mw_vp8l_get_info": ([I, I, I, I, I, I, I], I),
    "mw_vp8l_workspace_size": ([I] * 9, I),
    "mw_vp8l_decode": ([I] * 12, I),
    "mw_vp8l_decode_alpha": ([I] * 13, I),
    "mw_vp8l_inspect_state": ([I] * 10, I),
    "mw_vp8l_inspect_transform": ([I] * 3, I),
    "mw_vp8l_read_bits": ([I] * 5, I),
    "mw_vp8l_build_huffman_table": ([I] * 7, I),
    "mw_vp8l_decoder_field": ([I] * 2, I),
    "mw_vp8l_dump_pixels": ([I] * 3, I),
    "mw_vp8l_inverse_transform": ([I] * 6, None),
    "mw_vp8l_inspect_group": ([I] * 3, I),
    "mw_vp8l_dump_transform_data": ([I] * 4, I),
    "mw_vp8l_br_state": ([I] * 2, I),
    "mw_kernel_run": ([I] * 6, None),
    "mw_struct_size": ([I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOLIBWEBP_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOLIBWEBP_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/mojowebp_capi.mojo` into `dist/libmojo-libwebp.so` if stale."""
    sources = [ENTRYPOINT] + [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(os.path.join(SRC, "webplib"))
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC, ENTRYPOINT, "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def _bind(handle: ctypes.CDLL) -> ctypes.CDLL:
    for name, (argtypes, restype) in _SIGNATURES.items():
        fn = getattr(handle, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return handle


def lib() -> ctypes.CDLL:
    """The compiled library, rebuilt if the file on disk is not ours."""
    global _lib
    if _lib is None:
        try:
            _lib = _bind(ctypes.CDLL(build()))
        except AttributeError:
            # dist/ holds a build without our entry point: force ours.
            _lib = _bind(ctypes.CDLL(build(force=True)))
    return _lib


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojowebp._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
