#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/mojowebp_capi.mojo is the single entry point; -I src resolves the
# webplib.* module imports that hold the kernels.
mojo build --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/mojowebp_capi.mojo" \
    -o "$repo_dir/dist/libmojo-libwebp.so"
