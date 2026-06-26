#!/usr/bin/env bash
# Build the landing-page WASM mesher and sync it into the site's static assets.
# Run after changing the Rust engine (src/lib.rs) or surf2d.
set -euo pipefail
cd "$(dirname "$0")"
wasm-pack build --release --target web
dest="../../site/static/wasm"
mkdir -p "$dest"
cp pkg/rapidmesh_wasm.js pkg/rapidmesh_wasm_bg.wasm "$dest/"
echo "synced rapidmesh_wasm.{js,wasm} -> $dest"
