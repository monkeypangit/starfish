#!/bin/sh
# Builds the web viewer into web/dist/: the wasm module and its JS glue, the
# page, and the model exports with a files.txt each (the page fetches them).
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli at the version
# of wasm-bindgen in Cargo.lock (cargo install wasm-bindgen-cli --version =X).
# Serve web/dist/ over HTTP (WebGPU wants localhost or HTTPS), e.g.
#   python3 -m http.server -d web/dist 8000
set -eu
cd "$(dirname "$0")/.."

cargo build --release -p starfish-web --target wasm32-unknown-unknown
rm -rf web/dist
mkdir -p web/dist
wasm-bindgen --target web --no-typescript --out-dir web/dist \
    target/wasm32-unknown-unknown/release/starfish_web.wasm
cp web/index.html web/dist/
for export in bayer_wgpu/f16 bayer_wgpu/f32 xtrans_wgpu/f16 xtrans_wgpu/f32; do
    mkdir -p "web/dist/models/$export"
    cp "training/$export"/* "web/dist/models/$export/"
    ls "training/$export" > "web/dist/models/$export/files.txt"
done
du -sh web/dist
