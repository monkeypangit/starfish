#!/bin/sh
# Copies the exported networks from the training repo (default ../starfish-neural-demosaic,
# or the first argument) into models/, without the test.bin self-checks.
set -eu
cd "$(dirname "$0")/.."
src="${1:-../starfish-neural-demosaic}"
for dir in "$src"/*_wgpu; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    rm -rf "models/$name"
    mkdir -p "models/$name"
    for prec in f16 f32; do
        [ -d "$dir/$prec" ] && cp -r "$dir/$prec" "models/$name/$prec"
    done
    echo "$name"
done
