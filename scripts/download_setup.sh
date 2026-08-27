#!/bin/bash

# Download ~/.openvm setup artifacts
FULL_VERSION="2.1.0"
OPENVM_VERSION="${FULL_VERSION%.*}"
CACHE_DIR="$HOME/.openvm/v$OPENVM_VERSION"
HALO2_DIR="halo2/src/v$OPENVM_VERSION-base"
mkdir -p "$CACHE_DIR/$HALO2_DIR/interfaces"
mkdir -p ~/.openvm/params

BASE_URL="https://openvm-public-artifacts-us-east-1.s3.us-east-1.amazonaws.com/v$FULL_VERSION"

for file in "internal_recursive.pk" "internal_recursive.vk" "root.pk" "halo2.pk"; do
    URL="$BASE_URL/$file"
    LOCAL="$CACHE_DIR/$file"
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done

for file in "Halo2Verifier.sol" "interfaces/IOpenVmHalo2Verifier.sol" "OpenVmHalo2Verifier.sol" "verifier.bytecode.json"; do
    URL="$BASE_URL/$HALO2_DIR/$file"
    LOCAL="$CACHE_DIR/$HALO2_DIR/$file"
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done

for k in {10..24}; do
    file="kzg_bn254_${k}.srs"
    URL="$BASE_URL/params/$file"
    LOCAL=~/.openvm/params/$file
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done
