#!/bin/bash

# Download ~/.openvm setup artifacts
HALO2_DIR="halo2/src/v1.4"
mkdir -p ~/.openvm
mkdir -p ~/.openvm/$HALO2_DIR
mkdir -p ~/.openvm/$HALO2_DIR/interfaces
mkdir -p ~/.openvm/params

BASE_URL="https://openvm-public-artifacts-us-east-1.s3.us-east-1.amazonaws.com/v1.4.1"

for file in "agg_stark.pk" "agg_stark.vk" "agg_halo2.pk" "root.asm"; do
    URL="$BASE_URL/$file"
    LOCAL=~/.openvm/$file
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done

for file in "Halo2Verifier.sol" "interfaces/IOpenVmHalo2Verifier.sol" "OpenVmHalo2Verifier.sol" "verifier.bytecode.json"; do
    URL="$BASE_URL/$HALO2_DIR/$file"
    LOCAL=~/.openvm/$HALO2_DIR/$file
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done

for k in {10..23}; do
    file="kzg_bn254_${k}.srs"
    URL="$BASE_URL/params/$file"
    LOCAL=~/.openvm/params/$file
    wget "$URL" -O "$LOCAL" || curl -L "$URL" -o "$LOCAL"
done
