#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

proof_path="$temp_dir/fibonacci.stark.proof"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml

cargo openvm setup

cargo openvm prove stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path"

cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path"
