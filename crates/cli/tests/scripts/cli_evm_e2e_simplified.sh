#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

proof_path="$temp_dir/fibonacci.evm.proof"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml

cargo openvm setup --evm

cargo openvm prove evm \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path"

cargo openvm verify evm \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path"
