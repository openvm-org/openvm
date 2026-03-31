#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

proof_path="$temp_dir/fibonacci.app.proof"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml

cargo openvm prove app \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path"

cargo openvm verify app \
  --manifest-path tests/programs/multi/Cargo.toml \
  --proof "$proof_path"
