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

output=$(cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path" \
  --lean-verified 2>&1 | tee /dev/stderr)

# ensure the Lean verifier actually ran rather than the flag being ignored
grep -q "Lean verifier accepted the proof" <<<"$output"
