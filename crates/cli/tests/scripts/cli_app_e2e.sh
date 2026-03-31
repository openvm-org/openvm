#!/usr/bin/env bash
set -euo pipefail

: "${EXE_PATH:?EXE_PATH must be set}"

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

cargo openvm keygen \
  --config tests/programs/fibonacci/openvm.toml \
  --output-dir "$temp_dir"

cargo openvm run \
  --exe "$EXE_PATH" \
  --config tests/programs/fibonacci/openvm.toml

cargo openvm prove app \
  --app-pk "$temp_dir/app.pk" \
  --exe "$EXE_PATH" \
  --proof "$temp_dir/fibonacci.app.proof"

cargo openvm verify app \
  --app-vk "$temp_dir/app.vk" \
  --proof "$temp_dir/fibonacci.app.proof"
