#!/usr/bin/env bash
set -euo pipefail

: "${EXE_PATH:?EXE_PATH must be set}"

cargo openvm run \
  --exe "$EXE_PATH" \
  --config tests/programs/fibonacci/openvm.toml

cargo openvm run \
  --exe "$EXE_PATH" \
  --config tests/programs/fibonacci/openvm.toml \
  --mode pure
