#! /bin/bash

# IMPORTANT: Run this from the workspace root
#
# Usage:
#   ./scripts/ptx.sh <path/to/file.cu>
#   ./scripts/ptx.sh --out foo.ptx <path/to/file.cu>
#   ./scripts/ptx.sh --cuda-arch 89 --threads 16 <path/to/file.cu>

set -euo pipefail

cargo run -p openvm-scripts --features cuda --bin ptx-details -- "$@"
