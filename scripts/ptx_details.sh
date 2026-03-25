#! /bin/bash

# IMPORTANT: Run this from the workspace root
#
# Usage:
#   ./scripts/ptx_details.sh <path/to/file.cu>
#   ./scripts/ptx_details.sh --out foo.ptx <path/to/file.cu>
#   ./scripts/ptx_details.sh --cuda-arch 89 --threads 16 <path/to/file.cu>

set -euo pipefail

cargo run -p openvm-scripts --features cuda --bin ptx-details -- "$@"
