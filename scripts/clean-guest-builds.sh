#!/usr/bin/env bash
# Remove cached guest build artifacts from the workspace target directory.
# Guest builds target riscv64im-unknown-openvm-elf and store transpiled output under openvm/.
#
# Usage: ./scripts/clean-guest-builds.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TARGET_DIR="${REPO_ROOT}/target"

dirs_to_clean=(
    "${TARGET_DIR}/riscv64im-unknown-openvm-elf"
    "${TARGET_DIR}/openvm"
)

for dir in "${dirs_to_clean[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir"
        rm -rf "$dir"
    fi
done

echo "Guest build cache cleaned."
