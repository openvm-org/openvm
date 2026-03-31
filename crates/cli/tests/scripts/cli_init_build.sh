#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

cargo openvm init "$temp_dir" --name cli-package

manifest_path="$temp_dir/Cargo.toml"
config_path="$temp_dir/openvm.toml"

if [[ "${USE_LOCAL_OPENVM:-}" == "1" ]]; then
  local_openvm_path="$(cd ../toolchain/openvm && pwd)"
  perl -0pi -e 's#openvm = \{ git = "https://github.com/openvm-org/openvm\.git".*?\}#openvm = { path = "'"$local_openvm_path"'", features = ["std"] }#' "$manifest_path"
fi

cargo openvm build \
  --config "$config_path" \
  --manifest-path "$manifest_path"
