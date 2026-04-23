#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

cargo openvm init "$temp_dir" --name cli-package

manifest_path="$temp_dir/Cargo.toml"
config_path="$temp_dir/openvm.toml"

if [[ "${USE_LOCAL_OPENVM:-}" == "1" ]]; then
  local_openvm_path="$(cd ../toolchain/openvm && pwd)"
  echo "Using local openvm dependency at $local_openvm_path"
  perl -0pi -e 's#openvm = \{ git = "https://github.com/openvm-org/openvm\.git".*?\}#openvm = { path = "'"$local_openvm_path"'" }#' "$manifest_path"
else
  echo "Using generated openvm dependency from cargo openvm init"
fi

cargo openvm build \
  --config "$config_path" \
  --manifest-path "$manifest_path"
