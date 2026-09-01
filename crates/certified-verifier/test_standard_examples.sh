#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
examples_dir="$repo_root/examples"
standard_config="$repo_root/crates/sdk-config/src/openvm_standard.toml"

standard_examples=(ecc ecdsa i256 keccak pairing sha2 u256)
init_examples=(ecc ecdsa pairing)

work_dir="$(mktemp -d)"
backup_dir="$work_dir/init-backups"
keys_dir="$work_dir/keys"
target_dir="$work_dir/target"
mkdir -p "$backup_dir"

for example in "${init_examples[@]}"; do
  cp "$examples_dir/$example/openvm_init.rs" "$backup_dir/$example.rs"
done

restore_init_files() {
  for example in "${init_examples[@]}"; do
    cp "$backup_dir/$example.rs" "$examples_dir/$example/openvm_init.rs"
  done
}

cleanup() {
  restore_init_files
  rm -rf "$work_dir"
}
trap cleanup EXIT

# The application and aggregation-prefix keys depend on the VM configuration, not the executable,
# so generate the canonical standard keys once and reuse them for every example.
cargo openvm keygen \
  --manifest-path "$examples_dir/sha2/Cargo.toml" \
  --target-dir "$target_dir" \
  --config "$standard_config" \
  --output-dir "$keys_dir"
cargo openvm setup

for example in "${standard_examples[@]}"; do
  manifest="$examples_dir/$example/Cargo.toml"
  proof="$work_dir/$example.stark.proof"
  prove_args=(
    --manifest-path "$manifest"
    --target-dir "$target_dir"
    --config "$standard_config"
    --app-pk "$keys_dir/app.pk"
    --agg-prefix-pk "$keys_dir/agg_prefix.pk"
    --proof "$proof"
  )

  # Examples that call openvm::init!() need the standard configuration's generated init file at
  # the default manifest-relative path. Their checked-in, example-specific files are restored
  # immediately after proving and again by the EXIT trap if proving fails.
  case "$example" in
    ecc | ecdsa | pairing) ;;
    *) prove_args+=(--init-file-name "$work_dir/$example-openvm_init.rs") ;;
  esac

  echo "Proving and certified-verifying the $example example with the standard configuration"
  cargo openvm prove stark "${prove_args[@]}"
  case "$example" in
    ecc | ecdsa | pairing) cp "$backup_dir/$example.rs" "$examples_dir/$example/openvm_init.rs" ;;
  esac

  cargo openvm verify stark \
    --manifest-path "$manifest" \
    --target-dir "$target_dir" \
    --proof "$proof" \
    --certified
done
