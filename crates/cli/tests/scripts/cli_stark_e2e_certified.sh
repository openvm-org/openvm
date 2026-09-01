#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

proof_path="$temp_dir/fibonacci.stark.proof"
canonical_standard_config="$temp_dir/openvm-standard.toml"
riscv_proof_path="$temp_dir/fibonacci-riscv.stark.proof"
riscv_config="$temp_dir/openvm-riscv32.toml"
init_file="$temp_dir/openvm_init.rs"

cp ../sdk-config/src/openvm_standard.toml "$canonical_standard_config"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml \
  --config "$canonical_standard_config"

cargo openvm setup

cargo openvm prove stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --config "$canonical_standard_config" \
  --init-file-name "$init_file" \
  --proof "$proof_path"

output=$(cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path" \
  --certified 2>&1 | tee /dev/stderr)

# ensure the certified verifier actually ran rather than the flag being ignored
grep -q "Certified verifier accepted the proof" <<<"$output"

# A proof generated with the canonical RISC-V configuration is still a valid
# STARK proof, but it is outside the standard configuration covered by certified verification.
printf '[app_vm_config.rv32i]\n[app_vm_config.rv32m]\n[app_vm_config.io]\n' \
  > "$riscv_config"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml \
  --config "$riscv_config"

cargo openvm prove stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --config "$riscv_config" \
  --init-file-name "$init_file" \
  --proof "$riscv_proof_path"

# Establish that rejection below comes from the certified verifier's config scope,
# not from ordinary STARK verification or a malformed proof.
cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$riscv_proof_path"

if output=$(cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$riscv_proof_path" \
  --certified 2>&1); then
  echo "Certified verification unexpectedly accepted the RISC-V VM config" >&2
  exit 1
fi
printf '%s\n' "$output" >&2
grep -q "baseline does not match the canonical standard pipeline" <<<"$output"
