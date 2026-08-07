#!/usr/bin/env bash
set -euo pipefail

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

proof_path="$temp_dir/fibonacci.stark.proof"
canonical_riscv_config="$temp_dir/openvm-riscv32.toml"
non_riscv_proof_path="$temp_dir/fibonacci-non-riscv.stark.proof"
non_riscv_config="$temp_dir/openvm-non-riscv.toml"

# The multi-program fixture omits the IO extension, but the FV verifier only
# covers the canonical RV32IM+IO configuration.
printf '[app_vm_config.rv32i]\n[app_vm_config.rv32m]\n[app_vm_config.io]\n' \
  > "$canonical_riscv_config"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml \
  --config "$canonical_riscv_config"

cargo openvm setup

cargo openvm prove stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --config "$canonical_riscv_config" \
  --proof "$proof_path"

output=$(cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$proof_path" \
  --fv-verified 2>&1 | tee /dev/stderr)

# ensure the FV verifier actually ran rather than the flag being ignored
grep -q "FV verifier accepted the proof" <<<"$output"

# A proof generated with an additional VM extension is still a valid STARK
# proof, but it is outside the canonical RISC-V configuration covered by the FV verifier.
cp "$canonical_riscv_config" "$non_riscv_config"
printf '\n[app_vm_config.keccak]\n' >> "$non_riscv_config"

cargo openvm keygen \
  --manifest-path tests/programs/multi/Cargo.toml \
  --config "$non_riscv_config"

cargo openvm prove stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --config "$non_riscv_config" \
  --proof "$non_riscv_proof_path"

# Establish that rejection below comes from the FV verifier's config scope,
# not from ordinary STARK verification or a malformed proof.
cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$non_riscv_proof_path"

if output=$(cargo openvm verify stark \
  --manifest-path tests/programs/multi/Cargo.toml \
  --example fibonacci \
  --proof "$non_riscv_proof_path" \
  --fv-verified 2>&1); then
  echo "FV verification unexpectedly accepted a non-RISC-V VM config" >&2
  exit 1
fi
printf '%s\n' "$output" >&2
grep -q "baseline does not match the canonical riscv32 pipeline" <<<"$output"
