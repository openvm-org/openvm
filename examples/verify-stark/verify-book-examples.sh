#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
examples_dir="$repo_root/examples"

host_manifest="$script_dir/host/Cargo.toml"
host_bin="$script_dir/host/target/release/main"
artifacts_dir="$script_dir/openvm"
sdk_pk="$artifacts_dir/sdk.pk"
vmexe="$artifacts_dir/verify-stark.vmexe"
baseline="$artifacts_dir/verify-stark.baseline.json"
verify_stark_agg_vk="$artifacts_dir/internal_recursive.vk"
child_agg_vk="$HOME/.openvm/internal_recursive.vk"
host_features="${VERIFY_STARK_HOST_FEATURES-cuda}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <num_def_circuits> <num_proves_per_circuit>"
  echo "Example: $0 5 1,0,3,0,2"
  exit 1
fi

num_def_circuits="$1"
num_proves_per_circuit="$2"

IFS=',' read -r -a prove_counts <<< "$num_proves_per_circuit"
if [[ "${#prove_counts[@]}" -ne "$num_def_circuits" ]]; then
  echo "num_proves_per_circuit length (${#prove_counts[@]}) must equal num_def_circuits ($num_def_circuits)"
  exit 1
fi

mkdir -p "$artifacts_dir"

cargo openvm setup --force

host_build_cmd=(cargo build --release --manifest-path "$host_manifest" --bin main)
if [[ -n "$host_features" ]]; then
  host_build_cmd+=(--features "$host_features")
fi
"${host_build_cmd[@]}"

"$host_bin" keygen \
  --child-agg-vk "$child_agg_vk" \
  --num-def-circuits "$num_def_circuits" \
  --sdk-pk "$sdk_pk" \
  --openvm-toml "$artifacts_dir/openvm.toml" \
  --agg-vk "$verify_stark_agg_vk"
"$host_bin" build \
  --sdk-pk "$sdk_pk" \
  --vmexe "$vmexe" \
  --baseline "$baseline"

for manifest in "$examples_dir"/*/Cargo.toml; do
  example_dir="$(dirname "$manifest")"
  example="$(basename "$example_dir")"
  target_name="$example-example"

  echo "Verifying $example with verify-stark"
  (
    cd "$example_dir"
    cargo openvm build
    cargo openvm keygen
    cargo openvm prove stark --proof "$example_dir/$target_name.stark.proof"
  )

  verify_stark_proof="$artifacts_dir/$target_name.verify-stark.stark.proof"
  "$host_bin" prove \
    --sdk-pk "$sdk_pk" \
    --vmexe "$vmexe" \
    --child-agg-vk "$child_agg_vk" \
    --child-baseline "$example_dir/openvm/release/$target_name.baseline.json" \
    --input-proof "$example_dir/$target_name.stark.proof" \
    --num-proves-per-circuit "$num_proves_per_circuit" \
    --output-proof "$verify_stark_proof"
  cargo openvm verify stark \
    --manifest-path "$manifest" \
    --agg-vk "$verify_stark_agg_vk" \
    --app-baseline "$baseline" \
    --proof "$verify_stark_proof"
done
