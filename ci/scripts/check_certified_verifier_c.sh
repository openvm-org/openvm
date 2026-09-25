#!/usr/bin/env bash

set -euo pipefail

openvm_fv_dir="${1:?usage: check_certified_verifier_c.sh <openvm-fv checkout>}"
vendored_dir="crates/certified-verifier/csrc"
vm_trace="$openvm_fv_dir/.lake/build/bin/vm_verify.trace"
dump_trace="$openvm_fv_dir/.lake/build/bin/vm_dump_proof.trace"

test -d "$vendored_dir"
test -f "$vm_trace"
test -f "$dump_trace"

scratch_dir="$(mktemp -d)"
trap 'rm -rf -- "$scratch_dir"' EXIT

for trace in "$vm_trace" "$dump_trace"; do
  jq -r '.inputs | .. | arrays | .[0]? | strings | select(endswith(".c.o.export"))' "$trace"
done | sort -u > "$scratch_dir/generated-objects"
test -s "$scratch_dir/generated-objects"

while IFS= read -r object; do
  generated_c="${object%.o.export}"
  relative_path="${generated_c##*/.lake/build/ir/}"
  test "$relative_path" != "$generated_c"
  test -f "$generated_c"
  test -f "$vendored_dir/$relative_path"
  cmp "$generated_c" "$vendored_dir/$relative_path"
  printf '%s\n' "$relative_path"
done < "$scratch_dir/generated-objects" | sort -u > "$scratch_dir/generated-files"

find "$vendored_dir" -type f -name '*.c' \
  | sed "s#^$vendored_dir/##" \
  | sort > "$scratch_dir/vendored-files"

diff -u "$scratch_dir/generated-files" "$scratch_dir/vendored-files"

echo "Vendored C matches the openvm-fv vm_verify and vm_dump_proof link closures."
