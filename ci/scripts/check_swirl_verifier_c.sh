#!/usr/bin/env bash

set -euo pipefail

swirl_dir="${1:?usage: check_swirl_verifier_c.sh <swirl-rbr-fv checkout>}"
generated_dir="$swirl_dir/.lake/build/ir"
vendored_dir="crates/certified-verifier/csrc"

test -d "$generated_dir"
test -d "$vendored_dir"

scratch_dir="$(mktemp -d)"
trap 'rm -rf -- "$scratch_dir"' EXIT

for executable in swirl_verify swirl_dump_proof; do
  trace="$swirl_dir/.lake/build/bin/$executable.trace"
  test -f "$trace"
  jq -r '.inputs[] | select(.[0] == "linkObjs") | .[1][][0]' "$trace" \
    | sed -E 's#^.*/ir/##; s#\.o\.export$##'
done | sort -u > "$scratch_dir/generated-files"
find "$vendored_dir" -type f -name '*.c' \
  | sed "s#^$vendored_dir/##" \
  | sort > "$scratch_dir/vendored-files"

diff -u "$scratch_dir/generated-files" "$scratch_dir/vendored-files"

while IFS= read -r relative_path; do
  cmp "$generated_dir/$relative_path" "$vendored_dir/$relative_path"
done < "$scratch_dir/generated-files"

echo "Vendored C matches the swirl_verify and swirl_dump_proof link closures."
