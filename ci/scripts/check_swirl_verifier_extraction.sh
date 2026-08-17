#!/usr/bin/env bash

set -euo pipefail

openvm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
swirl_dir="${1:?usage: check_swirl_verifier_extraction.sh <swirl-rbr-fv checkout>}"
swirl_dir="$(cd "$swirl_dir" && pwd)"

# Ask Cargo for the exact stark-backend source selected by this lockfile. The
# precise Git revision is part of `source`; `cargo pkgid` only reports the
# package identity and omits that revision.
locked_source="$(
  cd "$openvm_root"
  cargo metadata --locked --format-version 1 | jq -er '
    [.packages[] | select(.name == "openvm-stark-backend") | .source]
    | if length == 1 then .[0]
      else error("expected exactly one openvm-stark-backend package")
      end
  '
)"
if [[ ! "$locked_source" =~ \#([0-9a-f]{40})$ ]]; then
  echo "Could not resolve the openvm-stark-backend revision from Cargo metadata:" >&2
  echo "$locked_source" >&2
  exit 1
fi
locked_revision="${BASH_REMATCH[1]}"

echo "Checking the Lean model against OpenVM's locked stark-backend revision $locked_revision."
"$swirl_dir/verifier-lean/extraction/verify.sh" --stark-backend-rev "$locked_revision"
