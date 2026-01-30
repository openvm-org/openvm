#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/verify_pre_hash.sh <base_ref> <tagged_ref>
# Runs keygen on examples and benchmarks for both refs and compares vk pre_hash.

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <base_ref> <tagged_ref>" >&2
  exit 1
fi

BASE_REF="$1"
TAGGED_REF="$2"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp $ROOT/crates/sdk/src/bin/vk_dump.rs /tmp/vk_dump.rs

get_pre_hash() {
  # Copy the script for branches that don't have it. We need to build from source since the serialization of AppProvingKey has small differences.
  cargo run -p openvm-sdk --bin vk_dump --release -- "$1" | awk '/^pre_hash:/{print $0}'
}

run_for_ref() {
  local ref="$1"
  local outdir="$2"
  # Use detached checkout to avoid branch-lock issues with other worktrees.
  git switch --detach "$ref"
  cp /tmp/vk_dump.rs $ROOT/crates/sdk/src/bin/vk_dump.rs
  cargo install --force --locked --path crates/cli

  mkdir -p "$outdir/examples" "$outdir/benchmarks"

  for example in "$ROOT"/examples/*/; do
    [ -f "${example}/Cargo.toml" ] || continue
    local name; name="$(basename "$example")"
    echo "Building/keygen example: $name"
    (cd "$example" && cargo openvm keygen --output-dir "$outdir/examples/$name")
    # Capture pre_hash with the vk_dump built from the same ref to avoid cross-branch
    # deserialization issues when fields change.
    if [ -f "$outdir/examples/$name/app.vk" ]; then
      get_pre_hash "$outdir/examples/$name/app.vk" > "$outdir/examples/$name/pre_hash.txt"
    fi
  done

  for bench in "$ROOT"/benchmarks/guest/*/; do
    [ -f "${bench}/Cargo.toml" ] || continue
    local name; name="$(basename "$bench")"
    echo "Building/keygen benchmark: $name"
    (cd "$bench" && cargo openvm keygen --output-dir "$outdir/benchmarks/$name")
    if [ -f "$outdir/benchmarks/$name/app.vk" ]; then
      get_pre_hash "$outdir/benchmarks/$name/app.vk" > "$outdir/benchmarks/$name/pre_hash.txt"
    fi
  done
  rm $ROOT/crates/sdk/src/bin/vk_dump.rs
  git reset --hard
}

compare_dir() {
  local kind="$1"
  local base_root="$2"
  local tagged_root="$3"
  local failed=0
  for path in "$base_root/$kind"/*/; do
    [ -d "$path" ] || continue
    local name; name="$(basename "$path")"
    local base_vk="$base_root/$kind/$name/app.vk"
    local tagged_vk="$tagged_root/$kind/$name/app.vk"
    if [ ! -f "$tagged_vk" ]; then
      echo "❌ missing tagged vk for $kind/$name"
      failed=1
      continue
    fi
    local base_hash tagged_hash base_hash_file tagged_hash_file
    base_hash_file="$base_root/$kind/$name/pre_hash.txt"
    tagged_hash_file="$tagged_root/$kind/$name/pre_hash.txt"
    if [ ! -f "$base_hash_file" ] || [ ! -f "$tagged_hash_file" ]; then
      echo "❌ missing pre_hash record for $kind/$name"
      failed=1
      continue
    fi
    base_hash="$(cat "$base_hash_file")"
    tagged_hash="$(cat "$tagged_hash_file")"
    if [ "$base_hash" = "$tagged_hash" ]; then
      echo "✅ $kind/$name pre_hash matches ($base_hash)"
    else
      echo "❌ $kind/$name pre_hash differs"
      echo "    base  : $base_hash"
      echo "    tagged: $tagged_hash"
      failed=1
    fi
  done
  return $failed
}

BASE_OUT="$ROOT/base-outputs"
TAGGED_OUT="$ROOT/tagged-outputs"

run_for_ref "$BASE_REF" "$BASE_OUT"
run_for_ref "$TAGGED_REF" "$TAGGED_OUT"

# Switch back to original HEAD at end.
git switch --detach "$TAGGED_REF" >/dev/null 2>&1 || true

echo "Comparing pre_hash values..."
compare_dir examples "$BASE_OUT" "$TAGGED_OUT"
compare_dir benchmarks "$BASE_OUT" "$TAGGED_OUT"
