#!/usr/bin/env bash
# Generates `crates/toolchain/openvm/src/memcpy.s` and `memset.s` from
# musl-libc sources, deterministically.
#
# Usage:
#   scripts/generate_libc_intrinsics.sh                                  # use defaults (v1.2.6)
#   scripts/generate_libc_intrinsics.sh --musl-ref v1.2.5                # pin to a different release
#   scripts/generate_libc_intrinsics.sh --musl-ref master                # track upstream master
#   scripts/generate_libc_intrinsics.sh --musl-ref <40-char SHA>         # pin to a specific commit
#   scripts/generate_libc_intrinsics.sh --clang clang-14                 # pin a specific clang
#
# Defaults:
#   --musl-ref     v1.2.6 (the latest musl-libc release as of writing)
#   --clang        clang
#
# `--musl-ref` accepts any cgit-resolvable ref — a release tag (`v1.2.x`),
# a branch name, or a commit SHA. Tags and branches are resolved to a
# concrete SHA via `git ls-remote` before fetching, and the resolved SHA
# is what gets recorded in the generated file's header (so the output
# stays deterministic even when tracking a moving ref).
#
# The actual clang version used is recorded in each output `.s` via the
# trailing `.ident` directive, so the checked-in files self-describe the
# toolchain that produced them.
#
# The script:
#   1. Fetches musl's `src/string/{memcpy,memset}.c` at the pinned commit,
#      plus the matching `COPYRIGHT` file for license attribution.
#   2. Strips `#include` lines and prepends minimal inline typedefs so each
#      source can be compiled with `-nostdlib -fno-builtin`.
#   3. Compiles each file with:
#         clang -target riscv64 -march=rv64im -O3 -S \
#               -nostdlib -fno-builtin -funroll-loops
#   4. Renames LLVM's per-translation-unit local labels (`.LBB0_*`,
#      `.LJTI0_*`, `.Lfunc_end0`) to function-prefixed variants so they do
#      not collide with the same auto-generated names in user code under
#      fat LTO (`error: symbol '.LJTI0_0' is already defined`).
#   5. Writes the final `.s` with a header that pins the musl commit, the
#      reproduction recipe, and embeds the musl COPYRIGHT inline.
#
# To verify the working tree is in sync with this script:
#   scripts/generate_libc_intrinsics.sh && git diff --exit-code \
#     crates/toolchain/openvm/src/memcpy.s crates/toolchain/openvm/src/memset.s

set -euo pipefail

MUSL_REF="v1.2.6"
CLANG="clang"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --musl-ref)
      [[ $# -ge 2 ]] || { echo "error: --musl-ref needs a value" >&2; exit 2; }
      MUSL_REF="$2"; shift 2 ;;
    --clang)
      [[ $# -ge 2 ]] || { echo "error: --clang needs a value" >&2; exit 2; }
      CLANG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done

# Resolve the user-supplied ref to a concrete commit SHA so the embedded
# header is reproducible even when MUSL_REF is a tag or branch name. If
# the input already looks like a 40-char SHA, skip the lookup.
if [[ "${MUSL_REF}" =~ ^[0-9a-f]{40}$ ]]; then
  MUSL_COMMIT="${MUSL_REF}"
else
  echo "gen: resolving musl ref '${MUSL_REF}' via git ls-remote"
  # `refs/tags/${ref}^{}` is the peeled form — for an annotated tag this returns
  # the underlying commit SHA, not the tag object SHA. Prefer it when present
  # so we always embed a commit SHA in the file header.
  ls_remote_out="$(git ls-remote https://git.musl-libc.org/git/musl \
                   "refs/tags/${MUSL_REF}^{}" \
                   "refs/tags/${MUSL_REF}" \
                   "refs/heads/${MUSL_REF}" \
                   "${MUSL_REF}" 2>/dev/null)"
  resolved="$(echo "${ls_remote_out}" | awk '$2 ~ /\^\{\}$/ { print $1; exit }')"
  if [[ -z "${resolved}" ]]; then
    resolved="$(echo "${ls_remote_out}" | awk 'NR==1 { print $1 }')"
  fi
  if [[ -z "${resolved}" ]]; then
    echo "error: could not resolve musl ref '${MUSL_REF}' (tried tag, branch, and direct lookup)" >&2
    exit 1
  fi
  MUSL_COMMIT="${resolved}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${REPO_ROOT}/crates/toolchain/openvm/src"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

if [[ "${MUSL_REF}" == "${MUSL_COMMIT}" ]]; then
  echo "gen: musl commit ${MUSL_COMMIT}"
else
  echo "gen: musl commit ${MUSL_COMMIT} (from --musl-ref ${MUSL_REF})"
fi
echo "gen: clang -> $("${CLANG}" --version | head -1)"
echo "gen: writing to ${OUTDIR}"

# Fetch musl COPYRIGHT once (shared between both files).
curl -fsSL \
  "https://git.musl-libc.org/cgit/musl/plain/COPYRIGHT?id=${MUSL_COMMIT}" \
  -o "${TMP}/COPYRIGHT"

# Per-function inline typedefs: musl headers (string.h / stdint.h / endian.h)
# pull in too much for a freestanding build. We provide the minimal set each
# source needs.
TYPEDEFS_MEMCPY=$(cat <<'EOF'
typedef unsigned long size_t;
typedef unsigned int uint32_t;
typedef unsigned long uintptr_t;
EOF
)

TYPEDEFS_MEMSET=$(cat <<'EOF'
typedef unsigned long size_t;
typedef unsigned int uint32_t;
typedef unsigned long uintptr_t;
typedef unsigned long long uint64_t;
EOF
)

regen_one() {
  local fn="$1"        # memcpy | memset
  local typedefs="$2"  # block of typedef lines

  local src_url="https://git.musl-libc.org/cgit/musl/plain/src/string/${fn}.c?id=${MUSL_COMMIT}"
  local c="${TMP}/${fn}_no_includes.c"
  local s="${TMP}/${fn}.s"

  echo "gen: fetching ${fn}.c"
  # Drop musl's `#include` lines and prepend our minimal typedefs in their place.
  {
    printf '// Derived from musl-libc commit %s (src/string/%s.c),\n' "${MUSL_COMMIT}" "${fn}"
    printf '// with #include lines replaced by inline typedefs so the file can\n'
    printf '// be compiled freestanding (-nostdlib -fno-builtin).\n\n'
    printf '%s\n\n' "${typedefs}"
    curl -fsSL "${src_url}" | sed '/^#include /d'
  } > "${c}"

  echo "gen: compiling ${fn}.c -> ${fn}.s (riscv64im, O3, -funroll-loops)"
  "${CLANG}" -target riscv64 -march=rv64im -O3 -S \
    -nostdlib -fno-builtin -funroll-loops \
    "${c}" -o "${s}"

  echo "gen: renaming local labels (.LBB0_/.LJTI0_/.Lfunc_end0 -> ${fn}-prefixed), normalizing .file/.ident"
  # All three label classes share the function-index "0_*" suffix shape. Under
  # fat LTO multiple translation units can emit colliding ".LBB0_*",
  # ".LJTI0_*", ".Lfunc_end0" names; prefix every local label that contains
  # "0_" (or "_end0") with the function name to make it unique.
  #
  # We also normalize two clang-specific bits to keep the output reproducible
  # across clang distributions:
  #   - `.file "..._no_includes.c"`: clang embeds the on-disk filename of the
  #     scratch source. Rewrite to a stable name.
  #   - `.ident "..."`: clang stamps its own vendor + version string. Drop it
  #     so output bytes depend only on the musl source and the LLVM codegen,
  #     not on the clang distribution (Homebrew vs Debian vs Apple, etc.).
  local tab; tab="$(printf '\t')"
  sed \
    -e "s|\.LBB0_|.LBB${fn}0_|g" \
    -e "s|\.LJTI0_|.LJTI${fn}0_|g" \
    -e "s|\.Lfunc_end0|.L${fn}func_end0|g" \
    -e "s|\.file[[:space:]]\{1,\}\"[^\"]*_no_includes\.c\"|.file${tab}\"musl_${fn}.c\"|" \
    -e "/^[[:space:]]*\.ident[[:space:]]/d" \
    "${s}" > "${s}.new" && mv "${s}.new" "${s}"

  echo "gen: writing ${OUTDIR}/${fn}.s"
  {
    printf '// This is musl-libc commit %s:\n' "${MUSL_COMMIT}"
    printf '//\n'
    printf '// src/string/%s.c\n' "${fn}"
    printf '//\n'
    printf '// This was compiled into assembly with:\n'
    printf '//\n'
    printf '//     clang -target riscv64 -march=rv64im -O3 -S %s.c -nostdlib -fno-builtin -funroll-loops\n' "${fn}"
    printf '//\n'
    printf '// and local labels (.LBB0_*, .LJTI0_*, .Lfunc_end0) were prefixed\n'
    printf '// with the function name so they do not collide with the same\n'
    printf '// auto-generated names in user code under fat LTO.\n'
    printf '//\n'
    printf '// Regenerated by scripts/generate_libc_intrinsics.sh.\n'
    printf '//\n'
    # Prepend "// " to non-blank COPYRIGHT lines and just "//" to blank ones,
    # so the result has no trailing whitespace.
    awk '{ if (length($0)) print "// " $0; else print "//" }' "${TMP}/COPYRIGHT"
    printf '\n'
    cat "${s}"
  } > "${OUTDIR}/${fn}.s"
}

regen_one memcpy "${TYPEDEFS_MEMCPY}"
regen_one memset "${TYPEDEFS_MEMSET}"

echo "gen: done"
