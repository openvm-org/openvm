#!/usr/bin/env bash
# Local pre-CI check: runs fmt, clippy, and tests only on crates changed vs a target branch.
# Usage: ./scripts/pre-push.sh [target-branch]  (default: develop-v2.0.0-beta)
#
# To install as a git pre-push hook:
#
#   ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
#
# Or, if you use a git worktree or want a wrapper that passes the remote branch:
#
#   cat > .git/hooks/pre-push << 'HOOK'
#   #!/usr/bin/env bash
#   exec "$(git rev-parse --show-toplevel)/scripts/pre-push.sh"
#   HOOK
#   chmod +x .git/hooks/pre-push
#
# To bypass the hook for a single push:  git push --no-verify
set -euo pipefail

TARGET="${1:-develop-v2.0.0-beta}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Match CI environment variables for faster test runs
export OPENVM_SKIP_DEBUG="${OPENVM_SKIP_DEBUG:-1}"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC} $1"; }
fail() { echo -e "${RED}FAIL${NC} $1"; }
info() { echo -e "${BOLD}==> $1${NC}"; }
warn() { echo -e "${YELLOW}WARN${NC} $1"; }

# --- GPU detection ---
EXTRA_FEATURES=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    info "NVIDIA GPU detected — will enable cuda features where supported"
    HAS_GPU=1
else
    info "No NVIDIA GPU detected — CPU-only mode"
    HAS_GPU=0
fi

# --- Find merge base and changed files ---
MERGE_BASE="$(git merge-base HEAD "$TARGET")"
CHANGED_FILES="$(git diff --name-only "$MERGE_BASE" HEAD)"

if [ -z "$CHANGED_FILES" ]; then
    info "No files changed vs $TARGET — nothing to check."
    exit 0
fi

echo "$CHANGED_FILES" | head -20
TOTAL=$(echo "$CHANGED_FILES" | wc -l)
if [ "$TOTAL" -gt 20 ]; then
    echo "  ... and $((TOTAL - 20)) more files"
fi

# --- Map changed files to crate directories ---
# Use simple arrays instead of associative arrays for Bash 3.x compatibility (macOS)
CRATE_DIR_LIST=""
while IFS= read -r file; do
    dir="$file"
    while true; do
        dir="$(dirname "$dir")"
        if [ "$dir" = "." ] || [ "$dir" = "/" ]; then
            break
        fi
        toml="$dir/Cargo.toml"
        if [ -f "$toml" ] && grep -q '^\[package\]' "$toml"; then
            # Deduplicate: only add if not already present
            case "$CRATE_DIR_LIST" in
                *"|$dir|"*) ;;
                *) CRATE_DIR_LIST="${CRATE_DIR_LIST}|$dir|" ;;
            esac
            break
        fi
    done
done <<< "$CHANGED_FILES"

if [ -z "$CRATE_DIR_LIST" ]; then
    info "No Rust crates changed (only non-crate files modified) — nothing to check."
    exit 0
fi

# --- Extract crate names and filter ---
# Parallel arrays: CRATE_NAMES[i] and CRATE_DIRS[i]
CRATE_NAMES=()
CRATE_DIRS=()
HEAVY_CRATE_LIST="" # "|name1|name2|" for membership checks

# Parse unique dirs from CRATE_DIR_LIST
IFS='|' read -ra _raw_dirs <<< "$CRATE_DIR_LIST"
for dir in "${_raw_dirs[@]}"; do
    [ -z "$dir" ] && continue
    # Skip benchmarks and guest programs (need nightly + rust-src)
    if [[ "$dir" == benchmarks/* ]] || [[ "$dir" == */programs/* ]] || [[ "$dir" == */programs ]]; then
        warn "Skipping $dir (benchmark or guest program)"
        continue
    fi
    name=$(grep '^name\s*=' "$dir/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [ -n "$name" ]; then
        CRATE_NAMES+=("$name")
        CRATE_DIRS+=("$dir")
        # Integration test crates and guest-lib crates get heavy profile (CI runs these with limited threads)
        if [[ "$dir" == extensions/*/tests ]] || [[ "$dir" == guest-libs/*/tests ]] \
            || [[ "$dir" == guest-libs/* ]]; then
            HEAVY_CRATE_LIST="${HEAVY_CRATE_LIST}|$name|"
        fi
    fi
done

if [ ${#CRATE_NAMES[@]} -eq 0 ]; then
    info "All changed crates were skipped — nothing to check."
    exit 0
fi

# Helper: check if a crate is heavy
is_heavy() { case "$HEAVY_CRATE_LIST" in *"|$1|"*) return 0 ;; *) return 1 ;; esac; }

info "Changed crates (${#CRATE_NAMES[@]}):"
for i in "${!CRATE_NAMES[@]}"; do
    extra=""
    is_heavy "${CRATE_NAMES[$i]}" && extra=" (heavy)"
    echo "  ${CRATE_NAMES[$i]}  [${CRATE_DIRS[$i]}]$extra"
done

# --- Helper: compute features for a crate ---
crate_features() {
    local dir="$1"
    local toml="$dir/Cargo.toml"
    local feats=()
    # Check for each feature in the crate's Cargo.toml
    for f in parallel cuda touchemall; do
        if [ "$f" = "cuda" ] || [ "$f" = "touchemall" ]; then
            [ "$HAS_GPU" -eq 0 ] && continue
        fi
        if grep -qE "^${f}\s*=" "$toml" 2>/dev/null; then
            feats+=("$f")
        fi
    done
    if [ ${#feats[@]} -gt 0 ]; then
        local IFS=','
        echo "${feats[*]}"
    fi
}

ERRORS=0

# --- Step 1: Format check ---
info "Step 1/3: cargo +nightly fmt --all -- --check"
if cargo +nightly fmt --all -- --check; then
    pass "formatting"
else
    fail "formatting"
    ERRORS=$((ERRORS + 1))
fi

# --- Step 2: Clippy on changed crates ---
info "Step 2/3: clippy on changed crates"
for i in "${!CRATE_NAMES[@]}"; do
    name="${CRATE_NAMES[$i]}"
    dir="${CRATE_DIRS[$i]}"
    feats="$(crate_features "$dir")"
    feat_args=()
    [ -n "$feats" ] && feat_args=(--features "$feats")

    echo -n "  clippy $name "
    [ -n "$feats" ] && echo -n "(+$feats) "
    if cargo clippy -p "$name" --all-targets --tests "${feat_args[@]}" -- -D warnings; then
        pass ""
    else
        fail ""
        ERRORS=$((ERRORS + 1))
    fi
done

# --- Step 3: Tests on changed crates ---
info "Step 3/3: tests on changed crates"

# Detect test runner
if command -v cargo-nextest &>/dev/null; then
    USE_NEXTEST=1
else
    warn "cargo-nextest not found — falling back to cargo test"
    USE_NEXTEST=0
fi

for i in "${!CRATE_NAMES[@]}"; do
    name="${CRATE_NAMES[$i]}"
    dir="${CRATE_DIRS[$i]}"
    feats="$(crate_features "$dir")"
    feat_args=()
    [ -n "$feats" ] && feat_args=(--features "$feats")

    echo -n "  test $name "
    [ -n "$feats" ] && echo -n "(+$feats) "

    if [ "$USE_NEXTEST" -eq 1 ]; then
        profile_args=()
        if is_heavy "$name"; then
            profile_args=(--profile=heavy)
            echo -n "(heavy) "
        fi
        if cargo nextest run --cargo-profile=fast -p "$name" "${feat_args[@]}" "${profile_args[@]}"; then
            pass ""
        else
            fail ""
            ERRORS=$((ERRORS + 1))
        fi
    else
        if cargo test --profile fast -p "$name" "${feat_args[@]}"; then
            pass ""
        else
            fail ""
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

# --- Summary ---
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}$ERRORS check(s) failed.${NC}"
    exit 1
fi
