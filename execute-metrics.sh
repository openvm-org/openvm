set -euo pipefail

ensure_coremark_submodules() {
  # The coremark-openvm repo is a git submodule, and it itself contains a nested
  # `coremark` submodule with the C sources needed for the build.
  #
  # If either is missing, initialize/update recursively.
  if [[ ! -f "coremark-openvm/Cargo.toml" ]] || [[ ! -f "coremark-openvm/coremark/core_main.c" ]]; then
    echo "[execute-metrics] Initializing coremark submodules..."
    git submodule sync --recursive
    git submodule update --init --recursive coremark-openvm
  fi
}

ensure_coremark_submodules

cargo install --path crates/cli/
OUTPUT_PATH="metrics-vanilla.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml

cargo install --path crates/cli/ --features aot
OUTPUT_PATH="metrics-aot.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml

cargo +nightly install --path crates/cli/ --features tco
OUTPUT_PATH="metrics-tco.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml
