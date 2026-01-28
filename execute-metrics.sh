set -euo pipefail

ensure_riscv_gcc() {
  if command -v riscv64-unknown-elf-gcc >/dev/null 2>&1; then
    return
  fi

  echo "[execute-metrics] riscv64-unknown-elf-gcc not found; installing..."
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y gcc-riscv64-unknown-elf
    else
      apt-get update
      apt-get install -y gcc-riscv64-unknown-elf
    fi
  else
    # Don't hard-fail here: on non-Debian systems we can't auto-install, but the user may already
    # have another valid RISC-V GCC toolchain installed or may prefer to install manually.
    echo "[execute-metrics] WARNING: missing riscv64-unknown-elf-gcc and no apt-get found."
    echo "[execute-metrics] WARNING: please install a RISC-V GCC toolchain (riscv64-unknown-elf-gcc) and re-run if needed."
  fi
}

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

ensure_riscv_gcc
ensure_coremark_submodules

cargo install --path crates/cli/
OUTPUT_PATH="metrics-vanilla.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml

cargo install --path crates/cli/ --features aot
OUTPUT_PATH="metrics-aot.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml

cargo +nightly install --path crates/cli/ --features tco
OUTPUT_PATH="metrics-tco.json" cargo openvm run --manifest-path coremark-openvm/Cargo.toml
