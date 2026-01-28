#!/usr/bin/env bash

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

ensure_root_vmexe() {
  # We keep a prebuilt `VmExe` at repo root so we can skip rebuilding/transpiling
  # the guest program when collecting execution metrics.
  #
  # This uses the CLI's `--output-dir` to copy the transpiled `.vmexe` to `./`.
  if [[ ! -f "coremark-openvm.vmexe" ]]; then
    ensure_riscv_gcc
    ensure_coremark_submodules
    echo "[execute-metrics] coremark-openvm.vmexe missing; building once..."
    cargo openvm build --manifest-path coremark-openvm/Cargo.toml --output-dir .
  fi
}


cargo install --path crates/cli/
ensure_root_vmexe
OUTPUT_PATH="_coremark-metrics-vanilla.json" cargo openvm run --exe coremark-openvm.vmexe --manifest-path coremark-openvm/Cargo.toml
OUTPUT_PATH="_client-eth-metrics-vanilla.json" cargo openvm run --exe client-eth.vmexe --manifest-path client-eth/Cargo.toml --input client-eth-input.json

cargo install --path crates/cli/ --features aot
ensure_root_vmexe
OUTPUT_PATH="_coremark-metrics-aot.json" cargo openvm run --exe coremark-openvm.vmexe --manifest-path coremark-openvm/Cargo.toml
OUTPUT_PATH="_client-eth-metrics-aot.json" cargo openvm run --exe client-eth.vmexe --manifest-path client-eth/Cargo.toml --input client-eth-input.json

cargo +nightly install --path crates/cli/ --features tco
ensure_root_vmexe
OUTPUT_PATH="_coremark-metrics-tco.json" cargo openvm run --exe coremark-openvm.vmexe --manifest-path coremark-openvm/Cargo.toml
OUTPUT_PATH="_client-eth-metrics-tco.json" cargo openvm run --exe client-eth.vmexe --manifest-path client-eth/Cargo.toml --input client-eth-input.json
