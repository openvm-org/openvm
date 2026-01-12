# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenVM is a modular zkVM (zero-knowledge virtual machine) framework designed for customization and extensibility. It features a no-CPU architecture where functionality is added via extensions rather than a central processing unit. The framework uses STARK proofs built on Plonky3.

## Build and Development Commands

```bash
# Build the workspace
cargo build

# Run all tests with nextest (recommended)
cargo nextest run

# Run tests for a specific crate
cargo nextest run -p openvm-circuit

# Run a single test
cargo nextest run -p openvm-circuit test_name

# Faster test iterations (CI uses these env vars)
OPENVM_FAST_TEST=1 OPENVM_SKIP_DEBUG=1 cargo nextest run

# Lint (requires nightly for fmt)
cargo +nightly fmt --all -- --check
cargo clippy --all-targets --all --tests -- -D warnings

# Build docs
cargo doc --workspace --exclude "openvm-benchmarks" --exclude "*-tests"
```

### CLI Usage

```bash
# Run CLI from source
cargo run --bin cargo-openvm -- --help

# Install CLI locally
cd crates/cli && cargo install --force --locked --path .

# Then use as
cargo openvm
```

### Profiles

- `release`: Fastest runtime (thin LTO, optimized)
- `fast`: Quick iteration with O1 optimization and incremental compilation
- `profiling`: Release with debug symbols for flamegraphs
- `maxperf`: Maximum performance (fat LTO, single codegen unit - slow compile)

## Architecture Overview

### Extension System (Three-Trait Pattern)

Each VM extension implements three traits:
1. **`VmExecutionExtension`**: Runtime instruction handling via executors
2. **`VmCircuitExtension`**: AIR constraints for the zkVM circuit
3. **`VmProverExtension`**: Trace generation for specific prover backends (CPU/GPU)

Extensions are composed via the `#[derive(VmConfig)]` macro which handles executor enum generation and trait implementations.

### Execution Modes

The VM has three execution modes with corresponding executor traits:
- **Pure Execution** (`Executor<F>`): Fast execution without overhead, uses precomputed function pointers
- **Metered Execution** (`MeteredExecutor<F>`): Tracks trace heights for segmentation/continuations
- **Preflight Execution** (`PreflightExecutor<F, RA>`): Generates execution records for trace generation

### Integration API (Adapter/Core Pattern)

Most chips use the adapter/core separation:
- **Adapter**: Handles system interactions (memory bus, program bus, execution bus)
- **Core**: Implements instruction-specific arithmetic constraints

Components: `VmAdapterAir`, `VmCoreAir`, `VmAirWrapper`, `VmChipWrapper`

Naming convention: For functionality `Foo`, create `FooExecutor`, `FooFiller`, `FooCoreAir`, then typedef `FooChip` and `FooAir`.

### Directory Structure

- `crates/vm/`: Core VM circuit framework and system chips
- `crates/sdk/`: Developer SDK for proving programs
- `crates/cli/`: `cargo openvm` CLI tool
- `crates/toolchain/`: Guest program toolchain (build, transpiler, platform runtime)
- `crates/continuations/`: Aggregation programs for continuations
- `crates/circuits/`: Circuit primitives (mod-builder, poseidon2-air, primitives)
- `extensions/`: All non-system functionality (rv32im, native, keccak256, sha256, bigint, algebra, ecc, pairing)
- `guest-libs/`: Libraries for use in guest programs
- `examples/`: Example guest programs

### Extension Structure

Each extension typically has:
- `circuit/`: Circuit extension implementation
- `transpiler/`: RISC-V to OpenVM instruction transpilation
- `guest/`: Guest-side library with intrinsics
- `tests/`: Integration tests

## Key Concepts

- **Instruction = Opcode + Operands**: Each opcode maps to a specific executor
- **Chips**: Handle opcode groups, contain AIR constraints and trace generation logic
- **Buses**: Virtual communication channels (memory, program, execution) for AIR interactions
- **Phantom Sub-Instructions**: Instructions affecting runtime but with no AIR constraints (except PC advancement)

## IDE Setup

For `rustfmt` with nightly options, add to `.vscode/settings.json`:
```json
{
  "rust-analyzer.rustfmt.extraArgs": ["+nightly"]
}
```

For CUDA development, add `"rust-analyzer.cargo.features": ["cuda"]` and run `scripts/generate_clangd.sh` for clangd linting.

## Testing Notes

- Heavy tests use `--profile heavy` (2 parallel tests)
- Tests matching `~persistent` get 16x thread weight
- Nextest is the preferred test runner
