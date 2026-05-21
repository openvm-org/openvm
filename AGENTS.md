# AGENTS.md

## Agent Quickstart (Read First)

- Prefer targeted commands (`-p <crate>` or `cd <crate-dir>`) over workspace-wide runs; full workspace builds/tests are slow.
- Use the CI-like build profile: `cargo build --profile fast -p <crate>` (note: `--cargo-profile` is a **nextest** flag, not a Cargo build flag).
- Formatting requires nightly rustfmt: `cargo +nightly fmt --all` (stable `cargo fmt` will fail due to unstable options in `rustfmt.toml`).
- Run tests with nextest when available: `cargo nextest run --cargo-profile=fast -p <crate>`; fallback: `cargo test -p <crate>`.
- Speed knob for local runs: `OPENVM_SKIP_DEBUG=1` (see below).
- After major code edits, run `./scripts/pre-push.sh` to check formatting, linting, and tests on changed crates before pushing.

## Project Overview

OpenVM is a modular zkVM (zero-knowledge virtual machine) framework built on STARK proofs. It features a no-CPU architecture where all functionality (including RISC-V support) is implemented via composable extensions. The proof system is built on top of Plonky3 and the [stark-backend](https://github.com/openvm-org/stark-backend).

## Build & Development Commands

### Rust Toolchain

- Rust 1.91.1 (stable), specified in `rust-toolchain.toml`
- Nightly is only needed for:
  - `rustfmt` (unstable formatting options)
  - some workflows that build `cargo-openvm` with experimental features (pinned nightly: `nightly-2026-01-18`)
- Guest program compilation uses the prebuilt `openvm-nightly-2026-01-18` toolchain from the [openvm-org/rust](https://github.com/openvm-org/rust) fork, installed via `cargo openvm toolchain install` (or `ci/install-openvm-toolchain.sh` in CI). The fork's tarball ships `std`/`core`/`alloc`/`panic_abort` rlibs for `riscv64im-unknown-openvm-elf`.

### Building

```bash
cargo build -p <crate>                          # debug build (targeted)
cargo build --profile fast -p <crate>           # optimized dev build (opt-level=1, incremental, good for testing)
cargo build --release -p <crate>                # release build
```

### Formatting & Linting

```bash
cargo +nightly fmt --all                    # format (nightly required for unstable options)
cargo +nightly fmt --all -- --check         # check formatting
cargo clippy -p openvm-circuit --all-targets --tests -- -D warnings   # lint (targeted)
cargo clippy --workspace --all-targets --tests -- -D warnings         # lint (slower)
cargo shear                                 # check for unused dependencies (install via: cargo install cargo-shear)
```

Formatting uses unstable options: `group_imports = "StdExternalCrate"`, `imports_granularity = "Crate"`. Configure IDE with `rust-analyzer.rustfmt.extraArgs: ["+nightly"]`.

### Testing

Tests use [cargo-nextest](https://nexte.st/). The `fast` cargo profile is used in CI for test runs (optimized dev build).
Install with `cargo install cargo-nextest`.
Note: in nextest, `--cargo-profile` selects the Cargo build profile; `--profile` selects the nextest runner profile (e.g. `heavy`).

**Do not** run all workspace tests at once unless doing a final integration test. Build and test only specific crates during development.

```bash
# Run tests for a specific crate (most common pattern)
cargo nextest run --cargo-profile=fast -p openvm-circuit       # VM crate tests
cargo nextest run --cargo-profile=fast -p openvm-riscv-circuit # extension circuit tests

# If nextest isn't installed, fall back to cargo test
cargo test -p openvm-circuit

# Run a single test by name
cargo nextest run --cargo-profile=fast -p openvm-circuit -- test_name

# Run tests in a working directory (as CI does)
cd extensions/riscv/circuit && cargo nextest run --cargo-profile=fast

# Integration tests for extensions (requires the openvm rust toolchain)
cargo openvm toolchain install
cd extensions/riscv/tests && cargo nextest run --cargo-profile=fast --profile=heavy

# Run with parallelism (used in CI)
cargo nextest run --cargo-profile=fast --features parallel
```

### Environment Variables for Tests

- `OPENVM_SKIP_DEBUG=1`: Skips debug-mode constraint checking in `air_test` (faster CI runs)
- `OPENVM_RUST_TOOLCHAIN`: Override the rustup toolchain name used by `cargo openvm build`. Default is the `openvm-<rustup-name>` compiled into `cargo-openvm`. Set this to swap in a custom rustc fork.
- `OPENVM_RUSTC_TARGET`: Override the rustc target triple used by `cargo openvm build`. Default is `riscv64im-unknown-openvm-elf`, the tier-3 target defined in the [openvm-org/rust](https://github.com/openvm-org/rust) fork. Any rv64im-flavored target works, but the chosen toolchain must either ship prebuilt target rlibs or have the `rust-src` component installed.

### Nextest Profiles

- Default profile: standard parallel execution
- `heavy` profile: `--test-threads=2`, used for integration tests that are memory-intensive

## Architecture

### Key Crate Hierarchy

- `crates/vm` (`openvm-circuit`): VM circuit framework, system chips, trait definitions
- `crates/sdk` (`openvm-sdk`): Developer SDK, proof aggregation, on-chain verifier generation
- `crates/cli` (`cargo-openvm`): CLI for compile/execute/prove
- `crates/toolchain/`: Transpiler, instructions, platform runtime, build tools
- `crates/circuits/primitives`: Primitive chips/sub-chips reusable across circuits
- `crates/circuits/mod-builder`: Modular arithmetic circuit builder
- `crates/continuations`: Aggregation programs for multi-segment proving

The main VM crate with architecture traits and system implementation is `openvm-circuit` in `crates/vm/`.
See `docs/repo/layout.md` for full crate layout of the project.

### Core Concepts

The VM has **no CPU**. All instruction handling (including base RISC-V) is provided by **extensions**. Each extension has up to four components:

| Component    | Purpose                                                       | Example path                    |
| ------------ | ------------------------------------------------------------- | ------------------------------- |
| `circuit`    | AIR constraints + chips for proving                           | `extensions/riscv/circuit/`    |
| `transpiler` | Converts RISC-V ELF instructions to OpenVM instructions       | `extensions/riscv/transpiler/` |
| `guest`      | Rust library with intrinsics for guest programs               | `extensions/riscv/guest/`      |
| `tests`      | Integration tests with guest programs in `programs/examples/` | `extensions/riscv/tests/`      |

### Extension Framework (Three Traits)

Each extension implements three traits, each independent and for a different phase:

1. **`VmExecutionExtension`** - registers executors for new opcodes (runtime execution)
2. **`VmCircuitExtension`** - registers AIRs (constraint system, determines verifying key)
3. **`VmProverExtension`** - registers chips for trace generation (can vary by prover backend: CPU vs GPU)

Extensions are composed into a `VmConfig` using the `#[derive(VmConfig)]` macro:

```rust
#[derive(VmConfig)]
pub struct MyConfig {
    #[config]
    pub rv64im: Rv64ImConfig,  // existing config (implements VmConfig)
    #[extension]
    pub my_ext: MyExtension,   // new extension
}
```

### Chip Architecture (Adapter + Core pattern)

Most chips follow the adapter/core split via the integration API:

- **AdapterAir** handles system interactions (memory bus, execution bus, program bus)
- **CoreAir** implements instruction-specific arithmetic constraints
- Combined via `VmAirWrapper<AdapterAir, CoreAir>` and `VmChipWrapper<F, Filler>`

Naming convention: `FooExecutor`, `FooFiller`, `FooCoreAir` for a chip named `Foo`.

### Three Execution Modes

1. **Pure execution** (`Executor<F>`): Runs program, returns final state. Uses precomputed function pointers.
2. **Metered execution** (`MeteredExecutor<F>`): Tracks per-chip trace heights for segmentation.
3. **Preflight execution** (`PreflightExecutor<F, RA>`): Generates execution records for trace generation.

### Guest Programs

Guest programs (run inside the VM) use `#![no_main]` / `#![no_std]` with `openvm::entry!(main)`:

```rust
#![cfg_attr(all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

pub fn main() { /* ... */ }
```

Guest programs are compiled to RISC-V ELF, then transpiled to OpenVM instructions. They live in `programs/examples/` within test crates.

### Cfg conventions

Guest code is gated on **two predicates**:

1. **`cfg(any(openvm_intrinsics, target_os = "openvm"))`** — "we are building for the openvm runtime". True when either:
   - the target is `riscv64im-unknown-openvm-elf` (the openvm-fork's built-in target spec sets `target_os = "openvm"`), or
   - the `openvm_intrinsics` cfg is set explicitly — `cargo openvm build` sets this rustflag; an opt-in path for using openvm runtime on a generic rv64im target (e.g. `riscv64im-unknown-none-elf`) would also set it.

   All openvm-specific codegen — custom RISC-V opcodes via inline asm, the custom `_start`, memset/memcpy, the `entry!` wrapper that exposes `main` — gates on this. Host `cargo check`/`clippy` matches neither cfg and excludes guest code cleanly.

2. **`cfg(feature = "std")`** — std is being linked. Orthogonal to the runtime predicate. The `openvm` crate's `std` feature forwards to std-linking deps.

Combined gates appear in a handful of places:

- `all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm"))` — gates `#![no_main]` in guest programs and the `entry!` wrapper definition (the variant that emits `#[no_mangle] fn main()` so the custom `_start` can call it). In std mode, Rust's runtime emits `main`; we must not duplicate it.
- `all(feature = "std", target_os = "openvm")` — gates the `pal_abi` module. The std PAL only exists for `target_os = "openvm"` (it ships in the openvm-fork's `library/std/src/sys/pal/openvm/`); a generic `target_os = "none"` build cannot link std regardless of `openvm_intrinsics`, so `pal_abi` would be dead code there.

**Cargo.toml target-specific dependency tables** can only see target cfgs (Cargo doesn't pass `--cfg` flags to manifest evaluation), so use the target-only form: `[target.'cfg(all(target_arch = "riscv64", any(target_os = "none", target_os = "openvm")))'.dependencies]` for guest-only deps, `[target.'cfg(not(any(target_os = "none", target_os = "openvm")))'.dependencies]` for host-only deps. Including the `target_os = "none"` term lets a future generic rv64im opt-in (via `openvm_intrinsics`) still pick up the right deps without further Cargo.toml edits.

**check-cfg** — guest crates need both names declared so stable-rustc host checks don't warn on them:

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(openvm_intrinsics)', 'cfg(target_os, values("openvm"))'] }
```

`cargo openvm init` adds this automatically.

### Integration Test Pattern

```rust
let elf = build_example_program_at_path(get_programs_dir!(), "program_name")?;
let exe = VmExe::from_elf(elf, Transpiler::<F>::default()
    .with_extension(Rv64ITranspilerExtension)
    .with_extension(Rv64MTranspilerExtension)
    .with_extension(Rv64IoTranspilerExtension))?;
let config = Rv64ImConfig::default();
air_test(Rv64ImCpuBuilder, config, exe);
```

### Proof Field

The primary field is **BabyBear** (31-bit prime field from Plonky3). Final on-chain verification uses BN254 via Halo2.

### Versioning

OpenVM uses semver naming but with ZK-specific semantics: patch versions preserve verifying key (`MultiStarkVerifyingKey`) compatibility. See `VERSIONING.md`.

## CUDA/GPU Support

CUDA is behind the `cuda` feature flag, disabled by default. Feature-gate non-CUDA-compatible code with `#[cfg(not(feature = "cuda"))]`. GPU prover extensions implement `VmProverExtension` separately (e.g., `Rv64ImGpuProverExt`).

## File Structure

AIR constraints should always be in a separate file (`air.rs`) from execution or trace generation logic. Trace generation and CUDA prover code should be kept in files named `trace.rs`, `cuda_abi.rs` or folders named `tracegen`, `cuda`. Execution logic should be in files named `execution.rs` or `preflight.rs`.
