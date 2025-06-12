# Changelog

All notable changes to OpenVM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.2.1 (TBD)

### Breaking Changes

#### CLI

- **New `init` command**: Added `cargo openvm init` command for creating new OpenVM packages with proper project structure and dependencies. This command initializes a new Rust project with OpenVM configuration.

- **New `commit` command**: Added `cargo openvm commit` command for viewing the Bn254 commit of an OpenVM executable. This command generates and displays commitment information for built executables.

- **Build command output changes**: The `cargo openvm build` command now outputs `AppExecutionCommit` in JSON format and stores ELF and vmexe files in updated locations. The old output (`exe_commit.bytes`) was incorrect.

- **Setup command enhancements**: The `cargo openvm setup` command now supports skipping halo2 proving keys and outputs halo2 PK and STARK PK as separate files.

- **Hex output format**: The `cargo openvm commit` and `cargo openvm prove stark` commands now consistently output commit values in hexadecimal format. Previously, some outputs used different formats, which could cause parsing issues in downstream tools.

- **Prove command output paths**: The `cargo openvm prove` command now outputs proofs to `${bin_name}.app.proof` instead of `app.proof`, where `bin_name` is the file stem of the executable. For example, if your binary is `my_program`, the proof will be saved as `my_program.app.proof` instead of the generic `app.proof`. This applies to both `stark` and `evm` proof types.

#### SDK

- No breaking changes to the SDK public API in this release.

#### Library Interfaces

- **Guest bindings refactor**: Major breaking changes to guest library interfaces. The `setup_*` functions have been removed from guest bindings and are now called on-demand within relevant binding functions. Additionally, custom opcode initialization is now simplified through the inclusion of `openvm_init.rs` files and the `openvm::init!()` macro. This affects:
  - Code that previously called `setup_*` functions explicitly
  - Projects that need custom opcode initialization (now use `openvm::init!()`)
  - Benchmarks and examples that depend on the old guest library interface

### Migration Guide

#### CLI Migration
- **New commands**: Use `cargo openvm init <directory>` to create new OpenVM projects with proper structure
- **Output format changes**: Update scripts parsing `cargo openvm commit` and `cargo openvm prove stark` output to handle hexadecimal format
- **Prove command paths**: Update automation expecting proof files at `app.proof` to look for `${binary_name}.app.proof` instead

#### SDK Migration
- Update import statements:
  ```rust
  // Old
  use openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
  
  // New
  use openvm_circuit::arch::DEFAULT_MAX_NUM_PUBLIC_VALUES;
  ```

#### Library Migration
- **Remove explicit setup calls**: Remove any explicit calls to `setup_*` functions in your guest code, as these are now called automatically on-demand
- **Add initialization macro**: Include `openvm::init!();` in your guest program's main function to initialize custom opcodes
- **Update build process**: The build process now generates `openvm_init.rs` files automatically based on your `openvm.toml` configuration

## v1.1.2 (2025-05-08)

- The solidity verifier contract no longer has any awareness of the OpenVM patch version. `{MAJOR_VERSION}.{MINOR_VERSION}` is the minimum information necessary to identify the verifier contract since any verifier contract changes will be accompanied by a minor version bump.

## v1.1.1 (2025-05-03)

- Adds `OpenVmHalo2Verifier` generation to the SDK which is a thin wrapper around the original `Halo2Verifier` contract exposing a more user-friendly interface.
- Updates the CLI to generate the new `OpenVmHalo2Verifier` contract during `cargo openvm setup`.
- Removes the ability to generate the old `Halo2Verifier` contract from the SDK and CLI.
- Changes the `EvmProof` struct to align with the interface of the `OpenVmHalo2Verifier` contract.
- Formats the verifier contract during generation for better readability on block explorers.
- For verifier contract compilation, explicitly sets the `solc` config via standard-json input for metadata consistency.

## v1.1.0 (2025-05-02)

### Security Fixes
- Fixes security vulnerability [OpenVM allows the byte decomposition of pc in AUIPC chip to overflow](https://github.com/advisories/GHSA-jf2r-x3j4-23m7)
