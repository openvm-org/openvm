# Changelog

All notable changes to OpenVM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.2.1 (TBD)

### Breaking Changes

#### CLI

- **New `init` command**: Added `cargo openvm init` command for creating new OpenVM packages with proper project structure and dependencies. This command initializes a new Rust project with OpenVM configuration.

- **New `commit` command**: Added `cargo openvm commit` command for viewing the Bn254 commit of an OpenVM executable. This command generates and displays commitment information for built executables.

- **Setup command renamed**: The `EvmProvingSetupCmd` has been renamed to `SetupCmd`. Scripts and automation that reference the old command structure will need to be updated.

- **Hex output format**: All CLI commit outputs are now consistently formatted in hexadecimal. Previously, some outputs used different formats, which could cause parsing issues in downstream tools.

- **Prove command improvements**: The `prove` command now uses the binary name for default output file paths instead of generic names. This provides better organization when working with multiple executables.

#### SDK

- **Import path change**: The import path for `DEFAULT_MAX_NUM_PUBLIC_VALUES` has changed from `openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES` to `openvm_circuit::arch::DEFAULT_MAX_NUM_PUBLIC_VALUES`. Code using the old import path will fail to compile.

- **Configuration structure updates**: The `AggregationTreeConfig` struct now includes additional CLI help text and argument groupings with `help_heading = "Aggregation Tree Options"`. While this doesn't break functionality, it may affect tools that parse CLI help output.

#### Library Interfaces

- **Guest bindings refactor**: This is a major breaking change. Guest library components have been removed from this repository and moved to a separate repository as part of a guest library reorganization. This affects:
  - Benchmarks that depend on guest libraries
  - Examples that use guest library functions
  - Any user code that imports guest library modules directly

### Migration Guide

#### CLI Migration
- Update any scripts that call CLI commands to use the new command names
- Expect hexadecimal output format from commit commands and update parsing logic accordingly
- Update references from `EvmProvingSetupCmd` to `SetupCmd` in configuration or documentation

#### SDK Migration
- Update import statements:
  ```rust
  // Old
  use openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
  
  // New
  use openvm_circuit::arch::DEFAULT_MAX_NUM_PUBLIC_VALUES;
  ```

#### Library Migration
- **Guest Libraries**: Migrate to the new guest library repository. The exact migration path depends on which guest library components your code uses. Consult the new guest library repository documentation for specific migration instructions.
- **Benchmarks and Examples**: These will need to be redesigned to use the new guest library structure.

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
