# Changelog

All notable changes to OpenVM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows a versioning principles documented in [VERSIONING.md](./VERSIONING.md).

## v1.3.0 (2025-07-15)

No circuit constraints or verifying keys were changed in this release.

A substantial refactor has been done to the guest libraries to separate the low level Rust bindings for OpenVM intrinsic instructions from the higher level user interface. For each VM extension, the `openvm-$name-guest` crate is now a _primitives library_ containing only the Rust bindings for the intrinsic instructions and essential logic related to the extension (e.g., ECDSA signature verification). We introduce new _guest libraries_ as standalone Rust crates which provide the high-level interfaces guest programs should use to interact with the associated VM extensions.

Users are advised to switch to using the new guest libraries.

### Added
- (ISA) Added OpenVM phantom sub-instructions `HintNonQr` and `HintSqrt` to the algebra (modular arithmetic) extension. Added corresponding RISC-V custom instructions `hint_non_qr` and `hint_sqrt`.
- (Guest Libraries) We introduce the following new guest libraries:
  - `openvm-keccak256`: guest library for the Keccak256 hash function.
  - `openvm-sha2`: guest library providing access to a set of accelerated SHA-2 family hash functions. Currently, the SHA-256 hash function is supported.
  - `openvm-pairing`: guest library for the elliptic curve pairing check operation.
  - `ff_derive`: patch of [ff_derive](https://crates.io/crates/ff_derive) using the algebra extension.
  - `k256`: patch of [k256](https://crates.io/crates/k256) using the algebra and ECC extensions.
  - `p256`: patch of [p256](https://crates.io/crates/p256) using the algebra and ECC extensions.
  - `ruint`: patch of [ruint](https://crates.io/crates/ruint) using the big integer extension.
  - `openvm-verify-stark`: a new guest library providing a `define_verify_stark_proof!` macro which generates a user-named function `$verify_stark` that can be used to verify an OpenVM STARK proof from within a Rust program. The `$verify_stark` function is accelerated using the native field arithmetic extension.
- (CLI) New `cargo openvm init` and `cargo openvm commit` commands.
- (CLI) New `cargo openvm prove stark` and `cargo openvm verify stark` commands to generate a single final STARK proof without Halo2 SNARK wrapper.
- (SDK) New functions `generate_e2e_stark_proof` and `verify_e2e_stark_proof`

### Changed
- (Toolchain) The `openvm` crate and `cargo openvm build` command have been updated to support both `getrandom` `v0.2` and `v0.3`.
- (Primitives Libraries) In the algebra and elliptic curve primitive libraries, the `setup_*` functions have been removed from guest bindings and are now called on-demand within other relevant binding functions. Additionally, custom opcode initialization is now simplified through the inclusion of `openvm_init.rs` files and the `openvm::init!()` macro. Read the book for more details.
- (CLI) The build command `cargo openvm build` now stores build artifacts in the `target/` to match cargo conventions.
- (CLI) The `cargo openvm setup` command now supports skipping halo2 proving keys and outputs halo2 PK and STARK PK as separate files.
- (CLI) The `cargo openvm commit` and `cargo openvm prove stark` commands now consistently output commit values in hexadecimal format.
- (CLI) The `cargo openvm prove` command now outputs proofs to `${bin_name}.app.proof` instead of `app.proof`.

### Removed
- (ISA) Removed OpenVM phantom sub-instructions `HintDecompress` and `HintNonQr` from the elliptic curve extension. Removed corresponding RISC-V custom instructions `hint_decompress` and `hint_non_qr`.

## v1.2.0 (2025-06-02)

### Security Fixes
This release makes fixes for security advisories:
- Plonky3: https://github.com/Plonky3/Plonky3/security/advisories/GHSA-f69f-5fx9-w9r9
- OpenVM: https://github.com/openvm-org/openvm/security/advisories/GHSA-4w7p-8f9q-f4g2 (recursion circuit fixes corresponding to Plonky3)

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
