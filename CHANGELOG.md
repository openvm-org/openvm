# Changelog

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
