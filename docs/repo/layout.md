## Project Layout

The main components of the repository are:

- [Project Layout](#project-layout)
  - [Documentation](#documentation)
  - [Benchmarks](#benchmarks)
  - [CI](#ci)
  - [Profiling](#profiling)
  - [CLI](#cli)
  - [SDK](#sdk)
  - [Toolchain](#toolchain)
  - [Circuit Framework](#circuit-framework)
  - [Circuit Foundations](#circuit-foundations)
  - [Recursion](#recursion)
  - [Continuations](#continuations)
  - [Examples](#examples)
  - [Extensions](#extensions)
    - [RV64IM](#rv64im)
    - [Deferral](#deferral)
    - [Keccak256](#keccak256)
    - [Big Integers](#big-integers)
    - [Algebra (Modular Arithmetic)](#algebra-modular-arithmetic)
    - [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
    - [SHA-2](#sha-2)
    - [Elliptic Curve Pairing](#elliptic-curve-pairing)
  - [Guest Libraries](#guest-libraries)
  - [Verification & Configuration](#verification--configuration)

### Documentation

Contributor documentation is in [`docs`](../../docs) and user documentation is in [`docs/vocs`](../../docs/vocs).

### Benchmarks

Benchmark guest programs and benchmark scripts are in [`benchmarks`](../../benchmarks), which contains:
- [`openvm-benchmarks-execute`](../../benchmarks/execute): Execution benchmarks (no proving) using divan.
- [`openvm-benchmarks-prove`](../../benchmarks/prove): Proving benchmark binaries.
- [`openvm-benchmarks-utils`](../../benchmarks/utils): Shared utilities for building guest ELFs.

### CI

Scripts for CI use and metrics post-processing are in [`ci`](../../ci).

### Profiling

- [`openvm-prof`](../../crates/prof): Tools to post-process metrics emitted by the VM for performance profiling.

### CLI

Command-line binary to compile, execute, and prove guest programs is in [`cli`](../../crates/cli).

### SDK

- [`openvm-sdk`](../../crates/sdk): The developer SDK for the VM. It provides the final interface for proving an arbitrary program for a target VM, including a local aggregation scheduling implementation for continuations. The SDK includes functionality to generate the final onchain SNARK verifier contract.

### Toolchain

- [`openvm`](../../crates/toolchain/openvm): The OpenVM standard library to be imported by guest programs. Contains `main` function entrypoint and standard intrinsic functions for IO.
- [`openvm-platform`](../../crates/toolchain/platform): Rust runtime for RV64IM target using OpenVM intrinsic for system termination. This crate is re-exported by the `openvm` crate.
- [`openvm-build`](../../crates/toolchain/build): Library of build tools for compiling Rust to the RISC-V target, built on top of `cargo`.
- [`openvm-transpiler`](../../crates/toolchain/transpiler): Transpiler for converting RISC-V ELF with custom instructions into OpenVM executable with OpenVM instructions. This crate contains the `TranspilerExtension` trait and a `Transpiler` struct which supports adding custom `TranspilerExtension` implementations.
- [`openvm-instructions`](../../crates/toolchain/instructions): OpenVM instruction struct and trait definitions. Also includes some system instruction definitions.
- [`openvm-instructions-derive`](../../crates/toolchain/instructions/derive): Procedural macros to derive traits for OpenVM instructions.
- [`openvm-macros-common`](../../crates/toolchain/macros): Common library for parsing utilities shared across procedural macros used for custom instruction setup in guest programs.
- [`openvm-toolchain-tests`](../../crates/toolchain/tests): Includes all official RISC-V 64-bit IM test vectors and transpiler tests. Also, provides utilities for writing integration tests for custom extensions.
- [`openvm-custom-insn`](../../crates/toolchain/custom_insn): Custom instruction macros for use in guest programs.

### Circuit Framework

- [`openvm-circuit`](../../crates/vm): The VM circuit framework. It includes the struct and trait definitions used throughout the architecture, as well as the system chips.
- [`openvm-circuit-derive`](../../crates/vm/derive): Procedural macros to derive traits in the VM circuit framework.

### Circuit Foundations

- [`openvm-circuit-primitives`](../../crates/circuits/primitives): Primitive chips and sub-chips for standalone use in any circuit.
- [`openvm-circuit-primitives-derive`](../../crates/circuits/primitives/derive): Procedural macros for use in circuit to derive traits.
- [`openvm-poseidon2-air`](../../crates/circuits/poseidon2-air): Wrapper around `p3_poseidon2_air` only intended for use in OpenVM with BabyBear.
- [`openvm-mod-circuit-builder`](../../crates/circuits/mod-builder): General builder for generating a chip for any modular arithmetic expression for a modulus known at compile time.

### Recursion
- [`openvm-recursion-circuit`](../../crates/recursion): Sub-circuit used to verify child STARK proofs.

### Continuations
- [`openvm-continuations`](../../crates/continuations): Continuation-specific AIRs and utilities for aggregating segment proofs for all VMs in the framework.

### Examples

- [`examples`](../../examples): Examples of guest programs using the OpenVM framework. All of the examples can be built and run using the CLI.

### Extensions

The toolchain, ISA, and VM are simultaneously extendable. All non-system functionality is implemented via extensions, which may be moved to standalone repositories in the future but are presently in this repository for maintainer convenience.

#### Procedural macros for algebraic structs

- [`openvm-algebra-moduli-macros`](../../extensions/algebra/moduli-macros): Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.
- [`openvm-algebra-complex-macros`](../../extensions/algebra/complex-macros): Procedural macros for use in guest program to generate complex field struct with custom intrinsics for compile-time modulus.
- [`openvm-ecc-sw-macros`](../../extensions/ecc/sw-macros): Procedural macros for use in guest program to generate short Weierstrass curve struct with custom intrinsics for compile-time curve.

#### RV64IM

- [`openvm-riscv-circuit`](../../extensions/riscv/circuit): Circuit extension for RV64IM instructions and IO instructions.
- [`openvm-riscv-transpiler`](../../extensions/riscv/transpiler): Transpiler extension for RV64IM instructions and IO instructions.
- [`openvm-riscv-guest`](../../extensions/riscv/guest): Guest library for RV64IM instructions and IO instructions. This is re-exported by the `openvm` crate for convenience.
- [`openvm-riscv-adapters`](../../extensions/riscv-adapters): Circuit adapters for other circuit extensions to use to be compatible with the RISC-V 64-bit architecture.
- [`openvm-riscv-integration-tests`](../../extensions/riscv/tests): Integration tests for the RV64IM extension.

#### Deferral

- [`openvm-deferral-circuit`](../../extensions/deferral/circuit): Circuit extension for deferred computation. Provides chips for deferral calls and outputs.
- [`openvm-deferral-transpiler`](../../extensions/deferral/transpiler): Transpiler extension for deferral instructions.
- [`openvm-deferral-guest`](../../extensions/deferral/guest): Guest library with deferral instruction definitions and types.
- [`openvm-deferral-integration-tests`](../../extensions/deferral/tests): Integration tests for the deferral extension.

#### Keccak256

- [`openvm-keccak256-circuit`](../../extensions/keccak256/circuit): Circuit extension for the `keccak256` hash function.
- [`openvm-keccak256-transpiler`](../../extensions/keccak256/transpiler): Transpiler extension for the `keccak256` hash function.
- [`openvm-keccak256-guest`](../../extensions/keccak256/guest): Guest library with intrinsic function for the `keccak256` hash function.

#### SHA-2

- [`openvm-sha2-air`](../../crates/circuits/sha2-air): Standalone SHA-2 AIR implementation.
- [`openvm-sha2-circuit`](../../extensions/sha2/circuit): Circuit extension for SHA-2.
- [`openvm-sha2-transpiler`](../../extensions/sha2/transpiler): Transpiler extension for SHA-2.
- [`openvm-sha2-guest`](../../extensions/sha2/guest): Guest library for SHA-2.

#### Big Integers

- [`openvm-bigint-circuit`](../../extensions/bigint/circuit): Circuit extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-transpiler`](../../extensions/bigint/transpiler): Transpiler extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-guest`](../../extensions/bigint/guest): Guest library with `I256` and `U256` big integers operations using intrinsics for underlying operations.

#### Algebra (Modular Arithmetic)

- [`openvm-algebra-circuit`](../../extensions/algebra/circuit): Circuit extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and complex field extension operations.
- [`openvm-algebra-transpiler`](../../extensions/algebra/transpiler): Transpiler extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and complex field extension operations.
- [`openvm-algebra-guest`](../../extensions/algebra/guest): Guest library with traits for modular arithmetic and complex field extension operations.
- [`openvm-algebra-tests`](../../extensions/algebra/tests): Integration tests for the algebra extension.

#### Elliptic Curve Cryptography

- [`openvm-ecc-circuit`](../../extensions/ecc/circuit): Circuit extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-transpiler`](../../extensions/ecc/transpiler): Transpiler extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-guest`](../../extensions/ecc/guest): Guest library with traits for elliptic curve cryptography. Includes implementations of ECDSA and multi-scalar multiplication.
- [`openvm-ecc-integration-tests`](../../extensions/ecc/tests): Integration tests for the elliptic curve cryptography extension.

#### Elliptic Curve Pairing

- [`openvm-pairing-circuit`](../../extensions/pairing/circuit): Circuit extension for optimal Ate pairing on BN254 and BLS12-381 curves.
- [`openvm-pairing-transpiler`](../../extensions/pairing/transpiler): Transpiler extension for optimal Ate pairing on BN254 and BLS12-381.
- [`openvm-pairing-guest`](../../extensions/pairing/guest): Guest library with optimal Ate pairing on BN254 and BLS12-381 and associated constants. Also includes elliptic curve operations for VM runtime with the `halo2curves` feature gate.

### Guest Libraries

Forked or custom libraries optimized for guest program execution inside the VM.

- [`openvm-ff-derive`](../../guest-libs/ff_derive): OpenVM fork of `ff_derive` for finite field arithmetic.
- [`k256`](../../guest-libs/k256): OpenVM fork of `k256`.
- [`openvm-keccak256`](../../guest-libs/keccak256): OpenVM library for keccak256.
- [`p256`](../../guest-libs/p256): OpenVM fork of `p256`.
- [`openvm-pairing`](../../guest-libs/pairing): OpenVM library for elliptic curve pairing.
- [`ruint`](../../guest-libs/ruint): OpenVM fork of `ruint`.
- [`openvm-sha2`](../../guest-libs/sha2): OpenVM library for SHA-2.
- [`openvm-verify-stark-circuit`](../../guest-libs/verify-stark/circuit): Circuit extension for verifying STARKs in-guest.
- [`openvm-verify-stark-guest`](../../guest-libs/verify-stark/guest): Guest library for verifying STARKs.

### Verification & Configuration

- [`openvm-sdk-config`](../../crates/sdk-config): SDK configuration types, separated for lighter downstream dependencies.
- [`openvm-static-verifier`](../../crates/static-verifier): Static verifier generation.
- [`openvm-verify-stark-host`](../../crates/verify): Lightweight crate to verify a STARK proof for an OpenVM virtual machine.
