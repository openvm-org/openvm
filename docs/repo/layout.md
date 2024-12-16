## Project Layout

The main components of the repository are:

- [Project Layout](#project-layout)
  - [Documentation](#documentation)
  - [Benchmarks](#benchmarks)
  - [CI](#ci)
  - [CLI](#cli)
  - [SDK](#sdk)
  - [Toolchain](#toolchain)
  - [Circuit Framework](#circuit-framework)
  - [Circuit Foundations](#circuit-foundations)
  - [Extensions](#extensions)
    - [RV32IM](#rv32im)
    - [Native Recursion](#native-recursion)
    - [Keccak256](#keccak256)
    - [Big Integers](#big-integers)
    - [Algebra (Modular Arithmetic)](#algebra-modular-arithmetic)
    - [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
    - [Elliptic Curve Pairing](#elliptic-curve-pairing)

### Documentation

Contributor documentation is in [`docs`](../../docs) and user documentation is in [`book`](../../book).

### Benchmarks

Benchmark guest programs and benchmark scripts are in [`openvm-benchmarks`](../../benchmarks).

### CI

Scripts for CI use and metrics post-processing are in [`ci`](../../ci).

### CLI

Command-line binary to compile, execute, and prove guest programs is in [`cli`](../../crates/cli).

### SDK

- [`sdk`](../../crates/sdk): The developer SDK for the VM. It includes the OpenVM aggregation programs to support continuations for all VMs in the framework, and well as a local aggregation scheduling implementation. It provides the final interface for proving an arbitrary program for a target VM. The SDK includes functionality to generate the final onchain SNARK verifier contract.

### Toolchain

- [`openvm`](../../crates/toolchain/openvm): The OpenVM standard library to be imported by guest programs. Contains `main` function entrypoint and standard intrinsic functions for IO.
- [`openvm-platform`](../../crates/toolchain/platform): Rust runtime for RV32IM target using OpenVM intrinsic for system termination. This crate is re-exported by the `openvm` crate.
- [`openvm-build`](../../crates/toolchain/build): Library of build tools for compiling Rust to the RISC-V target, built on top of `cargo`.
- [`openvm-transpiler`](../../crates/toolchain/transpiler): Transpiler for converting RISC-V ELF with custom instructions into OpenVM executable with OpenVM instructions. This crate contains the `TranspilerExtension` trait and a `Transpiler` struct which supports adding custom `TranspilerExtension` implementations.
- [`openvm-instructions`](../../crates/toolchain/instructions): OpenVM instruction struct and trait definitions. Also includes some system instruction definitions.
- [`openvm-instructions-derive`](../../crates/toolchain/instructions/derive): Procedural macros to derive traits for OpenVM instructions.
- [`openvm-macros-common`](../../crates/toolchain/macros): Common library for parsing utilities shared across procedural macros used for custom instruction setup in guest programs.
- [`openvm-toolchain-tests`](../../crates/toolchain/tests): Testing of Rust toolchain including all official RISC-V 32-bit IM test vectors. Currently this is a monolithic crate with tests across many different extensions. We will soon refactor the tests to be more modular.

### Circuit Framework

- [`openvm-circuit`](../../crates/vm): The VM circuit framework. It includes the struct and trait definitions used throughout the architecture, as well as the system chips.
- [`openvm-circuit-derive`](../../crates/vm/derive): Procedural macros to derive traits in the VM circuit framework.

### Circuit Foundations

- [`openvm-circuit-primitives`](../../crates/circuits/primitives): Primitive chips and sub-chips for standalone use in any circuit.
- [`openvm-circuit-primitives-derive`](../../crates/circuits/primitives/derive): Procedural macros for use in circuit to derive traits.
- [`openvm-poseidon2-air`](../../crates/circuits/poseidon2-air): Standalone poseidon2 AIR implementation which is configurable based on the desired maximum constraint degree.
- [`openvm-mod-circuit-builder`](../../crates/circuits/mod-builder): General builder for generating a chip for any modular arithmetic expression for a modulus known at compile time.

### Extensions

The toolchain, ISA, and VM are simultaenously extendable. All non-system functionality is implemented via extensions, which may be moved to standalone repositories in the future but are presently in this repository for maintainer convenience.

#### RV32IM

- [`openvm-rv32im-circuit`](../../extensions/rv32im/circuit): Circuit extension for RV32IM instructions and IO instructions.
- [`openvm-rv32im-transpiler`](../../extensions/rv32im/transpiler): Transpiler extension for RV32IM instructions and IO instructions.
- [`openvm-rv32im-guest`](../../extensions/rv32im/guest): Guest library for RV32IM instructions and IO instructions. This is re-exported by the `openvm` crate for convenience.

#### Native Recursion

- [`openvm-native-circuit`](../../extensions/native/circuit/): Circuit extension for native instructions operating on field elements.
- [`openvm-native-compiler`](../../extensions/native/compiler/): Implementation of compiler from a Rust embedded DSL to OpenVM assembly targeting the native kernel extension. The eDSL also has a static mode to support compilation to a Halo2 circuit.
- [`openvm-native-recursion`](../../extensions/native/recursion): Library written in the native eDSL with functions to verify arbitrary STARK proofs. The library also supports compilation to a Halo2 circuit.

#### Keccak256

- [`openvm-keccak256-circuit`](../../extensions/keccak256/circuit): Circuit extension for the `keccak256` hash function.
- [`openvm-keccak256-transpiler`](../../extensions/keccak256/transpiler): Transpiler extension for the `keccak256` hash function.
- [`openvm-keccak256-guest`](../../extensions/keccak256/guest): Guest library with intrinsic function for the `keccak256` hash function.

#### Big Integers

- [`openvm-bigint-circuit`](../../extensions/bigint/circuit): Circuit extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-transpiler`](../../extensions/bigint/transpiler): Transpiler extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-guest`](../../extensions/bigint/guest): Guest library with `I256` and `U256` big integers operations using intrinsics for underlying operations.

#### Algebra (Modular Arithmetic)

- [`openvm-algebra-circuit`](../../extensions/algebra/circuit): Circuit extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and complex field extension operations.
- [`openvm-algebra-transpiler`](../../extensions/algebra/transpiler): Transpiler extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and complex field extension operations.
- [`openvm-algebra-guest`](../../extensions/algebra/guest): Guest library with traits for modular arithmetic and complex field extension operations.
- [`openvm-algebra-moduli-setup`](../../extensions/algebra/moduli-setup): Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.
- [`openvm-algebra-complex-macros`](../../extensions/algebra/guest/src/field/complex-macros): Procedural macros for use in guest program to generate complex field struct with custom intrinsics for compile-time modulus.

#### Elliptic Curve Cryptography

- [`openvm-ecc-circuit`](../../extensions/ecc/circuit): Circuit extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-transpiler`](../../extensions/ecc/transpiler): Transpiler extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-guest`](../../extensions/ecc/guest): Guest library with traits for elliptic curve cryptography. Includes implementations of ECDSA and multi-scalar multiplication.
- [`openvm-ecc-sw-setup`](../../extensions/ecc/sw-setup): Procedural macros for use in guest program to generate short Weierstrass curve struct with custom intrinsics for compile-time curve.

#### Elliptic Curve Pairing

- [`openvm-pairing-circuit`](../../extensions/pairing/circuit): Circuit extension for optimal Ate pairing on BN254 and BLS12-381 curves.
- [`openvm-pairing-transpiler`](../../extensions/pairing/transpiler): Transpiler extension for optimal Ate pairing on BN254 and BLS12-381.
- [`openvm-pairing-guest`](../../extensions/pairing/guest): Guest library with optimal Ate pairing on BN254 and BLS12-381 and associated constants. Also includes elliptic curve operations for VM runtime with the `halo2curves` feature gate.
