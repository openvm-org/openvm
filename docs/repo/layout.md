## Project Layout

The main components of the repository are:

- [Project Layout](#project-layout)
  - [Documentation](#documentation)
  - [Benchmarks](#benchmarks)
  - [CI](#ci)
  - [CLI](#cli)
  - [VM SDK](#vm-sdk)
  - [Rust Toolchain](#rust-toolchain)
  - [VM Framework](#vm-framework)
  - [Circuit Foundations](#circuit-foundations)
  - [Proof System](#proof-system)
  - [Extensions](#extensions)
    - [RV32IM](#rv32im)
    - [Native Recursion](#native-recursion)
    - [Keccak256](#keccak256)
    - [Big Integers](#big-integers)
    - [Modular Arithmetic](#modular-arithmetic)
    - [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
    - [Pairing](#pairing)

### Documentation

Contributor documentation is in [`docs`](../../docs) and end-user documentation is in [`book`](../../book).

### Benchmarks

Benchmark guest programs and benchmark scripts are in [`openvm-benchmarks`](../../benchmarks).

### CI

Scripts for CI use and metrics post-processing are in [`ci`](../../ci).

### CLI

Command-line binary to compile, execute, and prove guest programs is in [`cli`](../../crates/cli).

### VM SDK

- [`sdk`](../../crates/sdk): The developer SDK for the VM. It includes the OpenVM aggregation programs to support continuations for all VMs in the framework, and well as local aggregation scheduling implementation. It provides the final interface for proving an arbitrary program for a target VM. Includes utilities to generate final onchain SNARK verifier contract.

### Rust Toolchain

- [`openvm`](../../crates/toolchain/openvm): The OpenVM standard library to be imported by guest programs. Contains `main` function entrypoint and standard intrinsic functions for IO.
- [`openvm-platform`](../../crates/toolchain/platform): Rust runtime for RV32IM target using OpenVM intrinsic for system termination.
- [`openvm-transpiler`](../../crates/toolchain/transpiler): Transpiler for converting RISC-V ELF with custom instructions into OpenVM executable with OpenVM instructions.
- [`openvm-macros-common`](../../crates/toolchain/macros): Common library for parsing utilities shared across procedural macros used for custom instruction setup in guest programs.
- [`openvm-toolchain-tests`](../../crates/toolchain/tests): Testing of Rust toolchain including all official RISC-V 32-bit IM test vectors.

### VM Framework

- [`openvm-circuit`](../../crates/vm): The VM circuit framework. It includes the struct and trait definitions used throughout the architecture, as well as the system chips.
- [`openvm-circuit-derive`](../../crates/vm/derive): Procedural macros to derive traits in the VM circuit framework.
- [`openvm-instructions`](../../crates/toolchain/instructions): OpenVM instruction struct and trait definitions.
- [`openvm-instructions-derive`](../../crates/toolchain/instructions/derive): Procedural macros to derive traits for OpenVM instructions.

### Circuit Foundations

- [`openvm-circuit-primitives`](../../crates/circuits/primitives): Primitive chips and sub-chips for standalone use in any circuit.
- [`openvm-circuit-primitives-derive`](../../crates/circuits/derive): Procedural macros for use in circuit to derive traits.
- [`openvm-poseidon2-air`](../../crates/circuits/poseidon2-air): Standalone poseidon2 AIR implementation.
- [`openvm-mod-circuit-builder`](../../crates/circuits/mod-builder): General builder for generating chip for any modular arithmetic expression for a compile-time modulus.

### Proof System

- [`openvm-stark-backend`](../../crates/stark-backend): General purpose STARK proving system with multi-trace and logup support, built on top of plonky3.
- [`openvm-stark-sdk`](../../crates/stark-sdk): Low-level SDK for use with STARK backend to generate proofs for specific STARK configurations.

### Extensions

The toolchain, ISA, and VM are simultaenously extendable. All non-system functionality is implemented via extensions, which may be moved to standalone repositories in the future but are presently in this repository for maintainer convenience.

#### RV32IM

- [`openvm-rv32im-circuit`](../../extensions/rv32im/circuit): VM circuit extension for RV32IM instructions, including IO operations.
- [`openvm-rv32im-transpiler`](../../extensions/rv32im/transpiler): Transpiler extension for RV32IM instructions.
- [`openvm-rv32im-guest`](../../extensions/rv32im/guest): Guest library for RV32IM instructions.

#### Native Recursion

- [`openvm-native-circuit`](../../extensions/native/circuit/): VM circuit extension for native instructions operating on field elements.
- [`openvm-native-compiler`](../../extensions/native/compiler/): Implementation of compiler from a Rust embedded DSL to OpenVM assembly targeting the native kernel extension. The eDSL also has a static mode to support compilation to a Halo2 circuit.
- [`openvm-native-recursion`](../../extensions/native/recursion): Library written in the native eDSL with functions to verify arbitrary STARK proofs. Library supports compilation to Halo2 circuit.

#### Keccak256

- [`openvm-keccak256-circuit`](../../extensions/keccak256/circuit): VM circuit extension for `keccak256` hash function.
- [`openvm-keccak256-transpiler`](../../extensions/keccak256/transpiler): Transpiler extension for `keccak256` hash function.
- [`openvm-keccak256-guest`](../../extensions/keccak256/guest): Guest library with intrinsic function for `keccak256` hash function.

#### Big Integers

- [`openvm-bigint-circuit`](../../extensions/bigint/circuit): VM circuit extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-transpiler`](../../extensions/bigint/transpiler): Transpiler extension for `I256` and `U256` big integer operations.
- [`openvm-bigint-guest`](../../extensions/bigint/guest): Guest library with `I256` and `U256` big integers operations using intrinsics for underlying operations.

#### Modular Arithmetic

- [`openvm-algebra-circuit`](../../extensions/algebra/circuit): VM circuit extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and Fp2 operations.
- [`openvm-algebra-transpiler`](../../extensions/algebra/transpiler): Transpiler extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and Fp2 operations.
- [`openvm-algebra-guest`](../../extensions/algebra/guest): Guest library with traits for modular arithmetic and Fp2 operations.
- [`openvm-algebra-moduli-setup`](../../extensions/algebra/moduli-setup): Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.

#### Elliptic Curve Cryptography

- [`openvm-ecc-circuit`](../../extensions/ecc/circuit): VM circuit extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-transpiler`](../../extensions/ecc/transpiler): Transpiler extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`openvm-ecc-guest`](../../extensions/ecc/guest): Guest library with elliptic curve constants for Secp256k1 and functions using custom intrinsics, including ECDSA.
- [`openvm-ecc-sw-setup`](../../extensions/ecc/sw-setup): Procedural macros for use in guest program to generate short Weierstrass curve struct with custom intrinsics for compile-time curve.

#### Pairing

- [`openvm-pairing-circuit`](../../extensions/pairing/circuit): VM circuit extension for optimal Ate pairing on arbitrary compile-time elliptic curves, including BN254 and BLS12-381.
- [`openvm-pairing-transpiler`](../../extensions/pairing/transpiler): Transpiler extension for optimal Ate pairing on arbitrary compile-time elliptic curves, including BN254 and BLS12-381.
- [`openvm-pairing-guest`](../../extensions/pairing/guest): Guest library with optimal Ate pairing on elliptic curves, including BN254 and BLS12-381 and associated constants. Also includes elliptic curve operations for VM runtime with the `halo2curves` feature gate.
