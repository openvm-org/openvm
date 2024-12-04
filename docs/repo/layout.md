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
    - [Recursion](#recursion)
    - [Keccak256](#keccak256)
    - [Big Integers](#big-integers)
    - [Modular Arithmetic](#modular-arithmetic)
    - [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
    - [Pairing](#pairing)

### Documentation

Contributor documentation is in [`docs`](../../docs) and end-user documentation is in [`book`](../../book).

### Benchmarks

Benchmark guest programs and benchmark scripts are in [`axvm-benchmarks`](../../benchmarks).

### CI

Scripts for CI use and metrics post-processing are in [`ci`](../../ci).

### CLI

Command-line binary to compile, execute, and prove guest programs is in [`cargo-axiom`](../../crates/cargo-axiom).

### VM SDK

- [`axvm-sdk`](../../crates/axvm-sdk): The developer SDK for the VM. It includes the axVM aggregation programs to support continuations for all VMs in the framework, and well as local aggregation scheduling implementation. It provides the final interface for proving an arbitrary program for a target VM. Includes utilities to generate final onchain SNARK verifier contract.

### Rust Toolchain

- [`axvm`](../../crates/toolchain/axvm): The axVM standard library to be imported by guest programs. Contains `main` function entrypoint and standard intrinsic functions for IO.
- [`axvm-platform`](../../crates/toolchain/platform): Rust runtime for RV32IM target using axVM intrinsic for system termination.
- [`axvm-transpiler`](../../crates/toolchain/transpiler): Transpiler for converting RISC-V ELF with custom instructions into axVM executable with axVM instructions.
- [`axvm-macros-common`](../../crates/toolchain/macros): Common library for parsing utilities shared across procedural macros used for custom instruction setup in guest programs.
- [`axvm-toolchain-tests`](../../crates/toolchain/tests): Testing of Rust toolchain including all official RISC-V 32-bit IM test vectors.

### VM Framework

- [`axvm-circuit`](../../crates/vm): The VM circuit framework. It includes the struct and trait definitions used throughout the architecture, as well as the system chips.
- [`axvm-circuit-derive`](../../crates/vm/derive): Procedural macros to derive traits in the VM circuit framework.
- [`axvm-instructions`](../../crates/toolchain/instructions): axVM instruction struct and trait definitions.
- [`axvm-instructions-derive`](../../crates/toolchain/instructions/derive): Procedural macros to derive traits for axVM instructions.

### Circuit Foundations

- [`ax-circuit-primitives`](../../crates/circuits/primitives): Primitive chips and sub-chips for standalone use in any circuit.
- [`ax-circuit-derive`](../../crates/circuits/derive): Procedural macros for use in circuit to derive traits.
- [`ax-poseidon2-air`](../../crates/circuits/poseidon2-air): Standalone poseidon2 AIR implementation.
- [`ax-mod-circuit-builder`](../../crates/circuits/mod-builder): General builder for generating chip for any modular arithmetic expression for a compile-time modulus. 

### Proof System

- [`ax-stark-backend`](../../crates/stark-backend): General purpose STARK proving system with multi-trace and logup support, built on top of plonky3.
- [`ax-stark-sdk`](../../crates/stark-sdk): Low-level SDK for use with STARK backend to generate proofs for specific STARK configurations.

### Extensions

The toolchain, ISA, and VM are simultaenously extendable. All non-system functionality is implemented via extensions, which may be moved to standalone repositories in the future but are presently in this repository for maintainer convenience.

#### RV32IM

- [`axvm-rv32im-circuit`](../../extensions/rv32im/circuit): VM circuit extension for RV32IM instructions, including IO operations.
- [`axvm-rv32im-transpiler`](../../extensions/rv32im/transpiler): Transpiler extension for RV32IM instructions.
- [`axvm-rv32im-guest`](../../extensions/rv32im/guest): Guest library for RV32IM instructions.

#### Native Recursion

- [`axvm-native-circuit`](../../extensions/native/circuit/): VM circuit extension for native instructions operating on field elements.
- [`axvm-native-compiler`](../../extensions/native/compiler/): Implementation of compiler from a Rust embedded DSL to axVM assembly targeting the native kernel extension. The eDSL also has a static mode to support compilation to a Halo2 circuit.
- [`axvm-native-recursion`](../../extensions/native/recursion): Library written in the native eDSL with functions to verify arbitrary STARK proofs. Library supports compilation to Halo2 circuit.

#### Keccak256

- [`axvm-keccak256-circuit`](../../extensions/keccak256/circuit): VM circuit extension for `keccak256` hash function.
- [`axvm-keccak256-transpiler`](../../extensions/keccak256/transpiler): Transpiler extension for `keccak256` hash function.
- [`axvm-keccak256-guest`](../../extensions/keccak256/guest): Guest library with intrinsic function for `keccak256` hash function.

#### Big Integers

- [`axvm-bigint-circuit`](../../extensions/bigint/circuit): VM circuit extension for `I256` and `U256` big integer operations.
- [`axvm-bigint-transpiler`](../../extensions/bigint/transpiler): Transpiler extension for `I256` and `U256` big integer operations.
- [`axvm-bigint-guest`](../../extensions/bigint/guest): Guest library with `I256` and `U256` big integers operations using intrinsics for underlying operations.

#### Modular Arithmetic

- [`axvm-algebra-circuit`](../../extensions/algebra/circuit): VM circuit extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and Fp2 operations.
- [`axvm-algebra-transpiler`](../../extensions/algebra/transpiler): Transpiler extension for modular arithmetic for arbitrary compile-time modulus. Supports modular arithmetic and Fp2 operations.
- [`axvm-algebra-guest`](../../extensions/algebra/guest): Guest library with traits for modular arithmetic and Fp2 operations.
- [`axvm-algebra-moduli-setup`](../../extensions/algebra/moduli-setup): Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.

#### Elliptic Curve Cryptography

- [`axvm-ecc-circuit`](../../extensions/ecc/circuit): VM circuit extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`axvm-ecc-transpiler`](../../extensions/ecc/transpiler): Transpiler extension for Weierstrass elliptic curve operations for arbitrary compile-time curve.
- [`axvm-ecc-guest`](../../extensions/ecc/guest): Guest library with elliptic curve functions using custom intrinsics, including ECDSA.
- [`axvm-ecc-execution`](../../extensions/ecc/execution): Elliptic curve operations for use in VM runtime execution.
- [`axvm-ecc-constants`](../../extensions/ecc/constants): Constants for elliptic curves, including BN254, BLS12-381, and Secp256k1.
- [`axvm-ecc-sw-setup`](../../extensions/ecc/sw-setup): Procedural macros for use in guest program to generate short Weierstrass curve struct with custom intrinsics for compile-time curve.

#### Pairing

- [`axvm-pairing-circuit`](../../extensions/pairing/circuit): VM circuit extension for optimal Ate pairing on arbitrary compile-time elliptic curves, including BN254 and BLS12-381.
- [`axvm-pairing-transpiler`](../../extensions/pairing/transpiler): Transpiler extension for optimal Ate pairing on arbitrary compile-time elliptic curves, including BN254 and BLS12-381.
- [`axvm-pairing-guest`](../../extensions/pairing/guest): Guest library with optimal Ate pairing on elliptic curves, including BN254 and BLS12-381.
