## Project Layout

The main components of the repository are:

- [Project Layout](#project-layout)
  - [Documentation](#documentation)
  - [Benchmarks](#benchmarks)
  - [CI](#ci)
  - [CLI](#cli)
  - [VM SDK](#vm-sdk)
  - [VM Framework](#vm-framework)
  - [Rust Toolchain](#rust-toolchain)
  - [Extensions](#extensions)
    - [RV32IM](#rv32im)
    - [Recursion](#recursion)
    - [Keccak256](#keccak256)
    - [Big Integers](#big-integers)
    - [Modular Arithmetic](#modular-arithmetic)
    - [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
  - [Circuit Foundations](#circuit-foundations)
  - [Proof System](#proof-system)

### Documentation

Contributor documentation is in [`docs`](../../docs) and end-user documentation is in [`book`](../../book).

### Benchmarks

Benchmark guest programs and benchmark scripts are in [`axvm-benchmarks`](../../benchmarks).

### CI

Scripts for CI use and metrics post-processing are in [`ci`](../../ci).

### CLI

Command-line binary to compile, execute, and prove guest programs is in [`cargo-axiom`](../../cargo-axiom).

### VM SDK

- [`axvm-sdk`](../../axvm-sdk): The developer SDK for the VM. It includes the axVM aggregation programs to support continuations for all VMs in the framework, and well as local aggregation scheduling implementation. It provides the final interface for proving an arbitrary program for a target VM. Includes utilities to generate final onchain SNARK verifier contract.

### VM Framework

- [`axvm-circuit`](../../vm): The VM circuit framework. It includes the struct and trait definitions used throughout the architecture, as well as the system chips.
- [`axvm-circuit-derive`](../../vm/derive): Procedural macros to derive traits in the VM circuit framework.
- [`axvm-instructions`](../../toolchain/instructions): axVM instruction struct and trait definitions.
- [`axvm-instructions-derive`](../../toolchain/instructions/derive): Procedural macros to derive traits for axVM instructions.

### Rust Toolchain

- [`axvm`](../../toolchain/riscv/axvm): The axVM standard library to be imported by guest programs. Contains `main` function entrypoint and standard intrinsic functions for IO.
- [`axvm-platform`](../../toolchain/riscv/platform): Rust runtime for RV32IM target using axVM intrinsic for system termination.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler): Transpiler for converting RISC-V ELF with custom instructions into axVM executable with axVM instructions.
  - currently transpiler extensions for custom transpilation are still in this crate, but they will be refactored into extension crates
- [`axvm-macros-common`](../../toolchain/riscv/macros): Common library for parsing utilities shared across procedural macros used for custom instruction setup in guest programs.
- [`axvm-toolchain-tests`](../../toolchain/tests): Testing of Rust toolchain including all official RISC-V 32-bit IM test vectors.

### Extensions

The toolchain, ISA, and VM are simultaenously extendable. This repository maintains several extensions. These can be moved to standalone repositories in the future but are kept in this repository for maintainer convenience.

#### RV32IM

- [`axvm-circuit`](../../vm/src/extensions/rv32im): VM extension for RV32IM is currently in the main crate but will be refactored into a standalone crate.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler/src/rrs.rs): Transpiler of RV32IM instructions is in the main transpiler crate but will be refactored out.

#### Recursion

- [`axvm-circuit`](../../vm/): VM extension for native kernel instructions is currently in the main crate but will be refactored into a standalone crate.
- [`axvm-native-compiler`](../../toolchain/native-compiler/): Implementation of compiler from a Rust embedded DSL to axVM assembly. The eDSL only targets the native kernel extension. The eDSL also has a static mode to support compilation to a Halo2 circuit.
- [`axvm-recursion`](../../lib/recursion): Library written in the native eDSL with functions to verify arbitrary STARK proofs. Library supports compilation to Halo2 circuit.

#### Keccak256

- [`axvm-circuit`](../../vm/): To be refactored into a standalone crate.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler/): To be refactored into a standalone crate.
- [`axvm`](../../toolchain/riscv/axvm/): Intrinsic function for `keccak256` to be refactored into a standalone crate.

#### Big Integers

- [`axvm-circuit`](../../vm/): VM extension for `I256, U256` to be refactored into a standalone crate.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler/): To be refactored into a standalone crate.
- [`axvm`](../../toolchain/riscv/axvm/): `I256, U256` struct implementations using intrinsics for underlying operations to be refactored into a standalone crate.

#### Modular Arithmetic

- [`axvm-circuit`](../../vm/): VM extension for modular arithmetic for arbitrary compile-time modulus, to be refactored into a standalone crate.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler/): To be refactored into a standalone crate.
- [`axvm-algebra`](../../lib/algebra): Guest library with traits for algebra (e.g., modular arithmetic, field extension).
- [`axvm-moduli-setup`](../../toolchain/riscv/toolchain/macros/moduli-setup): Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.

#### Elliptic Curve Cryptography

- [`ax-ecc-primitives`](../../circuits/ecc): VM extension for Weierstrass elliptic curve operations for arbitrary compile-time curve, VM extension for BN254 and BLS12-381 pairing operations, to be refactored.
- [`ax-ecc-execution`](../../lib/ecc-execution): Elliptic curve operations for use in VM runtime execution.
- [`axvm-transpiler`](../../toolchain/riscv/transpiler/): To be refactored into a standalone crate.
- [`axvm-ecc`](../../lib/ecc): Guest library with elliptic curve functions using custom intrinsics. Includes ECDSA and pairing.
- [`axvm-sw-setup`](../../toolchain/riscv/macros/sw-setup): Procedural macros for use in guest program to generate short Weierstrass curve struct with custom intrinsics for compile-time curve.

### Circuit Foundations

- [`ax-circuit-primitives`](../../circuits/primitives): Primitive chips and sub-chips for standalone use in any circuit.
- [`ax-circuit-derive`](../../circuits/derive): Procedural macros for use in circuit to derive traits.
- [`ax-poseidon2-air`](../../circuits/hashes/poseidon2-air): Standalone poseidon2 AIR implementation.
- [`ax-ecc-primitives`](../../circuits/ecc): General builder for generating chip for any modular arithmetic expression for a compile-time modulus. To be refactored.

### Proof System

- [`ax-stark-backend`](../../stark-backend): General purpose STARK proving system with multi-trace and logup support, built on top of plonky3.
- [`ax-stark-sdk`](../../stark-sdk): Low-level SDK for use with STARK backend to generate proofs for specific STARK configurations.
