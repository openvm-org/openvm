<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Using already existing extensions](#using-already-existing-extensions)
  - [`openvm-algebra`](#openvm-algebra)
  - [`openvm-ecc`](#openvm-ecc)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Using already existing extensions

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## `openvm-algebra`

This crate allows one to create and use structs for convenient modular arithmetic operations, and also for their complex extensions (for example, if $p$ is a prime number, `openvm-algebra` provides methods for modular arithmetic in the field $\mathbb{F}_p[x]/(x^2 + 1)$).

To declare a modular arithmetic struct, one needs to use the `moduli_declare!` macro. A usage example is given below:

```rust
moduli_declare! {
    Bls12381 { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bn254 { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}
```

This creates two structs, `Bls12381` and `Bn254`, each representing the modular arithmetic class (implementing `Add`, `Sub` and so on). The modulus for each struct is specified in the `modulus` parameter of the macro. It should be a string literal in either decimal or hexadecimal format (in the latter case, it must start with `0x`).

The arithmetic operations for these classes, when compiling for the `zkvm` target, are converted into RISC-V asm instructions which are distinguished by the `funct7` field. The corresponding "distinguishers assignment" is happening when another macro is called:

```rust
moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "21888242871839275222246405745257275088696311157297823662689037894645226208583"
}
```

This macro **must be called exactly once** in the program, and it must contain all the moduli that have ever been declared in the `moduli_declare!` macros across all the compilation units. It is possible to `declare` a number in decimal and `init` it in hexadecimal, and vice versa.

When `moduli_init!` is called, the moduli in it are enumerated from `0`. For each chip that is used, the first instruction that this chip receives must be a `setup` instruction -- this adds a record to the trace that guarantees that the modulus this chip uses is exactly the one we `init`ed.

To send a setup instruction for the $i$-th struct, one needs to call the `setup_<i>()` function (for instance, `setup_1()`). There is also a function `setup_all_moduli()` that calls all the available `setup` functions.

## `openvm-ecc`

**TODO**: fill in.