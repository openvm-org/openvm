# Using already existing extensions

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## `openvm-algebra`

This crate allows one to create and use structs for convenient modular arithmetic operations, and also for their complex extensions (for example, if $p$ is a prime number, `openvm-algebra` provides methods for modular arithmetic in the field $\mathbb{F}_p[x]/(x^2 + 1)$).

To declare a modular arithmetic struct, one needs to use the `moduli_declare!` macro. A usage example is given below:

```rust
moduli_declare! {
    Bls12_381Fp { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bn254Fp { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}
```

This creates two structs, `Bls12381_Fp` and `Bn254_Fp`, each representing the modular arithmetic class. These classes implement `Add`, `Sub` and other basic arithmetic operations; the underlying functions used for this are a part of the `IntMod` trait. The modulus for each struct is specified in the `modulus` parameter of the macro. It should be a string literal in either decimal or hexadecimal format (in the latter case, it must start with `0x`).

The arithmetic operations for these classes, when compiling for the `zkvm` target, are converted into RISC-V asm instructions which are distinguished by the `funct7` field. The corresponding "distinguishers assignment" is happening when another macro is called:

```rust
moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "21888242871839275222246405745257275088696311157297823662689037894645226208583"
}
```

This macro **must be called exactly once** in the final executable program, and it must contain all the moduli that have ever been declared in the `moduli_declare!` macros across all the compilation units. It is possible to `declare` a number in decimal and `init` it in hexadecimal, and vice versa.

When `moduli_init!` is called, the moduli in it are enumerated from `0`. For each chip that is used, the first instruction that this chip receives must be a `setup` instruction -- this adds a record to the trace that guarantees that the modulus this chip uses is exactly the one we `init`ed.

To send a setup instruction for the $i$-th struct, one needs to call the `setup_<i>()` function (for instance, `setup_1()`). There is also a function `setup_all_moduli()` that calls all the available `setup` functions.

To summarize:

- `moduli_declare!` declares a struct for a modular arithmetic class. It can be called multiple times across the compilation units.
- `moduli_init!` initializes the data required for transpiling the program into the RISC-V assembly. **Every modulus ever `declare`d in the program must be among the arguments of `moduli_init!`**.
- `setup_<i>()` sends a setup instruction for the $i$-th struct. Here, **$i$-th struct is the one that corresponds to the $i$-th modulus in `moduli_init!`**. The order of `moduli_declare!` invocations or the arguments in them does not matter.
- `setup_all_moduli()` sends setup instructions for all the structs.

## `openvm-ecc`

This crate allows one to create and use structs for elliptic curve cryptography. More specifically, it only supports curves where the defining equation is in short [Weierstrass curves](https://en.wikipedia.org/wiki/Weierstrass_form) (that is, `a = 0`).

To declare an elliptic curve struct, one needs to use the `sw_declare!` macro. A usage example is given below:

```rust
sw_declare! {
    Bls12_381G1Affine { mod_type = Bls12_381Fp, b = BLS12_381_B },
    Bn254G1Affine { mod_type = Bn254Fp, b = BN254_B },
}
```

Similar to the `moduli_declare!` macro, the `sw_declare!` macro creates a struct for an elliptic curve. The `mod_type` parameter specifies the type of the modulus for this curve, and the `b` parameter specifies the free coefficient of the curve equation; both of these parameters are required. The `mod_type` parameter must be a struct that implements the `IntMod` trait. The `b` parameter must be a constant.

The arithmetic operations for these classes, when compiling for the `zkvm` target, are converted into RISC-V asm instructions which are distinguished by the `funct7` field. The corresponding "distinguishers assignment" is happening when another macro is called:

```rust
sw_init! {
    Bls12_381Fp, Bn254Fp,
}
```

Again, this macro **must be called exactly once** in the final executable program, and it must contain all the curves that have ever been declared in the `sw_declare!` macros across all the compilation units.

When `sw_init!` is called, the curves in it are enumerated from `0`. For each chip that is used, the first instruction that this chip receives must be a `setup` instruction -- this adds a record to the trace that guarantees that the curve this chip uses is exactly the one we `init`ed.

To send a setup instruction for the $i$-th struct, one needs to call the `setup_sw_<i>()` function (for instance, `setup_sw_1()`). There is also a function `setup_all_curves()` that calls all the available `setup` functions.

To summarize:

- `sw_declare!` declares a struct for an elliptic curve. It can be called multiple times across the compilation units.
- `sw_init!` initializes the data required for transpiling the program into the RISC-V assembly. **Every curve ever `declare`d in the program must be among the arguments of `sw_init!`**.
- `setup_sw_<i>()` sends a setup instruction for the $i$-th struct. Here, **$i$-th struct is the one that corresponds to the $i$-th curve in `sw_init!`**. The order of `sw_declare!` invocations or the arguments in them does not matter.
- `setup_all_curves()` sends setup instructions for all the structs.
