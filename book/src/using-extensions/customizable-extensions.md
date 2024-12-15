# Using already existing extensions

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## `openvm-algebra`

This crate allows one to create and use structs for convenient modular arithmetic operations, and also for their complex extensions (for example, if $p$ is a prime number, `openvm-algebra` provides methods for modular arithmetic in the field $\mathbb{F}_p[x]/(x^2 + 1)$).

### Available traits and methods

- `IntMod` trait: contains type `Repr`, constants `MODULUS`, `NUM_LIMBS`, `ZERO`, `ONE`, and basic methods for constructing an object and arithmetic operations. `Repr` is usually `[u8; NUM_LIMBS]` and indicates the underlying representation of the number. `MODULUS: Repr` is the modulus of the struct, `ZERO` and `ONE` are the additive and multiplicative identities (both are of type `Repr`). To construct a struct, methods `from_repr`, `from_le_bytes`, `from_be_bytes`, `from_u8`, `from_u32`, `from_u64` are available.

- `Field` trait: contains constants `ZERO` and `ONE`, and methods for basic arithmetic operations.

<!-- TODO: FieldExtension trait -->

<!-- TODO: exp_bytes is only intended for host? -->

### Modular arithmetic

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

### Complex field extension

To declare a complex field extension struct, one needs to use the `complex_declare!` macro. A usage example is given below:

```rust
complex_declare! {
    Bn254_Fp2 { mod_type = Bn254_Fp }
}
```

This creates a struct `Bn254_Fp2`, which represents the complex field extension class. The `mod_type` parameter must be a struct that implements the `IntMod` trait.

The arithmetic operations for these classes, when compiling for the `zkvm` target, are converted into RISC-V asm instructions which are distinguished by the `funct7` field. The corresponding "distinguishers assignment" is happening when another macro is called:

```rust
complex_init! {
    Bn254_Fp2 { mod_idx = 0 },
}
```

This macro **must be called exactly once** in the final executable program, and it must contain all the moduli that have ever been declared in the `complex_declare!` macros across all the compilation units. This macro must be called after `moduli_init!`, and `mod_idx` must be the index of the modulus in the `moduli_init!` macro (and is unrelated to the order of `moduli_declare!` invocations or the modular structs in them).

When `complex_init!` is called, the structs in it are enumerated from `0`. For each chip that is used, the first instruction that this chip receives must be a `setup` instruction -- this adds a record to the trace that guarantees that the modulus this chip uses is exactly the one we `init`ed.

To send a setup instruction for the $i$-th struct, one needs to call the `setup_complex_<i>()` function (for instance, `setup_complex_1()`). There is also a function `setup_all_complex_extensions()` that calls all the available `setup` functions.

To summarize:

- `complex_declare!` declares a struct for a complex field extension class. It can be called multiple times across the compilation units.
- `complex_init!` initializes the data required for transpiling the program into the RISC-V assembly. **Every struct ever `declare`d in the program must be among the arguments of `complex_init!`**.
- `setup_complex_<i>()` sends a setup instruction for the $i$-th struct. Here, **$i$-th struct is the one that corresponds to the $i$-th modulus in `complex_init!`**. The order of `complex_declare!` invocations or the arguments in them does not matter.
- `setup_all_complex_extensions()` sends setup instructions for all the structs.

### A toy example of a guest program using `openvm-algebra` extension

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;

openvm::entry!(main);

// This macro will create two structs, `Mod1` and `Mod2`,
// one for arithmetic modulo 998244353, and the other for arithmetic modulo 1000000007.
openvm_algebra_moduli_setup::moduli_declare! {
    Mod1 { modulus = "998244353" },
    Mod2 { modulus = "1000000007" }
}

// This macro will initialize the moduli.
// Now, `Mod1` is the "zeroth" modular struct, and `Mod2` is the "first" one.
openvm_algebra_moduli_setup::moduli_init! {
    "998244353", "1000000007"
}

// This macro will create two structs, `Complex1` and `Complex2`,
// one for arithmetic in the field $\mathbb{F}_{998244353}[x]/(x^2 + 1)$,
// and the other for arithmetic in the field $\mathbb{F}_{1000000007}[x]/(x^2 + 1)$.
openvm_algebra_complex_macros::complex_declare! {
    Complex1 { mod_type = Mod1 },
    Complex2 { mod_type = Mod2 },
}

// The order of these structs does not matter,
// given that we specify the `mod_idx` parameters properly.
openvm_algebra_complex_macros::complex_init! {
    Complex2 { mod_idx = 1 }, Complex1 { mod_idx = 0 },
}

pub fn main() {
    setup_all_complex_extensions();
    let a = Complex1::new(Mod1::ZERO, Mod1::from_u32(0x3b8) * Mod1::from_u32(0x100000)); // a = -i in the corresponding field
    let b = Complex2::new(Mod2::ZERO, Mod2::from_u32(1000000006)); // b = -i in the corresponding field
    assert_eq!(a.clone() * &a * &a * &a * &a, a); // a^5 = a
    assert_eq!(b.clone() * &b * &b * &b * &b, b); // b^5 = b
    // Note that these assertions would fail, have we provided the `mod_idx` parameters wrongly.
}
```

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
