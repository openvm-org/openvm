# `openvm-algebra-complex-macros`

Procedural macros for use in guest program to generate modular arithmetic struct with custom intrinsics for compile-time modulus.

The workflow of this macro is very similar to the [`openvm-algebra-moduli-macros`](../moduli-macros/README.md) crate. We recommend reading it first.

## Example

```rust
openvm_algebra_moduli_macros::moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" }
}
openvm_algebra_moduli_macros::moduli_init!(
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
);

openvm_algebra_complex_macros::complex_declare! {
    Complex { mod_type = Secp256k1Coord }
}

openvm_algebra_complex_macros::complex_init! {
    Complex { mod_idx = 0 },
}

pub fn main() {
    setup_all_moduli();
    setup_all_complex_extensions();
    // ...
}
```

## Full story

Again, the principle is the same as in the [`openvm-algebra-moduli-macros`](../moduli-macros/README.md) crate. Here we emphasize the core differences.

The crate provides two macros: `complex_declare!` and `complex_init!`. The signatures are:

- `complex_declare!` receives comma-separated list of moduli classes descriptions. Each description looks like `ComplexStruct { mod_type = ModulusName }`. Here `ModulusName` is the name of any struct that implements `trait IntMod` -- in particular, the ones created by `moduli_declare!` do, and `ComplexStruct` is the name for the complex arithmetic struct to create.

- `complex_init!` receives comma-separated list of struct descriptions. Each description looks like `ComplexStruct { mod_idx = idx }`. Here `ComplexStruct` is the name of the complex struct used in `complex_declare!`, and `idx` is the index of the modulus **in the `moduli_init!` macro**.

What happens under the hood:

1. `complex_declare!` macro creates a struct with two field `c0` and `c1` of type `mod_type`. In the example it would be

```rust
struct Complex {
    c0: Secp256k1Coord,
    c1: Secp256k1Coord,
}
```

Similar to `moduli_declare!`, this macro also creates extern functions for arithmetic operations -- but in this case they are named after the complex type, not after any hexadecimal (since the macro has no way to obtain it from the name of the modulus type anyway):

```rust
extern "C" {
    fn complex_add_extern_func_Complex(rd: usize, rs1: usize, rs2: usize);
    fn complex_sub_extern_func_Complex(rd: usize, rs1: usize, rs2: usize);
    fn complex_mul_extern_func_Complex(rd: usize, rs1: usize, rs2: usize);
    fn complex_div_extern_func_Complex(rd: usize, rs1: usize, rs2: usize);
}
```

2. Again, `complex_init!` macro implements these extern functions and defines the setup functions for the complex arithmetic struct.

```rust
#[cfg(target_os = "zkvm")]
mod openvm_intrinsics_ffi_complex {
    fn complex_add_extern_func_Complex(rd: usize, rs1: usize, rs2: usize) {
        // send the instructions for the corresponding complex chip
        // If this struct was `init`ed k-th, these operations will be sent to the k-th complex chip
    }
    // implement the other functions
}
pub fn setup_complex_0() {
    // send the setup instructions
}
pub fn setup_all_complex_extensions() {
    setup_complex_0();
    // call all other setup_complex_* for all the items in the moduli_init! macro
}
```

3. Obviously, `mod_idx` in the `complex_init!` must match the position of the corresponding modulus in the `moduli_init!` macro. The order of the items in `complex_init!` affects what `setup_complex_*` function will correspond to what complex class. Also, it **must match** the order of the moduli in the chip configuration -- more specifically, in the modular extension parameters (the order of numbers in `Fp2Extension::supported_modulus`, which is usually defined with the whole `app_vm_config` in the `openvm.toml` file). However, it again imposes the restriction that we only can invoke `complex_init!` once. Again analogous to the moduli setups, we must call `setup_complex_*` for each used complex extension before doing anything with entities of that class (or one can call `setup_all_complex_extensions` to setup all of them, if all are used).

4. Note that, due to the nature of function names, the name of the struct used in `complex_init!` must be the same as in `complex_declare!`. To illustrate, the following code will **fail** to compile:

```rust
// moduli related macros...

complex_declare! {
    Bn254Fp2 { mod_type = Bn254Fp },
}

pub type Fp2 = Bn254Fp2;

complex_init! {
    Fp2 { mod_idx = 0 },
}
```

The reason is that, for example, the function `complex_add_extern_func_Bn254Fp2` remains unimplemented, but we implement `complex_add_extern_func_Fp2` instead.
