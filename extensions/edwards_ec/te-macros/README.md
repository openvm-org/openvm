# `openvm-te-macros`

Procedural macros for use in guest program to generate short twisted Edwards elliptic curve struct with custom intrinsics for compile-time modulus.

The workflow of this macro is very similar to the [`openvm-algebra-moduli-macros`](../../algebra/moduli-macros/README.md) crate. We recommend reading it first.

## Example

```rust
// ...

moduli_declare! {
    Ed25519Coord { modulus = "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED" },
    Ed25519Scalar { modulus = "0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED" },
}

// Note that from_const_bytes is little endian
pub const CURVE_A: Ed25519Coord = Ed25519Coord::from_const_bytes(hex!(
    "ECFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7F"
));
pub const CURVE_D: Ed25519Coord = Ed25519Coord::from_const_bytes(hex!(
    "A3785913CA4DEB75ABD841414D0A700098E879777940C78C73FE6F2BEE6C0352"
));

sw_declare! {
    Ed25519Point { mod_type = Ed25519Coord, a = CURVE_A, d = CURVE_D },
}

openvm_algebra_guest::moduli_macros::moduli_init! {
    "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED",
    "0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED",
}

openvm_weierstrass_guest::te_macros::te_init! {
    Ed25519Point,
}

pub fn main() {
    setup_all_moduli();
    setup_all_te_curves();
    // ...
}
```

## Full story

Again, the principle is the same as in the [`openvm-algebra-moduli-macros`](../moduli-macros/README.md) crate. Here we emphasize the core differences.

The crate provides two macros: `te_declare!` and `te_init!`. The signatures are:

- `te_declare!` receives comma-separated list of moduli classes descriptions. Each description looks like `TeStruct { mod_type = ModulusName, a = a_expr, d = d_expr }`. Here `ModulusName` is the name of any struct that implements `trait IntMod` -- in particular, the ones created by `moduli_declare!` do. Parameters `a` and `d` correspond to the coefficients of the equation defining the curve. They **must be compile-time constants**. Both the parameters `a` and `d` are required.

- `te_init!` receives comma-separated list of struct names. The struct name must exactly match the name in `te_declare!` -- type defs are not allowed (see point 5 below).

What happens under the hood:

1. `te_declare!` macro creates a struct with two field `x` and `y` of type `mod_type`. This struct denotes a point on the corresponding elliptic curve. In the example it would be

```rust
struct Ed25519Point {
    x: Ed25519Coord,
    y: Ed25519Coord,
}
```

Similar to `moduli_declare!`, this macro also creates extern functions for arithmetic operations -- but in this case they are named after the te type, not after any hexadecimal (since the macro has no way to obtain it from the name of the modulus type anyway):

```rust
extern "C" {
    fn te_add_extern_func_Ed25519Point(rd: usize, rs1: usize, rs2: usize);
    fn hint_decompress_extern_func_Ed25519Point(rs1: usize, rs2: usize);
}
```

2. Again, `te_init!` macro implements these extern functions and defines the setup functions for the te struct.

```rust
#[cfg(target_os = "zkvm")]
mod openvm_intrinsics_ffi_2 {
    use :openvm_weierstrass_guest::{OPCODE, TE_FUNCT3, TeBaseFunct7};

    #[no_mangle]
    extern "C" fn te_add_extern_func_Ed25519Point(rd: usize, rs1: usize, rs2: usize) {
        // ...
    }
    // other externs
}
#[allow(non_snake_case)]
pub fn setup_te_Ed25519Point() {
    #[cfg(target_os = "zkvm")]
    {
        // ...
    }
}
pub fn setup_all_te_curves() {
    setup_te_Ed25519Point();
    // other setups
}
```

3. Again, if using the Rust bindings, then the `te_setup_extern_func_*` function for every curve is automatically called on first use of any of the curve's intrinsics.

4. The order of the items in `te_init!` **must match** the order of the moduli in the chip configuration -- more specifically, in the modular extension parameters (the order of `CurveConfig`s in `EccExtension::supported_te_curves`, which is usually defined with the whole `app_vm_config` in the `openvm.toml` file).

5. Note that, due to the nature of function names, the name of the struct used in `te_init!` must be the same as in `te_declare!`. To illustrate, the following code will **fail** to compile:

```rust
// ...

te_declare! {
    Ed25519Point { mod_type = Ed25519Coord, a = CURVE_A, d = CURVE_D },
}

pub type Te = Ed25519Point;

te_init! {
    Te,
}
```

The reason is that, for example, the function `sw_add_extern_func_Secp256k1Point` remains unimplemented, but we implement `sw_add_extern_func_Sw`.

6. `cargo openvm build` will automatically generate a call to `te_init!` based on `openvm.toml`.
Note that `openvm.toml` must contain the name of each struct created by `te_declare!` as a string (in the example at the top of this document, its `"Ed25519Point"`).
The SDK also supports this feature.
