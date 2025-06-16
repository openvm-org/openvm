# `openvm-ecc-sw-macros`

Procedural macros for use in guest program to generate short Weierstrass elliptic curve struct with custom intrinsics for compile-time modulus.

The workflow of this macro is very similar to the [`openvm-algebra-moduli-macros`](../../algebra/moduli-macros/README.md) crate. We recommend reading it first.

## Example

```rust
// ...

moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" },
    Secp256k1Scalar { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141" },
}

const CURVE_B: Secp256k1Coord = Secp256k1Coord::from_const_bytes(seven_le());

sw_declare! {
    Secp256k1Point { mod_type = Secp256k1Coord, b = CURVE_B },
}

openvm::init!();
/* The init! macro will expand to:
openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}

openvm_ecc_guest::sw_macros::sw_init! {
    Secp256k1Point,
}
*/

pub fn main() {
    // ...
}
```

## Full story

Again, the principle is the same as in the [`openvm-algebra-moduli-macros`](../../algebra/moduli-macros/README.md) crate. Here we emphasize the core differences.

The crate provides two macros: `sw_declare!` and `sw_init!`. The signatures are:

- `sw_declare!` receives comma-separated list of moduli classes descriptions. Each description looks like `SwStruct { mod_type = ModulusName, a = a_expr, b = b_expr }`. Here `ModulusName` is the name of a struct that implements `trait IntMod` -- in particular, the ones created by `moduli_declare!` do -- and has `NUM_LIMBS` divisible by 4. Parameters `a` and `b` correspond to the coefficients of the equation defining the curve. They **must be compile-time constants**. The parameter `a` may be omitted, in which case it defaults to `0` (or, more specifically, to `<ModulusName as IntMod>::ZERO`). The parameter `b` is required.

- `sw_init!` receives comma-separated list of struct names. The struct name must exactly match the name in `sw_declare!` -- type defs are not allowed (see point 5 below).

What happens under the hood:

1. `sw_declare!` macro creates a struct with two field `x` and `y` of type `mod_type`. This struct denotes a point on the corresponding elliptic curve. In the example it would be

```rust
struct Secp256k1Point {
    x: Secp256k1Coord,
    y: Secp256k1Coord,
}
```

Similar to `moduli_declare!`, this macro also creates extern functions for arithmetic operations -- but in this case they are named after the sw type, not after any hexadecimal (since the macro has no way to obtain it from the name of the modulus type anyway):

```rust
extern "C" {
    fn sw_add_extern_func_Secp256k1Point(rd: usize, rs1: usize, rs2: usize);
    fn sw_double_extern_func_Secp256k1Point(rd: usize, rs1: usize);
}
```

2. Again, `sw_init!` macro implements these extern functions and defines the setup functions for the sw struct.

```rust
#[allow(non_snake_case)]
#[cfg(target_os = "zkvm")]
mod openvm_intrinsics_ffi_2 {
    use :openvm_ecc_guest::{OPCODE, SW_FUNCT3, SwBaseFunct7};

    #[no_mangle]
    extern "C" fn sw_add_extern_func_Secp256k1Point(rd: usize, rs1: usize, rs2: usize) {
        // ...
    }
    // other externs

    #[no_mangle]
    extern "C" fn sw_setup_extern_func_Secp256k1Point() {
        #[cfg(target_os = "zkvm")]
        {
            // ...
        }
    }
}
```

3. Again, if using the Rust bindings, then the `sw_setup_extern_func_*` function for every curve is automatically called on first use of any of the curve's intrinsics.

4. The order of the items in `sw_init!` **must match** the order of the moduli in the chip configuration -- more specifically, in the modular extension parameters (the order of `CurveConfig`s in `WeierstrassExtension::supported_curves`, which is usually defined with the whole `app_vm_config` in the `openvm.toml` file).

5. Note that, due to the nature of function names, the name of the struct used in `sw_init!` must be the same as in `sw_declare!`. To illustrate, the following code will **fail** to compile:

```rust
// ...

sw_declare! {
    Secp256k1Point { mod_type = Secp256k1Coord, b = CURVE_B },
}

pub type Sw = Secp256k1Point;

sw_init! {
    Sw,
}
```

The reason is that, for example, the function `sw_add_extern_func_Secp256k1Point` remains unimplemented, but we implement `sw_add_extern_func_Sw`.

6. `cargo openvm build` will automatically generate a call to `sw_init!` based on `openvm.toml`.
Note that `openvm.toml` must contain the name of each struct created by `sw_declare!` as a string (in the example at the top of this document, its `"Secp256k1"`).
The SDK also supports this feature.
