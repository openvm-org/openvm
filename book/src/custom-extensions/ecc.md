# Elliptic Curve Cryptography

The OpenVM Elliptic Curve Cryptography Extension provides support for elliptic curve operations through the `openvm-ecc-guest` crate.

The secp256k1 and secp256r1 curves are supported out of the box, and developers can enable arbitrary Weierstrass curves by configuring this extension with the modulus for the coordinate field and the coefficients in the curve equation.

## Available traits and methods

- `Group` trait:
  This represents an element of a [group](<https://en.wikipedia.org/wiki/Group_(mathematics)>) where the operation is addition. Therefore the trait includes functions for `add`, `sub`, and `double`.

  - `IDENTITY` is the identity element of the group.

- `CyclicGroup` trait:
  It's a group that has a generator, so it defines `GENERATOR` and `NEG_GENERATOR`.

- `WeierstrassPoint` trait:
  It represents an affine point on a Weierstrass elliptic curve and it extends `Group`.

  - `Coordinate` type is the type of the coordinates of the point, and it implements `IntMod`.
  - `x()`, `y()` are used to get the affine coordinates
  - `from_xy` is a constructor for the point, which checks if the point is either identity or on the affine curve.
  - The point supports elliptic curve operations through intrinsic functions `add_ne_nonidentity` and `double_nonidentity`.
  - `decompress`: Sometimes an elliptic curve point is compressed and represented by its `x` coordinate and the odd/even parity of the `y` coordinate. `decompress` is used to decompress the point back to `(x, y)`.

- `msm`: for multi-scalar multiplication.

- `ecdsa`: for doing ECDSA signature verification and public key recovery from signature.

## Macros

For elliptic curve cryptography, the `openvm-ecc-guest` crate provides macros similar to those in [`openvm-algebra-guest`](./algebra.md):

1. **Declare**: Use `sw_declare!` to define elliptic curves over the previously declared moduli. For example:

```rust
sw_declare! {
    Bls12_381G1Affine { mod_type = Bls12_381Fp, b = BLS12_381_B },
    P256Affine { mod_type = P256Coord, a = P256_A, b = P256_B },
}
```

Each declared curve must specify the `mod_type` (implementing `IntMod`) and a constant `b` for the Weierstrass curve equation \\(y^2 = x^3 + ax + b\\). `a` is optional and defaults to 0 for short Weierstrass curves.
This creates `Bls12_381G1Affine` and `P256Affine` structs which implement the `Group` and `WeierstrassPoint` traits. The underlying memory layout of the structs uses the memory layout of the `Bls12_381Fp` and `P256Coord` structs, respectively.

2. **Init**: Called once, the `init!` macro produces a call to `sw_init!` that enumerates these curves and allows the compiler to produce optimized instructions:

```rust
init!();
/* This expands to
sw_init! {
    Bls12_381G1Affine, P256Affine,
}
*/
```

**Summary**:

- `sw_declare!`: Declares elliptic curve structures.
- `init!`: Initializes them once, linking them to the underlying moduli.

To use elliptic curve operations on a struct defined with `sw_declare!`, it is expected that the struct for the curve's coordinate field was defined using `moduli_declare!`. In particular, the coordinate field needs to be initialized and set up as described in the [algebra extension](./algebra.md) chapter.

For the basic operations provided by the `WeierstrassPoint` trait, the scalar field is not needed. For the ECDSA functions in the `ecdsa` module, the scalar field must also be declared, initialized, and set up.

## ECDSA

The ECC extension supports ECDSA signature verification on any elliptic curve, and pre-defined implementations are provided for the secp256k1 and secp256r1 curves.
To verify an ECDSA signature, first call the `VerifyingKey::recover_from_prehash_noverify` associated function to recover the verifying key, then call the `VerifyingKey::verify_prehashed` method on the recovered verifying key.

## Example program

See a working example [here](https://github.com/openvm-org/openvm/blob/main/examples/ecc/src/main.rs).

To use the ECC extension, add the following dependencies to `Cargo.toml`:

```toml
openvm-algebra-guest = { git = "https://github.com/openvm-org/openvm.git" }
openvm-ecc-guest = { git = "https://github.com/openvm-org/openvm.git", features = ["k256"] }
```

One can define their own ECC structs but we will use the Secp256k1 struct from `openvm-ecc-guest` and thus the `k256` feature should be enabled.

```rust,no_run,noplayground
{{ #include ../../../examples/ecc/src/main.rs:imports }}
{{ #include ../../../examples/ecc/src/main.rs:init }}
```

`moduli_init!` is called for both the coordinate and scalar field because they were declared in the `k256` module, although we will not be using the scalar field below.

With the above we can start doing elliptic curve operations like adding points:

```rust,no_run,noplayground
{{ #include ../../../examples/ecc/src/main.rs:main }}
```

### Config parameters

For the guest program to build successfully, all used moduli and curves must be declared in the `.toml` config file in the following format:

```toml
[app_vm_config.modular]
supported_moduli = ["115792089237316195423570985008687907853269984665640564039457584007908834671663", "115792089237316195423570985008687907852837564279074904382605163141518161494337"]

[[app_vm_config.ecc.supported_curves]]
struct_name = "Secp256k1Point"
modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"
scalar = "115792089237316195423570985008687907852837564279074904382605163141518161494337"
a = "0"
b = "7"
```

The `supported_moduli` parameter is a list of moduli that the guest program will use. As mentioned in the [algebra extension](./algebra.md) chapter, the order of moduli in `[app_vm_config.modular]` must match the order in the `moduli_init!` macro.

The `ecc.supported_curves` parameter is a list of supported curves that the guest program will use. They must be provided in decimal format in the `.toml` file. For multiple curves create multiple `[[app_vm_config.ecc.supported_curves]]` sections. The order of curves in `[[app_vm_config.ecc.supported_curves]]` must match the order in the `sw_init!` macro.
Also, the `struct_name` field must be the name of the elliptic curve struct created by `sw_declare!`.
In this example, the `Secp256k1Point` struct is created in `openvm_ecc_guest::k256`.
