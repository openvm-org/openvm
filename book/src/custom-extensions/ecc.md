# Elliptic Curve Cryptography

The OpenVM Elliptic Curve Cryptography Extension provides support for elliptic curve operations through the `openvm-ecc-guest` crate.

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
    Bn254G1Affine { mod_type = Bn254Fp, b = BN254_B },
}
```

Each declared curve must specify the `mod_type` (implementing `IntMod`) and a constant `b` for the Weierstrass curve equation \\(y^2 = x^3 + b\\).
This creates `Bls12_381G1Affine` and `Bn254G1Affine` structs which implement the `Group` and `WeierstrassPoint` traits. The underlying memory layout of the structs uses the memory layout of the `Bls12_381Fp` and `Bn254Fp` structs, respectively.

2. **Init**: Called once, it enumerates these curves and allows the compiler to produce optimized instructions:

```rust
sw_init! {
    Bls12_381Fp, Bn254Fp,
}
```

3. **Setup**: Similar to the moduli and complex extensions, runtime setup instructions ensure that the correct curve parameters are being used, guaranteeing secure operation.

**Summary**:

- `sw_declare!`: Declares elliptic curve structures.
- `sw_init!`: Initializes them once, linking them to the underlying moduli.
- `setup_sw_<i>()`/`setup_all_curves()`: Secures runtime correctness.

To use elliptic curve operations on a struct defined with `sw_declare!`, it is expected that the struct for the curve's coordinate field was defined using `moduli_declare!`. In particular, the coordinate field needs to be initialized and set up as described in the [algebra extension](./algebra.md) chapter.

For the basic operations provided by the `WeierstrassPoint` trait, the scalar field is not needed. For the ECDSA functions in the `ecdsa` module, the scalar field must also be declared, initialized, and set up.

## Example program

See a working example [here](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/tests/programs/examples/ec.rs).

To use the ECC extension, add the following dependencies to `Cargo.toml`:

```toml
openvm-algebra-guest = { git = "https://github.com/openvm-org/openvm.git" }
openvm-ecc-guest = { git = "https://github.com/openvm-org/openvm.git", features = ["k256"] }
```

One can define their own ECC structs but we will use the Secp256k1 struct from `openvm-ecc-guest` and thus the `k256` feature should be enabled.

```rust
use openvm_ecc_guest::{
    k256::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar},
    Group, weierstrass::WeierstrassPoint,
};

openvm_algebra_guest::moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}

openvm_ecc_guest::sw_setup::sw_init! {
    Secp256k1Coord,
}
```

We `moduli_init!` both the coordinate and scalar field because they were declared in the `k256` module, although we will not be using the scalar field below.

With the above we can start doing elliptic curve operations like adding points:

```rust
pub fn main() {
    setup_all_moduli();
    setup_all_curves();
    let x1 = Secp256k1Coord::from_u32(1);
    let y1 = Secp256k1Coord::from_le_bytes(&hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let p1 = Secp256k1Point::from_xy_nonidentity(x1, y1).unwrap();

    let x2 = Secp256k1Coord::from_u32(2);
    let y2 = Secp256k1Coord::from_le_bytes(&hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    let p2 = Secp256k1Point::from_xy_nonidentity(x2, y2).unwrap();

    let p3 = &p1 + &p2;
}
```

### Config parameters

For the guest program to build successfully, all used moduli and curves must be declared in the `.toml` config file in the following format:

```toml
[app_vm_config.modular]
supported_modulus = ["115792089237316195423570985008687907853269984665640564039457584007908834671663", "115792089237316195423570985008687907852837564279074904382605163141518161494337"]

[[app_vm_config.ecc.supported_curves]]
modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"
scalar = "115792089237316195423570985008687907852837564279074904382605163141518161494337"
a = "0"
b = "7"
```

The `supported_modulus` parameter is a list of moduli that the guest program will use. The `ecc.supported_curves` parameter is a list of supported curves that the guest program will use. They must be provided in decimal format in the `.toml` file. For multiple curves create multiple `[[app_vm_config.ecc.supported_curves]]` sections.
