# The ECC Extension

The ECC extension provides support for elliptic curve operations. This guide will show you how to use it in your guest programs. See a working example [here](https://github.com/openvm-org/openvm/blob/c19c9ac60b135bb0f38fc997df5eb149db8144b4/crates/toolchain/tests/programs/examples/ec.rs).

## Setup

Steps to setup a guest program can be found [here](../writing-apps/write-program.md#guest-program-setup).
Additionally, to use the ECC extension, add the following dependencies to `Cargo.toml`:

```toml
openvm-algebra-moduli-setup = { git = "https://github.com/openvm-org/openvm.git" }
openvm-ecc-sw-setup = { git = "https://github.com/openvm-org/openvm.git" }
openvm-algebra-guest = { git = "https://github.com/openvm-org/openvm.git" }
openvm-ecc-guest = { git = "https://github.com/openvm-org/openvm.git" }
```

Previous [section](./customizable-extensions.md) explains how to use macros like `moduli_declare!` and `sw_declare!` to declare ECC structs and their moduli. One can define their own ECC structs but we will use the Secp256k1 struct from `openvm-ecc-guest`.

```rust
use openvm_ecc_guest::{
    k256::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar}
    Group,
};

openvm_algebra_moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}

openvm_ecc_sw_setup::sw_init! {
    Secp256k1Coord,
}
```

With the above we can start doing elliptic curve operations like adding points:
```rust
let x1 = Secp256k1Coord::from_u32(1);
let y1 = Secp256k1Coord::from_le_bytes(&hex!(
    "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
));
let p1 = Secp256k1Point { x: x1, y: y1 };

let x2 = Secp256k1Coord::from_u32(2);
let y2 = Secp256k1Coord::from_le_bytes(&hex!(
    "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
));
let p2 = Secp256k1Point { x: x2, y: y2 };

let p3 = &p1 + &p2;
```

Some additional functionalities like scalar multiplication and ECDSA are also supported.

## Run the program