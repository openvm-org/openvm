# Elliptic Curve Pairing

The pairing extension enables usage of the optimal Ate pairing check on the BN254 and BLS12-381 elliptic curves. The following field extension tower for \\(\mathbb{F}\_{p^{12}}\\) is used for pairings in this crate:

$$
\mathbb{F_{p^2}} = \mathbb{F_{p}}[u]/(u^2 - \beta)\\\\
\mathbb{F_{p^6}} = \mathbb{F_{p^2}}[v]/(v^3 - \xi)\\\\
\mathbb{F_{p^{12}}} = \mathbb{F_{p^6}}[w]/(w^2 - v)
$$

A full guest program example is available here: [pairing_check.rs](https://github.com/openvm-org/openvm/blob/c19c9ac60b135bb0f38fc997df5eb149db8144b4/crates/toolchain/tests/programs/examples/pairing_check.rs)

## Guest program setup

We'll be working with an example using the BLS12-381 elliptic curve. This is in addition to the setup that needs to be done in the [Writing a Program](../writing-apps/write-program.md) section.

In the guest program, we will import the `PairingCheck` and `IntMod` traits, along with the BLS12-381 curve structs (**IMPORTANT:** this requires the `bls12_381` feature enabled in Cargo.toml for the `openvm-pairing-guest` dependency), and a few other values that we will need:

```rust title="guest program"
use openvm_pairing_guest::{
    pairing::PairingCheck,
    bls12_381::{Bls12_381, Fp, Fp2},
};
use openvm_ecc_guest::AffinePoint;
use openvm_algebra_guest::IntMod;
use openvm::io::read;
```

Additionally, we'll need to initialize our moduli and `Fp2` struct via the following macros. For a more in-depth description of these macros, please see the [OpenVM Algebra](./algebra.md) section.

```rust
// These correspond to the BLS12-381 coordinate and scalar moduli, respectively
openvm_algebra_moduli_setup::moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}

openvm_algebra_complex_macros::complex_init! {
    Bls12_381Fp2 { mod_idx = 0 },
}
```

And we'll run the required setup functions at the top of the guest program's `main()` function:

```rust
setup_0();
setup_all_complex_extensions();
```

There are two moduli defined internally in the `Bls12_381` feature. The `moduli_init!` macro thus requires both of them to be initialized. However, we do not need the scalar field of BLS12-381 (which is at index 1), and thus we only initialize the modulus from index 0, thus we only use `setup_0()` (as opposed to `setup_all_moduli()`, which will save us some columns when generating the trace).

## Input values

The inputs to the pairing check are `AffinePoint`s in \\(\mathbb{F}\_p\\) and \\(\mathbb{F}\_{p^2}\\). They can be constructed via the `AffinePoint::new` function, with the inner `Fp` and `Fp2` values constructed via various `from_...` functions.

We can create a new struct to hold these `AffinePoint`s for the purpose of this guide. You may instead put them into a custom struct to serve your use case.

```rust
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PairingCheckInput {
    p0: AffinePoint<Fp>,
    p1: AffinePoint<Fp2>,
    q0: AffinePoint<Fp>,
    q1: AffinePoint<Fp2>,
}
```

## Pairing check

Most users that use the pairing extension will want to assert that a pairing is valid (the final exponentiation equals one). With the `PairingCheck` trait imported from the previous section, we have access to the `pairing_check` function on the `Bls12_381` struct. After reading in the input struct, we can use its values in the `pairing_check`:

```rust
let res = Bls12_381::pairing_check(
    &[p0, p1],
    &[q0, q1],
);
assert!(res.is_ok())
```

## Additional functionality

We also have access to each of the specific functions that the pairing check utilizes for either the BN254 or BLS12-381 elliptic curves.

### Multi-Miller loop

The multi-Miller loop requires the MultiMillerLoop trait can also be ran separately via:

```rust
let f = Bls12_381::multi_miller_loop(
    &[p0, p1],
    &[q0, q1],
);
```

## Running via CLI

### Config parameters

For the guest program to build successfully, we'll need to create an `openvm.toml` configuration file somewhere. It contains all of the necessary configuration information for enabling the OpenVM components that are used in the pairing check.

```toml
# openvm.toml
[app_vm_config.pairing]
supported_curves = ["Bls12_381"]

[app_vm_config.modular]
supported_modulus = [
    "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
]

[app_vm_config.fp2]
supported_modulus = [
    "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
]
```

Also note that since this is a complicated computation, the `keygen` step requires quite a lot of memory. Run it with `RUST_MIN_STACK` set to a large value, e.g.
```bash
RUST_MIN_STACK=8388608 cargo openvm keygen
```

### Full example code

This example code contains hardcoded values and no inputs as an example that can be run via the CLI.

```rust
#![no_main]
#![no_std]

use hex_literal::hex;
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::{
    bls12_381::{Bls12_381, Fp, Fp2},
    pairing::PairingCheck,
};

openvm::entry!(main);

openvm_algebra_moduli_setup::moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}

openvm_algebra_complex_macros::complex_init! {
    Bls12_381Fp2 { mod_idx = 0 },
}

pub fn main() {
    setup_0();
    setup_all_complex_extensions();

    let p0 = AffinePoint::new(
        Fp::from_be_bytes(&hex!("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb")),
        Fp::from_be_bytes(&hex!("08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"))
    );
    let p1 = AffinePoint::new(
        Fp2::from_coeffs([
            Fp::from_be_bytes(&hex!("1638533957d540a9d2370f17cc7ed5863bc0b995b8825e0ee1ea1e1e4d00dbae81f14b0bf3611b78c952aacab827a053")),
            Fp::from_be_bytes(&hex!("0a4edef9c1ed7f729f520e47730a124fd70662a904ba1074728114d1031e1572c6c886f6b57ec72a6178288c47c33577"))
        ]),
        Fp2::from_coeffs([
            Fp::from_be_bytes(&hex!("0468fb440d82b0630aeb8dca2b5256789a66da69bf91009cbfe6bd221e47aa8ae88dece9764bf3bd999d95d71e4c9899")),
            Fp::from_be_bytes(&hex!("0f6d4552fa65dd2638b361543f887136a43253d9c66c411697003f7a13c308f5422e1aa0a59c8967acdefd8b6e36ccf3"))
        ]),
    );
    let q0 = AffinePoint::new(
        Fp::from_be_bytes(&hex!("0572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4e")),
        Fp::from_be_bytes(&hex!("166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28"))
    );
    let q1 = AffinePoint::new(
        Fp2::from_coeffs([
            Fp::from_be_bytes(&hex!("024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8")),
            Fp::from_be_bytes(&hex!("13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e"))
        ]),
        Fp2::from_coeffs([
            Fp::from_be_bytes(&hex!("0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801")),
            Fp::from_be_bytes(&hex!("0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be"))
        ]),
    );

    let res = Bls12_381::pairing_check(&[p0, -q0], &[p1, q1]);
    assert!(res.is_ok());
}
```
