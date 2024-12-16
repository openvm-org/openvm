# Elliptic Curve Pairing

The pairing extension enables usage of the optimal Ate pairing check on the BN254 and BLS12-381 elliptic curves. The following field extension tower for $\mathbb{F}_{p^{12}}$ is used for pairings in this crate:

$$
\mathbb{F_{p^2}} = \mathbb{F_{p}}[u]/(u^2 - \beta)\\
\mathbb{F_{p^6}} = \mathbb{F_{p^2}}[v]/(v^3 - \xi)\\
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

Additionally, we'll need to initialize our moduli and `Fp2` struct via the following macros. For a more in-depth description of these macros, please see the [Customizable Extensions](./customizable-extensions.md) section.

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

There are two moduli defined internally in the Bls12_381 feature. The `moduli_init!` macro thus requires both of them to be initialized. However, we do not need the scalar field of BLS12-381 (which is at index 1), and thus we only initialize the modulus from index 0, thus we only use `setup_0()` (as opposed to `setup_all_moduli()`, which will save us some columns when generating the trace).

## Input values

The inputs to the pairing check are `AffinePoint`s in $\mathbb{F}_p$ and $\mathbb{F}_{p^2}$. They can be constructed via the `AffinePoint::new` function, with the inner `Fp` and `Fp2` values constructed via various `from_...` functions.

We can create a new struct outside of our guest program that will hold these `AffinePoint`s and import it into our guest program:

```rust
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PairingCheckInput {
    p0: AffinePoint<Fp>,
    p1: AffinePoint<Fp2>,
    q0: AffinePoint<Fp>,
    q1: AffinePoint<Fp2>,
}
```

Our guest program imports the struct and can read it in via:

```rust
let io: PairingCheckInput = read();
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

### Final exponentiation

Final exponentiation can be run separately by importing the `FinalExp` trait.

```rust
use openvm_pairing_guest::pairing::FinalExp;
let (c, s) = Bls12_381::assert_final_exp_is_one(
    &f,
    &[p0, p1],
    &[q0, q1],
);
```

Where $c$ is the residue witness and $s$ is the scaling factor (BLS12-381) or cubic non-residue power (BN254), and the input $f$ is the result of the multi-Miller loop. The `p0`, `p1`, `q0`, `q1` points are the same as those used in the multi_miller_loop function.

## CLI

### Configuration

In order to run the pairing check in CLI, we'll need to create an `openvm.toml` configuration file somewhere. Its contents contains all of the necessary configuration information for enabling the OpenVM components that are used in the pairing check.

```toml
# openvm.toml
[app_vm_config.pairing]
supported_curves = ["Bls12_381"]

[app_vm_config.weierstrass]

[app_vm_config.modular]
supported_modulus = [
    "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
]

[app_vm_config.fp2]
supported_modulus = [
    "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
]
```

### Inputs

### Prove

Proving can be run via this CLI command.

```bash

```

### Verify
