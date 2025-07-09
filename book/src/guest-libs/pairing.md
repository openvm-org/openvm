# Elliptic Curve Pairing

We'll be working with an example using the BLS12-381 elliptic curve. This is in addition to the setup that needs to be done in the [Writing a Program](../writing-apps/write-program.md) section.

In the guest program, we will import the `PairingCheck` and `IntMod` traits, along with the BLS12-381 curve structs (**IMPORTANT:** this requires the `bls12_381` feature enabled in Cargo.toml for the `openvm-pairing` dependency), and a few other values that we will need:

```rust,no_run,noplayground title="guest program"
{{ #include ../../../examples/pairing/src/main.rs:imports }}
```

Additionally, we'll need to initialize our moduli and `Fp2` struct via the following macros. For a more in-depth description of these macros, please see the [OpenVM Algebra](./algebra.md) section.

```rust,no_run,noplayground
{{ #include ../../../examples/pairing/src/main.rs:init }}
```

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

```rust,no_run,noplayground
{{ #include ../../../examples/pairing/src/main.rs:pairing_check }}
```

## Additional functionality

We also have access to each of the specific functions that the pairing check utilizes for either the BN254 or BLS12-381 elliptic curves.

### Multi-Miller loop

The multi-Miller loop requires the MultiMillerLoop trait can also be run separately via:

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
[app_vm_config.rv32i]
[app_vm_config.rv32m]
[app_vm_config.io]
[app_vm_config.pairing]
supported_curves = ["Bls12_381"]

[app_vm_config.modular]
supported_moduli = [
    "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
]

[app_vm_config.fp2]
supported_moduli = [
    ["Bls12_381Fp2", "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"],
]
```

Also note that since this is a complicated computation, the `keygen` step requires quite a lot of memory. Run it with `RUST_MIN_STACK` set to a large value, e.g.

```bash
RUST_MIN_STACK=8388608 cargo openvm keygen
```

### Full example program

This [example code](https://github.com/openvm-org/openvm/blob/main/examples/pairing/src/main.rs) contains hardcoded values and no inputs as an example that can be run via the CLI.

```rust,no_run,noplayground
{{ #include ../../../examples/pairing/src/main.rs:pre }}
{{ #include ../../../examples/pairing/src/main.rs:imports }}

{{ #include ../../../examples/pairing/src/main.rs:init }}

{{ #include ../../../examples/pairing/src/main.rs:main }}
```
