# Acceleration Using Pre-Built Extensions

OpenVM ships with a set of pre-built extensions maintained by the OpenVM team. Below, we highlight six of these extensions designed to accelerate common arithmetic and cryptographic operations that are notoriously expensive to execute. Some of these extensions have corresponding guest libraries which provide convenient, high-level interfaces for your guest program to interact with the extension.

- [`openvm-keccak-guest`](./keccak.md) - Keccak256 hash function. See the [Keccak256 guest library](../guest-libs/keccak256.md) for usage details.
- [`openvm-sha256-guest`](./sha256.md) - SHA-256 hash function. See the [SHA-2 guest library](../guest-libs/sha2.md) for usage details.
- [`openvm-bigint-guest`](./bigint.md) - Big integer arithmetic for 256-bit signed and unsigned integers. See the [ruint guest library](../guest-libs/ruint.md) for using accelerated 256-bit integer ops in rust.
- [`openvm-algebra-guest`](./algebra.md) - Modular arithmetic and complex field extensions.
- [`openvm-ecc-guest`](./ecc.md) - Elliptic curve cryptography. See the [k256](../guest-libs/k256.md) and [p256](../guest-libs/p256.md) guest libraries for using this extension over the respective curves.
- [`openvm-pairing-guest`](./pairing.md) - Elliptic curve optimal Ate pairings. See the [pairing guest library](../guest-libs/pairing.md) for usage details.

## Optimizing Modular Arithmetic

Some of these extensions—specifically `algebra`, `ecc`, and `pairing`—perform modular arithmetic, which can be significantly optimized when the modulus is known at compile time.  Therefore, these extensions provide a framework to inform the compiler about all the moduli and associated arithmetic structures we intend to use. To achieve this, two steps are involved:

1. **Declare**: Introduce a modular arithmetic or related structure, along with its modulus and functionality. This can be done in any library or binary file.
2. **Init**: Performed exactly once in the final binary. It aggregates all previously declared structures, assigns them stable indices, and sets up linkage so that they can be referenced in generated code.

These steps ensure both performance and security: performance because the modulus is known at compile time, and security because runtime checks confirm that the correct structures have been initialized.

Our design for the configuration procedure above was inspired by the [EVMMAX proposal](https://github.com/jwasinger/EIPs/blob/evmmax-2/EIPS/eip-6601.md).

### Automating the `init!` step

The `openvm` crate provides an `init!` macro to automate the **init** step:
1. Call `openvm::init!()` exactly once in the code of the final program binary.
2. When [compiling the program](../writing-apps/build.md), `cargo openvm build` will read the [configuration file](#configuration) to automatically generate the correct init code and write it to `<INIT_FILE_NAME>`, which defaults to `openvm_init.rs` in the manifest directory.
3. The `openvm::init!()` macro will include the `openvm_init.rs` file into the final binary to complete the init process. You can call `openvm::init!(INIT_FILE_NAME)` to include init code from a different file if needed.

## Configuration

To use these extensions, you must populate an `openvm.toml` in your package root directory (where the `Cargo.toml` file is located).
We will explain in each extension how to configure the `openvm.toml` file.

A template `openvm.toml` file using the default VM extensions shipping with OpenVM is as follows:

```toml
[app_vm_config.rv32i]

[app_vm_config.rv32m]
range_tuple_checker_sizes = [256, 8192]

[app_vm_config.io]

[app_vm_config.keccak]

[app_vm_config.sha256]

[app_vm_config.native]

[app_vm_config.bigint]
range_tuple_checker_sizes = [256, 8192]

[app_vm_config.modular]
supported_moduli = ["<modulus_1>", "<modulus_2>", ...]

[app_vm_config.fp2]
supported_moduli = ["<modulus_1>", "<modulus_2>", ...]

[app_vm_config.pairing]
supported_curves = ["Bls12_381", "Bn254"]

[[app_vm_config.ecc.supported_curves]]
struct_name = "<curve_name_1>"
modulus = "<modulus_1>"
scalar = "<scalar_1>"
a = "<a_1>"
b = "<b_1>"

[[app_vm_config.ecc.supported_curves]]
struct_name = "<curve_name_2>"
modulus = "<modulus_2>"
scalar = "<scalar_2>"
a = "<a_2>"
b = "<b_2>"
```

`rv32i`, `io`, and `rv32m` need to be always included if you make an `openvm.toml` file while the rest are optional and should be included if you want to use the corresponding extension.
All moduli and scalars must be provided in decimal format. Currently `pairing` supports only pre-defined `Bls12_381` and `Bn254` curves. To add more `ecc` curves you need to add more `[[app_vm_config.ecc.supported_curves]]` entries.
