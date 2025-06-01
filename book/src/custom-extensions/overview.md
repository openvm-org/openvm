# Using Existing Extensions

You can seamlessly integrate certain performance-optimized extensions maintained by the OpenVM team to enhance your arithmetic operations and cryptographic computations.

In this chapter, we will explain how to use the following existing extensions:

- [`openvm-keccak-guest`](./keccak.md) - Keccak256 hash function.
- [`openvm-sha256-guest`](./sha256.md) - SHA2-256 hash function.
- [`openvm-bigint-guest`](./bigint.md) - Big integer arithmetic for 256-bit signed and unsigned integers.
- [`openvm-algebra-guest`](./algebra.md) - Modular arithmetic and complex field extensions.
- [`openvm-ecc-guest`](./ecc.md) - Elliptic curve cryptography.
- [`openvm-pairing-guest`](./pairing.md) - Elliptic curve optimal Ate pairings.

Some extensions such as `openvm-keccak-guest`, `openvm-sha256-guest`, and `openvm-bigint-guest` can be enabled without specifying any additional configuration.

On the other hand certain arithmetic operations, particularly modular arithmetic, can be optimized significantly when the modulus is known at compile time. This approach requires a framework to inform the compiler about all the moduli and associated arithmetic structures we intend to use. To achieve this, three steps are involved:

1. **Declare**: Introduce a modular arithmetic or related structure, along with its modulus and functionality. This can be done in any library or binary file.
2. **Init**: Performed exactly once in the final binary. It aggregates all previously declared structures, assigns them stable indices, and sets up linkage so that they can be referenced in generated code.
3. **Setup**: A one-time runtime procedure for security. This ensures that the compiled code matches the virtual machine’s expectations and that each instruction set is tied to the correct modulus or extension.

These steps ensure both performance and security: performance because the modulus is known at compile time, and security because runtime checks confirm that the correct structures have been initialized.

Our design for the configuration procedure above was inspired by the [EVMMAX proposal](https://github.com/jwasinger/EIPs/blob/evmmax-2/EIPS/eip-6601.md).

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
modulus = "<modulus_1>"
scalar = "<scalar_1>"
a = "<a_1>"
b = "<b_1>"

[[app_vm_config.ecc.supported_curves]]
modulus = "<modulus_2>"
scalar = "<scalar_2>"
a = "<a_2>"
b = "<b_2>"
```

`rv32i`, `io`, and `rv32m` need to be always included if you make an `openvm.toml` file while the rest are optional and should be included if you want to use the corresponding extension.
All moduli and scalars must be provided in decimal format. Currently `pairing` supports only pre-defined `Bls12_381` and `Bn254` curves. To add more `ecc` curves you need to add more `[[app_vm_config.ecc.supported_curves]]` entries.
