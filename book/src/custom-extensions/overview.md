# Using Existing Extensions

You can seamlessly integrate certain performance-optimized extensions maintained by the OpenVM team to enhance your arithmetic operations and cryptographic computations.

In this chapter, we will explain how to use the following existing extensions:

- [`openvm-keccak-guest`](./keccak.md) - Keccak256 hash function.
- [`openvm-bigint-guest`](./bigint.md) - Big integer arithmetic for 256-bit signed and unsigned integers.
- [`openvm-algebra-guest`](./algebra.md) - Modular arithmetic and complex field extensions.
- [`openvm-ecc-guest`](./ecc.md) - Elliptic curve cryptography.
- [`openvm-pairing-guest`](./pairing.md) - Elliptic curve optimal Ate pairings.

Some extensions such as `openvm-keccak-guest` and `openvm-bigint-guest` can be enabled without specifying any additional configuration.

On the other hand certain arithmetic operations, particularly modular arithmetic, can be optimized significantly when the modulus is known at compile time. This approach requires a framework to inform the compiler about all the moduli and associated arithmetic structures we intend to use. To achieve this, three steps are involved:

1. **Declare**: Introduce a modular arithmetic or related structure, along with its modulus and functionality. This can be done in any library or binary file.
2. **Init**: Performed exactly once in the final binary. It aggregates all previously declared structures, assigns them stable indices, and sets up linkage so that they can be referenced in generated code.
3. **Setup**: A one-time runtime procedure for security. This ensures that the compiled code matches the virtual machine’s expectations and that each instruction set is tied to the correct modulus or extension.

These steps ensure both performance and security: performance because the modulus is known at compile time, and security because runtime checks confirm that the correct structures have been initialized.

Our design for the configuration procedure above was inspired by the [EVMMAX proposal](https://github.com/jwasinger/EIPs/blob/evmmax-2/EIPS/eip-6601.md).

## Configuration

To use these extensions, you must populate a `openvm.toml` in your package root directory (where the `Cargo.toml` file is located).
We will explain in each extension how to configure the `openvm.toml` file.

The template `openvm.toml` file is as follows:

```toml
[app_vm_config.rv32i]
[app_vm_config.rv32m]
[app_vm_config.io]
# ...
```
