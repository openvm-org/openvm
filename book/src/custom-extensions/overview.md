# Using Existing Extensions

You can seamlessly integrate certain performance-optimized extensions maintained by the OpenVM team to enhance your arithmetic operations and cryptographic computations.

Certain arithmetic operations, particularly modular arithmetic, can be optimized significantly when the modulus is known at compile time.  This approach requires a framework to inform the compiler about all the moduli and associated arithmetic structures we intend to use. To achieve this, three steps are involved:

1. **Declare**: Introduce a modular arithmetic or related structure, along with its modulus and functionality. This can be done in any library or binary file.
2. **Init**: Performed exactly once in the final binary. It aggregates all previously declared structures, assigns them stable indices, and sets up linkage so that they can be referenced in generated code.
3. **Setup**: A one-time runtime procedure for security. This ensures that the compiled code matches the virtual machine’s expectations and that each instruction set is tied to the correct modulus or extension.

These steps ensure both performance and security: performance because the modulus is known at compile time, and security because runtime checks confirm that the correct structures have been initialized.

The list of existing extensions:

- [`openvm-algebra`](./algebra.md)
- [`openvm-bigint`](./bigint.md)
- [`openvm-keccak`](./keccak.md)
- [`openvm-pairing`](./pairing.md)
- [`openvm-ecc`](./ecc.md)
