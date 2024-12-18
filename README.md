# OpenVM

[**Install**](https://book.openvm.dev/getting-started/install.html)
| [User Book](https://book.openvm.dev)
| [Contributor Docs](./docs)
| [Crate Docs](https://docs.openvm.dev/openvm)

OpenVM is a performant and modular zkVM framework built for customization and extensibility.

## Key Features

- **Modular no-CPU Architecture**: Unlike traditional machine architectures, the OpenVM architecture has no central processing unit. This design choice allows for seamless integration of custom chips, **without forking or modifying the core architecture**.

- **Extensible Instruction Set**: The instruction set architecture (ISA) is designed to be extended with new custom instructions that integrate directly with the virtual machine. Current extensions available for OpenVM include:

  - RISC-V support via RV32IM
  - A native field arithmetic extension for proof recursion and aggregation
  - The Keccak-256 hash function
  - Int256 arithmetic
  - Modular arithmetic over arbitrary fields
  - Elliptic curve operations, including multi-scalar multiplication and ECDSA scalar multiplication.
  - Pairing operations on the BN254 and BLS12-381 curves.

- **Rust Frontend**: ISA extensions are directly accessible through a Rust frontend via [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function), providing a smooth developer experience.

- **On-chain Verification**: Every VM made using the framework comes with out-of-the-box support for unbounded program proving with verification on Ethereum.

## Security Status

As of December 2024, OpenVM has not been audited and is currently not recommended for production use. We plan to continue development towards a production-ready release in 2025.

## For Users

See the [Book](https://book.openvm.dev) for more information on how to use OpenVM.

## For Contributors

See the [Contributor Docs](./docs) for more information on the project. A good starting point is [Project Layout](./docs/repo/layout.md).

## Acknowledgements

OpenVM is a new zkVM design framework. In the process of building it, we studied and learned from the designs and implementations of other projects. We would like to thank these projects for sharing their code for open source development:

- [Plonky3](https://github.com/Plonky3/Plonky3): The [STARK backend](https://github.com/openvm-org/stark-backend) and circuit writing interfaces are built on top of Plonky3, where we benefited from their modular design at the polynomial IOP level.
- [Valida](https://github.com/valida-xyz/valida): Many ideas around chips and chip interactions were pioneered by Valida and we were greatly inspired by their designs. Some parts of our ISA architecture also had inspirations from their ZK-specific ISA.
- [RISC Zero](https://github.com/risc0/risc0): We are extremely grateful to the RISC Zero team for merging their zkVM focused toolchain into [Rust upstream](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html). Our Rust toolchain integration builds upon their work.
- [SP1](https://github.com/succinctlabs/sp1): We gained inspirations from various parts of SP1's design and interfaces. The native compiler and eDSL we use for the Native Field VM Extension originated from their recursion compiler.
