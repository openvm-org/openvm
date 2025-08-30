# OpenVM

[![Telegram Chat][tg-badge]][tg-url] [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/openvm-org/openvm)

[**Install**](https://book.openvm.dev/getting-started/install.html)
| [Book](https://book.openvm.dev)
| [Contributor Docs](./docs)
| [Rustdocs](https://docs.openvm.dev/openvm)
| [Whitepaper](https://openvm.dev/whitepaper.pdf)

[tg-badge]: https://img.shields.io/endpoint?color=neon&logo=telegram&label=chat&url=https://tg.sumanjay.workers.dev/openvm

OpenVM is a performant and modular zkVM framework built for customization and extensibility.

## Key Features

- **Modular no-CPU Architecture**: Unlike traditional machine architectures, the OpenVM architecture has no central processing unit. This design choice allows for seamless integration of custom chips, **without forking or modifying the core architecture**.

- **Extensible Instruction Set**: The instruction set architecture (ISA) is designed to be extended with new custom instructions that integrate directly with the virtual machine. Current extensions available for OpenVM include:

  - RISC-V support via RV32IM
  - A native field arithmetic extension for proof recursion and aggregation
  - The Keccak-256 and SHA2-256 hash functions
  - Int256 arithmetic
  - Modular arithmetic over arbitrary fields
  - Elliptic curve operations, including multi-scalar multiplication and ECDSA scalar multiplication.
  - Pairing operations on the BN254 and BLS12-381 curves.

- **Rust Frontend**: ISA extensions are directly accessible through a Rust frontend via [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function), providing a smooth developer experience.

- **On-chain Verification**: Every VM made using the framework comes with out-of-the-box support for unbounded program proving with verification on Ethereum.

## Status

As of June 2025, OpenVM v1.2.0 and later are recommended for production use. OpenVM completed an external [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-cantina-report.pdf) on [Cantina](https://cantina.xyz/) from January to March 2025 as well as an internal [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-internal/README.md) by members of the [Axiom](https://axiom.xyz/) team during the same timeframe.

## For Users

See the [Book](https://book.openvm.dev) for more information on how to use OpenVM.

## For Contributors

See the [Contributor Docs](./docs) for more information on the project. A good starting point is [Project Layout](./docs/repo/layout.md). See the [Assets](https://github.com/openvm-org/openvm/tree/main/assets) folder for OpenVM's logo and favicon.

## Security

See [SECURITY.md](./SECURITY.md).

## Acknowledgements

OpenVM is a new zkVM design framework. In the process of building it, we studied and learned from the designs and implementations of other projects. We would like to thank these projects for sharing their code for open source development:

- [Plonky3](https://github.com/Plonky3/Plonky3): The [STARK backend](https://github.com/openvm-org/stark-backend) and circuit writing interfaces are built on top of Plonky3, where we benefited from their modular design at the polynomial IOP level.
- [Valida](https://github.com/valida-xyz/valida): Many ideas around chips and chip interactions were pioneered by Valida and we were greatly inspired by their designs. Some parts of our ISA architecture also had inspirations from their ZK-specific ISA.
- [RISC Zero](https://github.com/risc0/risc0): We are extremely grateful to the RISC Zero team for merging their zkVM focused toolchain into [Rust upstream](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html). Our Rust toolchain integration builds upon their work.
- [SP1](https://github.com/succinctlabs/sp1): We gained inspirations from various parts of SP1's design and interfaces. The native compiler and eDSL we use for the Native Field VM Extension originated from their recursion compiler.

[tg-url]: https://t.me/openvm
