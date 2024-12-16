# OpenVM

[**Install**](https://book.openvm.dev/getting-started/install.html)
| [User Book](https://book.openvm.dev)
| [Contributor Docs](./docs)
| [Crate Docs](https://docs.openvm.dev/openvm)

## Acknowledgements

OpenVM is a new zkVM design framework. In the process of building it, we studied and learned from the designs and implementations of other projects. We would like to thank these projects for sharing their code for open source development:

- [Plonky3](https://github.com/Plonky3/Plonky3): The [STARK backend](https://github.com/openvm-org/stark-backend) and circuit writing interfaces are built on top of Plonky3, where we benefited from their modular design at the polynomial IOP level.
- [Valida](https://github.com/valida-xyz/valida): Many ideas around chips and chip interactions were pioneered by Valida and we were greatly inspired by their designs. Some parts of our ISA architecture also had inspirations from their ZK-specific ISA.
- [RISC Zero](https://github.com/risc0/risc0): We are extremely grateful to the RISC Zero team for merging their zkVM focused toolchain into [Rust upstream](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html). Our Rust toolchain integration builds upon their work.
- [SP1](https://github.com/succinctlabs/sp1): We gained inspirations from various parts of SP1's design and interfaces. The native compiler and eDSL we use for the Native Field VM Extension originated from their recursion compiler.
