# OpenVM

_A performant and modular zkVM framework built for customization and extensibility_

OpenVM is an open-source zero-knowledge virtual machine (zkVM) framework focused on modularity at every level of the stack. OpenVM is designed for customization and extensibility without sacrificing performance or maintainability.

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

## Using This Book

The following chapters will guide you through:

- [Getting started](./getting-started/install.md).
- [Writing applications](./writing-apps/overview.md) in Rust targeting OpenVM and generating proofs.
- [Using existing extensions](./custom-extensions/overview.md) to optimize your Rust programs.
- [How to add custom VM extensions](./advanced-usage/new-extension.md).

## Security Status

As of December 2024, OpenVM has not been audited and is currently not recommended for production use. We plan to continue development towards a production-ready release in 2025.

> 📖 **About this book**
>
> The book is continuously rendered [here](https://book.openvm.dev/)!
> You can contribute to this book on [GitHub][gh-book].

[gh-book]: https://github.com/openvm-org/openvm/tree/main/book