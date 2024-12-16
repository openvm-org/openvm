# OpenVM

_A modular toolkit for extensible zkVMs_

OpenVM is an open-source zero-knowledge virtual machine (zkVM) framework focused on modularity at every level of the stack. OpenVM is designed for customization and extensibility without sacrificing performance or maintainability.

## Key Features

- **Modular no-CPU Architecture**: Unlike traditional machine architectures, the OpenVM architecture has no central processing unit. This design choice allows for seamless integration of custom chips, **without forking or modifying the core architecture**.

- **Extensible Instruction Set**: The instruction set architecture (ISA) is designed to be extended with new custom instructions that integrate directly with the virtual machine.

- **Rust Frontend**: ISA extensions are directly accessible through a Rust frontend via [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function), providing a smooth developer experience.

- **On-chain Verification**: Every VM made using the framework comes with out-of-the-box support for unbounded program proving with verification on Ethereum.

## Using This Book

The following chapters will guide you through:

- [Getting started](./getting-started/install.md).
- [Writing applications](./writing-apps/overview.md) in Rust targeting OpenVM and generating proofs.
- [Using existing extensions](./custom-extensions/overview.md) to optimize your Rust programs.
- [How to add custom VM extensions](./advanced-usage/new-extension.md).
