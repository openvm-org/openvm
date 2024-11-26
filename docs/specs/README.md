# Design and Specifications

- [Virtual Machines](./vm/README.md)
- [Aggregation](./aggregation.md) - design for VM-based proof aggregation.

# Toolchain + VM Extensions

axvm provides the framework to be able to simultaneously extend the machine architecture, instruction set, and compiler toolchain so that the new architecture is directly accessible as a first-class citizen from the language frontend.

To write a new extension, you will need to add three crates:

- `lib` (better name?) -- the no_std guest library that should be importable from guest program with target_os=zkvm
  - this should include all custom RISC-V instruction declarations, intrinsic macro definitions, and intrinsic function wrappers
- transpiler extension - implement `TranspilerExtension` trait to specify how newly introduced custom RISC-V instructions should be transpiled into custom axVM instructions.
  - this crate needs access to the custom `Opcode` trait implementations
- VM extension - implement `VmExtension` trait and define new chips and assign them to handle the new opcodes.
