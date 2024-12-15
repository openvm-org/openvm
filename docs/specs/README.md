# Design and Specifications

## Design

OpenVM provides the modular framework to co-design and build a custom zkVM, ISA, and supporting programming language frontend simultaneously.

- The [Circuit Architecture](./circuit.md) explains the VM circuit architecture, which focuses on maximizing modularity and composability. The architecture supports adding arbitrary chips to handle custom instructions, as long as they fall within our ISA framework.
  - There are a few required system chips: Program, Connector, Range Checker, Memory (which can be several chips depending on the configuration)
- The [Instruction Set Architecture](./ISA.md) specifies our ISA framework and also lists the currently supported instructions in different VM extensions. Our ISA is specialized for zkVMs and provide additional flexibility over traditional machine architectures:
  - Support for multiple traditional machine architectures _simultaneously_ with multiple memory address spaces. These address spaces also allow support for both register and stack based architectures.
  - Variable word size, which allows flexible support for different register sizes and also higher bandwidth memory accesses.
- Programming language support is provided using Rust as the language frontend. We compile Rust to 32-bit RISC-V ELF binary via LLVM. To provide intrinsic support for custom instructions within Rust, we use Rust to insert custom RISC-V directives to the LLVM assembler, which are assembled into the ELF. We use an extendable [transpiler](./RISCV.md) to convert the RISC-V ELF into OpenVM assembly. While intrinsic instructions are custom, they still respect the RISC-V architecture. Our framework supports the addition of additional frontends to generate OpenVM assembly, which can be included within Rust itself via procedural macros or as a standalone frontend to generate OpenVM assembly.
  - [RISC-V custom instructions and transpiler](./RISCV.md)
- We provide a general recursion library written in a standalone Rust eDSL for OpenVM native kernel instructions. The library supports inner aggregation of arbitrary STARK proofs, outer aggregation using Halo2, and on-chain verification of the aggregated SNARK proof.
- All VMs within our framework support proving of programs with unbounded cycle count using continuations. Our [continuations design](./continuations.md) maximizes proving parallelism and does not rely on any interactive communication between continuation segments.

### Extensions

The framework is designed to be extendable via external crates _without forking_.
VM extensions provide a way to simultaneously extend the VM with new chips, opcodes, and toolchain support for these opcodes.
A new extension of the overall architecture consists of three components:

- Library: the guest library that compiles program code (usually in Rust) into RISC-V assembly with custom instructions.
- Transpiler extension: extend the transpiler to specify how newly introduced custom RISC-V instructions should be transpiled into custom OpenVM instructions.
- VM extension: define new chips and assign them to handle the new opcodes.
