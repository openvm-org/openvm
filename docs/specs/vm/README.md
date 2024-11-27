# Modular Virtual Machines

This section describes our modular framework that allows to build STARK virtual machines with customizable opcodes.

- [ISA](./ISA.md) discusses the instruction set architecture that all virtual machines in this framework must follow, and the memory model we use. This spec is purely at the computer architecture level and does not discuss the STARK implementations (although the design choices are influenced by these considerations).
- [RISC-V](./RISCV.md) defines custom RISC-V instructions for axVM and the transpilation spec from RV32IM and these custom RISC-V instructions to axVM instructions.
- [STARK](./stark.md) discusses the design of how components of the virtual machine are implemented as STARKs.
  - [(DRAFT) Paper](./axVM_STARK_Architecture_DRAFT.pdf) formalizes polynomial representation of the VM architecture and the system chips.
- [Continuations](./continuations.md) discusses how continuations can be enabled for any virtual machine.
