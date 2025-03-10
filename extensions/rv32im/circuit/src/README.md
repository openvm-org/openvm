# RV32IM Extension Circuit

This directory contains the circuit implementation of the RV32IM extension.

## Design

The RV32IM chips consist of a series of an adapter chip and a core chip.

- The adapter chip is responsible for adapting the input and output of the core chip to the format expected by the VM, and handling any interactions with the VM.
- The core chip is responsible for implementing the logic of the RISC-V instructions.

## Circuit statements

This section describes the statements that each circuit is responsible for proving.
Details about the constraints and assumptions for each statement are available in the implementation of the circuit.

### Adapter

#### 1. [ALU adapter](./adapters/alu.rs)

Given `rs1`, `rs2`, and `rd` pointers, and a boolean indicating if `rs2` is an immediate value,
this circuit proves the following:

- A memory read from register `rs1` is performed
- If `rs2` is not an immediate value, a memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the operation
- The instruction is executed successfully by the VM and the program counter is incremented by 4

#### 2. [Branch adapter](./adapters/branch.rs)

Given `rs1`, `rs2`, and the destination pointer `to_pc`, this circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- The instruction is executed successfully by the VM and the program counter is set to `to_pc`

#### 3. [JALR adapter](./adapters/jalr.rs)

Given `rd`, `rs1`, the destination pointer `to_pc`, and a flag indicating if `rd` is `x0`, this circuit proves the following:

- A memory read from register `rs1` is performed
- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is executed successfully by the VM and the program counter is set to `to_pc`

#### 4. [Load/store adapter](./adapters/loadstore.rs)

Given `rd`, `rs1`, immediate value `imm`, address space `mem_as`, and a flag indicating if the instruction is a load or store, this circuit proves the following:

If the instruction is a load:

- A memory read from register `rs1` is performed
- A memory read from `mem_as` is performed at `rs1 + imm`
- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is executed successfully by the VM and the program counter is incremented by 4

If the instruction is a store:

- A memory read from register `rs1` is performed
- A memory read from register `rd` is performed
- A memory write to `mem_as` is performed at `rs1 + imm`
- The instruction is executed successfully by the VM and the program counter is incremented by 4

#### 5. [Multiplication adapter](./adapters/mul.rs)

Given `rd`, `rs1`, `rs2`,this circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the multiplication
- The instruction is executed successfully by the VM and the program counter is incremented by 4

#### 6. [Rdwrite adapter](./adapters/rdwrite.rs)

Given `rd`, the destination pointer `to_pc`, and a flag indicating if `rd` is `x0`, this circuit proves the following:

- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is executed successfully by the VM and the program counter is set to `to_pc`
