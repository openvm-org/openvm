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

### Core

**Note:** For the core chips, we do not need to constrain the instructions operands (which is given in the statement), since they are already constrained by the adapter through execution bus. The main goal is to constrain the result matches the specification of the instruction.

#### 1. [Base ALU](./base_alu/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `a` is the decomposition of the result
- `opcode` indicating the operation to be performed

This circuit proves that:

- `compose(a) == compose(b) op compose(c)`
- `a` limbs are in range `[0, 2^RV32_CELL_BITS)`

#### 2. [Branch_Eq](./branch_eq/core.rs)

Given:

- `a`, `b` are decompositions of the operands
- `opcode_beq_flag` and `opcode_bne_flag` indicating if the instruction is a branch equal or branch not equal
- `imm` is the immediate value
- `to_pc` is the destination pointer

This circuit proves that:

- If `opcode_beq_flag` is true and `a` is equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If `opcode_bne_flag` is true and `a` is not equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 3. [Branch_Lt](./branch_lt/core.rs)

Given:

- `a`, `b` are decompositions of the operands
- Flags indicating if the instruction is one of `blt`, `bltu`, `bge`, `bgeu`
- `imm` is the immediate value
- `to_pc` is the destination pointer

This circuit proves that:

- If the instruction is `blt` and `compose(a) < compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bltu` and `compose(a) < compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bge` and `compose(a) >= compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bgeu` and `compose(a) >= compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 4. [Divrem](./divrem/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `q` is the quotient
- `r` is the remainder
- `a` is the decomposition of the result
- Flags indicating if the instruction is `div`, `divu`, `rem`, `remu`

This circuit proves that:

- `compose(b) = compose(c) * compose(q) + compose(r)`
- `0 <= |compose(r)| < |compose(c)|`
- If `compose(c) == 0`, then `compose(q) == -1` for signed operations and `compose(q) == 2^32 - 1` for unsigned operations
- `q` and `r` limbs are in range `[0, 2^RV32_CELL_BITS)`
- `a = q` if the instruction is `div` or `divu`
- `a = r` if the instruction is `rem` or `remu`

#### 5. [JAL_LUI](./jal_lui/core.rs)

Given:

- `rd` is the decomposition of the operand
- `imm` is the immediate value
- `to_pc` is the destination pointer
- `opcode` indicating the operation to be performed

This circuit proves that:

- If `opcode` is `jal`, then `to_pc == pc + imm` and `decompose(rd) == pc + 4`
- If `opcode` is `lui`, then `to_pc == pc + 4` and `decompose(rd) == imm * 2^8`

#### 6. [JALR](./jalr/core.rs)

Given:

- `rd`, `rs1` are decompositions of the operands
- `imm` is the immediate value
- `to_pc_limbs` are the decomposition of the destination pointer

This circuit proves that:

- `compose(to_pc_limbs) == compose(rs1) + imm`
- `compose(rd) == pc + 4`
- `to_pc_limbs` limbs are in range `[0, 2^RV32_CELL_BITS)`

#### 7. [AUIPC](./auipc/core.rs)

Given:

- `rd` is the decomposition of the operands
- `imm_limbs` are the decomposition of the immediate value
- `pc_limbs` are the decomposition of the program counter

This circuit proves that:

- `compose(rd) == compose(pc_limbs) + compose(imm_limbs) * 2^8`
- `compose(pc_limbs) == pc` and `compose(pc_limbs) < 2^PC_MAX_BITS`
- `rd`, `imm_limbs`, `pc_limbs` limbs are in range `[0, 2^RV32_CELL_BITS)`

#### 8. [Less_than](./less_than/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `a` is the result
- `opcode` indicating the operation to be performed

This circuit proves that:

- If `opcode` is `slt` and `compose(b) < compose(c)` (signed comparison), then `a` is 1.
- If `opcode` is `sltu`, then `compose(b) < compose(c)` (unsigned comparison), then `a` is 1.
- Otherwise, `a` is 0.

#### 9. [Load_sign_extend](./load_sign_extend/core.rs) and [Loadstore](./loadstore/core.rs)

Given:

- `read_data` is the data read from `aligned(mem_as[rs1 + imm])` if the instruction is load, otherwise it is the data read from `rd`
- `write_data` is the data to be written to `rd` if the instruction is load, otherwise it is the data to be written to `mem_as[rs1 + imm]`
- Flags indicating which instruction is being executed

This circuit proves that `write_data == shift(read_data)` with shift amount adjusted for the instruction.

#### 10. [Mul](./mul/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `a` is the decomposition of the lower 32 bits of the result
- `opcode` indicating the operation to be performed

This circuit proves that:

- `compose(a) == compose(b) * compose(c) % 2^32`
- `a` limbs are in range `[0, 2^RV32_CELL_BITS)`

#### 11. [MULH](./mulh/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `a` is the decomposition of the upper 32 bits of the result
- `opcode` indicating the operation to be performed

This circuit proves that:

- `compose(a) == compose(b) * compose(c) / 2^32`
- `a` limbs are in range `[0, 2^RV32_CELL_BITS)`

#### 12. [Shift](./shift/core.rs)

Given:

- `b`, `c` are decompositions of the operands
- `a` is the decomposition of the result
- `opcode` indicating the operation to be performed

This circuit proves that:

- If `opcode` is `sll`, then `compose(a) == compose(b) << (compose(c) % 32)`
- If `opcode` is `srl`, then `compose(a) == compose(b) >> (compose(c) % 32)`
- If `opcode` is `sra`, then `compose(a) == sign_extend(compose(b) >> (compose(c) % 32))`
- `a` limbs are in range `[0, 2^RV32_CELL_BITS)`
