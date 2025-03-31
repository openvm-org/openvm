# RV32IM Extension Circuit

This directory contains the circuit implementation of the RV32IM extension.

## Design

The RV32IM chips is composed of two main components: an adapter chip and a core chip

- The adapter chip adapts the core chip's I/O to the VM's expected format and manages interactions with the VM.
- The core chip is responsible for implementing the logic of the RISC-V instructions.

## Circuit statements

This section outlines the specific statements that each circuit is designed to prove.
For further details, including the underlying constraints and assumptions, please refer to the circuit implementation.

### Adapter

#### 1. [ALU adapter](./adapters/alu.rs)

Given

- `rs1`, `rs2`, and `rd` are register addresses
- `rs2_as` is a boolean indicating if `rs2` is an immediate value
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- If `rs2_as` is false, a memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the operation
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 2. [Branch adapter](./adapters/branch.rs)

Given

- `rs1`, `rs2`, and `rd` are register addresses
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`.

#### 3. [JALR adapter](./adapters/jalr.rs)

Given

- `rd`, `rs1` are register addresses
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`

#### 4. [Load/store adapter](./adapters/loadstore.rs)

Given

- `rd`, `rs1` are register addresses
- `imm` is an immediate value
- `mem_as` is an address space
- `is_load` is a boolean indicating if the instruction is a load
- `from_pc` is the current program address

This circuit proves the following:

- If `is_load` is true:
  - `mem_as` is in `{0, 1, 2}`
  - A memory read from register `rs1` is performed
  - A memory read from `mem_as` is performed at address `val(rs1) + imm` where `val(rs1)` is the value read from register `rs1`
  - A memory write to register `rd` is performed if `rd` is not `x0`
  - The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

- Otherwise:
  - `mem_as` is in `{2, 3, 4}`
  - A memory read from register `rs1` is performed
  - A memory read from register `rd` is performed
  - A memory write to `mem_as` is performed at address `val(rs1) + imm` where `val(rs1)` is the value read from register `rs1`
  - The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 5. [Multiplication adapter](./adapters/mul.rs)

Given

- `rd`, `rs1`, `rs2` are register addresses
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the multiplication
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 6. [Rdwrite adapter](./adapters/rdwrite.rs)

Given

- `rd` is a register address
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`

### Core

**Note:** For the core chips, it is not necessary to constrain the instruction operands (as specified in the statement), because the adapter already constrains them via the execution bus. The primary objective is to ensure that the result conforms to the instruction's specification.

#### 1. [Base ALU](./base_alu/core.rs)

Given:

- `b` and `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `a` is the decomposition of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- `compose(a) == compose(b) op compose(c)`
- Each limb of `a` is within the range `[0, 2^RV32_CELL_BITS)`

#### 2. [Branch Eq](./branch_eq/core.rs)

Given:

- `a` and `b` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `opcode_beq_flag` and `opcode_bne_flag` indicate if the instruction is `beq` or `bne`
- `imm` is the immediate value
- `to_pc` is the destination program address

This circuit proves that:

- If `opcode_beq_flag` is true and `a` is equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If `opcode_bne_flag` is true and `a` is not equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 3. [Branch Lt](./branch_lt/core.rs)

Given:

- `a` and `b` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- Flags indicating if the instruction is one of `blt`, `bltu`, `bge`, `bgeu`
- `imm` is the immediate value
- `to_pc` is the destination program address

This circuit proves that:

- If the instruction is `blt` and `compose(a) < compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bltu` and `compose(a) < compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bge` and `compose(a) >= compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bgeu` and `compose(a) >= compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 4. [Divrem](./divrem/core.rs)

Given:

- `b` and `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `q` is the decomposition of the quotient
- `r` is the decomposition of the remainder
- `a` is the decomposition of the result
- Flags indicating if the instruction is `div`, `divu`, `rem`, `remu`

This circuit proves that:

- `compose(b) = compose(c) * compose(q) + compose(r)`
- `0 <= |compose(r)| < |compose(c)|`
- If `compose(c) == 0`, then `compose(q) == -1` for signed operations and `compose(q) == 2^32 - 1` for unsigned operations
- Each limb of `q` and `r` is in the range `[0, 2^RV32_CELL_BITS)`
- `a = q` if the instruction is `div` or `divu`
- `a = r` if the instruction is `rem` or `remu`

#### 5. [JAL_LUI](./jal_lui/core.rs)

Given:

- `rd` is the decomposition of the result
- `imm` is the immediate value
- `to_pc` is the destination program address
- `opcode` indicates the operation to be performed

This circuit proves that:

- Each limb of `rd` is in the range `[0, 2^RV32_CELL_BITS)`
- If `opcode` is `jal`, then
  - `to_pc == pc + imm`
  - `compose(rd) == pc + 4`
  - The most significant limb of `rd` is in the range `[0, 2^(PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))`
- If `opcode` is `lui`, then
  - `to_pc == pc + 4`
  - `compose(rd) == imm * 2^8`

#### 6. [JALR](./jalr/core.rs)

Given:

- `rs1` is the decomposition of the operand, with its limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `rd` is the decomposition of the result
- `imm` is the immediate value
- `to_pc_limbs` is the decomposition into 16-bit limbs of the destination program address

This circuit proves that:

- `compose(to_pc_limbs) == compose(rs1) + imm`
- `compose(rd) == pc + 4`
- Each limb of `rd` is in the range `[0, 2^RV32_CELL_BITS)`
- The most significant limb of `rd` is in the range `[0, 2^(PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))`
- `to_pc_limbs[0]` is in the range `[0, 2^15)`
- `to_pc_limbs[1]` is in the range `[0, 2^(PC_BITS - 16))`

#### 7. [AUIPC](./auipc/core.rs)

Given:

- `rd` is the decomposition of the result
- `imm_limbs` are the decomposition of the immediate value
- `pc_limbs` are the decomposition of the program counter

This circuit proves that:

- `compose(rd) == compose(pc_limbs) + compose(imm_limbs) * 2^8`
- `compose(pc_limbs) == pc`
- Each limb of `rd`, `imm_limbs`, and `pc_limbs` is in the range `[0, 2^RV32_CELL_BITS)`
- The most significant limb of `pc_limbs` is in the range `[0, 2^(PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))`

#### 8. [Less than](./less_than/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `a` is the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- If `opcode` is `slt` and `compose(b) < compose(c)` (signed comparison), then `a` is 1.
- If `opcode` is `sltu` and `compose(b) < compose(c)` (unsigned comparison), then `a` is 1.
- Otherwise, `a` is 0.

#### 9. [Load sign extend](./load_sign_extend/core.rs) and [Loadstore](./loadstore/core.rs)

Given:

- `read_data` is the data read from `mem_as[aligned(val(rs1) + imm)]` if the instruction is load, otherwise it is the data read from register `rd`
- `write_data` is the data to be written to register `rd` if the instruction is load, otherwise it is the data to be written to `mem_as[aligned(val(rs1) + imm)]`
- `opcode` indicates the operation to be performed

This circuit proves that `write_data` equals `shift(read_data)`, where the shift amount is adjusted according to the instruction.

#### 10. [Multiplication](./mul/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `a` is the decomposition of the lower 32 bits of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- `compose(a) == (compose(b) * compose(c)) % 2^32`
- Each limb of `a` is in the range `[0, 2^RV32_CELL_BITS)`

#### 11. [MULH](./mulh/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `a` is the decomposition of the upper 32 bits of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- `compose(a) == floor((compose(b) * compose(c)) / 2^32)`
- Each limb of `a` is in the range `[0, 2^RV32_CELL_BITS)`

#### 12. [Shift](./shift/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV32_CELL_BITS)`
- `a` is the decomposition of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- If `opcode` is `sll`, then `compose(a) == compose(b) << (compose(c) % (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS))`
- If `opcode` is `srl`, then `compose(a) == compose(b) >> (compose(c) % (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS))`
- If `opcode` is `sra`, then `compose(a) == sign_extend(compose(b) >> (compose(c) % (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS)))`
- Each limb of `a` is in the range `[0, 2^RV32_CELL_BITS)`
