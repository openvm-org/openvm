# RV64IM Extension Circuit

This directory contains the circuit implementation of the RV64IM extension.

## Design

The RV64IM chips is composed of two main components: an adapter chip and a core chip

- The adapter chip adapts the core chip's I/O to the VM's expected format and manages interactions with the VM.
- The core chip is responsible for implementing the logic of the RISC-V instructions.

## Circuit statements

This section outlines the specific statements that each circuit is designed to prove.
For further details, including the underlying constraints and assumptions, please refer to the circuit implementation.

### Adapter

#### 1. ALU register adapters

- [Byte limbs](./adapters/alu_reg.rs)
- [u16 limbs](./adapters/alu_reg_u16.rs)

Given

- `rs1`, `rs2`, and `rd` are register addresses
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the operation
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 2. ALU immediate adapters

- [Byte limbs](./adapters/alu_imm.rs)
- [u16 limbs](./adapters/alu_imm_u16.rs)

Given

- `rs1` and `rd` are register addresses
- `imm` is the immediate operand supplied by the core
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory write to register `rd` is performed with the result of the operation
- The immediate supplied by the core is bound to the instruction on the execution bus
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 3. ALU W adapters

- [Register operands](./adapters/alu_w_reg_u16.rs)
- [Immediate operand](./adapters/alu_w_imm_u16.rs)

Given

- `rs1` and `rd` are register addresses
- The register adapter also receives the `rs2` register address
- The immediate adapter receives the immediate operand from the core
- `from_pc` is the current program address

This circuit proves the following:

- A u16-cell memory read from register `rs1` is performed and its upper 32 bits are preserved for the read interaction
- The register adapter reads `rs2` and preserves its upper 32 bits for the read interaction
- The immediate adapter binds the immediate supplied by the core to the instruction on the execution bus
- The low 32-bit result is sign-extended to a full 64-bit u16-cell register write by constraining the result sign bit from the most significant low-word limb
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 4. [Branch adapter](./adapters/branch.rs)

Given

- `rs1`, `rs2`, and `rd` are register addresses
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`.

#### 5. [JALR adapter](./adapters/jalr.rs)

Given

- `rd`, `rs1` are register addresses
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`

#### 6. [Load adapter](./adapters/load.rs)

Given

- `rd`, `rs1` are register addresses
- `imm` is an immediate value
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from the RV64 memory address space is performed at address `val(rs1) + imm`
- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 7. [Store adapter](./adapters/store.rs)

Given

- `rs1`, `rs2` are register addresses
- `imm` is an immediate value
- `mem_as` is the target address space, constrained to be the RV64 memory address space (`2`) or the public values address space (`3`)
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- A memory write to `mem_as` is performed at address `val(rs1) + imm`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 8. [Multiplication adapter](./adapters/mul.rs)

Given

- `rd`, `rs1`, `rs2` are register addresses
- `from_pc` is the current program address

This circuit proves the following:

- A memory read from register `rs1` is performed
- A memory read from register `rs2` is performed
- A memory write to register `rd` is performed with the result of the multiplication
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `from_pc + 4`

#### 9. [Rdwrite adapter](./adapters/rdwrite.rs)

Given

- `rd` is a register address
- `from_pc` is the current program address
- `to_pc` is the destination program address

This circuit proves the following:

- A memory write to register `rd` is performed if `rd` is not `x0`
- The instruction is correctly fetched from the program ROM at address `from_pc` and the program counter is set to `to_pc`

### Core

**Note:** For the core chips, it is not necessary to constrain the instruction operands (as specified in the statement), because the adapter already constrains them via the execution bus. The primary objective is to ensure that the result conforms to the instruction's specification.

#### 1. [Add/Sub](./add_sub/core.rs)

Given:

- `b` and `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^U16_BITS)`
- `a` is the decomposition of the result
- `opcode_add_flag` and `opcode_sub_flag` indicate if the instruction is `add` or `sub`

This circuit proves that:

- `compose(a) == compose(b) + compose(c)` for `add`, modulo the register width
- `compose(a) == compose(b) - compose(c)` for `sub`, modulo the register width
- Each limb of `a` is within the range `[0, 2^U16_BITS)`

#### 2. Bitwise Logic

- [Register operands](./bitwise_logic/core.rs)
- [Immediate operand](./bitwise_logic_imm/core.rs)

Given:

- `b` is the byte decomposition of `rs1`
- `c` is the byte decomposition of `rs2` for register instructions
- `c_low` and `imm_sign` encode the signed 12-bit immediate for immediate instructions
- `a` is the byte decomposition of the result
- `opcode_xor_flag`, `opcode_or_flag`, and `opcode_and_flag` indicate if the instruction is `xor`, `or`, or `and`

This circuit proves that:

- `a[i] == b[i] ^ c[i]` for `xor`, and the equivalent operation with the sign-extended immediate for `xori`
- `a[i] == b[i] | c[i]` for `or`, and the equivalent operation with the sign-extended immediate for `ori`
- `a[i] == b[i] & c[i]` for `and`, and the equivalent operation with the sign-extended immediate for `andi`

#### 3. [Branch Eq](./branch_eq/core.rs)

Given:

- `a` and `b` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- `opcode_beq_flag` and `opcode_bne_flag` indicate if the instruction is `beq` or `bne`
- `imm` is the immediate value
- `to_pc` is the destination program address

This circuit proves that:

- If `opcode_beq_flag` is true and `a` is equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If `opcode_bne_flag` is true and `a` is not equal to `b`, then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 4. [Branch Lt](./branch_lt/core.rs)

Given:

- `a` and `b` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- Flags indicating if the instruction is one of `blt`, `bltu`, `bge`, `bgeu`
- `imm` is the immediate value
- `to_pc` is the destination program address

This circuit proves that:

- If the instruction is `blt` and `compose(a) < compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bltu` and `compose(a) < compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bge` and `compose(a) >= compose(b)` (signed comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`
- If the instruction is `bgeu` and `compose(a) >= compose(b)` (unsigned comparison), then `to_pc == pc + imm`, otherwise `to_pc == pc + 4`

#### 5. [Divrem](./divrem/core.rs)

Given:

- `b` and `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- `q` is the decomposition of the quotient
- `r` is the decomposition of the remainder
- `a` is the decomposition of the result
- Flags indicating if the instruction is `div`, `divu`, `rem`, `remu`

This circuit proves that:

- `compose(b) = compose(c) * compose(q) + compose(r)`
- `0 <= |compose(r)| < |compose(c)|`
- If `compose(c) == 0`, then `compose(q) == -1` for signed operations and `compose(q) == 2^64 - 1` for unsigned operations
- Each limb of `q` and `r` is in the range `[0, 2^RV64_BYTE_BITS)`
- `a = q` if the instruction is `div` or `divu`
- `a = r` if the instruction is `rem` or `remu`

#### 6. [JAL_LUI](./jal_lui/core.rs)

Given:

- `rd` is the decomposition of the result
- `imm` is the immediate value
- `to_pc` is the destination program address
- `opcode` indicates the operation to be performed

This circuit proves that:

- Each limb of `rd` is in the range `[0, 2^RV64_BYTE_BITS)`
- If `opcode` is `jal`, then
  - `to_pc == pc + imm`
  - `compose(rd) == pc + 4`
  - The most significant limb of `rd` is in the range `[0, 2^(PC_BITS - RV64_BYTE_BITS * (RV64_WORD_NUM_LIMBS - 1))`
- If `opcode` is `lui`, then
  - `to_pc == pc + 4`
  - `compose(rd) == imm * 2^12`

#### 7. [JALR](./jalr/core.rs)

Given:

- `rs1` is the decomposition of the operand, with its limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- `rd` is the decomposition of the result
- `imm` is the immediate value
- `to_pc_least_sig_bit` is the least significant bit of `compose(rs1) + imm`
- `to_pc_limbs` is the decomposition of the remaining destination program address bits, where `to_pc_limbs[0]` is 15 bits and `to_pc_limbs[1]` contains the upper bits

This circuit proves that:

- `to_pc_least_sig_bit + 2 * compose(to_pc_limbs) == compose(rs1) + imm`
- The destination program address is `2 * compose(to_pc_limbs)`, so the least significant bit is cleared as required by `jalr`
- `compose(rd) == pc + 4`
- Each limb of `rd` is in the range `[0, 2^RV64_BYTE_BITS)`
- The most significant limb of `rd` is in the range `[0, 2^(PC_BITS - RV64_BYTE_BITS * (RV64_WORD_NUM_LIMBS - 1))`
- `to_pc_limbs[0]` is in the range `[0, 2^15)`
- `to_pc_limbs[1]` is in the range `[0, 2^(PC_BITS - 16))`

#### 8. [AUIPC](./auipc/core.rs)

Given:

- `rd` is the decomposition of the result
- `imm_limbs` are the decomposition of the immediate value
- `pc_limbs` are the decomposition of the program counter

This circuit proves that:

- `compose(rd) == compose(pc_limbs) + compose(imm_limbs) * 2^8`
- `compose(pc_limbs) == pc`
- Each limb of `rd`, `imm_limbs`, and `pc_limbs` is in the range `[0, 2^RV64_BYTE_BITS)`
- The most significant limb of `pc_limbs` is in the range `[0, 2^(PC_BITS - RV64_BYTE_BITS * (RV64_WORD_NUM_LIMBS - 1))`

#### 9. Less Than

- [Register operands](./less_than/core.rs)
- [Immediate operand](./less_than_imm/core.rs)

Given:

- `b` is the decomposition of `rs1`
- `c` is the decomposition of `rs2` for register instructions
- `imm_low11` and `imm_sign` encode the signed 12-bit immediate for immediate instructions
- `a` is the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- If `opcode` is `slt` and `compose(b) < compose(c)` (signed comparison), then `a` is 1.
- If `opcode` is `sltu` and `compose(b) < compose(c)` (unsigned comparison), then `a` is 1.
- `slti` and `sltiu` perform the corresponding comparison against the sign-extended immediate.
- Otherwise, `a` is 0.

#### 10. [Load](./load/mod.rs)

The RV64 unsigned-load circuit is split by access width: [byte](./load/byte/core.rs), [halfword](./load/halfword/mod.rs), [word](./load/word/mod.rs), and [doubleword](./load/doubleword/mod.rs).

Given:

- `read_data` is the aligned memory block
- `opcode` indicates the operation to be performed

These circuits prove that:

- The selected opcode matches the access width handled by the chip
- The shift selector matches the low address bits for byte, halfword, and word accesses
- The value written to `rd` is the selected memory value zero-extended to 64 bits

#### 11. [Store](./store/mod.rs)

The RV64 store circuit is split by access width: [byte](./store/byte/core.rs), [halfword](./store/halfword/mod.rs), [word](./store/word/mod.rs), and [doubleword](./store/doubleword/mod.rs).

Given:

- `read_data` is the data read from register `rs2`
- `prev_data` is the previous aligned memory block
- `opcode` indicates the operation to be performed

These circuits prove that:

- The selected opcode matches the access width handled by the chip
- The shift selector matches the low address bits for byte, halfword, and word accesses
- The value written to memory is the previous memory block with the selected bytes replaced by the value from `rs2`

#### 12. [Load sign extend](./load_sign_extend/mod.rs)

The RV64 signed-load circuit is split by access width: [byte](./load_sign_extend/byte/core.rs), [halfword](./load_sign_extend/halfword/mod.rs), and [word](./load_sign_extend/word/mod.rs).

Given:

- `read_data` is the data read from the RV64 memory address space at `aligned(val(rs1) + imm)`
- `opcode` indicates the operation to be performed

These circuits prove that:

- The selected opcode matches the signed-load width handled by the chip
- The shift selector matches the low address bits for byte, halfword, and word accesses
- The loaded value is sign-extended to 64 bits before it is written to `rd`
- The sign bit decomposition used for extension is range-checked

#### 13. [Multiplication](./mul/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- `a` is the decomposition of the lower 64 bits of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- `compose(a) == (compose(b) * compose(c)) % 2^64`
- Each limb of `a` is in the range `[0, 2^RV64_BYTE_BITS)`

#### 14. [MULH](./mulh/core.rs)

Given:

- `b`, `c` are decompositions of the operands, with their limbs assumed to be in the range `[0, 2^RV64_BYTE_BITS)`
- `a` is the decomposition of the upper 64 bits of the result
- `opcode` indicates the operation to be performed

This circuit proves that:

- Let `u64(x) = compose(x)`, and let `i64(x)` denote the signed 64-bit integer with bit decomposition `x`.
- If `opcode` is `mulh`, then `compose(a) = floor((i64(b) * i64(c) mod 2^128) / 2^64)`.
- If `opcode` is `mulhsu`, then `compose(a) = floor((i64(b) * u64(c) mod 2^128) / 2^64)`.
- If `opcode` is `mulhu`, then `compose(a) = floor((u64(b) * u64(c)) / 2^64)`.
- Each limb of `a` is in the range `[0, 2^RV64_BYTE_BITS)`

#### 15. Shift Logical

- [Register operand](./shift_logical/core.rs)
- [Immediate operand](./shift_logical_imm/core.rs)

Given:

- `b` is the decomposition of `rs1`
- `c` is the decomposition of the register shift amount for `sll` and `srl`
- The shift-marker columns encode the immediate shift amount for `slli` and `srli`
- `a` is the decomposition of the result
- The shift-marker sum indicates whether the row is valid, and `opcode_sll_flag` distinguishes a
  left shift from a right shift on valid rows

This circuit proves that:

- If the instruction is `sll`, then `compose(a) == compose(b) << (compose(c) % (NUM_LIMBS * LIMB_BITS))`, modulo the register width
- If the instruction is `srl`, then `compose(a) == compose(b) >> (compose(c) % (NUM_LIMBS * LIMB_BITS))`
- `slli` and `srli` perform the corresponding shift by the immediate amount bound by the marker columns.
- Each limb of `a` is in the range `[0, 2^LIMB_BITS)`

To stay sound at `LIMB_BITS = 16` over BabyBear, each limb of `b` is decomposed into the part that crosses the limb boundary (`carry`) and the part that stays (`aux`); both parts are range checked, and each limb of `a` is recombined additively from them so that no constraint term reaches the field modulus.

#### 16. Shift Right Arithmetic

- [Register operand](./shift_right_arithmetic/core.rs)
- [Immediate operand](./shift_right_arithmetic_imm/core.rs)

Given:

- `b` is the decomposition of `rs1`
- `c` is the decomposition of the register shift amount for `sra`
- The shift-marker columns encode the immediate shift amount for `srai`
- `a` is the decomposition of the result
- `is_valid` indicates if the row corresponds to a real instruction

This circuit proves that:

- `compose(a) == sign_extend(compose(b) >> (compose(c) % (NUM_LIMBS * LIMB_BITS)))`, where the vacated high bits are filled with the sign bit of `b`
- `srai` performs the same arithmetic shift using the immediate amount bound by the marker columns.
- The sign column `b_sign` equals the most significant bit of `b`
- Each limb of `a` is in the range `[0, 2^LIMB_BITS)`

The same carry/aux decomposition as the shift logical chip is used to keep every constraint term below the field modulus at `LIMB_BITS = 16`.

The full-width register and immediate shift chips use the corresponding u16 ALU adapters. The W register variants `sllw`/`srlw`/`sraw` ([shift_w](./shift_w/mod.rs)) reuse the register cores over 32-bit words (`NUM_LIMBS = 2`) with the W register adapter. The immediate variants `slliw`/`srliw`/`sraiw` reuse the immediate cores with the W immediate adapter.
