# RISC-V to OpenVM Transpilation

VM extensions consisting of intrinsics are transpiled from [custom RISC-V instructions](./RISCV.md) using a modular transpiler from the RISC-V ELF format to OpenVM assembly. This document specifies the behavior of the transpiler and uses the following notation:

- Let `ind(rd)` denote the register index, which is in `0..32`. In particular, it fits in one field element.
- We use `itof` for the function that takes 12-bits (or 21-bits in case of J-type) to a signed integer and then mapping to the corresponding field element. So `0b11…11` goes to `-1` in `F`.
- We use `sign_extend_24` to convert a 12-bit integer into a 24-bit integer via sign extension. We use this in conjunction with `utof`, which converts 24 bits into an unsigned integer and then maps it to the corresponding field element. Note that each 24-bit unsigned integer fits in one field element.
- We use `sign_extend_16` for the analogous conversion into a 16-bit integer via sign extension.
- We use `zero_extend_24` to convert an unsigned integer with at most 24 bits into a 24-bit unsigned integer by zero extension. This is used in conjunction with `utof` to convert unsigned integers to field elements.
- The notation `imm[0:4]` means the lowest 5 bits of the immediate.

The transpilation will only be valid for programs where:

- The program code does not have program address greater than or equal to `2^PC_BITS`.
- The program does not access memory outside the range `[0, 2^addr_max_bits)`.

We now specify the transpilation for system instructions and the default set of VM extensions.

## System Instructions

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| terminate      | TERMINATE `_, _, utof(imm)`                                      |
| hintstorew     | HINT_STOREW_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 2`    |
| reveal         | REVEAL_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 3`        |
| hintinput      | PHANTOM `_, _, HintInputRv32 as u16`                             |
| printstr       | PHANTOM `ind(rd), ind(rs1), PrintStrRv32 as u16`                 |

## RV32IM Extension

Transpilation from RV32IM to OpenVM assembly follows the mapping below, which is generally a 1-1 translation. 

### Handling of the `x0` register

The transpilation handles writes to `x0` as follows:

- Instructions that write to `x0` with no side effects are transpiled to the PHANTOM instruction with `c = 0x00` (`Nop`).
- Instructions that write to `x0` with side effects (JAL, JALR, AUIPC) are transpiled to the corresponding custom instruction without a write to `[0:4]_1`.

Because `[0:4]_1` is initialized to `0` and never written to, this guarantees that reads from `x0` yield `0` and enforces that any OpenVM program transpiled from RV32IM conforms to the RV32IM specification for `x0`.


| RISC-V Inst | OpenVM Instruction                                                         |
| ----------- | -------------------------------------------------------------------------- |
| add         | ADD_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sub         | SUB_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| xor         | XOR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| or          | OR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                                |
| and         | AND_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sll         | SLL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| srl         | SRL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sra         | SRA_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| slt         | SLT_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                               |
| sltu        | SLTU_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1`                              |
| addi        | ADD_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| xori        | XOR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| ori         | OR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`               |
| andi        | AND_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| slli        | SLL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| srli        | SRL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| srai        | SRA_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0`         |
| slti        | SLT_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`              |
| sltiu       | SLTU_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0`             |
| lb          | LOADB_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lh          | LOADH_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lw          | LOADW_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`            |
| lbu         | LOADBU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`           |
| lhu         | LOADHU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2`           |
| sb          | STOREB_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| sh          | STOREH_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| sw          | STOREW_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2`          |
| beq         | BEQ_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bne         | BNE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| blt         | BLT_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bge         | BGE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                             |
| bltu        | BLTU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                            |
| bgeu        | BGEU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                            |
| jal         | JAL_RV32 `ind(rd), 0, itof(imm), 1, 0, (rd != x0)`                         |
| jalr        | JALR_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 0, (rd != x0)` |
| lui         | LUI_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31])), 1, 0, 1`           |
| auipc       | AUIPC_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31]) << 4), 1`          |
| mul         | MUL_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| mulh        | MULH_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |
| mulhsu      | MULHSU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                               |
| mulhu       | MULHU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                |
| div         | DIV_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| divu        | DIVU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |
| rem         | REM_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                  |
| remu        | REMU_RV32 `ind(rd), ind(rs1), ind(rs2), 1`                                 |

## Keccak Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| keccak256      | KECCAK256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |

## SHA2-256 Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| sha256         | SHA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |

## BigInt Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| add256         | ADD256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| sub256         | SUB256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| xor256         | XOR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| or256          | OR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                   |
| and256         | AND256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| sll256         | SLL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| srl256         | SRL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| sra256         | SRA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| slt256         | SLT256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| sltu256        | SLTU256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                 |
| mul256         | MUL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`                  |
| beq256         | BEQ256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2`                |

## Algebra Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| addmod\<N\>    | ADDMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| submod\<N\>    | SUBMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| mulmod\<N\>    | MULMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| divmod\<N\>    | DIVMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| iseqmod\<N\>   | ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`            |
| setup\<N\>     | SETUP_ADDSUB,MULDIV,ISEQ_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` |

## Elliptic Curve Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| sw_add_ne\<C\> | SW_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| sw_double\<C\> | SW_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`                 |

## Pairing Extension

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| hint_final_exp | PHANTOM `ind(rs1), pairing_idx, HintFinalExp as u16`             |
