# RISC-V to OpenVM Transpilation

VM extensions consisting of intrinsics are transpiled from [custom RISC-V instructions](./RISCV.md) using a modular transpiler from the RISC-V ELF format to OpenVM assembly. This document specifies the behavior of the transpiler and uses the following notation:

- Let `ind(rd)` denote `4 * (register index)`, which is in `0..128`. In particular, it fits in one field element.
- We use `itof` for the function that takes 12-bits (or 21-bits in case of J-type) to a signed integer and then mapping to the corresponding field element. So `0b11…11` goes to `-1` in `F`.
- We use `sign_extend_24` to convert a 12-bit integer into a 24-bit integer via sign extension. We use this in conjunction with `utof`, which converts 24 bits into an unsigned integer and then maps it to the corresponding field element. Note that each 24-bit unsigned integer fits in one field element.
- We use `sign_extend_16` for the analogous conversion into a 16-bit integer via sign extension.
- We use `zero_extend_24` to convert an unsigned integer with at most 24 bits into a 24-bit unsigned integer by zero extension. This is used in conjunction with `utof` to convert unsigned integers to field elements.
- We use `sign_of(imm)` to get the sign bit of the immediate `imm`.
- The notation `imm[0:4]` means the lowest 5 bits of the immediate.
- For a phantom instruction `ins`, `disc(ins)` is the discriminant specified in the [ISA specification](./ISA.md#system-instructions).
- For a phantom instruction `ins` and a 16-bit `c_upper`, `phantom_c(c_upper, ins) = c_upper << 16 | disc(ins)` is the corresponding 32-bit operand `c` for PHANTOM.

The transpilation will only be valid for programs where:

- The program code does not have program address greater than or equal to `2^PC_BITS`.
- The program does not access memory outside the range `[0, 2^addr_max_bits)`.

We now specify the transpilation for system instructions and the default set of VM extensions.

## System Instructions

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| terminate      | TERMINATE `_, _, utof(imm)`                                      |

## RV32IM Extension

Transpilation from RV32IM to OpenVM assembly follows the mapping below, which is generally 
a 1-1 translation between RV32IM instructions and OpenVM instructions. The main exception relates
to handling of the `x0` register, which discards writes and has value `0` in all reads.
We handle writes to `x0` in transpilation as follows:

- Instructions that write to `x0` with no side effects are transpiled to the PHANTOM instruction with `c = 0x00` (`Nop`).
- Instructions that write to a register which might be `x0` with side effects (JAL, JALR) are transpiled to the corresponding custom instruction whose write behavior is controlled by a flag specifying whether the target register is `x0`.

Because `[0:4]_1` is initialized to `0` and never written to, this guarantees that reads from `x0` yield `0` and enforces that any OpenVM program transpiled from RV32IM conforms to the RV32IM specification for `x0`.

### System Level Extensions to RV32IM

| RISC-V Inst | OpenVM Instruction                                        |
| ----------- | --------------------------------------------------------- |
| hintstorew  | HINT_STOREW_RV32 `0, ind(rd), _, 1, 2`                    |
| hintbuffer  | HINT_BUFFER_RV32 `ind(rs1), ind(rd), _, 1, 2`             |
| reveal      | REVEAL_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 3, 0, sign_of(imm)` |
| hintinput   | PHANTOM `_, _, disc(Rv32HintInput)`                       |
| printstr    | PHANTOM `ind(rd), ind(rs1), disc(Rv32PrintStr)`           |
| hintrandom  | PHANTOM `ind(rd), _, disc(Rv32HintRandom)`                |

### Standard RV32IM Instructions

| RISC-V Inst | OpenVM Instruction                                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| add         | ADD_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| sub         | SUB_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| xor         | XOR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| or          | OR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                       |
| and         | AND_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| sll         | SLL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| srl         | SRL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| sra         | SRA_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| slt         | SLT_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| sltu        | SLTU_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                     |
| addi        | ADD_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                     |
| xori        | XOR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                     |
| ori         | OR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                      |
| andi        | AND_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                     |
| slli        | SLL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                |
| srli        | SRL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                |
| srai        | SRA_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                |
| slti        | SLT_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                     |
| sltiu       | SLTU_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                    |
| lb          | LOADB_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`  |
| lh          | LOADH_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`  |
| lw          | LOADW_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`  |
| lbu         | LOADBU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)` |
| lhu         | LOADHU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)` |
| sb          | STOREB_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)`                                                   |
| sh          | STOREH_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)`                                                   |
| sw          | STOREW_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 0, sign_of(imm)`                                                   |
| beq         | BEQ_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                       |
| bne         | BNE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                       |
| blt         | BLT_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                       |
| bge         | BGE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                       |
| bltu        | BLTU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                      |
| bgeu        | BGEU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                                      |
| jal         | JAL_RV32 `ind(rd), 0, itof(imm), 1, 0, (rd != x0)`                                                                                   |
| jalr        | JALR_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 0, (rd != x0)`                                                           |
| lui         | LUI_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31])), 1, 0, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                  |
| auipc       | AUIPC_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31]) << 4), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                 |
| mul         | MUL_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                         |
| mulh        | MULH_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                        |
| mulhsu      | MULHSU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                      |
| mulhu       | MULHU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                       |
| div         | DIV_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                         |
| divu        | DIVU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                        |
| rem         | REM_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                         |
| remu        | REMU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                        |

## OpenVM Intrinsic VM Extensions

The following sections specify the transpilation of the default set of intrinsic extensions
to OpenVM. In order to preserve correctness of handling of `x0`, the transpilation must respect
the constraint that any instruction that writes to a register must:

- Transpile to `Nop` if the register is `x0` and there are no side effects.
- Transpile to an OpenVM assembly instruction that does not write to `[0:4]_1` and processes side effects if the register is `x0` and there are side effects.

Each VM extension's behavior is specified below.

### Keccak Extension

| RISC-V Inst | OpenVM Instruction                                 |
| ----------- | -------------------------------------------------- |
| keccak256   | KECCAK256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2` |

### SHA2-256 Extension

| RISC-V Inst | OpenVM Instruction                              |
| ----------- | ----------------------------------------------- |
| sha256      | SHA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2` |

### BigInt Extension

| RISC-V Inst | OpenVM Instruction                                |
| ----------- | ------------------------------------------------- |
| add256      | ADD256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sub256      | SUB256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| xor256      | XOR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| or256       | OR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`    |
| and256      | AND256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sll256      | SLL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| srl256      | SRL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sra256      | SRA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| slt256      | SLT256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sltu256     | SLTU256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`  |
| mul256      | MUL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| beq256      | BEQ256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2` |

### Algebra Extension

#### Modular Arithmetic

| RISC-V Inst  | OpenVM Instruction                                                                                                                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | ADDMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| submod\<N\>  | SUBMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| mulmod\<N\>  | MULMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| divmod\<N\>  | DIVMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| iseqmod\<N\> | ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                                                                                                      |
| setup\<N\>   | SETUP_ADDSUBMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 0`, SETUP_MULDIVMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 1`, SETUP_ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 2` |

#### Complex Extension Field Arithmetic

| RISC-V Inst  | OpenVM Instruction                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| addcomplex   | ADD\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| subcomplex   | SUB\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| mulcomplex   | MUL\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| divcomplex   | DIV\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| setupcomplex | SETUP_ADDSUB_RV32\<Fp2\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 0`, SETUP_MULDIV_RV32\<Fp2\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 1` |

### Elliptic Curve Extension

| RISC-V Inst     | OpenVM Instruction                                                                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\>  | EC_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                               |
| sw_double\<C\>  | EC_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`                                                                                                      |
| setup\<C\>      | SETUP_EC_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) != 0`, SETUP_EC_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 0` |
| hint_decompress | PHANTOM `ind(rd), ind(rs1), phantom_c(curve_idx, HintDecompress)`                                                                                     |

### Pairing Extension

| RISC-V Inst                | OpenVM Instruction                                                       |
| -------------------------- | ------------------------------------------------------------------------ |
| miller_double_step         | MILLER_DOUBLE_STEP_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`                |
| miller_double_and_add_step | MILLER_DOUBLE_AND_ADD_STEP_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2` |
| fp12_mul                   | FP12_MUL_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                   |
| evaluate_line              | EVALUATE_LINE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`              |
| mul_013_by_013             | MUL_013_BY_013_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| mul_by_01234               | MUL_BY_01234_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| mul_023_by_023             | MUL_023_BY_023_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| mul_by_02345               | MUL_BY_02345_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
| hint_final_exp             | PHANTOM `ind(rs1), pairing_idx, phantom_c(pairing_idx, HintFinalExp)`    |
