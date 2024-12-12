# RISC-V Custom Instructions

OpenVM intrinsic opcodes are callable from a RISC-V ELF as custom RISC-V instructions.
For these instructions, we will use the standard 32-bit RISC-V encoding unless otherwise specified.
We follow Chapter 21 of the [RISC-V spec v2.2](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf) on how to extend the ISA, aiming to avoid collisions with existing standard instruction formats. As suggested by Chapter 19, for instructions which fit into 32-bits, we will use the _custom-0_ opcode[6:0] prefix **0001011** and _custom-1_ opcode[6:0] prefix **0101011**. Note that instructions are parsed from right to left, and opcode[1:0] = 11 is typical for standard 32-bit instructions (opcode[1:0]=00 is used for compressed 16-bit instructions). We will use _custom-0_ for intrinsics that don’t require additional configuration parameters, and _custom-1_ for ones that do (e.g., prime field arithmetic and elliptic curve arithmetic).

Almost all intrinsics will be R-type or I-type.

R-type format: `[funct7] [rs2] [rs1] [funct3] [rd] [opcode]`.
I-type format: `[imm[11:0]] [rs1] [funct3] [rd] [opcode]`.

We will use funct3 as the top level distinguisher between opcode classes, and then funct7 (if R-type) or imm (if I-type) for more specific specification. In the tables below, the funct7 column will specify the value of imm[11:0] when the instruction is I-type.

We start with the instructions using _custom-0_ opcode[6:0] prefix **0001011**..

## System

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes                                                                                                |
| ----------- | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------- |
| terminate   | I   | 0001011     | 000    | `code`    | terminate with exit code `code`                                                                                             |
| hintstorew  | I   | 0001011     | 001    |           | Stores next 4-byte word from hint stream in user memory at `[rd + imm]_2` (`i32` addition).                                 |
| reveal      | I   | 0001011     | 010    |           | Stores the 4-byte word `rs1` at address `rd + imm` in user IO space.                                                        |
| hintinput   | I   | 0001011     | 011    | 0x0       | Pop next vector from input stream and reset hint stream to the vector.                                                      |
| printstr    | I   | 0001011     | 011    | 0x1       | Tries to convert `[rd..rd + rs1]_2` to UTF-8 string and print to host stdout. Will print error message if conversion fails. |

## Hashes

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                |
| ----------- | --- | ----------- | ------ | ------ | ------------------------------------------- |
| keccak256   | R   | 0001011     | 100    | 0x0    | `[rd:32]_2 = keccak256([rs1..rs1 + rs2]_2)` |

## 256-bit Integers

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                              |
| ----------- | --- | ----------- | ------ | ------ | --------------------------------------------------------- |
| add256      | R   | 0001011     | 101    | 0x00   | `[rd:32]_2 = [rs1:32]_2 + [rs2:32]_2`                     |
| sub256      | R   | 0001011     | 101    | 0x01   | `[rd:32]_2 = [rs1:32]_2 - [rs2:32]_2`                     |
| xor256      | R   | 0001011     | 101    | 0x02   | `[rd:32]_2 = [rs1:32]_2 ^ [rs2:32]_2`                     |
| or256       | R   | 0001011     | 101    | 0x03   | `[rd:32]_2 = [rs1:32]_2 \| [rs2:32]_2`                    |
| and256      | R   | 0001011     | 101    | 0x04   | `[rd:32]_2 = [rs1:32]_2 & [rs2:32]_2`                     |
| sll256      | R   | 0001011     | 101    | 0x05   | `[rd:32]_2 = [rs1:32]_2 << [rs2:32]_2`                    |
| srl256      | R   | 0001011     | 101    | 0x06   | `[rd:32]_2 = [rs1:32]_2 >> [rs2:32]_2`                    |
| sra256      | R   | 0001011     | 101    | 0x07   | `[rd:32]_2 = [rs1:32]_2 >> [rs2:32]_2` MSB extends        |
| slt256      | R   | 0001011     | 101    | 0x08   | `[rd:32]_2 = i256([rs1:32]_2) < i256([rs2:32]_2) ? 1 : 0` |
| sltu256     | R   | 0001011     | 101    | 0x09   | `[rd:32]_2 = u256([rs1:32]_2) < u256([rs2:32]_2) ? 1 : 0` |
| mul256      | R   | 0001011     | 101    | 0x10   | `[rd:32]_2 = ([rs1:32]_2 * [rs2:32]_2)[0:255]`            |

We support a single branch instruction, `beq256`, which is B-type.

| RISC-V Inst | FMT | opcode[6:0] | funct3 | RISC-V description and notes             |
| ----------- | --- | ----------- | ------ | ---------------------------------------- |
| beq256      | B   | 0001011     | 110    | `if([rs1:32]_2 == [rs2:32]_2) pc += imm` |

## Modular Arithmetic

We next proceed to the instructions using _custom-1_ opcode[6:0] prefix **0101011**..

Modular arithmetic instructions depend on the modulus `N`. The ordered list of supported moduli should be saved in the `.openvm` section of the ELF file in the serialized format. This is achieved by the `moduli_declare!` macro: for example, the following code

```rust
moduli_declare! {
    Bls12381 { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bn254 { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}
```

generates classes `Bls12381` and `Bn254` that represent the elements of the corresponding modular fields. Hexadecimal and decimal formats are supported.

For each created modular class, one must call a corresponding `setup_*` function once at the beginning of the program. For example, for the structs above this would be `setup_0()` and `setup_1()`. This function generates the `setup` intrinsics which are distinguished by the `rs2` operand that specifies the chip this instruction is passed to..

We use `config.mod_idx(N)` to denote the index of `N` in this list. In the list below, `idx` denotes `config.mod_idx(N)`.

**Note:** The output for the first 4 instructions is not guaranteed to be less than `N`. See [above](#modular-arithmetic) for more details.

| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                    |
| ------------ | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | R   | 0101011     | 000    | `idx*8`   | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 + [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| submod\<N\>  | R   | 0101011     | 000    | `idx*8+1` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 - [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| mulmod\<N\>  | R   | 0101011     | 000    | `idx*8+2` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 * [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| divmod\<N\>  | R   | 0101011     | 000    | `idx*8+3` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 / [rs2: N::NUM_LIMBS]_2 (mod N)` (undefined when `gcd([rs2: N::NUM_LIMBS]_2, N) != 1`)                                                                                                                                                                                            |
| iseqmod\<N\> | R   | 0101011     | 000    | `idx*8+4` | `rd = [rs1: N::NUM_LIMBS]_2 == [rs2: N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. Enforces that `[rs1: N::NUM_LIMBS]_2` and `[rs2: N::NUM_LIMBS]_2` are both less than `N` and then sets `rd` equal to boolean comparison value.                                                                                                           |
| setup\<N\>   | R   | 0101011     | 000    | `idx*8+5` | `assert([rs1: N::NUM_LIMBS]_2 == N)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes something (can be anything) into `[rd: N::NUM_LIMBS]_2` if `ind(rs2) = 0,1` (for add_sub, mul_div) or it overwrites the register value of `rd` if `ind(rs2) = 2` (for iseq). |

Since `funct7` is 7-bits, up to 16 moduli can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Short Weierstrass Elliptic Curve Arithmetic

Short Weierstrass elliptic curve arithmetic depends on elliptic curve `C`. The instruction set and VM can be simultaneously configured _ahead of time_ to support a fixed ordered list of supported curves. We use `config.curve_idx(C)` to denote the index of `C` in this list. In the list below, `idx` denotes `config.curve_idx(C)`.

| RISC-V Inst     | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------- | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\>  | R   | 0101011     | 001    | `idx*8`   | `EcPoint([rd:2*C::COORD_SIZE]_2) = EcPoint([rs1:2*C::COORD_SIZE]_2) + EcPoint([rs2:2*C::COORD_SIZE]_2)`. Assumes that input affine points are not identity and do not have same x-coordinate.                                                                                                                                                                                                                                                                                                                                                                                           |
| sw_double\<C\>  | R   | 0101011     | 001    | `idx*8+1` | `EcPoint([rd:2*C::COORD_SIZE]_2) = 2 * EcPoint([rs1:2*C::COORD_SIZE]_2)`. Assumes that input affine point is not identity. `rs2` is unused and must be set to `x0`.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| setup\<C\>      | R   | 0101011     | 001    | `idx*8+2` | `assert([rs1: C::COORD_SIZE]_2 == C::MODULUS)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes something (can be anything) into `[rd: 2*C::COORD_SIZE]_2`. It is required for proper functionality that `[rs1 + C::COORD_SIZE: C::COORD_SIZE]_2 != C::Fp::ZERO`. If `ind(rs2) != 0`, then this instruction is setup for `sw_add_ne`. Otherwise it is setup for `sw_double`. When `ind(rs2) != 0`, it is required that `[rs2: C::COORD_SIZE]_2 != C::MODULUS` and `[rs2 + C::COORD_SIZE: C::COORD_SIZE]_2 != C::Fp::ZERO`. |
| hint_decompress | R   | 0101011     | 001    | `idx*8+3` | Read `x: C::Fp` from `[rs1: C::COORD_SIZE]_2` and `rec_id: u8` from `[rs2]_2`. Reset the hint stream to equal the unique `y: C::Fp` such that `(x, y)` is a point on `C` and `y` has the same parity as `rec_id`, if it exists. Otherwise reset hint stream to arbitrary `C::Fp`. `rd` should be `x0`.                                                                                                                                                                                                                                                                                  |

Since `funct7` is 7-bits, up to 16 curves can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Complex Extension Field Arithmetic

Complex extension field arithmetic over `Fp2` depends on `Fp` where `-1` is not a quadratic residue. The instruction set and VM can be simultaneously configured _ahead of time_ to support `Fp2` arithmetic for a subset of the `Fp` with modular arithmetic enabled. We use **the same** `config.mod_idx(Fp::MODULUS)` to denote the index of `Fp2` in this list. In the list below, `idx` denotes `config.mod_idx(Fp::MODULUS)`.

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                              |
| ----------- | --- | ----------- | ------ | --------- | ----------------------------------------------------------------------------------------- |
| add         | R   | 0101011     | 010    | `idx*8`   | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x + y` to `[rd..]_2` |
| sub         | R   | 0101011     | 010    | `idx*8+1` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x - y` to `[rd..]_2` |
| mul         | R   | 0101011     | 010    | `idx*8+2` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x * y` to `[rd..]_2` |
| div         | R   | 0101011     | 010    | `idx*8+3` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x / y` to `[rd..]_2` |

## Optimal Ate Pairing

Instruction for accelerating optimal Ate pairing depend on a pairing friend elliptic curve `C` and associated `Fp, Fp2, Fp12` and constant `XI: Fp2`. Presently only the curves BN254 and BLS12-381 are supported, with `pairing_idx(Bn254) = 0` and `pairing_idx(Bls12_381) = 1`. In the list below, `idx` denotes `pairing_idx(C)`.

| RISC-V Inst                | FMT | opcode[6:0] | funct3 | funct7        | RISC-V description and notes                                                                                                                                                                                                 |
| -------------------------- | --- | ----------- | ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| miller_double_step         | R   | 0101011     | 011    | `idx*16`      | Read `S: EcPoint<Fp2>` from `[rs1..]_2`. Write `miller_double_step(S): (EcPoint<Fp2>, UnevaluatedLine<Fp2>)` to `[rd..]_2`. `rs2` must be zero.                                                                              |
| miller_double_and_add_step | R   | 0101011     | 011    | `idx*16 + 1`  | Read `S: EcPoint<Fp2>` from `[rs1..]_2` and `Q: EcPoint<Fp2>` from `[rs2..]_2`. Write `miller_double_and_add_step(S, Q): (EcPoint<Fp2>, UnevaluatedLine<Fp2>, UnevaluatedLine<Fp2>)` to `[rd..]_2`.                          |
| fp12_mul                   | R   | 0101011     | 011    | `idx*16 + 2`  | Read `x: Fp12` from `[rs1..]_2` and `y: Fp12` from `[rs2..]_2`. Write `x * y: Fp12` to `[rd..]_2`.                                                                                                                           |
| evaluate_line              | R   | 0101011     | 011    | `idx*16 + 3`  | Read `line: UnevaluatedLine<Fp2>` from `[rs1..]_2` and `(x_over_y, x_inv): (Fp, Fp)` from `[rs2..]_2`. Write `evaluate_line(line, x_over_y, x_inv): EvaluatedLine<Fp2>` to `[rd..]_2`.                                       |
| mul_013_by_013             | R   | 0101011     | 011    | `idx*16 + 4`  | Read `line_0: EvaluatedLine<Fp2>` from `[rs1..]_2` and `line_1: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_013_by_013(line_0, line_1): [Fp2; 5]` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **D-type**. |
| mul_by_013                 | R   | 0101011     | 011    | `idx*16 + 5`  | Read `f: Fp12` from `[rs1..]_2` and `line: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_by_013(f, line): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **D-type**.                                     |
| mul_by_01234               | R   | 0101011     | 011    | `idx*16 + 6`  | Read `f: Fp12` from `[rs1..]_2` and `x: [Fp2; 5]` from `[rs2..]_2`. Write `mul_by_01234(f, x): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **D-type**.                                                   |
| mul_023_by_023             | R   | 0101011     | 011    | `idx*16 + 7`  | Read `line_0: EvaluatedLine<Fp2>` from `[rs1..]_2` and `line_1: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_023_by_023(line_0, line_1): [Fp2; 5]` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **M-type**. |
| mul_by_023                 | R   | 0101011     | 011    | `idx*16 + 8`  | Read `f: Fp12` from `[rs1..]_2` and `line: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_by_023(f, line): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **M-type**.                                     |
| mul_by_02345               | R   | 0101011     | 011    | `idx*16 + 9`  | Read `f: Fp12` from `[rs1..]_2` and `x: [Fp2; 5]` from `[rs2..]_2`. Write `mul_by_02345(f, x): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **M-type**.                                                   |
| hint_final_exp             | R   | 0101011     | 011    | `idx*16 + 10` | Read `f: Fp12` from `[rs1..]_2` and reset hint stream to equal `hint_final_exp(f) = (residue_witness, scaling_factor): (Fp12, Fp12)` flattened into bytes. `rd, rs2` should be `x0`.                                         |

# RISC-V to OpenVM Transpilation

We describe the transpilation of the RV32IM instruction set and our [custom RISC-V instructions](#risc-v-custom-instructions) to the OpenVM instruction set.

We use the following notation for the transpilation:

- Let `ind(rd)` denote the register index, which is in `0..32`. In particular, it fits in one field element.
- We use `itof` for the function that takes 12-bits (or 21-bits in case of J-type) to a signed integer and then mapping to the corresponding field element. So `0b11…11` goes to `-1` in `F`.
- We use `sign_extend_24` to convert a 12-bit integer into a 24-bit integer via sign extension. We use this in conjunction with `utof`, which converts 24 bits into an unsigned integer and then maps it to the corresponding field element. Note that each 24-bit unsigned integer fits in one field element.
- We use `sign_extend_16` for the analogous conversion into a 16-bit integer via sign extension.
- We use `zero_extend_24` to convert an unsigned integer with at most 24 bits into a 24-bit unsigned integer by zero extension. This is used in conjunction with `utof` to convert unsigned integers to field elements.
- The notation `imm[0:4]` means the lowest 5 bits of the immediate.

The transpilation will only be valid for programs where:

- The program code does not have program address greater than or equal to `2^PC_BITS`.
- The program does not access memory outside the range `[0, 2^addr_max_bits)`.

## RV32IM Transpilation

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

## Custom Instruction Transpilation

| RISC-V Inst    | OpenVM Instruction                                               |
| -------------- | ---------------------------------------------------------------- |
| terminate      | TERMINATE `_, _, utof(imm)`                                      |
| hintstorew     | HINTSTOREW_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 2`    |
| reveal         | REVEAL_RV32 `0, ind(rd), utof(sign_extend_16(imm)), 1, 3`        |
| hintinput      | PHANTOM `_, _, HintInputRv32 as u16`                             |
| printstr       | PHANTOM `ind(rd), ind(rs1), PrintStrRv32 as u16`                 |
| keccak256      | KECCAK256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`               |
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
| addmod\<N\>    | ADDMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| submod\<N\>    | SUBMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| mulmod\<N\>    | MULMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| divmod\<N\>    | DIVMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`             |
| iseqmod\<N\>   | ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`            |
| setup\<N\>     | SETUP_ADDSUB,MULDIV,ISEQ_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` |
| sw_add_ne\<C\> | SW_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`          |
| sw_double\<C\> | SW_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`                 |
| hint_final_exp | PHANTOM `ind(rs1), pairing_idx, HintFinalExp as u16`             |
