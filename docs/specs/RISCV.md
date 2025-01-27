# RISC-V Custom Instructions for VM Extensions

VM extensions in OpenVM consisting of intrinsics are supported in the Rust frontend by inserting custom RISC-V directives to be passed through LLVM into the RISC-V ELF using a standard 32-bit RISC-V encoding. This document specifies the custom instruction format used for the default set of intrinsic VM extensions. These custom instructions will be transpiled to OpenVM assembly using the modular transpiler specified [here](./transpiler.md).

The custom instruction format in OpenVM conforms to the extension convention in the [RISC-V spec v2.2](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf) (Chapter 21) to avoid collisions with existing RISC-V extensions. The format is specified as follows:

- Intrinsics use _custom-0_ opcode[6:0] prefix **0001011** and _custom-1_ opcode[6:0] prefix **0101011**. Intrinsics which do not require additional configuration parameters use _custom-0_, and ones which do (e.g., prime field arithmetic and elliptic curve arithmetic) use _custom-1_.
- We use funct3 as the top level distinguisher between opcode classes, and then funct7 (if R-type) or imm (if I-type or B-type) for more specific specification. In the tables below, the funct7 column specifies the value of imm[11:0] when the instruction is I-type.

We now specify the custom instructions for the default set of VM extensions.

## System Instructions

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes                                                                                                |
| ----------- | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------- |
| terminate   | I   | 0001011     | 000    | `code`    | terminate with exit code `code`                                                                                             |

## RV32IM Extension

In addition to the standard RV32IM opcodes, we support the following additional instructions to handle system interactions

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes                                                                                                |
| ----------- | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------- |
| hintstorew  | I   | 0001011     | 001    | 0x0       | Stores next 4-byte word from hint stream in user memory at `[rd]_2`. Only valid if next 4 values in hint stream are bytes.                                  |
| hintbuffer  | I   | 0001011     | 001    | 0x1       | Stores next `4 * rs1` bytes from hint stream in user memory at `[rd..rd + 4 * rs1]_2`. Only valid if next `4 * rs1` values in hint stream are bytes and `rs1` is  non-zero.                                 |
| reveal      | I   | 0001011     | 010    |           | Stores the 4-byte word `rs1` at address `rd + imm` in user IO space.                                                        |
| hintinput   | I   | 0001011     | 011    | 0x0       | Pop next vector from input stream and reset hint stream to the vector.                                                      |
| printstr    | I   | 0001011     | 011    | 0x1       | Tries to convert `[rd..rd + rs1]_2` to UTF-8 string and print to host stdout. Will print error message if conversion fails. |
| hintrandom  | I   | 0001011     | 011    | 0x2       | Resets the hint stream to `4 * rd` random bytes from `rand::rngs::OsRng` on the host. |

## Keccak Extension

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                |
| ----------- | --- | ----------- | ------ | ------ | ------------------------------------------- |
| keccak256   | R   | 0001011     | 100    | 0x0    | `[rd:32]_2 = keccak256([rs1..rs1 + rs2]_2)` |

## SHA2-256 Extension

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                |
| ----------- | --- | ----------- | ------ | ------ | ------------------------------------------- |
| sha256      | R   | 0001011     | 100    | 0x1    | `[rd:32]_2 = sha256([rs1..rs1 + rs2]_2)`    |

## BigInt Extension

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

## Algebra Extension

Modular arithmetic instructions depend on the modulus `N`. The ordered list of supported moduli should be saved in the `.openvm` section of the ELF file in the serialized format. This is achieved by the `moduli_declare!` macro; for example, the following code

```rust
moduli_declare! {
    Bls12381 { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bn254 { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}
```

generates classes `Bls12381` and `Bn254` that represent the elements of the corresponding modular fields. Hexadecimal and decimal formats are supported.

### Field Arithmetic

For each created modular class, one must call a corresponding `setup_*` function once at the beginning of the program. For example, for the structs above this would be `setup_0()` and `setup_1()`. This function generates the `setup` intrinsics which are distinguished by the `rs2` operand that specifies the chip this instruction is passed to..

We use `config.mod_idx(N)` to denote the index of `N` in this list. In the list below, `idx` denotes `config.mod_idx(N)`.

**Note:** The output for the first 4 instructions is not guaranteed to be less than `N`. See the [ISA specification](./ISA.md#algebra-extension) for more details.

| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                    |
| ------------ | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | R   | 0101011     | 000    | `idx*8`   | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 + [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| submod\<N\>  | R   | 0101011     | 000    | `idx*8+1` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 - [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| mulmod\<N\>  | R   | 0101011     | 000    | `idx*8+2` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 * [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                  |
| divmod\<N\>  | R   | 0101011     | 000    | `idx*8+3` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 / [rs2: N::NUM_LIMBS]_2 (mod N)` (undefined when `gcd([rs2: N::NUM_LIMBS]_2, N) != 1`)                                                                                                                                                                                            |
| iseqmod\<N\> | R   | 0101011     | 000    | `idx*8+4` | `rd = [rs1: N::NUM_LIMBS]_2 == [rs2: N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. If `rd != x0`, enforces that `[rs1: N::NUM_LIMBS]_2` and `[rs2: N::NUM_LIMBS]_2` are both less than `N` and then sets `rd` equal to boolean comparison value. If `rd = x0`, this is a no-op.                                   |
| setup\<N\>   | R   | 0101011     | 000    | `idx*8+5` | `assert([rs1: N::NUM_LIMBS]_2 == N)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: N::NUM_LIMBS]_2` if `ind(rs2) = 0,1` (for add_sub, mul_div) or it overwrites the register value of `rd` with an unconstrained value if `ind(rs2) = 2` (for iseq). If `ind(rs2) = 2`, then the instruction is **invalid** if `rd = x0`. |

Since `funct7` is 7-bits, up to 16 moduli can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

### Complex Extension Field Arithmetic

Complex extension field arithmetic over `Fp2` depends on `Fp` where `-1` is not a quadratic residue. The extension can be configured to support `Fp2` arithmetic for a subset of the `Fp` with modular arithmetic enabled. We use **the same** `config.mod_idx(Fp::MODULUS)` to denote the index of `Fp2` in this list. In the list below, `idx` denotes `config.mod_idx(Fp::MODULUS)`.

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                              |
| ----------- | --- | ----------- | ------ | --------- | ----------------------------------------------------------------------------------------- |
| addcomplex  | R   | 0101011     | 010    | `idx*8`   | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x + y` to `[rd..]_2` |
| subcomplex  | R   | 0101011     | 010    | `idx*8+1` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x - y` to `[rd..]_2` |
| mulcomplex  | R   | 0101011     | 010    | `idx*8+2` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x * y` to `[rd..]_2` |
| divcomplex  | R   | 0101011     | 010    | `idx*8+3` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x / y` to `[rd..]_2` |
| setupcomplex| R   | 0101011     | 010    | `idx*8+4` | `assert([rs1: Fp::NUM_LIMBS]_2 == Fp::MODULUS)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: Fp::NUM_LIMBS]_2`. |

## Elliptic Curve Extension

The elliptic curve extension supports arithmetic over short Weierstrass curves, which requires specification of the elliptic curve `C`. The extension must be configured to support a fixed ordered list of supported curves. We use `config.curve_idx(C)` to denote the index of `C` in this list. In the list below, `idx` denotes `config.curve_idx(C)`.

| RISC-V Inst     | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\>  | R   | 0101011     | 001    | `idx*8`   | `EcPoint([rd:2*C::COORD_SIZE]_2) = EcPoint([rs1:2*C::COORD_SIZE]_2) + EcPoint([rs2:2*C::COORD_SIZE]_2)`. Assumes that input affine points are not identity and do not have same x-coordinate.                                                                                                                                                                                                                                                                                                                                                                       |
| sw_double\<C\>  | R   | 0101011     | 001    | `idx*8+1` | `EcPoint([rd:2*C::COORD_SIZE]_2) = 2 * EcPoint([rs1:2*C::COORD_SIZE]_2)`. Assumes that input affine point is not identity. `rs2` is unused and must be set to `x0`.                                                                                                                                                                                                                                                                                                                                                                                                 |
| setup\<C\>      | R   | 0101011     | 001    | `idx*8+2` | `assert([rs1: C::COORD_SIZE]_2 == C::MODULUS)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: 2*C::COORD_SIZE]_2`. If `ind(rs2) != 0`, then this instruction is setup for `sw_add_ne`. Otherwise it is setup for `sw_double`. When `ind(rs2) != 0` (add_ne), it is required for proper functionality that `[rs2: C::COORD_SIZE]_2 != [rs1: C::COORD_SIZE]_2`; otherwise (double), it is required that `[rs1 + C::COORD_SIZE: C::COORD_SIZE]_2 != C::Fp::ZERO` |
| hint_decompress | R   | 0101011     | 001    | `idx*8+3` | Read `x: C::Fp` from `[rs1: C::COORD_SIZE]_2` and `rec_id: u8` from `[rs2]_2`. Reset the hint stream to equal the unique `y: C::Fp` such that `(x, y)` is a point on `C` and `y` has the same parity as `rec_id`, if it exists. Otherwise reset hint stream to arbitrary `C::Fp`. `rd` should be `x0`.                                                                                                                                                                                                                                                              |

Since `funct7` is 7-bits, up to 16 curves can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Pairing Extension

Instructions for accelerating optimal Ate pairing depend on a pairing friendly elliptic curve `C` and associated `Fp, Fp2, Fp12` and constant `XI: Fp2`. Presently only the curves BN254 and BLS12-381 are supported, with `pairing_idx(Bn254) = 0` and `pairing_idx(Bls12_381) = 1`. In the list below, `idx` denotes `pairing_idx(C)`.

| RISC-V Inst                | FMT | opcode[6:0] | funct3 | funct7        | RISC-V description and notes                                                                                                                                                                                                 |
| -------------------------- | --- | ----------- | ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| miller_double_step         | R   | 0101011     | 011    | `idx*16`      | Read `S: EcPoint<Fp2>` from `[rs1..]_2`. Write `miller_double_step(S): (EcPoint<Fp2>, UnevaluatedLine<Fp2>)` to `[rd..]_2`. `rs2` must be zero.                                                                              |
| miller_double_and_add_step | R   | 0101011     | 011    | `idx*16 + 1`  | Read `S: EcPoint<Fp2>` from `[rs1..]_2` and `Q: EcPoint<Fp2>` from `[rs2..]_2`. Write `miller_double_and_add_step(S, Q): (EcPoint<Fp2>, UnevaluatedLine<Fp2>, UnevaluatedLine<Fp2>)` to `[rd..]_2`.                          |
| fp12_mul                   | R   | 0101011     | 011    | `idx*16 + 2`  | Read `x: Fp12` from `[rs1..]_2` and `y: Fp12` from `[rs2..]_2`. Write `x * y: Fp12` to `[rd..]_2`.                                                                                                                           |
| evaluate_line              | R   | 0101011     | 011    | `idx*16 + 3`  | Read `line: UnevaluatedLine<Fp2>` from `[rs1..]_2` and `(x_over_y, x_inv): (Fp, Fp)` from `[rs2..]_2`. Write `evaluate_line(line, x_over_y, x_inv): EvaluatedLine<Fp2>` to `[rd..]_2`.                                       |
| mul_013_by_013             | R   | 0101011     | 011    | `idx*16 + 4`  | Read `line_0: EvaluatedLine<Fp2>` from `[rs1..]_2` and `line_1: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_013_by_013(line_0, line_1): [Fp2; 5]` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **D-type**. |
| mul_by_01234               | R   | 0101011     | 011    | `idx*16 + 6`  | Read `f: Fp12` from `[rs1..]_2` and `x: [Fp2; 5]` from `[rs2..]_2`. Write `mul_by_01234(f, x): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **D-type**.                                                   |
| mul_023_by_023             | R   | 0101011     | 011    | `idx*16 + 7`  | Read `line_0: EvaluatedLine<Fp2>` from `[rs1..]_2` and `line_1: EvaluatedLine<Fp2>` from `[rs2..]_2`. Write `mul_023_by_023(line_0, line_1): [Fp2; 5]` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **M-type**. |
| mul_by_02345               | R   | 0101011     | 011    | `idx*16 + 9`  | Read `f: Fp12` from `[rs1..]_2` and `x: [Fp2; 5]` from `[rs2..]_2`. Write `mul_by_02345(f, x): Fp12` to `[rd..]_2`. Only enabled if the sextic twist of `C` is **M-type**.                                                   |
| hint_final_exp             | R   | 0101011     | 011    | `idx*16 + 10` | Read `f: Fp12` from `[rs1..]_2` and reset hint stream to equal `hint_final_exp(f) = (residue_witness, scaling_factor): (Fp12, Fp12)` flattened into bytes. `rd, rs2` should be `x0`.                                         |

