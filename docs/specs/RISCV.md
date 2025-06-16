# RISC-V Custom Code for VM Extensions

VM extensions in OpenVM are supported in the Rust frontend by inserting custom RISC-V machine code to be passed through LLVM into the RISC-V ELF using a standard 32-bit RISC-V encoding. This document specifies the custom machine code used for the default set of VM extensions. This custom code will be transpiled to OpenVM assembly using the modular transpiler specified [here](./transpiler.md).
The default VM extensions that support transpilation are:

- [RV32IM](#rv32im-extension): An extension supporting the 32-bit RISC-V ISA with multiplication.
- [Keccak-256](#keccak-extension): An extension implementing the Keccak-256 hash function compatibly with RISC-V memory.
- [SHA2-256](#sha2-256-extension): An extension implementing the SHA2-256 hash function compatibly with RISC-V memory.
- [BigInt](#bigint-extension): An extension supporting 256-bit signed and unsigned integer arithmetic, including multiplication. This extension respects the RISC-V memory format.
- [Algebra](#algebra-extension): An extension supporting modular arithmetic over arbitrary fields and their complex field extensions. This extension respects the RISC-V memory format.
- [Elliptic curve](#elliptic-curve-extension): An extension for elliptic curve operations over Weierstrass curves, including addition and doubling. This can be used to implement multi-scalar multiplication and ECDSA scalar multiplication. This extension respects the RISC-V memory format.
- [Pairing](#pairing-extension): An extension containing opcodes used to implement the optimal Ate pairing on the BN254 and BLS12-381 curves. This extension respects the RISC-V memory format.

## Classification of Custom RISC-V Machine Code

We divide the types of custom RISC-V machine code associated with VM extensions into two categories:

- **Intrinsic Instruction:** the custom machine code is a single custom RISC-V instruction, compliant with the RISC-V specification.
- **Kernel Code:** the custom machine code is a 32-bit aligned binary sequence with bit length a multiple of 32. The machine code does not need to conform to any RISC-V ISA specification. Kernel code is used as a means to statically link foreign OpenVM assembly code into the ELF without a custom linker. Kernel code cannot be executed directly by a RISC-V machine without additional toolchain support from the [transpiler](./transpiler.md#openvm-kernel-code-transpilation).

## Conventions for RISC-V Intrinsic Instructions

The RISC-V instruction format used for intrinsic instructions in OpenVM conforms to the convention for non-standard ISA extensions described in Chapters 34-35 of the [RISC-V Instruction Set Manual Volume I: Unprivileged ISA](https://lf-riscv.atlassian.net/wiki/spaces/HOME/pages/16154769/RISC-V+Technical+Specifications) (Version 20240411) to avoid collisions with existing RISC-V extensions. The format is specified as follows:

- Intrinsics are non-standard brownfield ISA extensions of the 30-bit encoding space of the base ISA.
- Intrinsics use _custom-0_ opcode[6:0] prefix **0001011** and _custom-1_ opcode[6:0] prefix **0101011**. Intrinsics which do not require additional configuration parameters use _custom-0_, and ones which do (e.g., prime field arithmetic and elliptic curve arithmetic) use _custom-1_.
- We use funct3 as the top level distinguisher between opcode classes, and then funct7 (if R-type) or imm (if I-type or B-type) for more specific specification.

We now specify the custom instructions for the default set of VM extensions.

## System Instructions

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes    |
| ----------- | --- | ----------- | ------ | --------- | ------------------------------- |
| terminate   | I   | 0001011     | 000    | `code`    | terminate with exit code `code` |

The `terminate` instruction is a requested trap that will cause an orderly termination to guest execution with the specified exit code.

## RV32IM Extension

The RV32IM extension supports the RV32I Base Integer Instruction Set, Version 2.1 with `XLEN=32`
and the "M" Extension for Integer Multiplication and Division, Version 2.0 with `XLEN=32`, following
the specification of the [RISC-V Instruction Set Manual Volume I: Unprivileged ISA](https://lf-riscv.atlassian.net/wiki/spaces/HOME/pages/16154769/RISC-V+Technical+Specifications) (Version 20240411).

**Memory Alignment**: Chapter 2.6 of _loc. cit._ specifies that the behavior of
loads and stores whose effective addresses are not naturally aligned to the referenced datatype
(i.e., the effective address is not divisible by the size of the access in bytes) depends on the
execution environment interface (EEI). The OpenVM execution environment does not support misaligned
loads and stores. More specifically, guest execution considers misaligned accesses invalid
and host execution will raise an exception resulting in a fatal trap.

### IO
In addition to the standard RV32IM opcodes, we support the following additional intrinsic instructions to handle interactions between the guest and host environments.
These instructions require the host execution environment to maintain the following data structures
as part of its state:

- `input_stream`: a non-interactive queue of byte vectors which is provided at the start of
  execution. This may be considered as the non-interactive input to the guest program.
- `hint_stream`: a queue of bytes populated during execution
  via instructions such as `hintinput`.
- user IO space: a fixed length array of bytes, with length `num_public_values` which is a configuration constant of the execution environment. The length must equal `8` times a power of two. The IO space can be overwritten, and the final state of the IO space is persisted after execution halts.

The guest execution has no control over what the host provides in `input_stream` and `hint_stream`, so
the guest must take care to validate all data and account for behavior in cases of untrusted input.

| RISC-V Inst | FMT | opcode[6:0] | funct3 | imm[0:11] | RISC-V description and notes                                                                                                                                               |
| ----------- | --- | ----------- | ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hintstorew  | I   | 0001011     | 001    | 0x0       | Stores next 4-byte word from hint stream in user memory at `[rd]_2`. The address `rd` does not have any alignment requirements. |
| hintbuffer  | I   | 0001011     | 001    | 0x1       | Stores next `4 * rs1` bytes from hint stream in user memory at `[rd..rd + 4 * rs1]_2`. Only valid if next `4 * rs1` values in hint stream are bytes and `rs1` is non-zero. The address `rd` does not have any alignment requirements. |
| reveal      | I   | 0001011     | 010    |           | Stores the 4-byte word `rs1` at address `rd + imm` in user IO space. The address `rd + imm` must be aligned to a 4-byte boundary. |
| hintinput   | I   | 0001011     | 011    | 0x0       | Pop next vector from input stream and reset hint stream to the vector.                                                                                                     |
| printstr    | I   | 0001011     | 011    | 0x1       | Tries to convert `[rd..rd + rs1]_2` to UTF-8 string and print to host stdout. Will print error message if conversion fails.                                                |
| hintrandom  | I   | 0001011     | 011    | 0x2       | Resets the hint stream to `4 * rd` random bytes from `rand::rngs::OsRng` on the host.                                                                                      |

| RISC-V Inst  | FMT | opcode[6:0] | funct3  | funct7 | RISC-V description and notes                                                                                                 |
|--------------|-----|-------------|---------|--------|------------------------------------------------------------------------------------------------------------------------------|
| nativestorew | R   | 0001011     | 111     | 0x2    | Stores the 4-byte word `rs1` at address `rd` in native address space. The address `rd` must be aligned to a 4-byte boundary. |

`nativestorew` connects RV32 address space and native address space. We put it in RV32 extension because its 
implementation is here. But we use `funct3 = 111` because the native extension has an available slot.

## Keccak Extension

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                |
| ----------- | --- | ----------- | ------ | ------ | ------------------------------------------- |
| keccak256   | R   | 0001011     | 100    | 0x0    | `[rd:32]_2 = keccak256([rs1..rs1 + rs2]_2)` |

## SHA2-256 Extension

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes             |
| ----------- | --- | ----------- | ------ | ------ | ---------------------------------------- |
| sha256      | R   | 0001011     | 100    | 0x1    | `[rd:32]_2 = sha256([rs1..rs1 + rs2]_2)` |

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

## Native (Kernel) Extension

The following are _not_ intrinsic instructions, but custom RISC-V instructions used to frame the insertion of custom kernel code. They are not meant to be used alone. See the [transpiler](./transpiler.md#openvm-kernel-code-transpilation) for more details.

These use the _custom-0_ opcode prefix and funct3 = 0b111.

| RISC-V Inst | FMT | opcode[6:0] | funct3 | funct7 | RISC-V description and notes                              |
| ----------- | --- | ----------- | ------ | ------ | --------------------------------------------------------- |
| lfii        | R   | 0001011     | 111    | 0      | Long Form Instruction Indicator. `rd = rs1 = rs2 = 0`     |
| gi          | R   | 0001011     | 111    | 1      | Gap Indicator. `rd = rs1 = rs2 = 0`                       |

`nativestorew` also uses `funct3 = 111`. It's listed in the RV32 extension.

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

For each created modular class, one must call a corresponding `setup_*` function before using the intrinsics.
For example, for the structs above this would be `setup_0()` and `setup_1()`. This function generates the `setup` intrinsics which are distinguished by the `rs2` operand that specifies the chip this instruction is passed to.
For developer convenience, in the Rust function bindings for these intrinsics, each modulus's `setup_*` function is automatically called on the first use of any of its intrinsics.

We use `config.mod_idx(N)` to denote the index of `N` in this list. In the list below, `idx` denotes `config.mod_idx(N)`.

**Note:** The output for the first 4 instructions is not guaranteed to be less than `N`. See the [ISA specification](./ISA.md#algebra-extension) for more details.

| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------ | --- | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | R   | 0101011     | 000    | `idx*8`   | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 + [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                                                                                                              |
| submod\<N\>  | R   | 0101011     | 000    | `idx*8+1` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 - [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                                                                                                              |
| mulmod\<N\>  | R   | 0101011     | 000    | `idx*8+2` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 * [rs2: N::NUM_LIMBS]_2 (mod N)`                                                                                                                                                                                                                                                                                                                                              |
| divmod\<N\>  | R   | 0101011     | 000    | `idx*8+3` | `[rd: N::NUM_LIMBS]_2 = [rs1: N::NUM_LIMBS]_2 / [rs2: N::NUM_LIMBS]_2 (mod N)` (undefined when `gcd([rs2: N::NUM_LIMBS]_2, N) != 1`)                                                                                                                                                                                                                                                                                        |
| iseqmod\<N\> | R   | 0101011     | 000    | `idx*8+4` | `rd = [rs1: N::NUM_LIMBS]_2 == [rs2: N::NUM_LIMBS]_2 (mod N) ? 1 : 0`. If `rd != x0`, enforces that `[rs1: N::NUM_LIMBS]_2` and `[rs2: N::NUM_LIMBS]_2` are both less than `N` and then sets `rd` equal to boolean comparison value. If `rd = x0`, this is a no-op.                                                                                                                                                         |
| setup\<N\>   | R   | 0101011     | 000    | `idx*8+5` | `assert([rs1: N::NUM_LIMBS]_2 == N)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: N::NUM_LIMBS]_2` if `ind(rs2) = 0,1` (for add_sub, mul_div) or it overwrites the register value of `rd` with an unconstrained value if `ind(rs2) = 2` (for iseq). If `ind(rs2) = 2`, then the instruction is **invalid** if `rd = x0`. |
| hint_non_qr\<N\> | R   | 0101011     | 000    | `idx*8+6` | Reset the hint stream to equal `non_qr` where `non_qr` is a quadratic nonresidue modulo `N`. The same `non_qr` is returned in each execution of this instruction. `rd`, `rs1`, and `rs2` should be `x0`. |
| hint_sqrt\<N\> | R   | 0101011     | 000    | `idx*8+7` | Read `x = [rs1: N::NUM_LIMBS]_2`. If `x` is a quadratic residue modulo `N` then reset the hint stream to `[1u0, 0u8, 0u8, 0u8]` concatenated with a square root of `x`. If `x` is not a quadratic residue, then reset the hint stream to `[0u8; 4]` concatenated with a square root of `x * non_qr` where `non_qr` is the quadratic nonresidue returned by `hint_non_qr<N>`. `rd` and `rs2` should be `x0`. |

Since `funct7` is 7-bits, up to 16 moduli can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

### Complex Extension Field Arithmetic

Complex extension field arithmetic over `Fp2` depends on `Fp` where `-1` is not a quadratic residue. The extension can be configured to support `Fp2` arithmetic for a subset of the `Fp` with modular arithmetic enabled. We use **the same** `config.mod_idx(Fp::MODULUS)` to denote the index of `Fp2` in this list. In the list below, `idx` denotes `config.mod_idx(Fp::MODULUS)`.

| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                       |
| ------------ | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| addcomplex   | R   | 0101011     | 010    | `idx*8`   | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x + y` to `[rd..]_2`                                                                                                                          |
| subcomplex   | R   | 0101011     | 010    | `idx*8+1` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x - y` to `[rd..]_2`                                                                                                                          |
| mulcomplex   | R   | 0101011     | 010    | `idx*8+2` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x * y` to `[rd..]_2`                                                                                                                          |
| divcomplex   | R   | 0101011     | 010    | `idx*8+3` | Read `x: Fp2` from `[rs1..]_2` and `y: Fp2` from `[rs2..]_2`. Write `x / y` to `[rd..]_2`                                                                                                                          |
| setupcomplex | R   | 0101011     | 010    | `idx*8+4` | `assert([rs1: Fp::NUM_LIMBS]_2 == Fp::MODULUS)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: Fp::NUM_LIMBS]_2`. |

## Elliptic Curve Extension

The elliptic curve extension supports arithmetic over short Weierstrass curves, which requires specification of the elliptic curve `C`. The extension must be configured to support a fixed ordered list of supported curves. We use `config.curve_idx(C)` to denote the index of `C` in this list. In the list below, `idx` denotes `config.curve_idx(C)`.

| RISC-V Inst     | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------- | --- | ----------- | ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\>  | R   | 0101011     | 001    | `idx*8`   | `EcPoint([rd:2*C::COORD_SIZE]_2) = EcPoint([rs1:2*C::COORD_SIZE]_2) + EcPoint([rs2:2*C::COORD_SIZE]_2)`. Assumes that input affine points are not identity and do not have same x-coordinate.                                                                                                                                                                                                                                                                                                                                                                  |
| sw_double\<C\>  | R   | 0101011     | 001    | `idx*8+1` | `EcPoint([rd:2*C::COORD_SIZE]_2) = 2 * EcPoint([rs1:2*C::COORD_SIZE]_2)`. Assumes that input affine point is not identity. `rs2` is unused and must be set to `x0`.                                                                                                                                                                                                                                                                                                                                                                                            |
| setup\<C\>      | R   | 0101011     | 001    | `idx*8+2` | `assert([rs1: C::COORD_SIZE]_2 == C::MODULUS)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: 2*C::COORD_SIZE]_2`. If `ind(rs2) != 0`, then this instruction is setup for `sw_add_ne`. Otherwise it is setup for `sw_double`. When `ind(rs2) != 0` (add_ne), it is required for proper functionality that `[rs2: C::COORD_SIZE]_2 != [rs1: C::COORD_SIZE]_2`; otherwise (double), it is required that `[rs1 + C::COORD_SIZE: C::COORD_SIZE]_2 != C::Fp::ZERO` |

Since `funct7` is 7-bits, up to 16 curves can be supported simultaneously. We use `idx*8` to leave some room for future expansion.

## Pairing Extension

Instructions for accelerating optimal Ate pairing depend on a pairing friendly elliptic curve `C` and associated `Fp, Fp2, Fp12` and constant `XI: Fp2`. Presently only the curves BN254 and BLS12-381 are supported, with `pairing_idx(Bn254) = 0` and `pairing_idx(Bls12_381) = 1`. In the list below, `idx` denotes `pairing_idx(C)`.

| RISC-V Inst                | FMT | opcode[6:0] | funct3 | funct7       | RISC-V description and notes                                                                                                                                                                                                                                   |
| -------------------------- | --- | ----------- | ------ | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hint_final_exp             | R   | 0101011     | 011    | `idx*16`     | Read `p: Fp` from `[rs1..]_2` and `q: Fp2` from `[rs2..]_2`, then compute `f: Fp12 = multi_miller_loop(p, q)`. Reset the hint stream to equal `hint_final_exp(f) = (residue_witness, scaling_factor): (Fp12, Fp12)` flattened into bytes. `rd` should be `x0`. |
