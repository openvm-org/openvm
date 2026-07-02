# RISC-V ELF Transpiler

Author: <https://github.com/manh9203>

## 1. Introduction

### Scope: RISC-V ELF Transpiler framework

- [RISC-V docs](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md)
- [Transpiler docs](https://github.com/openvm-org/openvm/blob/main/docs/specs/transpiler.md)
- [Transpiler toolchain](https://github.com/openvm-org/openvm/tree/main/crates/toolchain/transpiler)
- [RV32IM transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/rv32im/transpiler)
- [Algebra transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/algebra/transpiler)
- [Bigint transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/bigint/transpiler)
- [Keccak256 transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/keccak256/transpiler)
- [Sha256 transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/sha256/transpiler)
- [Ecc transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/ecc/transpiler)
- [Pairing transpiler](https://github.com/openvm-org/openvm/tree/main/extensions/pairing/transpiler)

### Describe the main focus and any additional context

VM Extension (transpiler): general RISC-V transpiler framework - Transpiler trait

- Carefully document how x0 needs to be handled

VM Extension (intrinsic functions): document how to add new custom RISC-V instructions

- What assumptions / requirements are on these instructions, including in relation to other existing instructions and extensions
- Requirements on transpiler properties

## 2. Findings

Classify by **severity** according to [cantina criteria](https://docs.cantina.xyz/cantina-docs/cantina-competitions/judging-process/finding-severity-criteria) in terms of likelihood and impact.

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

### 2.1 `LOAD` instruction behavior does not comply with RISC-V specifications

**Severity:** Low

**Context:** <https://cantina.xyz/code/c486d600-bed0-4fc6-aed1-de759fd29fa2/findings/54>

**Description:** Our implementation of `LOAD` turns it into a `NOP` when `rd` is `x0`.
To comply with RISC-V specifications, we should still attempt the memory read (and potentially raise an exception), and then omit the register write if `rd` is `x0`.

**Proof of concept:** <https://github.com/openvm-org/openvm/blob/main/crates/toolchain/transpiler/src/util.rs#L59-L74>

**Recommendation:** We should enable transpilation of `LOAD` instructions with `rd = x0`, and then handle it in `LoadStoreAdapter::postprocess`. May consult the implementation of `jal` and `jalr` for how to handle `x0` in the postprocess stage.

**Resolution:** https://github.com/openvm-org/openvm/commit/95ea04e813e91c99c3d6cb21c003e4f75275ee82

### 2.2 `reveal` instruction doesn't match specs in `transpiler.md`

**Severity:** Informational

**Context:** Docs review on `transpiler.md` and `RISCV.md`

**Description:** The `reveal` RISC-V instruction is transpiled to `STOREW` [in the RV32IM transpiler](https://github.com/openvm-org/openvm/blob/main/extensions/rv32im/transpiler/src/lib.rs#L179-L191). However, `transpiler.md` specifies that `reveal` should be transpiled to a dedicated `REVEAL_RV32` opcode [here](https://github.com/openvm-org/openvm/blob/main/docs/specs/transpiler.md?plain=1#L45).

**Recommendation:** We should update the docs to match the actual implementation.

**Resolution:** https://github.com/openvm-org/openvm/commit/f35b9dd45abc990949c956e4d5cd78dfcbaf36f4

### 2.3 `setup<C>` and `hint_decompress` in ecc transpiler doesn't match specs in `transpiler.md`

**Severity:** Informational

**Context:** Docs review on `transpiler.md` and `ecc.md`

**Description:** The [`setup<C>`](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/transpiler/src/lib.rs#L69-L83) and [`hint_decompress`](https://github.com/openvm-org/openvm/blob/main/extensions/ecc/transpiler/src/lib.rs#L60-L68) implementations in the ecc transpiler don't match the specs in `transpiler.md` [here](https://github.com/openvm-org/openvm/blob/main/docs/specs/transpiler.md?plain=1#L169-L170). The specs should be

| RISC-V Inst     | OpenVM Instruction                                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| setup\<C\>      | SETUP_EC_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `ind(rs2) != 0`, SETUP_EC_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `ind(rs2) = 0` |
| hint_decompress | PHANTOM `ind(rs1), ind(rs2), phantom_c(curve_idx, HintDecompress)`                                                                                                |

**Recommendation:** We should update the specs to match the actual implementation.

**Resolution:** https://github.com/openvm-org/openvm/commit/f35b9dd45abc990949c956e4d5cd78dfcbaf36f4

### 2.4 `hint_final_exp` in pairing transpiler doesn't match specs in `RISCV.md` and `transpiler.md`

**Severity:** Informational

**Context:** Docs review on `RISCV.md` and `transpiler.md`

**Description:** The `hint_final_exp` implementation [in the pairing transpiler](https://github.com/openvm-org/openvm/blob/main/extensions/pairing/transpiler/src/lib.rs#L98-L107) doesn't match the specs in `RISCV.md` [here](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md?plain=1#L136) and `transpiler.md` [here](https://github.com/openvm-org/openvm/blob/main/docs/specs/transpiler.md?plain=1#L184). The actual implementation takes in `ind(rs1), ind(rs2), pairing_idx` while the specs doesn't mention `ind(rs2)` at all.

**Recommendation:** We should update the specs to match the actual implementation.

**Resolution:** https://github.com/openvm-org/openvm/commit/f35b9dd45abc990949c956e4d5cd78dfcbaf36f4

### 2.5 `funct7` of RISC-V instructions for pairing extension in `RISCV.md` is incorrect

**Severity:** Informational

**Context:** Docs review on `RISCV.md`

**Description:** The `funct7` of RISC-V instructions for the pairing extension in `RISCV.md` [is incorrect](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md?plain=1#L128-L136). It should be from `idx*16` to `idx*16 + 9` (`idx*16 + 5` is missing).

**Recommendation:** We should fix the typos in `RISCV.md`.

**Resolution:** https://github.com/openvm-org/openvm/commit/f35b9dd45abc990949c956e4d5cd78dfcbaf36f4

## 3. Discussion

Discussion is for general discussion or additional writing about what has been studied, considered, or understood that did not result in a concrete finding. Discussions are useful both to see what were important areas that are security critical and why they were satisfied. In the good case, the review should have few findings but lots of discussion.

### 3.1 `x0` handling

Currently, we have two way to handle a RISC-V instruction that has `rd = x0`:

1. Transpile to a `NOP` if it has no side effects
2. Transpile to an OpenVM instruction that executes the expected side effect and does not change the value of `[0:4]_1` (side effect here could include raising an exception)

For the available RISC-V instructions that OpenVM supports, it is guaranteed that only `rd` can be written to during its execution, so the two cases above are sufficient. However, we need to carefully reconsider that every time we add a new instruction.

On the second case, the following instructions that allow `rd = x0` and can still possibly write into `rd`:

- `setup<N>`: allow writing into `rd` if `rs2 = 2`. If `rd = x0`, the instruction becomes **invalid** (from [RISC-V specs](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md?plain=1#L93)).

>| RISC-V Inst  | FMT | opcode[6:0] | funct3 | funct7    | RISC-V description and notes                                                                                                                                                                                                                                                                                                    |
>| ------------ | --- | ----------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
>| setup\<N\>   | R   | 0101011     | 000    | `idx*8+5` | `assert([rs1: N::NUM_LIMBS]_2 == N)` in the chip defined by the register index of `rs2`. For the sake of implementation convenience it also writes an unconstrained value into `[rd: N::NUM_LIMBS]_2` if `ind(rs2) = 0,1` (for add_sub, mul_div) or it overwrites the register value of `rd` with an unconstrained value if `ind(rs2) = 2` (for iseq). If `ind(rs2) = 2`, then the instruction is **invalid** if `rd = x0`. |

- `jal` and `jalr`: these instructions load the return address into `rd`. They are transpiled into OpenVM instructions with `f = (rd != 0)` as a flag to handle `x0` during execution if `rd = x0` (from [transpiler specs](https://github.com/openvm-org/openvm/blob/main/docs/specs/transpiler.md?plain=1#L87-L88)).

>| RISC-V Inst | OpenVM Instruction                                                         |
>| ----------- | -------------------------------------------------------------------------- |
>| jal         | JAL_RV32 `ind(rd), 0, itof(imm), 1, 0, (rd != x0)`                         |
>| jalr        | JALR_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 0, (rd != x0)` |

- `load` instructions: these instructions load a value from memory into `rd`. They are transpiled into OpenVM instructions with `f = (rd != x0)` as a flag to handle `x0` during execution if `rd = x0`. That change is a part of fixing [Finding 2.1](#21-load-instruction-behavior-does-not-comply-with-risc-v-specifications).

>| RISC-V Inst | OpenVM Instruction                                                                                 |
>| ----------- | -------------------------------------------------------------------------------------------------- |
>| lb          | LOADB_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`          |
>| lh          | LOADH_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`          |
>| lw          | LOADW_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`          |
>| lbu         | LOADBU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`         |
>| lhu         | LOADHU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`         |

### 3.2 Adding new custom RISC-V instructions

From [RISC-V specs](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md), the custom RISC-V instructions need to conform to the extension convention in the [RISC-V spec v2.2](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf) (Chapter 21) to avoid collisions with existing RISC-V extensions. The format is specified as follows:

- Intrinsics use _custom-0_ opcode[6:0] prefix **0001011** and _custom-1_ opcode[6:0] prefix **0101011**. Intrinsics which do not require additional configuration parameters use _custom-0_, and ones which do (e.g., prime field arithmetic and elliptic curve arithmetic) use _custom-1_.
- We use funct3 as the top level distinguisher between opcode classes, and then funct7 (if R-type) or imm (if I-type or B-type) for more specific specification.

### 3.3 Handling address spaces, memory, and the program counter

OpenVM uses address spaces to represent different memory regions on RISC-V-based machines. Specifically, it designates address space 1 for registers and address space 2 for memory, as defined in the [Design and Specification docs](https://github.com/openvm-org/openvm/blob/main/docs/specs/README.md?plain=1#L10). According to the [RISC-V spec](https://github.com/openvm-org/openvm/blob/main/docs/specs/RISCV.md), these address spaces are correctly mapped to register and memory usage during the execution of RISC-V instructions.

The program counter is not yet documented in the main branch, but the necessary updates are included in this [PR](https://github.com/axiom-crypto/openvm-private/pull/20).
