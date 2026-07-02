# Instruction Set Architecture

Author: [Xinding Wei](https://github.com/nyunyunyunyu)

## 1. Introduction

Scope: https://github.com/openvm-org/openvm
Commit: 336f1a475e5aa3513c4c5a266399f4128c119bba

## 2. Findings

Classify by **severity** according to [cantina criteria](https://docs.cantina.xyz/cantina-docs/cantina-competitions/judging-process/finding-severity-criteria) in terms of likelihood and impact.

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

### 2.1 Rv32LoadStoreChip could break address space invariants

**Severity:** Informational/Impact: Low/Likelihood: Low

**Context:** [link](https://cantina.xyz/code/c486d600-bed0-4fc6-aed1-de759fd29fa2/openvm/extensions/rv32im/circuit/src/adapters/loadstore.rs#L183)

**Description:** `Rv32LoadStoreChip` chip doesn't assert supported address spaces. An valid instruction could read from an address space which requires elements larger than 1 byte or write into an address space which requires elements smaller than 1 byte. This could break address space invariants which other chips take.

This could only happen when users intend to add some invalid instruction.

**Proof of concept:** N/A

**Recommendation:** Add address space assertions in the chip implementation.

**Resolution:** https://github.com/openvm-org/openvm/commit/171ec20ffea9fed8c67292d4d91dfb9236029ef1

Added constraints for `mem_as` in `Rv32LoadStoreAdapterChip` to avoid breaking invariants of each address space.
- `mem_as` for `LOAD` instructions should be in `0/1/2`
- `mem_as` for `STORE` instructions should be in `2/3/4`

Added constraints for `is_load` and `is_valid` in `LoadSignExtendCoreChip` and `LoadStoreCoreChip`.

### 2.2 Hosts allow access out-of-bound memory
**Severity:** Informational

**Context:** [link1](https://github.com/openvm-org/openvm/blob/c9339e6ee8c52ee047eab2fefc94fea0926f04b8/crates/vm/src/system/memory/controller/mod.rs#L385) [link2](https://github.com/openvm-org/openvm/blob/c9339e6ee8c52ee047eab2fefc94fea0926f04b8/crates/vm/src/system/memory/controller/mod.rs#L427)

**Description:** The current implementation only asserts the start pointer is less than `2^pointer_max_bits`. When `N > 1`, it doesn't check the entire range is in bounds. 

However it is not possible to access an address greater than `2^pointer_max_bits` in this way. The lower level `Memory` uses `PagedVec`, which will do array indexing checks.
Array out of bounds will still panic.

**Proof of concept:** N/A

**Recommendation:** Fix assertion condition to `ptr_u32 + N < (1 << self.mem_config.pointer_max_bits)`.


## 3. Discussion

This report checks if all existing instructions follow all invariants. Basically the instructions need to consider:

- `pc` stuff. Most opcodes just move `DEFAULT_STEP_OFFSET`. For branch opcodes, only `to_pc` of `JALR_RV32` is based on memory and that instruction also constraints that there is no overflow.

- Address space specific constraints. Currently only address space 1/2 have constraints. Only opcodes in `Rv32LoadStoreChip` use address spaces from operands and the found issue has already been listed above.

- Hint stuff. The only variable hint opcode is `HINT_BUFFER_RV32`. If an exploit wants to make `HINT_BUFFER_RV32` overflow, it must write into a address > `2^max_pointer_bits`. So `HINT_BUFFER_RV32` is safe.

- Public values. There are just `PUBLISH` for non-continuation and `REVEAL` for continuation. Nothing needs special discussion. 
