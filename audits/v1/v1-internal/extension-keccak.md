# VM Extension: Keccak

Author: [shuklaayush](https://github.com/shuklaayush)

## 1. Introduction

Scope: [`extensions/keccak256`](https://github.com/openvm-org/openvm/blob/main/extensions/keccak256/)

Commit: [efdcdd76320729e2b323835da5a368d5780e1e4d](https://github.com/openvm-org/openvm/commit/efdcdd76320729e2b323835da5a368d5780e1e4d)

This review examines the Keccak256 extension implementation, focusing on its AIR constraints, validating core assumptions, and analyzing security properties.

## 2. Findings

### 2.1 Redundant check for multiple padding bytes

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/efdcdd76320729e2b323835da5a368d5780e1e4d/extensions/keccak256/circuit/src/air.rs#L264-L267

**Description:** The padding validation logic contains a redundant condition. The `when(has_multiple_padding_bytes.clone())` check is unnecessary because by definition, if a padding byte appears at any index less than KECCAK_RATE_BYTES - 1, multiple padding bytes must exist. The `is_first_padding_byte` check alone is sufficient.

**Recommendation:** Remove the redundant `when(has_multiple_padding_bytes.clone())` condition from the assertion.

**Resolution:** None

### 2.2 Missing `pc` in `KeccakInstructionCols::assert_eq`

**Severity:** Medium

**Context:** https://github.com/openvm-org/openvm/blob/ecd33f43eaee7ceaeb4ef98a3f7c7bdac1cd7c01/extensions/keccak256/circuit/src/columns.rs#L132-L147

**Description:** The `assert_eq` method in `KeccakInstructionCols` doesn't constrain `pc` to remain same across keccak-f rounds.

**Recommendation:** Add an `assert_eq` for `pc` to ensure it remains constant

```
builder.assert_eq(self.pc, other.pc);
```

**Resolution:** https://github.com/openvm-org/openvm/pull/1472
https://github.com/openvm-org/openvm/commit/deaa157e78290a677e0334acbe397e5f327e6dac

### 2.3 Duplicate `remaining_len` decrement constraint

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/ecd33f43eaee7ceaeb4ef98a3f7c7bdac1cd7c01/extensions/keccak256/circuit/src/air.rs#L233-L240

**Description:** The constraint that decrements `remaining_len` by `KECCAK_RATE_BYTES` in the `constrain_padding` function is duplicated. The same constraint already exists in `constrain_block_transition`(https://github.com/openvm-org/openvm/blob/ecd33f43eaee7ceaeb4ef98a3f7c7bdac1cd7c01/extensions/keccak256/circuit/src/air.rs#L169).

**Recommendation:** Remove the redundant constraint from `constrain_padding`.

**Resolution:** None

### 2.4 Potentially unnecessary `partial_block` columns

**Severity:** Informational

**Context:** https://github.com/openvm-org/openvm/blob/ecd33f43eaee7ceaeb4ef98a3f7c7bdac1cd7c01/extensions/keccak256/circuit/src/air.rs#L560-L582

**Description:** In the `constrain_input_read` function, the implementation uses additional `partial_block` columns to handle partial word reads. Since `word` is already degree 2, a simpler approach might be directly defining `word` as `(1 - is_padding[i]) * block_bytes[i]`.

**Recommendation:** Remove the `partial_block` columns and use the suggested approach.

**Resolution:** None

## 3. Discussion

**Columns:**

- `inner`: Columns for keccak-f permutation
- `sponge`: Columns for sponge and padding
- `instruction`: Columns for instruction interface and register access
- `mem_oc`: Auxiliary columns for offline memory checking

### `eval_keccak_f`

Constraints the keccak-f permutation for a given preimage. This uses the KeccakAir implementation from plonky3 to enforce the keccak permutation constraints.

### `constrain_padding`

Constrains that the padding bytes added to the sponge construction align with the 10\*1 padding specified in the Keccak specification. This is adapted from the [`keccak_sponge_stark`](https://github.com/0xPolygonZero/zk_evm/blob/ef388619ffbd5305209519a3a5bc0396185d68ac/evm_arithmetization/src/keccak_sponge/keccak_sponge_stark.rs) implementation in plonky2 zk_evm.

### `constrain_consistency_across_rounds`

Ensures that instruction columns remain consistent across a keccak-f permutation round.

### `constrain_absorb`

Constraints the absorb step of keccak. Constraints that the input preimage to keccak-f are the xor of the input rate bytes and the old state. Also constrains that the capacity bytes remains same across permutation rounds.

### `eval_instruction`

Receives the Keccak256 instruction from the execution bus and performs memory reads to retrieve the instruction parameters (dst, src, len). It also performs range checks on their significant limbs to ensure values don't exceed the maximum allowed pointer size.

### `constrain_input_read`

Constraints that input bytes are properly read from memory into block_bytes. The function ensures reads happen in word sizes (4 bytes at a time) and handles partial word reads when input length is not a multiple of 4 bytes. It guarantees that only non-padding bytes are read from memory and employs the `partial_block` column for handling partial reads at the end of the input.

### `constrain_output_write`

Adds constraints to write the 32-byte Keccak digest to destination memory when `export` is enabled only on the final block's last round.

### `constrain_block_transition`

Constraints the transition between consecutive Keccak blocks during the absorb phase of the sponge construction. When processing a message longer than the RATE (136 bytes), this function enforces several continuity requirements between blocks: it maintains instruction state values like destination addresses and pointers, increments the source pointer by KECCAK_RATE_BYTES to point to the next chunk of input data, updates the timestamp for proper operation sequencing, and decrements the remaining length counter by KECCAK_RATE_BYTES to track how much of the input message still needs to be processed.
