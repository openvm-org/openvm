# VM Extension: Native

Author: @zlangley

## 1. Introduction

Scope: openvm/extensions/native/circuit/src
Commit: 830053d599606fd5c7dc8f8346710f9d6854beae

The focus of this review is the native extension chips and transpiler.

## 2. Findings

Classify by **severity** according to [cantina criteria](https://docs.cantina.xyz/cantina-docs/cantina-competitions/judging-process/finding-severity-criteria) in terms of likelihood and impact.

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

### 2.1 `very_first_timestamp` not sufficiently constrained in `NativePoseidon2Air` for top-level blocks

**Severity:** High
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/air.rs

**Description:**
In top-level blocks, `very_first_timestamp` is only pinned to the execution bridge, but not constrained to other timestamp columns.
This means, in particular, a malicious prover can set `end_timestamp` to be arbitrarily large.

**Recommendation:**
Add the following constraints to `NativePoseidon2Air`:
```rust
    builder
        .when(local.start_top_level)
        .assert_eq(local.very_first_timestamp + AB::F::from_canonical_usize(NUM_INITIAL_READS), local.start_timestamp);

    when_top_level_not_end.assert_eq(next.very_first_timestamp, very_first_timestamp);
```

**Resolution:** https://github.com/openvm-org/openvm/pull/1435
https://github.com/openvm-org/openvm/commit/d768af4de1500044a49ab642915174e83eb86bcb

### 2.2 NativePoseidon2Air constraints allow padding rows with `is_exhausted` cells

**Severity:** Medium
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/air.rs

**Description:**
In inside-row blocks, nothing prevents a non-first row from starting with
`is_exhausted[0] = 0`. While the goal is to constrain that inside-row blocks
compute the Poseidon2 hash, this missing constraint means they may not.
However, all the attacker can do is effectively repeatedly apply the Poseidon2
permutation to the final hash, which is unlikely to be useful in an attack.

**Recommendation:**
Add a constraint that `is_exhausted[0] = 0` or simply remove the `is_exhausted[0]` variable.

**Resolution:** https://github.com/openvm-org/openvm/pull/1436
https://github.com/openvm-org/openvm/commit/2514e5d371d5706d22f573746213cd062eea4142

### 2.3 NativePoseidon2Chip uses very slow algorithm for computing inverse of `opened_element_size_inv`

**Severity:** Low
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/chip.rs#L272

**Description:**
The algorithm for computing the inverse of `opened_element_size_inv` runs in
time `O(opened_element_size_inv.inverse())`.  This is very fast when we expect
the inverse to be 1 or 4, as we generally do. But the chip does not disallow
other values, so if a program provides, say, p-1 as an input, execution would
be extremely slow.

**Recommendation:**
If `opened_element_size_inv != 1` and `opened_element_size_inv != 4`, either
error or compute inverse using `.inverse()`.

**Resolution:** None

### 2.4 NativePoseidon2Chip panics if stream is empty

**Severity:** Low
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/chip.rs#L304

**Description:**
Chip can panic if stream is empty. Run-time should fail gracefully with an
error if input stream is not provided rather than panicking.

**Recommendation:**
Return an error rather than panicking.

**Resolution:** None


### 2.5 NativePoseidon2Air does not constrain `is_compress` to be boolean when `simple`

**Severity:** High
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/air.rs

**Description:**
The (shared) flag `is_compress` is not constrained to be boolean for `simple`
rows, but is used as boolean there when we compute the opcode for execution
bridge and for interaction count.

**Recommendation:**
Add the following constraint:
```rust
builder.when(simple).assert_bool(is_compress);
```

**Resolution:** https://github.com/openvm-org/openvm/pull/1434
https://github.com/openvm-org/openvm/commit/01a6d5d3de2b2153a22cb2d2c90a80d605ea864c


### 2.6 FriReducedOpeningChip panics if stream is empty

**Severity:** Low
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/fri/mod.rs#L573

**Description:**
Both of the following two lines panic of `hint_space` or `hint_steam` [sic] are
not sufficiently long:
```rust
    let hint_steam = &mut streams.hint_space[hint_id];
    hint_steam.drain(0..length).collect()
```

**Recommendation:**
Return an error such as `Err(HintOutOfBounds)` to fail gracefully rather than
panicking.

**Resolution:** None

### 2.7 FriReducedOpeningAir allows workload -> disabled transition

**Severity:** High
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/fri/mod.rs

**Description:**
A valid block is `workload -> ... -> workload -> ins1 -> ins2`. After all
blocks are finished are the disabled rows. But the constraints allow `workload
-> disabled`, so an adversary can stop the last block short, causing reads/writes
to be consumed by a row not associated with any instruction.

**Recommendation:**

Add a constraint like:
```
builder
    .when(local.is_workload_row)
    .assert_one(next.is_workload_row + next.is_ins_row);
```

**Resolution:** https://github.com/openvm-org/openvm/pull/1433
https://github.com/openvm-org/openvm/commit/dab4162dd77e163131b5a76b1325bb2a80a2ea69

### 2.8 `sibling_is_on_right` 

**Severity:** Low
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/circuit/src/poseidon2/air.rs

**Description:**
The `sibling_is_on_right` flag is never constrained to be boolean. However, it
doesn't seem to lead to any exploit. In addition to be constrained to a memory
read, it only is involved in the following constraints:
```rust
for i in 0..CHUNK {
    builder
        .when(next.incorporate_sibling)
        .when(next_top_level_specific.sibling_is_on_right)
        .assert_eq(next_right_input[i], left_output[i]);
    builder
        .when(next.incorporate_sibling)
        .when(AB::Expr::ONE - next_top_level_specific.sibling_is_on_right)
        .assert_eq(next_left_input[i], left_output[i]);
}
```
Making `sibling_is_on_right` non-boolean only serves to impose _more_
constraints, so the prover is never incentivized to make it non-boolean.

**Recommendation:**
Can be left as is if documented, or add `assert_bool` which could increase
clarity and better align with expectations.

**Resolution:** None


### 2.9 Variable register processing

**Severity:** High
**Context:** https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/native/transpiler/src/lib.rs

**Description:**
There are two issues in the processing of `vri`-encoded operands performed by the following code block:
```rust
 if instruction_stream[j] == VARIABLE_REGISTER_INDICATOR {
     let register = (instruction_stream[j + 1] >> 7) & 0x1f;
     let offset = instruction_stream[j + 1] >> 20;
     let mut operand = (RV32_REGISTER_NUM_LIMBS as u32 * register) + offset;
     if offset >= 1 << 12 {
         operand -= 1 << 12;
     }
     operands.push(F::from_canonical_u32(operand));
     j += 2;
 }
```

1) Register is read from `rd` rather than `rs1`.
2) Sign extension logic is not correct.

**Recommendation:**

Remove if unused or apply the following patch:

```rust
 if instruction_stream[j] == VARIABLE_REGISTER_INDICATOR {
-    let register = (instruction_stream[j + 1] >> 7) & 0x1f;
+    let register = (instruction_stream[j + 1] >> 15) & 0x1f;
     let offset = instruction_stream[j + 1] >> 20;
     let mut operand = (RV32_REGISTER_NUM_LIMBS as u32 * register) + offset;
-    if offset >= 1 << 12 {
+    if offset >= 1 << 11 {
         operand -= 1 << 12;
     }
     operands.push(F::from_canonical_u32(operand));
     j += 2;
 }
```

**Resolution:** https://github.com/openvm-org/openvm/pull/1450
https://github.com/openvm-org/openvm/commit/960abb0b9edb8ef8bca3d62054a20fb3edf372fc


## 3. Discussion

This review included all chips under `native/circuit/src` as well as
`native/transpiler`.

Aside from the issues found above, one observation from reviewing the AIRs is
that the AIRs are not always consistent with whether or not they enforce if an
instruction is syntactically valid. For example, many AIRs constrain that an
unused operand is 0 (as in the spec), and `BranchNativeAdapterAir` constrains
`d` and `e` to be either 0 or 4. But other AIRs do not enforce syntactic
validity: `NativeLoadStoreAir` does not constrain that `a == 0` when `opcode ==
HINT_STOREW`, and `NativePoseidon2Air` does not constrain that `g` is either the
inverse of 1 or 4 when `opcode == VERIFY_BATCH`. This is not necessarily a
security issue, but it does mean that in general the AIRs do not enforce the
ISA and the validity of a program must be checked separately.
