# Justification of Soundness

The soundness of `Sha2BlockHasherSubAir`'s constraints is not obvious.
This document aims to make it clearer.

## Summary of Constraints

The main constraints are summarized below.

### Constraint 1: Digest Row Hash Computation

In `eval_digest_row()` on lines 148-185, we constrain:

```
next.prev_hash + local.work_vars = next.final_hash
```

when `next` is a digest row.

### Constraint 2: Dummy Row Consistency

In `eval_transitions()` on lines 293-304, we constrain:

```
local.work_vars.a == next.work_vars.a
local.work_vars.e == next.work_vars.e
```

when `next` is a dummy row.

This ensures that all dummy rows have the same values in `work_vars.a` and `work_vars.e`, and moreover that these values match the last digest row's `hash` field. (Since the `hash` field on digest rows is the same as `work_vars` on round rows.)

### Constraint 3: Block Chaining

In `eval_prev_hash()`, we constrain, via an interaction on digest rows, that:

```
curr_block.digest_row.hash == next_block.digest_row.prev_hash
```

That is, the next block's digest row's `prev_hash` field is equal to the current block's digest row's `hash` field.

On the last block, this constraint wraps around, and constrains that:

```
last_block.digest_row.hash == first_block.digest_row.prev_hash
```

### Constraint 4: Work Variables Update

In `eval_work_vars()`, we constrain:

```
constraint_word_addition(local, next)
```

on **all** rows.

We constrain this on all rows because the constraint degree is already too high to narrow down the rows on which to enforce this constraint. On round rows, this constraint ensures the work vars are updated correctly.

However, on other rows, even though the constraint doesn't constrain anything meaningful, we still need to ensure that the constraint passes. In order to do this, we fill in certain fields on certain rows with values that satisfy the constraint.

In particular:
- When `next` is a digest row, we fill in `next.work_vars.carry_a` and `next.work_vars.carry_e` with slack values.
- When `next` is a dummy row, we also fill in `next.work_vars.carry_a` and `next.work_vars.carry_e` with slack values. However, in this case, all these values will be the same on all the dummy rows (due to constraint 2), so we compute them once (on the first dummy row) and copy them into all the other dummy rows. 


## Soundness

We will show that the four constraints above imply that the hash of each block is computed correctly.

We will walk through the justification with an example trace consisting of three blocks and three dummy rows. The argument generalizes readily to traces with more blocks or dummy rows.

### Example Trace

Suppose our trace looks like this:

```
Block 1:
  Round rows (64 rounds):
    first round row:  work_vars
    ...
    last round row:   work_vars
  
  Digest row:
    hash:       [3a]
    final_hash: (computed from prev_hash + work_vars)
    prev_hash:  [3c]
    carry_a/e:  (slack values)

Block 2:
  Round rows (64 rounds):
    first round row:  work_vars
    ...
    last round row:   work_vars
  Digest row:
    hash:       [3b]
    final_hash: (computed from prev_hash + work_vars)
    prev_hash:  [3a]
    carry_a/e:  (slack values)

Block 3:
  Round rows (64 rounds):
    first round row:  work_vars
    ...
    last round row:   work_vars
  
  Digest row:
    hash:       [3c][2a]
    final_hash: (computed from prev_hash + work_vars)
    prev_hash:  [3b]
    carry_a/e:  (slack values)

Dummy rows:
  Dummy row 1:
    work_vars:  [2a][2b]
    carry_a/e:  (slack values)
  
  Dummy row 2:
    work_vars:  [2b][2c]
    carry_a/e:  (slack values)
  
  Dummy row 3:
    work_vars:  [2c]
    carry_a/e:  (slack values)
```

**Legend:**
- Square brackets `[X]` indicate fields affected by constraint X (e.g., `[3a]` = affected by constraint 3a)
- Multiple annotations on a field (e.g., `[3c][2a]`) indicate it's affected by multiple constraints

### Constraint Applications

**Constraint 1** gives:

```
block1.digest_row.prev_hash + block1.last_round_row.work_vars == block1.digest_row.final_hash
block2.digest_row.prev_hash + block2.last_round_row.work_vars == block2.digest_row.final_hash
block3.digest_row.prev_hash + block3.last_round_row.work_vars == block3.digest_row.final_hash
```

**Constraint 2** gives:

```
[2a] block3.digest_row.hash == dummy_row_1.work_vars
[2b] dummy_row_1.work_vars == dummy_row_2.work_vars
[2c] dummy_row_2.work_vars == dummy_row_3.work_vars
```

**Constraint 3** gives:

```
[3a] block1.digest_row.hash == block2.digest_row.prev_hash
[3b] block2.digest_row.hash == block3.digest_row.prev_hash
[3c] block3.digest_row.hash == block1.digest_row.prev_hash
```

### Constraining Rounds

First, we claim that all 64 rounds for each block are constrained correctly. That is, we claim that the `work_vars` in each of these rounds are updated correctly.

**Constraint 4** gives that rounds 5 to 64 inclusive of each block are constrained correctly, since these rounds occur when `local` and `next` are in the same block.

For the first four rounds of each block, we must examine the case when `next` is the first row of a block. There are two subcases: either `local` is a digest row, or a dummy row.

#### Case 1: `local` is a digest row

If `local` is a digest row, for example when `local` is the digest row of block 1 (and so `next` is the first row of block 2), then by constraint 3b, we have:

```
local.hash == block3.digest_row.prev_hash
```

So, the `constraint_word_addition(local, next)` constrains that `next.work_vars` is updated from `local.work_vars` (i.e. `local.hash`) by consuming the first 4 words of input. Since `local.work_vars` stores the `prev_hash` of block 3, this constraint ensures that the first 4 rounds are constrained correctly.

A similar argument works for when `local` is the digest row of block2.

#### Case 2: `local` is a dummy row

If `local` is a dummy row, then it is the last dummy row, and `next` is the first row of block 1. In this case, constraint 2 gives us that:

```
local.work_vars == dummy_row_3.work_vars
                == dummy_row_2.work_vars
                == dummy_row_1.work_vars
                == block3.digest_row.hash
```

Then constraint 3c gives:

```
block3.digest_row.hash == block1.digest_row.prev_hash
```

which overall gives:

```
local.work_vars == block1.digest_row.prev_hash
```

So, the first 4 rounds of block 1 are constrained correctly.

**Conclusion:** All rounds are constrained correctly.

### Final Hash

Now, we can show that each block's `final_hash` is correct.

We already argued that, on each block, `last_round_row.work_vars` is correctly computed from the `prev_hash`. Now, **constraint 1** gives that each block's `final_hash` is constructed by adding the `last_round_row.work_vars` to the `prev_state`. This is exactly correct, by the SHA-2 specification.
