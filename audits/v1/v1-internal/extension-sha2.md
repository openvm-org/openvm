# VM Extension: SHA2

Author: [Avaneesh](https://github.com/Avaneesh-axiom)

## 1. Introduction

Scope: Sha256 VM chip and sub air
Commit: [v1.0.0-rc.1](https://github.com/openvm-org/openvm/releases/tag/v1.0.0-rc.1)

We will justify the soundness of the Sha256 air's constraints.

## 2. Findings

### 2.1 Padded message length is underconstrained 

**Severity**: High

**Context**: [SHA-256 VM extension padding constraints](https://github.com/openvm-org/openvm/blob/30576cc6ce838f213bf05b2e4ad035d95498c8b3/extensions/sha256/circuit/src/sha256_chip/air.rs#L363C1-L367C1)

**Description**: 
As part of the padding for a SHA-256 message, a 64-bit number is appended that denotes the length of the unpadded message in bits.
We constrain that `actual_len * 8 = appended_len` where `actual_len` is the observed length of the unpadded message in bytes and `appended_len` is this appended number.
For this constraint to be sound, we must ensure that `appended_len` is a multiple of 8.
Currently, the constraint to do this has a typo and so it only constraints that `appended_len` is a multiple of 4.

**Proof of concept**: N/A

**Recommendation**: Fix the constraint to ensure that `appended_len` is a multiple of 8.

**Resolution**: [fixed by this PR](https://github.com/openvm-org/openvm/pull/1400)
https://github.com/openvm-org/openvm/commit/fad6c8941b94797aadba0a04085bbebb324ae534

### 2.2 Trace generation used by the subair tests is wrong 

**Severity:** Medium

**Context:** [Sha256-air trace generation](https://github.com/openvm-org/openvm/blob/5e5558e8c4998797eb9ec3918c662c9ea818a81e/crates/circuits/sha256-air/src/trace.rs#L464)

**Description:** The Sha256-air fails when tested on messages that span multiple blocks. The Sha256-air's `generate_trace` function takes in a vector of records where the input message is provided as an array of bytes. The function then splits the input message into chunks of 4 bytes and converts them into words (aka `u32`'s) so the rest of the trace generation can operate on words. However, the limbs converted into words in little-endian order, which is incorrect.

**Note:** The `generate_trace` function mentioned above is currently only used in testing the Sha256-air and is not used anywhere else in the system.

**Proof of concept:** Almost any multi-block message will cause the Sha256-air testing to fail. For example, can change some of the random records to `false` in [`random_records`](https://github.com/openvm-org/openvm/blob/5e5558e8c4998797eb9ec3918c662c9ea818a81e/crates/circuits/sha256-air/src/tests.rs#L93) for `is_last_block`.

**Recommendation:** Need to change the order of the limbs when converting the input message into words. Also, add tests for testing multi-block messages.

**Resolution:** This issue was fixed by [this commit](https://github.com/openvm-org/openvm/commit/4afbcba53c8c64cd60fa02421b034ff518b98548)

## 3. Discussion

We will summarize some complicated constraints here and justify that they are sound.

### 3.1 Padding constraints

First, we will consider the constraints on `padding_occurred` (see `Sha256VmAir::eval_padding_transitions`).
We constrain that `padding_occurred`:
- is boolean
- is 1 on the last round row of the last block
- is 0 on the digest row of the last block
- if it's 1 in the current row, then it's 1 in the next row, unless the next row is the digest row of the last block.
- can only be different from the previous row in the first 4 rows of a block.
This gives us that `padding_occurred` is 1 on a suffix of rows in each message, excluding the digest row of the last block.
Furthermore, the suffix starts in the first 4 rows of some block.

Next, we will consider the constraints on `pad_flags` (see `Sha256VmAir::eval_padding_transitions` and `Sha256VmAir::eval_padding_row`).
We constrain that `pad_flags` is `NotConsidered` on all rows except the first 4 rows of a block.
On the first four rows of a block, we constrain that `pad_flags`:
- is `EntirePadding` if `padding_occurred = 1` on the current and previous rows
- is `FirstPadding*` if `padding_occurred = 1` on the current row and `padding_occurred = 0` on the previous row
- is `NotPadding` if `padding_occurred = 0` on the current row
- is `*LastRow` on the fourth row of the last block

It is clear that these constraints are sound.

### 3.2 Length of the input message

On the first row with `padding_occurred = 1`, we constrain that the length of the padding is correct.
In particular, we constrain that the length of the message processed so far (in bytes) is equal to `control.len`.
We also constrain that `control.len` is constant over all rows in all blocks.
And on the last digest row, we constrain that `control.len = len_data` and `len_data = memory[rs2_ptr]`.
So, the length of the unpadded message is equal to `memory[rs2_ptr]`, and so `rs2_ptr` determines the length of the input message in bytes, as specified in the ISA spec [here](https://github.com/axiom-crypto/openvm-private/blob/main/docs/specs/ISA.md#sha2-256-extension).

### 3.3 Row index constraints

We first constrain the type of the rows.
There are three types: round rows, digest rows, and padding rows.
The first two are determined by flags, and the last is derived as `not(is_round_row) and not(is_digest_row)`.
We constrain that the first row is a round row, and that only the following transitions between adjacent row types are valid:
- `round -> round`
- `round -> digest`
- `digest -> round`
- `digest -> padding`
- `padding -> padding`

This allows us to construct an expression, `row_idx_delta`, that takes on the value:
- 1 for `round -> round`
- 1 for `round -> digest`
- -16 for `digest -> round`
- 1 for `digest -> padding`
- 0 for `padding -> padding`

Then we constrain that `row_idx = 0` on the first row and that it increases by `row_idx_delta` between rows.

We also constrain:
- `is_round_row`, `is_digest_row`, and `is_padding_row` are boolean
- `is_round_row = 1` iff `row_idx` is in `[0, 15]`
- `is_digest_row = 1` iff `row_idx` is `16`
- `is_padding_row = 1` iff `row_idx` is `17`
This, together with the `row_idx_delta` constraint, ensures that each block consists of 16 round rows followed by a digest row, and that the padding rows follow the last block.

### 3.4 Work variables constraints

On every row, we constrain the next row's `a` to the current row's 
`h + sig_1(e) + ch(e, f, g) + K + W + sig_0(a) + Maj(a, b, c)`
and the next row's `e` to the current row's 
`d + h + sig_1(e) + ch(e, f, g) + K + W`
as in the SHA-256 spec.
So, the work variables update correctly between round rows. 

On the first round row, the previous row is the previous block's digest row, but the constraints on the working variables treat it as a round row.
This is okay because where a round row stores work variables (in `work_vars`), a digest row stores the final hash state of its block (in `hash`).
So the first round row's constraints correctly pick up the hash state from the previous block.

Next, we need to ensure that every digest row's `hash` is correct.
- On non-last blocks, the digest row's `hash` is constrained as `hash = final_hash` and `final_hash` is constrained as `final_hash = prev_hash + work_vars`.
Here, `work_vars` are the working variables taken from the last round row and `prev_hash` is constrained via interactions to the previous block's `hash`.
So, `hash` is correctly constrained to be the final hash of the current block.
- On last blocks, the digest row's `hash` is constrained equal to the initial hash state, `SHA256_H`, a constant.
This is correct since the next block is the first block of a message, so it's `prev_hash` should be `SHA256_H`.

Note that since the work variables constraints are applied to every row, they will fail on each digest row unless we fill in `carry_a` and `carry_e` with dummy values.
