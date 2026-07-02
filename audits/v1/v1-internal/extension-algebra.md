# VM Extension: Algebra (Modular Arithmetic and Complex Extensions)

Author: [Avaneesh](https://github.com/Avaneesh-axiom)

## 1. Introduction

Scope: [Algebra VM extension](https://github.com/openvm-org/openvm/tree/b89512352757bc60d929e9607df676ea7f37246c/extensions/algebra)

Commit: [fad6c8](https://github.com/openvm-org/openvm/commit/fad6c8941b94797aadba0a04085bbebb324ae534)

## 2. Findings

### 2.1 The setup row constraints of `ModularIsEqualCoreChip` are unsound

**Severity:** High

**Context:** [`ModularIsEqualCoreChip`](https://github.com/openvm-org/openvm/blob/830053d599606fd5c7dc8f8346710f9d6854beae/extensions/algebra/circuit/src/modular_chip/is_eq.rs)

**Description:** The setup row of the `ModularIsEqualCoreChip` is unsound. It does not correctly constrain that the modulus is correct.
The problem is that the casework on `prefix_sum` that allows constraining `b[i] = N[i]` for the correct indices (when `i > b_diff_idx`) fails on the setup row.
The intended behaviour on the setup row is that `c_lt_mark = 2` and `lt_marker` has exactly one 2 (at `c_diff_idx`) and the remaining entries are 0s.
There are three ways in which the setup row is underconstrained:
1. `c_lt_mark` is not constrained to be 2 
2. Even if `c_lt_mark = 2`, `sum_i lt_marker[i]` is not constrained to be 2 
3. Even if the above two constraints hold, `b[i] = N[i]` is only constrained for `i > c_diff_idx` while it should be constrained for all `i`.

**Proof of concept:** See the tests added in [this commit](https://github.com/openvm-org/openvm/pull/1409/commits/cdbdfe8119316324638524aaf9cffbd53929cc88).
These should fail to verify, but they currently pass.

**Recommendation:** 
On setup rows, add the constraints `c_lt_mark = 2` and `sum_i lt_marker[i] = 2`.
This (together with the existing constraints) will ensure `lt_marker` has one 2 and that the remaining entries are 0s.
Then modify the `b[i] = N[i]` constraint to constrain `b[i] = N[i]` when `prefix_sum` is not 1 or `final_sum - is_setup` (currently it applies the constraint when `prefix_sum` is not 1 or `final_sum`).
On setup rows, `final_sum - is_setup = 2 - 1 = 1` and `prefix_sum` is 0 or 2 for all `i` so `b[i] = N[i]` will be constrained for all `i`.
See the [soundness proof](#31-soundness-of-is_eq-chip) below for more details.

**Resolution:** https://github.com/openvm-org/openvm/pull/1409
https://github.com/openvm-org/openvm/commit/949e5f73ce1c96e83747be84dd220a82b774c67d

## 3. Discussion

### 3.1 Soundness of `is_eq` chip

Since the `ModularIsEqualCoreAir` air's constraints are non-obvious, we will a proof of their soundness.
In particular, we will justify the constraints that ensure `b` and `c` are less than the modulus, `N`.
The constraints that ensure that `cmp_result = 1` if and only if `b == c` are enforced in `IsEqArraySubAir` which is outside the scope of this review.

Note that we assume that the bugs found above in Section 2.1 have been fixed.

Recall the idea of this chip:
- `lt_marker` is an array of length equal to the number of limbs of `b` and `c`, and it contains all zeros except exactly one 1 and optionally a 2.
- To prove `b < N`, we set `lt_marker[b_diff_idx] = 1` where `b_diff_idx` is such that `b[b_diff_idx] < N[b_diff_idx]` and `b[j] == N[j]` for all `j > b_diff_idx` (where higher indices correspond to more significant limbs).
- Similarly, to prove `c < N`, we set `lt_marker[c_diff_idx] = 2` where `c_diff_idx` is such that `c[c_diff_idx] < N[c_diff_idx]` and `c[j] == N[j]` for all `j > c_diff_idx`.
    - There is an edge case when `b_diff_idx == c_diff_idx`.
    To handle this, we actually set `lt_marker[c_diff_idx] = c_lt_mark` where `c_lt_mark` is 1 if `b_diff_idx == c_diff_idx` and 2 otherwise.

Next, we will summarize how this idea is implemented as constraints and justify that they are sound.
The following constraints are applied to all rows with `is_valid = 1` (including setup rows) unless otherwise specified.

We constrain that `lt_marker[i]` is 0, 1, or `c_lt_mark`.

When `is_setup = 0`, we constrain that:
- `c_lt_mark` is 1 or 2.
- When `c_lt_mark = 1`, `sum_i lt_marker[i] = 1`, which implies that `lt_marker` has exactly one non-zero entry and it is a 1.
- When `c_lt_mark = 2`, we constrain
    - `sum_i lt_marker[i] * (lt_marker[i] - 1) = 2`.
    Since `lt_marker[i]` is in `{0, 1, 2}`, we have that `lt_marker[i] * (lt_marker[i] - 1)` is 0 or 2 and it is 2 exactly when `lt_marker[i] = 2`.
    So this constraint ensures that one entry of `lt_marker` is 2.
    - `sum_i lt_marker[i] = 3` which, together with the previous constraint, ensures that one entry of `lt_marker` is 1.

So far, we have constrained that, on a non-setup row, `lt_marker` either has exactly one nonzero entry and it is 1, or it has two nonzero entries, a 1 and a 2.
Let `b_diff_idx` be such that `lt_marker[b_diff_idx] = 1` and let `c_diff_idx` be such that `lt_marker[c_diff_idx] = 2` if such an index exists, otherwise let `c_diff_idx = b_diff_idx`.

Next, we iterate `i` from the most significant to the least significant limb's index (`NUM_LIMBS - 1` to 0), and we maintain a prefix sum `prefix_sum[i]` of `lt_marker` (since we are iterating backwards, this is really a suffix sum).
Let's define `final_sum = sum_j lt_marker[j]` to be the sum over all the entries of `lt_marker`.
Note that `prefix_sum[i]` is either 0, 1, or `final_sum`.

We claim that `prefix_sum[i]` is in `{1, final_sum}` if and only if `b_diff_idx >= i`.
We consider the three cases.
1. If `c_lt_mark = 1` then `prefix_sum[i] = 0` when `i > b_diff_idx` and `prefix_sum = 1` when `b_diff_idx >= i`.
2. If `c_lt_mark = 2` and `b_diff_idx > c_diff_idx` then
    - `prefix_sum = 0` when `i > b_diff_idx`,
    - `prefix_sum = 1` when `b_diff_idx >= i > c_diff_idx`,
    - `prefix_sum = final_sum = 3` when `c_diff_idx >= i`.
3. If `c_lt_mark = 2` and `c_diff_idx > b_diff_idx` then
    - `prefix_sum = 0` when `i > c_diff_idx`,
    - `prefix_sum = 2` when `c_diff_idx >= i > b_diff_idx`,
    - `prefix_sum = final_sum = 3` when `b_diff_idx >= i`.

By inspection, the claim is true.

Similarly, we claim that `prefix_sum[i]` is in `{c_lt_mark, final_sum}` if and only if `c_diff_idx >= i`.
We consider the three cases.
1. If `c_lt_mark = 1` then `c_diff_idx = b_diff_idx` and so the claim follows from the previous claim.
2. If `c_lt_mark = 2` and `b_diff_idx > c_diff_idx` then see case 2 above.
3. If `c_lt_mark = 2` and `c_diff_idx > b_diff_idx` then see case 3 above.

By inspection, the claim is true.

To constrain `b < N`, we add the following constraints:
- when `prefix_sum` is not 1 or `final_sum - is_setup`, constrain `b[i] = N[i]`.
By our claim, on non-setup rows, this is equivalent to constraining `b[i] = N[i]` for `i > b_diff_idx`.
- when `lt_marker[i]` is not 0 or 2 (and hence must be 1), constrain `b_lt_diff = N[i] - b[i]`.
This index `i` is the proposed `b_diff_idx`.
- when `is_setup = 0`, range check `b_lt_diff` to be in `[1, 2^LIMB_BITS - 1)` which implies `b_lt_diff > 0`. 

Thus, we have constrained that `b[i] == N[i]` for some `i` (namely `b_diff_idx`) and `b[j] < N[j]` for all `j > i` on non-setup rows as needed.

To constrain `c < N`, we add the following constraints:
- when `prefix_sum` is not `c_lt_mark` or `final_sum`, `c[i] = N[i]`.
By our claim, this is equivalent to constraining `c[i] = N[i]` for `i > c_diff_idx`.
- when `lt_marker[i]` is not 0 or 3 (and hence must be 2), constrain `c_lt_diff = N[i] - c[i]`.
This index `i` is the proposed `c_diff_idx`.
- when `is_setup = 0`, range check `c_lt_diff` to be in `[1, 2^LIMB_BITS - 1)` which implies `c_lt_diff > 0`. 

Thus, we have constrained that `c[i] == N[i]` for some `i` (namely `c_diff_idx`) and `c[j] < N[j]` for all `j > i` on non-setup rows as needed.

### Setup row constraints

On the setup row, we constrain:
- `c_lt_mark = 2`, which implies that `lt_marker[i]` is in `{0, 1, 2}` for all `i`
- `sum_i lt_marker[i] * (lt_marker[i] - 1) = 2` which implies that `lt_marker` has exactly one 2, by a similar argument as in the non-setup case.
- `sum_i lt_marker[i] = 2` which, together with the previous constraints, implies that `lt_marker` has exactly one 2 and the remaining entries are 0s.


Recall our constraint that when `prefix_sum` is not 1 or `final_sum - is_setup`, we constrain `b[i] = N[i]`.
On setup rows, since `final_sum - is_setup = 2 - 1 = 1`, and `prefix_sum[i]` is 0 or 2 for all `i`, this constraint is applied for all `i`. 
Thus, we must have `b == N` on the setup row, as needed.

Note that `c < N` is not constrained on the setup row since we omit the range check on `c_lt_diff`.
