# Recursion Program

Author: [Xinding Wei](https://github.com/nyunyunyunyu)

## 1. Introduction

Scope: https://github.com/openvm-org/openvm

Commit: 18096f4194b76b743463c0fb39955f24d010e9bf

Review recursion DSL program in `extensions/native/recursion/src`.

## 2. Findings

### 2.1 Out of bound access of `pows_of_two` when using `air_proof.log_degree` as an index

**Severity:** Medium

**Context:** 
`check_trace_height_constraints` checks if heights of AIRs meet the requirements of VKs.

**Description:** 
In `check_trace_height_constraints`, [here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/stark/mod.rs#L812) uses `air_proof.log_degree` as an index to access `pows_of_two`, which length is `MAX_TWO_ADICITY`. But before this access,
only [here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/stark/mod.rs#L177) asserts that `air_proof.log_degree < MAX_TWO_ADICITY + 1`. So it's possible to access an out-of-bound address when `air_proof.log_degree == MAX_TWO_ADICITY`.

**Recommendation:** 
Increase the length of `pows_of_two` to `MAX_TWO_ADICITY + 1`.

**Resolution:** https://github.com/openvm-org/openvm/pull/1452
https://github.com/openvm-org/openvm/commit/7c84a653268df0fb0d9a88bd3230f5180bf47947

### 2.2 `height_maxes` could reject valid heights

**Severity:** Medium

**Context:** 
Verifier needs to check height of each AIR is less than a threshold in order to avoid overflow.

**Description:** 
[Here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/view.rs#L91) sets `height_max` to `C::F::ORDER_U32 / max_coefficient`. 
Because we check `height < height_max`, `C::F::ORDER_U32 / max_coefficient` is a valid value for `height`.
`C::F::ORDER_U32 / max_coefficient * max_coefficient < C::F::ORDER_U32` because `C::F::ORDER_U32` is a prime and
`max_coefficient > 1`.

**Recommendation:** 
Set `height_max` to `C::F::ORDER_U32 / max_coefficient + 1`. 

**Resolution:** https://github.com/openvm-org/openvm/pull/1455
https://github.com/openvm-org/openvm/commit/84f07ea57c55d99155bf8bd2d8008d6ed6f5e61e

### 2.3 Shape of `opening.values` is not fully validated

**Severity:** Medium

**Context:** 
`opening.values` is from user inputs, which could be malicious.

**Description:** 
The shape of `opening.values.preprocessed` is not fully validated. When preparing the rounds for preprocessed traces [here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/stark/mod.rs#L405) could access an out-of-bound index.

[Here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/stark/mod.rs#L413) lengths of `prep.local` and `prep.next` should equal to the width of the preprocessed trace. 

The shape of `opening.values.after_challenge` is also not fully validated. [Here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/stark/mod.rs#L554) the width of trace is not validated.

**Recommendation:** 
Add validation of the shape of `opening.values.preprocessed` and `opening.values.after_challenge`.

**Resolution:** 
https://github.com/openvm-org/openvm/pull/1464
https://github.com/openvm-org/openvm/commit/20a84369d6a986a279a6a8f95ac9ac05d20272c8

### 2.4 Shape of `proof.commit_phase_commits` is not validated

**Severity:** Medium

**Context:** 
`proof.commit_phase_commits` is from user inputs, which could be malicious.

**Description:** 
The length of `proof.commit_phase_commits` is not validated and [here](https://github.com/openvm-org/openvm/blob/18096f4194b76b743463c0fb39955f24d010e9bf/extensions/native/recursion/src/fri/two_adic_pcs.rs#L125) we directly use it as the maximum log trace height.

**Recommendation:** 
Assert `proof.commit_phase_commits` equals to the maximum log trace height.

**Resolution:** 
https://github.com/openvm-org/openvm/pull/1468
https://github.com/openvm-org/openvm/commit/9c9638bf519f909e2df723b86a64babf181414cc

### 2.5 After challenge commitment array length not checked against `num_phases`

**Severity:** Low

**Context:** https://github.com/openvm-org/openvm/blob/70c0d62cd0001e3defb2cf0f8e08b1c969e0a87a/extensions/native/recursion/src/stark/mod.rs#L350

**Description:**
In recursion program, the number of after challenge commitments isn't checked to equal the number of phases as recorded in the vkey. 
Currently `num_phases` is required to be `<=1`, so the only scenario is if `num_phases = 1` while the length of after challenge commitments is 0.
There would be an out of bounds array access. This will not lead to a soundness issue because the control flow is already determined by `num_phases` from the vkey, and even if an out of bounds memory access were made, it is equivalent to if an array of the correct length 1 were provided, but where the value was incorrect -- this would still be caught by the verification algorithm.

**Recommendation:**
Add an array length check for clarity.

**Resolution:** https://github.com/openvm-org/openvm/pull/1503
https://github.com/openvm-org/openvm/commit/485e5e524c8f0d1ec1f8725fa76a5c7f5f9c485d

## 3. Discussion

### 3.1 Out of bound access

When using `builder.get`/`builder.set` to read/write an element from an `Array`, the index has to be less than the length of the array. Otherwise, there could be potential exploits:
- The exploit could put an unexpected value into an irrelated object in the nearby memory and let it be the returned value.
- If the index could be an arbitrary value, the exploit could overwrite any address and take the control flow.

Another case is that, when using `iter_zip`, the runtime doesn't check if all arrays have the same length. 

During the review of this report, we had went through all arrays with a variable length and verified the 
related access won't be out-of-bound.

This is a dynamic-only issue because the compiler will panic at compile if there is any out-of-bound access.

### 3.2 Constraint Evaluation

AIR Constraints only depend on the VK. So constraint evaluation of a specific AIR could be represented as a long list of 
instructions without loops or branches. In the recursion verifier program, we use `RecursiveVerifierConstraintFolder` to
convert constraints into a big expression. The logic is simple and safe at DSL level.
