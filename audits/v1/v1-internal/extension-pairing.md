# VM Extension: Pairing

Author: [Avaneesh](https://github.com/Avaneesh-axiom)

## 1. Introduction

Scope: Pairing VM extension
Commit: [latest main](https://github.com/openvm-org/openvm/commit/b92feee7496903f6de42aef66b0c0ac146ed1438) at time of writing

## 2. Findings

### 2.1 Incorrect constant

**Severity:** Low

**Context:** [guest/src/bn254/mod.rs](https://github.com/openvm-org/openvm/blob/b92feee7496903f6de42aef66b0c0ac146ed1438/extensions/pairing/guest/src/bn254/mod.rs#L143C1-L144C1)

**Description:** The `FROBENIUS_COEFF_FQ6_C1[0]` constant is incorrect.
However, the impact is low since this constant is not currently used.

**Proof of concept:** See the failing test in [this PR](https://github.com/openvm-org/openvm/pull/1471)

**Recommendation:** Update the constant

**Resolution:** [fixed in this PR](https://github.com/openvm-org/openvm/pull/1471)
https://github.com/openvm-org/openvm/commit/179294ae7249cee1a54680377e18a8da7785c6f6

### 2.2 Pairing hint is trusted

**Severity:** High

**Context:** 
[BLS12-381](https://github.com/openvm-org/openvm/blob/17626b9400222bd78ed2766be497b6db9b259254/extensions/pairing/guest/src/bls12_381/pairing.rs#L365C1-L365C13)
and
[BN254](https://github.com/openvm-org/openvm/blob/17626b9400222bd78ed2766be497b6db9b259254/extensions/pairing/guest/src/bn254/pairing.rs#L394C1-L395C1)

**Description:**
We use the fact that the pairing is equal to 1 iff there exist `c` and `u` satisfying certain conditions.
The values `c` and `u` are hinted by the prover.
However, a malicious prover can provide an invalid hint and cause the pairing check to fail.
The problem is that the guest code trusts the hint. 

**Proof of concept:** N/A

**Recommendation:** Add a fallback that uses square-and-multiply for final exponentiation when the hint is invalid.

**Resolution:** [fixed by this commit](https://github.com/openvm-org/openvm/commit/9242cd910b9bcc0af1768a80698617da5d0aa689)

Added a fallback for final exponentiation in the pairing extension for
the case that the hint fails to prove that the final exponentiation is
equal to 1.

This is a temporary fix. We will scope out a better approach after the
security reviews.

### 2.3 Malicious prover can cause pairing hint to panic

**Severity:** High

**Context:** 
[BLS12-381](https://github.com/openvm-org/openvm/blob/17626b9400222bd78ed2766be497b6db9b259254/extensions/pairing/guest/src/bls12_381/pairing.rs#L355C1-L356C1)
and
[BN254](https://github.com/openvm-org/openvm/blob/17626b9400222bd78ed2766be497b6db9b259254/extensions/pairing/guest/src/bn254/pairing.rs#L376C1-L377C1)

**Description:** The pairing check for both BLS12-381 and BN254 panics if the `c` part of the pairing check hint is 0, when it tries to invert it.
So, a malicious prover can cause a panic in the guest code by providing a hint with `c = 0`.
This means the prover can prove that the guest code panics when it shouldn't. 

**Proof of concept:** N/A

**Recommendation:** Use the fallback from Finding 2.2 when `c = 0`.

**Resolution:** [fixed by this commit](https://github.com/openvm-org/openvm/commit/76c079ad18c17362179d1e4b087eb244726695f4)

## 3. Discussion
