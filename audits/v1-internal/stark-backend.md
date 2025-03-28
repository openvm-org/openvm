# STARK Backend

Author: @zlangley

## 1. Introduction

Scope: stark-backend
Commit: 196b6f6569a869fb693479ac6112d11adf293e61

The focus of this review is all components of the STARK backend.

## 2. Findings

### 2.1 LogUp bus index usage

**Severity:** High
**Context:**
https://github.com/openvm-org/stark-backend/blob/196b6f6569a869fb693479ac6112d11adf293e61/crates/stark-backend/src/interaction/fri_log_up.rs

**Description:**
The LogUp summands are of the form `m / (X^b + h_beta(sigma))`, where `m` is a
multiplicity, `b` a bus index, `sigma` a message, and `h_beta` a polynomial in
`\beta` with coefficients given by the message `sigma`.

The goal is that summands for one bus cannot cancel algebraically with summands
from another bus. But these terms are not independent for different bus indices;
if $X^b + c$ can be factored into $P(X) \cdot Q(X)$, then we can always write
$1/(X^b + c)$ into a linear combination of $1/P(X)$ and $1/Q(X)$ via partial
fraction decomposition.

**Recommendation:**
Move the bus index into the message so that all the terms of the LogUp sum have
a linear term in the denominator.

**Resolution:** https://github.com/openvm-org/stark-backend/commit/2649571cba8b2e19dd018351f211e27a596d2e29

### 2.2 LogUp soundness may not be 100 bits

**Severity:** High
**Context:**
https://github.com/openvm-org/stark-backend/blob/196b6f6569a869fb693479ac6112d11adf293e61/crates/stark-backend/src/interaction/fri_log_up.rs

**Description:**
The LogUp sum computes $\sum_i m_i / (\alpha + h_{\beta}(\sigma_i))$, where $m_i$
is a multiplicity and $\sigma_i$ is a message. The values $m_i$ and $\sigma_i$
are derived from the trace. The prover convinces the verifier of the value of
this sum on a random $\alpha$ and $\beta$, and the verifier checks that the
value is zero.

The idea is that if the sum is not algebraically zero (treating $\alpha$ and
$\beta$ as variables), then evaluation at a random $\alpha$ and $\beta$ should
be nonzero.

The probability of this depends on the number of poles that the LogUp sum can
have (as a function of $\alpha$ and $\beta$). In general this seems like it may
be as high as $(k - 1) \cdot \ell$, where $k$ is the number of distinct
messages and $\ell$ and $\ell$ is the length of the longest message (with bus
index included, after the fix of Finding 2.1 above).

For OpenVM, $k$ may be as large as, say $2^{31}$ and $\ell$ as large as, say,
$100$. This gives about 85 bits of security.

**Recommendation:**
Add a proof-of-work phase before sampling $\alpha$ and $\beta$.

**Resolution:** https://github.com/openvm-org/stark-backend/commit/eee8f7c4692e939f81f7d727690ff4a4aa745ca8#diff-aefc00d21174ef68348e6600b989e2b96f996b4be9b70ed9826f4d386742f6d4R51

### 2.3 LogUp completeness is not one

**Severity:** Medium
**Context:**
https://github.com/openvm-org/stark-backend/blob/196b6f6569a869fb693479ac6112d11adf293e61/crates/stark-backend/src/interaction/fri_log_up.rs

**Description:**
For any choice of multiplicity and messages, there exist $\alpha$ and $\beta$
such that the LogUp sum has a pole at $(\alpha, \beta)$. The prover obtains
$\alpha$ and $\beta$ via Fiat-Shamir. If the trace data results in sampling a
pole, the prover will not be able to prove this witness, even if valid.

**Recommendation:**
The proof-of-work step above resolves the issue, as then $\alpha$ and $\beta$
are determined in part from the prover's private randomness.

**Resolution:** https://github.com/openvm-org/stark-backend/commit/eee8f7c4692e939f81f7d727690ff4a4aa745ca8#diff-aefc00d21174ef68348e6600b989e2b96f996b4be9b70ed9826f4d386742f6d4R51


### 2.4 Observe AIR IDs in Fiat-Shamir transcript

**Severity:** Low
**Context:** https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/verifier/mod.rs#L54

**Description:**
To protect against weak Fiat-Shamir, everything in the proof input should be observed in the Fiat-Shamir transcript.
The `Proof` contains the `air_ids` of the AIRs used in the proof, where the protocol will verify the proof against this subset of the AIRs specified in the verification key.
This finding is similar in nature to [Cantina #152](https://cantina.xyz/code/c486d600-bed0-4fc6-aed1-de759fd29fa2/findings/152).

We also ensure that everything in the `Proof` is observed:
- `commitments`: [preprocessed](https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/verifier/mod.rs#L88), [main](https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/verifier/mod.rs#L92), [after challenge](https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/interaction/fri_log_up.rs#L195) assuming `<=1` phase, [quotient](https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/verifier/mod.rs#L145)
  - We note that `observe_slice` does not observe the length of the slice, which is acceptable for commitments because the number of commitments is recorded in the verifying key, _assuming_ that the proof shape is validated against the verifying key (see Finding 2.5 below)
- `opening: OpeningProof`:
  - `proof: PcsProof<SC>` is of concrete type `FriProof` and is observed as part of FRI verify: [commit_phase_commits](https://github.com/Plonky3/Plonky3/blob/1ba4e5c9500fd956a7c1eb121e08653e5974728d/fri/src/verifier.rs#L40), [final_poly](https://github.com/Plonky3/Plonky3/blob/1ba4e5c9500fd956a7c1eb121e08653e5974728d/fri/src/verifier.rs#L49), [pow_witness](https://github.com/Plonky3/Plonky3/blob/1ba4e5c9500fd956a7c1eb121e08653e5974728d/fri/src/verifier.rs#L56).
    - `query_proofs` consists entirely of sibling hashes in Merkle proofs, which are not observed because they are determined from the opening value and Merkle root.
  - `opened_values` is observed as part of FRI verify [here](https://github.com/Plonky3/Plonky3/blob/1ba4e5c/fri/src/two_adic_pcs.rs#L405)

**Recommendation:**
Observe the `air_ids` and also the number of AIRs.

**Resolution:**
- https://github.com/openvm-org/stark-backend/pull/56 (https://github.com/openvm-org/stark-backend/commit/2c535dc35542bf2d9c957104a327ce99e8bc7c59)
- https://github.com/openvm-org/openvm/pull/1502 (https://github.com/openvm-org/openvm/commit/70c0d62cd0001e3defb2cf0f8e08b1c969e0a87a)

### 2.5 Proof shape validation

**Severity:** Low
**Context:** https://github.com/openvm-org/stark-backend/blob/fdb808bec40ff21dce7e462c2c18dbb997207adb/crates/stark-backend/src/verifier/mod.rs#L54

**Description:**
The proof shape (lengths of arrays) needs to be validated against the verifying key. This was done in the recursive verifier but not in the native Rust verifier.

**Recommendation:**
Validate the proof shape in the native Rust verifier.

**Resolution:** https://github.com/openvm-org/stark-backend/pull/57
https://github.com/openvm-org/stark-backend/commit/46f581fa46db1b61293f12b6fbc092542ad0aa45
