# Stacking Module

**Reference verifier correspondence:** `verify_stacked_reduction`
(`verifier/stacked_reduction.rs`) plus two outer-caller operations: the column opening
observations in `verify_zerocheck_and_logup` (after the sumcheck loop) and the μ PoW / mu
sampling preamble of `verify_whir`. Reduces column opening claims via univariate round and
multilinear sumcheck, then batches stacking openings into the WHIR claim.

**AIRs (6):** OpeningClaimsAir, UnivariateRoundAir, SumcheckRoundsAir, StackingClaimsAir,
EqBaseAir, EqBitsAir.
**Per-AIR details:** [Stacking AIRs](./airs.md).

## Interface

All quantities below are per proof (indexed by `proof_idx`). Bus names are in parentheses for
cross-reference with [bus-inventory.md](../../bus-inventory.md).

### VK constants (baked into circuit)

- `l_skip` — log-blowup / skip factor
- `n_stack` — number of multilinear sumcheck rounds (stacking height)
- `w_stack` — total number of stacked columns across all commitments
- `mu_pow_bits` — proof-of-work difficulty for the mu batching challenge (0 means no PoW)
- `stacking_index_mult` — multiplier for WHIR stacking indices (derived from WHIR round-0
  query count and `k_whir`)
- Per-commit stacked layout structure: sorted column ordering, per-AIR partition widths,
  rotation flags

### Inputs

- **Pipeline handoff:** `tidx_stack` — starting transcript index for this proof's Stacking
  activity, equal to `tidx` after BC's last sumcheck sample
  (`StackingModuleBus`, received by OpeningClaimsAir)
- **Sumcheck challenges from BC:** `r_0, ..., r_{n_max}` in D_EF — the constraint sumcheck
  randomness (`ConstraintSumcheckRandomnessBus`, received by SumcheckRoundsAir and EqBaseAir)
- **Eq-negation results:** shared computation with BC — EqBaseAir receives eq-neg evaluations
  (`EqNegResultBus`, from BC's EqNegAir)

From ProofShape (per child-AIR per proof, indexed by sort_idx `i`):
- `(i, air_idx, num_interactions, need_rot)` (`AirShapeBus`, received by
  OpeningClaimsAir) — `need_rot` gates whether rotation claims are sent on
  `ColumnClaimsBus`
- `(i, part_idx, commit_idx, n[i], lifted_height[i], log_lifted_height[i])`
  (`LiftedHeightsBus`, received by OpeningClaimsAir) — one message per trace partition
- `(commit_idx, stacking_row_idx)` (`StackingIndicesBus`, received by StackingClaimsAir)
  — maps each commitment's columns to their row indices in the stacked matrix

### Derived quantities

- `|D| = 2^l_skip` — univariate domain size (omega order)

### Outputs

- **WHIR handoff:** `tidx_whir` and `whir_claim` in D_EF — the transcript position after all
  Stacking activity and the batched stacking claim. `tidx_whir` is `tidx_stack` plus the
  slot counts from the transcript schedule below (phases A + B + C). The exact computation
  is in `StackingModule::preflight` (`stacking/mod.rs`).
  (`WhirModuleBus`, sent by StackingClaimsAir)
- **Mu challenge:** `mu` in D_EF — the stacking batching challenge
  (`WhirMuBus`, sent by StackingClaimsAir)
- **Opening point `u`:** `u[0..l_skip + n_stack]` in D_EF — the evaluation point for WHIR,
  corresponding to the reference verifier's `u_cube`. Sent via `WhirOpeningPointBus` by
  two AIRs:
  - `idx = 0..l_skip-1`: `u_0^{2^i}` for `i = 0, ..., l_skip-1` — sent by EqBaseAir.
    Note: `idx = 0` carries `u_0` itself (= `u_0^{2^0}`).
  - `idx = l_skip..l_skip+n_stack-1`: `u_j` for sumcheck rounds `j = 1, ..., n_stack` —
    sent by SumcheckRoundsAir (with `idx = round + l_skip - 1`, where `round` is 1-based).
  - Total: `l_skip + n_stack` messages, matching the reference verifier's
    `u_cube = [u0^{2^0}, ..., u0^{2^{l_skip-1}}, u1, ..., u_{n_stack}]`.
- **Column opening claims:** per-AIR per-partition `(col_claim, rot_claim)` in D_EF — the
  claimed polynomial evaluations at `xi` that WHIR will verify
  (`ColumnClaimsBus`, sent by OpeningClaimsAir)
- **Eq-negation base randomness:** `(u, r^2)` — sent to BC's EqNegAir for shared eq-neg
  computation (`EqNegBaseRandBus`, sent by EqBaseAir)

## Extraction

From the satisfying Stacking traces, read off proof fields. Note: the Stacking module
handles transcript operations that span two different `Proof` sub-structs — some fields go
into `StackingProof`, others into `BatchConstraintProof` or `WhirProof` (see the
module-boundary note in [README.md](../../README.md#witness-extraction)).

| Extracted field | Source AIR | Destination in `Proof` |
| --- | --- | --- |
| `column_openings[i][part]` | OpeningClaimsAir | `batch_constraint_proof` (observed by BC in reference verifier) |
| `univariate_round_coeffs` | UnivariateRoundAir | `stacking_proof` |
| `sumcheck_round_polys[j] = [s_j(1), s_j(2)]` | SumcheckRoundsAir | `stacking_proof` |
| `stacking_openings[commit][col]` | StackingClaimsAir | `stacking_proof` |
| `mu_pow_witness` | StackingClaimsAir | `whir_proof` (handled by `verify_whir` in reference verifier) |

**Connection to outgoing buses.** OpeningClaimsAir computes `s_0 = RLC(t_claims, lambda)` from
the column openings and sends it on `SumcheckClaimsBus` (module_idx 0) to start the sumcheck
chain. It also sends each `(col_claim, rot_claim)` on `ColumnClaimsBus` to WHIR.
StackingClaimsAir receives the final sumcheck claim via `SumcheckClaimsBus` (module_idx 2),
verifies it against the stacking openings and claim coefficients, then computes
`whir_claim = RLC(stacking_openings, mu)` and sends it on `WhirModuleBus`. It also sends `mu`
on `WhirMuBus`. EqBaseAir and SumcheckRoundsAir send the opening point coordinates `u[i]` on
`WhirOpeningPointBus`.

## Contract

The Stacking module's transcript region covers three phases: column openings observation
(handled by the reference verifier's outer caller), `verify_stacked_reduction` itself, and
μ PoW / mu sampling (also outer caller). The circuit's Stacking AIRs constrain all three
phases.

Stacking's transcript operations form a single contiguous region starting at `tidx_stack`:

```
--- Phase A (outer caller in reference verifier, OpeningClaimsAir in circuit): ---

(1) observe col_claim[0..D_EF], rot_claim[0..D_EF]  (per column, in sort_idx order)
(2) sample lambda                                    (D_EF)

--- Phase B (verify_stacked_reduction): ---

(3) observe univariate_round_coeffs                  (by UnivariateRoundAir)
    sample u_0                                       (D_EF)
(4) for round j = 1, ..., n_stack:                   (by SumcheckRoundsAir)
      observe s_j(1), s_j(2)                         (2 * D_EF)
      sample u_j                                     (D_EF)
(5) observe stacking_openings[commit][col]           (by StackingClaimsAir)

--- Phase C (outer caller in reference verifier, StackingClaimsAir in circuit): ---

(6) observe mu_pow_witness, sample mu_pow_sample     (if mu_pow_bits > 0)
(7) sample mu                                        (D_EF)
```

Phases A and C are outer-caller logic in the reference verifier but are constrained by
Stacking AIRs in the circuit (see the module-boundary note in
[README.md](../../README.md#witness-extraction)).

The reference verifier steps (spanning all three phases) are:

1. **Observe column openings and sample `lambda`** (phase A). The column openings are
   observed in the sort order guaranteed by ProofShape (`sort_idx` is a valid sort
   permutation of trace heights). Column claims with `need_rot` also have a `rot_claim`;
   others have `rot_claim = 0`.

2. **Compute `s_0`.** `s_0 = sum_i (col_claim_i + rot_claim_i * lambda) * lambda^{2i}` —
   the random linear combination of all column opening claims with the sampled `lambda`.

3. **Univariate round** (phase B begins). Check that `|D| * (a_0 + a_{|D|}) == s_0` (the
   sum of the univariate polynomial over the multiplicative subgroup `D` equals `s_0`). Then
   evaluate the polynomial at `u_0` via Horner: `claim = s_0(u_0)`.

4. **Multilinear sumcheck rounds.** For `j = 1, ..., n_stack`:
   - Compute `s_j(0) = claim - s_j(1)` (from the round polynomial)
   - Interpolate quadratically at `u_j`: `claim = interpolate([s_j(0), s_j(1), s_j(2)], u_j)`

5. **Final verification.** Compute `q_coeffs[j']` — the claim coefficients — from
   `lambda`, `u`, `r`, and the stacking layout (involving `eq_mle`, `eq_prism`,
   `rot_kernel_prism`, and `in_uni` evaluations). Check that
   `claim == sum_{j'} stacking_openings[j'] * q_coeffs[j']` (phase B ends).

6. **μ PoW check** (phase C). Verify `mu_pow_witness` satisfies the PoW condition
   (if `mu_pow_bits > 0`).

7. **Compute WHIR claim.** `whir_claim = RLC(stacking_openings, mu)` — the random linear
   combination using the sampled `mu`.

**C1 (`WhirModuleBus`).** `tidx_whir` is the transcript index after (7). `whir_claim` is the
batched stacking claim computed from (5) and (7).

**C2 (`WhirMuBus`).** `mu` is the stacking batching challenge sampled in (7).

**C3 (`WhirOpeningPointBus`).** `u[0..l_skip + n_stack]` are the evaluation point
coordinates: `u_0` powers-of-2 from EqBaseAir (`idx = 0..l_skip-1`) and sumcheck
challenges `u_1, ..., u_{n_stack}` from SumcheckRoundsAir (`idx = l_skip..l_skip+n_stack-1`).

**C4 (`ColumnClaimsBus`).** Column opening claims are the values observed in (1).

**C5 (`EqNegBaseRandBus`).** Eq-negation base randomness `(u_0, r_0^2)` — `u_0` from (3),
`r_0` from BC's constraint sumcheck.

## Module-level argument

Each reference verifier check is mapped to the responsible AIR(s):

- **`s_0` computation** from column openings and `lambda` (RLC with `lambda^{2i}`
  powers): OpeningClaimsAir.
- **Univariate round** (sum-over-D check `|D| * (a_0 + a_{|D|}) == s_0`, Horner
  evaluation at `u_0`): UnivariateRoundAir.
- **Multilinear sumcheck rounds** (`n_stack` rounds of quadratic interpolation at
  each `u_j`): SumcheckRoundsAir.
- **Final verification** (`q_coeffs` from eq/rot-kernel evaluations, dot product
  `claim == sum_{j'} stacking_openings[j'] * q_coeffs[j']`): StackingClaimsAir.
  The `q_coeffs` depend on eq polynomial evaluations from EqBaseAir and EqBitsAir.
- **`u_cube` assembly**: EqBaseAir (sends `u_0^{2^i}` on `WhirOpeningPointBus`),
  SumcheckRoundsAir (sends `u_j` on `WhirOpeningPointBus`).
- **μ PoW check** and **`whir_claim` batching** with `mu`: StackingClaimsAir.

The `q_coeffs` computation depends on the column-to-stacking-index mapping being
consistent with the reference verifier's `StackedLayout`. This is guaranteed by ProofShape
(`sort_idx` is a valid sort permutation) together with `LiftedHeightsBus` (correct
lifted heights per column) and `StackingIndicesBus` (correct stacking row indices per
commitment).

**Internal buses:** `StackingModuleTidxBus` (threads `tidx` through module_idx 0→1→2),
`SumcheckClaimsBus` (threads sumcheck claim through module_idx 0→1→2). Together these
connect OpeningClaimsAir → UnivariateRoundAir → SumcheckRoundsAir → StackingClaimsAir.
Defined in `stacking/bus.rs`.

**Lookup dependencies:** `ExpBitsLenBus` (for μ PoW check).
