# BatchConstraint Module

**Reference verifier correspondence:** the post-GKR portion of `verify_zerocheck_and_logup`
(`verifier/batch_constraints.rs`) — beta_logup sampling, symbolic constraint evaluation,
interaction folding, univariate sumcheck, multilinear sumcheck. (Everything before the
constraint evaluation loop — logup PoW, alpha sampling, `verify_gkr`, xi extension — is
handled by GKR.)

**AIRs (13):** SymbolicExpressionAir, FractionsFolderAir, UnivariateSumcheckAir,
MultilinearSumcheckAir, EqNsAir, Eq3bAir, EqSharpUniAir, EqSharpUniReceiverAir, EqUniAir,
ExpressionClaimAir, InteractionsFoldingAir, ConstraintsFoldingAir, EqNegAir.
**Per-AIR details:** [BatchConstraint AIRs](./airs.md).

## Interface

All quantities below are per proof (indexed by `proof_idx`). Bus names are in parentheses for
cross-reference with [bus-inventory.md](../../bus-inventory.md).

### VK constants (baked into circuit)

- `l_skip` — log-blowup / skip factor
- `max_constraint_degree`
- Per-AIR symbolic constraint structure: number of constraints, constraint expression trees,
  interaction structure (number of interactions, message widths), trace partition widths

**On the child constraint commitment.** The per-AIR symbolic constraint structure is not
hardcoded into the verifier circuit as constants. Instead, SymbolicExpressionAir stores the
entire constraint DAG (node types, operand indices, variable entries, interaction structure)
in its **cached trace**. The commitment to this cached trace is observed on the transcript
as part of ProofShape's preamble (via `CommitmentsBus`). The enclosing circuit exposes
this commitment so that a downstream verifier can check it matches the expected child VK's
constraints. This is analogous to the VK pre-hash proviso on the main claim.

### Inputs from GKR

- **GKR handoff:** `tidx_bc` and `gkr_claim = [p_claim, q_claim - alpha_logup]` in D_EF —
  the transcript position and aggregated logup claim returned by `verify_gkr`, or `[0, 0]`
  when `n_logup = 0` (`BatchConstraintModuleBus`)
- **Evaluation point:** `xi[0..n_global + l_skip]` in D_EF — the random point at which
  polynomials are opened. Assembled from the last GKR layer's challenges and any xi extension
  (`XiRandomnessBus`)
- **Transcript positions:** `tidx_cf` for lambda/main-schedule sampling
  (`ConstraintsFoldingInputBus`); `tidx_if` for beta_logup sampling
  (`InteractionsFoldingInputBus`). These two positions are non-contiguous (see Contract)

### Inputs from ProofShape

Per child-AIR per proof (indexed by sort_idx `i`):
- Which child-AIRs are present: `is_present[i]` (`AirPresenceBus`)
- Structural constants (all VK-fixed): `air_idx`, `num_interactions`, `need_rot`
  (`AirShapeBus`); `n_lift[i]` (`NLiftBus`); `(n_lift[i], n_logup, num_interactions[i])`
  (`Eq3bShapeBus`)
- Hypercube dimensions: `n_abs[i]`, `n_sign_bit[i]` (`HyperdimBus`)
- Public values: `(air_idx, pv_idx, value)` (`PublicValuesBus`)

Global (per proof):
- `n_max`, `n_logup` (`EqNsNLogupMaxBus`, `ExpressionClaimNMaxBus`)
- `num_present_airs` (`FractionFolderInputBus`)

### Derived quantities

- `n_global = max(n_logup, n_max)` — the xi vector from GKR already has this many + `l_skip`
  entries, so the reference verifier's xi extension loop is a no-op for BC

### Outputs (to Stacking)

- **Pipeline handoff:** `tidx_stack` — the transcript index after all BC activity
  (`StackingModuleBus`)
- **Sumcheck challenges:** `r_0, ..., r_{n_max}` in D_EF — sampled in schedule (5)–(6)
  (`ConstraintSumcheckRandomnessBus`)
- **Univariate selectors:** `sel_uni(n, r_0)` for each dimension `n` — eq-polynomial
  evaluations used by Stacking for lifted polynomial evaluation (`SelUniBus`)
- **Eq-negation results:** shared computation between BC and Stacking for
  negative-hypercube equality handling (`EqNegResultBus` out, `EqNegBaseRandBus` in from
  Stacking)

## Extraction

From the satisfying BC traces, read off a `BatchConstraintProof`:

| Extracted field | Source AIR | Description |
| --- | --- | --- |
| `numerator_term_per_air[i]` | FractionsFolderAir | per-AIR constraint folding claim (D_EF); computed by ConstraintsFoldingAir, observed here |
| `denominator_term_per_air[i]` | FractionsFolderAir | per-AIR interaction folding claim (D_EF); computed by InteractionsFoldingAir, observed here |
| `univariate_round_coeffs` | UnivariateSumcheckAir | univariate sumcheck polynomial coefficients |
| `sumcheck_round_polys[j]` | MultilinearSumcheckAir | multilinear round evaluations, `j = 1..n_max` |
| `column_openings[i][part]` | *(Stacking: OpeningClaimsAir)* | per-AIR per-partition column opening values |

Note: `column_openings` is part of `BatchConstraintProof` in the reference verifier, but in
the circuit the column opening observations are performed by Stacking's OpeningClaimsAir
(see [Stacking](../stacking/README.md) schedule (1) and the module-boundary note in
[README.md](../../README.md#witness-extraction)).

## Contract

Execute the non-GKR portion of `verify_zerocheck_and_logup` with the extracted data. BC's
transcript operations occupy two disjoint regions, both determined by GKR's outgoing bus
messages:

```
(1) sample beta_logup                                      (D_EF, at tidx_if — within GKR's range)

--- contiguous region starting at tidx_cf: ---

(2) sample lambda                                          (D_EF)
(3) for each present AIR i:
      observe numerator_term[i], denominator_term[i]       (2 * D_EF)
(4) sample mu                                              (D_EF)
(5) observe univariate_round_coeffs                        (univariate_degree + 1 elements)
    sample r_0                                             (D_EF)
(6) for j = 1, ..., n_max:
      observe sumcheck_round_polys[j]                      (2 * D_EF)
      sample r_j                                           (D_EF)
```

Note: (1) is interleaved within GKR's transcript range (immediately after alpha_logup in the
reference verifier schedule). (2)–(6) form a contiguous region.

Column opening observations (after the sumcheck in the reference verifier) are handled by
Stacking's OpeningClaimsAir.

In the reference verifier, alpha and beta are sampled consecutively; in the circuit, alpha is sampled by
GKR while beta is sampled by BC at `tidx_if`. The rest of the BC schedule begins at `tidx_cf`,
after all GKR activity.

**C1 (`StackingModuleBus`).** `tidx_stack` is the transcript index after (6), computed
from `tidx_cf` by summing the slot counts in (2)–(6). The exact formula is in
`BatchConstraintModule::preflight`.

**C2 (`ConstraintSumcheckRandomnessBus`).** `r_0, ..., r_{n_max}` are the sumcheck
challenges sampled in (5)–(6) — pinned by the transcript sponge at their respective `tidx`
positions.

**C3 (`SelUniBus`).** `sel_uni(n, r_0)` values for negative-`n` dimensions (where
`log_height < l_skip`) — computed by EqNegAir as `eval_eq_uni_at_one(k, r_0^{2^{l_skip-k}})`
for each dimension `k`. These do not depend on `xi`.

**C4 (`EqNegResultBus`).** Eq-negation evaluations — computed by EqNegAir from `xi`, `r`,
and the eq-neg base randomness received from Stacking (C5).

**C5 (`EqNegBaseRandBus`).** Receives eq-negation base randomness from Stacking.

## Module-level argument

Each reference verifier check is mapped to the responsible AIR(s):

**(a) Per-AIR constraint evaluation** (`VerifierConstraintEvaluator` in
`verifier/evaluator.rs`, called from `verify_zerocheck_and_logup` at
`verifier/batch_constraints.rs:314`). For each present AIR:
- Compute `is_first_row` and `is_last_row` from `rs` and `omega_skip` via
  `progression_exp_2` (geometric-sum selectors over the skip domain).
- Walk the symbolic expression DAG, resolving variables to `column_openings` (offset 0 =
  value, offset 1 = rotated value), public values (via `PublicValuesBus` from
  PublicValuesAir), and the computed selectors.
- Lambda-fold: `expr = Σ_j λ^j * nodes[constraint_idx[j]]`.
- Weight: `numerator_term[i] = eq_ns[n_lift] * expr`.

Circuit: SymbolicExpressionAir evaluates the DAG (from cached trace) at column openings
received via buses. ConstraintsFoldingAir folds with `lambda` and weights by `eq_ns`
(from EqNsAir).

**(b) Per-AIR interaction evaluation** (in `verify_zerocheck_and_logup`,
`verifier/batch_constraints.rs:332–357`). For each
interaction in the AIR:
- `num = eval(count)`, `denom = Σ_k β^k * eval(msg_k) + (bus_index+1) * β^{msg_len}`.
- Weight by `eq_3b[interaction_idx]` (the multilinear eq polynomial selecting the
  interaction's stacked index within the logup dimension).
- Push `num_total * norm_factor * eq_sharp_ns[n_lift]` and
  `denom_total * eq_sharp_ns[n_lift]`.

Circuit: SymbolicExpressionAir evaluates interaction expressions (from cached trace).
InteractionsFoldingAir folds with `beta_logup` and weights by `eq_3b` (from Eq3bAir).
ExpressionClaimAir applies `norm_factor` and `eq_sharp_ns` (from EqSharpUniAir /
EqSharpUniReceiverAir / EqUniAir via EqNsAir).

**(c) Eq polynomial precomputation** (in `verify_zerocheck_and_logup`,
`verifier/batch_constraints.rs:207–254`).
- `eq_3b_per_trace`: `eval_eq_mle(xi[l_skip+n_lift..l_skip+n_logup], b_vec)` for each
  interaction's stacked index.
- `eq_ns[i]`: product of `eval_eq_uni(l_skip, xi[0], r_0)` with multilinear eq factors
  `eval_eq_mle([xi[l_skip+j-1]], [r_j])` for `j = 1..i`, times a front-loaded batch
  correction `prod(r_j for j > i)`.
- `eq_sharp_ns[i]`: same but with `eval_eq_sharp_uni(omega_skip_pows, xi[..l_skip], r_0)`
  as the base factor.
- Lift adjustment: when `n < 0`, use `l = l_skip + n`, `r_0^{2^|n|}` instead of full
  `rs`, and `norm_factor = 1/2^|n|`.

Circuit: EqNsAir (eq_ns), EqSharpUniAir + EqSharpUniReceiverAir + EqUniAir (eq_sharp_ns),
Eq3bAir (eq_3b), EqNegAir (negative-n normalization).

**(d) Claim assembly** (in `verify_zerocheck_and_logup`,
`verifier/batch_constraints.rs:112–137`). Observe per-AIR
`(numerator_term[i], denominator_term[i])`, verify GKR claim consistency
(`p_xi_claim == 0`, `q_xi_claim == alpha_logup`), sample `mu`, compute
`sum_claim = Σ_k μ^k * x_k` where `x_k` interleaves numerator and denominator terms.

Circuit: FractionsFolderAir combines GKR claim with per-AIR claims via `mu`.

**(e) Sumcheck verification** (in `verify_zerocheck_and_logup`,
`verifier/batch_constraints.rs:139–205`).
- Univariate: check `|D| * (a_0 + a_{|D|}) == sum_claim`, evaluate `s_0(r_0)` via Horner.
- Multilinear: `n_max` rounds of Lagrange interpolation at sampled `r_j`.

Circuit: UnivariateSumcheckAir (univariate round), MultilinearSumcheckAir (multilinear
rounds).

**(f) Final reduced claim check** (in `verify_zerocheck_and_logup`,
`verifier/batch_constraints.rs:359–367`).
`evaluated_claim = Σ_k μ^k * x_k` over `interactions_evals ++ constraints_evals`; check
`cur_sum == evaluated_claim`.

Circuit: ExpressionClaimAir assembles the mu-weighted sum and sends the final claim to
the sumcheck chain, which verifies consistency.

**Lookup dependencies:** `PowerCheckerBus` (eq polynomial computations).
