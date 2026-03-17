# WHIR Module

**Reference verifier correspondence:** `verify_whir` (`verifier/whir.rs`) after the μ PoW /
μ sampling / initial claim preamble — per-round sumcheck, codeword commitment verification,
query verification, and final polynomial check. (The preamble — μ PoW, μ sampling, initial
claim computation — is handled by Stacking's StackingClaimsAir.)

**AIRs (8):** WhirRoundAir, SumcheckAir, WhirQueryAir, InitialOpenedValuesAir,
NonInitialOpenedValuesAir, WhirFoldingAir, FinalPolyMleEvalAir, FinalPolyQueryEvalAir.
**Per-AIR details:** [WHIR AIRs](./airs.md).

## Interface

All quantities below are per proof (indexed by `proof_idx`). Bus names are in parentheses for
cross-reference with [bus-inventory.md](../../bus-inventory.md).

### VK constants (baked into circuit)

- `k_whir` — binary fold factor (number of sumcheck steps per WHIR round)
- `log_blowup` — log RS code blowup factor
- `log_final_poly_len` — log of final polynomial coefficient count
- `num_whir_rounds` — number of WHIR rounds
- `folding_pow_bits` — PoW difficulty for folding challenges
- `query_phase_pow_bits` — PoW difficulty for query phases
- Per-round: `num_queries[i]`

### Inputs

- **Pipeline handoff:** `tidx_whir` and `whir_claim` in D_EF — the transcript position after
  Stacking's μ sampling and the batched stacking claim `RLC(stacking_openings, mu)`
  (`WhirModuleBus`, received by WhirRoundAir)
- **Mu challenge:** `mu` in D_EF — the stacking batching challenge, used to weight initial-round
  opened values (`WhirMuBus`, received by InitialOpenedValuesAir)
- **Opening point:** `u[0..m]` in D_EF — the evaluation point from Stacking's sumcheck
  (`WhirOpeningPointBus`, received by SumcheckAir and FinalPolyMleEvalAir)

From ProofShape (per proof):
- `(major_idx, minor_idx, commitment)` (`CommitmentsBus`, looked up by WhirRoundAir) —
  stacking commitments at `major_idx = 0` and WHIR round commitments at
  `major_idx = round + 1`, used as Merkle roots for query verification
- `(commit_idx, stacking_row_idx)` (`StackingIndicesBus`, looked up by
  InitialOpenedValuesAir) — maps stacking columns to row indices for initial-round
  query dispatch

Other bus dependencies: `MerkleVerifyBus` (perm), `Poseidon2PermuteBus` (lookup),
`Poseidon2CompressBus` (lookup), `ExpBitsLenBus` (lookup).

### Derived quantities

- `m = l_skip + n_stack` — the polynomial variable count
- `t = k_whir * num_whir_rounds` — total sumcheck steps (equals `m - log_final_poly_len`)
- `initial_log_rs_domain_size = m + log_blowup`

### Outputs

None. WHIR is the terminal module. Its sole obligation is to verify
the WHIR proof internally — there are no outgoing bus messages to other modules.

## Extraction

From the satisfying WHIR traces, read off most fields of `WhirProof` (the `mu_pow_witness`
field is extracted from Stacking's StackingClaimsAir — see [Stacking](../stacking/README.md)):

| Extracted field | Source AIR | Description |
| --- | --- | --- |
| `whir_sumcheck_polys[i*k+j] = [ev1, ev2]` | SumcheckAir | per-step sumcheck evaluations (D_EF) |
| `folding_pow_witnesses[i*k+j]` | SumcheckAir | folding PoW witnesses |
| `codeword_commits[i]` | WhirRoundAir | committed codewords (non-final rounds) |
| `ood_values[i]` | WhirRoundAir | OOD evaluation claims (non-final rounds, D_EF) |
| `query_phase_pow_witnesses[i]` | WhirRoundAir | query phase PoW witnesses |
| `initial_round_opened_rows` | InitialOpenedValuesAir | per-commit per-query opened values (round 0) |
| `initial_round_merkle_proofs` | InitialOpenedValuesAir | Merkle proofs for initial round |
| `codeword_opened_values[i]` | NonInitialOpenedValuesAir | opened values (non-initial rounds, D_EF) |
| `codeword_merkle_proofs[i]` | NonInitialOpenedValuesAir | Merkle proofs for non-initial rounds |
| `final_poly` | FinalPolyMleEvalAir | final polynomial coefficients (D_EF) |

**Connection to internal buses.** WhirRoundAir orchestrates each round: it receives the initial
claim from `WhirModuleBus`, dispatches sumcheck claims to SumcheckAir via `WhirSumcheckBus`,
dispatches query verification to WhirQueryAir via `VerifyQueriesBus`, and collects the final
polynomial evaluations from FinalPolyMleEvalAir (`FinalPolyMleEvalBus`) and
FinalPolyQueryEvalAir (`FinalPolyQueryEvalBus`). The claim is threaded through the round
via these buses and the gamma-weighted accumulation in WhirQueryAir.

WhirQueryAir dispatches per-query Merkle verification to InitialOpenedValuesAir (round 0) or
NonInitialOpenedValuesAir (non-initial rounds) via `VerifyQueryBus`. The opened values are
folded by WhirFoldingAir via `WhirFoldingBus`, using alphas from SumcheckAir via `WhirAlphaBus`.

SumcheckAir accumulates `eq_partial = prod mobius_eq_1(u[j], alpha_j)` across all sumcheck
steps and sends the result to FinalPolyMleEvalAir via `WhirEqAlphaUBus`.
FinalPolyMleEvalAir evaluates the final polynomial MLE at `u[t..]` via a binary tree
(`FinalPolyFoldingBus`), using coefficients stored in `WhirFinalPolyBus`.
FinalPolyQueryEvalAir evaluates the final polynomial at each query point via Horner's method,
using alphas from `WhirAlphaBus` and gammas from `WhirGammaBus`.

## Contract

Note: In the reference verifier, `verify_whir` begins with μ PoW verification, μ sampling,
and initial claim computation. In the circuit, these are performed by Stacking's
StackingClaimsAir (see [Stacking](../stacking/README.md) schedule (6)–(7)); WHIR receives the
initial claim via `WhirModuleBus`.

WHIR's transcript operations form a single contiguous region starting at `tidx_whir`:

```
for round i = 0, ..., num_whir_rounds - 1:
  (1) for step j = 0, ..., k_whir - 1:                  (by SumcheckAir)
        observe ev1, ev2                                 (2 * D_EF)
        folding PoW                                      (if folding_pow_bits > 0)
        sample alpha_{i*k+j}                             (D_EF)
  (2) if i < num_whir_rounds - 1:                        (by WhirRoundAir)
        observe codeword_commit[i]                       (digest)
        sample z0[i]                                     (D_EF)
        observe ood_value[i]                             (D_EF)
  (3) if i == num_whir_rounds - 1:                       (by FinalPolyMleEvalAir)
        observe final_poly coefficients                  (2^log_final_poly_len * D_EF)
  (4) query phase PoW                                    (if query_phase_pow_bits > 0)
      for query q = 0, ..., num_queries[i] - 1:          (by WhirQueryAir)
        sample query_index[q]                            (bits)
  (5) sample gamma[i]                                    (D_EF, by WhirRoundAir)
```

The reference verifier steps (after μ PoW, μ sampling, and initial claim computation) are:

1. **Per-round sumcheck.** For round `i`, steps `j = 0, ..., k_whir - 1`:
   - Compute `ev0 = claim - ev1`
   - Update `claim = interpolate([ev0, ev1, ev2], alpha_{i*k+j})`

2. **Commitment / final polynomial.** Non-final rounds: observe codeword commitment, sample
   OOD point `z0`, observe OOD value `y0`. Final round: observe `final_poly` coefficients.

3. **Query verification.** For each query `q` in round `i`:
   - Derive `zi_root = omega^{index}` and `zi = zi_root^{2^k}` (where `omega` is the
     two-adic generator of order `2^{log_rs_domain_size}`, `index` has
     `log_rs_domain_size - k` bits, and `log_rs_domain_size` decreases by 1 each round)
   - Round 0: for each stacking commitment, Merkle-verify the opened rows against the
     stacking commitment, then compute `codeword_vals[j] = sum_c mu^c * opened_rows[j][c]`;
     fold via `binary_k_fold(codeword_vals, alphas_round, zi_root)` → `yi`
   - Non-initial rounds: Merkle-verify opened values against `codeword_commit[i-1]`;
     fold via `binary_k_fold(opened_values, alphas_round, zi_root)` → `yi`

4. **Claim update.** After all queries in round `i`:
   - Non-final rounds: `claim += y0 * gamma`
   - All rounds: `claim += sum_q yi[q] * gamma^{q+2}`

5. **Final check.** Verify `acc == claim` where:
   - `prefix = eval_mobius_eq_mle(u[..t], alphas[..t])` — the Möbius equality polynomial
     (distinct from the standard `eval_eq_mle` used below)
   - `suffix = eval_mle_evals_at_point(final_poly, u[t..])`
   - `acc = prefix * suffix`
   - For each round `i` (with `j = (i+1)*k`): for z0 (non-final rounds) and each query zi,
     accumulate `acc += gamma_pow * eval_eq_mle(alphas[j..t], z_pow[..t-j])
     * horner_eval(final_poly, z_pow_max)` where `z_pow` is the powers-of-2 sequence of `z`
     and `eval_eq_mle` is the standard multilinear equality polynomial

## Module-level argument

Each reference verifier check is mapped to the responsible AIR(s):

- **Per-round sumcheck** (quadratic interpolation at each `alpha` to update `claim`):
  SumcheckAir.
- **Commitment verification** (Merkle proofs against stacking commitments or codeword
  commitments): WhirQueryAir dispatches to InitialOpenedValuesAir (round 0) or
  NonInitialOpenedValuesAir (non-initial rounds), which send Merkle paths via
  `MerkleVerifyBus`. MerkleVerifyAir then checks each compression step via
  `Poseidon2CompressBus` and authenticates the resulting root against `CommitmentsBus`, so
  the opened values are bound to the commitment observed for that round.
- **Binary k-fold** (`binary_k_fold` reduces `2^k` opened values to a single `yi`
  using alphas and coset structure): WhirFoldingAir (receives opened values and alphas,
  computes the fold).
- **Claim accumulation** (gamma-weighted sum of OOD values `y0` and query evaluations
  `yi`): WhirRoundAir (orchestrates per-round accumulation via `VerifyQueriesBus`
  and WhirQueryAir).
- **Final check** (`acc == prefix * suffix + gamma-weighted query/OOD contributions`):
  FinalPolyMleEvalAir computes `suffix = eval_mle(final_poly, u[t..])` via binary
  tree. SumcheckAir computes `prefix = eval_mobius_eq_mle(u[..t], alphas[..t])`.
  FinalPolyQueryEvalAir evaluates the final polynomial at each query point via
  Horner's method. WhirRoundAir collects these and checks the final claim.

**Internal buses** (defined in `whir/bus.rs`):

| Bus | Sender | Receiver | Role |
| --- | --- | --- | --- |
| `WhirSumcheckBus` | WhirRoundAir | SumcheckAir | dispatch sumcheck claims per round |
| `VerifyQueriesBus` | WhirRoundAir | WhirQueryAir | dispatch per-round query verification |
| `VerifyQueryBus` | WhirQueryAir | Initial/NonInitialOpenedValuesAir | dispatch per-query Merkle verification |
| `WhirFoldingBus` | WhirQueryAir | WhirFoldingAir | opened values for binary k-fold |
| `WhirAlphaBus` | SumcheckAir | WhirFoldingAir, FinalPolyQueryEvalAir | alpha challenges |
| `WhirGammaBus` | WhirRoundAir | FinalPolyQueryEvalAir | gamma challenges |
| `WhirEqAlphaUBus` | SumcheckAir | FinalPolyMleEvalAir | eq_partial accumulation |
| `FinalPolyFoldingBus` | FinalPolyMleEvalAir | FinalPolyMleEvalAir | binary tree MLE eval |
| `WhirFinalPolyBus` | FinalPolyMleEvalAir | FinalPolyQueryEvalAir | final poly coefficients (lookup) |
| `FinalPolyMleEvalBus` | FinalPolyMleEvalAir | WhirRoundAir | MLE eval result |
| `FinalPolyQueryEvalBus` | FinalPolyQueryEvalAir | WhirRoundAir | per-query Horner eval |

**Bus dependencies:** `MerkleVerifyBus` (permutation, for commitment openings),
`Poseidon2PermuteBus` (lookup, for initial-round row hashing),
`Poseidon2CompressBus` (lookup, for non-initial round value hashing and Merkle compression),
`CommitmentsBus` (lookup, for Merkle root authentication), `ExpBitsLenBus` (lookup, for PoW
checks and omega derivation).
