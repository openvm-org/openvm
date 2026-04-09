# Proof Shape Validation

**Reference verifier correspondence:** `verify_proof_shape` (`verifier/proof_shape.rs`) —
validates that all proof arrays have lengths consistent with the VK before downstream
verification functions access them.

## Why this exists in the reference verifier

The reference verifier uses Rust arrays/vectors. If the prover supplies a proof with
wrong-length arrays, downstream code would panic on out-of-bounds access.
`verify_proof_shape` is a defensive check that rejects malformed proofs early.

## Why the circuit handles this differently

In the circuit, proof data lives in AIR traces rather than arrays. Each AIR has a fixed
number of rows (determined by VK constants and/or ProofShape-derived quantities). The data
is communicated between AIRs via bus interactions. This creates three enforcement mechanisms:

1. **VK-fixed trace structure.** Many trace dimensions are purely functions of VK constants
   (e.g., `num_whir_rounds`, `k_whir`, `max_constraint_degree`). These are baked into the
   circuit and cannot be wrong.

2. **AIR loop constraints.** AIRs use `NestedForLoopSubAir` or equivalent index constraints
   to enforce that loop indices increment correctly and reach the expected boundary values.
   A trace with the wrong number of rows would violate these boundary constraints.

3. **Bus interaction balancing.** When one AIR sends N messages on a bus, the receiver must
   consume exactly N messages. If the prover supplies traces with mismatched dimensions,
   the bus interactions won't balance and the witness is unsatisfying.

The net effect: proof shape validation is distributed across the circuit's structural
constraints rather than concentrated in a single check.

## Per-check correspondence

Below, each check in `verify_proof_shape` is listed with its circuit enforcement mechanism.

### VData checks (in `verify_proof_shape`, lines 316–367)

| Line | Reference verifier check | Circuit enforcement |
| --- | --- | --- |
| 316 | `trace_vdata.len() == num_airs` | VK-fixed: ProofShapeAir has one row per AIR (VK constant `num_airs`). |
| 321 | `public_values.len() == num_airs` | VK-fixed: PublicValuesAir iterates over all AIRs. PV count per AIR is VK-fixed. |
| 332 | Required AIRs have `is_present = 1` | ProofShapeAir constraint: `assert_one(is_present)` when `is_required` (PS1b). |
| 336 | Absent AIRs have no public values | PublicValuesAir only sends `PublicValuesBus` messages gated by presence. |
| 343 | `cached_commitments.len() == num_cached_mains()` | VK-fixed: ProofShapeAir has `max_cached` commitment columns per AIR row; the number of `CommitmentsBus` sends per AIR is VK-determined. |
| 351 | `log_height <= l_skip + n_stack` | Stacking's OpeningClaimsAir: `num_bits = l_skip + n_stack - log_lifted_height` is range-checked via `EqBitsLookupBus`. A negative value would fail the lookup. ProofShapeAir also range-checks `log_height` to `[0, 32)` via `PowerCheckerBus`. |
| 360 | `num_public_values == pvs.len()` | VK-fixed: ProofShapeAir sends `num_pvs` (VK constant) on `NumPublicValuesBus`; PublicValuesAir receives and processes exactly that many PVs. |

### GKR proof shape (in `verify_proof_shape`, lines 380–411)

| Line | Reference verifier check | Circuit enforcement |
| --- | --- | --- |
| 391 | `claims_per_layer.len() == num_gkr_rounds` | GkrLayerAir: `layer_idx` starts at 0 (`gkr/layer/air.rs:142`, in `GkrLayerAir::eval`), increments by 1 (`gkr/layer/air.rs:144`, in `GkrLayerAir::eval`). GkrInputAir computes `num_layers = n_logup + l_skip` (`gkr/input/air.rs:168`, in `GkrInputAir::eval`) and sends on `GkrLayerInputBus`; receives back on `GkrLayerOutputBus` with `layer_idx_end = num_layers - 1`. Permutation bus balancing forces exactly `num_layers` rows. |
| 396 | `sumcheck_polys.len() == num_gkr_rounds - 1` | GkrLayerSumcheckAir: processes layers 1..K-1 (layer 0 has no sumcheck). Layer count forced by `GkrSumcheckChallengeBus` chain pairing with GkrLayerAir. |
| 404 | `poly[i].len() == i + 1` (triangular) | GkrLayerSumcheckAir: `round` starts at 0 (`gkr/sumcheck/air.rs:157`, in `GkrLayerSumcheckAir::eval`), increments by 1 (`gkr/sumcheck/air.rs:159`, in `GkrLayerSumcheckAir::eval`). Boundary: `assert_eq(round, layer_idx - 1)` on the last round (`gkr/sumcheck/air.rs:165`, in `GkrLayerSumcheckAir::eval`). This forces layer j to have exactly j sub-rounds. |

### BatchConstraint proof shape (in `verify_proof_shape`, lines 413–516)

| Line | Reference verifier check | Circuit enforcement |
| --- | --- | --- |
| 419 | `numerator_term_per_air.len() == num_airs_present` | FractionsFolderAir: `air_idx` decrements by 1 each row (`batch_constraint/fractions_folder/air.rs:108`, in `FractionsFolderAir::eval`), and must reach 0 on the last row (`batch_constraint/fractions_folder/air.rs:110`, in `FractionsFolderAir::eval`). On the first row, it receives `num_present_airs = air_idx + 1` via `FractionFolderInputBus` (`batch_constraint/fractions_folder/air.rs:187–194`, in `FractionsFolderAir::eval`), matched against ProofShapeAir's send (`proof_shape/proof_shape/air.rs:1003–1010`, in `ProofShapeAir::eval`). Forces exactly `num_present_airs` rows. |
| 426 | `denominator_term_per_air.len() == num_airs_present` | Same as above (numerator and denominator are paired in FractionsFolderAir rows). |
| 433 | `univariate_round_coeffs.len() == s_0_deg + 1` | UnivariateSumcheckAir: `coeff_idx` starts at `univariate_deg` (`batch_constraint/sumcheck/univariate/air.rs:120`, in `UnivariateSumcheckAir::eval`), decrements by 1 (`batch_constraint/sumcheck/univariate/air.rs:124`, in `UnivariateSumcheckAir::eval`), and must reach 0 on the last row (`batch_constraint/sumcheck/univariate/air.rs:126`, in `UnivariateSumcheckAir::eval`). `univariate_deg = (max_constraint_degree + 1) * (2^l_skip - 1)` is VK-determined when the AIR is constructed (`batch_constraint/mod.rs:338`, in `BatchConstraintModule::airs`). Forces exactly `univariate_deg + 1` rows. |
| 440 | `sumcheck_round_polys.len() == n_max` | MultilinearSumcheckAir: per round, `eval_idx` starts at 0 (`batch_constraint/sumcheck/multilinear/air.rs:143`, in `MultilinearSumcheckAir::eval`), increments by 1 (`batch_constraint/sumcheck/multilinear/air.rs:148`, in `MultilinearSumcheckAir::eval`), and must reach `s_deg = max_constraint_degree + 1` on the last eval (`batch_constraint/sumcheck/multilinear/air.rs:153`, in `MultilinearSumcheckAir::eval`). `n_max` flows via `ExpressionClaimNMaxBus` → ExpressionClaimAir → `SumcheckClaimBus` chain: claims received at round i, sent at round i+1. ExpressionClaimAir sends final claim at round `n_max`, forcing exactly `n_max` rounds. |
| 447 | `column_openings.len() == num_airs_present` | Stacking's OpeningClaimsAir processes exactly `num_airs_present` AIRs (see stacking.md). |
| 457 | `evals[i].len() == max_constraint_degree + 1` | MultilinearSumcheckAir: `eval_idx` constrained to reach `max_constraint_degree` per round. VK-fixed. |
| 471 | `part_openings.len() == num_parts()` | VK-fixed: the number of trace partitions per AIR is a VK constant. SymbolicExpressionAir's cached trace encodes partition structure. OpeningClaimsAir iterates over VK-determined parts. |
| 479 | `part_openings[0].len() == common_main_width * openings_per_col` | VK-fixed: common main width is a VK constant per AIR. |
| 488 | `part_openings[1].len() == preprocessed_width * openings_per_col` | VK-fixed: preprocessed width is a VK constant. |
| 505 | `cached_openings[i].len() == width * openings_per_col` | VK-fixed: cached main widths are VK constants. |

### Stacking proof shape (in `verify_proof_shape`, lines 518–603)

| Line | Reference verifier check | Circuit enforcement |
| --- | --- | --- |
| 522 | `univariate_round_coeffs.len() == 2*(2^l_skip - 1) + 1` | UnivariateRoundAir: `coeff_idx` constrained from 0 to `2*(2^l_skip - 1)`. VK-fixed (`l_skip`). |
| 529 | `sumcheck_round_polys.len() == n_stack` | SumcheckRoundsAir: round index constrained from 1 to `n_stack`. VK-fixed. |
| 571 | `total_stacked_width <= w_stack` | StackingClaimsAir: `global_col_idx` constrained to reach exactly `w_stack - 1` on the last valid row. `w_stack` is VK-fixed. If the actual stacked width exceeds `w_stack`, the boundary constraint fails. |
| 580 | `stacking_openings.len() == layouts.len()` | StackingClaimsAir: iterates over commits with VK-determined commit structure. The commit index loop boundary is constrained. |
| 594 | `openings[i].len() == stacked_matrix_width` | StackingClaimsAir: per-commit column count is determined by the VK-fixed stacking layout. |

### WHIR proof shape (in `verify_proof_shape`, lines 605–786)

| Line | Reference verifier check | Circuit enforcement |
| --- | --- | --- |
| 614 | `whir_sumcheck_polys.len() == num_whir_sumcheck_rounds` | SumcheckAir: trace height = `k_whir * num_whir_rounds` per proof. Step index constrained to increment through all rounds. VK-fixed. |
| 619 | `codeword_commits.len() == num_whir_rounds - 1` | WhirRoundAir: commitment observation gated by `is_not_final_round`. Round index constrained from 0 to `num_whir_rounds - 1`. VK-fixed. |
| 624 | `ood_values.len() == num_whir_rounds - 1` | Same as above (OOD values observed in same non-final-round block). |
| 629 | `folding_pow_witnesses.len() == num_whir_sumcheck_rounds` | SumcheckAir: one PoW witness per sumcheck step. Same trace height as sumcheck polys. |
| 634 | `query_phase_pow_witnesses.len() == num_whir_rounds` | WhirRoundAir: one PoW witness per round. Trace height = `num_whir_rounds`. |
| 639 | `initial_round_opened_rows.len() == num_commits` | InitialOpenedValuesAir: 5-level `NestedForLoopSubAir` (proof → query → coset → commit → chunk). Commit count determined by VK-fixed layout. `NestedForLoopSubAir` enforces contiguous rows and proper counter transitions at boundaries. |
| 644 | `initial_round_merkle_proofs.len() == num_commits` | Same (Merkle proofs paired with opened rows in InitialOpenedValuesAir). |
| 651 | `codeword_opened_values.len() == num_whir_rounds - 1` | NonInitialOpenedValuesAir: 4-level `NestedForLoopSubAir` (proof → round → query → coset). Round counter `whir_round - 1` enforced by nested loop. |
| 656 | `codeword_merkle_proofs.len() == num_whir_rounds - 1` | Same (paired with opened values). |
| 661 | `final_poly.len() == 2^log_final_poly_len` | FinalPolyMleEvalAir: binary tree with `2^(log_final_poly_len+1) - 1` nodes. Leaf count = `2^log_final_poly_len`. VK-fixed. |
| 675 | `opened_rows[commit].len() == num_queries[0]` | InitialOpenedValuesAir: query loop processes `num_queries[0]` queries per commit. VK-fixed (`num_queries[0]`). |
| 683 | `merkle_proofs[commit].len() == num_queries[0]` | Same (paired). |
| 694 | `rows[query].len() == 2^k_whir` | InitialOpenedValuesAir: `coset_idx` must reach `2^k - 1` when transitioning between queries (`whir/initial_opened_values/air.rs:173–175`, in `InitialOpenedValuesAir::eval`). VK-fixed (`k_whir`). |
| 705 | `row[j].len() == stacked_width[commit]` | InitialOpenedValuesAir: column count per commit determined by VK-fixed stacking layout. |
| 720 | `merkle_proof.len() == merkle_depth` | MerkleVerifyAir: `total_depth` is a field in `MerkleVerifyBusMessage` (`bus.rs:414–436`, in `MerkleVerifyBusMessage`). Constraint: `height + 1 == total_depth` on the root row (`transcript/merkle_verify/air.rs:128–131`, in `MerkleVerifyAir::eval`). InitialOpenedValuesAir sends `total_depth = initial_log_domain_size + 1` (`whir/initial_opened_values/air.rs:311`, in `InitialOpenedValuesAir::eval`). Height increments by 1 per row. Forces correct depth. |
| 742 | `opened_values[round].len() == num_queries[round]` | NonInitialOpenedValuesAir: query loop per round within 4-level `NestedForLoopSubAir`. `num_queries[round]` is VK-fixed. |
| 750 | `merkle_proofs[round].len() == num_queries[round]` | Same (paired). |
| 761 | `opened_values[query].len() == 2^k_whir` | NonInitialOpenedValuesAir: `coset_idx` must reach `2^k - 1` when transitioning between queries (`whir/non_initial_opened_values/air.rs:80–82`, in `NonInitialOpenedValuesAir::eval`). VK-fixed. |
| 774 | `merkle_proof.len() == merkle_depth` | MerkleVerifyAir: same mechanism as line 720. NonInitialOpenedValuesAir sends `total_depth = initial_log_domain_size + 1 - whir_round` (`whir/non_initial_opened_values/air.rs:201–202`, in `NonInitialOpenedValuesAir::eval`). Decreasing depth per round matches reference verifier. |

## Summary

Every check in `verify_proof_shape` falls into one of three categories:

1. **VK-fixed** (majority): The dimension is a pure function of VK constants. The circuit's
   trace structure is built from these constants, so the dimension is correct by construction.

2. **ProofShape-derived**: The dimension depends on prover-supplied data (e.g., `num_present_airs`,
   `n_max`, `n_logup`). ProofShapeAir computes these and sends them on buses. Receiving AIRs
   constrain their loop boundaries to match. Bus balancing ensures consistency.

3. **Cross-module bus**: The dimension is enforced by bus interaction balancing between modules
   (e.g., column openings count between BC and Stacking, Merkle proof depth via MerkleVerifyBus).
