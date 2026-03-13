# Recursion AIR Map

## Overview

This file is a flat reference for the recursion circuit's **39 AIRs** and the buses they
touch. It complements the module docs in `modules/`:

- module `README.md` pages explain interface, extraction, contract, and module-level
  correspondence
- module `airs.md` pages explain per-AIR mechanics, trace columns, walkthroughs, and local
  bus summaries
- this page answers: which AIR exists, where its source lives, and which buses it
  sends/receives/provides/looks up

The main verifier pipeline is:

```
ProofShape -> GKR -> BatchConstraint -> Stacking -> WHIR
                                                      |
                                        Transcript (shared by all)
```

Primitive lookup providers support multiple stages of that pipeline. See
[README.md](./README.md) for the high-level overview and [bus-inventory.md](./bus-inventory.md)
for bus message formats.

---

## Complete AIR List by Module

### BatchConstraint Module (13 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 1 | `SymbolicExpressionAir` | `batch_constraint/expr_eval/symbolic_expression/air.rs` | Evaluates symbolic constraint/interaction expressions from the DAG |
| 2 | `FractionsFolderAir` | `batch_constraint/fractions_folder/air.rs` | Folds interaction fractions with mu; bridges GKR to batch constraint |
| 3 | `UnivariateSumcheckAir` | `batch_constraint/sumcheck/univariate/air.rs` | Univariate sumcheck rounds (front-loaded, handles l_skip) |
| 4 | `MultilinearSumcheckAir` | `batch_constraint/sumcheck/multilinear/air.rs` | Multilinear sumcheck rounds with Lagrange coefficients |
| 5 | `EqNsAir` | `batch_constraint/eq_airs/eq_ns/air.rs` | Multivariate eq polynomial evaluation from xi and r randomness |
| 6 | `Eq3bAir` | `batch_constraint/eq_airs/eq_3b/air.rs` | 3-bit equality polynomial evaluations for interaction indexing |
| 7 | `EqSharpUniAir` | `batch_constraint/eq_airs/eq_sharp_uni/air.rs` | Sharp univariate eq polynomial with roots of unity |
| 8 | `EqSharpUniReceiverAir` | `batch_constraint/eq_airs/eq_sharp_uni/air.rs` | Receives and accumulates EqSharpUni products |
| 9 | `EqUniAir` | `batch_constraint/eq_airs/eq_uni/air.rs` | Univariate eq polynomial evaluation |
| 10 | `ExpressionClaimAir` | `batch_constraint/expression_claim/air.rs` | Folds expression claims with mu for algebraic batching |
| 11 | `InteractionsFoldingAir` | `batch_constraint/expr_eval/interactions_folding/air.rs` | Folds interaction evaluations into expression claims |
| 12 | `ConstraintsFoldingAir` | `batch_constraint/expr_eval/constraints_folding/air.rs` | Folds constraint evaluations into expression claims |
| 13 | `EqNegAir` | `batch_constraint/eq_airs/eq_neg/air.rs` | Negative hypercube eq polynomial evaluation |


### Transcript Module (3 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 14 | `TranscriptAir` | `transcript/transcript/air.rs` | Fiat-Shamir transcript: observe/sample operations |
| 15 | `Poseidon2Air` | `transcript/poseidon2.rs` | Poseidon2 permutation and compression lookups |
| 16 | `MerkleVerifyAir` | `transcript/merkle_verify/air.rs` | Merkle tree path verification |

### ProofShape Module (3 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 17 | `ProofShapeAir` | `proof_shape/proof_shape/air.rs` | Validates proof structure; populates global data buses |
| 18 | `PublicValuesAir` | `proof_shape/pvs/air.rs` | Observes public values into transcript |
| 19 | `RangeCheckerAir` | `primitives/range/air.rs` | 8-bit range check lookup table |

### GKR Module (4 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 20 | `GkrInputAir` | `gkr/input/air.rs` | Initializes GKR sumcheck from module bus messages |
| 21 | `GkrLayerAir` | `gkr/layer/air.rs` | Layer-to-layer GKR protocol transitions |
| 22 | `GkrLayerSumcheckAir` | `gkr/sumcheck/air.rs` | Per-layer sumcheck with cubic interpolation |
| 23 | `GkrXiSamplerAir` | `gkr/xi_sampler/air.rs` | Samples xi randomness challenges for GKR |

### Stacking Module (6 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 24 | `OpeningClaimsAir` | `stacking/opening/air.rs` | Processes column opening claims from stacked traces |
| 25 | `UnivariateRoundAir` | `stacking/univariate/air.rs` | Univariate sumcheck rounds for stacking |
| 26 | `SumcheckRoundsAir` | `stacking/sumcheck/air.rs` | Multilinear sumcheck rounds for stacking |
| 27 | `StackingClaimsAir` | `stacking/claims/air.rs` | Finalizes stacking claims; bridges to WHIR module |
| 28 | `EqBaseAir` | `stacking/eq_base/air.rs` | Base eq polynomial evaluation with rotation |
| 29 | `EqBitsAir` | `stacking/eq_bits/air.rs` | Bit-decomposed eq polynomial for stacking |

### WHIR Module (8 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 30 | `WhirRoundAir` | `whir/whir_round/air.rs` | Per-round WHIR protocol control (commitments, challenges) |
| 31 | `SumcheckAir` (WHIR) | `whir/sumcheck/air.rs` | WHIR sumcheck with alpha folding |
| 32 | `WhirQueryAir` | `whir/query/air.rs` | Generates and dispatches WHIR queries |
| 33 | `InitialOpenedValuesAir` | `whir/initial_opened_values/air.rs` | First-round opened values with Poseidon2 permute + Merkle |
| 34 | `NonInitialOpenedValuesAir` | `whir/non_initial_opened_values/air.rs` | Non-initial round opened values with Poseidon2 compress + Merkle |
| 35 | `WhirFoldingAir` | `whir/folding/air.rs` | WHIR polynomial folding tree |
| 36 | `FinalPolyMleEvalAir` | `whir/final_poly_mle_eval/air.rs` | MLE evaluation of the final polynomial |
| 37 | `FinalPolyQueryEvalAir` | `whir/final_poly_query_eval/air.rs` | Query evaluation of the final polynomial |

### Shared Lookup Providers (2 AIRs)

| # | AIR | File | Role |
|---|-----|------|------|
| 38 | `PowerCheckerAir` | `primitives/pow/air.rs` | Power-of-base lookup table |
| 39 | `ExpBitsLenAir` | `primitives/exp_bits_len/air.rs` | Exponentiation with bit-length for PoW checks |

---

## Bus Connectivity per AIR

Below, each AIR lists every bus it touches. Direction: S=send, R=receive, P=provide (add_key_with_lookups), L=lookup (lookup_key).

### BatchConstraint Module

**SymbolicExpressionAir**
- `SymbolicExpressionBus` (P, L)
- `ColumnClaimsBus` (R)
- `HyperdimBus` (L)
- `AirShapeBus` (L)
- `AirPresenceBus` (L)
- `InteractionsFoldingBus` (S)
- `ConstraintsFoldingBus` (S)
- `PublicValuesBus` (R)
- `SelHypercubeBus` (L)
- `SelUniBus` (L)

**FractionsFolderAir**
- `TranscriptBus` (observe/sample)
- `FractionFolderInputBus` (R)
- `UnivariateSumcheckInputBus` (S)
- `SumcheckClaimBus` (S)
- `BatchConstraintConductorBus` (P)
- `BatchConstraintModuleBus` (R)

**UnivariateSumcheckAir**
- `UnivariateSumcheckInputBus` (R)
- `SumcheckClaimBus` (R, S)
- `StackingModuleBus` (S)
- `TranscriptBus` (observe/sample)
- `ConstraintSumcheckRandomnessBus` (S)
- `BatchConstraintConductorBus` (P)

**MultilinearSumcheckAir**
- `SumcheckClaimBus` (R, S)
- `TranscriptBus` (observe/sample)
- `ConstraintSumcheckRandomnessBus` (S)
- `BatchConstraintConductorBus` (P)
- `StackingModuleBus` (R, S)

**EqNsAir**
- `EqZeroNBus` (R)
- `XiRandomnessBus` (R)
- `BatchConstraintConductorBus` (L)
- `SelHypercubeBus` (P)
- `EqNOuterBus` (P)

**Eq3bAir**
- `Eq3bBus` (S)
- `Eq3bShapeBus` (R)
- `BatchConstraintConductorBus` (L)

**EqSharpUniAir**
- `XiRandomnessBus` (R)
- `EqSharpUniBus` (S)
- `BatchConstraintConductorBus` (L, P)

**EqSharpUniReceiverAir**
- `BatchConstraintConductorBus` (P)
- `EqSharpUniBus` (R)
- `EqZeroNBus` (R, S)

**EqUniAir**
- `BatchConstraintConductorBus` (L)
- `EqZeroNBus` (S)

**ExpressionClaimAir**
- `ExpressionClaimNMaxBus` (R)
- `ExpressionClaimBus` (R)
- `BatchConstraintConductorBus` (L)
- `SumcheckClaimBus` (R)
- `EqNOuterBus` (L)
- `PowerCheckerBus` (L)
- `HyperdimBus` (L)

**InteractionsFoldingAir**
- `InteractionsFoldingBus` (R)
- `InteractionsFoldingInputBus` (R)
- `AirShapeBus` (L)
- `TranscriptBus` (sample)
- `ExpressionClaimBus` (S)
- `Eq3bBus` (R)

**ConstraintsFoldingAir**
- `ConstraintsFoldingBus` (R)
- `ConstraintsFoldingInputBus` (R)
- `AirShapeBus` (L)
- `TranscriptBus` (sample)
- `ExpressionClaimBus` (S)
- `EqNOuterBus` (L)
- `NLiftBus` (R)

**EqNegAir**
- `EqNegResultBus` (S)
- `EqNegBaseRandBus` (R)
- `EqNegInternalBus` (S, R)
- `SelUniBus` (P)

### Transcript Module

**TranscriptAir**
- `TranscriptBus` (S)
- `Poseidon2PermuteBus` (L)
- `FinalTranscriptStateBus` (S, optional/continuations)

**Poseidon2Air**
- `Poseidon2PermuteBus` (P)
- `Poseidon2CompressBus` (P)

**MerkleVerifyAir**
- `MerkleVerifyBus` (R, S)
- `CommitmentsBus` (L)
- `Poseidon2CompressBus` (L)
- `RightShiftBus` (L)

### ProofShape Module

**ProofShapeAir**
- `ProofShapePermutationBus` (S, R)
- `RangeCheckerBus` (L)
- `TranscriptBus` (R)
- `AirShapeBus` (P)
- `AirPresenceBus` (P)
- `HyperdimBus` (P)
- `LiftedHeightsBus` (P)
- `CommitmentsBus` (P)
- `PowerCheckerBus` (L)
- `CachedCommitBus` (S)
- `NumPublicValuesBus` (S)
- `GkrModuleBus` (S)
- `ExpressionClaimNMaxBus` (S)
- `EqNsNLogupMaxBus` (S)
- `NLiftBus` (S)
- `FractionFolderInputBus` (S)
- `Eq3bShapeBus` (S)
- `PreHashBus` (S, optional/continuations)
- `StartingTidxBus` (S, R)

**PublicValuesAir**
- `NumPublicValuesBus` (R)
- `PublicValuesBus` (S)
- `TranscriptBus` (R)

**RangeCheckerAir**
- `RangeCheckerBus` (P)

### GKR Module

**GkrInputAir**
- `GkrModuleBus` (R)
- `BatchConstraintModuleBus` (S)
- `ConstraintsFoldingInputBus` (S)
- `InteractionsFoldingInputBus` (S)
- `TranscriptBus` (observe/sample)
- `ExpBitsLenBus` (L)
- `GkrLayerInputBus` (S)
- `GkrLayerOutputBus` (R)
- `GkrXiSamplerBus` (S, R)

**GkrLayerAir**
- `XiRandomnessBus` (S)
- `TranscriptBus` (observe/sample)
- `GkrLayerInputBus` (R)
- `GkrLayerOutputBus` (S)
- `GkrSumcheckInputBus` (S)
- `GkrSumcheckChallengeBus` (S)
- `GkrSumcheckOutputBus` (R)

**GkrLayerSumcheckAir**
- `TranscriptBus` (observe/sample)
- `XiRandomnessBus` (S)
- `GkrSumcheckInputBus` (R)
- `GkrSumcheckOutputBus` (S)
- `GkrSumcheckChallengeBus` (R, S)

**GkrXiSamplerAir**
- `XiRandomnessBus` (S)
- `TranscriptBus` (sample)
- `GkrXiSamplerBus` (R, S)

### Stacking Module

**OpeningClaimsAir**
- `LiftedHeightsBus` (L)
- `StackingModuleBus` (R)
- `ColumnClaimsBus` (S)
- `TranscriptBus` (R)
- `AirShapeBus` (L)
- `StackingModuleTidxBus` (S, R)
- `ClaimCoefficientsBus` (S)
- `SumcheckClaimsBus` (S)
- `EqKernelLookupBus` (L)
- `EqBitsLookupBus` (L)

**UnivariateRoundAir**
- `TranscriptBus` (R)
- `StackingModuleTidxBus` (R, S)
- `SumcheckClaimsBus` (R, S)
- `EqRandValuesLookupBus` (P)
- `EqKernelLookupBus` (L)

**SumcheckRoundsAir**
- `ConstraintSumcheckRandomnessBus` (R)
- `WhirOpeningPointBus` (S)
- `TranscriptBus` (R)
- `StackingModuleTidxBus` (R, S)
- `SumcheckClaimsBus` (R, S)
- `EqBaseBus` (R)
- `EqRandValuesLookupBus` (P)
- `EqKernelLookupBus` (P)

**StackingClaimsAir**
- `StackingIndicesBus` (P)
- `WhirModuleBus` (S)
- `WhirMuBus` (S)
- `TranscriptBus` (R)
- `ExpBitsLenBus` (L)
- `StackingModuleTidxBus` (R)
- `ClaimCoefficientsBus` (R)
- `SumcheckClaimsBus` (R)

**EqBaseAir**
- `ConstraintSumcheckRandomnessBus` (R)
- `WhirOpeningPointBus` (S)
- `EqBaseBus` (S)
- `EqRandValuesLookupBus` (L)
- `EqKernelLookupBus` (P)
- `EqNegBaseRandBus` (S)
- `EqNegResultBus` (R)

**EqBitsAir**
- `EqBitsInternalBus` (R, S)
- `EqBitsLookupBus` (P)
- `EqRandValuesLookupBus` (L)

### WHIR Module

**WhirRoundAir**
- `WhirModuleBus` (R)
- `CommitmentsBus` (P)
- `TranscriptBus` (observe/sample)
- `ExpBitsLenBus` (L)
- `WhirSumcheckBus` (S)
- `VerifyQueriesBus` (S)
- `FinalPolyMleEvalBus` (S)
- `FinalPolyQueryEvalBus` (S)
- `WhirQueryBus` (S)
- `WhirGammaBus` (S)

**SumcheckAir (WHIR)**
- `WhirSumcheckBus` (R)
- `WhirOpeningPointBus` (R)
- `TranscriptBus` (observe/sample)
- `ExpBitsLenBus` (L)
- `WhirAlphaBus` (P)
- `WhirEqAlphaUBus` (S)

**WhirQueryAir**
- `TranscriptBus` (sample)
- `ExpBitsLenBus` (L)
- `WhirQueryBus` (S)
- `VerifyQueriesBus` (R)
- `VerifyQueryBus` (S)

**InitialOpenedValuesAir**
- `StackingIndicesBus` (L)
- `WhirMuBus` (R)
- `VerifyQueryBus` (R)
- `WhirFoldingBus` (S)
- `Poseidon2PermuteBus` (L)
- `MerkleVerifyBus` (S)

**NonInitialOpenedValuesAir**
- `VerifyQueryBus` (R)
- `WhirFoldingBus` (S)
- `Poseidon2CompressBus` (L)
- `MerkleVerifyBus` (S)

**WhirFoldingAir**
- `WhirAlphaBus` (L)
- `WhirFoldingBus` (R, S)

**FinalPolyMleEvalAir**
- `WhirOpeningPointBus` (R)
- `WhirOpeningPointLookupBus` (P, L)
- `TranscriptBus` (observe)
- `FinalPolyMleEvalBus` (R)
- `WhirEqAlphaUBus` (R)
- `WhirFinalPolyBus` (P)
- `FinalPolyFoldingBus` (R, S)

**FinalPolyQueryEvalAir**
- `WhirQueryBus` (R)
- `WhirAlphaBus` (L)
- `WhirGammaBus` (R)
- `WhirFinalPolyBus` (L)
- `FinalPolyQueryEvalBus` (R)

### Shared Lookup Providers

**PowerCheckerAir**
- `PowerCheckerBus` (P)
- `RangeCheckerBus` (P)

**ExpBitsLenAir**
- `ExpBitsLenBus` (P)
- `RightShiftBus` (P)

---

## Major Inter-AIR Bus Connections

This table highlights buses whose producer and consumer AIRs are different. It is a quick
index for tracing the major wiring paths across the recursion circuit; for full bus contracts,
see [bus-inventory.md](./bus-inventory.md).

| Bus | Producer AIR(s) | Consumer AIR(s) |
|-----|-----------------|-----------------|
| `TranscriptBus` | `TranscriptAir` | Nearly all protocol AIRs (`observe` / `sample`) |
| `Poseidon2PermuteBus` | `Poseidon2Air` | `TranscriptAir`, `InitialOpenedValuesAir` |
| `Poseidon2CompressBus` | `Poseidon2Air` | `MerkleVerifyAir`, `NonInitialOpenedValuesAir` |
| `MerkleVerifyBus` | `InitialOpenedValuesAir`, `NonInitialOpenedValuesAir`, `MerkleVerifyAir` | `MerkleVerifyAir` |
| `CommitmentsBus` | `ProofShapeAir`, `WhirRoundAir` | `MerkleVerifyAir` |
| `AirShapeBus` | `ProofShapeAir` | `SymbolicExpressionAir`, `InteractionsFoldingAir`, `ConstraintsFoldingAir`, `OpeningClaimsAir` |
| `AirPresenceBus` | `ProofShapeAir` | `SymbolicExpressionAir` |
| `Eq3bShapeBus` | `ProofShapeAir` | `Eq3bAir` |
| `EqNsNLogupMaxBus` | `ProofShapeAir` | `EqNsAir` |
| `ConstraintsFoldingInputBus` | `GkrInputAir` | `ConstraintsFoldingAir` |
| `InteractionsFoldingInputBus` | `GkrInputAir` | `InteractionsFoldingAir` |
| `HyperdimBus` | `ProofShapeAir` | `SymbolicExpressionAir`, `ExpressionClaimAir` |
| `LiftedHeightsBus` | `ProofShapeAir` | `OpeningClaimsAir` |
| `RangeCheckerBus` | `RangeCheckerAir` | `ProofShapeAir`, `PowerCheckerAir` |
| `PowerCheckerBus` | `PowerCheckerAir` | `ProofShapeAir`, `ExpressionClaimAir` |
| `ExpBitsLenBus` | `ExpBitsLenAir` | `GkrInputAir`, `StackingClaimsAir`, `WhirRoundAir`, `SumcheckAir` (WHIR), `WhirQueryAir` |
| `RightShiftBus` | `ExpBitsLenAir` | `MerkleVerifyAir` |
| `GkrModuleBus` | `ProofShapeAir` | `GkrInputAir` |
| `BatchConstraintModuleBus` | `GkrInputAir` | `FractionsFolderAir` |
| `XiRandomnessBus` | `GkrLayerAir`, `GkrLayerSumcheckAir`, `GkrXiSamplerAir` | `EqNsAir`, `EqSharpUniAir` |
| `StackingModuleBus` | `UnivariateSumcheckAir`, `MultilinearSumcheckAir` | `OpeningClaimsAir`, `MultilinearSumcheckAir` |
| `ConstraintSumcheckRandomnessBus` | `UnivariateSumcheckAir`, `MultilinearSumcheckAir` | `SumcheckRoundsAir`, `EqBaseAir` |
| `ColumnClaimsBus` | `OpeningClaimsAir` | `SymbolicExpressionAir` |
| `PublicValuesBus` | `PublicValuesAir` | `SymbolicExpressionAir` |
| `StackingIndicesBus` | `StackingClaimsAir` | `InitialOpenedValuesAir` |
| `WhirModuleBus` | `StackingClaimsAir` | `WhirRoundAir` |
| `WhirMuBus` | `StackingClaimsAir` | `InitialOpenedValuesAir` |
| `WhirOpeningPointBus` | `SumcheckRoundsAir`, `EqBaseAir` | `SumcheckAir` (WHIR), `FinalPolyMleEvalAir` |
| `EqNegBaseRandBus` | `EqBaseAir` | `EqNegAir` |
| `EqNegResultBus` | `EqNegAir` | `EqBaseAir` |
| `ExpressionClaimBus` | `InteractionsFoldingAir`, `ConstraintsFoldingAir` | `ExpressionClaimAir` |
| `FractionFolderInputBus` | `ProofShapeAir` | `FractionsFolderAir` |
| `ExpressionClaimNMaxBus` | `ProofShapeAir` | `ExpressionClaimAir` |
| `NLiftBus` | `ProofShapeAir` | `ConstraintsFoldingAir` |
| `WhirAlphaBus` | `SumcheckAir` (WHIR) | `WhirFoldingAir`, `FinalPolyQueryEvalAir` |
| `WhirFoldingBus` | `InitialOpenedValuesAir`, `NonInitialOpenedValuesAir`, `WhirFoldingAir` | `WhirFoldingAir` |
| `VerifyQueryBus` | `WhirQueryAir` | `InitialOpenedValuesAir`, `NonInitialOpenedValuesAir` |

---

## Summary

| Module / Category | AIR Count | Internal Buses | Key External Interfaces |
|-------------------|-----------|---------------|------------------------|
| BatchConstraint | 13 | 11 | Receives: `BatchConstraintModuleBus`, `XiRandomnessBus`. Sends: `StackingModuleBus`, `ConstraintSumcheckRandomnessBus` |
| Transcript | 3 | 0 | Provides: `TranscriptBus`, Poseidon2 buses, `MerkleVerifyBus` |
| ProofShape | 3 | 3 | Provides: `AirShapeBus`, `HyperdimBus`, `CommitmentsBus`, etc. Sends: `GkrModuleBus` |
| GKR | 4 | 6 | Receives: `GkrModuleBus`. Sends: `BatchConstraintModuleBus`, `XiRandomnessBus` |
| Stacking | 6 | 8 | Receives: `StackingModuleBus`. Sends: `WhirModuleBus`, `WhirMuBus`, `ColumnClaimsBus` |
| WHIR | 8 | 12 | Receives: `WhirModuleBus`. Uses: `MerkleVerifyBus`, Poseidon2 buses |
| Shared Lookup Providers | 2 | 0 | Provides: `PowerCheckerBus`, `ExpBitsLenBus`, `RightShiftBus` |
| **Total** | **39** | **40** | |
