# Recursion Crate Internal Audit Mapping

## Overview

The recursion crate implements a verifier sub-circuit composed of **39 AIRs** organized across **6 modules** plus 2 system-level AIRs. AIRs communicate via typed buses (permutation checks or lookups).

The modules are chained in a pipeline:
```
ProofShape -> GKR -> BatchConstraint -> Stacking -> WHIR
                                                      |
                                        Transcript (shared by all)
```

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

### System-Level (2 AIRs)

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
- `ConstraintsFoldingInputBus` (S)
- `InteractionsFoldingInputBus` (S)
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
- `StackingIndicesBus` (R)
- `WhirMuBus` (R/L)
- `VerifyQueryBus` (S)
- `WhirFoldingBus` (R)
- `Poseidon2PermuteBus` (S)
- `MerkleVerifyBus` (S)

**NonInitialOpenedValuesAir**
- `VerifyQueryBus` (S)
- `WhirFoldingBus` (R)
- `Poseidon2CompressBus` (S)
- `MerkleVerifyBus` (S)

**WhirFoldingAir**
- `WhirAlphaBus` (L)
- `WhirFoldingBus` (R, S)

**FinalPolyMleEvalAir**
- `WhirOpeningPointBus` (R, S)
- `WhirOpeningPointLookupBus` (L)
- `TranscriptBus` (observe)
- `FinalPolyMleEvalBus` (R)
- `WhirEqAlphaUBus` (R)
- `WhirFinalPolyBus` (P)
- `FinalPolyFoldingBus` (R, S)

**FinalPolyQueryEvalAir**
- `WhirQueryBus` (R)
- `WhirAlphaBus` (L)
- `WhirGammaBus` (R)
- `WhirFinalPolyBus` (S)
- `FinalPolyQueryEvalBus` (R)

### System-Level

**PowerCheckerAir**
- `PowerCheckerBus` (P)
- `RangeCheckerBus` (P)

**ExpBitsLenAir**
- `ExpBitsLenBus` (P)
- `RightShiftBus` (P)

---

## Logical Groups (Partitioned by Bus Connectivity)

Groups are kept to at most 4 AIRs unless deeper coupling requires a larger group.

### Group 1: Transcript Infrastructure
> TranscriptAir, Poseidon2Air, MerkleVerifyAir

Tightly coupled through `Poseidon2PermuteBus` and `Poseidon2CompressBus`. TranscriptAir produces the Fiat-Shamir transcript (sends on `TranscriptBus`, consumed by `Poseidon2PermuteBus`). Poseidon2Air provides both permute and compress lookups. MerkleVerifyAir uses compress lookups and sends commitment verification results.

```
TranscriptAir --[Poseidon2PermuteBus]--> Poseidon2Air
MerkleVerifyAir --[Poseidon2CompressBus]--> Poseidon2Air
MerkleVerifyAir --[CommitmentsBus]--> (ProofShapeAir, WhirRoundAir)
```

### Group 2: Proof Shape & Range Check
> ProofShapeAir, PublicValuesAir, RangeCheckerAir

ProofShapeAir is the "data loader" - it populates many global data buses (`AirShapeBus`, `HyperdimBus`, `LiftedHeightsBus`, `CommitmentsBus`). PublicValuesAir handles public value observation. RangeCheckerAir provides the 8-bit range lookup table used by ProofShapeAir and PowerCheckerAir.

```
ProofShapeAir --[NumPublicValuesBus]--> PublicValuesAir
ProofShapeAir --[RangeCheckerBus]--> RangeCheckerAir
ProofShapeAir --[GkrModuleBus]--> (GkrInputAir)
ProofShapeAir --[FractionFolderInputBus]--> (FractionsFolderAir)
```

### Group 3: Primitive Lookup Tables
> PowerCheckerAir, ExpBitsLenAir

Standalone lookup providers. PowerCheckerAir also uses `RangeCheckerBus`. Both are consumed by many AIRs across modules.

```
PowerCheckerAir --[RangeCheckerBus]--> RangeCheckerAir (Group 2)
PowerCheckerAir provides: PowerCheckerBus (used by ExpressionClaimAir, ProofShapeAir)
ExpBitsLenAir provides: ExpBitsLenBus (used by GkrInputAir, WhirRoundAir, SumcheckAir, WhirQueryAir, StackingClaimsAir), RightShiftBus (used by MerkleVerifyAir)
```

### Group 4: GKR Protocol
> GkrInputAir, GkrLayerAir, GkrLayerSumcheckAir, GkrXiSamplerAir

Tightly coupled through 6 internal buses (`GkrLayerInputBus`, `GkrLayerOutputBus`, `GkrSumcheckInputBus`, `GkrSumcheckOutputBus`, `GkrSumcheckChallengeBus`, `GkrXiSamplerBus`). Externally: receives from `GkrModuleBus` (ProofShapeAir), sends to `BatchConstraintModuleBus` (FractionsFolderAir), and produces `XiRandomnessBus` challenges.

```
GkrInputAir --[GkrLayerInput/OutputBus]--> GkrLayerAir
GkrLayerAir --[GkrSumcheckInput/Output/ChallengeBus]--> GkrLayerSumcheckAir
GkrInputAir --[GkrXiSamplerBus]--> GkrXiSamplerAir
All --[XiRandomnessBus]--> (EqNsAir, EqSharpUniAir)
```

### Group 5: Batch Constraint - Sumcheck Pipeline
> FractionsFolderAir, UnivariateSumcheckAir, MultilinearSumcheckAir, ExpressionClaimAir

The core sumcheck pipeline. FractionsFolderAir receives the GKR claim and kicks off the batched constraint sumcheck. UnivariateSumcheck handles the front-loaded univariate rounds (l_skip), MultilinearSumcheck handles remaining rounds. ExpressionClaimAir folds individual expression claims with the mu batching challenge.

```
FractionsFolderAir --[UnivariateSumcheckInputBus]--> UnivariateSumcheckAir
FractionsFolderAir --[SumcheckClaimBus]--> UnivariateSumcheckAir / MultilinearSumcheckAir
UnivariateSumcheckAir --[SumcheckClaimBus]--> MultilinearSumcheckAir
ExpressionClaimAir <--[ExpressionClaimBus]-- (InteractionsFoldingAir, ConstraintsFoldingAir)
All share: BatchConstraintConductorBus, ConstraintSumcheckRandomnessBus
```

### Group 6: Batch Constraint - Expression Evaluation
> SymbolicExpressionAir, InteractionsFoldingAir, ConstraintsFoldingAir

SymbolicExpressionAir evaluates the constraint DAG and sends results to InteractionsFoldingAir and ConstraintsFoldingAir, which fold them into ExpressionClaimBus messages for ExpressionClaimAir (Group 5).

```
SymbolicExpressionAir --[InteractionsFoldingBus]--> InteractionsFoldingAir
SymbolicExpressionAir --[ConstraintsFoldingBus]--> ConstraintsFoldingAir
InteractionsFoldingAir --[ExpressionClaimBus]--> (ExpressionClaimAir)
ConstraintsFoldingAir --[ExpressionClaimBus]--> (ExpressionClaimAir)
SymbolicExpressionAir --[ColumnClaimsBus]--> (OpeningClaimsAir)
```

### Group 7: Batch Constraint - Eq Polynomials (Univariate)
> EqUniAir, EqSharpUniAir, EqSharpUniReceiverAir, Eq3bAir

Evaluate univariate equality polynomials needed for the batch constraint sumcheck. Connected through `EqZeroNBus`, `EqSharpUniBus`, and `BatchConstraintConductorBus`.

```
EqUniAir --[EqZeroNBus]--> EqSharpUniReceiverAir / EqNsAir
EqSharpUniAir --[EqSharpUniBus]--> EqSharpUniReceiverAir
Eq3bAir --[Eq3bBus]--> (InteractionsFoldingAir)
All read from: BatchConstraintConductorBus, XiRandomnessBus
```

### Group 8: Batch Constraint - Eq Polynomials (Multivariate & Negative)
> EqNsAir, EqNegAir

EqNsAir evaluates multivariate eq polynomials. EqNegAir evaluates negative-hypercube eq polynomials. Cross-module connection: EqNegAir communicates with EqBaseAir (Stacking) through `EqNegBaseRandBus` and `EqNegResultBus`.

```
EqNsAir --[EqNOuterBus]--> (ExpressionClaimAir, ConstraintsFoldingAir)
EqNsAir --[SelHypercubeBus]--> (SymbolicExpressionAir)
EqNegAir --[EqNegBaseRandBus/EqNegResultBus]--> (EqBaseAir in Stacking)
EqNegAir --[SelUniBus]--> (SymbolicExpressionAir)
```

### Group 9: Stacking - Claims & Opening
> OpeningClaimsAir, UnivariateRoundAir, SumcheckRoundsAir, StackingClaimsAir

The stacking protocol pipeline. OpeningClaimsAir receives from `StackingModuleBus` and produces column opening claims. UnivariateRoundAir and SumcheckRoundsAir perform the stacking sumcheck. StackingClaimsAir finalizes and sends to WHIR via `WhirModuleBus`.

```
OpeningClaimsAir --[ClaimCoefficientsBus]--> StackingClaimsAir
OpeningClaimsAir --[SumcheckClaimsBus]--> UnivariateRoundAir / SumcheckRoundsAir / StackingClaimsAir
UnivariateRoundAir --[SumcheckClaimsBus]--> SumcheckRoundsAir
All share: StackingModuleTidxBus (sequencing)
StackingClaimsAir --[WhirModuleBus]--> (WhirRoundAir)
StackingClaimsAir --[WhirMuBus]--> (InitialOpenedValuesAir)
```

### Group 10: Stacking - Eq Helpers
> EqBaseAir, EqBitsAir

Support AIRs for stacking sumcheck equality evaluations. EqBaseAir computes base eq evaluations with rotation; EqBitsAir handles bit-decomposed eq evaluations.

```
EqBaseAir --[EqBaseBus]--> SumcheckRoundsAir (Group 9)
EqBaseAir --[EqNegBaseRandBus/EqNegResultBus]--> EqNegAir (Group 8)
EqBitsAir --[EqBitsLookupBus]--> OpeningClaimsAir (Group 9)
Both use: EqRandValuesLookupBus, EqKernelLookupBus, ConstraintSumcheckRandomnessBus, WhirOpeningPointBus
```

### Group 11: WHIR - Protocol Control
> WhirRoundAir, SumcheckAir (WHIR)

WhirRoundAir is the top-level WHIR round controller (receives `WhirModuleBus`, handles commitments, dispatches sumcheck/query/final-poly tasks). SumcheckAir performs the WHIR sumcheck.

```
WhirRoundAir --[WhirSumcheckBus]--> SumcheckAir
SumcheckAir --[WhirAlphaBus]--> (WhirFoldingAir, FinalPolyQueryEvalAir)
SumcheckAir --[WhirEqAlphaUBus]--> (FinalPolyMleEvalAir)
WhirRoundAir --[VerifyQueriesBus]--> (WhirQueryAir)
WhirRoundAir --[WhirGammaBus]--> (FinalPolyQueryEvalAir)
```

### Group 12: WHIR - Query Verification
> WhirQueryAir, InitialOpenedValuesAir, NonInitialOpenedValuesAir

WhirQueryAir generates queries and dispatches them. InitialOpenedValuesAir handles first-round evaluations (uses Poseidon2 permute + Merkle). NonInitialOpenedValuesAir handles subsequent rounds (uses Poseidon2 compress + Merkle).

```
WhirQueryAir --[VerifyQueryBus]--> InitialOpenedValuesAir / NonInitialOpenedValuesAir
InitialOpenedValuesAir --[WhirFoldingBus]--> (WhirFoldingAir)
NonInitialOpenedValuesAir --[WhirFoldingBus]--> (WhirFoldingAir)
InitialOpenedValuesAir --[Poseidon2PermuteBus]--> (Poseidon2Air)
NonInitialOpenedValuesAir --[Poseidon2CompressBus]--> (Poseidon2Air)
Both --[MerkleVerifyBus]--> (MerkleVerifyAir)
```

### Group 13: WHIR - Polynomial Folding & Final Poly
> WhirFoldingAir, FinalPolyMleEvalAir, FinalPolyQueryEvalAir

WhirFoldingAir builds the folding tree from opened values. FinalPolyMleEvalAir evaluates the final polynomial at the MLE point. FinalPolyQueryEvalAir evaluates it at query points.

```
WhirFoldingAir --[WhirFoldingBus]--> (InitialOpenedValuesAir, NonInitialOpenedValuesAir)
WhirFoldingAir --[WhirAlphaBus]--> (SumcheckAir)
FinalPolyMleEvalAir --[WhirFinalPolyBus]--> FinalPolyQueryEvalAir
FinalPolyMleEvalAir --[FinalPolyFoldingBus]--> (self, internal tree)
FinalPolyQueryEvalAir --[WhirQueryBus]--> (WhirRoundAir)
FinalPolyQueryEvalAir --[WhirGammaBus]--> (WhirRoundAir)
```

---

## Cross-Group Bus Connections

The following buses bridge between logical groups:

| Bus | Producer (Group) | Consumer (Group) |
|-----|-------------------|-------------------|
| `TranscriptBus` | TranscriptAir (1) | Nearly all AIRs (observe/sample) |
| `Poseidon2PermuteBus` | Poseidon2Air (1) | TranscriptAir (1), InitialOpenedValuesAir (12) |
| `Poseidon2CompressBus` | Poseidon2Air (1) | MerkleVerifyAir (1), NonInitialOpenedValuesAir (12) |
| `MerkleVerifyBus` | MerkleVerifyAir (1) | InitialOpenedValuesAir (12), NonInitialOpenedValuesAir (12) |
| `CommitmentsBus` | ProofShapeAir (2) | MerkleVerifyAir (1), WhirRoundAir (11) |
| `AirShapeBus` | ProofShapeAir (2) | SymbolicExpressionAir (6), InteractionsFoldingAir (6), ConstraintsFoldingAir (6), OpeningClaimsAir (9) |
| `AirPresenceBus` | ProofShapeAir (2) | SymbolicExpressionAir (6) |
| `Eq3bShapeBus` | ProofShapeAir (2) | Eq3bAir (7) |
| `EqNsNLogupMaxBus` | ProofShapeAir (2) | EqNsAir (8) |
| `ConstraintsFoldingInputBus` | ProofShapeAir (2) | ConstraintsFoldingAir (6) |
| `InteractionsFoldingInputBus` | ProofShapeAir (2) | InteractionsFoldingAir (6) |
| `HyperdimBus` | ProofShapeAir (2) | SymbolicExpressionAir (6), ExpressionClaimAir (5) |
| `LiftedHeightsBus` | ProofShapeAir (2) | OpeningClaimsAir (9) |
| `RangeCheckerBus` | RangeCheckerAir (2) | ProofShapeAir (2), PowerCheckerAir (3) |
| `PowerCheckerBus` | PowerCheckerAir (3) | ProofShapeAir (2), ExpressionClaimAir (5) |
| `ExpBitsLenBus` | ExpBitsLenAir (3) | GkrInputAir (4), StackingClaimsAir (9), WhirRoundAir (11), SumcheckAir (11), WhirQueryAir (12) |
| `RightShiftBus` | ExpBitsLenAir (3) | MerkleVerifyAir (1) |
| `GkrModuleBus` | ProofShapeAir (2) | GkrInputAir (4) |
| `BatchConstraintModuleBus` | GkrInputAir (4) | FractionsFolderAir (5) |
| `XiRandomnessBus` | GKR AIRs (4) | EqNsAir (8), EqSharpUniAir (7) |
| `StackingModuleBus` | UnivariateSumcheck/MultilinearSumcheck (5) | OpeningClaimsAir (9) |
| `ConstraintSumcheckRandomnessBus` | Sumcheck AIRs (5) | SumcheckRoundsAir (9), EqBaseAir (10) |
| `ColumnClaimsBus` | OpeningClaimsAir (9) | SymbolicExpressionAir (6) |
| `PublicValuesBus` | PublicValuesAir (2) | SymbolicExpressionAir (6) |
| `StackingIndicesBus` | StackingClaimsAir (9) | InitialOpenedValuesAir (12) |
| `WhirModuleBus` | StackingClaimsAir (9) | WhirRoundAir (11) |
| `WhirMuBus` | StackingClaimsAir (9) | InitialOpenedValuesAir (12) |
| `WhirOpeningPointBus` | SumcheckRoundsAir/EqBaseAir (9/10) | SumcheckAir-WHIR (11), FinalPolyMleEvalAir (13) |
| `EqNegBaseRandBus` | EqBaseAir (10) | EqNegAir (8) |
| `EqNegResultBus` | EqNegAir (8) | EqBaseAir (10) |
| `ExpressionClaimBus` | InteractionsFolding/ConstraintsFolding (6) | ExpressionClaimAir (5) |
| `FractionFolderInputBus` | ProofShapeAir (2) | FractionsFolderAir (5) |
| `ExpressionClaimNMaxBus` | ProofShapeAir (2) | ExpressionClaimAir (5) |
| `NLiftBus` | ProofShapeAir (2) | ConstraintsFoldingAir (6) |
| `WhirAlphaBus` | SumcheckAir-WHIR (11) | WhirFoldingAir (13), FinalPolyQueryEvalAir (13) |
| `WhirFoldingBus` | WhirFoldingAir (13) | InitialOpenedValuesAir (12), NonInitialOpenedValuesAir (12) |
| `VerifyQueryBus` | WhirQueryAir (12) | InitialOpenedValuesAir (12), NonInitialOpenedValuesAir (12) |

---

## Summary

| Module | AIR Count | Internal Buses | Key External Interfaces |
|--------|-----------|---------------|------------------------|
| BatchConstraint | 13 | 11 | Receives: BatchConstraintModuleBus, XiRandomnessBus. Sends: StackingModuleBus, ConstraintSumcheckRandomnessBus |
| Transcript | 3 | 0 | Provides: TranscriptBus, Poseidon2 buses, MerkleVerifyBus |
| ProofShape | 3 | 3 | Provides: AirShapeBus, HyperdimBus, CommitmentsBus, etc. Sends: GkrModuleBus |
| GKR | 4 | 6 | Receives: GkrModuleBus. Sends: BatchConstraintModuleBus, XiRandomnessBus |
| Stacking | 6 | 8 | Receives: StackingModuleBus. Sends: WhirModuleBus, WhirMuBus, ColumnClaimsBus |
| WHIR | 8 | 12 | Receives: WhirModuleBus. Uses: MerkleVerifyBus, Poseidon2 buses |
| System | 2 | 0 | Provides: PowerCheckerBus, ExpBitsLenBus, RightShiftBus |
| **Total** | **39** | **40** | |

