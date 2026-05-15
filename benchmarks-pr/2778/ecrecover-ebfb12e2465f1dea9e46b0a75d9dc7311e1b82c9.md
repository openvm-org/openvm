| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.24 |  1.24 |  1.24 |
| app_proof |  0.48 |  0.48 |  0.48 |
| leaf |  0.27 |  0.27 |  0.27 |
| internal_for_leaf |  0.22 |  0.22 |  0.22 |
| internal_recursive.0 |  0.15 |  0.15 |  0.15 |
| internal_recursive.1 |  0.13 |  0.13 |  0.13 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  469 |  469 |  469 |  469 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insns` |  112,210 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  18.43 | -          |  18.43 |  18.43 |
| `execute_preflight_insns` |  112,210 |  112,210 |  112,210 |  112,210 |
| `execute_preflight_time_ms` |  12 |  12 |  12 |  12 |
| `execute_preflight_insn_mi/s` |  12.75 | -          |  12.75 |  12.75 |
| `trace_gen_time_ms   ` |  148 |  148 |  148 |  148 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  173 |  173 |  173 |  173 |
| `prover.main_trace_commit_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints_time_ms` |  123 |  123 |  123 |  123 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.round0_time_ms` |  77 |  77 |  77 |  77 |
| `prover.rap_constraints.mle_rounds_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  267 |  267 |  267 |  267 |
| `execute_preflight_time_ms` |  6 |  6 |  6 |  6 |
| `trace_gen_time_ms   ` |  36 |  36 |  36 |  36 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  231 |  231 |  231 |  231 |
| `prover.main_trace_commit_time_ms` |  76 |  76 |  76 |  76 |
| `prover.rap_constraints_time_ms` |  118 |  118 |  118 |  118 |
| `prover.openings_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.logup_gkr_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.round0_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  25 |  25 |  25 |  25 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  219 |  219 |  219 |  219 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  205 |  205 |  205 |  205 |
| `prover.main_trace_commit_time_ms` |  79 |  79 |  79 |  79 |
| `prover.rap_constraints_time_ms` |  95 |  95 |  95 |  95 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.round0_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.mle_rounds_time_ms` |  45 |  45 |  45 |  45 |
| `prover.openings.stacked_reduction_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  152 |  152 |  152 |  152 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  142 |  142 |  142 |  142 |
| `prover.main_trace_commit_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints_time_ms` |  68 |  68 |  68 |  68 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  130 |  130 |  130 |  130 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  121 |  121 |  121 |  121 |
| `prover.main_trace_commit_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9/ecrecover-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 2.84 | app_proof.0 |
| set initial memory | 2.84 | app_proof.0 |
| prover.rap_constraints | 1.79 | internal_for_leaf.0.prover |
| frac_sumcheck.gkr_rounds | 1.66 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 1.66 | leaf.0.prover |
| prover.gkr_input_evals | 1.64 | leaf.0.prover |
| frac_sumcheck.segment_tree | 1.64 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.42 | internal_for_leaf.0.prover |
| prover.openings | 1.41 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 1.41 | internal_for_leaf.0.prover |
| prover.merkle_tree | 1.41 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 1.37 | internal_for_leaf.0.prover |
| prover.before_gkr_input_evals | 1.24 | internal_for_leaf.0.prover |
| prover.stacked_commit | 1.24 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 1.23 | internal_for_leaf.0.prover |
| tracegen.pow_checker | 0.54 | leaf.0 |
| tracegen.exp_bits_len | 0.54 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.54 | leaf.0 |
| tracegen.whir_folding | 0.47 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.47 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.47 | leaf.0 |
| tracegen.public_values | 0.47 | leaf.0 |
| tracegen.range_checker | 0.47 | leaf.0 |
| tracegen.proof_shape | 0.47 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 176 | 9 | 0 | 267,111 | 228,567 | 9 | 82 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 12 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 13 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 14 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 15 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 17 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| 18 | KeccakfOpAir | 110 | 27 | 2 | 
| 19 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | XorinVmAir | 357 | 92 | 3 | 
| 21 | Rv64HintStoreAir | 17 | 15 | 3 | 
| 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 4 | 3 | 
| 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 17 | 9 | 3 | 
| 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 12 | 10 | 3 | 
| 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 14 | 25 | 3 | 
| 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 11 | 11 | 3 | 
| 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 29 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 30 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| 31 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 19 | 30 | 3 | 
| 32 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| 33 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| 34 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 35 | PhantomAir | 3 |  | 1 | 
| 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 37 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 
| 6 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 50 | 118 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 9 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 50 | 118 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 72 | 9 |  |  | 0 |  | 350 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 469 |  |  | 6 | 112,210 | 18.43 | 0 | 481 |  | 
| internal_for_leaf |  |  |  | 219 |  |  |  |  |  |  |  |  |  | 219 | 
| internal_recursive.0 |  |  |  | 152 |  |  |  |  |  |  |  |  |  | 153 | 
| internal_recursive.1 |  |  |  | 131 |  |  |  |  |  |  |  |  |  | 131 | 
| leaf |  |  | 267 |  |  |  |  |  |  |  |  |  |  | 267 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 1 | 
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 30 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 14 | 
| app_proof | XorinVmAir | 0 | 0 | 

| group | air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 69 | 213 | 4 | 
| agg_keygen | 1 | VmPvsAir | 30 | 54 | 4 | 
| agg_keygen | 10 | EqSharpUniReceiverAir | 3 | 25 | 4 | 
| agg_keygen | 11 | EqUniAir | 3 | 31 | 4 | 
| agg_keygen | 12 | ExpressionClaimAir | 7 | 68 | 4 | 
| agg_keygen | 13 | InteractionsFoldingAir | 13 | 94 | 4 | 
| agg_keygen | 14 | ConstraintsFoldingAir | 10 | 42 | 4 | 
| agg_keygen | 15 | EqNegAir | 8 | 83 | 4 | 
| agg_keygen | 16 | TranscriptAir | 17 | 84 | 4 | 
| agg_keygen | 17 | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| agg_keygen | 18 | MerkleVerifyAir | 6 | 22 | 3 | 
| agg_keygen | 19 | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| agg_keygen | 2 | UnsetPvsAir | 1 | 2 | 2 | 
| agg_keygen | 20 | PublicValuesAir | 4 | 18 | 4 | 
| agg_keygen | 21 | RangeCheckerAir<8> | 1 | 3 | 2 | 
| agg_keygen | 22 | GkrInputAir | 19 | 19 | 4 | 
| agg_keygen | 23 | GkrLayerAir | 30 | 38 | 4 | 
| agg_keygen | 24 | GkrLayerSumcheckAir | 21 | 59 | 4 | 
| agg_keygen | 25 | GkrXiSamplerAir | 7 | 17 | 4 | 
| agg_keygen | 26 | OpeningClaimsAir | 22 | 98 | 4 | 
| agg_keygen | 27 | UnivariateRoundAir | 13 | 54 | 4 | 
| agg_keygen | 28 | SumcheckRoundsAir | 21 | 69 | 4 | 
| agg_keygen | 29 | StackingClaimsAir | 17 | 57 | 4 | 
| agg_keygen | 3 | SymbolicExpressionAir<BabyBearParameters> | 52 | 32 | 4 | 
| agg_keygen | 30 | EqBaseAir | 8 | 89 | 4 | 
| agg_keygen | 31 | EqBitsAir | 5 | 24 | 4 | 
| agg_keygen | 32 | WhirRoundAir | 31 | 28 | 4 | 
| agg_keygen | 33 | SumcheckAir | 19 | 47 | 4 | 
| agg_keygen | 34 | WhirQueryAir | 5 | 51 | 4 | 
| agg_keygen | 35 | InitialOpenedValuesAir | 13 | 145 | 4 | 
| agg_keygen | 36 | NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| agg_keygen | 37 | WhirFoldingAir | 4 | 15 | 3 | 
| agg_keygen | 38 | FinalPolyMleEvalAir | 13 | 19 | 4 | 
| agg_keygen | 39 | FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| agg_keygen | 4 | FractionsFolderAir | 17 | 41 | 4 | 
| agg_keygen | 40 | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| agg_keygen | 41 | ExpBitsLenAir | 2 | 44 | 3 | 
| agg_keygen | 5 | UnivariateSumcheckAir | 14 | 46 | 4 | 
| agg_keygen | 6 | MultilinearSumcheckAir | 14 | 60 | 4 | 
| agg_keygen | 7 | EqNsAir | 10 | 65 | 4 | 
| agg_keygen | 8 | Eq3bAir | 3 | 65 | 4 | 
| agg_keygen | 9 | EqSharpUniAir | 5 | 48 | 4 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 69 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 102,400 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 180,224 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 9,863,168 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 606,208 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 2,048 | 63 | 129,024 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 1,572,864 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 729,088 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 409,600 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.0 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 68 | 
| internal_recursive.0 | 11 | EqUniAir | 1 | prover | 4 | 16 | 64 | 
| internal_recursive.0 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.0 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.0 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 102,400 | 
| internal_recursive.0 | 15 | EqNegAir | 1 | prover | 8 | 40 | 320 | 
| internal_recursive.0 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.0 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 32,768 | 301 | 9,863,168 | 
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
| internal_recursive.0 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| internal_recursive.0 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 1,024 | 
| internal_recursive.0 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| internal_recursive.0 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 26 | 
| internal_recursive.0 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 1,472 | 
| internal_recursive.0 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 11,520 | 
| internal_recursive.0 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 10 | 
| internal_recursive.0 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 129,024 | 
| internal_recursive.0 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 216 | 
| internal_recursive.0 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 1,824 | 
| internal_recursive.0 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 17,920 | 
| internal_recursive.0 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 1,572,864 | 
| internal_recursive.0 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 204 | 
| internal_recursive.0 | 31 | EqBitsAir | 1 | prover | 2,048 | 16 | 32,768 | 
| internal_recursive.0 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 184 | 
| internal_recursive.0 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 608 | 
| internal_recursive.0 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 16,384 | 89 | 1,458,176 | 
| internal_recursive.0 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 28,672 | 
| internal_recursive.0 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 63,488 | 
| internal_recursive.0 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 8,704 | 
| internal_recursive.0 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 737,280 | 
| internal_recursive.0 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 1,856 | 
| internal_recursive.0 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| internal_recursive.0 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 131,072 | 
| internal_recursive.0 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 384 | 
| internal_recursive.0 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 4,224 | 
| internal_recursive.0 | 7 | EqNsAir | 1 | prover | 32 | 41 | 1,312 | 
| internal_recursive.0 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 409,600 | 
| internal_recursive.0 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 68 | 
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.1 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 68 | 
| internal_recursive.1 | 11 | EqUniAir | 1 | prover | 4 | 16 | 64 | 
| internal_recursive.1 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.1 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.1 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 102,400 | 
| internal_recursive.1 | 15 | EqNegAir | 1 | prover | 8 | 40 | 320 | 
| internal_recursive.1 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.1 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 16,384 | 301 | 4,931,584 | 
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
| internal_recursive.1 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| internal_recursive.1 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 1,024 | 
| internal_recursive.1 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| internal_recursive.1 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 26 | 
| internal_recursive.1 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 1,472 | 
| internal_recursive.1 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 11,520 | 
| internal_recursive.1 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 10 | 
| internal_recursive.1 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 129,024 | 
| internal_recursive.1 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 216 | 
| internal_recursive.1 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 1,824 | 
| internal_recursive.1 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 17,920 | 
| internal_recursive.1 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 1,572,864 | 
| internal_recursive.1 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 204 | 
| internal_recursive.1 | 31 | EqBitsAir | 1 | prover | 4,096 | 16 | 65,536 | 
| internal_recursive.1 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 184 | 
| internal_recursive.1 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 608 | 
| internal_recursive.1 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.1 | 35 | InitialOpenedValuesAir | 1 | prover | 8,192 | 89 | 729,088 | 
| internal_recursive.1 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 28,672 | 
| internal_recursive.1 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 63,488 | 
| internal_recursive.1 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 8,704 | 
| internal_recursive.1 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 737,280 | 
| internal_recursive.1 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 1,856 | 
| internal_recursive.1 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| internal_recursive.1 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 131,072 | 
| internal_recursive.1 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 384 | 
| internal_recursive.1 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 4,224 | 
| internal_recursive.1 | 7 | EqNsAir | 1 | prover | 32 | 41 | 1,312 | 
| internal_recursive.1 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 409,600 | 
| internal_recursive.1 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 68 | 
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 69 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 11,520 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 7,864,320 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 1,458,176 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 327,680 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 32 | 274 | 8,768 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 210 | 3,360 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 16 | 43 | 688 | 
| app_proof | 17 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 18 | KeccakfOpAir | prover | 0 | 8 | 284 | 2,272 | 
| app_proof | 19 | KeccakfPermAir | prover | 0 | 128 | 2,634 | 337,152 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 2,048 | 21 | 43,008 | 
| app_proof | 20 | XorinVmAir | prover | 0 | 8 | 669 | 5,352 | 
| app_proof | 21 | Rv64HintStoreAir | prover | 0 | 128 | 27 | 3,456 | 
| app_proof | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 4,096 | 16 | 65,536 | 
| app_proof | 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 2,048 | 24 | 49,152 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 4,096 | 18 | 73,728 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 0 | 16,384 | 26 | 425,984 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 1,024 | 48 | 49,152 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 65,536 | 56 | 3,670,016 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 256 | 62 | 15,872 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 32 | 131,072 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 16,384 | 73 | 1,196,032 | 
| app_proof | 31 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | prover | 0 | 1,024 | 38 | 38,912 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 4,096 | 45 | 184,320 | 
| app_proof | 33 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 65,536 | 48 | 3,145,728 | 
| app_proof | 34 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 35 | PhantomAir | prover | 0 | 16 | 6 | 96 | 
| app_proof | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 2,048 | 300 | 614,400 | 
| app_proof | 37 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2,048 | 551 | 1,128,448 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 647 | 662,528 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 32 | 120 | 3,840 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 274 | 4,384 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 210 | 3,360 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 4,096 | 120 | 491,520 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 14 | 219 | 14 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 152 | 10 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 130 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 36 | 267 | 35 | 5 | 0 | 6 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,560,251 | 205 | 79 | 0 | 0 | 95 | 31 | 30 | 45 | 19 | 0 | 29 | 18 | 11 | 2 | 8 | 79 | 95 | 0 | 2 | 18 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,703 | 142 | 45 | 0 | 0 | 68 | 25 | 25 | 28 | 14 | 0 | 28 | 20 | 8 | 1 | 6 | 45 | 68 | 0 | 2 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 121 | 31 | 0 | 0 | 62 | 24 | 23 | 24 | 13 | 0 | 27 | 19 | 7 | 1 | 5 | 31 | 62 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 46,796,539 | 231 | 76 | 0 | 0 | 118 | 44 | 43 | 29 | 44 | 0 | 36 | 25 | 10 | 4 | 6 | 76 | 118 | 0 | 5 | 43 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,738 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,851,522 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 18,092,548 | 173 | 21 | 0 | 2 | 123 | 77 | 76 | 17 | 27 | 0 | 28 | 18 | 9 | 1 | 7 | 22 | 123 | 0 | 2 | 27 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 6,824,482 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 148 | 469 | 148 | 135 | 0 | 4 | 12 | 112,210 | 12.75 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25909007796)
