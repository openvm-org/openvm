| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.42 |  1.42 |  1.42 |
| app_proof |  0.65 |  0.65 |  0.65 |
| leaf |  0.26 |  0.26 |  0.26 |
| internal_for_leaf |  0.23 |  0.23 |  0.23 |
| internal_recursive.0 |  0.15 |  0.15 |  0.15 |
| internal_recursive.1 |  0.13 |  0.13 |  0.13 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  593 |  593 |  593 |  593 |
| `execute_metered_time_ms` |  60 | -          | -          | -          |
| `execute_metered_insns` |  592,827 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  9.85 | -          |  9.85 |  9.85 |
| `execute_preflight_insns` |  592,827 |  592,827 |  592,827 |  592,827 |
| `execute_preflight_time_ms` |  75 |  75 |  75 |  75 |
| `execute_preflight_insn_mi/s` |  9.43 | -          |  9.43 |  9.43 |
| `trace_gen_time_ms   ` |  140 |  140 |  140 |  140 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  242 |  242 |  242 |  242 |
| `prover.main_trace_commit_time_ms` |  50 |  50 |  50 |  50 |
| `prover.rap_constraints_time_ms` |  161 |  161 |  161 |  161 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  75 |  75 |  75 |  75 |
| `prover.rap_constraints.round0_time_ms` |  56 |  56 |  56 |  56 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  12 |  12 |  12 |  12 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  261 |  261 |  261 |  261 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  29 |  29 |  29 |  29 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  231 |  231 |  231 |  231 |
| `prover.main_trace_commit_time_ms` |  74 |  74 |  74 |  74 |
| `prover.rap_constraints_time_ms` |  115 |  115 |  115 |  115 |
| `prover.openings_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.logup_gkr_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.round0_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  30 |  30 |  30 |  30 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  225 |  225 |  225 |  225 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  210 |  210 |  210 |  210 |
| `prover.main_trace_commit_time_ms` |  80 |  80 |  80 |  80 |
| `prover.rap_constraints_time_ms` |  96 |  96 |  96 |  96 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.round0_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.mle_rounds_time_ms` |  45 |  45 |  45 |  45 |
| `prover.openings.stacked_reduction_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  150 |  150 |  150 |  150 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  140 |  140 |  140 |  140 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  134 |  134 |  134 |  134 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  125 |  125 |  125 |  125 |
| `prover.main_trace_commit_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  23 |  23 |  23 |  23 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9/pairing-ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 2.84 | app_proof.0 |
| set initial memory | 2.84 | app_proof.0 |
| frac_sumcheck.gkr_rounds | 1.88 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 1.88 | app_proof.prover.0 |
| prover.gkr_input_evals | 1.82 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 1.82 | app_proof.prover.0 |
| prover.rap_constraints | 1.79 | internal_for_leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.41 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 1.40 | internal_for_leaf.0.prover |
| prover.merkle_tree | 1.40 | internal_for_leaf.0.prover |
| prover.openings | 1.40 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 1.37 | internal_for_leaf.0.prover |
| prover.before_gkr_input_evals | 1.24 | internal_for_leaf.0.prover |
| prover.stacked_commit | 1.24 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 1.23 | internal_for_leaf.0.prover |
| tracegen.exp_bits_len | 0.53 | leaf.0 |
| tracegen.pow_checker | 0.53 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.53 | leaf.0 |
| tracegen.whir_folding | 0.46 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.46 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.46 | leaf.0 |
| tracegen.public_values | 0.46 | leaf.0 |
| tracegen.proof_shape | 0.46 | leaf.0 |
| tracegen.range_checker | 0.46 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 147 | 9 | 0 | 267,111 | 228,056 | 9 | 63 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 11 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 50 | 118 | 3 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 14 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 15 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 17 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 19 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | Rv64HintStoreAir | 17 | 15 | 3 | 
| 21 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 4 | 3 | 
| 22 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 17 | 9 | 3 | 
| 23 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 12 | 10 | 3 | 
| 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 14 | 25 | 3 | 
| 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 11 | 11 | 3 | 
| 26 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 28 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| 29 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 30 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 19 | 30 | 3 | 
| 31 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| 32 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| 33 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 34 | PhantomAir | 3 |  | 1 | 
| 35 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 36 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 371 | 194 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 243 | 130 | 3 | 
| 8 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 50 | 118 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 81 | 9 |  |  | 0 |  | 367 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 593 |  |  | 60 | 592,827 | 9.85 | 0 | 660 |  | 
| internal_for_leaf |  |  |  | 225 |  |  |  |  |  |  |  |  |  | 225 | 
| internal_recursive.0 |  |  |  | 150 |  |  |  |  |  |  |  |  |  | 150 | 
| internal_recursive.1 |  |  |  | 134 |  |  |  |  |  |  |  |  |  | 134 | 
| leaf |  |  | 261 |  |  |  |  |  |  |  |  |  |  | 261 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 29 | 

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
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 180,224 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 7,864,320 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 1,458,176 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 32 | 29 | 928 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 327,680 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv64IsEqualModAdapterU16Air<2, 4, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 32 | 120 | 3,840 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 274 | 280,576 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 64 | 210 | 13,440 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | prover | 0 | 256 | 55 | 14,080 | 
| app_proof | 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 19 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 16,384 | 21 | 344,064 | 
| app_proof | 20 | Rv64HintStoreAir | prover | 0 | 128 | 27 | 3,456 | 
| app_proof | 21 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 8,192 | 16 | 131,072 | 
| app_proof | 22 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 8,192 | 24 | 196,608 | 
| app_proof | 23 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 8,192 | 18 | 147,456 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 8,192 | 32 | 262,144 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 0 | 32,768 | 26 | 851,968 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 2,048 | 48 | 98,304 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 524,288 | 56 | 29,360,128 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 8,192 | 73 | 598,016 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 16,384 | 32 | 524,288 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | prover | 0 | 512 | 38 | 19,456 | 
| app_proof | 31 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 2,048 | 45 | 92,160 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 131,072 | 48 | 6,291,456 | 
| app_proof | 33 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 34 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 35 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 16,384 | 300 | 4,915,200 | 
| app_proof | 36 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16,384 | 519 | 8,503,296 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 8,192 | 391 | 3,203,072 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 14 | 225 | 14 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 150 | 9 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 134 | 8 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 29 | 261 | 29 | 6 | 0 | 3 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,560,251 | 210 | 80 | 0 | 0 | 96 | 31 | 31 | 45 | 19 | 0 | 33 | 22 | 11 | 2 | 8 | 80 | 96 | 0 | 2 | 18 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,703 | 140 | 45 | 0 | 0 | 67 | 25 | 24 | 27 | 13 | 0 | 26 | 18 | 8 | 1 | 6 | 46 | 67 | 0 | 2 | 12 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 125 | 31 | 0 | 0 | 62 | 23 | 23 | 24 | 13 | 0 | 31 | 23 | 7 | 1 | 5 | 31 | 62 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 44,535,899 | 231 | 73 | 0 | 0 | 115 | 43 | 42 | 29 | 43 | 0 | 41 | 30 | 10 | 3 | 6 | 74 | 115 | 0 | 5 | 42 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,738 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,229,666 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 61,566,738 | 242 | 50 | 0 | 0 | 161 | 56 | 55 | 29 | 75 | 0 | 29 | 17 | 12 | 5 | 6 | 50 | 161 | 0 | 3 | 74 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 23,125,069 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 140 | 593 | 140 | 136 | 0 | 4 | 75 | 592,827 | 9.43 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ebfb12e2465f1dea9e46b0a75d9dc7311e1b82c9

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25909007796)
