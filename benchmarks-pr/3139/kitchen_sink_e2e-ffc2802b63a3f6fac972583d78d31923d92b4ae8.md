| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  9.77 |  9.53 |  9.53 |
| app_proof |  2.49 |  2.25 |  2.25 |
| leaf |  0.47 |  0.47 |  0.47 |
| internal_for_leaf |  0.19 |  0.19 |  0.19 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |
| root |  1.45 |  1.45 |  1.45 |
| halo2_outer |  3.39 |  3.39 |  3.39 |
| halo2_wrapper |  1.55 |  1.55 |  1.55 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  828 |  2,484 |  2,245 |  69 |
| `compile_metered_time_ms` |  0 |  0 |  0 |  0 |
| `execute_metered_time_ms` |  8.33 | -          | -          | -          |
| `execute_metered_insns` |  659,991 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  25.69 | -          |  77.07 |  0 |
| `set_initial_memory_time_ms` |  22.67 |  68 |  58 |  2 |
| `execute_preflight_insns` |  659,991 |  1,979,973 |  1,979,971 |  1 |
| `execute_preflight_time_ms` |  27 |  81 |  81 |  0 |
| `execute_preflight_insn_mi/s` |  24.44 | -          |  24.21 |  0.11 |
| `postflight_time_ms  ` |  27 |  81 |  73 |  4 |
| `postflight_memory_chronology_time_ms` |  2 |  6 |  6 |  0 |
| `postflight_program_index_time_ms` |  2.67 |  8 |  4 |  0 |
| `trace_gen_time_ms   ` |  27.67 |  83 |  66 |  5 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  717.67 |  2,153 |  1,952 |  57 |
| `prover.main_trace_commit_time_ms` |  182.33 |  547 |  494 |  7 |
| `prover.rap_constraints_time_ms` |  360.67 |  1,082 |  1,028 |  24 |
| `prover.openings_time_ms` |  173.33 |  520 |  428 |  18 |
| `prover.rap_constraints.logup_gkr_time_ms` |  50.33 |  151 |  126 |  11 |
| `prover.rap_constraints.round0_time_ms` |  241.67 |  725 |  709 |  7 |
| `prover.rap_constraints.mle_rounds_time_ms` |  67.33 |  202 |  193 |  4 |
| `prover.openings.stacked_reduction_time_ms` |  29.67 |  89 |  83 |  3 |
| `prover.openings.stacked_reduction.round0_time_ms` |  15 |  45 |  45 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  14 |  42 |  38 |  2 |
| `prover.openings.whir_time_ms` |  143.33 |  430 |  344 |  15 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  471 |  471 |  471 |  471 |
| `execute_preflight_time_ms` |  17 |  17 |  17 |  17 |
| `trace_gen_time_ms   ` |  102 |  102 |  102 |  102 |
| `generate_blob_total_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  369 |  369 |  369 |  369 |
| `prover.main_trace_commit_time_ms` |  96 |  96 |  96 |  96 |
| `prover.rap_constraints_time_ms` |  185 |  185 |  185 |  185 |
| `prover.openings_time_ms` |  86 |  86 |  86 |  86 |
| `prover.rap_constraints.logup_gkr_time_ms` |  66 |  66 |  66 |  66 |
| `prover.rap_constraints.round0_time_ms` |  79 |  79 |  79 |  79 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  69 |  69 |  69 |  69 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  190 |  190 |  190 |  190 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  172 |  172 |  172 |  172 |
| `prover.main_trace_commit_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints_time_ms` |  80 |  80 |  80 |  80 |
| `prover.openings_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  35 |  35 |  35 |  35 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  120 |  120 |  120 |  120 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  108 |  108 |  108 |  108 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  107 |  107 |  107 |  107 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  97 |  97 |  97 |  97 |
| `prover.main_trace_commit_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,453 |  1,453 |  1,453 |  1,453 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,356 |  1,356 |  1,356 |  1,356 |
| `prover.main_trace_commit_time_ms` |  715 |  715 |  715 |  715 |
| `prover.rap_constraints_time_ms` |  114 |  114 |  114 |  114 |
| `prover.openings_time_ms` |  526 |  526 |  526 |  526 |
| `prover.rap_constraints.logup_gkr_time_ms` |  58 |  58 |  58 |  58 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  517 |  517 |  517 |  517 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,386 |  3,386 |  3,386 |  3,386 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,550 |  1,550 |  1,550 |  1,550 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ffc2802b63a3f6fac972583d78d31923d92b4ae8/kitchen_sink_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 12.62 | app_proof.prover..0 |
| prover.rap_constraints | 9.64 | app_proof.prover..0 |
| prover.merkle_tree | 8.79 | app_proof.prover..0 |
| prover.prove_whir_opening | 8.79 | app_proof.prover..0 |
| prover.openings | 8.79 | app_proof.prover..0 |
| prover.rs_code_matrix | 8.78 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 7.71 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 7.71 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 7.65 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 7.65 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 7.61 | app_proof.prover..0 |
| prover.gkr_input_evals | 7.61 | app_proof.prover..0 |
| postflight | 5.23 | app_proof..0 |
| tracegen | 5.07 | app_proof..0 |
| generate mem proving ctxs | 5.07 | app_proof..0 |
| set initial memory | 4.91 | app_proof..0 |
| prover.before_gkr_input_evals | 4.83 | app_proof.prover..0 |
| tracegen.whir_final_poly_query_eval | 1.57 | leaf.0 |
| tracegen.pow_checker | 1.57 | leaf.0 |
| tracegen.exp_bits_len | 1.57 | leaf.0 |
| tracegen.whir_folding | 1.50 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.50 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.50 | leaf.0 |
| tracegen.proof_shape | 1.41 | leaf.0 |
| tracegen.public_values | 1.41 | leaf.0 |
| tracegen.range_checker | 1.41 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | memory_to_vec_partition_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 554 | 35 | 0 | 27 | 0 | 0 | 0 | 2 | 4 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 0 | RootVerifierPvsAir |  | 109 | 37 | 4 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | RevealAir |  | 25 | 3 | 2 | 
| 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 101 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | HintStoreAir | 1 | 18 | 12 | 3 | 
| 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 551 | 247 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 359 | 151 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 375 | 167 | 3 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 247 | 103 | 3 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 222 | 102 | 3 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 11 | 3 | 
| 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 158 | 70 | 3 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 127 | 4 | 
| 19 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 15 | 2 | 
| 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 20 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 21 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 69 | 155 | 3 | 
| 22 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 27 | 7 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 23 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 26 | 10 | 3 | 
| 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 286 | 150 | 3 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 24 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 25 | 7 | 3 | 
| 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 190 | 102 | 3 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 25 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 24 | 10 | 3 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 25 | 11 | 3 | 
| 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 24 | 7 | 3 | 
| 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 23 | 10 | 3 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 24 | 11 | 3 | 
| 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 30 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 31 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 32 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 33 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 34 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 35 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 36 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 37 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 38 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 39 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 883 | 487 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 40 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 41 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 42 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 43 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 43 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 103 | 307 | 3 | 
| 44 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 102 | 582 | 3 | 
| 45 | PhantomAir |  | 3 | 1 | 2 | 
| 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 133 | 1 | 2 | 
| 46 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 46 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 50 | 59 | 3 | 
| 47 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 47 | 21 | 3 | 
| 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 72 | 56 | 3 | 
| 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 101 | 4 | 2 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 744 | 342 | 2 | 
| 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 85 | 35 | 3 | 
| 51 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 52 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 53 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 54 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 152 | 4 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 88 | 4 | 3 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 111 | 1 | 2 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 359 | 34 | 3 | 
| 64 | RevealAir |  | 25 | 3 | 2 | 
| 65 | HintStoreAir | 1 | 18 | 12 | 3 | 
| 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 68 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
| 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 11 | 3 | 
| 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 15 | 2 | 
| 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 27 | 7 | 3 | 
| 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 26 | 10 | 3 | 
| 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 25 | 7 | 3 | 
| 79 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 24 | 10 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 25 | 11 | 3 | 
| 81 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 24 | 7 | 3 | 
| 82 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 23 | 10 | 3 | 
| 83 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 24 | 11 | 3 | 
| 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 87 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
| 90 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 91 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 92 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 93 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 98 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 99 | PhantomAir |  | 3 | 1 | 2 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | root_time_ms | prepare_preflight_time_ms | populate_inputs_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | graph_witness_gen_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | collect_pvs_time_ms | build_graph_program_time_ms | apply_merkle_precomputation_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 60 |  |  |  |  |  |  |  |  | 263 |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof | 0 |  |  |  |  |  |  |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_keygen |  |  |  |  |  |  |  |  |  |  |  | 76,724 |  |  |  |  |  |  |  |  | 4,424 |  |  | 
| halo2_outer |  |  |  | 3,386 |  |  |  |  |  | 3 |  |  |  | 23 | 148 |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper |  |  |  | 1,550 |  |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  |  | 190 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 190 | 
| internal_recursive.0 |  |  |  |  |  |  | 120 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 120 | 
| internal_recursive.1 |  |  |  |  |  |  | 107 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 107 | 
| leaf |  |  |  |  |  | 471 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 471 | 
| root |  | 96 | 11 | 1,453 | 10 |  |  | 1,453 |  |  |  |  |  |  |  | 1 | 0 | 2 | 0 |  |  | 0 | 1,453 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> | halo2_keygen | 0 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> | root_keygen | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 0 | 
| app_proof | HintStoreAir | halo2_keygen | 0 | 0 | 
| app_proof | HintStoreAir | root_keygen | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 8 | 
| app_proof | KeccakfOpAir | halo2_keygen | 0 | 0 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | KeccakfPermAir | halo2_keygen | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | PhantomAir | halo2_keygen | 0 | 0 | 
| app_proof | PhantomAir | root_keygen | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 5 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> | halo2_keygen | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> | root_keygen | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> | halo2_keygen | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> | root_keygen | 0 | 1 | 
| app_proof | RevealAir |  | 0 | 0 | 
| app_proof | RevealAir | halo2_keygen | 0 | 0 | 
| app_proof | RevealAir | root_keygen | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> | halo2_keygen | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> | halo2_keygen | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> |  | 0 | 3 | 
| app_proof | Sha2MainAir<Sha256Config> | halo2_keygen | 0 | 0 | 
| app_proof | Sha2MainAir<Sha512Config> |  | 0 | 0 | 
| app_proof | Sha2MainAir<Sha512Config> | halo2_keygen | 0 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 0 | 1 | 
| app_proof | VariableRangeCheckerAir | halo2_keygen | 0 | 1 | 
| app_proof | VariableRangeCheckerAir | root_keygen | 0 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 4 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | root_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 7 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 1 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | halo2_keygen | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> | halo2_keygen | 0 | 0 | 
| app_proof | XorinVmAir |  | 0 | 0 | 
| app_proof | XorinVmAir | halo2_keygen | 0 | 0 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 70 | 218 | 4 | 
| agg_keygen | 1 | VmPvsAir | 1 | 30 | 54 | 4 | 
| agg_keygen | 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| agg_keygen | 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| agg_keygen | 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| agg_keygen | 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| agg_keygen | 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| agg_keygen | 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| agg_keygen | 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| agg_keygen | 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| agg_keygen | 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 127 | 4 | 
| agg_keygen | 2 | UnsetPvsAir | 1 | 1 | 2 | 2 | 
| agg_keygen | 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| agg_keygen | 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| agg_keygen | 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| agg_keygen | 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| agg_keygen | 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| agg_keygen | 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| agg_keygen | 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| agg_keygen | 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| agg_keygen | 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| agg_keygen | 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| agg_keygen | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 52 | 32 | 4 | 
| agg_keygen | 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| agg_keygen | 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| agg_keygen | 32 | WhirRoundAir | 1 | 31 | 30 | 4 | 
| agg_keygen | 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| agg_keygen | 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| agg_keygen | 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| agg_keygen | 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| agg_keygen | 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| agg_keygen | 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| agg_keygen | 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| agg_keygen | 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| agg_keygen | 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| agg_keygen | 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| agg_keygen | 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| agg_keygen | 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| agg_keygen | 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| agg_keygen | 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| agg_keygen | 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 

| group | air_id | air_name | bus_index | bus_name | program | segment | metered_bus_interaction_memory_unpadded_bytes | metered_bus_interaction_memory_padding_bytes | metered_bus_interaction_cells_unpadded | metered_bus_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 2 | Program |  | 0 | 208,128 | 54,016 | 6,504 | 1,688 | 
| app_proof | 0 | ProgramAir | 2 | Program | halo2_keygen | 0 | 32 |  | 1 |  | 
| app_proof | 0 | ProgramAir | 2 | Program | root_keygen | 0 | 32 |  | 1 |  | 
| app_proof | 1 | VmConnectorAir | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 0 | Execution | halo2_keygen | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 0 | Execution | root_keygen | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program | halo2_keygen | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program | root_keygen | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange |  | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange | halo2_keygen | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange | root_keygen | 0 | 128 |  | 4 |  | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression |  | 0 | 142,592 | 119,552 | 4,456 | 3,736 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression | halo2_keygen | 0 | 2,880 | 1,216 | 90 | 38 | 
| app_proof | 101 | VariableRangeCheckerAir | 3 | VariableRange |  | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 101 | VariableRangeCheckerAir | 3 | VariableRange | halo2_keygen | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 |  | 156 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 30,080 |  | 940 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 |  | 156 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 17,792 |  | 556 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 |  | 108 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 20,352 |  | 636 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 |  | 108 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 12,160 |  | 380 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 256 |  | 8 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 2,816 |  | 88 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 128 |  | 4 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 3,584 |  | 112 |  | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 4,032 | 64 | 126 | 2 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 60,480 | 960 | 1,890 | 30 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 2,016 | 32 | 63 | 1 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 381,024 | 6,048 | 11,907 | 189 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 4,032 | 64 | 126 | 2 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 60,480 | 960 | 1,890 | 30 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 2,016 | 32 | 63 | 1 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 252,000 | 4,000 | 7,875 | 125 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory |  | 0 | 263,552 | 260,736 | 8,236 | 8,148 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory | halo2_keygen | 0 | 2,048 |  | 64 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory | root_keygen | 0 | 2,048 |  | 64 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle |  | 0 | 131,776 | 130,368 | 4,118 | 4,074 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle | halo2_keygen | 0 | 1,024 |  | 32 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle | root_keygen | 0 | 1,024 |  | 32 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression |  | 0 | 131,776 | 130,368 | 4,118 | 4,074 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression | halo2_keygen | 0 | 1,024 |  | 32 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression | root_keygen | 0 | 1,024 |  | 32 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 1 | Memory |  | 0 | 6,720 | 960 | 210 | 30 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 3 | VariableRange |  | 0 | 8,064 | 1,152 | 252 | 36 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,688 |  | 84 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 15,424 |  | 482 |  | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,032 | 1,344 | 126 | 42 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 13,920 | 4,640 | 435 | 145 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle |  | 0 | 418,176 | 368,256 | 13,068 | 11,508 | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle | halo2_keygen | 0 | 8,064 | 4,224 | 252 | 132 | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle | root_keygen | 0 | 8,064 | 4,224 | 252 | 132 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression |  | 0 | 139,392 | 122,752 | 4,356 | 3,836 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression | halo2_keygen | 0 | 2,688 | 1,408 | 84 | 44 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression | root_keygen | 0 | 2,688 | 1,408 | 84 | 44 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 | 1,664 | 156 | 52 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 79,488 | 26,496 | 2,484 | 828 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 44 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup | root_keygen | 0 | 4,194,304 |  | 131,072 |  | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 3 | VariableRange |  | 0 | 883,200 | 247,296 | 27,600 | 7,728 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | Execution |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | Memory |  | 0 | 192,000 | 53,760 | 6,000 | 1,680 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 10 | RangeTuple |  | 0 | 204,800 | 57,344 | 6,400 | 1,792 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 2 | Program |  | 0 | 6,400 | 1,792 | 200 | 56 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 3 | VariableRange |  | 0 | 230,400 | 64,512 | 7,200 | 2,016 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 6 | BitwiseLookup |  | 0 | 204,800 | 57,344 | 6,400 | 1,792 | 
| app_proof | 46 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression | root_keygen | 0 | 2,880 | 1,216 | 90 | 38 | 
| app_proof | 47 | VariableRangeCheckerAir | 3 | VariableRange | root_keygen | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | Execution |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 1 | Memory |  | 0 | 128,000 | 35,840 | 4,000 | 1,120 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 2 | Program |  | 0 | 6,400 | 1,792 | 200 | 56 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 3 | VariableRange |  | 0 | 153,600 | 43,008 | 4,800 | 1,344 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | Execution |  | 0 | 18,880 | 13,888 | 590 | 434 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 1 | Memory |  | 0 | 283,200 | 208,320 | 8,850 | 6,510 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 2 | Program |  | 0 | 9,440 | 6,944 | 295 | 217 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 3 | VariableRange |  | 0 | 368,160 | 270,816 | 11,505 | 8,463 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 3 | VariableRange |  | 0 | 460,800 | 129,024 | 14,400 | 4,032 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 6 | BitwiseLookup |  | 0 | 409,600 | 114,688 | 12,800 | 3,584 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,496 |  | 78 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 21,216 |  | 663 |  | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 3 | VariableRange |  | 0 | 665,600 | 186,368 | 20,800 | 5,824 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | Execution |  | 0 | 13,120 | 3,264 | 410 | 102 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | Memory |  | 0 | 39,360 | 9,792 | 1,230 | 306 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 10 | RangeTuple |  | 0 | 52,480 | 13,056 | 1,640 | 408 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 2 | Program |  | 0 | 6,560 | 1,632 | 205 | 51 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 3 | VariableRange |  | 0 | 39,360 | 9,792 | 1,230 | 306 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 52,480 | 13,056 | 1,640 | 408 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | 10 | RangeTuple |  | 0 | 67,108,864 |  | 2,097,152 |  | 
| app_proof | 56 | RangeTupleCheckerAir<2> | 10 | RangeTuple | halo2_keygen | 0 | 67,108,864 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 0 | Execution |  | 0 | 1,286,400 | 810,752 | 40,200 | 25,336 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 1 | Memory |  | 0 | 24,441,600 | 15,404,288 | 763,800 | 481,384 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 2 | Program |  | 0 | 643,200 | 405,376 | 20,100 | 12,668 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 3 | VariableRange |  | 0 | 28,300,800 | 17,836,544 | 884,400 | 557,392 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 8 | Sha2Block |  | 0 | 1,929,600 | 1,216,128 | 60,300 | 38,004 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 3 | VariableRange |  | 0 | 174,950,400 | 93,485,056 | 5,467,200 | 2,921,408 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 6 | BitwiseLookup |  | 0 | 87,475,200 | 46,742,528 | 2,733,600 | 1,460,704 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 8 | Sha2Block |  | 0 | 32,803,200 | 17,528,448 | 1,025,100 | 547,764 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 9 | Sha2SubAir |  | 0 | 21,868,800 | 11,685,632 | 683,400 | 365,176 | 
| app_proof | 61 | KeccakfOpAir | 0 | Execution |  | 0 | 606,080 | 442,496 | 18,940 | 13,828 | 
| app_proof | 61 | KeccakfOpAir | 1 | Memory |  | 0 | 15,758,080 | 11,504,896 | 492,440 | 359,528 | 
| app_proof | 61 | KeccakfOpAir | 2 | Program |  | 0 | 303,040 | 221,248 | 9,470 | 6,914 | 
| app_proof | 61 | KeccakfOpAir | 3 | VariableRange |  | 0 | 16,364,160 | 11,947,392 | 511,380 | 373,356 | 
| app_proof | 61 | KeccakfOpAir | 7 | KeccakfState |  | 0 | 606,080 | 442,496 | 18,940 | 13,828 | 
| app_proof | 62 | KeccakfPermAir | 7 | KeccakfState |  | 0 | 14,545,920 | 2,231,296 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir | 0 | Execution |  | 0 | 605,312 | 443,264 | 18,916 | 13,852 | 
| app_proof | 63 | XorinVmAir | 1 | Memory |  | 0 | 32,686,848 | 23,936,256 | 1,021,464 | 748,008 | 
| app_proof | 63 | XorinVmAir | 2 | Program |  | 0 | 302,656 | 221,632 | 9,458 | 6,926 | 
| app_proof | 63 | XorinVmAir | 3 | VariableRange |  | 0 | 33,897,472 | 24,822,784 | 1,059,296 | 775,712 | 
| app_proof | 63 | XorinVmAir | 6 | BitwiseLookup |  | 0 | 41,161,216 | 30,141,952 | 1,286,288 | 941,936 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | Execution |  | 0 | 8,093,376 | 295,232 | 252,918 | 9,226 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | Memory |  | 0 | 16,186,752 | 590,464 | 505,836 | 18,452 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 2 | Program |  | 0 | 4,046,688 | 147,616 | 126,459 | 4,613 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 3 | VariableRange |  | 0 | 16,186,752 | 590,464 | 505,836 | 18,452 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 36,420,192 | 1,328,544 | 1,138,131 | 41,517 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | Execution |  | 0 | 14,912 | 1,472 | 466 | 46 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 1 | Memory |  | 0 | 29,824 | 2,944 | 932 | 92 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 2 | Program |  | 0 | 7,456 | 736 | 233 | 23 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 52,192 | 5,152 | 1,631 | 161 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | Execution |  | 0 | 3,552,064 | 642,240 | 111,002 | 20,070 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | Memory |  | 0 | 7,104,128 | 1,284,480 | 222,004 | 40,140 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 2 | Program |  | 0 | 1,776,032 | 321,120 | 55,501 | 10,035 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 21,312,384 | 3,853,440 | 666,012 | 120,420 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | Execution |  | 0 | 20,456,896 | 13,097,536 | 639,278 | 409,298 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | Memory |  | 0 | 40,913,792 | 26,195,072 | 1,278,556 | 818,596 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 2 | Program |  | 0 | 10,228,448 | 6,548,768 | 319,639 | 204,649 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 92,056,032 | 58,938,912 | 2,876,751 | 1,841,841 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 0 | Execution |  | 0 | 2,045,440 | 51,712 | 63,920 | 1,616 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 1 | Memory |  | 0 | 2,045,440 | 51,712 | 63,920 | 1,616 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 2 | Program |  | 0 | 1,022,720 | 25,856 | 31,960 | 808 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 3 | VariableRange |  | 0 | 9,204,480 | 232,704 | 287,640 | 7,272 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 0 | Execution |  | 0 | 5,376,960 | 3,011,648 | 168,030 | 94,114 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 1 | Memory |  | 0 | 10,753,920 | 6,023,296 | 336,060 | 188,228 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 2 | Program |  | 0 | 2,688,480 | 1,505,824 | 84,015 | 47,057 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 3 | VariableRange |  | 0 | 21,507,840 | 12,046,592 | 672,120 | 376,456 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 0 | Execution |  | 0 | 6,644,160 | 1,744,448 | 207,630 | 54,514 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 1 | Memory |  | 0 | 6,644,160 | 1,744,448 | 207,630 | 54,514 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 2 | Program |  | 0 | 3,322,080 | 872,224 | 103,815 | 27,257 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 3 | VariableRange |  | 0 | 23,254,560 | 6,105,568 | 726,705 | 190,799 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 6,595,776 | 1,792,832 | 206,118 | 56,026 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 13,191,552 | 3,585,664 | 412,236 | 112,052 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | Program |  | 0 | 3,297,888 | 896,416 | 103,059 | 28,013 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 23,085,216 | 6,274,912 | 721,413 | 196,091 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 0 | Execution |  | 0 | 11,403,200 | 5,374,016 | 356,350 | 167,938 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 1 | Memory |  | 0 | 22,806,400 | 10,748,032 | 712,700 | 335,876 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 2 | Program |  | 0 | 5,701,600 | 2,687,008 | 178,175 | 83,969 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 3 | VariableRange |  | 0 | 22,806,400 | 10,748,032 | 712,700 | 335,876 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | Execution |  | 0 | 19,647,872 | 13,906,560 | 613,996 | 434,580 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 1 | Memory |  | 0 | 78,591,488 | 55,626,240 | 2,455,984 | 1,738,320 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 2 | Program |  | 0 | 9,823,936 | 6,953,280 | 306,998 | 217,290 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 3 | VariableRange |  | 0 | 98,239,360 | 69,532,800 | 3,069,980 | 2,172,900 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 6 | BitwiseLookup |  | 0 | 58,943,616 | 41,719,680 | 1,841,988 | 1,303,740 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | Execution |  | 0 | 18,573,440 | 14,980,992 | 580,420 | 468,156 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 1 | Memory |  | 0 | 74,293,760 | 59,923,968 | 2,321,680 | 1,872,624 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 2 | Program |  | 0 | 9,286,720 | 7,490,496 | 290,210 | 234,078 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 3 | VariableRange |  | 0 | 92,867,200 | 74,904,960 | 2,902,100 | 2,340,780 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 6 | BitwiseLookup |  | 0 | 46,433,600 | 37,452,480 | 1,451,050 | 1,170,390 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | Execution |  | 0 | 103,552 | 27,520 | 3,236 | 860 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 1 | Memory |  | 0 | 414,208 | 110,080 | 12,944 | 3,440 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 2 | Program |  | 0 | 51,776 | 13,760 | 1,618 | 430 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 3 | VariableRange |  | 0 | 517,760 | 137,600 | 16,180 | 4,300 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 6 | BitwiseLookup |  | 0 | 207,104 | 55,040 | 6,472 | 1,720 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | Execution |  | 0 | 1,455,424 | 641,728 | 45,482 | 20,054 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 1 | Memory |  | 0 | 5,821,696 | 2,566,912 | 181,928 | 80,216 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 2 | Program |  | 0 | 727,712 | 320,864 | 22,741 | 10,027 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 3 | VariableRange |  | 0 | 8,004,832 | 3,529,504 | 250,151 | 110,297 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 6 | BitwiseLookup |  | 0 | 2,183,136 | 962,592 | 68,223 | 30,081 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 0 | Execution |  | 0 | 591,424 | 457,152 | 18,482 | 14,286 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 1 | Memory |  | 0 | 1,774,272 | 1,371,456 | 55,446 | 42,858 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 2 | Program |  | 0 | 295,712 | 228,576 | 9,241 | 7,143 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 3 | VariableRange |  | 0 | 2,365,696 | 1,828,608 | 73,928 | 57,144 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 6 | BitwiseLookup |  | 0 | 591,424 | 457,152 | 18,482 | 14,286 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 0 | Execution |  | 0 | 454,080 | 70,208 | 14,190 | 2,194 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 1 | Memory |  | 0 | 1,362,240 | 210,624 | 42,570 | 6,582 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 2 | Program |  | 0 | 227,040 | 35,104 | 7,095 | 1,097 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 3 | VariableRange |  | 0 | 1,816,320 | 280,832 | 56,760 | 8,776 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 6 | BitwiseLookup |  | 0 | 227,040 | 35,104 | 7,095 | 1,097 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | Execution |  | 0 | 102,400 | 28,672 | 3,200 | 896 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 1 | Memory |  | 0 | 307,200 | 86,016 | 9,600 | 2,688 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 2 | Program |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 3 | VariableRange |  | 0 | 460,800 | 129,024 | 14,400 | 4,032 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 6 | BitwiseLookup |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | Memory |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 3 | VariableRange |  | 0 | 115,200 | 32,256 | 3,600 | 1,008 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | Execution |  | 0 | 48,512 | 17,024 | 1,516 | 532 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 1 | Memory |  | 0 | 97,024 | 34,048 | 3,032 | 1,064 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 2 | Program |  | 0 | 24,256 | 8,512 | 758 | 266 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 169,792 | 59,584 | 5,306 | 1,862 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 7 | RangeTuple | root_keygen | 0 | 33,554,432 |  | 1,048,576 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 6,656 | 1,536 | 208 | 48 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 19,968 | 4,608 | 624 | 144 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 2 | Program |  | 0 | 3,328 | 768 | 104 | 24 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 29,952 | 6,912 | 936 | 216 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | Execution |  | 0 | 1,352,512 | 744,640 | 42,266 | 23,270 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 1 | Memory |  | 0 | 4,057,536 | 2,233,920 | 126,798 | 69,810 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 2 | Program |  | 0 | 676,256 | 372,320 | 21,133 | 11,635 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 5,410,048 | 2,978,560 | 169,064 | 93,080 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | Execution |  | 0 | 1,118,784 | 978,368 | 34,962 | 30,574 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | Memory |  | 0 | 3,356,352 | 2,935,104 | 104,886 | 91,722 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 2 | Program |  | 0 | 559,392 | 489,184 | 17,481 | 15,287 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 3 | VariableRange |  | 0 | 3,356,352 | 2,935,104 | 104,886 | 91,722 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 4,475,136 | 3,913,472 | 139,848 | 122,296 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | Execution |  | 0 | 16,408,384 | 368,832 | 512,762 | 11,526 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 1 | Memory |  | 0 | 49,225,152 | 1,106,496 | 1,538,286 | 34,578 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 2 | Program |  | 0 | 8,204,192 | 184,416 | 256,381 | 5,763 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 82,041,920 | 1,844,160 | 2,563,810 | 57,630 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup |  | 0 | 4,194,304 |  | 131,072 |  | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup | halo2_keygen | 0 | 4,194,304 |  | 131,072 |  | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 102,400 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 180,224 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 38 | 622,592 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 45 | 2,880 | 
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
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 32,768 | 89 | 2,916,352 | 
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
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.0 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 68 | 
| internal_recursive.0 | 11 | EqUniAir | 1 | prover | 4 | 16 | 64 | 
| internal_recursive.0 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.0 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.0 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 102,400 | 
| internal_recursive.0 | 15 | EqNegAir | 1 | prover | 8 | 40 | 320 | 
| internal_recursive.0 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.0 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 32,768 | 301 | 9,863,168 | 
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 38 | 311,296 | 
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 45 | 2,880 | 
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
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.1 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 68 | 
| internal_recursive.1 | 11 | EqUniAir | 1 | prover | 4 | 16 | 64 | 
| internal_recursive.1 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 4,096 | 
| internal_recursive.1 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 303,104 | 
| internal_recursive.1 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 102,400 | 
| internal_recursive.1 | 15 | EqNegAir | 1 | prover | 8 | 40 | 320 | 
| internal_recursive.1 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.1 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 16,384 | 301 | 4,931,584 | 
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 38 | 311,296 | 
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 45 | 2,880 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 38 | 1,245,184 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 50 | 6,400 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 2,064,384 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 31,457,280 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 47 | 188 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 23,330,816 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 64 | 24 | 1,536 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| root | 0 | RootVerifierPvsAir | prover | 1 | 207 | 207 | 
| root | 1 | UserPvsCommitAir | prover | 8 | 30 | 240 | 
| root | 10 | EqSharpUniReceiverAir | prover | 4 | 17 | 68 | 
| root | 11 | EqUniAir | prover | 4 | 16 | 64 | 
| root | 12 | ExpressionClaimAir | prover | 128 | 32 | 4,096 | 
| root | 13 | InteractionsFoldingAir | prover | 8,192 | 37 | 303,104 | 
| root | 14 | ConstraintsFoldingAir | prover | 4,096 | 25 | 102,400 | 
| root | 15 | EqNegAir | prover | 8 | 40 | 320 | 
| root | 16 | TranscriptAir | prover | 4,096 | 44 | 180,224 | 
| root | 17 | Poseidon2Air<BabyBearParameters>, 1> | prover | 16,384 | 301 | 4,931,584 | 
| root | 18 | MerkleVerifyAir | prover | 8,192 | 38 | 311,296 | 
| root | 19 | ProofShapeAir<4, 8> | prover | 64 | 45 | 2,880 | 
| root | 2 | UserPvsInMemoryAir | prover | 32 | 20 | 640 | 
| root | 20 | PublicValuesAir | prover | 128 | 8 | 1,024 | 
| root | 21 | RangeCheckerAir<8> | prover | 256 | 2 | 512 | 
| root | 22 | GkrInputAir | prover | 1 | 26 | 26 | 
| root | 23 | GkrLayerAir | prover | 32 | 46 | 1,472 | 
| root | 24 | GkrLayerSumcheckAir | prover | 256 | 45 | 11,520 | 
| root | 25 | GkrXiSamplerAir | prover | 1 | 10 | 10 | 
| root | 26 | OpeningClaimsAir | prover | 2,048 | 63 | 129,024 | 
| root | 27 | UnivariateRoundAir | prover | 8 | 27 | 216 | 
| root | 28 | SumcheckRoundsAir | prover | 32 | 57 | 1,824 | 
| root | 29 | StackingClaimsAir | prover | 512 | 35 | 17,920 | 
| root | 3 | SymbolicExpressionAir<BabyBearParameters> | prover | 32,768 | 316 | 10,354,688 | 
| root | 30 | EqBaseAir | prover | 4 | 51 | 204 | 
| root | 31 | EqBitsAir | prover | 4,096 | 16 | 65,536 | 
| root | 32 | WhirRoundAir | prover | 4 | 46 | 184 | 
| root | 33 | SumcheckAir | prover | 16 | 38 | 608 | 
| root | 34 | WhirQueryAir | prover | 128 | 32 | 4,096 | 
| root | 35 | InitialOpenedValuesAir | prover | 8,192 | 89 | 729,088 | 
| root | 36 | NonInitialOpenedValuesAir | prover | 1,024 | 28 | 28,672 | 
| root | 37 | WhirFoldingAir | prover | 2,048 | 31 | 63,488 | 
| root | 38 | FinalPolyMleEvalAir | prover | 256 | 34 | 8,704 | 
| root | 39 | FinalPolyQueryEvalAir | prover | 16,384 | 45 | 737,280 | 
| root | 4 | FractionsFolderAir | prover | 64 | 29 | 1,856 | 
| root | 40 | PowerCheckerAir<2, 32> | prover | 32 | 4 | 128 | 
| root | 41 | ExpBitsLenAir | prover | 8,192 | 16 | 131,072 | 
| root | 5 | UnivariateSumcheckAir | prover | 16 | 24 | 384 | 
| root | 6 | MultilinearSumcheckAir | prover | 128 | 33 | 4,224 | 
| root | 7 | EqNsAir | prover | 32 | 41 | 1,312 | 
| root | 8 | Eq3bAir | prover | 16,384 | 25 | 409,600 | 
| root | 9 | EqSharpUniAir | prover | 4 | 17 | 68 | 

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 0 | ProgramAir | prover | halo2_keygen | 0 | 1 | 10 | 10 | 
| app_proof | 0 | ProgramAir | prover | root_keygen | 0 | 1 | 10 | 10 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 1 | VmConnectorAir | prover | halo2_keygen | 0 | 2 | 6 | 12 | 
| app_proof | 1 | VmConnectorAir | prover | root_keygen | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | halo2_keygen | 0 | 32 | 300 | 9,600 | 
| app_proof | 101 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 101 | VariableRangeCheckerAir | prover | halo2_keygen | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 718 | 1,436 | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 526 | 1,052 | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 486 | 972 | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 358 | 716 | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 4 | 105 | 420 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 277 | 17,728 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 213 | 13,632 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 4,096 | 38 | 155,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | halo2_keygen | 0 | 1 | 38 | 38 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | root_keygen | 0 | 1 | 38 | 38 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover |  | 0 | 8 | 145 | 1,160 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 369 | 738 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 273 | 1,092 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 8,192 | 33 | 270,336 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | halo2_keygen | 0 | 32 | 33 | 1,056 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | root_keygen | 0 | 32 | 33 | 1,056 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 978 | 3,912 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 44 | BitwiseOperationLookupAir<8> | prover | root_keygen | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | prover |  | 0 | 512 | 172 | 88,064 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover |  | 0 | 256 | 154 | 39,424 | 
| app_proof | 46 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | root_keygen | 0 | 32 | 300 | 9,600 | 
| app_proof | 47 | VariableRangeCheckerAir | prover | root_keygen | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover |  | 0 | 256 | 80 | 20,480 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover |  | 0 | 512 | 111 | 56,832 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | prover |  | 0 | 512 | 156 | 79,872 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 910 | 910 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | prover |  | 0 | 512 | 107 | 54,784 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 256 | 40 | 10,240 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover |  | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover | halo2_keygen | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover |  | 0 | 32,768 | 131 | 4,292,608 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover |  | 0 | 16,384 | 258 | 4,227,072 | 
| app_proof | 62 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover |  | 0 | 16,384 | 543 | 8,896,512 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 131,072 | 34 | 4,456,448 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 256 | 27 | 6,912 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 51 | 3,342,336 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 524,288 | 23 | 12,058,624 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 32,768 | 16 | 524,288 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 131,072 | 23 | 3,014,656 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 524,288 | 38 | 19,922,944 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 38 | 19,922,944 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 2,048 | 36 | 73,728 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 32,768 | 37 | 1,212,416 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 16,384 | 28 | 458,752 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 8,192 | 28 | 229,376 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 2,048 | 29 | 59,392 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 512 | 44 | 22,528 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1,024 | 22 | 22,528 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | root_keygen | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 128 | 33 | 4,224 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 32,768 | 28 | 917,504 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 42 | 1,376,256 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 262,144 | 29 | 7,602,176 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | prover | halo2_keygen | 0 | 65,536 | 18 | 1,179,648 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_cells_unpadded | metered_interaction_cells_padding | metered_constraint_eval_cells_unpadded | metered_constraint_eval_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 6,504 | 1,688 | 260,160 | 67,520 | 65,040 | 16,880 | 6,504 | 1,688 |  |  | 
| app_proof | 0 | ProgramAir | halo2_keygen | 0 | 1 |  | 40 |  | 10 |  | 1 |  |  |  | 
| app_proof | 0 | ProgramAir | root_keygen | 0 | 1 |  | 40 |  | 10 |  | 1 |  |  |  | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 1 | VmConnectorAir | halo2_keygen | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 1 | VmConnectorAir | root_keygen | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 495 | 165 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 4,456 | 3,736 | 5,347,200 | 4,483,200 | 1,336,800 | 1,120,800 | 4,456 | 3,736 | 147,048 | 123,288 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | halo2_keygen | 0 | 90 | 38 | 108,000 | 45,600 | 27,000 | 11,400 | 90 | 38 | 2,970 | 1,254 | 
| app_proof | 101 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 101 | VariableRangeCheckerAir | halo2_keygen | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 196 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 5,744 |  | 1,436 |  | 1,102 |  | 776 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 4,208 |  | 1,052 |  | 718 |  | 114 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 3,888 |  | 972 |  | 750 |  | 476 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,864 |  | 716 |  | 494 |  | 80 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 4 |  | 1,680 |  | 420 |  | 212 |  | 288 |  | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 69,804 | 1,108 | 17,451 | 277 | 13,986 | 222 | 5,355 | 85 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 53,676 | 852 | 13,419 | 213 | 9,954 | 158 | 2,898 | 46 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 2,059 | 2,037 | 312,968 | 309,624 | 78,242 | 77,406 | 16,472 | 16,296 | 8,236 | 8,148 | 
| app_proof | 2 | PersistentBoundaryAir<8> | halo2_keygen | 0 | 16 |  | 2,432 |  | 608 |  | 128 |  | 64 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | root_keygen | 0 | 16 |  | 2,432 |  | 608 |  | 128 |  | 64 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 258 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 7 | 1 | 4,060 | 580 | 1,015 | 145 | 483 | 69 | 728 | 104 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,952 |  | 738 |  | 572 |  | 388 |  | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 3,276 | 1,092 | 819 | 273 | 570 | 190 | 171 | 57 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 250 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 216 |  | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 4,356 | 3,836 | 574,992 | 506,352 | 143,748 | 126,588 | 17,424 | 15,344 | 39,204 | 34,524 | 
| app_proof | 3 | MemoryMerkleAir<8> | halo2_keygen | 0 | 84 | 44 | 11,088 | 5,808 | 2,772 | 1,452 | 336 | 176 | 756 | 396 | 
| app_proof | 3 | MemoryMerkleAir<8> | root_keygen | 0 | 84 | 44 | 11,088 | 5,808 | 2,772 | 1,452 | 336 | 176 | 756 | 396 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 120 | 40 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 206 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 254 |  | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 123 | 41 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 208 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 82 |  | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 11,736 | 3,912 | 2,934 | 978 | 2,649 | 883 | 855 | 285 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 210 |  | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 120 | 40 | 
| app_proof | 44 | BitwiseOperationLookupAir<8> | root_keygen | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 400 | 112 | 275,200 | 77,056 | 68,800 | 19,264 | 40,800 | 11,424 | 27,600 | 7,728 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 200 | 56 | 123,200 | 34,496 | 30,800 | 8,624 | 26,600 | 7,448 | 400 | 112 | 
| app_proof | 46 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | root_keygen | 0 | 90 | 38 | 108,000 | 45,600 | 27,000 | 11,400 | 90 | 38 | 2,970 | 1,254 | 
| app_proof | 47 | VariableRangeCheckerAir | root_keygen | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 200 | 56 | 64,000 | 17,920 | 16,000 | 4,480 | 9,400 | 2,632 | 3,800 | 1,064 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 295 | 217 | 130,980 | 96,348 | 32,745 | 24,087 | 21,240 | 15,624 | 2,360 | 1,736 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 400 | 112 | 249,600 | 69,888 | 62,400 | 17,472 | 40,400 | 11,312 | 1,600 | 448 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 1 |  | 3,640 |  | 910 |  | 744 |  | 332 |  | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 400 | 112 | 171,200 | 47,936 | 42,800 | 11,984 | 34,000 | 9,520 | 3,200 | 896 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 205 | 51 | 32,800 | 8,160 | 8,200 | 2,040 | 6,355 | 1,581 | 410 | 102 | 
| app_proof | 56 | RangeTupleCheckerAir<2> |  | 0 | 2,097,152 |  | 25,165,824 |  | 6,291,456 |  | 2,097,152 |  | 10,485,760 |  | 
| app_proof | 56 | RangeTupleCheckerAir<2> | halo2_keygen | 0 | 2,097,152 |  | 25,165,824 |  | 6,291,456 |  | 2,097,152 |  | 10,485,760 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> |  | 0 | 20,100 | 12,668 | 10,532,400 | 6,638,032 | 2,633,100 | 1,659,508 | 1,768,800 | 1,114,784 | 60,300 | 38,004 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 504 | 168 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 341,700 | 182,588 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 9,909,300 | 5,295,052 | 250,124,400 | 133,654,416 | 
| app_proof | 61 | KeccakfOpAir |  | 0 | 9,470 | 6,914 | 9,773,040 | 7,135,248 | 2,443,260 | 1,783,812 | 1,051,170 | 767,454 | 18,940 | 13,828 | 
| app_proof | 62 | KeccakfPermAir |  | 0 | 227,280 | 34,864 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 454,560 | 69,728 | 607,064,880 | 93,121,744 | 
| app_proof | 63 | XorinVmAir |  | 0 | 9,458 | 6,926 | 20,542,776 | 15,043,272 | 5,135,694 | 3,760,818 | 3,395,422 | 2,486,434 | 37,832 | 27,704 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 126,459 | 4,613 | 17,198,424 | 627,368 | 4,299,606 | 156,842 | 2,529,180 | 92,260 | 505,836 | 18,452 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 233 | 23 | 25,164 | 2,484 | 6,291 | 621 | 3,262 | 322 | 1,864 | 184 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 55,501 | 10,035 | 11,322,204 | 2,047,140 | 2,830,551 | 511,785 | 1,054,519 | 190,665 | 1,221,022 | 220,770 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 199 |  | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 319,639 | 204,649 | 29,406,788 | 18,827,708 | 7,351,697 | 4,706,927 | 5,114,224 | 3,274,384 | 1,598,195 | 1,023,245 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 31,960 | 808 | 2,045,440 | 51,712 | 511,360 | 12,928 | 447,440 | 11,312 | 127,840 | 3,232 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 84,015 | 47,057 | 7,729,380 | 4,329,244 | 1,932,345 | 1,082,311 | 1,260,225 | 705,855 | 420,075 | 235,285 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 103,815 | 27,257 | 7,474,680 | 1,962,504 | 1,868,670 | 490,626 | 1,245,780 | 327,084 | 1,038,150 | 272,570 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 103,059 | 28,013 | 12,367,080 | 3,361,560 | 3,091,770 | 840,390 | 1,442,826 | 392,182 | 824,472 | 224,104 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 178,175 | 83,969 | 17,104,800 | 8,061,024 | 4,276,200 | 2,015,256 | 1,959,925 | 923,659 | 1,247,225 | 587,783 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 306,998 | 217,290 | 46,663,696 | 33,028,080 | 11,665,924 | 8,257,020 | 8,288,946 | 5,866,830 | 1,841,988 | 1,303,740 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 290,210 | 234,078 | 44,111,920 | 35,579,856 | 11,027,980 | 8,894,964 | 7,545,460 | 6,086,028 | 1,741,260 | 1,404,468 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 1,618 | 430 | 232,992 | 61,920 | 58,248 | 15,480 | 40,450 | 10,750 | 9,708 | 2,580 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 558 | 186 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 22,741 | 10,027 | 3,365,668 | 1,483,996 | 841,417 | 370,999 | 568,525 | 250,675 | 136,446 | 60,162 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 9,241 | 7,143 | 1,034,992 | 800,016 | 258,748 | 200,004 | 175,579 | 135,717 | 55,446 | 42,858 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 7,095 | 1,097 | 794,640 | 122,864 | 198,660 | 30,716 | 127,710 | 19,746 | 42,570 | 6,582 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 1,600 | 448 | 185,600 | 51,968 | 46,400 | 12,992 | 30,400 | 8,512 | 9,600 | 2,688 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 400 | 112 | 70,400 | 19,712 | 17,600 | 4,928 | 6,400 | 1,792 | 8,800 | 2,464 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 758 | 266 | 66,704 | 23,408 | 16,676 | 5,852 | 10,612 | 3,724 | 3,790 | 1,330 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | root_keygen | 0 | 1,048,576 |  | 12,582,912 |  | 3,145,728 |  | 1,048,576 |  | 5,242,880 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 217 |  | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 104 | 24 | 13,728 | 3,168 | 3,432 | 792 | 1,872 | 432 | 832 | 192 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 21,133 | 11,635 | 2,366,896 | 1,303,120 | 591,724 | 325,780 | 359,261 | 197,795 | 169,064 | 93,080 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 17,481 | 15,287 | 2,936,808 | 2,568,216 | 734,202 | 642,054 | 402,063 | 351,601 | 69,924 | 61,148 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 256,381 | 5,763 | 29,740,196 | 668,508 | 7,435,049 | 167,127 | 4,871,239 | 109,497 | 2,051,048 | 46,104 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | halo2_keygen | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 0 | 
| app_proof | interpreter | halo2_keygen | 0 | 
| app_proof | interpreter | root_keygen | 0 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 190 | 17 | 5 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 120 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 102 | 471 | 102 | 17 | 6 | 17 | 15 | 15 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,594,365 | 172 | 45 | 0 | 0 | 80 | 28 | 27 | 37 | 14 | 0 | 45 | 35 | 10 | 2 | 7 | 45 | 45 | 80 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,386,961 | 108 | 19 | 0 | 0 | 56 | 20 | 20 | 23 | 11 | 0 | 32 | 24 | 8 | 1 | 6 | 20 | 19 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,759,057 | 97 | 14 | 0 | 0 | 54 | 20 | 19 | 22 | 11 | 0 | 28 | 20 | 7 | 1 | 6 | 14 | 14 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 167,516,989 | 369 | 96 | 0 | 0 | 185 | 79 | 78 | 39 | 66 | 0 | 86 | 69 | 17 | 7 | 9 | 96 | 96 | 185 | 0 | 3 | 66 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,643 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | halo2_section_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | prover |  |  | 6 |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  | 5 |  |  |  |  | 
| halo2_keygen | ifft_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 118 |  |  |  | 
| halo2_keygen | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_keygen | lagrange_to_coeff_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 118 |  |  |  | 
| halo2_keygen | multiexp_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_outer | add_blinding_factors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | advice_msms |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 674 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 34 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_normalize_commitments |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | coeff_to_extended_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | commit_vanishing_h_x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 284 |  |  |  | 
| halo2_outer | commit_vanishing_random_poly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_outer | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_outer | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 35 |  |  |  | 
| halo2_outer | convert_raw_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 927 |  |  |  | 
| halo2_outer | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3,232 |  |  |  | 
| halo2_outer | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_outer | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,037 |  |  |  | 
| halo2_outer | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  | 
| halo2_outer | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_outer | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_pk_from_host |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 205 |  |  |  | 
| halo2_outer | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 164 |  |  |  | 
| halo2_outer | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_msm_instance_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 896 |  |  |  | 
| halo2_outer | instance_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 161 |  |  |  | 
| halo2_outer | instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 59 |  |  |  | 
| halo2_outer | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lagrange_to_coeff |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 153 |  |  |  | 
| halo2_outer | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 103 |  |  |  | 
| halo2_outer | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 245 |  |  |  | 
| halo2_outer | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 92 |  |  |  | 
| halo2_outer | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 896 |  |  |  | 
| halo2_outer | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_outer | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 370 |  |  |  | 
| halo2_outer | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_outer | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation_quotient_poly_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 928 |  |  |  | 
| halo2_outer | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 153 |  |  |  | 
| halo2_outer | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 474 |  |  |  | 
| halo2_outer | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 370 |  |  |  | 
| halo2_outer | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 103 |  |  |  | 
| halo2_outer | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,037 |  |  |  | 
| halo2_outer | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 284 |  |  |  | 
| halo2_outer | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 353 |  |  |  | 
| halo2_outer | phase5_multiopen |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 297 |  |  |  | 
| halo2_outer | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  | 
| halo2_outer | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 205 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 297 |  |  |  | 
| halo2_outer | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 92 |  |  |  | 
| halo2_outer | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_outer | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | transcript_write_squeeze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 284 |  |  |  | 
| halo2_outer | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | add_blinding_factors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 13 |  |  |  | 
| halo2_wrapper | advice_msms |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 62 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_witness |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 29 |  |  |  | 
| halo2_wrapper | batch_normalize_commitments |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | coeff_to_extended_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | commit_vanishing_h_x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 198 |  |  |  | 
| halo2_wrapper | commit_vanishing_random_poly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 13 |  |  |  | 
| halo2_wrapper | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | convert_raw_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 274 |  |  |  | 
| halo2_wrapper | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,311 |  |  |  | 
| halo2_wrapper | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 29 |  |  |  | 
| halo2_wrapper | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 13 |  |  |  | 
| halo2_wrapper | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 296 |  |  |  | 
| halo2_wrapper | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_wrapper | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_pk_from_host |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 72 |  |  |  | 
| halo2_wrapper | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_msm_instance_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 259 |  |  |  | 
| halo2_wrapper | instance_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 39 |  |  |  | 
| halo2_wrapper | instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 144 |  |  |  | 
| halo2_wrapper | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lagrange_to_coeff |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 62 |  |  |  | 
| halo2_wrapper | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 
| halo2_wrapper | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 259 |  |  |  | 
| halo2_wrapper | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96 |  |  |  | 
| halo2_wrapper | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 
| halo2_wrapper | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_quotient_poly_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 13 |  |  |  | 
| halo2_wrapper | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 274 |  |  |  | 
| halo2_wrapper | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 62 |  |  |  | 
| halo2_wrapper | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 147 |  |  |  | 
| halo2_wrapper | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96 |  |  |  | 
| halo2_wrapper | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 
| halo2_wrapper | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 296 |  |  |  | 
| halo2_wrapper | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 198 |  |  |  | 
| halo2_wrapper | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 152 |  |  |  | 
| halo2_wrapper | phase5_multiopen |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 132 |  |  |  | 
| halo2_wrapper | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_wrapper | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 132 |  |  |  | 
| halo2_wrapper | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | synthesize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 149 |  |  |  | 
| halo2_wrapper | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | transcript_write_squeeze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 198 |  |  |  | 
| halo2_wrapper | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| root | prover | 18,541,863 | 1,356 | 715 | 0 | 1 | 114 | 22 | 21 | 33 | 58 | 0 | 526 | 517 | 8 | 1 | 7 | 715 | 715 | 114 | 0 | 133 |  | 12 | 0 | 0 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,045,458,492 | 1,952 | 494 | 0 | 73 | 1,028 | 709 | 707 | 193 | 126 | 0 | 428 | 344 | 83 | 45 | 38 | 494 | 494 | 1,028 | 0 | 1 | 124 | 0 | 0 | 
| app_proof | prover | halo2_keygen | 0 | 8,530,396 | 57 | 7 | 0 | 0 | 30 | 9 | 9 | 5 | 14 | 0 | 18 | 15 | 3 | 0 | 2 | 7 | 7 | 30 | 0 | 1 | 14 | 0 | 0 | 
| app_proof | prover | root_keygen | 0 | 5,384,668 | 144 | 45 | 0 | 0 | 24 | 7 | 7 | 4 | 11 | 0 | 74 | 71 | 3 | 0 | 2 | 46 | 45 | 24 | 0 | 6 | 10 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 85,613,970 | 2,013,265,921 | 
| app_proof | prover | halo2_keygen | 0 | 0 | 2,490,547 | 2,013,265,921 | 
| app_proof | prover | root_keygen | 0 | 0 | 1,441,971 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,535 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 2,245 | 25 | 1,979,971 | 77.07 | 0 | 2,326 | 
| app_proof | halo2_keygen | 69 | 0 | 1 | 0 | 0 | 152 | 
| app_proof | root_keygen | 170 | 0 | 1 | 0 | 0 | 172 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_whir_memory_bytes | metered_secondary_peak_memory_bytes | metered_rs_code_matrix_memory_bytes | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_gkr_memory_bytes | metered_batch_constraint_memory_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 58 | 2 | 66 | 2,245 | 3 | 58 | 0 | 73 | 0 | 6 | 0 | 125,829,120 | 8,514,437,120 | 8,388,608,000 | 10,140,004,592 | 2,561,181,696 | 12,701,186,288 | 3,293,426,240 | 5,241,766,836 | 2 | 0 | 0 | 3 | 63 | 81 | 1,979,971 | 24.21 | 0 | 0 | 
| app_proof | halo2_keygen | 0 | 2 | 1 | 5 | 69 | 2 | 2 | 0 | 4 | 4 | 0 |  | 125,829,120 | 482,894,176 | 83,886,080 | 517,087,656 | 58,256 | 517,145,912 | 482,894,176 | 374,909,488 | 2 | 0 |  | 2 | 2 | 0 | 1 | 0.13 | 0 | 0 | 
| app_proof | root_keygen | 0 | 8 | 7 | 12 | 170 | 8 | 8 | 0 | 4 | 4 | 0 |  | 536,870,912 | 671,088,640 | 134,217,728 | 692,706,056 | 51,408 | 692,757,464 | 449,077,600 | 332,966,448 | 7 | 0 |  | 7 | 3 | 0 | 1 | 0.11 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms | halo2_section_time_ms |
| --- | --- | --- | --- | --- | --- |
| kzg.g_device_first_touch |  |  |  |  | 27 | 
| kzg.g_lagrange_device_first_touch |  |  |  |  | 0 | 
| prover | 6 | 0 | 6 | 5 |  | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ffc2802b63a3f6fac972583d78d31923d92b4ae8

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33195648040)
