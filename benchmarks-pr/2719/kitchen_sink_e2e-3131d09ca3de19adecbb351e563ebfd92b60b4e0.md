| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  232.98 |  232.56 |  232.56 |
| app_proof |  2.09 |  1.66 |  1.66 |
| leaf |  0.39 |  0.39 |  0.39 |
| internal_for_leaf |  0.14 |  0.14 |  0.14 |
| internal_recursive.0 |  0.09 |  0.09 |  0.09 |
| internal_recursive.1 |  0.08 |  0.08 |  0.08 |
| root |  1.52 |  1.52 |  1.52 |
| halo2_outer |  177.14 |  177.14 |  177.14 |
| halo2_wrapper |  51.53 |  51.53 |  51.53 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,030 |  2,060 |  1,638 |  422 |
| `execute_metered_time_ms` |  26 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  74.54 | -          |  74.54 |  74.54 |
| `execute_preflight_insns` |  989,985.50 |  1,979,971 |  1,445,000 |  534,971 |
| `execute_preflight_time_ms` |  126.50 |  253 |  182 |  71 |
| `execute_preflight_insn_mi/s` |  31.73 | -          |  31.92 |  31.54 |
| `trace_gen_time_ms   ` |  28.50 |  57 |  46 |  11 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  786.50 |  1,573 |  1,274 |  299 |
| `prover.main_trace_commit_time_ms` |  277.50 |  555 |  483 |  72 |
| `prover.rap_constraints_time_ms` |  459.50 |  919 |  721 |  198 |
| `prover.openings_time_ms` |  48.50 |  97 |  69 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  83 |  166 |  129 |  37 |
| `prover.rap_constraints.round0_time_ms` |  271.50 |  543 |  424 |  119 |
| `prover.rap_constraints.mle_rounds_time_ms` |  104.50 |  209 |  168 |  41 |
| `prover.openings.stacked_reduction_time_ms` |  35 |  70 |  53 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  15.50 |  31 |  24 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  19 |  38 |  29 |  9 |
| `prover.openings.whir_time_ms` |  12.50 |  25 |  15 |  10 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  390 |  390 |  390 |  390 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  96 |  96 |  96 |  96 |
| `generate_blob_total_time_ms` |  10 |  10 |  10 |  10 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  294 |  294 |  294 |  294 |
| `prover.main_trace_commit_time_ms` |  127 |  127 |  127 |  127 |
| `prover.rap_constraints_time_ms` |  138 |  138 |  138 |  138 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  53 |  53 |  53 |  53 |
| `prover.rap_constraints.round0_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  15 |  15 |  15 |  15 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.whir_time_ms` |  12 |  12 |  12 |  12 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  137 |  137 |  137 |  137 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  119 |  119 |  119 |  119 |
| `prover.main_trace_commit_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  10 |  10 |  10 |  10 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  92 |  92 |  92 |  92 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  80 |  80 |  80 |  80 |
| `prover.main_trace_commit_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints_time_ms` |  49 |  49 |  49 |  49 |
| `prover.openings_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.whir_time_ms` |  8 |  8 |  8 |  8 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  82 |  82 |  82 |  82 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  73 |  73 |  73 |  73 |
| `prover.main_trace_commit_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints_time_ms` |  47 |  47 |  47 |  47 |
| `prover.openings_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.whir_time_ms` |  8 |  8 |  8 |  8 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,518 |  1,518 |  1,518 |  1,518 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,401 |  1,401 |  1,401 |  1,401 |
| `prover.main_trace_commit_time_ms` |  399 |  399 |  399 |  399 |
| `prover.rap_constraints_time_ms` |  92 |  92 |  92 |  92 |
| `prover.openings_time_ms` |  909 |  909 |  909 |  909 |
| `prover.rap_constraints.logup_gkr_time_ms` |  39 |  39 |  39 |  39 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  30 |  30 |  30 |  30 |
| `prover.openings.stacked_reduction_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  902 |  902 |  902 |  902 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  177,141 |  177,141 |  177,141 |  177,141 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  51,535 |  51,535 |  51,535 |  51,535 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/3131d09ca3de19adecbb351e563ebfd92b60b4e0/kitchen_sink_e2e-3131d09ca3de19adecbb351e563ebfd92b60b4e0.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 15.08 | app_proof.prover.0 |
| prover.merkle_tree | 14.15 | app_proof.prover.0 |
| prover.openings | 14.15 | app_proof.prover.0 |
| prover.prove_whir_opening | 14.15 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 12.89 | app_proof.prover.0 |
| prover.gkr_input_evals | 12.89 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 12.89 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 12.89 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 12.70 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 12.70 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 10.94 | app_proof.prover.0 |
| prover.stacked_commit | 10.94 | app_proof.prover.0 |
| prover.rs_code_matrix | 10.93 | app_proof.prover.0 |
| generate mem proving ctxs | 3.32 | app_proof.0 |
| set initial memory | 3.32 | app_proof.1 |
| tracegen.pow_checker | 1.84 | leaf.0 |
| tracegen.exp_bits_len | 1.84 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.84 | leaf.0 |
| tracegen.whir_folding | 1.71 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.71 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.71 | leaf.0 |
| tracegen.range_checker | 1.62 | leaf.0 |
| tracegen.proof_shape | 1.62 | leaf.0 |
| tracegen.public_values | 1.62 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | merkle_tree_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 611 | 8 | 3 | 0 | 3 | 1 | 0 | 2 | 0 | 0 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 0 | RootVerifierPvsAir | 108 | 36 | 2 | 
| 1 | UserPvsCommitAir | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 3 | 25 | 4 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 11 | EqUniAir | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 
| 12 | ExpressionClaimAir | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 548 | 286 | 3 | 
| 13 | InteractionsFoldingAir | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 356 | 190 | 3 | 
| 14 | ConstraintsFoldingAir | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 372 | 194 | 3 | 
| 15 | EqNegAir | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 244 | 130 | 3 | 
| 16 | TranscriptAir | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 219 | 117 | 3 | 
| 18 | MerkleVerifyAir | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 155 | 85 | 3 | 
| 19 | ProofShapeAir<4, 8> | 78 | 85 | 4 | 
| 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 3 | 13 | 4 | 
| 20 | PublicValuesAir | 4 | 18 | 4 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 3 | 2 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 22 | GkrInputAir | 19 | 19 | 4 | 
| 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 67 | 170 | 3 | 
| 23 | GkrLayerAir | 30 | 38 | 4 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 283 | 171 | 3 | 
| 24 | GkrLayerSumcheckAir | 21 | 59 | 4 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 187 | 123 | 3 | 
| 25 | GkrXiSamplerAir | 7 | 17 | 4 | 
| 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 26 | OpeningClaimsAir | 22 | 98 | 4 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 27 | UnivariateRoundAir | 13 | 54 | 4 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 28 | SumcheckRoundsAir | 21 | 69 | 4 | 
| 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 29 | StackingClaimsAir | 17 | 57 | 4 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 35 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 13 | 320 | 4 | 
| 30 | EqBaseAir | 8 | 89 | 4 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 31 | EqBitsAir | 5 | 24 | 4 | 
| 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 32 | WhirRoundAir | 31 | 28 | 4 | 
| 33 | SumcheckAir | 19 | 47 | 4 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 34 | WhirQueryAir | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 13 | 145 | 4 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 36 | NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 37 | WhirFoldingAir | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir | 13 | 19 | 4 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 39 | FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 4 | FractionsFolderAir | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 881 | 513 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 41 | ExpBitsLenAir | 2 | 44 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 148 | 2,127 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 130 | 16 | 2 | 
| 45 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> | 48 | 69 | 3 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 45 | 31 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 69 | 71 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 130 | 85 | 3 | 
| 49 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 5 | UnivariateSumcheckAir | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 741 | 381 | 2 | 
| 50 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 51 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 52 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 54 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| 55 | Sha2MainAir<Sha512Config> | 149 | 39 | 3 | 
| 56 | Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,481 | 3 | 
| 57 | Sha2MainAir<Sha256Config> | 85 | 23 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha256Config> | 29 | 754 | 3 | 
| 59 | KeccakfOpAir | 110 | 27 | 2 | 
| 6 | MultilinearSumcheckAir | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 60 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 61 | XorinVmAir | 357 | 92 | 3 | 
| 62 | Rv64HintStoreAir | 17 | 15 | 3 | 
| 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 14 | 5 | 3 | 
| 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 15 | 9 | 3 | 
| 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 12 | 11 | 2 | 
| 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 14 | 25 | 3 | 
| 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 26 | 22 | 3 | 
| 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 33 | 31 | 3 | 
| 7 | EqNsAir | 10 | 65 | 4 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 
| 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 29 | 77 | 3 | 
| 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 38 | 180 | 3 | 
| 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 19 | 30 | 3 | 
| 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 25 | 23 | 3 | 
| 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 32 | 34 | 3 | 
| 75 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 76 | PhantomAir | 3 | 1 | 2 | 
| 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 78 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 8 | Eq3bAir | 3 | 65 | 4 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 9 | EqSharpUniAir | 5 | 48 | 4 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 75 |  |  |  | 3 |  |  | 0 |  |  | 351 | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  |  |  | 422 |  |  |  |  |  |  |  |  | 26 | 1,979,971 | 74.54 | 0 |  |  | 2,105 |  | 
| halo2_keygen |  |  |  |  |  |  |  |  |  |  |  |  | 101,147 |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_outer |  |  | 177,141 |  |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 51,535 |  |  |  |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  |  | 137 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 137 | 
| internal_recursive.0 |  |  |  |  |  |  | 92 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 92 | 
| internal_recursive.1 |  |  |  |  |  |  | 82 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 82 | 
| leaf |  |  |  |  |  | 390 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 390 | 
| root | 116 | 9 | 1,518 | 8 |  |  |  |  | 1,518 |  |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 | 0 |  | 1,518 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 5 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 6 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 7 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 1 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 78 | 89 | 4 | 
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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 606,208 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 49 | 12,544 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 52 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 2,944 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 46,080 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 20 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 2,064,384 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 1,728 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 3,648 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 143,360 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 31,457,280 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 816 | 
| leaf | 31 | EqBitsAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 368 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 1,216 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 32,768 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 23,330,816 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 229,376 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 507,904 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 69,632 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 23,592,960 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| root | 0 | RootVerifierPvsAir | prover | 1 | 206 | 206 | 
| root | 1 | UserPvsCommitAir | prover | 8 | 30 | 240 | 
| root | 10 | EqSharpUniReceiverAir | prover | 4 | 17 | 68 | 
| root | 11 | EqUniAir | prover | 4 | 16 | 64 | 
| root | 12 | ExpressionClaimAir | prover | 128 | 32 | 4,096 | 
| root | 13 | InteractionsFoldingAir | prover | 8,192 | 37 | 303,104 | 
| root | 14 | ConstraintsFoldingAir | prover | 4,096 | 25 | 102,400 | 
| root | 15 | EqNegAir | prover | 8 | 40 | 320 | 
| root | 16 | TranscriptAir | prover | 4,096 | 44 | 180,224 | 
| root | 17 | Poseidon2Air<BabyBearParameters>, 1> | prover | 16,384 | 301 | 4,931,584 | 
| root | 18 | MerkleVerifyAir | prover | 8,192 | 37 | 303,104 | 
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

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 757 | 1,514 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 565 | 1,130 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 513 | 1,026 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 385 | 770 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 4 | 116 | 464 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 292 | 18,688 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 228 | 14,592 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 4,096 | 21 | 86,016 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover | 0 | 8 | 160 | 1,280 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 2 | 390 | 780 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 4 | 294 | 1,176 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 32 | 131,072 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,004 | 4,016 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 0 | 512 | 246 | 125,952 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 169 | 43,264 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover | 0 | 256 | 90 | 23,040 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover | 0 | 256 | 126 | 32,256 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 173 | 177,152 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 1 | 949 | 949 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 150 | 2,457,600 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 119,537,664 | 
| app_proof | 59 | KeccakfOpAir | prover | 0 | 8,192 | 284 | 2,326,528 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 60 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 61 | XorinVmAir | prover | 0 | 8,192 | 669 | 5,480,448 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 32,768 | 17 | 557,056 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 65,536 | 24 | 1,572,864 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 131,072 | 26 | 3,407,872 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 32,768 | 46 | 1,507,328 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 524,288 | 54 | 28,311,552 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 512 | 58 | 29,696 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 65,536 | 73 | 4,784,128 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 512 | 38 | 19,456 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 16,384 | 41 | 671,744 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 524,288 | 48 | 25,165,824 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 2,048 | 300 | 614,400 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 2,048 | 21 | 43,008 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 2,048 | 32 | 65,536 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 1 | 64 | 246 | 15,744 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 32 | 169 | 5,408 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover | 1 | 32 | 90 | 2,880 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover | 1 | 64 | 126 | 8,064 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 128 | 173 | 22,144 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 1 | 32 | 43 | 1,376 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 1 | 8,192 | 150 | 1,228,800 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 131,072 | 456 | 59,768,832 | 
| app_proof | 59 | KeccakfOpAir | prover | 1 | 4,096 | 284 | 1,163,264 | 
| app_proof | 60 | KeccakfPermAir | prover | 1 | 65,536 | 2,634 | 172,621,824 | 
| app_proof | 61 | XorinVmAir | prover | 1 | 4,096 | 669 | 2,740,224 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 16,384 | 17 | 278,528 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 32,768 | 24 | 786,432 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 32,768 | 18 | 589,824 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 32,768 | 32 | 1,048,576 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 65,536 | 26 | 1,703,936 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 8,192 | 46 | 376,832 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 262,144 | 54 | 14,155,776 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 64 | 58 | 3,712 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 16,384 | 73 | 1,196,032 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 1 | 64 | 38 | 2,432 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 8,192 | 41 | 335,872 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 262,144 | 48 | 12,582,912 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 2,048 | 300 | 614,400 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 137 | 18 | 6 | 1 | 2 | 1 | 2 | 
| internal_recursive.0 | 1 | 12 | 92 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 82 | 8 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 96 | 390 | 96 | 23 | 10 | 18 | 2 | 2 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,979 | 119 | 38 | 0 | 0 | 62 | 20 | 19 | 29 | 12 | 0 | 18 | 10 | 8 | 1 | 6 | 38 | 62 | 0 | 1 | 11 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,767 | 80 | 16 | 0 | 0 | 49 | 19 | 18 | 20 | 10 | 0 | 14 | 8 | 5 | 1 | 4 | 16 | 49 | 0 | 1 | 9 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,863 | 73 | 11 | 0 | 0 | 47 | 18 | 18 | 18 | 10 | 0 | 13 | 8 | 5 | 0 | 4 | 11 | 47 | 0 | 1 | 9 | 0 | 0 | 
| leaf | 0 | prover | 260,613,236 | 294 | 127 | 0 | 1 | 138 | 48 | 47 | 36 | 53 | 0 | 27 | 12 | 15 | 7 | 8 | 127 | 138 | 0 | 2 | 52 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,586 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,382 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,358 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 39,253,315 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| root | prover | 18,533,670 | 1,401 | 399 | 0 | 0 | 92 | 22 | 21 | 30 | 39 | 1 | 909 | 902 | 6 | 1 | 5 | 399 | 92 | 0 | 76 | 13 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 902,782,844 | 1,274 | 482 | 0 | 277 | 721 | 424 | 423 | 168 | 129 | 69 | 69 | 15 | 53 | 24 | 29 | 483 | 721 | 0 | 1 | 59 | 0 | 0 | 
| app_proof | prover | 1 | 279,963,980 | 299 | 72 | 0 | 1 | 198 | 119 | 118 | 41 | 37 | 0 | 28 | 10 | 17 | 7 | 9 | 72 | 198 | 0 | 1 | 37 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 60,351,486 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 29,472,010 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,534 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 46 | 1,638 | 46 | 134 | 0 | 2 | 182 | 1,445,000 | 31.54 | 
| app_proof | 1 | 11 | 422 | 11 | 40 | 0 | 2 | 71 | 534,971 | 31.92 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/3131d09ca3de19adecbb351e563ebfd92b60b4e0

Max Segment Length: 4194304

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26963124954)
