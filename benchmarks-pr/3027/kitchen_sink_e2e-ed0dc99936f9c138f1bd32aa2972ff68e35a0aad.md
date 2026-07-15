| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  13.46 |  13.46 |  13.46 |
| app_proof |  2.73 |  2.73 |  2.73 |
| leaf |  0.46 |  0.46 |  0.46 |
| internal_for_leaf |  0.19 |  0.19 |  0.19 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |
| root |  1.43 |  1.43 |  1.43 |
| halo2_outer |  6.65 |  6.65 |  6.65 |
| halo2_wrapper |  1.77 |  1.77 |  1.77 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,705 |  2,705 |  2,705 |  2,705 |
| `execute_metered_time_ms` |  27 | -          | -          | -          |
| `execute_metered_insns` |  2,579,903 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  93.100 | -          |  93.100 |  93.100 |
| `execute_preflight_insns` |  2,579,903 |  2,579,903 |  2,579,903 |  2,579,903 |
| `execute_preflight_time_ms` |  451 |  451 |  451 |  451 |
| `execute_preflight_insn_mi/s` |  34.05 | -          |  34.05 |  34.05 |
| `trace_gen_time_ms   ` |  75 |  75 |  75 |  75 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,008 |  2,008 |  2,008 |  2,008 |
| `prover.main_trace_commit_time_ms` |  555 |  555 |  555 |  555 |
| `prover.rap_constraints_time_ms` |  975 |  975 |  975 |  975 |
| `prover.openings_time_ms` |  477 |  477 |  477 |  477 |
| `prover.rap_constraints.logup_gkr_time_ms` |  119 |  119 |  119 |  119 |
| `prover.rap_constraints.round0_time_ms` |  654 |  654 |  654 |  654 |
| `prover.rap_constraints.mle_rounds_time_ms` |  200 |  200 |  200 |  200 |
| `prover.openings.stacked_reduction_time_ms` |  91 |  91 |  91 |  91 |
| `prover.openings.stacked_reduction.round0_time_ms` |  48 |  48 |  48 |  48 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  42 |  42 |  42 |  42 |
| `prover.openings.whir_time_ms` |  385 |  385 |  385 |  385 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  463 |  463 |  463 |  463 |
| `execute_preflight_time_ms` |  22 |  22 |  22 |  22 |
| `trace_gen_time_ms   ` |  107 |  107 |  107 |  107 |
| `generate_blob_total_time_ms` |  7 |  7 |  7 |  7 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  355 |  355 |  355 |  355 |
| `prover.main_trace_commit_time_ms` |  100 |  100 |  100 |  100 |
| `prover.rap_constraints_time_ms` |  166 |  166 |  166 |  166 |
| `prover.openings_time_ms` |  88 |  88 |  88 |  88 |
| `prover.rap_constraints.logup_gkr_time_ms` |  58 |  58 |  58 |  58 |
| `prover.rap_constraints.round0_time_ms` |  67 |  67 |  67 |  67 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  71 |  71 |  71 |  71 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  193 |  193 |  193 |  193 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  17 |  17 |  17 |  17 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  175 |  175 |  175 |  175 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  78 |  78 |  78 |  78 |
| `prover.openings_time_ms` |  50 |  50 |  50 |  50 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  39 |  39 |  39 |  39 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  119 |  119 |  119 |  119 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  108 |  108 |  108 |  108 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  106 |  106 |  106 |  106 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  97 |  97 |  97 |  97 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,425 |  1,425 |  1,425 |  1,425 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,337 |  1,337 |  1,337 |  1,337 |
| `prover.main_trace_commit_time_ms` |  714 |  714 |  714 |  714 |
| `prover.rap_constraints_time_ms` |  96 |  96 |  96 |  96 |
| `prover.openings_time_ms` |  526 |  526 |  526 |  526 |
| `prover.rap_constraints.logup_gkr_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  32 |  32 |  32 |  32 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  517 |  517 |  517 |  517 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,650 |  6,650 |  6,650 |  6,650 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,772 |  1,772 |  1,772 |  1,772 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ed0dc99936f9c138f1bd32aa2972ff68e35a0aad/kitchen_sink_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 13.47 | app_proof.prover.0 |
| prover.rap_constraints | 10.14 | app_proof.prover.0 |
| prover.merkle_tree | 9.36 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.36 | app_proof.prover.0 |
| prover.openings | 9.36 | app_proof.prover.0 |
| prover.rs_code_matrix | 9.35 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 8.51 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 8.51 | app_proof.prover.0 |
| prover.gkr_input_evals | 8.42 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 8.42 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 8.00 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 8.00 | app_proof.prover.0 |
| generate mem proving ctxs | 5.42 | app_proof.0 |
| set initial memory | 5.42 | app_proof.0 |
| prover.before_gkr_input_evals | 5.12 | app_proof.prover.0 |
| tracegen.exp_bits_len | 1.59 | leaf.0 |
| tracegen.pow_checker | 1.59 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.59 | leaf.0 |
| tracegen.whir_folding | 1.52 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.52 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.52 | leaf.0 |
| tracegen.public_values | 1.42 | leaf.0 |
| tracegen.range_checker | 1.42 | leaf.0 |
| tracegen.proof_shape | 1.42 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 606 | 33 | 26 | 0 | 0 | 0 | 2 | 4 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 0 | RootVerifierPvsAir |  | 109 | 37 | 4 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 691 | 322 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 499 | 226 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 467 | 218 | 3 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 339 | 154 | 3 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 266 | 129 | 3 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 202 | 97 | 3 | 
| 19 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 85 | 4 | 
| 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PhantomAir |  | 3 | 1 | 2 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> |  | 113 | 326 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> |  | 354 | 189 | 3 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> |  | 258 | 141 | 3 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 976 | 537 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 43 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, ShiftCoreAir<32, 8> |  | 163 | 2,139 | 3 | 
| 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> |  | 145 | 28 | 2 | 
| 45 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchLessThanCoreAir<32, 8> |  | 78 | 125 | 3 | 
| 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> |  | 76 | 55 | 3 | 
| 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> |  | 115 | 131 | 3 | 
| 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> |  | 145 | 97 | 3 | 
| 49 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 884 | 417 | 2 | 
| 50 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 52 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 53 | Sha2MainAir<Sha512Config> | 1 | 276 | 71 | 3 | 
| 54 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 55 | Sha2MainAir<Sha256Config> | 1 | 148 | 39 | 3 | 
| 56 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 57 | KeccakfOpAir |  | 310 | 52 | 2 | 
| 58 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 59 | XorinVmAir |  | 561 | 177 | 3 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 60 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 
| 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 66 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 
| 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 71 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 72 | PhantomAir |  | 3 | 1 | 2 | 
| 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 74 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 
| 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 57 |  |  |  |  |  |  |  | 249 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  | 2,705 |  |  |  |  |  |  |  | 27 | 2,579,903 | 93.100 | 0 |  |  | 2,763 |  | 
| halo2_keygen |  |  |  |  |  |  |  | 289 |  | 68,033 |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 323 |  | 
| halo2_outer |  |  | 6,650 |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 1,772 |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  | 193 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 193 | 
| internal_recursive.0 |  |  |  |  |  | 119 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 119 | 
| internal_recursive.1 |  |  |  |  |  | 106 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 107 | 
| leaf |  |  |  |  | 463 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 463 | 
| root | 88 | 10 | 1,425 | 9 |  |  | 1,425 |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 | 0 |  | 1,425 | 
| root_keygen |  |  |  |  |  |  |  | 389 |  |  |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 390 |  | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 21 | KeccakfOpAir | 0 | 3,977,400 | 
| app_proof | Sha2MainAir<Sha256Config> | 23 | Sha2MainAir<Sha256Config> | 0 | 6,753,600 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 44,986,864 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,279,576 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1,373,632 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,391,960 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 7,605,440 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,391,552 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 56 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 1,680 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 38 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 1,232 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 41 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 352 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 44 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 1,232 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 47 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 352 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 50 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 1,232 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 53 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 352 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 59 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 352 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 62 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 704 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,804,516 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 63,817,680 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 27 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 18,172 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 892,388 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 68 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 876 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 70 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 876 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 72 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 876 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 74 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 1,260 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 54 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 1,008 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 55 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 672 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 63 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 864 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 64 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 864 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 67 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 432 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 69 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 432 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 71 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 432 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 65 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 1,248 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 66 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 1,248 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 73 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 624 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 288,000 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 31 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 | 141,600 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 34 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 47,200 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 720 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 37 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 40 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 720 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 43 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 45 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 46 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 720 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 49 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 51 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 52 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 57 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 58 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 60 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 15,120 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 61 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 15,120 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 32 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 | 33,600 | 
| app_proof | XorinVmAir | 19 | XorinVmAir | 0 | 8,133,880 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 |  | 2 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 |  | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 5 | 5 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 |  | 1 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 |  | 0 | 
| app_proof | XorinVmAir | 0 |  | 5 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 70 | 217 | 4 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 92 | 4 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 48 | 6,144 | 
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
| leaf | 31 | EqBitsAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
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

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 3 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 63 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 63 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | MUL | 0 | 200 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | BEQ | 0 | 200 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | SLTU | 0 | 590 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | ADD | 0 | 600 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | AND | 0 | 200 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | SUB | 0 | 200 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | XOR | 0 | 200 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | MUL | 0 | 413 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 20,100 | 
| app_proof | 57 | KeccakfOpAir | KECCAKF | 0 | 9,470 | 
| app_proof | 59 | XorinVmAir | XORIN | 0 | 9,458 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 0 | 31,871 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 0 | 63,739 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 0 | 40,082 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 0 | 3,404 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGE | 0 | 16 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 0 | 626 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 0 | 12 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 0 | 189,482 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 93,632 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 116,167 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 0 | 26,302 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 0 | 494,969 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 0 | 11,128 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 0 | 531,229 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 0 | 20,416 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SRL | 0 | 6,000 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 0 | 43,838 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 600,407 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 0 | 140,859 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 0 | 61,681 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 61,785 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | XOR | 0 | 400 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 

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
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 919 | 1,838 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 727 | 1,454 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 623 | 1,246 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 495 | 990 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 4 | 208 | 832 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 350 | 22,400 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 286 | 18,304 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 21 | 172,032 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | prover | 0 | 8 | 296 | 2,368 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 474 | 948 | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 378 | 1,512 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 16,384 | 33 | 540,672 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,128 | 4,512 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 227 | 58,112 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 0 | 256 | 166 | 42,496 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 1,024 | 232 | 237,568 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 2,048 | 231 | 473,088 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 1,111 | 1,111 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 512 | 31 | 15,872 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 0 | 32,768 | 284 | 9,306,112 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 57 | KeccakfOpAir | prover | 0 | 16,384 | 561 | 9,191,424 | 
| app_proof | 58 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 59 | XorinVmAir | prover | 0 | 16,384 | 914 | 14,974,976 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 655,360 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 65,536 | 28 | 1,835,008 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 262,144 | 32 | 8,388,608 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 2,097,152 | 41 | 85,983,232 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 32,768 | 53 | 1,736,704 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 65,536 | 37 | 2,424,832 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| halo2_keygen | 0 | ProgramAir | prover | 0 | 1 | 10 | 10 | 
| halo2_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| halo2_keygen | 52 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| halo2_keygen | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| halo2_keygen | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| halo2_keygen | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| root_keygen | 0 | ProgramAir | prover | 0 | 1 | 10 | 10 | 
| root_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| root_keygen | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| root_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| root_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| root_keygen | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| root_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| root_keygen | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 1,572,864 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 10,064 | 6,320 | 251,600 | 158,000 | 402,560 | 252,800 | 100,640 | 63,200 | 364,820 | 229,100 | 10,064 | 6,320 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,733 | 1,577 | 7,572 | 2,524 | 1,893 | 631 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 1,878 |  | 3,004 |  | 751 |  | 21,605 |  | 596 |  | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 4,595 |  | 7,352 |  | 1,838 |  | 50,098 |  | 1,382 |  | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 3,635 |  | 5,816 |  | 1,454 |  | 36,178 |  | 998 |  | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 3,115 |  | 4,984 |  | 1,246 |  | 33,858 |  | 934 |  | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 2,475 |  | 3,960 |  | 990 |  | 24,578 |  | 678 |  | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4 |  | 2,080 |  | 3,328 |  | 832 |  | 11,745 |  | 324 |  | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 55,125 | 875 | 88,200 | 1,400 | 22,050 | 350 | 607,478 | 9,642 | 16,758 | 266 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 45,045 | 715 | 72,072 | 1,144 | 18,018 | 286 | 461,318 | 7,322 | 12,726 | 202 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,040 |  | 1,664 |  | 416 |  | 5,873 |  | 162 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 8,960 | 7,424 | 470,400 | 389,760 | 752,640 | 623,616 | 188,160 | 155,904 | 1,299,200 | 1,076,480 | 35,840 | 29,696 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,310 |  | 2,096 |  | 524 |  | 12,905 |  | 356 |  | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 7 | 1 | 5,180 | 740 | 8,288 | 1,184 | 2,072 | 296 | 28,674 | 4,096 | 791 | 113 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 2,370 |  | 3,792 |  | 948 |  | 25,665 |  | 708 |  | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 2,835 | 945 | 4,536 | 1,512 | 1,134 | 378 | 28,058 | 9,352 | 774 | 258 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,040 |  | 1,664 |  | 416 |  | 5,873 |  | 162 |  | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,310 |  | 2,096 |  | 524 |  | 12,905 |  | 356 |  | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 3,640 | 520 | 5,824 | 832 | 1,456 | 208 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 12,040 | 4,344 | 1,986,600 | 716,760 | 1,589,280 | 573,408 | 397,320 | 143,352 | 1,745,800 | 629,880 | 48,160 | 17,376 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,965 | 655 | 3,144 | 1,048 | 786 | 262 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,040 |  | 1,664 |  | 416 |  | 5,873 |  | 162 |  | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,310 |  | 2,096 |  | 524 |  | 12,905 |  | 356 |  | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 3,640 | 520 | 5,824 | 832 | 1,456 | 208 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,965 | 655 | 3,144 | 1,048 | 786 | 262 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,040 |  | 1,664 |  | 416 |  | 5,873 |  | 162 |  | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,310 |  | 2,096 |  | 524 |  | 12,905 |  | 356 |  | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 8,460 | 2,820 | 13,536 | 4,512 | 3,384 | 1,128 | 106,140 | 35,380 | 2,928 | 976 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 3,640 | 520 | 5,824 | 832 | 1,456 | 208 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,630 |  | 2,608 |  | 652 |  | 17,545 |  | 484 |  | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,965 | 655 | 3,144 | 1,048 | 786 | 262 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 200 | 56 | 113,500 | 31,780 | 181,600 | 50,848 | 45,400 | 12,712 | 1,051,250 | 294,350 | 29,000 | 8,120 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 | 200 | 56 | 83,000 | 23,240 | 132,800 | 37,184 | 33,200 | 9,296 | 551,000 | 154,280 | 15,200 | 4,256 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 | 590 | 434 | 342,200 | 251,720 | 547,520 | 402,752 | 136,880 | 100,688 | 2,459,563 | 1,809,237 | 67,850 | 49,910 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 1,200 | 848 | 693,000 | 489,720 | 1,108,800 | 783,552 | 277,200 | 195,888 | 6,307,500 | 4,457,300 | 174,000 | 122,960 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 2,778 |  | 4,444 |  | 1,111 |  | 32,045 |  | 884 |  | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 413 | 99 | 32,008 | 7,672 | 51,212 | 12,276 | 12,803 | 3,069 | 284,454 | 68,186 | 7,847 | 1,881 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | 0 | 20,100 | 12,668 | 28,542,000 | 17,988,560 | 22,833,600 | 14,390,848 | 5,708,400 | 3,597,712 | 107,836,500 | 67,963,820 | 2,974,800 | 1,874,864 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 341,700 | 182,588 | 779,076,000 | 416,300,640 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 359,212,125 | 191,945,635 | 9,909,300 | 5,295,052 | 
| app_proof | 57 | KeccakfOpAir | 0 | 9,470 | 6,914 | 13,281,675 | 9,696,885 | 21,250,680 | 15,515,016 | 5,312,670 | 3,878,754 | 106,419,125 | 77,696,075 | 2,935,700 | 2,143,340 | 
| app_proof | 58 | KeccakfPermAir | 0 | 227,280 | 34,864 | 2,993,277,600 | 459,158,880 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 16,477,800 | 2,527,640 | 454,560 | 69,728 | 
| app_proof | 59 | XorinVmAir | 0 | 9,458 | 6,926 | 21,611,530 | 15,825,910 | 34,578,448 | 25,321,456 | 8,644,612 | 6,330,364 | 192,340,253 | 140,848,867 | 5,305,938 | 3,885,486 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,733 | 1,577 | 7,572 | 2,524 | 1,893 | 631 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 31,871 | 897 | 1,593,550 | 44,850 | 2,549,680 | 71,760 | 637,420 | 17,940 | 13,863,885 | 390,195 | 382,452 | 10,764 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 63,739 | 1,797 | 4,461,730 | 125,790 | 7,138,768 | 201,264 | 1,784,692 | 50,316 | 36,968,620 | 1,042,260 | 1,019,824 | 28,752 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 43,486 | 22,050 | 1,956,870 | 992,250 | 3,130,992 | 1,587,600 | 782,748 | 396,900 | 15,763,675 | 7,993,125 | 434,860 | 220,500 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 190,136 | 72,008 | 15,210,880 | 5,760,640 | 24,337,408 | 9,217,024 | 6,084,352 | 2,304,256 | 89,601,590 | 33,933,770 | 2,471,768 | 936,104 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 209,799 | 52,345 | 13,636,935 | 3,402,425 | 21,819,096 | 5,443,880 | 5,454,774 | 1,360,970 | 83,657,352 | 20,872,568 | 2,307,789 | 575,795 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,063,628 | 1,033,524 | 109,021,870 | 105,936,210 | 174,434,992 | 169,497,936 | 43,608,748 | 42,374,484 | 655,460,755 | 636,909,165 | 18,081,676 | 17,569,908 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 26,416 | 6,352 | 3,500,120 | 841,640 | 5,600,192 | 1,346,624 | 1,400,048 | 336,656 | 22,981,920 | 5,526,240 | 633,984 | 152,448 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 43,838 | 21,698 | 4,055,015 | 2,007,065 | 6,488,024 | 3,211,304 | 1,622,006 | 802,826 | 28,604,295 | 14,157,945 | 789,084 | 390,564 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 1,878 |  | 3,004 |  | 751 |  | 21,605 |  | 596 |  | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 865,132 | 183,444 | 77,861,880 | 16,509,960 | 124,579,008 | 26,415,936 | 31,144,752 | 6,603,984 | 627,220,700 | 132,996,900 | 17,302,640 | 3,668,880 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 21,000 | 11,768 | 15,750,000 | 8,826,000 | 25,200,000 | 14,121,600 | 6,300,000 | 3,530,400 | 761,250 | 426,590 | 21,000 | 11,768 | 
| app_proof | 74 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,733 | 1,577 | 7,572 | 2,524 | 1,893 | 631 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 1,878 |  | 3,004 |  | 751 |  | 21,605 |  | 596 |  | 
| halo2_keygen | 0 | ProgramAir | 0 | 1 |  | 25 |  | 40 |  | 10 |  | 37 |  | 1 |  | 
| halo2_keygen | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | 0 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | 0 | 172 | 84 | 28,380 | 13,860 | 22,704 | 11,088 | 5,676 | 2,772 | 24,940 | 12,180 | 688 | 336 | 
| halo2_keygen | 52 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| halo2_keygen | 71 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| halo2_keygen | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| halo2_keygen | 74 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| root_keygen | 0 | ProgramAir | 0 | 1 |  | 25 |  | 40 |  | 10 |  | 37 |  | 1 |  | 
| root_keygen | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| root_keygen | 19 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| root_keygen | 2 | PersistentBoundaryAir<8> | 0 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| root_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| root_keygen | 22 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| root_keygen | 3 | MemoryMerkleAir<8> | 0 | 172 | 84 | 28,380 | 13,860 | 22,704 | 11,088 | 5,676 | 2,772 | 24,940 | 12,180 | 688 | 336 | 
| root_keygen | 7 | RangeTupleCheckerAir<2> | 0 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 17 | 193 | 17 | 5 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 10 | 119 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 106 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 107 | 463 | 107 | 18 | 7 | 22 | 14 | 14 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,981 | 175 | 46 | 0 | 0 | 78 | 27 | 27 | 36 | 13 | 0 | 50 | 39 | 10 | 2 | 7 | 46 | 46 | 78 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 108 | 20 | 0 | 0 | 55 | 20 | 20 | 23 | 11 | 0 | 32 | 24 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 97 | 14 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 28 | 20 | 7 | 1 | 5 | 15 | 14 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 170,433,085 | 355 | 99 | 0 | 0 | 166 | 67 | 66 | 39 | 58 | 0 | 88 | 71 | 17 | 7 | 9 | 100 | 99 | 166 | 0 | 3 | 58 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 37,912,451 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | halo2_section_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | prover |  |  | 6 |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  | 6 |  |  |  |  | 
| halo2_keygen | ifft_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 112 |  |  |  | 
| halo2_keygen | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_keygen | lagrange_to_coeff_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 112 |  |  |  | 
| halo2_keygen | multiexp_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 41 |  |  |  | 
| halo2_outer | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 47 |  |  |  | 
| halo2_outer | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5,836 |  |  |  | 
| halo2_outer | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 43 |  |  |  | 
| halo2_outer | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 
| halo2_outer | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 36 |  |  |  | 
| halo2_outer | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,676 |  |  |  | 
| halo2_outer | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 11 |  |  |  | 
| halo2_outer | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 282 |  |  |  | 
| halo2_outer | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 138 |  |  |  | 
| halo2_outer | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 99 |  |  |  | 
| halo2_outer | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 155 |  |  |  | 
| halo2_outer | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 324 |  |  |  | 
| halo2_outer | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97 |  |  |  | 
| halo2_outer | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 593 |  |  |  | 
| halo2_outer | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 65 |  |  |  | 
| halo2_outer | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |  | 
| halo2_outer | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 758 |  |  |  | 
| halo2_outer | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_outer | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,806 |  |  |  | 
| halo2_outer | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 234 |  |  |  | 
| halo2_outer | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 914 |  |  |  | 
| halo2_outer | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 758 |  |  |  | 
| halo2_outer | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 155 |  |  |  | 
| halo2_outer | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,039 |  |  |  | 
| halo2_outer | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 340 |  |  |  | 
| halo2_outer | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 479 |  |  |  | 
| halo2_outer | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 51 |  |  |  | 
| halo2_outer | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 281 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 28 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 337 |  |  |  | 
| halo2_outer | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97 |  |  |  | 
| halo2_outer | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 41 |  |  |  | 
| halo2_outer | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 20 |  |  |  | 
| halo2_outer | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 340 |  |  |  | 
| halo2_outer | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 92 |  |  |  | 
| halo2_outer | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 877 |  |  |  | 
| halo2_wrapper | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 16 |  |  |  | 
| halo2_wrapper | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,543 |  |  |  | 
| halo2_wrapper | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 27 |  |  |  | 
| halo2_wrapper | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_wrapper | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 418 |  |  |  | 
| halo2_wrapper | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 
| halo2_wrapper | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 105 |  |  |  | 
| halo2_wrapper | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 61 |  |  |  | 
| halo2_wrapper | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_wrapper | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 49 |  |  |  | 
| halo2_wrapper | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 106 |  |  |  | 
| halo2_wrapper | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 75 |  |  |  | 
| halo2_wrapper | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 90 |  |  |  | 
| halo2_wrapper | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_wrapper | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 152 |  |  |  | 
| halo2_wrapper | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_wrapper | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 266 |  |  |  | 
| halo2_wrapper | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 86 |  |  |  | 
| halo2_wrapper | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 202 |  |  |  | 
| halo2_wrapper | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 152 |  |  |  | 
| halo2_wrapper | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 49 |  |  |  | 
| halo2_wrapper | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 534 |  |  |  | 
| halo2_wrapper | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 215 |  |  |  | 
| halo2_wrapper | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 231 |  |  |  | 
| halo2_wrapper | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 20 |  |  |  | 
| halo2_wrapper | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 105 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 175 |  |  |  | 
| halo2_wrapper | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 75 |  |  |  | 
| halo2_wrapper | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |  | 
| halo2_wrapper | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 214 |  |  |  | 
| halo2_wrapper | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_wrapper | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 133 |  |  |  | 
| root | prover | 18,533,671 | 1,337 | 713 | 0 | 1 | 96 | 21 | 21 | 32 | 42 | 0 | 526 | 517 | 8 | 1 | 7 | 714 | 713 | 96 | 0 | 133 |  | 12 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,121,338,980 | 2,008 | 554 | 0 | 70 | 975 | 654 | 653 | 200 | 119 | 0 | 477 | 385 | 91 | 48 | 42 | 555 | 554 | 975 | 0 | 1 | 118 | 0 | 0 | 
| halo2_keygen | prover | 0 | 8,531,435 | 60 | 7 | 0 | 0 | 28 | 8 | 7 | 5 | 14 | 0 | 24 | 21 | 3 | 0 | 2 | 7 | 7 | 28 | 0 | 1 | 13 | 0 | 0 | 
| root_keygen | prover | 0 | 3,812,843 | 143 | 46 | 0 | 0 | 20 | 5 | 5 | 4 | 9 | 0 | 77 | 73 | 3 | 0 | 2 | 46 | 46 | 20 | 0 | 6 | 9 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 104,976,098 | 2,013,265,921 | 
| halo2_keygen | prover | 0 | 0 | 2,490,671 | 2,013,265,921 | 
| root_keygen | prover | 0 | 0 | 917,807 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,535 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 170 | 75 | 2,705 | 75 | 170 | 10,592,943,552 | 2,968,407,792 | 13,561,351,344 | 2,097,152 | 0 | 32 | 451 | 2,579,903 | 34.05 | 
| halo2_keygen | 0 | 115 | 108 | 289 | 108 | 115 | 126,900,030 | 285,353 | 127,185,383 | 2,097,152 | 0 | 1 | 3 | 1 | 9,223,372,036,854,775,807 | 
| root_keygen | 0 | 130 | 114 | 389 | 114 | 130 | 51,009,342 | 285,353 | 51,294,695 | 2,097,152 | 0 | 3 | 0 | 1 | 9,223,372,036,854,775,807 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ed0dc99936f9c138f1bd32aa2972ff68e35a0aad

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29451490250)
