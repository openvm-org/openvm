| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  234.79 |  233.19 |  233.19 |
| app_proof |  1.78 |  0.40 |  0.40 |
| leaf |  0.49 |  0.28 |  0.28 |
| internal_for_leaf |  0.21 |  0.21 |  0.21 |
| internal_recursive.0 |  0.10 |  0.10 |  0.10 |
| internal_recursive.1 |  0.09 |  0.09 |  0.09 |
| root |  1.48 |  1.48 |  1.48 |
| halo2_outer |  178.78 |  178.78 |  178.78 |
| halo2_wrapper |  51.87 |  51.87 |  51.87 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  244.14 |  1,709 |  328 |  230 |
| `execute_metered_time_ms` |  67 | -          | -          | -          |
| `execute_metered_insns` |  12,000,265 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  176.88 | -          |  176.88 |  176.88 |
| `execute_preflight_insns` |  1,714,323.57 |  12,000,265 |  1,747,000 |  1,518,265 |
| `execute_preflight_time_ms` |  50.86 |  356 |  52 |  45 |
| `execute_preflight_insn_mi/s` |  46.91 | -          |  47.01 |  46.80 |
| `trace_gen_time_ms   ` |  24 |  168 |  29 |  23 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  114.29 |  800 |  117 |  113 |
| `prover.main_trace_commit_time_ms` |  20 |  140 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  67 |  469 |  69 |  66 |
| `prover.openings_time_ms` |  26.14 |  183 |  27 |  26 |
| `prover.rap_constraints.logup_gkr_time_ms` |  47 |  329 |  47 |  47 |
| `prover.rap_constraints.round0_time_ms` |  10.43 |  73 |  12 |  10 |
| `prover.rap_constraints.mle_rounds_time_ms` |  9 |  63 |  9 |  9 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  35 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  14 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  3 |  21 |  3 |  3 |
| `prover.openings.whir_time_ms` |  20.86 |  146 |  21 |  20 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  247 |  494 |  279 |  215 |
| `execute_preflight_time_ms` |  1 |  2 |  1 |  1 |
| `trace_gen_time_ms   ` |  38.50 |  77 |  44 |  33 |
| `generate_blob_total_time_ms` |  2.50 |  5 |  3 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  208 |  416 |  235 |  181 |
| `prover.main_trace_commit_time_ms` |  63 |  126 |  75 |  51 |
| `prover.rap_constraints_time_ms` |  81 |  162 |  87 |  75 |
| `prover.openings_time_ms` |  63.50 |  127 |  73 |  54 |
| `prover.rap_constraints.logup_gkr_time_ms` |  19 |  38 |  21 |  17 |
| `prover.rap_constraints.round0_time_ms` |  34 |  68 |  36 |  32 |
| `prover.rap_constraints.mle_rounds_time_ms` |  26.50 |  53 |  28 |  25 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  20 |  11 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3.50 |  7 |  4 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  12 |  6 |  6 |
| `prover.openings.whir_time_ms` |  52.50 |  105 |  61 |  44 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  206 |  206 |  206 |  206 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  24 |  24 |  24 |  24 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  182 |  182 |  182 |  182 |
| `prover.main_trace_commit_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints_time_ms` |  75 |  75 |  75 |  75 |
| `prover.openings_time_ms` |  54 |  54 |  54 |  54 |
| `prover.rap_constraints.logup_gkr_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  45 |  45 |  45 |  45 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  101 |  101 |  101 |  101 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  90 |  90 |  90 |  90 |
| `prover.main_trace_commit_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints_time_ms` |  48 |  48 |  48 |  48 |
| `prover.openings_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  88 |  88 |  88 |  88 |
| `execute_preflight_time_ms` |  1 |  1 |  1 |  1 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  79 |  79 |  79 |  79 |
| `prover.main_trace_commit_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints_time_ms` |  46 |  46 |  46 |  46 |
| `prover.openings_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.whir_time_ms` |  14 |  14 |  14 |  14 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,478 |  1,478 |  1,478 |  1,478 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,358 |  1,358 |  1,358 |  1,358 |
| `prover.main_trace_commit_time_ms` |  402 |  402 |  402 |  402 |
| `prover.rap_constraints_time_ms` |  91 |  91 |  91 |  91 |
| `prover.openings_time_ms` |  864 |  864 |  864 |  864 |
| `prover.rap_constraints.logup_gkr_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.round0_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.mle_rounds_time_ms` |  30 |  30 |  30 |  30 |
| `prover.openings.stacked_reduction_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  857 |  857 |  857 |  857 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  178,783 |  178,783 |  178,783 |  178,783 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  51,865 |  51,865 |  51,865 |  51,865 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/70cccbf6f335c093b5f3fd272462b1b8ed5ef291/fibonacci_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 4.74 | app_proof.6 |
| set initial memory | 4.74 | app_proof.1 |
| prover.stacked_commit | 3.03 | leaf.0.prover |
| prover.merkle_tree | 2.60 | leaf.0.prover |
| prover.openings | 2.60 | leaf.0.prover |
| prover.prove_whir_opening | 2.60 | leaf.0.prover |
| prover.rs_code_matrix | 2.60 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 1.70 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 1.70 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 1.61 | app_proof.prover.0 |
| prover.gkr_input_evals | 1.61 | app_proof.prover.0 |
| prover.rap_constraints | 1.52 | internal_for_leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.31 | leaf.0.prover |
| prover.batch_constraints.round0 | 1.31 | leaf.0.prover |
| tracegen.whir_final_poly_query_eval | 0.88 | leaf.0 |
| tracegen.pow_checker | 0.88 | leaf.0 |
| tracegen.exp_bits_len | 0.88 | leaf.0 |
| prover.before_gkr_input_evals | 0.82 | leaf.0.prover |
| tracegen.whir_folding | 0.63 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.62 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.62 | leaf.0 |
| tracegen.proof_shape | 0.58 | leaf.0 |
| tracegen.public_values | 0.58 | leaf.0 |
| tracegen.range_checker | 0.58 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | merkle_tree_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 55 | 9 | 3 | 0 | 3 | 1 | 0 | 2 | 0 | 0 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 0 | RootVerifierPvsAir |  | 108 | 36 | 2 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 19 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 85 | 4 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PhantomAir |  | 3 | 1 | 2 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 35 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 81 |  |  |  | 3 |  |  | 0 |  |  | 339 | 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  |  |  | 231 |  |  |  |  |  |  |  |  | 67 | 12,000,265 | 176.88 | 0 |  |  | 1,800 |  | 
| halo2_keygen |  |  |  |  |  |  |  |  |  |  |  |  | 100,758 |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_outer |  |  | 178,783 |  |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 51,865 |  |  |  |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  |  | 206 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 206 | 
| internal_recursive.0 |  |  |  |  |  |  | 101 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 101 | 
| internal_recursive.1 |  |  |  |  |  |  | 88 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 88 | 
| leaf |  |  |  |  |  | 215 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 495 | 
| root | 118 | 9 | 1,478 | 8 |  |  |  |  | 1,478 |  |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 | 0 |  | 1,478 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 0 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 69 | 213 | 4 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 87 | 4 | 
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
| agg_keygen | 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 45 | 5,760 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 256 | 8 | 2,048 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 52 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 2,944 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 46,080 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 20 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 1,728 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 3,648 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 143,360 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 1,572,864 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 816 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 8,192 | 16 | 131,072 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 368 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 1,216 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 32,768 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 32,768 | 89 | 2,916,352 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 229,376 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 507,904 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 69,632 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 23,592,960 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 819,200 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 69 | 276 | 
| leaf | 0 | VerifierPvsAir | 1 | prover | 4 | 69 | 276 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 1 | VmPvsAir | 1 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 10 | EqSharpUniReceiverAir | 1 | prover | 64 | 17 | 1,088 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 11 | EqUniAir | 1 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 4,096 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 37 | 151,552 | 
| leaf | 13 | InteractionsFoldingAir | 1 | prover | 4,096 | 37 | 151,552 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 14 | ConstraintsFoldingAir | 1 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 15 | EqNegAir | 1 | prover | 64 | 40 | 2,560 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 16 | TranscriptAir | 1 | prover | 8,192 | 44 | 360,448 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 131,072 | 301 | 39,452,672 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 18 | MerkleVerifyAir | 1 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 43 | 5,504 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 128 | 43 | 5,504 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 20 | PublicValuesAir | 1 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 104 | 
| leaf | 22 | GkrInputAir | 1 | prover | 4 | 26 | 104 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 23 | GkrLayerAir | 1 | prover | 128 | 46 | 5,888 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 92,160 | 
| leaf | 24 | GkrLayerSumcheckAir | 1 | prover | 1,024 | 45 | 46,080 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 40 | 
| leaf | 25 | GkrXiSamplerAir | 1 | prover | 4 | 10 | 40 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 129,024 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 3,456 | 
| leaf | 27 | UnivariateRoundAir | 1 | prover | 128 | 27 | 3,456 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 7,296 | 
| leaf | 28 | SumcheckRoundsAir | 1 | prover | 64 | 57 | 3,648 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 286,720 | 
| leaf | 29 | StackingClaimsAir | 1 | prover | 8,192 | 35 | 286,720 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 8,192 | 60 | 491,520 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 8,192 | 60 | 491,520 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 30 | EqBaseAir | 1 | prover | 16 | 51 | 816 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 31 | EqBitsAir | 1 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 736 | 
| leaf | 32 | WhirRoundAir | 1 | prover | 16 | 46 | 736 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 33 | SumcheckAir | 1 | prover | 64 | 38 | 2,432 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 34 | WhirQueryAir | 1 | prover | 2,048 | 32 | 65,536 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 89 | 11,665,408 | 
| leaf | 35 | InitialOpenedValuesAir | 1 | prover | 65,536 | 89 | 5,832,704 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 36 | NonInitialOpenedValuesAir | 1 | prover | 8,192 | 28 | 229,376 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 37 | WhirFoldingAir | 1 | prover | 16,384 | 31 | 507,904 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 38 | FinalPolyMleEvalAir | 1 | prover | 4,096 | 34 | 139,264 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 39 | FinalPolyQueryEvalAir | 1 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 41 | ExpBitsLenAir | 1 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 256 | 24 | 6,144 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 7 | EqNsAir | 1 | prover | 128 | 41 | 5,248 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 8 | Eq3bAir | 1 | prover | 8,192 | 25 | 204,800 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 0 | 9 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 0 | 116,458 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 0 | 14 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 0 | 2 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 0 | 4 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 116,461 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 116,463 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 0 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 0 | 19 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 0 | 4 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 0 | 349,374 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 1,048,156 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 0 | 2 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 0 | 3 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 5 | 
| app_proof | 20 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_BUFFER | 0 | 2 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_STOREW | 0 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 0 | 8 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 1 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 116,467 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 1 | 349,400 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 1 | 1,048,199 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 2 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 2 | 116,466 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 2 | 116,467 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 2 | 349,401 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 2 | 1,048,199 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 3 | 116,466 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 3 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 3 | 116,466 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 3 | 349,399 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 3 | 1,048,202 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 4 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 4 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 4 | 116,467 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 4 | 349,400 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 4 | 1,048,199 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 5 | 116,467 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 5 | 116,466 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 5 | 116,467 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 5 | 349,401 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 5 | 1,048,199 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 6 | 10 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 6 | 101,210 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 6 | 6 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 6 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 6 | 101,215 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 6 | 101,212 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 6 | 7 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 6 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 6 | 10 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 6 | 21 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 6 | 303,627 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 6 | 910,925 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 6 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 6 | 4 | 

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
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 8 | 32 | 256 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 64 | 41 | 2,624 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 4 | 53 | 212 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 64 | 21 | 1,344 | 
| app_proof | 20 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 256 | 32 | 8,192 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 1,572,864 | 
| app_proof | 8 | Rv32HintStoreAir | prover | 0 | 4 | 32 | 128 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 8 | 20 | 160 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 131,072 | 18 | 2,359,296 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 262,144 | 26 | 6,815,744 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 2 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 2 | 131,072 | 18 | 2,359,296 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 2 | 262,144 | 26 | 6,815,744 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 2 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 2 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 3 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 3 | 131,072 | 18 | 2,359,296 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 3 | 262,144 | 26 | 6,815,744 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 3 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 3 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 4 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 4 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 4 | 131,072 | 18 | 2,359,296 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 4 | 262,144 | 26 | 6,815,744 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 4 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 4 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 4 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 4 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 4 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 4 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 4 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 5 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 5 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 5 | 131,072 | 18 | 2,359,296 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 5 | 262,144 | 26 | 6,815,744 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 5 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 5 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 5 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 5 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 5 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 5 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 5 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 6 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 6 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 6 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 6 | 131,072 | 18 | 2,359,296 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 6 | 2 | 32 | 64 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 6 | 262,144 | 26 | 6,815,744 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 6 | 64 | 41 | 2,624 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 6 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 6 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 6 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 6 | 64 | 21 | 1,344 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 6 | 256 | 300 | 76,800 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 6 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 6 | 256 | 32 | 8,192 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 6 | 524,288 | 3 | 1,572,864 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 6 | 4 | 20 | 80 | 

| group | air_id | air_name | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- | --- |
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 0 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 1 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 2 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 3 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 4 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 5 | 1 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 9 | 7 | 630 | 490 | 1,008 | 784 | 252 | 196 | 5,220 | 4,060 | 144 | 112 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 116,472 | 14,600 | 5,241,240 | 657,000 | 8,385,984 | 1,051,200 | 2,096,496 | 262,800 | 42,221,100 | 5,292,500 | 1,164,720 | 146,000 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 6 | 2 | 480 | 160 | 768 | 256 | 192 | 64 | 2,828 | 942 | 78 | 26 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 232,924 | 29,220 | 15,140,060 | 1,899,300 | 24,224,096 | 3,038,880 | 6,056,024 | 759,720 | 92,878,445 | 11,651,475 | 2,562,164 | 321,420 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 33 | 31 | 3,383 | 3,177 | 5,412 | 5,084 | 1,353 | 1,271 | 20,337 | 19,103 | 561 | 527 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 4 |  | 530 |  | 848 |  | 212 |  | 3,480 |  | 96 |  | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 349,374 | 174,914 | 32,317,095 | 16,179,545 | 51,707,352 | 25,887,272 | 12,926,838 | 6,471,818 | 227,966,535 | 114,131,385 | 6,288,732 | 3,148,452 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,166 | 410 | 94,334,940 | 36,900 | 150,935,904 | 59,040 | 37,733,976 | 14,760 | 759,920,350 | 297,250 | 20,963,320 | 8,200 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 512 |  | 26,880 |  | 43,008 |  | 10,752 |  | 74,240 |  | 2,048 |  | 
| app_proof | 20 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 1,200 | 848 | 900,000 | 636,000 | 1,440,000 | 1,017,600 | 360,000 | 254,400 | 43,500 | 30,740 | 1,200 | 848 | 
| app_proof | 22 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 688 | 336 | 110,080 | 53,760 | 88,064 | 43,008 | 22,016 | 10,752 | 99,760 | 48,720 | 2,752 | 1,344 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 0 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 8 | Rv32HintStoreAir | 0 | 3 | 1 | 480 | 160 | 384 | 128 | 96 | 32 | 1,958 | 652 | 54 | 18 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 400 |  | 640 |  | 160 |  | 3,480 |  | 96 |  | 
| app_proof | 0 | ProgramAir | 1 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 116,467 | 14,605 | 5,241,015 | 657,225 | 8,385,624 | 1,051,560 | 2,096,406 | 262,890 | 42,219,288 | 5,294,312 | 1,164,670 | 146,050 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 232,934 | 29,210 | 15,140,710 | 1,898,650 | 24,225,136 | 3,037,840 | 6,056,284 | 759,460 | 92,882,433 | 11,647,487 | 2,562,274 | 321,310 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 349,400 | 174,888 | 32,319,500 | 16,177,140 | 51,711,200 | 25,883,424 | 12,927,800 | 6,470,856 | 227,983,500 | 114,114,420 | 6,289,200 | 3,147,984 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,199 | 377 | 94,337,910 | 33,930 | 150,940,656 | 54,288 | 37,735,164 | 13,572 | 759,944,275 | 273,325 | 20,963,980 | 7,540 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 1 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 2 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 2 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 116,467 | 14,605 | 5,241,015 | 657,225 | 8,385,624 | 1,051,560 | 2,096,406 | 262,890 | 42,219,288 | 5,294,312 | 1,164,670 | 146,050 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 232,933 | 29,211 | 15,140,645 | 1,898,715 | 24,225,032 | 3,037,944 | 6,056,258 | 759,486 | 92,882,034 | 11,647,886 | 2,562,263 | 321,321 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 349,401 | 174,887 | 32,319,593 | 16,177,047 | 51,711,348 | 25,883,276 | 12,927,837 | 6,470,819 | 227,984,153 | 114,113,767 | 6,289,218 | 3,147,966 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,199 | 377 | 94,337,910 | 33,930 | 150,940,656 | 54,288 | 37,735,164 | 13,572 | 759,944,275 | 273,325 | 20,963,980 | 7,540 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 2 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 2 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 2 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 2 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 2 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 3 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 3 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 116,466 | 14,606 | 5,240,970 | 657,270 | 8,385,552 | 1,051,632 | 2,096,388 | 262,908 | 42,218,925 | 5,294,675 | 1,164,660 | 146,060 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 232,933 | 29,211 | 15,140,645 | 1,898,715 | 24,225,032 | 3,037,944 | 6,056,258 | 759,486 | 92,882,034 | 11,647,886 | 2,562,263 | 321,321 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 349,399 | 174,889 | 32,319,408 | 16,177,232 | 51,711,052 | 25,883,572 | 12,927,763 | 6,470,893 | 227,982,848 | 114,115,072 | 6,289,182 | 3,148,002 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,202 | 374 | 94,338,180 | 33,660 | 150,941,088 | 53,856 | 37,735,272 | 13,464 | 759,946,450 | 271,150 | 20,964,040 | 7,480 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 3 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 3 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 3 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 3 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 3 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 4 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 4 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 116,467 | 14,605 | 5,241,015 | 657,225 | 8,385,624 | 1,051,560 | 2,096,406 | 262,890 | 42,219,288 | 5,294,312 | 1,164,670 | 146,050 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 232,934 | 29,210 | 15,140,710 | 1,898,650 | 24,225,136 | 3,037,840 | 6,056,284 | 759,460 | 92,882,433 | 11,647,487 | 2,562,274 | 321,310 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 349,400 | 174,888 | 32,319,500 | 16,177,140 | 51,711,200 | 25,883,424 | 12,927,800 | 6,470,856 | 227,983,500 | 114,114,420 | 6,289,200 | 3,147,984 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,199 | 377 | 94,337,910 | 33,930 | 150,940,656 | 54,288 | 37,735,164 | 13,572 | 759,944,275 | 273,325 | 20,963,980 | 7,540 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 4 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 4 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 4 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 5 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 5 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 116,467 | 14,605 | 5,241,015 | 657,225 | 8,385,624 | 1,051,560 | 2,096,406 | 262,890 | 42,219,288 | 5,294,312 | 1,164,670 | 146,050 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 232,933 | 29,211 | 15,140,645 | 1,898,715 | 24,225,032 | 3,037,944 | 6,056,258 | 759,486 | 92,882,034 | 11,647,886 | 2,562,263 | 321,321 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 349,401 | 174,887 | 32,319,593 | 16,177,047 | 51,711,348 | 25,883,276 | 12,927,837 | 6,470,819 | 227,984,153 | 114,113,767 | 6,289,218 | 3,147,966 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,199 | 377 | 94,337,910 | 33,930 | 150,940,656 | 54,288 | 37,735,164 | 13,572 | 759,944,275 | 273,325 | 20,963,980 | 7,540 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 5 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 5 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 5 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 6 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 6 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 10 | 6 | 700 | 420 | 1,120 | 672 | 280 | 168 | 5,800 | 3,480 | 160 | 96 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 101,216 | 29,856 | 4,554,720 | 1,343,520 | 7,287,552 | 2,149,632 | 1,821,888 | 537,408 | 36,690,800 | 10,822,800 | 1,012,160 | 298,560 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 2 |  | 160 |  | 256 |  | 64 |  | 943 |  | 26 |  | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 202,427 | 59,717 | 13,157,755 | 3,881,605 | 21,052,408 | 6,210,568 | 5,263,102 | 1,552,642 | 80,717,767 | 23,812,153 | 2,226,697 | 656,887 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 52 | 12 | 5,330 | 1,230 | 8,528 | 1,968 | 2,132 | 492 | 32,045 | 7,395 | 884 | 204 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 303,627 | 220,661 | 28,085,498 | 20,411,142 | 44,936,796 | 32,657,828 | 11,234,199 | 8,164,457 | 198,116,618 | 143,981,302 | 5,465,286 | 3,971,898 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 910,926 | 137,650 | 81,983,340 | 12,388,500 | 131,173,344 | 19,821,600 | 32,793,336 | 4,955,400 | 660,421,350 | 99,796,250 | 18,218,520 | 2,753,000 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 6 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 6 | 512 |  | 26,880 |  | 43,008 |  | 10,752 |  | 74,240 |  | 2,048 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 1,200 | 848 | 900,000 | 636,000 | 1,440,000 | 1,017,600 | 360,000 | 254,400 | 43,500 | 30,740 | 1,200 | 848 | 
| app_proof | 22 | VariableRangeCheckerAir | 6 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 6 | 688 | 336 | 110,080 | 53,760 | 88,064 | 43,008 | 22,016 | 10,752 | 99,760 | 48,720 | 2,752 | 1,344 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 6 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 4 |  | 200 |  | 320 |  | 80 |  | 1,740 |  | 48 |  | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 24 | 206 | 23 | 9 | 1 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 11 | 101 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 88 | 8 | 1 | 0 | 1 | 0 | 0 | 
| leaf | 0 | 44 | 279 | 44 | 23 | 3 | 1 | 1 | 1 | 
| leaf | 1 | 33 | 215 | 32 | 17 | 2 | 1 | 1 | 1 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 53,005,300 | 182 | 51 | 0 | 0 | 75 | 23 | 22 | 36 | 15 | 0 | 54 | 45 | 9 | 2 | 7 | 51 | 75 | 0 | 1 | 15 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,767 | 90 | 16 | 0 | 0 | 48 | 17 | 17 | 20 | 10 | 0 | 25 | 19 | 5 | 1 | 4 | 16 | 48 | 0 | 1 | 9 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,863 | 79 | 11 | 0 | 0 | 46 | 17 | 16 | 18 | 10 | 0 | 20 | 14 | 5 | 0 | 4 | 11 | 46 | 0 | 1 | 9 | 0 | 0 | 
| leaf | 0 | prover | 147,934,246 | 235 | 74 | 0 | 0 | 87 | 36 | 35 | 28 | 21 | 0 | 73 | 61 | 11 | 4 | 6 | 75 | 87 | 0 | 2 | 20 | 0 | 0 | 
| leaf | 1 | prover | 98,728,630 | 181 | 51 | 0 | 0 | 75 | 32 | 31 | 25 | 17 | 0 | 54 | 44 | 9 | 3 | 6 | 51 | 75 | 0 | 2 | 16 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 5,632,323 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,382 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,358 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,806,469 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 7,964,181 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| root | prover | 18,533,670 | 1,358 | 402 | 0 | 0 | 91 | 18 | 18 | 30 | 41 | 0 | 864 | 857 | 6 | 1 | 5 | 402 | 91 | 0 | 76 | 11 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 70,295,622 | 117 | 20 | 0 | 0 | 69 | 12 | 12 | 9 | 47 | 0 | 27 | 21 | 5 | 2 | 3 | 20 | 69 | 0 | 1 | 46 | 0 | 0 | 
| app_proof | prover | 1 | 70,248,284 | 113 | 20 | 0 | 0 | 66 | 10 | 10 | 9 | 47 | 0 | 26 | 20 | 5 | 2 | 3 | 20 | 66 | 0 | 1 | 47 | 0 | 0 | 
| app_proof | prover | 2 | 70,248,284 | 114 | 20 | 0 | 0 | 67 | 10 | 10 | 9 | 47 | 0 | 26 | 21 | 5 | 2 | 3 | 20 | 67 | 0 | 1 | 46 | 0 | 0 | 
| app_proof | prover | 3 | 70,248,284 | 113 | 20 | 0 | 0 | 67 | 10 | 10 | 9 | 47 | 0 | 26 | 21 | 5 | 2 | 3 | 20 | 66 | 0 | 1 | 46 | 0 | 0 | 
| app_proof | prover | 4 | 70,248,284 | 114 | 20 | 0 | 0 | 66 | 10 | 10 | 9 | 47 | 0 | 26 | 21 | 5 | 2 | 3 | 20 | 66 | 0 | 1 | 46 | 0 | 0 | 
| app_proof | prover | 5 | 70,248,284 | 114 | 20 | 0 | 0 | 66 | 10 | 10 | 9 | 47 | 0 | 26 | 21 | 5 | 2 | 3 | 20 | 66 | 0 | 1 | 46 | 0 | 0 | 
| app_proof | prover | 6 | 70,295,004 | 115 | 20 | 0 | 0 | 68 | 11 | 11 | 9 | 47 | 0 | 26 | 21 | 5 | 2 | 3 | 20 | 68 | 0 | 1 | 46 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 35,531,965 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 4 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 5 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 6 | 0 | 35,531,668 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,534 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 24 | 328 | 24 | 134 | 1,411,127,546 | 162,713,442 | 1,573,840,988 | 2,097,152 | 0 | 1 | 52 | 1,747,000 | 46.94 | 
| app_proof | 1 | 23 | 230 | 23 | 40 | 1,409,791,968 | 161,775,035 | 1,571,567,003 | 2,097,152 | 0 | 1 | 52 | 1,747,000 | 46.89 | 
| app_proof | 2 | 23 | 230 | 23 | 40 | 1,409,792,265 | 161,774,738 | 1,571,567,003 | 2,097,152 | 0 | 1 | 52 | 1,747,000 | 46.80 | 
| app_proof | 3 | 23 | 230 | 23 | 40 | 1,409,792,837 | 161,774,166 | 1,571,567,003 | 2,097,152 | 0 | 1 | 52 | 1,747,000 | 47.01 | 
| app_proof | 4 | 23 | 230 | 23 | 40 | 1,409,791,968 | 161,775,035 | 1,571,567,003 | 2,097,152 | 0 | 1 | 52 | 1,747,000 | 46.83 | 
| app_proof | 5 | 23 | 230 | 23 | 40 | 1,409,792,265 | 161,774,738 | 1,571,567,003 | 2,097,152 | 0 | 1 | 51 | 1,747,000 | 46.100 | 
| app_proof | 6 | 29 | 231 | 29 | 40 | 1,233,288,671 | 340,539,078 | 1,573,827,749 | 2,097,152 | 0 | 1 | 45 | 1,518,265 | 46.90 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/70cccbf6f335c093b5f3fd272462b1b8ed5ef291

Max Segment Length: 1048576

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27281562352)
