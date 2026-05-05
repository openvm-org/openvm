| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  405.96 |  405.10 |  405.10 |
| app_proof |  2.48 |  1.63 |  1.63 |
| leaf |  0.65 |  0.65 |  0.65 |
| internal_for_leaf |  0.25 |  0.25 |  0.25 |
| internal_recursive.0 |  0.17 |  0.17 |  0.17 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |
| root |  106.03 |  106.03 |  106.03 |
| halo2_outer |  192.94 |  192.94 |  192.94 |
| halo2_wrapper |  103.28 |  103.28 |  103.28 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,223.50 |  2,447 |  1,590 |  857 |
| `execute_metered_time_ms` |  35 | -          | -          | -          |
| `execute_metered_insns` |  2,579,903 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  72.42 | -          |  72.42 |  72.42 |
| `execute_preflight_insns` |  1,289,951.50 |  2,579,903 |  1,540,000 |  1,039,903 |
| `execute_preflight_time_ms` |  138.50 |  277 |  164 |  113 |
| `execute_preflight_insn_mi/s` |  36.89 | -          |  37.30 |  36.47 |
| `trace_gen_time_ms   ` |  49 |  98 |  82 |  16 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  896 |  1,792 |  1,106 |  686 |
| `prover.main_trace_commit_time_ms` |  198.50 |  397 |  249 |  148 |
| `prover.rap_constraints_time_ms` |  616 |  1,232 |  760 |  472 |
| `prover.openings_time_ms` |  80.50 |  161 |  96 |  65 |
| `prover.rap_constraints.logup_gkr_time_ms` |  193.50 |  387 |  212 |  175 |
| `prover.rap_constraints.round0_time_ms` |  265.50 |  531 |  356 |  175 |
| `prover.rap_constraints.mle_rounds_time_ms` |  156 |  312 |  191 |  121 |
| `prover.openings.stacked_reduction_time_ms` |  66.50 |  133 |  83 |  50 |
| `prover.openings.stacked_reduction.round0_time_ms` |  25.50 |  51 |  27 |  24 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  41 |  82 |  56 |  26 |
| `prover.openings.whir_time_ms` |  13.50 |  27 |  14 |  13 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  649 |  649 |  649 |  649 |
| `execute_preflight_time_ms` |  25 |  25 |  25 |  25 |
| `trace_gen_time_ms   ` |  134 |  134 |  134 |  134 |
| `generate_blob_total_time_ms` |  16 |  16 |  16 |  16 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  514 |  514 |  514 |  514 |
| `prover.main_trace_commit_time_ms` |  259 |  259 |  259 |  259 |
| `prover.rap_constraints_time_ms` |  200 |  200 |  200 |  200 |
| `prover.openings_time_ms` |  54 |  54 |  54 |  54 |
| `prover.rap_constraints.logup_gkr_time_ms` |  82 |  82 |  82 |  82 |
| `prover.rap_constraints.round0_time_ms` |  60 |  60 |  60 |  60 |
| `prover.rap_constraints.mle_rounds_time_ms` |  58 |  58 |  58 |  58 |
| `prover.openings.stacked_reduction_time_ms` |  40 |  40 |  40 |  40 |
| `prover.openings.stacked_reduction.round0_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.whir_time_ms` |  13 |  13 |  13 |  13 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  247 |  247 |  247 |  247 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  25 |  25 |  25 |  25 |
| `generate_blob_total_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  221 |  221 |  221 |  221 |
| `prover.main_trace_commit_time_ms` |  70 |  70 |  70 |  70 |
| `prover.rap_constraints_time_ms` |  100 |  100 |  100 |  100 |
| `prover.openings_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.logup_gkr_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.round0_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.mle_rounds_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings.stacked_reduction_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  165 |  165 |  165 |  165 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  151 |  151 |  151 |  151 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  72 |  72 |  72 |  72 |
| `prover.openings_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  32 |  32 |  32 |  32 |
| `prover.openings.stacked_reduction_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  153 |  153 |  153 |  153 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  140 |  140 |  140 |  140 |
| `prover.main_trace_commit_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints_time_ms` |  70 |  70 |  70 |  70 |
| `prover.openings_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  30 |  30 |  30 |  30 |
| `prover.openings.stacked_reduction_time_ms` |  25 |  25 |  25 |  25 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  106,033 |  106,033 |  106,033 |  106,033 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  105,910 |  105,910 |  105,910 |  105,910 |
| `prover.main_trace_commit_time_ms` |  504 |  504 |  504 |  504 |
| `prover.rap_constraints_time_ms` |  12,472 |  12,472 |  12,472 |  12,472 |
| `prover.openings_time_ms` |  92,932 |  92,932 |  92,932 |  92,932 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12,403 |  12,403 |  12,403 |  12,403 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  46 |  46 |  46 |  46 |
| `prover.openings.stacked_reduction_time_ms` |  25 |  25 |  25 |  25 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.whir_time_ms` |  92,906 |  92,906 |  92,906 |  92,906 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  192,945 |  192,945 |  192,945 |  192,945 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  103,285 |  103,285 |  103,285 |  103,285 |
| `halo2_wrapper_k     ` |  23 |  23 |  23 |  23 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4/kitchen_sink_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 11.34 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 10.15 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 10.15 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 10.12 | app_proof.prover.0 |
| prover.gkr_input_evals | 10.12 | app_proof.prover.0 |
| prover.merkle_tree | 9.70 | app_proof.prover.0 |
| prover.openings | 9.70 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.70 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 8.94 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 8.94 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 7.48 | app_proof.prover.0 |
| prover.stacked_commit | 7.48 | app_proof.prover.0 |
| prover.rs_code_matrix | 7.47 | app_proof.prover.0 |
| generate mem proving ctxs | 5.61 | app_proof.0 |
| set initial memory | 5.61 | app_proof.1 |
| tracegen.exp_bits_len | 2.33 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.33 | leaf.0 |
| tracegen.pow_checker | 2.33 | leaf.0 |
| tracegen.whir_folding | 2.16 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 2.15 | leaf.0 |
| tracegen.whir_initial_opened_values | 2.15 | leaf.0 |
| tracegen.proof_shape | 1.96 | leaf.0 |
| tracegen.public_values | 1.96 | leaf.0 |
| tracegen.range_checker | 1.96 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | merkle_tree_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 705 | 12 | 3 | 0 | 3 | 3 | 0 | 3 | 0 | 0 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| ConstraintsFoldingAir | 11 | 47 | 4 | 
| Eq3bAir | 3 | 68 | 4 | 
| EqBaseAir | 8 | 106 | 4 | 
| EqBitsAir | 5 | 26 | 4 | 
| EqNegAir | 8 | 95 | 4 | 
| EqNsAir | 10 | 74 | 4 | 
| EqSharpUniAir | 5 | 50 | 4 | 
| EqSharpUniReceiverAir | 3 | 28 | 4 | 
| EqUniAir | 3 | 35 | 4 | 
| ExpBitsLenAir | 2 | 44 | 3 | 
| ExpressionClaimAir | 7 | 75 | 4 | 
| FinalPolyMleEvalAir | 14 | 21 | 4 | 
| FinalPolyQueryEvalAir | 5 | 137 | 4 | 
| FractionsFolderAir | 20 | 48 | 4 | 
| GkrInputAir | 21 | 22 | 4 | 
| GkrLayerAir | 36 | 43 | 4 | 
| GkrLayerSumcheckAir | 25 | 66 | 4 | 
| GkrXiSamplerAir | 8 | 17 | 4 | 
| InitialOpenedValuesAir | 13 | 156 | 4 | 
| InteractionsFoldingAir | 14 | 106 | 4 | 
| KeccakfOpAir | 310 | 52 | 2 | 
| KeccakfPermAir | 2 | 3,183 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| MerkleVerifyAir | 6 | 22 | 3 | 
| MultilinearSumcheckAir | 16 | 68 | 4 | 
| NonInitialOpenedValuesAir | 4 | 43 | 4 | 
| OpeningClaimsAir | 25 | 109 | 4 | 
| PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| ProgramAir | 1 |  | 1 | 
| ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| PublicValuesAir | 4 | 18 | 4 | 
| RangeCheckerAir<8> | 1 | 3 | 2 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| RootVerifierPvsAir | 108 | 36 | 2 | 
| Rv32HintStoreAir | 18 | 17 | 3 | 
| Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| Sha2MainAir<Sha256Config> | 148 | 39 | 3 | 
| Sha2MainAir<Sha512Config> | 276 | 71 | 3 | 
| StackingClaimsAir | 19 | 64 | 4 | 
| SumcheckAir | 22 | 52 | 4 | 
| SumcheckRoundsAir | 24 | 81 | 4 | 
| SymbolicExpressionAir<BabyBearParameters> | 13 | 321 | 4 | 
| TranscriptAir | 17 | 84 | 4 | 
| UnivariateRoundAir | 15 | 62 | 4 | 
| UnivariateSumcheckAir | 16 | 53 | 4 | 
| UserPvsCommitAir | 5 | 41 | 4 | 
| UserPvsInMemoryAir | 3 | 13 | 4 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 22 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 28 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 76 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 25 | 3 | 
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 9 | 2 | 
| VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 113 | 326 | 3 | 
| VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 81 | 222 | 3 | 
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 18 | 18 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 25 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 25 | 64 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 24 | 11 | 2 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 4 | 2 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 5 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 527 | 296 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 976 | 537 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 354 | 189 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 596 | 281 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 884 | 417 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 145 | 97 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 115 | 131 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 145 | 28 | 2 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, ShiftCoreAir<32, 8> | 163 | 2,139 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 266 | 129 | 3 | 
| VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 76 | 55 | 3 | 
| VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchLessThanCoreAir<32, 8> | 78 | 125 | 3 | 
| VmConnectorAir | 5 | 8 | 3 | 
| WhirFoldingAir | 4 | 17 | 3 | 
| WhirQueryAir | 5 | 57 | 4 | 
| WhirRoundAir | 34 | 29 | 4 | 
| XorinVmAir | 561 | 177 | 3 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 76 |  |  |  | 12 |  |  | 0 |  |  | 393 | 12 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  |  |  | 857 |  |  |  |  |  |  |  |  | 35 | 2,579,903 | 72.42 | 0 |  |  | 2,501 |  | 
| halo2_keygen |  |  |  |  |  |  |  |  |  |  |  |  | 138,476 |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_outer |  |  | 192,945 |  |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 103,285 |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  |  | 247 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 247 | 
| internal_recursive.0 |  |  |  |  |  |  | 165 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 165 | 
| internal_recursive.1 |  |  |  |  |  |  | 153 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 154 | 
| leaf |  |  |  |  |  | 649 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 649 | 
| root | 122 | 13 | 106,033 | 12 |  |  |  |  | 106,033 |  |  |  |  |  |  | 3 | 0 | 3 |  |  |  |  | 0 | 0 |  | 106,033 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 4 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 3 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 7 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 69 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 20 | 320 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 19 | 152 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 37 | 4,736 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 43 | 352,256 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 29 | 118,784 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 47 | 752 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 30 | 30 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 56 | 1,792 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 54 | 27,648 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 11 | 11 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 2,048 | 74 | 151,552 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 32 | 1,024 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 69 | 2,208 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 41 | 83,968 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 54 | 1,769,472 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 8 | 62 | 496 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 18 | 73,728 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 53 | 212 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 45 | 720 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 37 | 18,944 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 98 | 6,422,528 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 30 | 122,880 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 36 | 294,912 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 40 | 40,960 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 53 | 13,893,632 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 35 | 2,240 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 28 | 3,584 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 39 | 4,992 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 32 | 48 | 1,536 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 27 | 442,368 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 19 | 304 | 
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.0 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 20 | 80 | 
| internal_recursive.0 | 11 | EqUniAir | 1 | prover | 4 | 19 | 76 | 
| internal_recursive.0 | 12 | ExpressionClaimAir | 1 | prover | 128 | 37 | 4,736 | 
| internal_recursive.0 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 43 | 352,256 | 
| internal_recursive.0 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 29 | 118,784 | 
| internal_recursive.0 | 15 | EqNegAir | 1 | prover | 8 | 47 | 376 | 
| internal_recursive.0 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.0 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 32,768 | 301 | 9,863,168 | 
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 16,384 | 37 | 606,208 | 
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
| internal_recursive.0 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| internal_recursive.0 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 1,024 | 
| internal_recursive.0 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| internal_recursive.0 | 22 | GkrInputAir | 1 | prover | 1 | 30 | 30 | 
| internal_recursive.0 | 23 | GkrLayerAir | 1 | prover | 32 | 56 | 1,792 | 
| internal_recursive.0 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 54 | 13,824 | 
| internal_recursive.0 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 11 | 11 | 
| internal_recursive.0 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 74 | 151,552 | 
| internal_recursive.0 | 27 | UnivariateRoundAir | 1 | prover | 8 | 32 | 256 | 
| internal_recursive.0 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 69 | 2,208 | 
| internal_recursive.0 | 29 | StackingClaimsAir | 1 | prover | 256 | 41 | 10,496 | 
| internal_recursive.0 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 54 | 1,769,472 | 
| internal_recursive.0 | 30 | EqBaseAir | 1 | prover | 4 | 62 | 248 | 
| internal_recursive.0 | 31 | EqBitsAir | 1 | prover | 4,096 | 18 | 73,728 | 
| internal_recursive.0 | 32 | WhirRoundAir | 1 | prover | 4 | 53 | 212 | 
| internal_recursive.0 | 33 | SumcheckAir | 1 | prover | 16 | 45 | 720 | 
| internal_recursive.0 | 34 | WhirQueryAir | 1 | prover | 256 | 37 | 9,472 | 
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 8,192 | 98 | 802,816 | 
| internal_recursive.0 | 36 | NonInitialOpenedValuesAir | 1 | prover | 2,048 | 30 | 61,440 | 
| internal_recursive.0 | 37 | WhirFoldingAir | 1 | prover | 4,096 | 36 | 147,456 | 
| internal_recursive.0 | 38 | FinalPolyMleEvalAir | 1 | prover | 1,024 | 40 | 40,960 | 
| internal_recursive.0 | 39 | FinalPolyQueryEvalAir | 1 | prover | 131,072 | 53 | 6,946,816 | 
| internal_recursive.0 | 4 | FractionsFolderAir | 1 | prover | 64 | 35 | 2,240 | 
| internal_recursive.0 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| internal_recursive.0 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 131,072 | 
| internal_recursive.0 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 28 | 448 | 
| internal_recursive.0 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 39 | 4,992 | 
| internal_recursive.0 | 7 | EqNsAir | 1 | prover | 32 | 48 | 1,536 | 
| internal_recursive.0 | 8 | Eq3bAir | 1 | prover | 16,384 | 27 | 442,368 | 
| internal_recursive.0 | 9 | EqSharpUniAir | 1 | prover | 4 | 19 | 76 | 
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 32 | 
| internal_recursive.1 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 20 | 80 | 
| internal_recursive.1 | 11 | EqUniAir | 1 | prover | 4 | 19 | 76 | 
| internal_recursive.1 | 12 | ExpressionClaimAir | 1 | prover | 128 | 37 | 4,736 | 
| internal_recursive.1 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 43 | 352,256 | 
| internal_recursive.1 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 29 | 118,784 | 
| internal_recursive.1 | 15 | EqNegAir | 1 | prover | 8 | 47 | 376 | 
| internal_recursive.1 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 180,224 | 
| internal_recursive.1 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 16,384 | 301 | 4,931,584 | 
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 16,384 | 37 | 606,208 | 
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
| internal_recursive.1 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| internal_recursive.1 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 1,024 | 
| internal_recursive.1 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| internal_recursive.1 | 22 | GkrInputAir | 1 | prover | 1 | 30 | 30 | 
| internal_recursive.1 | 23 | GkrLayerAir | 1 | prover | 32 | 56 | 1,792 | 
| internal_recursive.1 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 54 | 13,824 | 
| internal_recursive.1 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 11 | 11 | 
| internal_recursive.1 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 74 | 151,552 | 
| internal_recursive.1 | 27 | UnivariateRoundAir | 1 | prover | 8 | 32 | 256 | 
| internal_recursive.1 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 69 | 2,208 | 
| internal_recursive.1 | 29 | StackingClaimsAir | 1 | prover | 256 | 41 | 10,496 | 
| internal_recursive.1 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 54 | 1,769,472 | 
| internal_recursive.1 | 30 | EqBaseAir | 1 | prover | 4 | 62 | 248 | 
| internal_recursive.1 | 31 | EqBitsAir | 1 | prover | 4,096 | 18 | 73,728 | 
| internal_recursive.1 | 32 | WhirRoundAir | 1 | prover | 4 | 53 | 212 | 
| internal_recursive.1 | 33 | SumcheckAir | 1 | prover | 16 | 45 | 720 | 
| internal_recursive.1 | 34 | WhirQueryAir | 1 | prover | 256 | 37 | 9,472 | 
| internal_recursive.1 | 35 | InitialOpenedValuesAir | 1 | prover | 8,192 | 98 | 802,816 | 
| internal_recursive.1 | 36 | NonInitialOpenedValuesAir | 1 | prover | 2,048 | 30 | 61,440 | 
| internal_recursive.1 | 37 | WhirFoldingAir | 1 | prover | 4,096 | 36 | 147,456 | 
| internal_recursive.1 | 38 | FinalPolyMleEvalAir | 1 | prover | 1,024 | 40 | 40,960 | 
| internal_recursive.1 | 39 | FinalPolyQueryEvalAir | 1 | prover | 131,072 | 53 | 6,946,816 | 
| internal_recursive.1 | 4 | FractionsFolderAir | 1 | prover | 64 | 35 | 2,240 | 
| internal_recursive.1 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| internal_recursive.1 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 131,072 | 
| internal_recursive.1 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 28 | 448 | 
| internal_recursive.1 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 39 | 4,992 | 
| internal_recursive.1 | 7 | EqNsAir | 1 | prover | 32 | 48 | 1,536 | 
| internal_recursive.1 | 8 | Eq3bAir | 1 | prover | 16,384 | 27 | 442,368 | 
| internal_recursive.1 | 9 | EqSharpUniAir | 1 | prover | 4 | 19 | 76 | 
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 20 | 640 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 19 | 304 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 37 | 18,944 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 131,072 | 43 | 5,636,096 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 29 | 950,272 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 47 | 1,504 | 
| leaf | 16 | TranscriptAir | 0 | prover | 65,536 | 44 | 2,883,584 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 47 | 12,032 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 30 | 60 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 56 | 3,584 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 54 | 55,296 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 11 | 22 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 74 | 2,424,832 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 32 | 2,048 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 69 | 4,416 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 41 | 167,936 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 68 | 35,651,584 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 62 | 992 | 
| leaf | 31 | EqBitsAir | 0 | prover | 65,536 | 18 | 1,179,648 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 53 | 424 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 45 | 1,440 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 37 | 37,888 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 524,288 | 98 | 51,380,224 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 30 | 245,760 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 36 | 589,824 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 40 | 81,920 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 53 | 27,787,264 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 35 | 4,480 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 28 | 7,168 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 39 | 9,984 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 48 | 3,072 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 27 | 14,155,776 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 19 | 608 | 

| group | air_id | air_name | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| root | 0 | RootVerifierPvsAir | prover | 1 | 206 | 206 | 
| root | 1 | UserPvsCommitAir | prover | 8 | 30 | 240 | 
| root | 10 | EqSharpUniReceiverAir | prover | 4 | 20 | 80 | 
| root | 11 | EqUniAir | prover | 4 | 19 | 76 | 
| root | 12 | ExpressionClaimAir | prover | 128 | 37 | 4,736 | 
| root | 13 | InteractionsFoldingAir | prover | 8,192 | 43 | 352,256 | 
| root | 14 | ConstraintsFoldingAir | prover | 4,096 | 29 | 118,784 | 
| root | 15 | EqNegAir | prover | 8 | 47 | 376 | 
| root | 16 | TranscriptAir | prover | 4,096 | 44 | 180,224 | 
| root | 17 | Poseidon2Air<BabyBearParameters>, 1> | prover | 16,384 | 301 | 4,931,584 | 
| root | 18 | MerkleVerifyAir | prover | 16,384 | 37 | 606,208 | 
| root | 19 | ProofShapeAir<4, 8> | prover | 64 | 44 | 2,816 | 
| root | 2 | UserPvsInMemoryAir | prover | 32 | 20 | 640 | 
| root | 20 | PublicValuesAir | prover | 128 | 8 | 1,024 | 
| root | 21 | RangeCheckerAir<8> | prover | 256 | 2 | 512 | 
| root | 22 | GkrInputAir | prover | 1 | 30 | 30 | 
| root | 23 | GkrLayerAir | prover | 32 | 56 | 1,792 | 
| root | 24 | GkrLayerSumcheckAir | prover | 256 | 54 | 13,824 | 
| root | 25 | GkrXiSamplerAir | prover | 1 | 11 | 11 | 
| root | 26 | OpeningClaimsAir | prover | 2,048 | 74 | 151,552 | 
| root | 27 | UnivariateRoundAir | prover | 8 | 32 | 256 | 
| root | 28 | SumcheckRoundsAir | prover | 32 | 69 | 2,208 | 
| root | 29 | StackingClaimsAir | prover | 256 | 41 | 10,496 | 
| root | 3 | SymbolicExpressionAir<BabyBearParameters> | prover | 32,768 | 318 | 10,420,224 | 
| root | 30 | EqBaseAir | prover | 4 | 62 | 248 | 
| root | 31 | EqBitsAir | prover | 4,096 | 18 | 73,728 | 
| root | 32 | WhirRoundAir | prover | 4 | 53 | 212 | 
| root | 33 | SumcheckAir | prover | 16 | 45 | 720 | 
| root | 34 | WhirQueryAir | prover | 256 | 37 | 9,472 | 
| root | 35 | InitialOpenedValuesAir | prover | 4,096 | 98 | 401,408 | 
| root | 36 | NonInitialOpenedValuesAir | prover | 2,048 | 30 | 61,440 | 
| root | 37 | WhirFoldingAir | prover | 4,096 | 36 | 147,456 | 
| root | 38 | FinalPolyMleEvalAir | prover | 1,024 | 40 | 40,960 | 
| root | 39 | FinalPolyQueryEvalAir | prover | 131,072 | 53 | 6,946,816 | 
| root | 4 | FractionsFolderAir | prover | 64 | 35 | 2,240 | 
| root | 40 | PowerCheckerAir<2, 32> | prover | 32 | 4 | 128 | 
| root | 41 | ExpBitsLenAir | prover | 8,192 | 16 | 131,072 | 
| root | 5 | UnivariateSumcheckAir | prover | 16 | 28 | 448 | 
| root | 6 | MultilinearSumcheckAir | prover | 128 | 39 | 4,992 | 
| root | 7 | EqNsAir | prover | 32 | 48 | 1,536 | 
| root | 8 | Eq3bAir | prover | 16,384 | 27 | 442,368 | 
| root | 9 | EqSharpUniAir | prover | 4 | 19 | 76 | 

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
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 32 | 262,144 | 
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
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 512 | 232 | 118,784 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 231 | 236,544 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 1,111 | 1,111 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 512 | 31 | 15,872 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 119,537,664 | 
| app_proof | 57 | KeccakfOpAir | prover | 0 | 8,192 | 561 | 4,595,712 | 
| app_proof | 58 | KeccakfPermAir | prover | 0 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 59 | XorinVmAir | prover | 0 | 8,192 | 914 | 7,487,488 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 655,360 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 65,536 | 28 | 1,835,008 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 32,768 | 18 | 589,824 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 131,072 | 26 | 3,407,872 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 32,768 | 53 | 1,736,704 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 32,768 | 37 | 1,212,416 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 524,288 | 36 | 18,874,368 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 4,096 | 21 | 86,016 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 4,096 | 32 | 131,072 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 64 | 227 | 14,528 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 1 | 64 | 166 | 10,624 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 1 | 256 | 232 | 59,392 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 512 | 231 | 118,272 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 1 | 128 | 31 | 3,968 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 1 | 16,384 | 284 | 4,653,056 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 262,144 | 456 | 119,537,664 | 
| app_proof | 57 | KeccakfOpAir | prover | 1 | 4,096 | 561 | 2,297,856 | 
| app_proof | 58 | KeccakfPermAir | prover | 1 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 59 | XorinVmAir | prover | 1 | 4,096 | 914 | 3,743,744 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 16,384 | 20 | 327,680 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 32,768 | 28 | 917,504 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 32,768 | 18 | 589,824 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 131,072 | 32 | 4,194,304 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 131,072 | 26 | 3,407,872 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 524,288 | 41 | 21,495,808 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 8,192 | 53 | 434,176 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 32,768 | 37 | 1,212,416 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 524,288 | 36 | 18,874,368 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 4,096 | 300 | 1,228,800 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 

| group | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| agg_keygen | ConstraintsFoldingAir | 11 | 47 | 4 | 
| agg_keygen | Eq3bAir | 3 | 68 | 4 | 
| agg_keygen | EqBaseAir | 8 | 106 | 4 | 
| agg_keygen | EqBitsAir | 5 | 26 | 4 | 
| agg_keygen | EqNegAir | 8 | 95 | 4 | 
| agg_keygen | EqNsAir | 10 | 74 | 4 | 
| agg_keygen | EqSharpUniAir | 5 | 50 | 4 | 
| agg_keygen | EqSharpUniReceiverAir | 3 | 28 | 4 | 
| agg_keygen | EqUniAir | 3 | 35 | 4 | 
| agg_keygen | ExpBitsLenAir | 2 | 44 | 3 | 
| agg_keygen | ExpressionClaimAir | 7 | 75 | 4 | 
| agg_keygen | FinalPolyMleEvalAir | 14 | 21 | 4 | 
| agg_keygen | FinalPolyQueryEvalAir | 5 | 137 | 4 | 
| agg_keygen | FractionsFolderAir | 20 | 48 | 4 | 
| agg_keygen | GkrInputAir | 21 | 22 | 4 | 
| agg_keygen | GkrLayerAir | 36 | 43 | 4 | 
| agg_keygen | GkrLayerSumcheckAir | 25 | 66 | 4 | 
| agg_keygen | GkrXiSamplerAir | 8 | 17 | 4 | 
| agg_keygen | InitialOpenedValuesAir | 13 | 156 | 4 | 
| agg_keygen | InteractionsFoldingAir | 14 | 106 | 4 | 
| agg_keygen | MerkleVerifyAir | 6 | 22 | 3 | 
| agg_keygen | MultilinearSumcheckAir | 16 | 68 | 4 | 
| agg_keygen | NonInitialOpenedValuesAir | 4 | 43 | 4 | 
| agg_keygen | OpeningClaimsAir | 25 | 109 | 4 | 
| agg_keygen | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| agg_keygen | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| agg_keygen | ProofShapeAir<4, 8> | 77 | 85 | 4 | 
| agg_keygen | PublicValuesAir | 4 | 18 | 4 | 
| agg_keygen | RangeCheckerAir<8> | 1 | 3 | 2 | 
| agg_keygen | StackingClaimsAir | 19 | 64 | 4 | 
| agg_keygen | SumcheckAir | 22 | 52 | 4 | 
| agg_keygen | SumcheckRoundsAir | 24 | 81 | 4 | 
| agg_keygen | SymbolicExpressionAir<BabyBearParameters> | 52 | 36 | 4 | 
| agg_keygen | TranscriptAir | 17 | 84 | 4 | 
| agg_keygen | UnivariateRoundAir | 15 | 62 | 4 | 
| agg_keygen | UnivariateSumcheckAir | 16 | 53 | 4 | 
| agg_keygen | UnsetPvsAir | 1 | 2 | 2 | 
| agg_keygen | VerifierPvsAir | 69 | 213 | 4 | 
| agg_keygen | VmPvsAir | 30 | 54 | 4 | 
| agg_keygen | WhirFoldingAir | 4 | 17 | 3 | 
| agg_keygen | WhirQueryAir | 5 | 57 | 4 | 
| agg_keygen | WhirRoundAir | 34 | 29 | 4 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 25 | 247 | 25 | 10 | 3 | 3 | 2 | 2 | 
| internal_recursive.0 | 1 | 13 | 165 | 13 | 3 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 13 | 153 | 12 | 3 | 0 | 3 | 0 | 0 | 
| leaf | 0 | 134 | 649 | 133 | 37 | 16 | 25 | 5 | 5 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 65,131,004 | 221 | 69 | 0 | 5 | 100 | 26 | 25 | 51 | 22 | 0 | 51 | 20 | 31 | 3 | 27 | 70 | 100 | 0 | 3 | 22 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 21,746,732 | 151 | 33 | 0 | 4 | 72 | 21 | 20 | 32 | 18 | 0 | 45 | 18 | 26 | 1 | 24 | 33 | 72 | 0 | 3 | 18 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 16,815,148 | 140 | 26 | 0 | 4 | 70 | 20 | 20 | 30 | 18 | 0 | 43 | 17 | 25 | 1 | 24 | 27 | 70 | 0 | 3 | 17 | 0 | 0 | 
| leaf | 0 | prover | 304,060,790 | 514 | 259 | 0 | 121 | 200 | 60 | 59 | 58 | 82 | 0 | 54 | 13 | 40 | 13 | 26 | 259 | 200 | 0 | 2 | 81 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 4,485,473 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,635,053 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 2,602,285 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 44,514,209 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| root | prover | 25,065,475 | 105,910 | 504 | 0 | 5 | 12,472 | 22 | 21 | 46 | 12,403 | 0 | 92,932 | 92,906 | 25 | 1 | 23 | 504 | 12,472 | 0 | 94 | 18 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 567,917,156 | 1,106 | 249 | 1 | 118 | 760 | 356 | 354 | 191 | 212 | 90 | 96 | 13 | 83 | 27 | 56 | 249 | 760 | 0 | 1 | 121 | 0 | 0 | 
| app_proof | prover | 1 | 537,266,124 | 686 | 148 | 0 | 23 | 472 | 175 | 175 | 121 | 175 | 95 | 65 | 14 | 50 | 24 | 26 | 148 | 472 | 0 | 1 | 79 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 54,936,802 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 40,900,042 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,697,213 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 82 | 1,590 | 82 | 237 | 0 | 2 | 164 | 1,540,000 | 36.47 | 
| app_proof | 1 | 16 | 857 | 16 | 40 | 0 | 2 | 113 | 1,039,903 | 37.30 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4

Max Segment Length: 4194304

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25387818431)
