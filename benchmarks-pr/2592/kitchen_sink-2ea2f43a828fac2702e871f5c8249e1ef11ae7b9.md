| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  3.10 |  2.64 |  2.65 |
| app_proof |  2.30 |  1.84 |  1.84 |
| leaf |  0.43 |  0.43 |  0.43 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.11 |  0.11 |  0.11 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,132 |  2,264 |  1,806 |  458 |
| `execute_metered_time_ms` |  34 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  74.67 | -          |  74.67 |  74.67 |
| `execute_preflight_insns` |  1,289,951.50 |  2,579,903 |  2,020,000 |  559,903 |
| `execute_preflight_time_ms` |  77 |  154 |  78 |  76 |
| `execute_preflight_insn_mi/s` |  37.05 | -          |  38.35 |  35.74 |
| `trace_gen_time_ms   ` |  57 |  114 |  93 |  21 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  885 |  1,770 |  1,452 |  318 |
| `prover.main_trace_commit_time_ms` |  289.50 |  579 |  497 |  82 |
| `prover.rap_constraints_time_ms` |  530 |  1,060 |  860 |  200 |
| `prover.openings_time_ms` |  64 |  128 |  93 |  35 |
| `prover.rap_constraints.logup_gkr_time_ms` |  157 |  314 |  272 |  42 |
| `prover.rap_constraints.round0_time_ms` |  278 |  556 |  446 |  110 |
| `prover.rap_constraints.mle_rounds_time_ms` |  93.50 |  187 |  141 |  46 |
| `prover.openings.stacked_reduction_time_ms` |  51 |  102 |  76 |  26 |
| `prover.openings.stacked_reduction.round0_time_ms` |  16 |  32 |  25 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  34.50 |  69 |  51 |  18 |
| `prover.openings.whir_time_ms` |  13 |  26 |  17 |  9 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  433 |  433 |  433 |  433 |
| `execute_preflight_time_ms` |  31 |  31 |  31 |  31 |
| `trace_gen_time_ms   ` |  112 |  112 |  112 |  112 |
| `generate_blob_total_time_ms` |  12 |  12 |  12 |  12 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  321 |  321 |  321 |  321 |
| `prover.main_trace_commit_time_ms` |  129 |  129 |  129 |  129 |
| `prover.rap_constraints_time_ms` |  144 |  144 |  144 |  144 |
| `prover.openings_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints.logup_gkr_time_ms` |  59 |  59 |  59 |  59 |
| `prover.rap_constraints.round0_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.whir_time_ms` |  15 |  15 |  15 |  15 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  157 |  157 |  157 |  157 |
| `execute_preflight_time_ms` |  4 |  4 |  4 |  4 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  138 |  138 |  138 |  138 |
| `prover.main_trace_commit_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints_time_ms` |  64 |  64 |  64 |  64 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  9 |  9 |  9 |  9 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  110 |  110 |  110 |  110 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  100 |  100 |  100 |  100 |
| `prover.main_trace_commit_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  8 |  8 |  8 |  8 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  105 |  105 |  105 |  105 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  8 |  8 |  8 |  8 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  96 |  96 |  96 |  96 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  49 |  49 |  49 |  49 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  10 |  10 |  10 |  10 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/2ea2f43a828fac2702e871f5c8249e1ef11ae7b9/kitchen_sink-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 16.50 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 15.52 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 15.52 | app_proof.prover.0 |
| prover.gkr_input_evals | 15.50 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 15.50 | app_proof.prover.0 |
| prover.merkle_tree | 14.65 | app_proof.prover.0 |
| prover.openings | 14.65 | app_proof.prover.0 |
| prover.prove_whir_opening | 14.65 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 14.02 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 14.02 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 11.39 | app_proof.prover.0 |
| prover.stacked_commit | 11.39 | app_proof.prover.0 |
| prover.rs_code_matrix | 11.37 | app_proof.prover.0 |
| generate mem proving ctxs | 5.33 | app_proof.0 |
| set initial memory | 5.32 | app_proof.1 |
| tracegen.pow_checker | 1.87 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.87 | leaf.0 |
| tracegen.exp_bits_len | 1.87 | leaf.0 |
| tracegen.whir_folding | 1.74 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.73 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.73 | leaf.0 |
| tracegen.range_checker | 1.62 | leaf.0 |
| tracegen.public_values | 1.62 | leaf.0 |
| tracegen.proof_shape | 1.62 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- |
| 3 | 0 | 264,807 | 227,073 | 3 | 64 | 

| air | generate_cached_trace_time_ms |
| --- | --- |
| SymbolicExpressionAir | 0 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| ConstraintsFoldingAir | 10 | 42 | 4 | 
| Eq3bAir | 3 | 65 | 4 | 
| EqBaseAir | 8 | 89 | 4 | 
| EqBitsAir | 5 | 24 | 4 | 
| EqNegAir | 8 | 83 | 4 | 
| EqNsAir | 10 | 65 | 4 | 
| EqSharpUniAir | 5 | 48 | 4 | 
| EqSharpUniReceiverAir | 3 | 25 | 4 | 
| EqUniAir | 3 | 31 | 4 | 
| ExpBitsLenAir | 2 | 44 | 3 | 
| ExpressionClaimAir | 7 | 68 | 4 | 
| FinalPolyMleEvalAir | 13 | 19 | 4 | 
| FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| FractionsFolderAir | 17 | 41 | 4 | 
| GkrInputAir | 19 | 15 | 4 | 
| GkrLayerAir | 30 | 38 | 4 | 
| GkrLayerSumcheckAir | 21 | 59 | 4 | 
| GkrXiSamplerAir | 7 | 17 | 4 | 
| InitialOpenedValuesAir | 13 | 145 | 4 | 
| InteractionsFoldingAir | 13 | 94 | 4 | 
| KeccakfOpAir | 310 | 52 | 2 | 
| KeccakfPermAir | 2 | 3,183 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| MerkleVerifyAir | 6 | 22 | 3 | 
| MultilinearSumcheckAir | 14 | 60 | 4 | 
| NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| OpeningClaimsAir | 22 | 98 | 4 | 
| PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| ProgramAir | 1 |  | 1 | 
| ProofShapeAir<4, 8> | 77 | 85 | 4 | 
| PublicValuesAir | 4 | 18 | 4 | 
| RangeCheckerAir<8> | 1 | 3 | 2 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| Rv32HintStoreAir | 18 | 17 | 3 | 
| Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| Sha2MainAir<Sha256Config> | 148 | 37 | 3 | 
| Sha2MainAir<Sha512Config> | 276 | 69 | 3 | 
| StackingClaimsAir | 17 | 57 | 4 | 
| SumcheckAir | 19 | 47 | 4 | 
| SumcheckRoundsAir | 21 | 69 | 4 | 
| SymbolicExpressionAir<BabyBearParameters> | 52 | 32 | 4 | 
| TranscriptAir | 17 | 84 | 4 | 
| UnivariateRoundAir | 13 | 54 | 4 | 
| UnivariateSumcheckAir | 14 | 46 | 4 | 
| UnsetPvsAir | 1 | 2 | 2 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VerifierPvsAir | 69 | 213 | 4 | 
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
| VmPvsAir | 30 | 54 | 4 | 
| WhirFoldingAir | 4 | 15 | 3 | 
| WhirQueryAir | 5 | 51 | 4 | 
| WhirRoundAir | 31 | 28 | 4 | 
| XorinVmAir | 561 | 177 | 3 | 

| group | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  |  | 458 | 34 | 2,579,903 | 74.67 | 0 | 2,304 |  | 
| internal_for_leaf |  | 157 |  |  |  |  |  |  | 157 | 
| internal_recursive.0 |  | 110 |  |  |  |  |  |  | 110 | 
| internal_recursive.1 |  | 105 |  |  |  |  |  |  | 105 | 
| leaf | 433 |  |  |  |  |  |  |  | 433 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 57 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 3 | 
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
| app_proof | XorinVmAir | 0 | 8 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 12 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 345 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 152 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 464 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 224 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 7,680 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 729,088 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 266,240 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 1,152 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 458,752 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 20,250,624 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 999,424 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 22,528 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 3,072 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 102 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 5,312 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 66,048 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 38 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 2,048 | 63 | 309,248 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 2,528 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 4,512 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 210,944 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 6,684,672 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 664 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 73,728 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 32,768 | 89 | 4,620,288 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 180,224 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 385,024 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 88,064 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 17,039,360 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 393,216 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 10,240 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 11,392 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 2,592 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 606,208 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 592 | 
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 345 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 152 | 
| internal_recursive.0 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 116 | 
| internal_recursive.0 | 11 | EqUniAir | 1 | prover | 4 | 16 | 112 | 
| internal_recursive.0 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| internal_recursive.0 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 729,088 | 
| internal_recursive.0 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 266,240 | 
| internal_recursive.0 | 15 | EqNegAir | 1 | prover | 8 | 40 | 576 | 
| internal_recursive.0 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 458,752 | 
| internal_recursive.0 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 32,768 | 301 | 10,125,312 | 
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 499,712 | 
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 22,528 | 
| internal_recursive.0 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| internal_recursive.0 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 3,072 | 
| internal_recursive.0 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| internal_recursive.0 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 102 | 
| internal_recursive.0 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 5,312 | 
| internal_recursive.0 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 33,024 | 
| internal_recursive.0 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 38 | 
| internal_recursive.0 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 309,248 | 
| internal_recursive.0 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 632 | 
| internal_recursive.0 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 4,512 | 
| internal_recursive.0 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 52,736 | 
| internal_recursive.0 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 6,684,672 | 
| internal_recursive.0 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 332 | 
| internal_recursive.0 | 31 | EqBitsAir | 1 | prover | 2,048 | 16 | 73,728 | 
| internal_recursive.0 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 680 | 
| internal_recursive.0 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 1,824 | 
| internal_recursive.0 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 6,656 | 
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 16,384 | 89 | 2,310,144 | 
| internal_recursive.0 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 45,056 | 
| internal_recursive.0 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 96,256 | 
| internal_recursive.0 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 22,016 | 
| internal_recursive.0 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 1,064,960 | 
| internal_recursive.0 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| internal_recursive.0 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| internal_recursive.0 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 196,608 | 
| internal_recursive.0 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 1,280 | 
| internal_recursive.0 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 11,392 | 
| internal_recursive.0 | 7 | EqNsAir | 1 | prover | 32 | 41 | 2,592 | 
| internal_recursive.0 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 606,208 | 
| internal_recursive.0 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 148 | 
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 345 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 152 | 
| internal_recursive.1 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 116 | 
| internal_recursive.1 | 11 | EqUniAir | 1 | prover | 4 | 16 | 112 | 
| internal_recursive.1 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| internal_recursive.1 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 729,088 | 
| internal_recursive.1 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 266,240 | 
| internal_recursive.1 | 15 | EqNegAir | 1 | prover | 8 | 40 | 576 | 
| internal_recursive.1 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 458,752 | 
| internal_recursive.1 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 16,384 | 301 | 5,062,656 | 
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 499,712 | 
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 22,528 | 
| internal_recursive.1 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| internal_recursive.1 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 3,072 | 
| internal_recursive.1 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| internal_recursive.1 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 102 | 
| internal_recursive.1 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 5,312 | 
| internal_recursive.1 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 33,024 | 
| internal_recursive.1 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 38 | 
| internal_recursive.1 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 309,248 | 
| internal_recursive.1 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 632 | 
| internal_recursive.1 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 4,512 | 
| internal_recursive.1 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 52,736 | 
| internal_recursive.1 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 6,684,672 | 
| internal_recursive.1 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 332 | 
| internal_recursive.1 | 31 | EqBitsAir | 1 | prover | 4,096 | 16 | 147,456 | 
| internal_recursive.1 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 680 | 
| internal_recursive.1 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 1,824 | 
| internal_recursive.1 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 6,656 | 
| internal_recursive.1 | 35 | InitialOpenedValuesAir | 1 | prover | 8,192 | 89 | 1,155,072 | 
| internal_recursive.1 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 45,056 | 
| internal_recursive.1 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 96,256 | 
| internal_recursive.1 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 22,016 | 
| internal_recursive.1 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 1,064,960 | 
| internal_recursive.1 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| internal_recursive.1 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| internal_recursive.1 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 196,608 | 
| internal_recursive.1 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 1,280 | 
| internal_recursive.1 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 11,392 | 
| internal_recursive.1 | 7 | EqNsAir | 1 | prover | 32 | 41 | 2,592 | 
| internal_recursive.1 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 606,208 | 
| internal_recursive.1 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 148 | 
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 690 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 304 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 928 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 448 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 30,720 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 131,072 | 37 | 11,665,408 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 2,129,920 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 2,304 | 
| leaf | 16 | TranscriptAir | 0 | prover | 65,536 | 44 | 7,340,032 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 162,004,992 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 3,997,696 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 47 | 90,880 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 204 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 10,624 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 132,096 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 76 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 4,947,968 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 5,056 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 9,024 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 421,888 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 140,509,184 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 1,328 | 
| leaf | 31 | EqBitsAir | 0 | prover | 65,536 | 16 | 2,359,296 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 1,360 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 3,648 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 53,248 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 36,962,304 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 360,448 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 770,048 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 176,128 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 34,078,720 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 12,416 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 786,432 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 20,480 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 22,784 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 5,184 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 19,398,656 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 1,184 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 229,376 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 10,956 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 3,135 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 919 | 7,366 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 727 | 5,446 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 623 | 4,982 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 495 | 3,702 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 4 | 208 | 2,128 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 350 | 90,496 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 286 | 70,016 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 1,064 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 21 | 303,104 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 1,948 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | prover | 0 | 8 | 296 | 5,984 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 474 | 3,780 | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 378 | 5,640 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 1,064 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 1,948 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 4,256 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 32 | 393,216 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 3,896 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 1,064 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 1,948 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 4,256 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 3,896 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 1,064 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 1,948 | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,128 | 20,128 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 4,256 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 2,588 | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 3,896 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 227 | 206,592 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 0 | 256 | 166 | 120,320 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 512 | 232 | 354,304 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 2,048 | 231 | 1,660,928 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 1,111 | 4,647 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 512 | 31 | 54,784 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 14,680,064 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 284 | 14,352,384 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 149,946,368 | 
| app_proof | 57 | KeccakfOpAir | prover | 0 | 8,192 | 561 | 14,753,792 | 
| app_proof | 58 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 692,584,448 | 
| app_proof | 59 | XorinVmAir | prover | 0 | 8,192 | 914 | 25,870,336 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 10,956 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 2,228,224 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 65,536 | 28 | 6,029,312 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 65,536 | 18 | 3,801,088 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 262,144 | 32 | 22,020,096 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 18,350,080 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 32,768 | 53 | 4,882,432 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 65,536 | 37 | 7,143,424 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 3,135 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,245,184 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 10,956 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 3,135 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 229,376 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 52 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 4,096 | 21 | 151,552 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 4,096 | 32 | 196,608 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 32 | 227 | 25,824 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 1 | 32 | 166 | 15,040 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 1 | 128 | 232 | 88,576 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 256 | 231 | 207,616 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 1 | 64 | 31 | 6,848 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 14,680,064 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 1 | 8,192 | 284 | 7,176,192 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 131,072 | 456 | 74,973,184 | 
| app_proof | 57 | KeccakfOpAir | prover | 1 | 4,096 | 561 | 7,376,896 | 
| app_proof | 58 | KeccakfPermAir | prover | 1 | 65,536 | 2,634 | 173,146,112 | 
| app_proof | 59 | XorinVmAir | prover | 1 | 4,096 | 914 | 12,935,168 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 8,192 | 20 | 557,056 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 16,384 | 28 | 1,507,328 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 16,384 | 18 | 950,272 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 65,536 | 32 | 5,505,024 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 65,536 | 26 | 4,587,520 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 262,144 | 41 | 28,573,696 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 4,096 | 53 | 610,304 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 16,384 | 37 | 1,785,856 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 262,144 | 36 | 30,408,704 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,703,936 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 4,096 | 300 | 1,245,184 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 2,097,152 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 157 | 17 | 6 | 1 | 4 | 2 | 
| internal_recursive.0 | 1 | 10 | 110 | 9 | 1 | 0 | 3 | 1 | 
| internal_recursive.1 | 1 | 8 | 105 | 8 | 1 | 0 | 2 | 0 | 
| leaf | 0 | 112 | 433 | 112 | 27 | 12 | 31 | 9 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 53,472,003 | 138 | 40 | 0 | 4 | 64 | 19 | 19 | 29 | 15 | 0 | 32 | 9 | 23 | 1 | 21 | 41 | 64 | 1 | 14 | 0 | 
| internal_recursive.0 | 1 | prover | 23,651,975 | 100 | 19 | 0 | 4 | 51 | 17 | 16 | 20 | 13 | 0 | 29 | 8 | 21 | 1 | 20 | 19 | 51 | 1 | 12 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 96 | 14 | 0 | 4 | 49 | 17 | 16 | 18 | 13 | 0 | 32 | 10 | 21 | 0 | 20 | 15 | 49 | 1 | 12 | 0 | 
| leaf | 0 | prover | 428,317,568 | 321 | 129 | 0 | 10 | 144 | 48 | 47 | 37 | 59 | 0 | 46 | 15 | 31 | 7 | 24 | 129 | 144 | 2 | 58 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,522 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 40,827,715 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,221,261,804 | 1,452 | 497 | 0 | 283 | 860 | 446 | 445 | 141 | 272 | 134 | 93 | 17 | 76 | 25 | 51 | 497 | 860 | 1 | 137 | 0 | 
| app_proof | prover | 1 | 370,741,140 | 318 | 81 | 0 | 17 | 200 | 110 | 110 | 46 | 42 | 0 | 35 | 9 | 26 | 7 | 18 | 82 | 200 | 1 | 42 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 69,896,418 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 23,505,642 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 93 | 1,806 | 93 | 184 | 0 | 3 | 76 | 2,020,000 | 35.74 | 
| app_proof | 1 | 21 | 458 | 21 | 40 | 0 | 1 | 78 | 559,903 | 38.35 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/2ea2f43a828fac2702e871f5c8249e1ef11ae7b9

Max Segment Length: 4194304

Instance Type: g6e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23567637984)
