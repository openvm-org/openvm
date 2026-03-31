| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.47 |  1.47 |  1.47 |
| app_proof |  0.64 |  0.64 |  0.64 |
| leaf |  0.27 |  0.27 |  0.27 |
| internal_for_leaf |  0.24 |  0.24 |  0.24 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.14 |  0.14 |  0.14 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  634 |  634 |  634 |  634 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  19.64 | -          |  19.64 |  19.64 |
| `execute_preflight_insns` |  122,348 |  122,348 |  122,348 |  122,348 |
| `execute_preflight_time_ms` |  15 |  15 |  15 |  15 |
| `execute_preflight_insn_mi/s` |  12.40 | -          |  12.40 |  12.40 |
| `trace_gen_time_ms   ` |  248 |  248 |  248 |  248 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  200 |  200 |  200 |  200 |
| `prover.main_trace_commit_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints_time_ms` |  127 |  127 |  127 |  127 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.round0_time_ms` |  73 |  73 |  73 |  73 |
| `prover.rap_constraints.mle_rounds_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  267 |  267 |  267 |  267 |
| `execute_preflight_time_ms` |  6 |  6 |  6 |  6 |
| `trace_gen_time_ms   ` |  33 |  33 |  33 |  33 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  233 |  233 |  233 |  233 |
| `prover.main_trace_commit_time_ms` |  67 |  67 |  67 |  67 |
| `prover.rap_constraints_time_ms` |  114 |  114 |  114 |  114 |
| `prover.openings_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.logup_gkr_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.round0_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  26 |  26 |  26 |  26 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  241 |  241 |  241 |  241 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  227 |  227 |  227 |  227 |
| `prover.main_trace_commit_time_ms` |  82 |  82 |  82 |  82 |
| `prover.rap_constraints_time_ms` |  99 |  99 |  99 |  99 |
| `prover.openings_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.logup_gkr_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  45 |  45 |  45 |  45 |
| `prover.openings.stacked_reduction_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  176 |  176 |  176 |  176 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  166 |  166 |  166 |  166 |
| `prover.main_trace_commit_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints_time_ms` |  69 |  69 |  69 |  69 |
| `prover.openings_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  145 |  145 |  145 |  145 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  8 |  8 |  8 |  8 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  137 |  137 |  137 |  137 |
| `prover.main_trace_commit_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/df6abc9dafa300daec6d88d50ac80569f5f552d3/ecrecover-df6abc9dafa300daec6d88d50ac80569f5f552d3.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 4.84 | app_proof.0 |
| set initial memory | 4.84 | app_proof.0 |
| prover.rap_constraints | 1.79 | internal_for_leaf.0.prover |
| frac_sumcheck.gkr_rounds | 1.67 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 1.67 | leaf.0.prover |
| frac_sumcheck.segment_tree | 1.61 | leaf.0.prover |
| prover.gkr_input_evals | 1.61 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.41 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 1.40 | internal_for_leaf.0.prover |
| prover.merkle_tree | 1.40 | internal_for_leaf.0.prover |
| prover.openings | 1.40 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 1.37 | internal_for_leaf.0.prover |
| prover.before_gkr_input_evals | 1.24 | internal_for_leaf.0.prover |
| prover.stacked_commit | 1.24 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 1.23 | internal_for_leaf.0.prover |
| tracegen.exp_bits_len | 0.50 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.50 | leaf.0 |
| tracegen.pow_checker | 0.50 | leaf.0 |
| tracegen.whir_folding | 0.43 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.43 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.43 | leaf.0 |
| tracegen.proof_shape | 0.43 | leaf.0 |
| tracegen.range_checker | 0.43 | leaf.0 |
| tracegen.public_values | 0.43 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 176 | 9 | 0 | 264,807 | 227,415 | 9 | 64 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| AccessAdapterAir<16> | 5 | 5 | 2 | 
| AccessAdapterAir<2> | 5 | 5 | 2 | 
| AccessAdapterAir<32> | 5 | 5 | 2 | 
| AccessAdapterAir<4> | 5 | 5 | 2 | 
| AccessAdapterAir<8> | 5 | 5 | 2 | 
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| KeccakVmAir | 321 | 4,247 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| ProgramAir | 1 |  | 1 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| Rv32HintStoreAir | 18 | 17 | 3 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 22 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 28 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 76 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 25 | 3 | 
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 9 | 2 | 
| VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 25 | 208 | 3 | 
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 18 | 18 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 25 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 25 | 64 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 24 | 11 | 2 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 4 | 2 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 5 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 415 | 268 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 158 | 108 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 428 | 239 | 2 | 
| VmConnectorAir | 5 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 69 | 9 |  |  | 0 |  | 339 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 634 |  |  | 6 | 122,348 | 19.64 | 0 | 646 |  | 
| internal_for_leaf |  |  |  | 241 |  |  |  |  |  |  |  |  |  | 241 | 
| internal_recursive.0 |  |  |  | 176 |  |  |  |  |  |  |  |  |  | 176 | 
| internal_recursive.1 |  |  |  | 146 |  |  |  |  |  |  |  |  |  | 146 | 
| leaf |  |  | 267 |  |  |  |  |  |  |  |  |  |  | 267 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakVmAir | 0 | 1 | 
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 30 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 13 | 

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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 10,125,312 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 147,456 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 1,155,072 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 345 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 152 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 464 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 224 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 7,680 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 1,458,176 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 532,480 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 1,152 | 
| leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 917,504 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 10,125,312 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,998,848 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 22,528 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 768 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 102 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 5,312 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 33,024 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 38 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 1,236,992 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 2,528 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 4,512 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 210,944 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 35,127,296 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 664 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 589,824 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 1,155,072 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 180,224 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 385,024 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 88,064 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 17,039,360 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 393,216 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 10,240 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 11,392 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 2,592 | 
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 2,424,832 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 592 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 458,752 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 625 | 2,393,088 | 
| app_proof | 11 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 32 | 166 | 8,512 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 16 | 263 | 14,320 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 16 | 199 | 9,200 | 
| app_proof | 14 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 4,096 | 166 | 1,089,536 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 32 | 263 | 28,640 | 
| app_proof | 16 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 16 | 199 | 9,200 | 
| app_proof | 18 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | prover | 0 | 8 | 39 | 1,080 | 
| app_proof | 19 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 64 | 31 | 6,848 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 4,096 | 20 | 131,072 | 
| app_proof | 20 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 3,670,016 | 
| app_proof | 21 | KeccakVmAir | prover | 0 | 128 | 3,163 | 569,216 | 
| app_proof | 22 | Rv32HintStoreAir | prover | 0 | 256 | 32 | 26,624 | 
| app_proof | 23 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 2,048 | 20 | 139,264 | 
| app_proof | 24 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 4,096 | 28 | 376,832 | 
| app_proof | 25 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 4,096 | 18 | 237,568 | 
| app_proof | 26 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 8,192 | 32 | 688,128 | 
| app_proof | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 16,384 | 26 | 1,146,880 | 
| app_proof | 28 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | prover | 0 | 2,048 | 36 | 221,184 | 
| app_proof | 29 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 65,536 | 41 | 7,143,424 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 32 | 196,608 | 
| app_proof | 30 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 8,192 | 53 | 1,220,608 | 
| app_proof | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 4,096 | 37 | 446,464 | 
| app_proof | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 65,536 | 36 | 7,602,176 | 
| app_proof | 33 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 34 | PhantomAir | prover | 0 | 16 | 6 | 288 | 
| app_proof | 35 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,245,184 | 
| app_proof | 36 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 6 | AccessAdapterAir<8> | prover | 0 | 16,384 | 17 | 606,208 | 
| app_proof | 7 | AccessAdapterAir<16> | prover | 0 | 4,096 | 25 | 184,320 | 
| app_proof | 8 | AccessAdapterAir<32> | prover | 0 | 2,048 | 41 | 124,928 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2,048 | 547 | 4,519,936 | 

| group | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| agg_keygen | ConstraintsFoldingAir | 10 | 42 | 4 | 
| agg_keygen | Eq3bAir | 3 | 65 | 4 | 
| agg_keygen | EqBaseAir | 8 | 89 | 4 | 
| agg_keygen | EqBitsAir | 5 | 24 | 4 | 
| agg_keygen | EqNegAir | 8 | 83 | 4 | 
| agg_keygen | EqNsAir | 10 | 65 | 4 | 
| agg_keygen | EqSharpUniAir | 5 | 48 | 4 | 
| agg_keygen | EqSharpUniReceiverAir | 3 | 25 | 4 | 
| agg_keygen | EqUniAir | 3 | 31 | 4 | 
| agg_keygen | ExpBitsLenAir | 2 | 44 | 3 | 
| agg_keygen | ExpressionClaimAir | 7 | 68 | 4 | 
| agg_keygen | FinalPolyMleEvalAir | 13 | 19 | 4 | 
| agg_keygen | FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| agg_keygen | FractionsFolderAir | 17 | 41 | 4 | 
| agg_keygen | GkrInputAir | 19 | 15 | 4 | 
| agg_keygen | GkrLayerAir | 30 | 38 | 4 | 
| agg_keygen | GkrLayerSumcheckAir | 21 | 59 | 4 | 
| agg_keygen | GkrXiSamplerAir | 7 | 17 | 4 | 
| agg_keygen | InitialOpenedValuesAir | 13 | 145 | 4 | 
| agg_keygen | InteractionsFoldingAir | 13 | 94 | 4 | 
| agg_keygen | MerkleVerifyAir | 6 | 22 | 3 | 
| agg_keygen | MultilinearSumcheckAir | 14 | 60 | 4 | 
| agg_keygen | NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| agg_keygen | OpeningClaimsAir | 22 | 98 | 4 | 
| agg_keygen | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| agg_keygen | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| agg_keygen | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| agg_keygen | PublicValuesAir | 4 | 18 | 4 | 
| agg_keygen | RangeCheckerAir<8> | 1 | 3 | 2 | 
| agg_keygen | StackingClaimsAir | 17 | 57 | 4 | 
| agg_keygen | SumcheckAir | 19 | 47 | 4 | 
| agg_keygen | SumcheckRoundsAir | 21 | 69 | 4 | 
| agg_keygen | SymbolicExpressionAir<BabyBearParameters> | 52 | 32 | 4 | 
| agg_keygen | TranscriptAir | 17 | 84 | 4 | 
| agg_keygen | UnivariateRoundAir | 13 | 54 | 4 | 
| agg_keygen | UnivariateSumcheckAir | 14 | 46 | 4 | 
| agg_keygen | UnsetPvsAir | 1 | 2 | 2 | 
| agg_keygen | VerifierPvsAir | 69 | 213 | 4 | 
| agg_keygen | VmPvsAir | 30 | 54 | 4 | 
| agg_keygen | WhirFoldingAir | 4 | 15 | 3 | 
| agg_keygen | WhirQueryAir | 5 | 51 | 4 | 
| agg_keygen | WhirRoundAir | 31 | 28 | 4 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 13 | 241 | 13 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 9 | 176 | 9 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 8 | 145 | 8 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 33 | 267 | 33 | 5 | 0 | 6 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 39,955,203 | 227 | 82 | 0 | 5 | 99 | 29 | 29 | 45 | 24 | 0 | 45 | 19 | 26 | 2 | 23 | 82 | 99 | 0 | 2 | 22 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 23,651,975 | 166 | 48 | 0 | 3 | 69 | 23 | 23 | 27 | 17 | 0 | 48 | 24 | 23 | 1 | 21 | 49 | 69 | 0 | 2 | 17 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 137 | 33 | 0 | 3 | 63 | 22 | 21 | 24 | 16 | 0 | 40 | 17 | 22 | 1 | 20 | 34 | 63 | 0 | 2 | 16 | 0 | 0 | 
| leaf | 0 | prover | 74,004,739 | 233 | 67 | 0 | 4 | 114 | 38 | 37 | 27 | 48 | 0 | 51 | 26 | 24 | 3 | 21 | 67 | 114 | 0 | 5 | 48 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,738 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,540,226 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 38,317,244 | 200 | 32 | 0 | 15 | 127 | 73 | 72 | 17 | 36 | 0 | 40 | 19 | 20 | 1 | 18 | 32 | 127 | 0 | 2 | 34 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 5,694,650 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 248 | 634 | 248 | 170 | 0 | 193 | 15 | 122,348 | 12.40 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/df6abc9dafa300daec6d88d50ac80569f5f552d3

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23823342381)
