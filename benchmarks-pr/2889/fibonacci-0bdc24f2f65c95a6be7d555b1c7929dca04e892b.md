| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  4.49 |  2.36 |  2.36 |
| app_proof |  3.15 |  1.01 |  1.01 |
| leaf |  0.67 |  0.67 |  0.67 |
| internal_for_leaf |  0.35 |  0.35 |  0.35 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  762.25 |  3,049 |  914 |  550 |
| `execute_metered_time_ms` |  99 | -          | -          | -          |
| `execute_metered_insns` |  12,000,265 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  120.91 | -          |  120.91 |  120.91 |
| `execute_preflight_insns` |  3,000,066.25 |  12,000,265 |  3,495,000 |  1,515,265 |
| `execute_preflight_time_ms` |  85.25 |  341 |  105 |  44 |
| `execute_preflight_insn_mi/s` |  47.97 | -          |  49.08 |  45.19 |
| `trace_gen_time_ms   ` |  138 |  552 |  172 |  120 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  468.50 |  1,874 |  531 |  292 |
| `prover.main_trace_commit_time_ms` |  97.25 |  389 |  112 |  58 |
| `prover.rap_constraints_time_ms` |  269.75 |  1,079 |  307 |  166 |
| `prover.openings_time_ms` |  100 |  400 |  115 |  66 |
| `prover.rap_constraints.logup_gkr_time_ms` |  194.50 |  778 |  222 |  116 |
| `prover.rap_constraints.round0_time_ms` |  50.50 |  202 |  59 |  32 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  96 |  27 |  17 |
| `prover.openings.stacked_reduction_time_ms` |  18.50 |  74 |  21 |  12 |
| `prover.openings.stacked_reduction.round0_time_ms` |  10.25 |  41 |  12 |  6 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  8 |  32 |  9 |  6 |
| `prover.openings.whir_time_ms` |  81.25 |  325 |  94 |  54 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  670 |  670 |  670 |  670 |
| `execute_preflight_time_ms` |  1 |  1 |  1 |  1 |
| `trace_gen_time_ms   ` |  66 |  66 |  66 |  66 |
| `generate_blob_total_time_ms` |  5 |  5 |  5 |  5 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  603 |  603 |  603 |  603 |
| `prover.main_trace_commit_time_ms` |  232 |  232 |  232 |  232 |
| `prover.rap_constraints_time_ms` |  181 |  181 |  181 |  181 |
| `prover.openings_time_ms` |  188 |  188 |  188 |  188 |
| `prover.rap_constraints.logup_gkr_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.round0_time_ms` |  83 |  83 |  83 |  83 |
| `prover.rap_constraints.mle_rounds_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings.stacked_reduction_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction.round0_time_ms` |  12 |  12 |  12 |  12 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  164 |  164 |  164 |  164 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  347 |  347 |  347 |  347 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  328 |  328 |  328 |  328 |
| `prover.main_trace_commit_time_ms` |  117 |  117 |  117 |  117 |
| `prover.rap_constraints_time_ms` |  107 |  107 |  107 |  107 |
| `prover.openings_time_ms` |  102 |  102 |  102 |  102 |
| `prover.rap_constraints.logup_gkr_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.round0_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints.mle_rounds_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings.stacked_reduction_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  89 |  89 |  89 |  89 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  180 |  180 |  180 |  180 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  169 |  169 |  169 |  169 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  66 |  66 |  66 |  66 |
| `prover.openings_time_ms` |  56 |  56 |  56 |  56 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  47 |  47 |  47 |  47 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  148 |  148 |  148 |  148 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  138 |  138 |  138 |  138 |
| `prover.main_trace_commit_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints_time_ms` |  61 |  61 |  61 |  61 |
| `prover.openings_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  37 |  37 |  37 |  37 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/0bdc24f2f65c95a6be7d555b1c7929dca04e892b/fibonacci-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 4.61 | app_proof.3 |
| set initial memory | 4.61 | app_proof.1 |
| frac_sumcheck.gkr_rounds | 2.98 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 2.98 | app_proof.prover.0 |
| prover.stacked_commit | 2.90 | leaf.0.prover |
| prover.gkr_input_evals | 2.77 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 2.77 | app_proof.prover.0 |
| prover.openings | 2.48 | leaf.0.prover |
| prover.merkle_tree | 2.48 | leaf.0.prover |
| prover.prove_whir_opening | 2.48 | leaf.0.prover |
| prover.rs_code_matrix | 2.48 | leaf.0.prover |
| prover.rap_constraints | 1.34 | leaf.0.prover |
| prover.batch_constraints.round0 | 1.19 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.19 | leaf.0.prover |
| tracegen.exp_bits_len | 0.76 | leaf.0 |
| tracegen.pow_checker | 0.76 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.76 | leaf.0 |
| prover.before_gkr_input_evals | 0.70 | leaf.0.prover |
| tracegen.whir_folding | 0.51 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.50 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.50 | leaf.0 |
| tracegen.proof_shape | 0.46 | leaf.0 |
| tracegen.range_checker | 0.46 | leaf.0 |
| tracegen.public_values | 0.46 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 58 | 9 | 0 | 267,271 | 229,777 | 9 | 82 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 14 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 19 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | PhantomAir |  | 3 | 1 | 2 | 
| 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 22 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 35 | 3 | 
| 4 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 5 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 6 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 7 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 8 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 77 | 9 |  |  | 0 |  | 381 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 550 |  |  | 99 | 12,000,265 | 120.91 | 0 | 3,155 |  | 
| internal_for_leaf |  |  |  | 347 |  |  |  |  |  |  |  |  |  | 347 | 
| internal_recursive.0 |  |  |  | 180 |  |  |  |  |  |  |  |  |  | 180 | 
| internal_recursive.1 |  |  |  | 148 |  |  |  |  |  |  |  |  |  | 148 | 
| leaf |  |  | 670 |  |  |  |  |  |  |  |  |  |  | 670 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 9 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 9 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 3 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 9 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 3 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 0 | 

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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 71 | 284 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 37 | 151,552 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 43 | 5,504 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 104 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 92,160 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 40 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 3,456 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 7,296 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 286,720 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 8,192 | 60 | 491,520 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 736 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 89 | 11,665,408 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 0 | 9 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 0 | 232,992 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 0 | 14 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 0 | 2 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 0 | 4 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 232,994 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 232,997 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 0 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 0 | 19 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 0 | 4 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 0 | 698,975 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 2,096,954 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 0 | 2 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 0 | 3 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 5 | 
| app_proof | 20 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_BUFFER | 0 | 2 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_STOREW | 0 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 0 | 8 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 1 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 233,000 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 1 | 699,000 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 1 | 2,097,000 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 2 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 2 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 2 | 233,000 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 2 | 699,000 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 2 | 2,097,000 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 3 | 10 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 3 | 101,010 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 3 | 6 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 3 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 3 | 101,015 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 3 | 101,012 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 3 | 7 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 3 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 3 | 10 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 3 | 21 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 3 | 303,027 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 3 | 909,125 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 3 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 3 | 4 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 262,144 | 18 | 4,718,592 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 8 | 32 | 256 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 524,288 | 26 | 13,631,488 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 64 | 41 | 2,624 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 4 | 53 | 212 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 2,097,152 | 36 | 75,497,472 | 
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
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 524,288 | 26 | 13,631,488 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 2,097,152 | 36 | 75,497,472 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 2 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 2 | 262,144 | 18 | 4,718,592 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 2 | 524,288 | 26 | 13,631,488 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 2 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 2,097,152 | 36 | 75,497,472 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 128 | 32 | 4,096 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 2 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 3 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 3 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 3 | 131,072 | 18 | 2,359,296 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 3 | 2 | 32 | 64 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 3 | 262,144 | 26 | 6,815,744 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 3 | 64 | 41 | 2,624 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 3 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 64 | 21 | 1,344 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 76,800 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 256 | 32 | 8,192 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 3 | 524,288 | 3 | 1,572,864 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 3 | 4 | 20 | 80 | 

| group | air_id | air_name | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- | --- |
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 0 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 1 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 2 | 1 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 9 | 7 | 630 | 490 | 1,008 | 784 | 252 | 196 | 5,220 | 4,060 | 144 | 112 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 233,006 | 29,138 | 10,485,270 | 1,311,210 | 16,776,432 | 2,097,936 | 4,194,108 | 524,484 | 84,464,675 | 10,562,525 | 2,330,060 | 291,380 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 6 | 2 | 480 | 160 | 768 | 256 | 192 | 64 | 2,828 | 942 | 78 | 26 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 465,991 | 58,297 | 30,289,415 | 3,789,305 | 48,463,064 | 6,062,888 | 12,115,766 | 1,515,722 | 185,813,912 | 23,245,928 | 5,125,901 | 641,267 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 33 | 31 | 3,383 | 3,177 | 5,412 | 5,084 | 1,353 | 1,271 | 20,337 | 19,103 | 561 | 527 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 4 |  | 530 |  | 848 |  | 212 |  | 3,480 |  | 96 |  | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 698,975 | 349,601 | 64,655,188 | 32,338,092 | 103,448,300 | 51,740,948 | 25,862,075 | 12,935,237 | 456,081,188 | 228,114,652 | 12,581,550 | 6,292,818 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,096,964 | 188 | 188,726,760 | 16,920 | 301,962,816 | 27,072 | 75,490,704 | 6,768 | 1,520,298,900 | 136,300 | 41,939,280 | 3,760 | 
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
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 233,000 | 29,144 | 10,485,000 | 1,311,480 | 16,776,000 | 2,098,368 | 4,194,000 | 524,592 | 84,462,500 | 10,564,700 | 2,330,000 | 291,440 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 466,000 | 58,288 | 30,290,000 | 3,788,720 | 48,464,000 | 6,061,952 | 12,116,000 | 1,515,488 | 185,817,500 | 23,242,340 | 5,126,000 | 641,168 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 699,000 | 349,576 | 64,657,500 | 32,335,780 | 103,452,000 | 51,737,248 | 25,863,000 | 12,934,312 | 456,097,500 | 228,098,340 | 12,582,000 | 6,292,368 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 2,097,000 | 152 | 188,730,000 | 13,680 | 301,968,000 | 21,888 | 75,492,000 | 5,472 | 1,520,325,000 | 110,200 | 41,940,000 | 3,040 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 1 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 2 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 2 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 233,000 | 29,144 | 10,485,000 | 1,311,480 | 16,776,000 | 2,098,368 | 4,194,000 | 524,592 | 84,462,500 | 10,564,700 | 2,330,000 | 291,440 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 466,000 | 58,288 | 30,290,000 | 3,788,720 | 48,464,000 | 6,061,952 | 12,116,000 | 1,515,488 | 185,817,500 | 23,242,340 | 5,126,000 | 641,168 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 699,000 | 349,576 | 64,657,500 | 32,335,780 | 103,452,000 | 51,737,248 | 25,863,000 | 12,934,312 | 456,097,500 | 228,098,340 | 12,582,000 | 6,292,368 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 2,097,000 | 152 | 188,730,000 | 13,680 | 301,968,000 | 21,888 | 75,492,000 | 5,472 | 1,520,325,000 | 110,200 | 41,940,000 | 3,040 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 2 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 2 | 128 |  | 6,720 |  | 10,752 |  | 2,688 |  | 18,560 |  | 512 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 300 | 212 | 225,000 | 159,000 | 360,000 | 254,400 | 90,000 | 63,600 | 10,875 | 7,685 | 300 | 212 | 
| app_proof | 22 | VariableRangeCheckerAir | 2 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 2 | 172 | 84 | 27,520 | 13,440 | 22,016 | 10,752 | 5,504 | 2,688 | 24,940 | 12,180 | 688 | 336 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 2 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 0 | ProgramAir | 3 | 6,443 | 1,749 | 161,075 | 43,725 | 257,720 | 69,960 | 64,430 | 17,490 | 233,559 | 63,401 | 6,443 | 1,749 | 
| app_proof | 1 | VmConnectorAir | 3 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 10 | 6 | 700 | 420 | 1,120 | 672 | 280 | 168 | 5,800 | 3,480 | 160 | 96 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 101,016 | 30,056 | 4,545,720 | 1,352,520 | 7,273,152 | 2,164,032 | 1,818,288 | 541,008 | 36,618,300 | 10,895,300 | 1,010,160 | 300,560 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 2 |  | 160 |  | 256 |  | 64 |  | 943 |  | 26 |  | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 202,027 | 60,117 | 13,131,755 | 3,907,605 | 21,010,808 | 6,252,168 | 5,252,702 | 1,563,042 | 80,558,267 | 23,971,653 | 2,222,297 | 661,287 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 52 | 12 | 5,330 | 1,230 | 8,528 | 1,968 | 2,132 | 492 | 32,045 | 7,395 | 884 | 204 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 303,027 | 221,261 | 28,029,998 | 20,466,642 | 44,847,996 | 32,746,628 | 11,211,999 | 8,186,657 | 197,725,118 | 144,372,802 | 5,454,486 | 3,982,698 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 909,126 | 139,450 | 81,821,340 | 12,550,500 | 130,914,144 | 20,080,800 | 32,728,536 | 5,020,200 | 659,116,350 | 101,101,250 | 18,182,520 | 2,789,000 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 3 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 3 | 512 |  | 26,880 |  | 43,008 |  | 10,752 |  | 74,240 |  | 2,048 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 1,200 | 848 | 900,000 | 636,000 | 1,440,000 | 1,017,600 | 360,000 | 254,400 | 43,500 | 30,740 | 1,200 | 848 | 
| app_proof | 22 | VariableRangeCheckerAir | 3 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 3 | 688 | 336 | 110,080 | 53,760 | 88,064 | 43,008 | 22,016 | 10,752 | 99,760 | 48,720 | 2,752 | 1,344 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 3 | 524,288 |  | 7,864,320 |  | 6,291,456 |  | 1,572,864 |  | 19,005,440 |  | 524,288 |  | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 4 |  | 200 |  | 320 |  | 80 |  | 1,740 |  | 48 |  | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 347 | 18 | 5 | 1 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 180 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 148 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 66 | 670 | 66 | 26 | 5 | 1 | 1 | 1 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,749 | 328 | 117 | 0 | 0 | 107 | 35 | 34 | 51 | 20 | 0 | 102 | 89 | 13 | 3 | 9 | 117 | 107 | 0 | 3 | 19 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 169 | 46 | 0 | 0 | 66 | 24 | 23 | 28 | 14 | 0 | 56 | 47 | 8 | 1 | 6 | 46 | 66 | 0 | 3 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 138 | 32 | 0 | 0 | 61 | 23 | 22 | 24 | 13 | 0 | 44 | 37 | 7 | 1 | 5 | 32 | 61 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 147,934,254 | 603 | 232 | 0 | 0 | 181 | 83 | 83 | 53 | 44 | 0 | 188 | 164 | 24 | 12 | 11 | 232 | 181 | 0 | 8 | 43 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,806,473 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 136,618,054 | 531 | 108 | 0 | 0 | 307 | 59 | 58 | 27 | 221 | 0 | 115 | 94 | 21 | 11 | 9 | 108 | 307 | 0 | 3 | 220 | 0 | 0 | 
| app_proof | prover | 1 | 136,570,716 | 524 | 112 | 0 | 0 | 301 | 55 | 55 | 26 | 219 | 0 | 110 | 89 | 20 | 12 | 8 | 112 | 301 | 0 | 3 | 218 | 0 | 0 | 
| app_proof | prover | 2 | 136,570,716 | 527 | 111 | 0 | 0 | 305 | 56 | 55 | 26 | 222 | 0 | 109 | 88 | 21 | 12 | 9 | 111 | 305 | 0 | 3 | 222 | 0 | 0 | 
| app_proof | prover | 3 | 70,295,004 | 292 | 58 | 0 | 0 | 166 | 32 | 32 | 17 | 116 | 0 | 66 | 54 | 12 | 6 | 6 | 58 | 166 | 0 | 3 | 114 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 70,134,973 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 70,132,426 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 70,132,426 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 35,531,668 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 135 | 914 | 135 | 155 | 2,770,197,065 | 323,292,691 | 3,093,489,756 | 2,097,152 | 0 | 3 | 92 | 3,495,000 | 45.19 | 
| app_proof | 1 | 120 | 791 | 120 | 41 | 2,768,862,357 | 322,353,414 | 3,091,215,771 | 2,097,152 | 0 | 1 | 105 | 3,495,000 | 48.96 | 
| app_proof | 2 | 125 | 794 | 125 | 41 | 2,768,862,357 | 322,353,414 | 3,091,215,771 | 2,097,152 | 0 | 1 | 100 | 3,495,000 | 48.65 | 
| app_proof | 3 | 172 | 550 | 172 | 41 | 1,230,956,171 | 342,871,578 | 1,573,827,749 | 2,097,152 | 0 | 2 | 44 | 1,515,265 | 49.08 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/0bdc24f2f65c95a6be7d555b1c7929dca04e892b

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27853297933)
