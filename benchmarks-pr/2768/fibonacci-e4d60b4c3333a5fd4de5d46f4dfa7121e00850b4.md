| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  7.08 |  2.78 |  2.78 |
| app_proof |  4.45 |  0.81 |  0.81 |
| leaf |  1.35 |  0.69 |  0.69 |
| internal_for_leaf |  0.68 |  0.68 |  0.68 |
| internal_recursive.0 |  0.31 |  0.31 |  0.31 |
| internal_recursive.1 |  0.28 |  0.28 |  0.28 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  626.57 |  4,386 |  744 |  604 |
| `execute_metered_time_ms` |  65 | -          | -          | -          |
| `execute_metered_insns` |  12,000,265 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  183.90 | -          |  183.90 |  183.90 |
| `execute_preflight_insns` |  1,714,323.57 |  12,000,265 |  1,747,000 |  1,518,265 |
| `execute_preflight_time_ms` |  48.86 |  342 |  54 |  42 |
| `execute_preflight_insn_mi/s` |  50.40 | -          |  50.70 |  49.45 |
| `trace_gen_time_ms   ` |  169.86 |  1,189 |  175 |  167 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  347.57 |  2,433 |  351 |  344 |
| `prover.main_trace_commit_time_ms` |  59.71 |  418 |  60 |  59 |
| `prover.rap_constraints_time_ms` |  242.43 |  1,697 |  246 |  240 |
| `prover.openings_time_ms` |  44 |  308 |  47 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  174.86 |  1,224 |  176 |  174 |
| `prover.rap_constraints.round0_time_ms` |  38.14 |  267 |  41 |  37 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28.57 |  200 |  29 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  21.71 |  152 |  24 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  9.43 |  66 |  10 |  9 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11.71 |  82 |  14 |  11 |
| `prover.openings.whir_time_ms` |  22 |  154 |  26 |  19 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  674.50 |  1,349 |  689 |  660 |
| `execute_preflight_time_ms` |  2 |  4 |  2 |  2 |
| `trace_gen_time_ms   ` |  66.50 |  133 |  80 |  53 |
| `generate_blob_total_time_ms` |  5 |  10 |  6 |  4 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  607 |  1,214 |  608 |  606 |
| `prover.main_trace_commit_time_ms` |  254 |  508 |  255 |  253 |
| `prover.rap_constraints_time_ms` |  263 |  526 |  264 |  262 |
| `prover.openings_time_ms` |  89 |  178 |  90 |  88 |
| `prover.rap_constraints.logup_gkr_time_ms` |  68 |  136 |  68 |  68 |
| `prover.rap_constraints.round0_time_ms` |  101.50 |  203 |  102 |  101 |
| `prover.rap_constraints.mle_rounds_time_ms` |  92.50 |  185 |  93 |  92 |
| `prover.openings.stacked_reduction_time_ms` |  53 |  106 |  54 |  52 |
| `prover.openings.stacked_reduction.round0_time_ms` |  22.50 |  45 |  23 |  22 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  30 |  60 |  30 |  30 |
| `prover.openings.whir_time_ms` |  35.50 |  71 |  36 |  35 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  683 |  683 |  683 |  683 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  39 |  39 |  39 |  39 |
| `generate_blob_total_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  643 |  643 |  643 |  643 |
| `prover.main_trace_commit_time_ms` |  269 |  269 |  269 |  269 |
| `prover.rap_constraints_time_ms` |  270 |  270 |  270 |  270 |
| `prover.openings_time_ms` |  103 |  103 |  103 |  103 |
| `prover.rap_constraints.logup_gkr_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.round0_time_ms` |  54 |  54 |  54 |  54 |
| `prover.rap_constraints.mle_rounds_time_ms` |  170 |  170 |  170 |  170 |
| `prover.openings.stacked_reduction_time_ms` |  49 |  49 |  49 |  49 |
| `prover.openings.stacked_reduction.round0_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.whir_time_ms` |  53 |  53 |  53 |  53 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  314 |  314 |  314 |  314 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  299 |  299 |  299 |  299 |
| `prover.main_trace_commit_time_ms` |  97 |  97 |  97 |  97 |
| `prover.rap_constraints_time_ms` |  120 |  120 |  120 |  120 |
| `prover.openings_time_ms` |  80 |  80 |  80 |  80 |
| `prover.rap_constraints.logup_gkr_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.round0_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.mle_rounds_time_ms` |  58 |  58 |  58 |  58 |
| `prover.openings.stacked_reduction_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.whir_time_ms` |  49 |  49 |  49 |  49 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  281 |  281 |  281 |  281 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  266 |  266 |  266 |  266 |
| `prover.main_trace_commit_time_ms` |  73 |  73 |  73 |  73 |
| `prover.rap_constraints_time_ms` |  110 |  110 |  110 |  110 |
| `prover.openings_time_ms` |  83 |  83 |  83 |  83 |
| `prover.rap_constraints.logup_gkr_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  52 |  52 |  52 |  52 |
| `prover.openings.stacked_reduction_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.whir_time_ms` |  53 |  53 |  53 |  53 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4/fibonacci-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 5.36 | internal_for_leaf.0.prover |
| generate mem proving ctxs | 5.11 | app_proof.6 |
| set initial memory | 5.11 | app_proof.1 |
| prover.batch_constraints.before_round0 | 4.40 | leaf.0.prover |
| frac_sumcheck.gkr_rounds | 4.40 | leaf.0.prover |
| prover.gkr_input_evals | 4.32 | leaf.0.prover |
| frac_sumcheck.segment_tree | 4.32 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 4.20 | leaf.0.prover |
| prover.batch_constraints.round0 | 4.20 | leaf.0.prover |
| prover.openings | 4.18 | leaf.0.prover |
| prover.merkle_tree | 4.18 | leaf.0.prover |
| prover.prove_whir_opening | 4.18 | leaf.0.prover |
| prover.before_gkr_input_evals | 3.61 | leaf.0.prover |
| prover.stacked_commit | 3.61 | leaf.0.prover |
| prover.rs_code_matrix | 3.58 | leaf.0.prover |
| tracegen.exp_bits_len | 1.36 | leaf.0 |
| tracegen.pow_checker | 1.36 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.36 | leaf.0 |
| tracegen.whir_folding | 1.01 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.01 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.01 | leaf.0 |
| tracegen.public_values | 0.96 | leaf.0 |
| tracegen.range_checker | 0.96 | leaf.0 |
| tracegen.proof_shape | 0.96 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 51 | 10 | 0 | 331,067 | 280,135 | 10 | 63 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| PersistentBoundaryAir<8> | 4 | 3 | 3 | 
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
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 18 | 18 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 25 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 25 | 64 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 24 | 11 | 2 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 4 | 2 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 5 | 3 | 
| VmConnectorAir | 5 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 87 | 39 |  |  | 0 |  | 475 | 39 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 607 |  |  | 65 | 12,000,265 | 183.90 | 0 | 4,461 |  | 
| internal_for_leaf |  |  |  | 683 |  |  |  |  |  |  |  |  |  | 683 | 
| internal_recursive.0 |  |  |  | 314 |  |  |  |  |  |  |  |  |  | 314 | 
| internal_recursive.1 |  |  |  | 281 |  |  |  |  |  |  |  |  |  | 281 | 
| leaf |  |  | 660 |  |  |  |  |  |  |  |  |  |  | 1,349 | 

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

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 20 | 640 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 16 | 19 | 304 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 37 | 9,472 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 43 | 704,512 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 29 | 237,568 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 32 | 47 | 1,504 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 44 | 5,632 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 256 | 8 | 2,048 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 2 | 30 | 60 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 64 | 56 | 3,584 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 54 | 55,296 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 11 | 22 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 74 | 303,104 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 32 | 2,048 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 69 | 4,416 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 41 | 167,936 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 54 | 1,769,472 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 16 | 62 | 992 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 8,192 | 18 | 147,456 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 8 | 53 | 424 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 32 | 45 | 1,440 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 37 | 37,888 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 98 | 6,422,528 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 30 | 245,760 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 36 | 589,824 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 40 | 81,920 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 53 | 27,787,264 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 35 | 4,480 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 28 | 7,168 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 39 | 9,984 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 64 | 48 | 3,072 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 27 | 884,736 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 19 | 608 | 
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
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 16,384 | 98 | 1,605,632 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 69 | 276 | 
| leaf | 0 | VerifierPvsAir | 1 | prover | 4 | 69 | 276 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 1 | VmPvsAir | 1 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 20 | 1,280 | 
| leaf | 10 | EqSharpUniReceiverAir | 1 | prover | 64 | 20 | 1,280 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 19 | 608 | 
| leaf | 11 | EqUniAir | 1 | prover | 16 | 19 | 304 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 37 | 9,472 | 
| leaf | 12 | ExpressionClaimAir | 1 | prover | 128 | 37 | 4,736 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 43 | 176,128 | 
| leaf | 13 | InteractionsFoldingAir | 1 | prover | 4,096 | 43 | 176,128 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 29 | 59,392 | 
| leaf | 14 | ConstraintsFoldingAir | 1 | prover | 2,048 | 29 | 59,392 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 47 | 3,008 | 
| leaf | 15 | EqNegAir | 1 | prover | 64 | 47 | 3,008 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 16 | TranscriptAir | 1 | prover | 8,192 | 44 | 360,448 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 18 | MerkleVerifyAir | 1 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 42 | 5,376 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 128 | 42 | 5,376 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 20 | PublicValuesAir | 1 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 30 | 120 | 
| leaf | 22 | GkrInputAir | 1 | prover | 4 | 30 | 120 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 56 | 7,168 | 
| leaf | 23 | GkrLayerAir | 1 | prover | 128 | 56 | 7,168 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 54 | 110,592 | 
| leaf | 24 | GkrLayerSumcheckAir | 1 | prover | 1,024 | 54 | 55,296 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 11 | 44 | 
| leaf | 25 | GkrXiSamplerAir | 1 | prover | 4 | 11 | 44 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 74 | 303,104 | 
| leaf | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 74 | 151,552 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 32 | 4,096 | 
| leaf | 27 | UnivariateRoundAir | 1 | prover | 128 | 32 | 4,096 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 69 | 8,832 | 
| leaf | 28 | SumcheckRoundsAir | 1 | prover | 64 | 69 | 4,416 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 41 | 335,872 | 
| leaf | 29 | StackingClaimsAir | 1 | prover | 8,192 | 41 | 335,872 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 8,192 | 68 | 557,056 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 8,192 | 68 | 557,056 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 62 | 1,984 | 
| leaf | 30 | EqBaseAir | 1 | prover | 16 | 62 | 992 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 18 | 73,728 | 
| leaf | 31 | EqBitsAir | 1 | prover | 4,096 | 18 | 73,728 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 53 | 848 | 
| leaf | 32 | WhirRoundAir | 1 | prover | 16 | 53 | 848 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 45 | 2,880 | 
| leaf | 33 | SumcheckAir | 1 | prover | 64 | 45 | 2,880 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 37 | 75,776 | 
| leaf | 34 | WhirQueryAir | 1 | prover | 2,048 | 37 | 75,776 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 98 | 12,845,056 | 
| leaf | 35 | InitialOpenedValuesAir | 1 | prover | 131,072 | 98 | 12,845,056 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 30 | 491,520 | 
| leaf | 36 | NonInitialOpenedValuesAir | 1 | prover | 16,384 | 30 | 491,520 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 36 | 1,179,648 | 
| leaf | 37 | WhirFoldingAir | 1 | prover | 32,768 | 36 | 1,179,648 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 40 | 163,840 | 
| leaf | 38 | FinalPolyMleEvalAir | 1 | prover | 4,096 | 40 | 163,840 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 53 | 55,574,528 | 
| leaf | 39 | FinalPolyQueryEvalAir | 1 | prover | 1,048,576 | 53 | 55,574,528 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 35 | 2,240 | 
| leaf | 4 | FractionsFolderAir | 1 | prover | 64 | 35 | 2,240 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 41 | ExpBitsLenAir | 1 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 28 | 14,336 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 256 | 28 | 7,168 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 39 | 19,968 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 512 | 39 | 19,968 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 48 | 6,144 | 
| leaf | 7 | EqNsAir | 1 | prover | 128 | 48 | 6,144 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 27 | 442,368 | 
| leaf | 8 | Eq3bAir | 1 | prover | 8,192 | 27 | 221,184 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 19 | 1,216 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 64 | 19 | 1,216 | 

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
| agg_keygen | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
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
| internal_for_leaf | 0 | 39 | 683 | 39 | 16 | 3 | 3 | 1 | 1 | 
| internal_recursive.0 | 1 | 14 | 314 | 14 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 14 | 281 | 14 | 3 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 80 | 689 | 80 | 38 | 6 | 2 | 1 | 1 | 
| leaf | 1 | 53 | 660 | 53 | 29 | 4 | 2 | 1 | 1 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 82,256,246 | 643 | 268 | 0 | 15 | 270 | 54 | 53 | 170 | 45 | 0 | 103 | 53 | 49 | 11 | 37 | 269 | 270 | 0 | 14 | 44 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 22,549,548 | 299 | 97 | 0 | 5 | 120 | 31 | 30 | 58 | 31 | 0 | 80 | 49 | 31 | 3 | 27 | 97 | 120 | 0 | 14 | 30 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 16,815,148 | 266 | 72 | 0 | 4 | 110 | 29 | 28 | 52 | 28 | 0 | 83 | 53 | 29 | 2 | 26 | 73 | 110 | 0 | 14 | 27 | 0 | 0 | 
| leaf | 0 | prover | 158,004,778 | 608 | 254 | 0 | 16 | 264 | 102 | 101 | 93 | 68 | 0 | 88 | 35 | 52 | 22 | 30 | 255 | 264 | 0 | 6 | 67 | 0 | 0 | 
| leaf | 1 | prover | 157,198,170 | 606 | 253 | 0 | 16 | 262 | 101 | 100 | 92 | 68 | 0 | 90 | 36 | 54 | 23 | 30 | 253 | 262 | 0 | 6 | 67 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 6,439,297 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,741,549 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 2,602,285 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,860,737 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 9,613,137 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 70,295,622 | 351 | 59 | 0 | 1 | 246 | 41 | 41 | 29 | 175 | 0 | 44 | 19 | 24 | 10 | 14 | 60 | 246 | 0 | 3 | 174 | 0 | 0 | 
| app_proof | prover | 1 | 70,248,284 | 346 | 60 | 0 | 1 | 242 | 38 | 37 | 28 | 175 | 0 | 43 | 22 | 21 | 9 | 11 | 60 | 242 | 0 | 3 | 174 | 0 | 0 | 
| app_proof | prover | 2 | 70,248,284 | 347 | 58 | 0 | 1 | 243 | 39 | 38 | 29 | 175 | 0 | 44 | 22 | 21 | 10 | 11 | 59 | 243 | 0 | 3 | 174 | 0 | 0 | 
| app_proof | prover | 3 | 70,248,284 | 351 | 60 | 0 | 1 | 242 | 37 | 37 | 29 | 175 | 0 | 47 | 26 | 21 | 9 | 11 | 60 | 242 | 0 | 3 | 175 | 0 | 0 | 
| app_proof | prover | 4 | 70,248,284 | 346 | 60 | 0 | 1 | 241 | 37 | 37 | 28 | 174 | 0 | 44 | 23 | 21 | 10 | 11 | 60 | 241 | 0 | 3 | 173 | 0 | 0 | 
| app_proof | prover | 5 | 70,248,284 | 344 | 59 | 0 | 1 | 240 | 37 | 37 | 29 | 174 | 0 | 43 | 22 | 21 | 9 | 11 | 59 | 240 | 0 | 3 | 173 | 0 | 0 | 
| app_proof | prover | 6 | 70,295,004 | 348 | 60 | 0 | 1 | 243 | 38 | 38 | 28 | 176 | 0 | 43 | 20 | 23 | 9 | 13 | 60 | 243 | 0 | 3 | 175 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 35,531,965 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 4 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 5 | 0 | 35,529,418 | 2,013,265,921 | 
| app_proof | prover | 6 | 0 | 35,531,668 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 167 | 744 | 167 | 170 | 0 | 3 | 54 | 1,747,000 | 49.45 | 
| app_proof | 1 | 168 | 606 | 168 | 40 | 0 | 1 | 50 | 1,747,000 | 50.65 | 
| app_proof | 2 | 170 | 607 | 170 | 41 | 0 | 1 | 49 | 1,747,000 | 50.70 | 
| app_proof | 3 | 170 | 611 | 170 | 40 | 0 | 1 | 49 | 1,747,000 | 50.59 | 
| app_proof | 4 | 169 | 607 | 169 | 40 | 0 | 1 | 49 | 1,747,000 | 50.37 | 
| app_proof | 5 | 170 | 604 | 170 | 40 | 0 | 1 | 49 | 1,747,000 | 50.53 | 
| app_proof | 6 | 175 | 607 | 175 | 40 | 0 | 1 | 42 | 1,518,265 | 50.52 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25387818431)
