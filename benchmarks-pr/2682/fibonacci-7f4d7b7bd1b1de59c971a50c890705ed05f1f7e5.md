| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.50 |  1.98 |  1.98 |
| app_proof |  3.85 |  0.73 |  0.73 |
| leaf |  0.94 |  0.54 |  0.54 |
| internal_for_leaf |  0.39 |  0.39 |  0.39 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  540.43 |  3,783 |  663 |  517 |
| `execute_metered_time_ms` |  65 | -          | -          | -          |
| `execute_metered_insns` |  12,000,265 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  182.96 | -          |  182.96 |  182.96 |
| `execute_preflight_insns` |  1,714,323.57 |  12,000,265 |  1,747,000 |  1,518,265 |
| `execute_preflight_time_ms` |  49 |  343 |  53 |  43 |
| `execute_preflight_insn_mi/s` |  49.66 | -          |  50.34 |  49.19 |
| `trace_gen_time_ms   ` |  161.86 |  1,133 |  168 |  159 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  269 |  1,883 |  278 |  265 |
| `prover.main_trace_commit_time_ms` |  58.29 |  408 |  59 |  58 |
| `prover.rap_constraints_time_ms` |  173.71 |  1,216 |  178 |  172 |
| `prover.openings_time_ms` |  35.86 |  251 |  40 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  126.29 |  884 |  128 |  125 |
| `prover.rap_constraints.round0_time_ms` |  28 |  196 |  31 |  27 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18.14 |  127 |  19 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  15.71 |  110 |  18 |  15 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5.86 |  41 |  6 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9.71 |  68 |  12 |  9 |
| `prover.openings.whir_time_ms` |  19.14 |  134 |  21 |  17 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  468.50 |  937 |  540 |  397 |
| `execute_preflight_time_ms` |  1 |  2 |  1 |  1 |
| `trace_gen_time_ms   ` |  48 |  96 |  59 |  37 |
| `generate_blob_total_time_ms` |  3 |  6 |  4 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  420.50 |  841 |  481 |  360 |
| `prover.main_trace_commit_time_ms` |  194 |  388 |  231 |  157 |
| `prover.rap_constraints_time_ms` |  161.50 |  323 |  180 |  143 |
| `prover.openings_time_ms` |  64 |  128 |  69 |  59 |
| `prover.rap_constraints.logup_gkr_time_ms` |  41 |  82 |  48 |  34 |
| `prover.rap_constraints.round0_time_ms` |  71.50 |  143 |  79 |  64 |
| `prover.rap_constraints.mle_rounds_time_ms` |  47.50 |  95 |  51 |  44 |
| `prover.openings.stacked_reduction_time_ms` |  35.50 |  71 |  39 |  32 |
| `prover.openings.stacked_reduction.round0_time_ms` |  10 |  20 |  12 |  8 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  25 |  50 |  26 |  24 |
| `prover.openings.whir_time_ms` |  28 |  56 |  30 |  26 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  385 |  385 |  385 |  385 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  28 |  28 |  28 |  28 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  357 |  357 |  357 |  357 |
| `prover.main_trace_commit_time_ms` |  159 |  159 |  159 |  159 |
| `prover.rap_constraints_time_ms` |  144 |  144 |  144 |  144 |
| `prover.openings_time_ms` |  54 |  54 |  54 |  54 |
| `prover.rap_constraints.logup_gkr_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.round0_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.mle_rounds_time_ms` |  71 |  71 |  71 |  71 |
| `prover.openings.stacked_reduction_time_ms` |  32 |  32 |  32 |  32 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  176 |  176 |  176 |  176 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  165 |  165 |  165 |  165 |
| `prover.main_trace_commit_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints_time_ms` |  71 |  71 |  71 |  71 |
| `prover.openings_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  149 |  149 |  149 |  149 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  139 |  139 |  139 |  139 |
| `prover.main_trace_commit_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5/fibonacci-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 4.83 | app_proof.6 |
| set initial memory | 4.83 | app_proof.1 |
| prover.rap_constraints | 3.78 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 3.76 | leaf.0.prover |
| frac_sumcheck.gkr_rounds | 3.76 | leaf.0.prover |
| frac_sumcheck.segment_tree | 3.70 | leaf.0.prover |
| prover.gkr_input_evals | 3.70 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 3.63 | leaf.0.prover |
| prover.batch_constraints.round0 | 3.63 | leaf.0.prover |
| prover.prove_whir_opening | 3.57 | leaf.0.prover |
| prover.openings | 3.57 | leaf.0.prover |
| prover.merkle_tree | 3.57 | leaf.0.prover |
| prover.before_gkr_input_evals | 3.14 | leaf.0.prover |
| prover.stacked_commit | 3.14 | leaf.0.prover |
| prover.rs_code_matrix | 3.10 | leaf.0.prover |
| tracegen.pow_checker | 0.98 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.98 | leaf.0 |
| tracegen.exp_bits_len | 0.98 | leaf.0 |
| tracegen.whir_folding | 0.72 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.72 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.71 | leaf.0 |
| tracegen.public_values | 0.66 | leaf.0 |
| tracegen.range_checker | 0.66 | leaf.0 |
| tracegen.proof_shape | 0.66 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 54 | 9 | 0 | 264,807 | 227,671 | 9 | 68 | 

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
| agg_keygen | 74 | 9 |  |  | 0 |  | 347 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 525 |  |  | 65 | 12,000,265 | 182.96 | 0 | 3,857 |  | 
| internal_for_leaf |  |  |  | 385 |  |  |  |  |  |  |  |  |  | 386 | 
| internal_recursive.0 |  |  |  | 176 |  |  |  |  |  |  |  |  |  | 176 | 
| internal_recursive.1 |  |  |  | 149 |  |  |  |  |  |  |  |  |  | 149 | 
| leaf |  |  | 397 |  |  |  |  |  |  |  |  |  |  | 939 | 

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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 690 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 304 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 928 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 448 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 15,360 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 1,458,176 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 532,480 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 2,304 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 917,504 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 20,250,624 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,998,848 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 44 | 45,056 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 256 | 8 | 6,144 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 204 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 10,624 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 132,096 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 76 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 618,496 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 5,056 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 9,024 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 421,888 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 6,684,672 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 1,328 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 8,192 | 16 | 294,912 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 1,360 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 3,648 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 53,248 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 32,768 | 89 | 4,620,288 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 360,448 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 770,048 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 176,128 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 34,078,720 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 12,416 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 786,432 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 20,480 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 22,784 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 5,184 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 1,212,416 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 1,184 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 69 | 1,380 | 
| leaf | 0 | VerifierPvsAir | 1 | prover | 4 | 69 | 1,380 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 608 | 
| leaf | 1 | VmPvsAir | 1 | prover | 4 | 32 | 608 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,856 | 
| leaf | 10 | EqSharpUniReceiverAir | 1 | prover | 64 | 17 | 1,856 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 896 | 
| leaf | 11 | EqUniAir | 1 | prover | 16 | 16 | 448 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 15,360 | 
| leaf | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 37 | 364,544 | 
| leaf | 13 | InteractionsFoldingAir | 1 | prover | 4,096 | 37 | 364,544 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 133,120 | 
| leaf | 14 | ConstraintsFoldingAir | 1 | prover | 2,048 | 25 | 133,120 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 4,608 | 
| leaf | 15 | EqNegAir | 1 | prover | 64 | 40 | 4,608 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 1,835,008 | 
| leaf | 16 | TranscriptAir | 1 | prover | 8,192 | 44 | 917,504 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 81,002,496 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 131,072 | 301 | 40,501,248 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 7,995,392 | 
| leaf | 18 | MerkleVerifyAir | 1 | prover | 65,536 | 37 | 3,997,696 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 42 | 44,800 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 128 | 42 | 44,800 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 3,072 | 
| leaf | 20 | PublicValuesAir | 1 | prover | 64 | 8 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 408 | 
| leaf | 22 | GkrInputAir | 1 | prover | 4 | 26 | 408 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 21,248 | 
| leaf | 23 | GkrLayerAir | 1 | prover | 128 | 46 | 21,248 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 264,192 | 
| leaf | 24 | GkrLayerSumcheckAir | 1 | prover | 1,024 | 45 | 132,096 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 152 | 
| leaf | 25 | GkrXiSamplerAir | 1 | prover | 4 | 10 | 152 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 618,496 | 
| leaf | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 309,248 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 10,112 | 
| leaf | 27 | UnivariateRoundAir | 1 | prover | 128 | 27 | 10,112 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 18,048 | 
| leaf | 28 | SumcheckRoundsAir | 1 | prover | 64 | 57 | 9,024 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 843,776 | 
| leaf | 29 | StackingClaimsAir | 1 | prover | 8,192 | 35 | 843,776 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 8,192 | 60 | 2,195,456 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 8,192 | 60 | 2,195,456 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 2,656 | 
| leaf | 30 | EqBaseAir | 1 | prover | 16 | 51 | 1,328 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 147,456 | 
| leaf | 31 | EqBitsAir | 1 | prover | 4,096 | 16 | 147,456 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 2,720 | 
| leaf | 32 | WhirRoundAir | 1 | prover | 16 | 46 | 2,720 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 7,296 | 
| leaf | 33 | SumcheckAir | 1 | prover | 64 | 38 | 7,296 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 106,496 | 
| leaf | 34 | WhirQueryAir | 1 | prover | 2,048 | 32 | 106,496 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 89 | 18,481,152 | 
| leaf | 35 | InitialOpenedValuesAir | 1 | prover | 65,536 | 89 | 9,240,576 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 720,896 | 
| leaf | 36 | NonInitialOpenedValuesAir | 1 | prover | 8,192 | 28 | 360,448 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,540,096 | 
| leaf | 37 | WhirFoldingAir | 1 | prover | 16,384 | 31 | 770,048 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 352,256 | 
| leaf | 38 | FinalPolyMleEvalAir | 1 | prover | 4,096 | 34 | 352,256 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 68,157,440 | 
| leaf | 39 | FinalPolyQueryEvalAir | 1 | prover | 1,048,576 | 45 | 68,157,440 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| leaf | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,572,864 | 
| leaf | 41 | ExpBitsLenAir | 1 | prover | 65,536 | 16 | 1,572,864 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 24 | 40,960 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 256 | 24 | 20,480 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 45,568 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 512 | 33 | 45,568 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 10,368 | 
| leaf | 7 | EqNsAir | 1 | prover | 128 | 41 | 10,368 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 606,208 | 
| leaf | 8 | Eq3bAir | 1 | prover | 8,192 | 25 | 303,104 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 2,368 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 64 | 17 | 2,368 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 16 | 28 | 1,472 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 7,602,176 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 8 | 32 | 672 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 18,350,080 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 64 | 41 | 6,976 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 4 | 53 | 596 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 64 | 21 | 2,368 | 
| app_proof | 20 | PhantomAir | prover | 0 | 1 | 6 | 18 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 77,824 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 256 | 32 | 12,288 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 3,670,016 | 
| app_proof | 8 | Rv32HintStoreAir | prover | 0 | 4 | 32 | 416 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 8 | 20 | 544 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 131,072 | 18 | 7,602,176 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 262,144 | 26 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 16 | 21 | 592 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 128 | 300 | 38,912 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 128 | 32 | 6,144 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 2 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 2 | 131,072 | 18 | 7,602,176 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 2 | 262,144 | 26 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 2 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 16 | 21 | 592 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 128 | 300 | 38,912 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 128 | 32 | 6,144 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 2 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 3 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 3 | 131,072 | 18 | 7,602,176 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 3 | 262,144 | 26 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 3 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 16 | 21 | 592 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 128 | 300 | 38,912 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 128 | 32 | 6,144 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 3 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 4 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 4 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 4 | 131,072 | 18 | 7,602,176 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 4 | 262,144 | 26 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 4 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 4 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 4 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 4 | 16 | 21 | 592 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 4 | 128 | 300 | 38,912 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 4 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 4 | 128 | 32 | 6,144 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 4 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 5 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 5 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 5 | 131,072 | 18 | 7,602,176 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 5 | 262,144 | 26 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 5 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 5 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 5 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 5 | 16 | 21 | 592 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 5 | 128 | 300 | 38,912 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 5 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 5 | 128 | 32 | 6,144 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 5 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 6 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 6 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 6 | 16 | 28 | 1,472 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 6 | 131,072 | 18 | 7,602,176 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 6 | 2 | 32 | 168 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 6 | 262,144 | 26 | 18,350,080 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 6 | 64 | 41 | 6,976 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 6 | 524,288 | 37 | 57,147,392 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 6 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 6 | 65,536 | 18 | 1,703,936 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 6 | 64 | 21 | 2,368 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 6 | 256 | 300 | 77,824 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 6 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 6 | 256 | 32 | 12,288 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 6 | 524,288 | 3 | 3,670,016 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 6 | 4 | 20 | 272 | 

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
| internal_for_leaf | 0 | 28 | 385 | 28 | 10 | 1 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 176 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 149 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 59 | 540 | 59 | 23 | 4 | 1 | 1 | 1 | 
| leaf | 1 | 37 | 397 | 36 | 17 | 2 | 1 | 1 | 1 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 75,533,952 | 357 | 158 | 0 | 9 | 144 | 40 | 39 | 71 | 32 | 0 | 54 | 21 | 32 | 5 | 27 | 159 | 144 | 0 | 2 | 32 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 23,651,975 | 165 | 47 | 0 | 3 | 71 | 25 | 24 | 27 | 18 | 0 | 45 | 20 | 24 | 1 | 22 | 48 | 71 | 0 | 2 | 17 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 139 | 34 | 0 | 3 | 62 | 21 | 20 | 24 | 17 | 0 | 42 | 19 | 22 | 1 | 20 | 34 | 62 | 0 | 2 | 16 | 0 | 0 | 
| leaf | 0 | prover | 187,179,962 | 481 | 231 | 0 | 15 | 180 | 79 | 78 | 51 | 48 | 0 | 69 | 30 | 39 | 12 | 26 | 231 | 180 | 0 | 5 | 47 | 0 | 0 | 
| leaf | 1 | prover | 130,607,498 | 360 | 156 | 0 | 10 | 143 | 64 | 63 | 44 | 34 | 0 | 59 | 26 | 32 | 8 | 24 | 157 | 143 | 0 | 5 | 33 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 5,632,195 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,809,925 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 7,967,637 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 212,423,482 | 278 | 59 | 0 | 1 | 178 | 31 | 30 | 18 | 128 | 0 | 40 | 21 | 18 | 5 | 12 | 59 | 178 | 0 | 2 | 126 | 0 | 0 | 
| app_proof | prover | 1 | 212,365,956 | 265 | 58 | 0 | 1 | 172 | 27 | 27 | 18 | 126 | 0 | 33 | 17 | 15 | 6 | 9 | 58 | 172 | 0 | 3 | 125 | 0 | 0 | 
| app_proof | prover | 2 | 212,365,956 | 269 | 58 | 0 | 1 | 173 | 27 | 27 | 18 | 127 | 0 | 36 | 20 | 15 | 6 | 9 | 59 | 173 | 0 | 3 | 125 | 0 | 0 | 
| app_proof | prover | 3 | 212,365,956 | 265 | 58 | 0 | 1 | 172 | 27 | 27 | 18 | 125 | 0 | 34 | 18 | 15 | 6 | 9 | 58 | 172 | 0 | 3 | 125 | 0 | 0 | 
| app_proof | prover | 4 | 212,365,956 | 265 | 58 | 0 | 1 | 173 | 27 | 27 | 18 | 126 | 0 | 33 | 17 | 15 | 6 | 9 | 58 | 173 | 0 | 3 | 125 | 0 | 0 | 
| app_proof | prover | 5 | 212,365,956 | 268 | 58 | 0 | 1 | 173 | 28 | 27 | 18 | 126 | 0 | 36 | 20 | 15 | 6 | 9 | 58 | 173 | 0 | 3 | 125 | 0 | 0 | 
| app_proof | prover | 6 | 212,421,676 | 273 | 58 | 0 | 1 | 175 | 29 | 29 | 19 | 126 | 0 | 39 | 21 | 17 | 6 | 11 | 58 | 175 | 0 | 3 | 125 | 0 | 0 | 

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
| app_proof | 0 | 159 | 663 | 159 | 171 | 0 | 3 | 53 | 1,747,000 | 49.56 | 
| app_proof | 1 | 161 | 517 | 161 | 40 | 0 | 1 | 50 | 1,747,000 | 50.34 | 
| app_proof | 2 | 162 | 522 | 162 | 41 | 0 | 1 | 49 | 1,747,000 | 50.25 | 
| app_proof | 3 | 161 | 517 | 161 | 40 | 0 | 1 | 49 | 1,747,000 | 49.77 | 
| app_proof | 4 | 161 | 518 | 161 | 40 | 0 | 1 | 49 | 1,747,000 | 49.26 | 
| app_proof | 5 | 161 | 521 | 161 | 40 | 0 | 1 | 50 | 1,747,000 | 49.21 | 
| app_proof | 6 | 168 | 525 | 168 | 40 | 0 | 1 | 43 | 1,518,265 | 49.19 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24199189210)
