| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  12.09 |  3.67 |  3.67 |
| app_proof |  9.89 |  1.98 |  1.98 |
| leaf |  1.38 |  0.88 |  0.88 |
| internal_for_leaf |  0.48 |  0.48 |  0.48 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,626 |  9,756 |  1,845 |  1,505 |
| `execute_metered_time_ms` |  138 | -          | -          | -          |
| `execute_metered_insns` |  14,793,960 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  106.84 | -          |  106.84 |  106.84 |
| `execute_preflight_insns` |  2,465,660 |  14,793,960 |  2,625,000 |  1,670,960 |
| `execute_preflight_time_ms` |  122 |  732 |  154 |  84 |
| `execute_preflight_insn_mi/s` |  41.47 | -          |  42.13 |  39.46 |
| `trace_gen_time_ms   ` |  265.50 |  1,593 |  317 |  238 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,175.17 |  7,051 |  1,274 |  1,136 |
| `prover.main_trace_commit_time_ms` |  282.50 |  1,695 |  283 |  282 |
| `prover.rap_constraints_time_ms` |  803 |  4,818 |  899 |  766 |
| `prover.openings_time_ms` |  88.67 |  532 |  92 |  86 |
| `prover.rap_constraints.logup_gkr_time_ms` |  279.67 |  1,678 |  382 |  258 |
| `prover.rap_constraints.round0_time_ms` |  356 |  2,136 |  361 |  343 |
| `prover.rap_constraints.mle_rounds_time_ms` |  166.17 |  997 |  168 |  164 |
| `prover.openings.stacked_reduction_time_ms` |  59.50 |  357 |  61 |  59 |
| `prover.openings.stacked_reduction.round0_time_ms` |  31.17 |  187 |  32 |  31 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  28.17 |  169 |  29 |  28 |
| `prover.openings.whir_time_ms` |  28.50 |  171 |  30 |  26 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  692 |  1,384 |  884 |  500 |
| `execute_preflight_time_ms` |  2 |  4 |  2 |  2 |
| `trace_gen_time_ms   ` |  78 |  156 |  106 |  50 |
| `generate_blob_total_time_ms` |  10.50 |  21 |  14 |  7 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  613.50 |  1,227 |  777 |  450 |
| `prover.main_trace_commit_time_ms` |  311.50 |  623 |  404 |  219 |
| `prover.rap_constraints_time_ms` |  224 |  448 |  282 |  166 |
| `prover.openings_time_ms` |  77 |  154 |  90 |  64 |
| `prover.rap_constraints.logup_gkr_time_ms` |  62.50 |  125 |  77 |  48 |
| `prover.rap_constraints.round0_time_ms` |  98 |  196 |  126 |  70 |
| `prover.rap_constraints.mle_rounds_time_ms` |  62.50 |  125 |  78 |  47 |
| `prover.openings.stacked_reduction_time_ms` |  46 |  92 |  55 |  37 |
| `prover.openings.stacked_reduction.round0_time_ms` |  16.50 |  33 |  22 |  11 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  29.50 |  59 |  33 |  26 |
| `prover.openings.whir_time_ms` |  30 |  60 |  34 |  26 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  481 |  481 |  481 |  481 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  31 |  31 |  31 |  31 |
| `generate_blob_total_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  449 |  449 |  449 |  449 |
| `prover.main_trace_commit_time_ms` |  225 |  225 |  225 |  225 |
| `prover.rap_constraints_time_ms` |  162 |  162 |  162 |  162 |
| `prover.openings_time_ms` |  61 |  61 |  61 |  61 |
| `prover.rap_constraints.logup_gkr_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.round0_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.mle_rounds_time_ms` |  86 |  86 |  86 |  86 |
| `prover.openings.stacked_reduction_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  179 |  179 |  179 |  179 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  166 |  166 |  166 |  166 |
| `prover.main_trace_commit_time_ms` |  50 |  50 |  50 |  50 |
| `prover.rap_constraints_time_ms` |  75 |  75 |  75 |  75 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  16 |  16 |  16 |  16 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  148 |  148 |  148 |  148 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  139 |  139 |  139 |  139 |
| `prover.main_trace_commit_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96/sha2_bench-b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| frac_sumcheck.gkr_rounds | 8.45 | app_proof.prover.5 |
| prover.batch_constraints.before_round0 | 8.45 | app_proof.prover.5 |
| frac_sumcheck.segment_tree | 8.35 | app_proof.prover.5 |
| prover.gkr_input_evals | 8.35 | app_proof.prover.5 |
| prover.batch_constraints.round0 | 7.19 | app_proof.prover.5 |
| prover.batch_constraints.fold_ple_evals | 7.19 | app_proof.prover.5 |
| prover.rap_constraints | 7.19 | app_proof.prover.5 |
| prover.merkle_tree | 6.15 | leaf.0.prover |
| prover.openings | 6.15 | leaf.0.prover |
| prover.prove_whir_opening | 6.15 | leaf.0.prover |
| prover.before_gkr_input_evals | 5.33 | leaf.0.prover |
| prover.stacked_commit | 5.33 | leaf.0.prover |
| prover.rs_code_matrix | 5.30 | leaf.0.prover |
| generate mem proving ctxs | 4.84 | app_proof.5 |
| set initial memory | 4.84 | app_proof.1 |
| tracegen.pow_checker | 1.45 | leaf.0 |
| tracegen.exp_bits_len | 1.45 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.45 | leaf.0 |
| tracegen.whir_folding | 1.20 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.19 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.19 | leaf.0 |
| tracegen.proof_shape | 1.00 | leaf.0 |
| tracegen.public_values | 1.00 | leaf.0 |
| tracegen.range_checker | 1.00 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 126 | 9 | 0 | 264,807 | 227,083 | 9 | 64 | 

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
| Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| Sha2MainAir<Sha256Config> | 148 | 39 | 3 | 
| Sha2MainAir<Sha512Config> | 276 | 71 | 3 | 
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
| agg_keygen | 68 | 9 |  |  | 0 |  | 338 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 1,505 |  |  | 138 | 14,793,960 | 106.84 | 0 | 9,901 |  | 
| internal_for_leaf |  |  |  | 481 |  |  |  |  |  |  |  |  |  | 481 | 
| internal_recursive.0 |  |  |  | 179 |  |  |  |  |  |  |  |  |  | 179 | 
| internal_recursive.1 |  |  |  | 148 |  |  |  |  |  |  |  |  |  | 148 | 
| leaf |  |  | 500 |  |  |  |  |  |  |  |  |  |  | 1,385 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 178 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 179 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 4 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 179 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 4 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 179 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 4 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 4 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 179 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 4 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 5 | 112 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 5 | 3 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 5 | 0 | 

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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 40,501,248 | 
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
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 89 | 9,240,576 | 
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
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 32,768 | 89 | 4,620,288 | 
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
| leaf | 0 | VerifierPvsAir | 1 | prover | 2 | 69 | 690 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 608 | 
| leaf | 1 | VmPvsAir | 1 | prover | 2 | 32 | 304 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,856 | 
| leaf | 10 | EqSharpUniReceiverAir | 1 | prover | 32 | 17 | 928 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 896 | 
| leaf | 11 | EqUniAir | 1 | prover | 16 | 16 | 448 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 15,360 | 
| leaf | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 1,458,176 | 
| leaf | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 729,088 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 532,480 | 
| leaf | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 266,240 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 4,608 | 
| leaf | 15 | EqNegAir | 1 | prover | 32 | 40 | 2,304 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 1,835,008 | 
| leaf | 16 | TranscriptAir | 1 | prover | 8,192 | 44 | 917,504 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 162,004,992 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 262,144 | 301 | 81,002,496 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 7,995,392 | 
| leaf | 18 | MerkleVerifyAir | 1 | prover | 65,536 | 37 | 3,997,696 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 42 | 44,800 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 42 | 22,400 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 3,072 | 
| leaf | 20 | PublicValuesAir | 1 | prover | 64 | 8 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 408 | 
| leaf | 22 | GkrInputAir | 1 | prover | 2 | 26 | 204 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 21,248 | 
| leaf | 23 | GkrLayerAir | 1 | prover | 64 | 46 | 10,624 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 264,192 | 
| leaf | 24 | GkrLayerSumcheckAir | 1 | prover | 1,024 | 45 | 132,096 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 152 | 
| leaf | 25 | GkrXiSamplerAir | 1 | prover | 2 | 10 | 76 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 1,236,992 | 
| leaf | 26 | OpeningClaimsAir | 1 | prover | 4,096 | 63 | 618,496 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 10,112 | 
| leaf | 27 | UnivariateRoundAir | 1 | prover | 64 | 27 | 5,056 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 18,048 | 
| leaf | 28 | SumcheckRoundsAir | 1 | prover | 64 | 57 | 9,024 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 843,776 | 
| leaf | 29 | StackingClaimsAir | 1 | prover | 4,096 | 35 | 421,888 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 17,563,648 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 65,536 | 60 | 17,563,648 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 2,656 | 
| leaf | 30 | EqBaseAir | 1 | prover | 16 | 51 | 1,328 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 147,456 | 
| leaf | 31 | EqBitsAir | 1 | prover | 2,048 | 16 | 73,728 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 2,720 | 
| leaf | 32 | WhirRoundAir | 1 | prover | 8 | 46 | 1,360 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 7,296 | 
| leaf | 33 | SumcheckAir | 1 | prover | 32 | 38 | 3,648 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 106,496 | 
| leaf | 34 | WhirQueryAir | 1 | prover | 1,024 | 32 | 53,248 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 524,288 | 89 | 73,924,608 | 
| leaf | 35 | InitialOpenedValuesAir | 1 | prover | 262,144 | 89 | 36,962,304 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 720,896 | 
| leaf | 36 | NonInitialOpenedValuesAir | 1 | prover | 8,192 | 28 | 360,448 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,540,096 | 
| leaf | 37 | WhirFoldingAir | 1 | prover | 16,384 | 31 | 770,048 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 352,256 | 
| leaf | 38 | FinalPolyMleEvalAir | 1 | prover | 2,048 | 34 | 176,128 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 68,157,440 | 
| leaf | 39 | FinalPolyQueryEvalAir | 1 | prover | 524,288 | 45 | 34,078,720 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 12,416 | 
| leaf | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,572,864 | 
| leaf | 41 | ExpBitsLenAir | 1 | prover | 32,768 | 16 | 786,432 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 24 | 40,960 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 256 | 24 | 20,480 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 45,568 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 256 | 33 | 22,784 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 10,368 | 
| leaf | 7 | EqNsAir | 1 | prover | 64 | 41 | 5,184 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 1,212,416 | 
| leaf | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 606,208 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 2,368 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 32 | 17 | 1,184 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 0 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 524,288 | 456 | 299,892,736 | 
| app_proof | 12 | Rv32HintStoreAir | prover | 0 | 2 | 32 | 208 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 2,048 | 21 | 75,776 | 
| app_proof | 20 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 1 | 53 | 149 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 24 | PhantomAir | prover | 0 | 1 | 6 | 18 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 77,824 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 2,048 | 32 | 98,304 | 
| app_proof | 6 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 2 | 31 | 214 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 1 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 524,288 | 456 | 299,892,736 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 262,144 | 26 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 2,048 | 21 | 75,776 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,703,936 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 77,824 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 2,048 | 32 | 98,304 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 2 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 2 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 2 | 524,288 | 456 | 299,892,736 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 2 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 2 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 2 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 2 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 2 | 262,144 | 26 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 2 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 2,048 | 21 | 75,776 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 2 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,703,936 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 256 | 300 | 77,824 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 2,048 | 32 | 98,304 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 2 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 3 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 3 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 3 | 524,288 | 456 | 299,892,736 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 3 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 3 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 3 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 3 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 3 | 262,144 | 26 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 3 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 2,048 | 21 | 75,776 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 3 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,703,936 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 77,824 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 2,048 | 32 | 98,304 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 3 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 4 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 4 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 4 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 4 | 524,288 | 456 | 299,892,736 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 4 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 4 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 4 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 4 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 4 | 262,144 | 26 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 4 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 4 | 2,048 | 21 | 75,776 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 4 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 4 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 4 | 65,536 | 18 | 1,703,936 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 4 | 256 | 300 | 77,824 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 4 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 4 | 2,048 | 32 | 98,304 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 4 | 524,288 | 3 | 3,670,016 | 
| app_proof | 0 | ProgramAir | prover | 5 | 8,192 | 10 | 114,688 | 
| app_proof | 1 | VmConnectorAir | prover | 5 | 2 | 6 | 52 | 
| app_proof | 10 | Sha2MainAir<Sha256Config> | prover | 5 | 32,768 | 284 | 28,704,768 | 
| app_proof | 11 | Sha2BlockHasherVmAir<Sha256Config> | prover | 5 | 524,288 | 456 | 299,892,736 | 
| app_proof | 13 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 5 | 32,768 | 20 | 2,228,224 | 
| app_proof | 14 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 5 | 65,536 | 28 | 6,029,312 | 
| app_proof | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 5 | 32,768 | 18 | 1,900,544 | 
| app_proof | 16 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 5 | 262,144 | 32 | 22,020,096 | 
| app_proof | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 5 | 262,144 | 26 | 18,350,080 | 
| app_proof | 18 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | prover | 5 | 8 | 36 | 864 | 
| app_proof | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 5 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 5 | 2,048 | 21 | 75,776 | 
| app_proof | 20 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 5 | 64 | 53 | 9,536 | 
| app_proof | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 5 | 65,536 | 37 | 7,143,424 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 5 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 5 | 65,536 | 18 | 1,703,936 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 5 | 512 | 300 | 155,648 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 5 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 5 | 2,048 | 32 | 98,304 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 5 | 524,288 | 3 | 3,670,016 | 

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
| internal_for_leaf | 0 | 31 | 481 | 30 | 11 | 2 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 13 | 179 | 12 | 2 | 1 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 148 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 106 | 884 | 106 | 39 | 14 | 2 | 3 | 3 | 
| leaf | 1 | 50 | 500 | 49 | 19 | 7 | 2 | 3 | 3 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 100,404,864 | 449 | 224 | 0 | 13 | 162 | 44 | 43 | 86 | 32 | 0 | 61 | 24 | 37 | 7 | 29 | 225 | 162 | 0 | 2 | 31 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 25,962,119 | 166 | 50 | 0 | 3 | 75 | 24 | 23 | 28 | 22 | 0 | 40 | 16 | 23 | 1 | 21 | 50 | 75 | 0 | 2 | 21 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 139 | 34 | 0 | 3 | 63 | 21 | 21 | 24 | 17 | 0 | 40 | 18 | 22 | 1 | 20 | 35 | 63 | 0 | 2 | 16 | 0 | 0 | 
| leaf | 0 | prover | 341,718,010 | 777 | 403 | 0 | 26 | 282 | 126 | 125 | 78 | 77 | 0 | 90 | 34 | 55 | 22 | 33 | 404 | 282 | 0 | 6 | 77 | 0 | 0 | 
| leaf | 1 | prover | 179,641,792 | 450 | 219 | 0 | 13 | 166 | 70 | 69 | 47 | 48 | 0 | 64 | 26 | 37 | 11 | 26 | 219 | 166 | 0 | 5 | 47 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 6,189,251 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,310 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 18,789,445 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 11,098,819 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 630,037,121 | 1,274 | 283 | 0 | 21 | 899 | 351 | 351 | 165 | 382 | 134 | 92 | 30 | 61 | 32 | 29 | 283 | 899 | 0 | 3 | 246 | 0 | 0 | 
| app_proof | prover | 1 | 630,036,532 | 1,162 | 282 | 0 | 19 | 790 | 359 | 358 | 168 | 262 | 0 | 89 | 29 | 59 | 31 | 28 | 282 | 790 | 0 | 3 | 261 | 0 | 0 | 
| app_proof | prover | 2 | 630,036,532 | 1,160 | 282 | 0 | 19 | 787 | 361 | 361 | 167 | 259 | 0 | 89 | 29 | 60 | 31 | 28 | 282 | 787 | 0 | 3 | 257 | 0 | 0 | 
| app_proof | prover | 3 | 630,036,532 | 1,160 | 282 | 0 | 19 | 789 | 361 | 361 | 168 | 258 | 0 | 88 | 29 | 59 | 31 | 28 | 282 | 789 | 0 | 3 | 257 | 0 | 0 | 
| app_proof | prover | 4 | 630,036,532 | 1,159 | 283 | 0 | 19 | 787 | 361 | 361 | 165 | 259 | 0 | 88 | 28 | 59 | 31 | 28 | 283 | 787 | 0 | 3 | 258 | 0 | 0 | 
| app_proof | prover | 5 | 630,124,756 | 1,136 | 283 | 0 | 20 | 766 | 343 | 343 | 164 | 258 | 0 | 86 | 26 | 59 | 31 | 28 | 283 | 766 | 0 | 3 | 256 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 69,034,351 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 69,034,250 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 69,034,250 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 69,034,250 | 2,013,265,921 | 
| app_proof | prover | 4 | 0 | 69,034,250 | 2,013,265,921 | 
| app_proof | prover | 5 | 0 | 69,036,186 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 317 | 1,845 | 317 | 168 | 0 | 3 | 84 | 2,623,000 | 39.46 | 
| app_proof | 1 | 245 | 1,603 | 245 | 41 | 0 | 2 | 154 | 2,625,000 | 41.61 | 
| app_proof | 2 | 264 | 1,601 | 264 | 41 | 0 | 2 | 135 | 2,625,000 | 41.76 | 
| app_proof | 3 | 265 | 1,602 | 265 | 41 | 0 | 2 | 135 | 2,625,000 | 42.13 | 
| app_proof | 4 | 264 | 1,600 | 264 | 41 | 0 | 2 | 134 | 2,625,000 | 41.91 | 
| app_proof | 5 | 238 | 1,505 | 238 | 41 | 0 | 2 | 90 | 1,670,960 | 41.92 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/b9e6f8bf17b92a0b19c86d9fb13fc10dd817cd96

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24267614252)
