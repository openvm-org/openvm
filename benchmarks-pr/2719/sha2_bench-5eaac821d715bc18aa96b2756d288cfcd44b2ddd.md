| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  11.60 |  4.83 |  4.83 |
| app_proof |  9.46 |  2.69 |  2.69 |
| leaf |  1.40 |  1.40 |  1.40 |
| internal_for_leaf |  0.40 |  0.40 |  0.40 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,343 |  9,372 |  2,600 |  2,230 |
| `execute_metered_time_ms` |  93 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  119.83 | -          |  119.83 |  119.83 |
| `execute_preflight_insns` |  2,791,990.25 |  11,167,961 |  2,851,000 |  2,614,961 |
| `execute_preflight_time_ms` |  160.50 |  642 |  203 |  96 |
| `execute_preflight_insn_mi/s` |  37.39 | -          |  38.35 |  34.72 |
| `trace_gen_time_ms   ` |  100 |  400 |  167 |  58 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,010.25 |  8,041 |  2,174 |  1,927 |
| `prover.main_trace_commit_time_ms` |  554.25 |  2,217 |  665 |  517 |
| `prover.rap_constraints_time_ms` |  1,312 |  5,248 |  1,362 |  1,267 |
| `prover.openings_time_ms` |  143.50 |  574 |  147 |  139 |
| `prover.rap_constraints.logup_gkr_time_ms` |  351.75 |  1,407 |  402 |  302 |
| `prover.rap_constraints.round0_time_ms` |  647.50 |  2,590 |  655 |  643 |
| `prover.rap_constraints.mle_rounds_time_ms` |  311.50 |  1,246 |  313 |  310 |
| `prover.openings.stacked_reduction_time_ms` |  105.50 |  422 |  109 |  101 |
| `prover.openings.stacked_reduction.round0_time_ms` |  59.25 |  237 |  61 |  57 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  45.75 |  183 |  48 |  44 |
| `prover.openings.whir_time_ms` |  37.50 |  150 |  40 |  34 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,404 |  1,404 |  1,404 |  1,404 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  146 |  146 |  146 |  146 |
| `generate_blob_total_time_ms` |  26 |  26 |  26 |  26 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,258 |  1,258 |  1,258 |  1,258 |
| `prover.main_trace_commit_time_ms` |  794 |  794 |  794 |  794 |
| `prover.rap_constraints_time_ms` |  348 |  348 |  348 |  348 |
| `prover.openings_time_ms` |  114 |  114 |  114 |  114 |
| `prover.rap_constraints.logup_gkr_time_ms` |  79 |  79 |  79 |  79 |
| `prover.rap_constraints.round0_time_ms` |  166 |  166 |  166 |  166 |
| `prover.rap_constraints.mle_rounds_time_ms` |  103 |  103 |  103 |  103 |
| `prover.openings.stacked_reduction_time_ms` |  77 |  77 |  77 |  77 |
| `prover.openings.stacked_reduction.round0_time_ms` |  35 |  35 |  35 |  35 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  41 |  41 |  41 |  41 |
| `prover.openings.whir_time_ms` |  36 |  36 |  36 |  36 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  395 |  395 |  395 |  395 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  24 |  24 |  24 |  24 |
| `generate_blob_total_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  370 |  370 |  370 |  370 |
| `prover.main_trace_commit_time_ms` |  184 |  184 |  184 |  184 |
| `prover.rap_constraints_time_ms` |  132 |  132 |  132 |  132 |
| `prover.openings_time_ms` |  52 |  52 |  52 |  52 |
| `prover.rap_constraints.logup_gkr_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings.stacked_reduction_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  184 |  184 |  184 |  184 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  172 |  172 |  172 |  172 |
| `prover.main_trace_commit_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints_time_ms` |  76 |  76 |  76 |  76 |
| `prover.openings_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.logup_gkr_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  154 |  154 |  154 |  154 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  144 |  144 |  144 |  144 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  64 |  64 |  64 |  64 |
| `prover.openings_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  23 |  23 |  23 |  23 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/5eaac821d715bc18aa96b2756d288cfcd44b2ddd/sha2_bench-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 11.26 | app_proof.prover.3 |
| frac_sumcheck.gkr_rounds | 10.96 | app_proof.prover.3 |
| prover.batch_constraints.before_round0 | 10.96 | app_proof.prover.3 |
| prover.gkr_input_evals | 10.74 | app_proof.prover.3 |
| frac_sumcheck.segment_tree | 10.74 | app_proof.prover.3 |
| prover.batch_constraints.round0 | 10.64 | app_proof.prover.3 |
| prover.batch_constraints.fold_ple_evals | 10.64 | app_proof.prover.3 |
| prover.merkle_tree | 10.04 | app_proof.prover.3 |
| prover.openings | 10.04 | app_proof.prover.3 |
| prover.prove_whir_opening | 10.04 | app_proof.prover.3 |
| prover.before_gkr_input_evals | 8.27 | leaf.0.prover |
| prover.stacked_commit | 8.27 | leaf.0.prover |
| prover.rs_code_matrix | 8.23 | leaf.0.prover |
| generate mem proving ctxs | 4.84 | app_proof.3 |
| set initial memory | 4.84 | app_proof.1 |
| tracegen.whir_final_poly_query_eval | 2.07 | leaf.0 |
| tracegen.exp_bits_len | 2.07 | leaf.0 |
| tracegen.pow_checker | 2.07 | leaf.0 |
| tracegen.whir_folding | 1.81 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.81 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.81 | leaf.0 |
| tracegen.range_checker | 1.64 | leaf.0 |
| tracegen.proof_shape | 1.64 | leaf.0 |
| tracegen.public_values | 1.64 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 131 | 9 | 0 | 267,175 | 229,382 | 9 | 63 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 148 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 84 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| 14 | Rv64HintStoreAir | 20 | 15 | 3 | 
| 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 16 | 9 | 3 | 
| 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 13 | 37 | 3 | 
| 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 11 | 15 | 3 | 
| 2 | PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 18 | 44 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| 27 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 28 | PhantomAir | 3 |  | 1 | 
| 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 30 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 26 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 33 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 32 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 20 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 23 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 74 | 9 |  |  | 0 |  | 353 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 2,230 |  |  | 93 | 11,167,961 | 119.83 | 0 | 9,472 |  | 
| internal_for_leaf |  |  |  | 395 |  |  |  |  |  |  |  |  |  | 395 | 
| internal_recursive.0 |  |  |  | 184 |  |  |  |  |  |  |  |  |  | 185 | 
| internal_recursive.1 |  |  |  | 154 |  |  |  |  |  |  |  |  |  | 154 | 
| leaf |  |  | 1,404 |  |  |  |  |  |  |  |  |  |  | 1,404 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 26 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 9 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 36 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 2 | 36 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 2 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 3 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 3 | 34 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 3 | 4 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 3 | 0 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 606,208 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 32,768 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 89 | 5,832,704 | 
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
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
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
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 32,768 | 89 | 2,916,352 | 
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
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
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
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 1,048,576 | 301 | 315,621,376 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 43 | 5,504 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 104 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 92,160 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 40 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 3,456 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 7,296 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 286,720 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 736 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 524,288 | 89 | 46,661,632 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 24 | 12,288 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 0 | 65,536 | 236 | 15,466,496 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 14 | Rv64HintStoreAir | prover | 0 | 2 | 38 | 76 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 65,536 | 25 | 1,638,400 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 262,144 | 23 | 6,029,312 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 262,144 | 44 | 11,534,336 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 2,048 | 20 | 40,960 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 1 | 66 | 66 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 131,072 | 77 | 10,092,544 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | prover | 0 | 2 | 53 | 106 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 65,536 | 49 | 3,211,264 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 2,048 | 32 | 65,536 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 1 | 47 | 47 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 1 | 65,536 | 236 | 15,466,496 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 25 | 1,638,400 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 131,072 | 32 | 4,194,304 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 262,144 | 23 | 6,029,312 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 1 | 262,144 | 44 | 11,534,336 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 1 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 2,048 | 20 | 40,960 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 131,072 | 77 | 10,092,544 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 65,536 | 49 | 3,211,264 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 2,048 | 32 | 65,536 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 2 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 2 | 65,536 | 236 | 15,466,496 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 2 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 2 | 65,536 | 25 | 1,638,400 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 2 | 131,072 | 32 | 4,194,304 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 2 | 262,144 | 23 | 6,029,312 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 2 | 262,144 | 44 | 11,534,336 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 2 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 2,048 | 20 | 40,960 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 2 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 2 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 2 | 131,072 | 77 | 10,092,544 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 65,536 | 49 | 3,211,264 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 2 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 2,048 | 32 | 65,536 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 3 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 3 | 65,536 | 236 | 15,466,496 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 3 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 3 | 65,536 | 25 | 1,638,400 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 3 | 131,072 | 32 | 4,194,304 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 3 | 262,144 | 23 | 6,029,312 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 3 | 262,144 | 44 | 11,534,336 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 3 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 2,048 | 20 | 40,960 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 3 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 3 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 3 | 4 | 66 | 264 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 3 | 131,072 | 77 | 10,092,544 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 65,536 | 49 | 3,211,264 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 3 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 512 | 300 | 153,600 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 2,048 | 32 | 65,536 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 3 | 1,048,576 | 3 | 3,145,728 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 24 | 395 | 24 | 7 | 2 | 2 | 3 | 3 | 
| internal_recursive.0 | 1 | 12 | 184 | 11 | 2 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 154 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 146 | 1,404 | 146 | 54 | 26 | 3 | 6 | 6 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 61,220,603 | 370 | 183 | 0 | 10 | 132 | 36 | 35 | 63 | 32 | 0 | 52 | 19 | 33 | 5 | 27 | 184 | 132 | 0 | 2 | 31 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,879 | 172 | 50 | 0 | 3 | 76 | 24 | 23 | 29 | 23 | 0 | 44 | 20 | 23 | 1 | 21 | 51 | 76 | 0 | 2 | 22 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 144 | 33 | 0 | 4 | 64 | 22 | 21 | 24 | 17 | 0 | 46 | 23 | 22 | 1 | 21 | 33 | 64 | 0 | 2 | 17 | 0 | 0 | 
| leaf | 0 | prover | 424,379,238 | 1,258 | 794 | 0 | 196 | 348 | 166 | 165 | 103 | 79 | 0 | 114 | 36 | 77 | 35 | 41 | 794 | 348 | 0 | 6 | 78 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 4,280,578 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,310 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 19,839,813 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 662,392,121 | 2,174 | 664 | 0 | 189 | 1,362 | 646 | 645 | 313 | 402 | 97 | 147 | 38 | 109 | 60 | 48 | 665 | 1,362 | 0 | 3 | 303 | 0 | 0 | 
| app_proof | prover | 1 | 662,391,820 | 2,008 | 517 | 0 | 39 | 1,351 | 643 | 643 | 313 | 393 | 112 | 139 | 38 | 101 | 57 | 44 | 517 | 1,351 | 0 | 3 | 281 | 0 | 0 | 
| app_proof | prover | 2 | 662,391,820 | 1,927 | 517 | 0 | 39 | 1,267 | 655 | 654 | 310 | 302 | 0 | 142 | 34 | 107 | 61 | 45 | 517 | 1,267 | 0 | 3 | 301 | 0 | 0 | 
| app_proof | prover | 3 | 662,468,884 | 1,932 | 517 | 0 | 41 | 1,268 | 646 | 646 | 310 | 310 | 0 | 146 | 40 | 105 | 59 | 46 | 518 | 1,268 | 0 | 3 | 310 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 98,662,793 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 98,662,666 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 98,662,666 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 98,663,022 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 167 | 2,600 | 167 | 162 | 0 | 3 | 96 | 2,851,000 | 34.72 | 
| app_proof | 1 | 58 | 2,311 | 58 | 41 | 0 | 1 | 203 | 2,851,000 | 38.20 | 
| app_proof | 2 | 86 | 2,231 | 86 | 41 | 0 | 1 | 176 | 2,851,000 | 38.35 | 
| app_proof | 3 | 89 | 2,230 | 89 | 41 | 0 | 1 | 167 | 2,614,961 | 38.29 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/5eaac821d715bc18aa96b2756d288cfcd44b2ddd

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25885138404)
