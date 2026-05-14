| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  11.52 |  4.73 |  4.73 |
| app_proof |  9.36 |  2.57 |  2.57 |
| leaf |  1.42 |  1.42 |  1.42 |
| internal_for_leaf |  0.40 |  0.40 |  0.40 |
| internal_recursive.0 |  0.19 |  0.19 |  0.19 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,316.25 |  9,265 |  2,476 |  2,212 |
| `execute_metered_time_ms` |  92 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  120.76 | -          |  120.76 |  120.76 |
| `execute_preflight_insns` |  2,791,990.25 |  11,167,961 |  2,851,000 |  2,614,961 |
| `execute_preflight_time_ms` |  163.50 |  654 |  205 |  99 |
| `execute_preflight_insn_mi/s` |  34.48 | -          |  35.33 |  32.20 |
| `trace_gen_time_ms   ` |  58.50 |  234 |  66 |  54 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,028 |  8,112 |  2,173 |  1,944 |
| `prover.main_trace_commit_time_ms` |  539.25 |  2,157 |  627 |  510 |
| `prover.rap_constraints_time_ms` |  1,345.75 |  5,383 |  1,400 |  1,292 |
| `prover.openings_time_ms` |  142 |  568 |  145 |  140 |
| `prover.rap_constraints.logup_gkr_time_ms` |  380 |  1,520 |  435 |  326 |
| `prover.rap_constraints.round0_time_ms` |  660 |  2,640 |  665 |  656 |
| `prover.rap_constraints.mle_rounds_time_ms` |  304.75 |  1,219 |  306 |  303 |
| `prover.openings.stacked_reduction_time_ms` |  106 |  424 |  108 |  104 |
| `prover.openings.stacked_reduction.round0_time_ms` |  59.75 |  239 |  61 |  59 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  45.75 |  183 |  48 |  45 |
| `prover.openings.whir_time_ms` |  35.50 |  142 |  37 |  34 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,421 |  1,421 |  1,421 |  1,421 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  145 |  145 |  145 |  145 |
| `generate_blob_total_time_ms` |  25 |  25 |  25 |  25 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,276 |  1,276 |  1,276 |  1,276 |
| `prover.main_trace_commit_time_ms` |  804 |  804 |  804 |  804 |
| `prover.rap_constraints_time_ms` |  355 |  355 |  355 |  355 |
| `prover.openings_time_ms` |  116 |  116 |  116 |  116 |
| `prover.rap_constraints.logup_gkr_time_ms` |  80 |  80 |  80 |  80 |
| `prover.rap_constraints.round0_time_ms` |  168 |  168 |  168 |  168 |
| `prover.rap_constraints.mle_rounds_time_ms` |  106 |  106 |  106 |  106 |
| `prover.openings.stacked_reduction_time_ms` |  79 |  79 |  79 |  79 |
| `prover.openings.stacked_reduction.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  42 |  42 |  42 |  42 |
| `prover.openings.whir_time_ms` |  36 |  36 |  36 |  36 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  403 |  403 |  403 |  403 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  24 |  24 |  24 |  24 |
| `generate_blob_total_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  378 |  378 |  378 |  378 |
| `prover.main_trace_commit_time_ms` |  187 |  187 |  187 |  187 |
| `prover.rap_constraints_time_ms` |  135 |  135 |  135 |  135 |
| `prover.openings_time_ms` |  56 |  56 |  56 |  56 |
| `prover.rap_constraints.logup_gkr_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  65 |  65 |  65 |  65 |
| `prover.openings.stacked_reduction_time_ms` |  34 |  34 |  34 |  34 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  187 |  187 |  187 |  187 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  174 |  174 |  174 |  174 |
| `prover.main_trace_commit_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints_time_ms` |  77 |  77 |  77 |  77 |
| `prover.openings_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints.logup_gkr_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  153 |  153 |  153 |  153 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  144 |  144 |  144 |  144 |
| `prover.main_trace_commit_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/db5ae422fd2d7aced19be94d9b779590a33ecb31/sha2_bench-db5ae422fd2d7aced19be94d9b779590a33ecb31.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| frac_sumcheck.gkr_rounds | 11.20 | app_proof.prover.3 |
| prover.batch_constraints.before_round0 | 11.20 | app_proof.prover.3 |
| prover.rap_constraints | 11.01 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 10.98 | app_proof.prover.3 |
| prover.gkr_input_evals | 10.98 | app_proof.prover.3 |
| prover.batch_constraints.fold_ple_evals | 10.44 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 10.44 | app_proof.prover.0 |
| prover.merkle_tree | 9.82 | app_proof.prover.0 |
| prover.openings | 9.82 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.82 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 8.30 | leaf.0.prover |
| prover.stacked_commit | 8.30 | leaf.0.prover |
| prover.rs_code_matrix | 8.27 | leaf.0.prover |
| generate mem proving ctxs | 2.84 | app_proof.3 |
| set initial memory | 2.84 | app_proof.1 |
| tracegen.exp_bits_len | 2.07 | leaf.0 |
| tracegen.pow_checker | 2.07 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.07 | leaf.0 |
| tracegen.whir_folding | 1.82 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.81 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.81 | leaf.0 |
| tracegen.public_values | 1.65 | leaf.0 |
| tracegen.range_checker | 1.65 | leaf.0 |
| tracegen.proof_shape | 1.65 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 9 | 0 | 267,111 | 229,500 | 9 | 63 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 148 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 84 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| 14 | Rv64HintStoreAir | 17 | 15 | 3 | 
| 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 4 | 3 | 
| 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 17 | 9 | 3 | 
| 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 12 | 10 | 3 | 
| 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 20 | 25 | 3 | 
| 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 19 | 11 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 29 | 77 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 38 | 180 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 25 | 30 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 25 | 23 | 3 | 
| 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 32 | 34 | 3 | 
| 27 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 28 | PhantomAir | 3 |  | 1 | 
| 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 30 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 81 | 9 |  |  | 0 |  | 366 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 2,212 |  |  | 92 | 11,167,961 | 120.76 | 0 | 9,365 |  | 
| internal_for_leaf |  |  |  | 403 |  |  |  |  |  |  |  |  |  | 403 | 
| internal_recursive.0 |  |  |  | 187 |  |  |  |  |  |  |  |  |  | 187 | 
| internal_recursive.1 |  |  |  | 153 |  |  |  |  |  |  |  |  |  | 154 | 
| leaf |  |  | 1,421 |  |  |  |  |  |  |  |  |  |  | 1,421 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 
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
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 36 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | 0 | 
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
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | 0 | 
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
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 0 | 65,536 | 188 | 12,320,768 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 14 | Rv64HintStoreAir | prover | 0 | 2 | 27 | 54 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 65,536 | 16 | 1,048,576 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 24 | 3,145,728 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 262,144 | 18 | 4,718,592 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 262,144 | 32 | 8,388,608 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 1 | 62 | 62 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 131,072 | 73 | 9,568,256 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluAdapterU16Air, LessThanCoreAir<4, 16> | prover | 0 | 2 | 38 | 76 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 65,536 | 45 | 2,949,120 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 1 | 43 | 43 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 1 | 65,536 | 188 | 12,320,768 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 16 | 1,048,576 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 131,072 | 24 | 3,145,728 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 262,144 | 32 | 8,388,608 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 1 | 262,144 | 26 | 6,815,744 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 131,072 | 73 | 9,568,256 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 65,536 | 45 | 2,949,120 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 2 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 2 | 65,536 | 188 | 12,320,768 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 2 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 2 | 65,536 | 16 | 1,048,576 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 2 | 131,072 | 24 | 3,145,728 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 2 | 262,144 | 18 | 4,718,592 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 2 | 262,144 | 32 | 8,388,608 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 2 | 262,144 | 26 | 6,815,744 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 2 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 2 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 2 | 131,072 | 73 | 9,568,256 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 65,536 | 45 | 2,949,120 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 2 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 3 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 3 | 65,536 | 188 | 12,320,768 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 3 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 3 | 65,536 | 16 | 1,048,576 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 3 | 131,072 | 24 | 3,145,728 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 3 | 262,144 | 18 | 4,718,592 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 3 | 262,144 | 32 | 8,388,608 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4, 16> | prover | 3 | 262,144 | 26 | 6,815,744 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 3 | 65,536 | 48 | 3,145,728 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 3 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 3 | 4 | 62 | 248 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 3 | 131,072 | 73 | 9,568,256 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 65,536 | 45 | 2,949,120 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 3 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 3 | 1,048,576 | 3 | 3,145,728 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 24 | 403 | 24 | 7 | 2 | 2 | 3 | 3 | 
| internal_recursive.0 | 1 | 12 | 187 | 12 | 2 | 1 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 153 | 8 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 145 | 1,421 | 145 | 54 | 25 | 2 | 6 | 6 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 61,220,603 | 378 | 186 | 0 | 11 | 135 | 36 | 36 | 65 | 32 | 0 | 56 | 22 | 34 | 5 | 28 | 187 | 135 | 0 | 2 | 31 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,879 | 174 | 51 | 0 | 4 | 77 | 24 | 23 | 29 | 23 | 0 | 45 | 21 | 24 | 1 | 22 | 51 | 77 | 0 | 2 | 22 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 144 | 34 | 0 | 4 | 67 | 24 | 23 | 24 | 18 | 0 | 42 | 19 | 22 | 1 | 21 | 34 | 67 | 0 | 2 | 17 | 0 | 0 | 
| leaf | 0 | prover | 425,198,438 | 1,276 | 804 | 0 | 196 | 355 | 168 | 167 | 106 | 80 | 0 | 116 | 36 | 79 | 36 | 42 | 804 | 355 | 0 | 6 | 79 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 4,280,578 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,310 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 19,938,117 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 644,972,797 | 2,173 | 627 | 0 | 158 | 1,400 | 658 | 658 | 306 | 435 | 113 | 145 | 36 | 108 | 60 | 48 | 627 | 1,400 | 0 | 3 | 320 | 0 | 0 | 
| app_proof | prover | 1 | 644,972,556 | 2,047 | 510 | 0 | 35 | 1,394 | 656 | 655 | 306 | 432 | 128 | 141 | 34 | 107 | 61 | 45 | 510 | 1,394 | 0 | 3 | 303 | 0 | 0 | 
| app_proof | prover | 2 | 644,972,556 | 1,948 | 510 | 0 | 35 | 1,297 | 665 | 665 | 304 | 327 | 0 | 140 | 35 | 105 | 59 | 45 | 510 | 1,297 | 0 | 3 | 326 | 0 | 0 | 
| app_proof | prover | 3 | 644,972,804 | 1,944 | 509 | 0 | 37 | 1,292 | 661 | 661 | 303 | 326 | 0 | 142 | 37 | 104 | 59 | 45 | 510 | 1,292 | 0 | 3 | 326 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 112,943,517 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 112,943,370 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 112,943,370 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 112,943,486 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 66 | 2,476 | 66 | 137 | 0 | 3 | 99 | 2,851,000 | 32.20 | 
| app_proof | 1 | 57 | 2,351 | 57 | 41 | 0 | 1 | 205 | 2,851,000 | 35.17 | 
| app_proof | 2 | 57 | 2,226 | 57 | 41 | 0 | 1 | 178 | 2,851,000 | 35.33 | 
| app_proof | 3 | 54 | 2,212 | 54 | 41 | 0 | 1 | 172 | 2,614,961 | 35.21 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/db5ae422fd2d7aced19be94d9b779590a33ecb31

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25877125387)
