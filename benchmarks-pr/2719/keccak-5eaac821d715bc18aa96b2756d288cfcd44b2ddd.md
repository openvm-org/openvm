| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  17 |  5.03 |  5.03 |
| app_proof |  13.88 |  2.70 |  2.70 |
| leaf |  2.26 |  1.46 |  1.46 |
| internal_for_leaf |  0.52 |  0.52 |  0.52 |
| internal_recursive.0 |  0.19 |  0.19 |  0.19 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,289.50 |  13,737 |  2,560 |  2,192 |
| `execute_metered_time_ms` |  140 | -          | -          | -          |
| `execute_metered_insns` |  14,365,133 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  102.43 | -          |  102.43 |  102.43 |
| `execute_preflight_insns` |  2,394,188.83 |  14,365,133 |  2,413,000 |  2,300,133 |
| `execute_preflight_time_ms` |  255 |  1,530 |  348 |  89 |
| `execute_preflight_insn_mi/s` |  34.99 | -          |  35.94 |  31.66 |
| `trace_gen_time_ms   ` |  54.17 |  325 |  156 |  33 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,918 |  11,508 |  2,151 |  1,867 |
| `prover.main_trace_commit_time_ms` |  716.83 |  4,301 |  884 |  683 |
| `prover.rap_constraints_time_ms` |  1,013 |  6,078 |  1,079 |  995 |
| `prover.openings_time_ms` |  187 |  1,122 |  189 |  185 |
| `prover.rap_constraints.logup_gkr_time_ms` |  181 |  1,086 |  235 |  170 |
| `prover.rap_constraints.round0_time_ms` |  518.17 |  3,109 |  524 |  511 |
| `prover.rap_constraints.mle_rounds_time_ms` |  312.67 |  1,876 |  332 |  308 |
| `prover.openings.stacked_reduction_time_ms` |  145.50 |  873 |  146 |  144 |
| `prover.openings.stacked_reduction.round0_time_ms` |  84.33 |  506 |  85 |  84 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  60.67 |  364 |  62 |  60 |
| `prover.openings.whir_time_ms` |  40.83 |  245 |  43 |  39 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,128 |  2,256 |  1,463 |  793 |
| `execute_preflight_time_ms` |  5.50 |  11 |  6 |  5 |
| `trace_gen_time_ms   ` |  139 |  278 |  185 |  93 |
| `generate_blob_total_time_ms` |  24.50 |  49 |  33 |  16 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  988 |  1,976 |  1,277 |  699 |
| `prover.main_trace_commit_time_ms` |  555.50 |  1,111 |  731 |  380 |
| `prover.rap_constraints_time_ms` |  327 |  654 |  419 |  235 |
| `prover.openings_time_ms` |  104 |  208 |  125 |  83 |
| `prover.rap_constraints.logup_gkr_time_ms` |  72.50 |  145 |  89 |  56 |
| `prover.rap_constraints.round0_time_ms` |  156.50 |  313 |  204 |  109 |
| `prover.rap_constraints.mle_rounds_time_ms` |  96 |  192 |  124 |  68 |
| `prover.openings.stacked_reduction_time_ms` |  69.50 |  139 |  87 |  52 |
| `prover.openings.stacked_reduction.round0_time_ms` |  30.50 |  61 |  41 |  20 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  39 |  78 |  46 |  32 |
| `prover.openings.whir_time_ms` |  34.50 |  69 |  38 |  31 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  524 |  524 |  524 |  524 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  39 |  39 |  39 |  39 |
| `generate_blob_total_time_ms` |  4 |  4 |  4 |  4 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  485 |  485 |  485 |  485 |
| `prover.main_trace_commit_time_ms` |  248 |  248 |  248 |  248 |
| `prover.rap_constraints_time_ms` |  178 |  178 |  178 |  178 |
| `prover.openings_time_ms` |  57 |  57 |  57 |  57 |
| `prover.rap_constraints.logup_gkr_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.round0_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.mle_rounds_time_ms` |  92 |  92 |  92 |  92 |
| `prover.openings.stacked_reduction_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  193 |  193 |  193 |  193 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  179 |  179 |  179 |  179 |
| `prover.main_trace_commit_time_ms` |  52 |  52 |  52 |  52 |
| `prover.rap_constraints_time_ms` |  77 |  77 |  77 |  77 |
| `prover.openings_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.logup_gkr_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  152 |  152 |  152 |  152 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  142 |  142 |  142 |  142 |
| `prover.main_trace_commit_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints_time_ms` |  66 |  66 |  66 |  66 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/5eaac821d715bc18aa96b2756d288cfcd44b2ddd/keccak-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 14.46 | app_proof.prover.4 |
| prover.merkle_tree | 13.10 | app_proof.prover.4 |
| prover.openings | 13.10 | app_proof.prover.4 |
| prover.prove_whir_opening | 13.10 | app_proof.prover.4 |
| frac_sumcheck.gkr_rounds | 12.03 | app_proof.prover.4 |
| prover.batch_constraints.before_round0 | 12.03 | app_proof.prover.4 |
| prover.gkr_input_evals | 11.96 | app_proof.prover.4 |
| frac_sumcheck.segment_tree | 11.96 | app_proof.prover.4 |
| prover.batch_constraints.round0 | 11.71 | app_proof.prover.4 |
| prover.batch_constraints.fold_ple_evals | 11.71 | app_proof.prover.4 |
| prover.before_gkr_input_evals | 9.97 | app_proof.prover.4 |
| prover.stacked_commit | 9.97 | app_proof.prover.4 |
| prover.rs_code_matrix | 9.95 | app_proof.prover.4 |
| generate mem proving ctxs | 4.84 | app_proof.0 |
| set initial memory | 4.84 | app_proof.1 |
| tracegen.exp_bits_len | 2.28 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.28 | leaf.0 |
| tracegen.pow_checker | 2.28 | leaf.0 |
| tracegen.whir_folding | 2.03 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 2.02 | leaf.0 |
| tracegen.whir_initial_opened_values | 2.02 | leaf.0 |
| tracegen.public_values | 1.68 | leaf.0 |
| tracegen.range_checker | 1.68 | leaf.0 |
| tracegen.proof_shape | 1.68 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 130 | 9 | 0 | 267,175 | 229,091 | 9 | 71 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | KeccakfOpAir | 210 | 27 | 2 | 
| 11 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 12 | XorinVmAir | 356 | 92 | 3 | 
| 13 | Rv64HintStoreAir | 20 | 15 | 3 | 
| 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 16 | 9 | 3 | 
| 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 13 | 37 | 3 | 
| 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 11 | 15 | 3 | 
| 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 2 | PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 21 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 18 | 44 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| 26 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 27 | PhantomAir | 3 |  | 1 | 
| 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 29 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 26 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 33 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 32 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 20 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 23 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 73 | 9 |  |  | 0 |  | 352 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 2,192 |  |  | 140 | 14,365,133 | 102.43 | 0 | 13,883 |  | 
| internal_for_leaf |  |  |  | 524 |  |  |  |  |  |  |  |  |  | 524 | 
| internal_recursive.0 |  |  |  | 193 |  |  |  |  |  |  |  |  |  | 193 | 
| internal_recursive.1 |  |  |  | 152 |  |  |  |  |  |  |  |  |  | 152 | 
| leaf |  |  | 793 |  |  |  |  |  |  |  |  |  |  | 2,256 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 6 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 15 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 15 | 
| app_proof | KeccakfOpAir | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 2 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 2 | 0 | 
| app_proof | XorinVmAir | 2 | 15 | 
| app_proof | KeccakfOpAir | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 3 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 3 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 3 | 0 | 
| app_proof | XorinVmAir | 3 | 15 | 
| app_proof | KeccakfOpAir | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 4 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 4 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 4 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 4 | 0 | 
| app_proof | XorinVmAir | 4 | 15 | 
| app_proof | KeccakfOpAir | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 5 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 5 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 5 | 5 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 5 | 0 | 
| app_proof | XorinVmAir | 5 | 15 | 

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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 44 | 5,632 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 368 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 1,216 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 32,768 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 89 | 11,665,408 | 
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
| leaf | 0 | VerifierPvsAir | 1 | prover | 2 | 69 | 138 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 1 | VmPvsAir | 1 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 10 | EqSharpUniReceiverAir | 1 | prover | 32 | 17 | 544 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 11 | EqUniAir | 1 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 12 | ExpressionClaimAir | 1 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 13 | InteractionsFoldingAir | 1 | prover | 16,384 | 37 | 606,208 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 14 | ConstraintsFoldingAir | 1 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 15 | EqNegAir | 1 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 16 | TranscriptAir | 1 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 1,048,576 | 301 | 315,621,376 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 18 | MerkleVerifyAir | 1 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 43 | 5,504 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 43 | 2,752 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 20 | PublicValuesAir | 1 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 104 | 
| leaf | 22 | GkrInputAir | 1 | prover | 2 | 26 | 52 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 23 | GkrLayerAir | 1 | prover | 64 | 46 | 2,944 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 92,160 | 
| leaf | 24 | GkrLayerSumcheckAir | 1 | prover | 1,024 | 45 | 46,080 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 40 | 
| leaf | 25 | GkrXiSamplerAir | 1 | prover | 2 | 10 | 20 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 2,064,384 | 
| leaf | 26 | OpeningClaimsAir | 1 | prover | 16,384 | 63 | 1,032,192 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 3,456 | 
| leaf | 27 | UnivariateRoundAir | 1 | prover | 64 | 27 | 1,728 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 7,296 | 
| leaf | 28 | SumcheckRoundsAir | 1 | prover | 64 | 57 | 3,648 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 286,720 | 
| leaf | 29 | StackingClaimsAir | 1 | prover | 4,096 | 35 | 143,360 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 30 | EqBaseAir | 1 | prover | 16 | 51 | 816 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 31 | EqBitsAir | 1 | prover | 2,048 | 16 | 32,768 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 736 | 
| leaf | 32 | WhirRoundAir | 1 | prover | 8 | 46 | 368 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 33 | SumcheckAir | 1 | prover | 32 | 38 | 1,216 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 34 | WhirQueryAir | 1 | prover | 1,024 | 32 | 32,768 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 1,048,576 | 89 | 93,323,264 | 
| leaf | 35 | InitialOpenedValuesAir | 1 | prover | 524,288 | 89 | 46,661,632 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 36 | NonInitialOpenedValuesAir | 1 | prover | 8,192 | 28 | 229,376 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 37 | WhirFoldingAir | 1 | prover | 16,384 | 31 | 507,904 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 38 | FinalPolyMleEvalAir | 1 | prover | 2,048 | 34 | 69,632 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 39 | FinalPolyQueryEvalAir | 1 | prover | 524,288 | 45 | 23,592,960 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 41 | ExpBitsLenAir | 1 | prover | 32,768 | 16 | 524,288 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 24 | 12,288 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 256 | 24 | 6,144 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 7 | EqNsAir | 1 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 131,072 | 25 | 3,276,800 | 
| leaf | 8 | Eq3bAir | 1 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 0 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 0 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 1 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 1 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 1 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 1 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 1 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 512 | 300 | 153,600 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 1 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 2 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 2 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 2 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 2 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 2 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 2 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 2 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 2 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 2 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 2 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 2 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 2 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 2 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 512 | 300 | 153,600 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 2 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 3 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 3 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 3 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 3 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 3 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 3 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 3 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 3 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 3 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 3 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 3 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 3 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 3 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 512 | 300 | 153,600 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 3 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 3 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 4 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 4 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 4 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 4 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 4 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 4 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 4 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 4 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 4 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 4 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 4 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 4 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 4 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 4 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 4 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 4 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 4 | 512 | 300 | 153,600 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 4 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 4 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 4 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 4 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 5 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 5 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 5 | 16,384 | 486 | 7,962,624 | 
| app_proof | 11 | KeccakfPermAir | prover | 5 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 5 | 16,384 | 741 | 12,140,544 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 5 | 65,536 | 25 | 1,638,400 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 5 | 131,072 | 32 | 4,194,304 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 5 | 65,536 | 23 | 1,507,328 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 5 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 5 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 5 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 5 | 256 | 20 | 5,120 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 5 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 5 | 65,536 | 77 | 5,046,272 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 5 | 32,768 | 49 | 1,605,632 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 5 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 5 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 5 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 5 | 512 | 32 | 16,384 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 5 | 16,384 | 47 | 770,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 5 | 1,048,576 | 3 | 3,145,728 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 39 | 524 | 38 | 13 | 4 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 13 | 193 | 13 | 2 | 1 | 2 | 1 | 2 | 
| internal_recursive.1 | 1 | 9 | 152 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 185 | 1,463 | 185 | 64 | 33 | 6 | 7 | 7 | 
| leaf | 1 | 93 | 793 | 93 | 32 | 16 | 5 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 81,415,028 | 485 | 248 | 0 | 14 | 178 | 49 | 48 | 92 | 36 | 0 | 57 | 18 | 39 | 7 | 31 | 248 | 178 | 0 | 2 | 35 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,879 | 179 | 52 | 0 | 4 | 77 | 24 | 23 | 29 | 24 | 0 | 49 | 24 | 24 | 1 | 22 | 52 | 77 | 0 | 2 | 22 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 142 | 34 | 0 | 4 | 66 | 23 | 22 | 24 | 18 | 0 | 40 | 18 | 22 | 1 | 21 | 35 | 66 | 0 | 2 | 17 | 0 | 0 | 
| leaf | 0 | prover | 476,578,662 | 1,277 | 731 | 0 | 48 | 419 | 204 | 204 | 124 | 89 | 0 | 125 | 38 | 87 | 41 | 46 | 731 | 419 | 0 | 6 | 88 | 0 | 0 | 
| leaf | 1 | prover | 240,255,732 | 699 | 379 | 0 | 24 | 235 | 109 | 108 | 68 | 56 | 0 | 83 | 31 | 52 | 20 | 32 | 380 | 235 | 0 | 5 | 56 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 7,020,739 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,310 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 28,064,581 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 15,736,387 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 857,743,372 | 2,151 | 884 | 0 | 262 | 1,079 | 511 | 510 | 332 | 235 | 66 | 187 | 41 | 146 | 84 | 61 | 884 | 1,079 | 0 | 3 | 168 | 0 | 0 | 
| app_proof | prover | 1 | 857,820,172 | 1,872 | 683 | 0 | 59 | 999 | 519 | 518 | 309 | 170 | 0 | 189 | 43 | 145 | 85 | 60 | 683 | 999 | 0 | 3 | 169 | 0 | 0 | 
| app_proof | prover | 2 | 857,820,172 | 1,873 | 684 | 0 | 59 | 1,003 | 524 | 523 | 308 | 170 | 0 | 185 | 40 | 144 | 84 | 60 | 684 | 1,003 | 0 | 3 | 169 | 0 | 0 | 
| app_proof | prover | 3 | 857,820,172 | 1,867 | 683 | 0 | 59 | 995 | 516 | 515 | 308 | 170 | 0 | 188 | 41 | 146 | 84 | 62 | 683 | 995 | 0 | 3 | 170 | 0 | 0 | 
| app_proof | prover | 4 | 857,820,172 | 1,870 | 683 | 0 | 59 | 1,000 | 520 | 519 | 309 | 170 | 0 | 185 | 39 | 146 | 85 | 60 | 683 | 1,000 | 0 | 3 | 169 | 0 | 0 | 
| app_proof | prover | 5 | 857,743,372 | 1,875 | 684 | 0 | 58 | 1,002 | 519 | 519 | 310 | 171 | 0 | 188 | 41 | 146 | 84 | 61 | 684 | 1,002 | 0 | 3 | 171 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 64,838,666 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 64,838,922 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 64,838,922 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 64,838,922 | 2,013,265,921 | 
| app_proof | prover | 4 | 0 | 64,838,922 | 2,013,265,921 | 
| app_proof | prover | 5 | 0 | 64,838,666 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 156 | 2,560 | 156 | 162 | 0 | 3 | 89 | 2,413,000 | 31.66 | 
| app_proof | 1 | 34 | 2,296 | 34 | 41 | 0 | 1 | 348 | 2,413,000 | 35.50 | 
| app_proof | 2 | 34 | 2,233 | 34 | 41 | 0 | 1 | 285 | 2,413,000 | 35.64 | 
| app_proof | 3 | 34 | 2,231 | 34 | 41 | 0 | 1 | 288 | 2,413,000 | 35.46 | 
| app_proof | 4 | 34 | 2,225 | 34 | 41 | 0 | 1 | 279 | 2,413,000 | 35.94 | 
| app_proof | 5 | 33 | 2,192 | 33 | 41 | 0 | 1 | 241 | 2,300,133 | 35.77 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/5eaac821d715bc18aa96b2756d288cfcd44b2ddd

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25885138404)
