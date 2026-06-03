| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  17.34 |  5.06 |  5.06 |
| app_proof |  14.17 |  2.66 |  2.66 |
| leaf |  2.37 |  1.60 |  1.60 |
| internal_for_leaf |  0.50 |  0.50 |  0.50 |
| internal_recursive.0 |  0.17 |  0.17 |  0.17 |
| internal_recursive.1 |  0.13 |  0.13 |  0.13 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,337.67 |  14,026 |  2,516 |  2,246 |
| `execute_metered_time_ms` |  143 | -          | -          | -          |
| `execute_metered_insns` |  14,365,133 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  100.26 | -          |  100.26 |  100.26 |
| `execute_preflight_insns` |  2,394,188.83 |  14,365,133 |  2,413,000 |  2,300,133 |
| `execute_preflight_time_ms` |  240.33 |  1,442 |  322 |  90 |
| `execute_preflight_insn_mi/s` |  32.70 | -          |  33.71 |  30.07 |
| `trace_gen_time_ms   ` |  36.83 |  221 |  51 |  34 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,002.67 |  12,016 |  2,239 |  1,936 |
| `prover.main_trace_commit_time_ms` |  699.17 |  4,195 |  858 |  666 |
| `prover.rap_constraints_time_ms` |  1,121.83 |  6,731 |  1,195 |  1,087 |
| `prover.openings_time_ms` |  180.83 |  1,085 |  185 |  178 |
| `prover.rap_constraints.logup_gkr_time_ms` |  290.67 |  1,744 |  352 |  262 |
| `prover.rap_constraints.round0_time_ms` |  513.33 |  3,080 |  517 |  506 |
| `prover.rap_constraints.mle_rounds_time_ms` |  316.33 |  1,898 |  335 |  307 |
| `prover.openings.stacked_reduction_time_ms` |  137.17 |  823 |  138 |  136 |
| `prover.openings.stacked_reduction.round0_time_ms` |  82.83 |  497 |  83 |  82 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  53.83 |  323 |  55 |  53 |
| `prover.openings.whir_time_ms` |  43.17 |  259 |  48 |  40 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,185 |  2,370 |  1,601 |  769 |
| `execute_preflight_time_ms` |  5 |  10 |  5 |  5 |
| `trace_gen_time_ms   ` |  134 |  268 |  178 |  90 |
| `generate_blob_total_time_ms` |  24 |  48 |  32 |  16 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,050.50 |  2,101 |  1,423 |  678 |
| `prover.main_trace_commit_time_ms` |  635 |  1,270 |  894 |  376 |
| `prover.rap_constraints_time_ms` |  322 |  644 |  414 |  230 |
| `prover.openings_time_ms` |  92 |  184 |  113 |  71 |
| `prover.rap_constraints.logup_gkr_time_ms` |  68 |  136 |  85 |  51 |
| `prover.rap_constraints.round0_time_ms` |  157.50 |  315 |  205 |  110 |
| `prover.rap_constraints.mle_rounds_time_ms` |  95.50 |  191 |  123 |  68 |
| `prover.openings.stacked_reduction_time_ms` |  54.50 |  109 |  72 |  37 |
| `prover.openings.stacked_reduction.round0_time_ms` |  30.50 |  61 |  41 |  20 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  23.50 |  47 |  30 |  17 |
| `prover.openings.whir_time_ms` |  37 |  74 |  41 |  33 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  502 |  502 |  502 |  502 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  40 |  40 |  40 |  40 |
| `generate_blob_total_time_ms` |  4 |  4 |  4 |  4 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  461 |  461 |  461 |  461 |
| `prover.main_trace_commit_time_ms` |  244 |  244 |  244 |  244 |
| `prover.rap_constraints_time_ms` |  173 |  173 |  173 |  173 |
| `prover.openings_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.round0_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.mle_rounds_time_ms` |  92 |  92 |  92 |  92 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  15 |  15 |  15 |  15 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  165 |  165 |  165 |  165 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  13 |  13 |  13 |  13 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  151 |  151 |  151 |  151 |
| `prover.main_trace_commit_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints_time_ms` |  73 |  73 |  73 |  73 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  134 |  134 |  134 |  134 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  124 |  124 |  124 |  124 |
| `prover.main_trace_commit_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/1ef15de87cad201df013dacfe95250c6fb7c14ef/keccak-1ef15de87cad201df013dacfe95250c6fb7c14ef.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 14.32 | app_proof.prover.0 |
| prover.openings | 12.97 | app_proof.prover.0 |
| prover.merkle_tree | 12.97 | app_proof.prover.0 |
| prover.prove_whir_opening | 12.97 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 12.85 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 12.85 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 12.72 | app_proof.prover.0 |
| prover.gkr_input_evals | 12.72 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 12.46 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 12.46 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 9.85 | app_proof.prover.0 |
| prover.stacked_commit | 9.85 | app_proof.prover.0 |
| prover.rs_code_matrix | 9.83 | app_proof.prover.0 |
| generate mem proving ctxs | 2.84 | app_proof.0 |
| set initial memory | 2.84 | app_proof.1 |
| tracegen.exp_bits_len | 2.28 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.28 | leaf.0 |
| tracegen.pow_checker | 2.28 | leaf.0 |
| tracegen.whir_folding | 2.03 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 2.02 | leaf.0 |
| tracegen.whir_initial_opened_values | 2.02 | leaf.0 |
| tracegen.public_values | 1.67 | leaf.0 |
| tracegen.range_checker | 1.67 | leaf.0 |
| tracegen.proof_shape | 1.67 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 135 | 9 | 0 | 267,143 | 229,659 | 9 | 68 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | KeccakfOpAir | 110 | 27 | 2 | 
| 11 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 12 | XorinVmAir | 357 | 92 | 3 | 
| 13 | Rv64HintStoreAir | 23 | 15 | 3 | 
| 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 18 | 9 | 3 | 
| 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 21 | 37 | 3 | 
| 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 19 | 15 | 3 | 
| 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 26 | 22 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 33 | 31 | 3 | 
| 21 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 29 | 77 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 38 | 180 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 26 | 44 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 25 | 23 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 32 | 34 | 3 | 
| 26 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 27 | PhantomAir | 3 |  | 1 | 
| 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 29 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 71 | 9 |  |  | 0 |  | 347 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 2,246 |  |  | 143 | 14,365,133 | 100.26 | 0 | 14,177 |  | 
| internal_for_leaf |  |  |  | 502 |  |  |  |  |  |  |  |  |  | 502 | 
| internal_recursive.0 |  |  |  | 165 |  |  |  |  |  |  |  |  |  | 165 | 
| internal_recursive.1 |  |  |  | 134 |  |  |  |  |  |  |  |  |  | 134 | 
| leaf |  |  | 769 |  |  |  |  |  |  |  |  |  |  | 2,370 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 2 | 
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
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 2 | 
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
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 2 | 2 | 
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
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 3 | 2 | 
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
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 4 | 2 | 
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
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 5 | 2 | 
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
| app_proof | 10 | KeccakfOpAir | prover | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 0 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 1 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 1 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 1 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 1 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 1 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 1 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 2 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 2 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 2 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 2 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 2 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 2 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 2 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 2 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 2 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 2 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 2 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 2 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 2 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 2 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 3 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 3 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 3 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 3 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 3 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 3 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 3 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 3 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 3 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 3 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 3 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 3 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 3 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 3 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 3 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 4 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 4 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 4 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 4 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 4 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 4 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 4 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 4 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 4 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 4 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 4 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 4 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 4 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 4 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 4 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 4 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 4 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 4 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 4 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 4 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 4 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 5 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 5 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 5 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover | 5 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover | 5 | 16,384 | 669 | 10,960,896 | 
| app_proof | 14 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 5 | 65,536 | 21 | 1,376,256 | 
| app_proof | 15 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 5 | 131,072 | 28 | 3,670,016 | 
| app_proof | 16 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 5 | 65,536 | 19 | 1,245,184 | 
| app_proof | 17 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 5 | 65,536 | 44 | 2,883,584 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 5 | 262,144 | 38 | 9,961,472 | 
| app_proof | 19 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 5 | 16,384 | 48 | 786,432 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 5 | 128 | 21 | 2,688 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 5 | 1,048,576 | 56 | 58,720,256 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 5 | 65,536 | 73 | 4,784,128 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 5 | 32,768 | 45 | 1,474,560 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 5 | 1,048,576 | 48 | 50,331,648 | 
| app_proof | 26 | BitwiseOperationLookupAir<8> | prover | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 5 | 256 | 300 | 76,800 | 
| app_proof | 29 | VariableRangeCheckerAir | prover | 5 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 5 | 256 | 32 | 8,192 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 5 | 16,384 | 43 | 704,512 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 5 | 1,048,576 | 3 | 3,145,728 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 40 | 502 | 40 | 13 | 4 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 13 | 165 | 13 | 2 | 1 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 134 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 178 | 1,601 | 178 | 63 | 32 | 5 | 7 | 7 | 
| leaf | 1 | 90 | 769 | 90 | 32 | 16 | 5 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 81,415,028 | 461 | 244 | 0 | 1 | 173 | 49 | 49 | 92 | 30 | 0 | 43 | 19 | 23 | 7 | 15 | 244 | 173 | 0 | 3 | 29 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,879 | 151 | 49 | 0 | 0 | 73 | 25 | 25 | 29 | 17 | 0 | 28 | 19 | 8 | 1 | 6 | 49 | 73 | 0 | 2 | 16 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 124 | 31 | 0 | 0 | 63 | 24 | 24 | 24 | 13 | 0 | 29 | 22 | 7 | 1 | 5 | 32 | 63 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 476,578,662 | 1,423 | 893 | 0 | 174 | 414 | 205 | 205 | 123 | 85 | 0 | 113 | 41 | 72 | 41 | 30 | 894 | 414 | 0 | 7 | 84 | 0 | 0 | 
| leaf | 1 | prover | 240,255,732 | 678 | 375 | 0 | 1 | 230 | 110 | 109 | 68 | 51 | 0 | 71 | 33 | 37 | 20 | 17 | 376 | 230 | 0 | 6 | 50 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 7,020,739 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,310 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 28,064,581 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 15,736,387 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 847,541,900 | 2,239 | 857 | 0 | 193 | 1,195 | 506 | 506 | 335 | 352 | 97 | 185 | 48 | 137 | 83 | 53 | 858 | 1,195 | 0 | 4 | 254 | 0 | 0 | 
| app_proof | prover | 1 | 847,541,900 | 1,954 | 666 | 0 | 3 | 1,108 | 515 | 514 | 330 | 262 | 0 | 179 | 42 | 137 | 83 | 54 | 666 | 1,108 | 0 | 4 | 260 | 0 | 0 | 
| app_proof | prover | 2 | 847,541,900 | 2,010 | 666 | 0 | 3 | 1,165 | 511 | 510 | 310 | 343 | 100 | 178 | 41 | 136 | 82 | 53 | 667 | 1,165 | 0 | 4 | 241 | 0 | 0 | 
| app_proof | prover | 3 | 847,541,900 | 1,939 | 667 | 0 | 3 | 1,088 | 515 | 514 | 309 | 263 | 0 | 183 | 45 | 137 | 83 | 54 | 667 | 1,088 | 0 | 4 | 261 | 0 | 0 | 
| app_proof | prover | 4 | 847,541,900 | 1,936 | 668 | 0 | 3 | 1,088 | 517 | 516 | 307 | 262 | 0 | 179 | 40 | 138 | 83 | 54 | 668 | 1,088 | 0 | 4 | 261 | 0 | 0 | 
| app_proof | prover | 5 | 847,541,900 | 1,938 | 669 | 0 | 3 | 1,087 | 516 | 516 | 307 | 262 | 0 | 181 | 43 | 138 | 83 | 55 | 669 | 1,087 | 0 | 4 | 260 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 92,182,282 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 92,182,282 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 92,182,282 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 92,182,282 | 2,013,265,921 | 
| app_proof | prover | 4 | 0 | 92,182,282 | 2,013,265,921 | 
| app_proof | prover | 5 | 0 | 92,182,282 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 51 | 2,516 | 51 | 135 | 0 | 3 | 90 | 2,413,000 | 30.07 | 
| app_proof | 1 | 34 | 2,353 | 34 | 41 | 0 | 1 | 322 | 2,413,000 | 33.28 | 
| app_proof | 2 | 34 | 2,354 | 34 | 41 | 0 | 1 | 268 | 2,413,000 | 32.82 | 
| app_proof | 3 | 34 | 2,281 | 34 | 41 | 0 | 1 | 266 | 2,413,000 | 32.84 | 
| app_proof | 4 | 34 | 2,276 | 34 | 41 | 0 | 1 | 264 | 2,413,000 | 33.47 | 
| app_proof | 5 | 34 | 2,246 | 34 | 41 | 0 | 1 | 232 | 2,300,133 | 33.71 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/1ef15de87cad201df013dacfe95250c6fb7c14ef

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26879453370)
