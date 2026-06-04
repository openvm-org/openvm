| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.29 |  4.38 |  4.38 |
| app_proof |  3.79 |  2.88 |  2.88 |
| leaf |  0.94 |  0.94 |  0.94 |
| internal_for_leaf |  0.28 |  0.28 |  0.28 |
| internal_recursive.0 |  0.15 |  0.15 |  0.15 |
| internal_recursive.1 |  0.13 |  0.13 |  0.13 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,880.50 |  3,761 |  2,856 |  905 |
| `execute_metered_time_ms` |  25 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  79.01 | -          |  79.01 |  79.01 |
| `execute_preflight_insns` |  989,985.50 |  1,979,971 |  1,445,000 |  534,971 |
| `execute_preflight_time_ms` |  75 |  150 |  82 |  68 |
| `execute_preflight_insn_mi/s` |  30.71 | -          |  30.96 |  30.47 |
| `trace_gen_time_ms   ` |  66 |  132 |  94 |  38 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,651.50 |  3,303 |  2,559 |  744 |
| `prover.main_trace_commit_time_ms` |  590 |  1,180 |  955 |  225 |
| `prover.rap_constraints_time_ms` |  925 |  1,850 |  1,404 |  446 |
| `prover.openings_time_ms` |  134.50 |  269 |  198 |  71 |
| `prover.rap_constraints.logup_gkr_time_ms` |  154 |  308 |  219 |  89 |
| `prover.rap_constraints.round0_time_ms` |  553.50 |  1,107 |  856 |  251 |
| `prover.rap_constraints.mle_rounds_time_ms` |  217 |  434 |  329 |  105 |
| `prover.openings.stacked_reduction_time_ms` |  101 |  202 |  157 |  45 |
| `prover.openings.stacked_reduction.round0_time_ms` |  56.50 |  113 |  88 |  25 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  44.50 |  89 |  69 |  20 |
| `prover.openings.whir_time_ms` |  33 |  66 |  40 |  26 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  940 |  940 |  940 |  940 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  115 |  115 |  115 |  115 |
| `generate_blob_total_time_ms` |  10 |  10 |  10 |  10 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  824 |  824 |  824 |  824 |
| `prover.main_trace_commit_time_ms` |  399 |  399 |  399 |  399 |
| `prover.rap_constraints_time_ms` |  346 |  346 |  346 |  346 |
| `prover.openings_time_ms` |  78 |  78 |  78 |  78 |
| `prover.rap_constraints.logup_gkr_time_ms` |  135 |  135 |  135 |  135 |
| `prover.rap_constraints.round0_time_ms` |  133 |  133 |  133 |  133 |
| `prover.rap_constraints.mle_rounds_time_ms` |  76 |  76 |  76 |  76 |
| `prover.openings.stacked_reduction_time_ms` |  40 |  40 |  40 |  40 |
| `prover.openings.stacked_reduction.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.whir_time_ms` |  38 |  38 |  38 |  38 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  281 |  281 |  281 |  281 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  21 |  21 |  21 |  21 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  260 |  260 |  260 |  260 |
| `prover.main_trace_commit_time_ms` |  117 |  117 |  117 |  117 |
| `prover.rap_constraints_time_ms` |  107 |  107 |  107 |  107 |
| `prover.openings_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints.logup_gkr_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.round0_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints.mle_rounds_time_ms` |  52 |  52 |  52 |  52 |
| `prover.openings.stacked_reduction_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  152 |  152 |  152 |  152 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  141 |  141 |  141 |  141 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  68 |  68 |  68 |  68 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  130 |  130 |  130 |  130 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  120 |  120 |  120 |  120 |
| `prover.main_trace_commit_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/1f998065deae7e89b008df7fa4287b0acbd7bf6b/kitchen_sink-1f998065deae7e89b008df7fa4287b0acbd7bf6b.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 15.08 | app_proof.prover.0 |
| prover.openings | 14.15 | app_proof.prover.0 |
| prover.merkle_tree | 14.15 | app_proof.prover.0 |
| prover.prove_whir_opening | 14.15 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 12.89 | app_proof.prover.0 |
| prover.gkr_input_evals | 12.89 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 12.89 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 12.89 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 12.70 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 12.70 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 10.94 | app_proof.prover.0 |
| prover.stacked_commit | 10.94 | app_proof.prover.0 |
| prover.rs_code_matrix | 10.93 | app_proof.prover.0 |
| generate mem proving ctxs | 3.32 | app_proof.0 |
| set initial memory | 3.32 | app_proof.1 |
| tracegen.pow_checker | 1.84 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.84 | leaf.0 |
| tracegen.exp_bits_len | 1.84 | leaf.0 |
| tracegen.whir_folding | 1.71 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.71 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.71 | leaf.0 |
| tracegen.public_values | 1.62 | leaf.0 |
| tracegen.range_checker | 1.62 | leaf.0 |
| tracegen.proof_shape | 1.62 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 651 | 9 | 0 | 267,175 | 229,939 | 9 | 61 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 548 | 286 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 356 | 190 | 3 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 372 | 194 | 3 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 244 | 130 | 3 | 
| 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 219 | 117 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 155 | 85 | 3 | 
| 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 67 | 170 | 3 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 283 | 171 | 3 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 187 | 123 | 3 | 
| 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 35 | 3 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 881 | 513 | 3 | 
| 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 51 | 118 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 195 | 117 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 131 | 85 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 148 | 2,127 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 130 | 16 | 2 | 
| 45 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> | 48 | 69 | 3 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 45 | 31 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 69 | 71 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 130 | 85 | 3 | 
| 49 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 741 | 381 | 2 | 
| 50 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 51 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 52 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 54 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| 55 | Sha2MainAir<Sha512Config> | 149 | 39 | 3 | 
| 56 | Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,481 | 3 | 
| 57 | Sha2MainAir<Sha256Config> | 85 | 23 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha256Config> | 29 | 754 | 3 | 
| 59 | KeccakfOpAir | 110 | 27 | 2 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 60 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 61 | XorinVmAir | 357 | 92 | 3 | 
| 62 | Rv64HintStoreAir | 17 | 15 | 3 | 
| 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 14 | 5 | 3 | 
| 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 15 | 9 | 3 | 
| 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 12 | 11 | 2 | 
| 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 14 | 25 | 3 | 
| 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 26 | 22 | 3 | 
| 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 33 | 31 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 
| 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 29 | 77 | 3 | 
| 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 38 | 180 | 3 | 
| 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 19 | 30 | 3 | 
| 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 25 | 23 | 3 | 
| 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 32 | 34 | 3 | 
| 75 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 76 | PhantomAir | 3 | 1 | 2 | 
| 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 78 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 464 | 280 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 501 | 257 | 2 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 82 | 9 |  |  | 0 |  | 402 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 905 |  |  | 25 | 1,979,971 | 79.01 | 0 | 3,791 |  | 
| internal_for_leaf |  |  |  | 281 |  |  |  |  |  |  |  |  |  | 281 | 
| internal_recursive.0 |  |  |  | 152 |  |  |  |  |  |  |  |  |  | 152 | 
| internal_recursive.1 |  |  |  | 130 |  |  |  |  |  |  |  |  |  | 130 | 
| leaf |  |  | 940 |  |  |  |  |  |  |  |  |  |  | 940 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 11 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 6 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 14 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 3 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 4 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 78 | 89 | 4 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 32,768 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 49 | 12,544 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 52 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 2,944 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 46,080 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 20 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 2,064,384 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 1,728 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 3,648 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 143,360 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 31,457,280 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 816 | 
| leaf | 31 | EqBitsAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 368 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 1,216 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 32,768 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 23,330,816 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 229,376 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 507,904 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 69,632 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 23,592,960 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 757 | 1,514 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 565 | 1,130 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 513 | 1,026 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 385 | 770 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 4 | 116 | 464 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 292 | 18,688 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 228 | 14,592 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 4,096 | 21 | 86,016 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover | 0 | 8 | 160 | 1,280 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 2 | 390 | 780 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 4 | 294 | 1,176 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 32 | 131,072 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,004 | 4,016 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 0 | 512 | 246 | 125,952 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 169 | 43,264 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover | 0 | 256 | 90 | 23,040 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover | 0 | 256 | 126 | 32,256 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 173 | 177,152 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 1 | 949 | 949 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 150 | 2,457,600 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 119,537,664 | 
| app_proof | 59 | KeccakfOpAir | prover | 0 | 8,192 | 284 | 2,326,528 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 60 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 61 | XorinVmAir | prover | 0 | 8,192 | 669 | 5,480,448 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 32,768 | 17 | 557,056 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 65,536 | 24 | 1,572,864 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 131,072 | 26 | 3,407,872 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 32,768 | 46 | 1,507,328 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 524,288 | 54 | 28,311,552 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 512 | 58 | 29,696 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 65,536 | 73 | 4,784,128 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 512 | 38 | 19,456 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 16,384 | 41 | 671,744 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 524,288 | 48 | 25,165,824 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 2,048 | 300 | 614,400 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 2,048 | 21 | 43,008 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 2,048 | 32 | 65,536 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 1 | 64 | 246 | 15,744 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 32 | 169 | 5,408 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover | 1 | 32 | 90 | 2,880 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover | 1 | 64 | 126 | 8,064 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 128 | 173 | 22,144 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 1 | 32 | 43 | 1,376 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 1 | 8,192 | 150 | 1,228,800 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 131,072 | 456 | 59,768,832 | 
| app_proof | 59 | KeccakfOpAir | prover | 1 | 4,096 | 284 | 1,163,264 | 
| app_proof | 60 | KeccakfPermAir | prover | 1 | 65,536 | 2,634 | 172,621,824 | 
| app_proof | 61 | XorinVmAir | prover | 1 | 4,096 | 669 | 2,740,224 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 16,384 | 17 | 278,528 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 32,768 | 24 | 786,432 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 32,768 | 18 | 589,824 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 32,768 | 32 | 1,048,576 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 65,536 | 26 | 1,703,936 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 8,192 | 46 | 376,832 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 262,144 | 54 | 14,155,776 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 64 | 58 | 3,712 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 16,384 | 73 | 1,196,032 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 1 | 64 | 38 | 2,432 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 8,192 | 41 | 335,872 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 262,144 | 48 | 12,582,912 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 2,048 | 300 | 614,400 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 21 | 281 | 20 | 6 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 10 | 152 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 130 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 115 | 940 | 115 | 24 | 10 | 18 | 2 | 2 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,979 | 260 | 117 | 0 | 0 | 107 | 34 | 34 | 52 | 20 | 0 | 35 | 21 | 13 | 3 | 9 | 117 | 107 | 0 | 2 | 19 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,767 | 141 | 46 | 0 | 0 | 68 | 25 | 25 | 27 | 14 | 0 | 27 | 18 | 8 | 1 | 6 | 46 | 68 | 0 | 2 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,863 | 120 | 31 | 0 | 0 | 63 | 24 | 24 | 24 | 13 | 0 | 25 | 18 | 7 | 1 | 5 | 31 | 63 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 260,613,236 | 824 | 399 | 0 | 1 | 346 | 133 | 133 | 76 | 135 | 0 | 78 | 38 | 40 | 21 | 18 | 399 | 346 | 0 | 7 | 133 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,586 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,382 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,358 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 39,253,315 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 902,782,844 | 2,559 | 955 | 0 | 305 | 1,404 | 856 | 854 | 329 | 219 | 67 | 198 | 40 | 157 | 88 | 69 | 955 | 1,404 | 0 | 3 | 151 | 0 | 0 | 
| app_proof | prover | 1 | 279,963,980 | 744 | 225 | 0 | 1 | 446 | 251 | 250 | 105 | 89 | 0 | 71 | 26 | 45 | 25 | 20 | 225 | 446 | 0 | 3 | 88 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 60,351,486 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 29,472,010 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 94 | 2,856 | 94 | 134 | 0 | 3 | 68 | 1,445,000 | 30.47 | 
| app_proof | 1 | 38 | 905 | 38 | 41 | 0 | 2 | 82 | 534,971 | 30.96 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/1f998065deae7e89b008df7fa4287b0acbd7bf6b

Max Segment Length: 4194304

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26959801658)
