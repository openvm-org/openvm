| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  2.87 |  2.13 |  2.13 |
| app_proof |  2.07 |  1.32 |  1.32 |
| leaf |  0.43 |  0.43 |  0.43 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.11 |  0.11 |  0.11 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,021.50 |  2,043 |  1,300 |  743 |
| `execute_metered_time_ms` |  23 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  83.91 | -          |  83.91 |  83.91 |
| `execute_preflight_insns` |  989,985.50 |  1,979,971 |  1,168,000 |  811,971 |
| `execute_preflight_time_ms` |  89 |  178 |  125 |  53 |
| `execute_preflight_insn_mi/s` |  32.19 | -          |  33.07 |  31.31 |
| `trace_gen_time_ms   ` |  30.50 |  61 |  45 |  16 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  791 |  1,582 |  1,022 |  560 |
| `prover.main_trace_commit_time_ms` |  226.50 |  453 |  304 |  149 |
| `prover.rap_constraints_time_ms` |  499.50 |  999 |  642 |  357 |
| `prover.openings_time_ms` |  63.50 |  127 |  74 |  53 |
| `prover.rap_constraints.logup_gkr_time_ms` |  103.50 |  207 |  143 |  64 |
| `prover.rap_constraints.round0_time_ms` |  283.50 |  567 |  352 |  215 |
| `prover.rap_constraints.mle_rounds_time_ms` |  111.50 |  223 |  146 |  77 |
| `prover.openings.stacked_reduction_time_ms` |  50 |  100 |  62 |  38 |
| `prover.openings.stacked_reduction.round0_time_ms` |  14.50 |  29 |  15 |  14 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  35 |  70 |  46 |  24 |
| `prover.openings.whir_time_ms` |  13 |  26 |  14 |  12 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  428 |  428 |  428 |  428 |
| `execute_preflight_time_ms` |  19 |  19 |  19 |  19 |
| `trace_gen_time_ms   ` |  109 |  109 |  109 |  109 |
| `generate_blob_total_time_ms` |  11 |  11 |  11 |  11 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  318 |  318 |  318 |  318 |
| `prover.main_trace_commit_time_ms` |  131 |  131 |  131 |  131 |
| `prover.rap_constraints_time_ms` |  144 |  144 |  144 |  144 |
| `prover.openings_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  57 |  57 |  57 |  57 |
| `prover.rap_constraints.round0_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.whir_time_ms` |  12 |  12 |  12 |  12 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  159 |  159 |  159 |  159 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  140 |  140 |  140 |  140 |
| `prover.main_trace_commit_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints_time_ms` |  65 |  65 |  65 |  65 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  10 |  10 |  10 |  10 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  113 |  113 |  113 |  113 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  103 |  103 |  103 |  103 |
| `prover.main_trace_commit_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  7 |  7 |  7 |  7 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  108 |  108 |  108 |  108 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  99 |  99 |  99 |  99 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  8 |  8 |  8 |  8 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/4bca1c4d7077460540e1c0b3df833a7aaba86d11/kitchen_sink-4bca1c4d7077460540e1c0b3df833a7aaba86d11.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 10.23 | app_proof.prover.0 |
| prover.merkle_tree | 8.94 | app_proof.prover.0 |
| prover.prove_whir_opening | 8.94 | app_proof.prover.0 |
| prover.openings | 8.94 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 8.65 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 8.65 | app_proof.prover.0 |
| prover.gkr_input_evals | 8.58 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 8.58 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 8.49 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 8.49 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 7.17 | app_proof.prover.0 |
| prover.stacked_commit | 7.17 | app_proof.prover.0 |
| prover.rs_code_matrix | 7.16 | app_proof.prover.0 |
| generate mem proving ctxs | 5.32 | app_proof.0 |
| set initial memory | 5.32 | app_proof.1 |
| tracegen.pow_checker | 1.86 | leaf.0 |
| tracegen.exp_bits_len | 1.86 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.86 | leaf.0 |
| tracegen.whir_folding | 1.73 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.73 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.73 | leaf.0 |
| tracegen.proof_shape | 1.63 | leaf.0 |
| tracegen.range_checker | 1.63 | leaf.0 |
| tracegen.public_values | 1.63 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 690 | 3 | 0 | 267,175 | 228,352 | 3 | 62 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | 547 | 286 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | 355 | 190 | 3 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 371 | 194 | 3 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 243 | 130 | 3 | 
| 16 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 218 | 117 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 154 | 85 | 3 | 
| 19 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 2 | PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 22 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 6, 8, 48>, ModularIsEqualCoreAir<48, 8, 8> | 65 | 314 | 3 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6, 8, 8>, FieldExpressionCoreAir> | 282 | 171 | 3 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6, 8, 8>, FieldExpressionCoreAir> | 186 | 123 | 3 | 
| 25 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 28 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 31 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12, 8, 8>, FieldExpressionCoreAir> | 880 | 513 | 3 | 
| 40 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 130 | 85 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 115 | 2,127 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 97 | 16 | 2 | 
| 45 | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchLessThanCoreAir<32, 8> | 46 | 117 | 3 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchEqualCoreAir<32> | 44 | 47 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, LessThanCoreAir<32, 8> | 67 | 119 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 97 | 85 | 3 | 
| 49 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 26 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | 740 | 381 | 2 | 
| 50 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 33 | 104 | 3 | 
| 51 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 32 | 11 | 2 | 
| 52 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 20 | 5 | 2 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 23 | 4 | 2 | 
| 54 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| 55 | Sha2MainAir<Sha512Config> | 148 | 39 | 3 | 
| 56 | Sha2BlockHasherVmAir<Sha512Config> | 53 | 1,480 | 3 | 
| 57 | Sha2MainAir<Sha256Config> | 84 | 23 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha256Config> | 29 | 753 | 3 | 
| 59 | KeccakfOpAir | 210 | 27 | 2 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| 60 | KeccakfPermAir | 2 | 3,183 | 3 | 
| 61 | XorinVmAir | 356 | 92 | 3 | 
| 62 | Rv64HintStoreAir | 20 | 15 | 3 | 
| 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 16 | 9 | 3 | 
| 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 13 | 37 | 3 | 
| 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 11 | 15 | 3 | 
| 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 
| 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| 72 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 18 | 44 | 3 | 
| 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| 75 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 76 | PhantomAir | 3 |  | 1 | 
| 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 78 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 71 | 3 |  |  | 0 |  | 341 | 3 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 743 |  |  | 23 | 1,979,971 | 83.91 | 0 | 2,071 |  | 
| internal_for_leaf |  |  |  | 159 |  |  |  |  |  |  |  |  |  | 159 | 
| internal_recursive.0 |  |  |  | 113 |  |  |  |  |  |  |  |  |  | 114 | 
| internal_recursive.1 |  |  |  | 108 |  |  |  |  |  |  |  |  |  | 108 | 
| leaf |  |  | 428 |  |  |  |  |  |  |  |  |  |  | 428 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 3 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 6, 8, 48>, ModularIsEqualCoreAir<48, 8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, LessThanCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchEqualCoreAir<32> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 3 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, LessThanCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchEqualCoreAir<32> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 7 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 77 | 86 | 4 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 138 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 48 | 12,288 | 
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
| leaf | 31 | EqBitsAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
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
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 583 | 2,332 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 679 | 679 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 811 | 1,622 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 619 | 1,238 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 551 | 1,102 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 423 | 846 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 4 | 188 | 752 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 64 | 314 | 20,096 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 64 | 250 | 16,000 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 2 | 188 | 376 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 20 | 163,840 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 226 | 452 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 6, 8, 48>, ModularIsEqualCoreAir<48, 8, 8> | prover | 0 | 8 | 264 | 2,112 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 420 | 840 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 324 | 1,296 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 2 | 188 | 376 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 226 | 452 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 8 | 188 | 1,504 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 32 | 262,144 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 226 | 904 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 2 | 188 | 376 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 226 | 452 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 8 | 188 | 1,504 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 226 | 904 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 2 | 188 | 376 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 226 | 452 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,056 | 4,224 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 8 | 188 | 1,504 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 290 | 580 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 226 | 904 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 0 | 512 | 268 | 137,216 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 191 | 48,896 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchEqualCoreAir<32> | prover | 0 | 256 | 142 | 36,352 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 256 | 196 | 50,176 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 195 | 199,680 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 1,003 | 1,003 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 47 | 12,032 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 236 | 3,866,624 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 119,537,664 | 
| app_proof | 59 | KeccakfOpAir | prover | 0 | 8,192 | 486 | 3,981,312 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 583 | 2,332 | 
| app_proof | 60 | KeccakfPermAir | prover | 0 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 61 | XorinVmAir | prover | 0 | 8,192 | 741 | 6,070,272 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 32,768 | 25 | 819,200 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 65,536 | 32 | 2,097,152 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 65,536 | 23 | 1,507,328 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 65,536 | 44 | 2,883,584 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 131,072 | 38 | 4,980,736 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 16,384 | 48 | 786,432 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 524,288 | 56 | 29,360,128 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 679 | 679 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 512 | 66 | 33,792 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 65,536 | 77 | 5,046,272 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | prover | 0 | 512 | 53 | 27,136 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 16,384 | 49 | 802,816 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 524,288 | 52 | 27,262,976 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 583 | 2,332 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 679 | 679 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 4,096 | 20 | 81,920 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 4,096 | 32 | 131,072 | 
| app_proof | 43 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, ShiftCoreAir<32, 8> | prover | 1 | 128 | 268 | 34,304 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 64 | 191 | 12,224 | 
| app_proof | 46 | VmAirWrapper<Rv64VecHeapBranchAdapterAir<2, 4, 8>, 2, 4, 8, 32>, BranchEqualCoreAir<32> | prover | 1 | 64 | 142 | 9,088 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, LessThanCoreAir<32, 8> | prover | 1 | 128 | 196 | 25,088 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, 2, 4, 4, 8, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 256 | 195 | 49,920 | 
| app_proof | 53 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 1 | 64 | 47 | 3,008 | 
| app_proof | 54 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 57 | Sha2MainAir<Sha256Config> | prover | 1 | 16,384 | 236 | 3,866,624 | 
| app_proof | 58 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 262,144 | 456 | 119,537,664 | 
| app_proof | 59 | KeccakfOpAir | prover | 1 | 4,096 | 486 | 1,990,656 | 
| app_proof | 60 | KeccakfPermAir | prover | 1 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 61 | XorinVmAir | prover | 1 | 4,096 | 741 | 3,035,136 | 
| app_proof | 63 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 16,384 | 25 | 409,600 | 
| app_proof | 64 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 65,536 | 32 | 2,097,152 | 
| app_proof | 65 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 65,536 | 23 | 1,507,328 | 
| app_proof | 66 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 1 | 65,536 | 44 | 2,883,584 | 
| app_proof | 67 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 1 | 131,072 | 38 | 4,980,736 | 
| app_proof | 68 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 16,384 | 48 | 786,432 | 
| app_proof | 69 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 262,144 | 56 | 14,680,064 | 
| app_proof | 70 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 128 | 66 | 8,448 | 
| app_proof | 71 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 32,768 | 77 | 2,523,136 | 
| app_proof | 72 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | prover | 1 | 128 | 53 | 6,784 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 16,384 | 49 | 802,816 | 
| app_proof | 74 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 524,288 | 52 | 27,262,976 | 
| app_proof | 75 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 77 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 4,096 | 300 | 1,228,800 | 
| app_proof | 78 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 159 | 18 | 6 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 10 | 113 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 108 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 109 | 428 | 109 | 24 | 11 | 19 | 4 | 5 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,915 | 140 | 40 | 0 | 4 | 65 | 20 | 19 | 29 | 16 | 0 | 33 | 10 | 23 | 1 | 21 | 41 | 65 | 0 | 1 | 15 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,703 | 103 | 19 | 0 | 4 | 54 | 19 | 18 | 20 | 13 | 0 | 29 | 7 | 21 | 1 | 20 | 19 | 54 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 99 | 14 | 0 | 4 | 54 | 21 | 20 | 18 | 13 | 0 | 30 | 8 | 21 | 0 | 21 | 15 | 54 | 0 | 1 | 13 | 0 | 0 | 
| leaf | 0 | prover | 263,565,172 | 318 | 130 | 0 | 10 | 144 | 49 | 48 | 37 | 57 | 0 | 43 | 12 | 31 | 7 | 24 | 131 | 144 | 0 | 2 | 56 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,522 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 40,270,659 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 565,122,580 | 1,022 | 304 | 0 | 173 | 642 | 352 | 350 | 146 | 143 | 46 | 74 | 12 | 62 | 15 | 46 | 304 | 642 | 0 | 1 | 96 | 0 | 0 | 
| app_proof | prover | 1 | 541,799,820 | 560 | 148 | 0 | 23 | 357 | 215 | 214 | 77 | 64 | 0 | 53 | 14 | 38 | 14 | 24 | 149 | 357 | 0 | 1 | 64 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 45,215,458 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 37,010,058 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 45 | 1,300 | 45 | 179 | 0 | 3 | 53 | 1,168,000 | 31.31 | 
| app_proof | 1 | 16 | 743 | 16 | 40 | 0 | 1 | 125 | 811,971 | 33.07 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/4bca1c4d7077460540e1c0b3df833a7aaba86d11

Max Segment Length: 4194304

Instance Type: g6e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25751597993)
