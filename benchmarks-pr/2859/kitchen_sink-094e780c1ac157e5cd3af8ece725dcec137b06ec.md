| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.43 |  3.80 |  3.80 |
| app_proof |  3.92 |  2.29 |  2.29 |
| leaf |  0.96 |  0.96 |  0.96 |
| internal_for_leaf |  0.28 |  0.28 |  0.28 |
| internal_recursive.0 |  0.15 |  0.15 |  0.15 |
| internal_recursive.1 |  0.13 |  0.13 |  0.13 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,939.50 |  3,879 |  2,251 |  1,628 |
| `execute_metered_time_ms` |  36 | -          | -          | -          |
| `execute_metered_insns` |  2,579,903 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  70.48 | -          |  70.48 |  70.48 |
| `execute_preflight_insns` |  1,289,951.50 |  2,579,903 |  1,540,000 |  1,039,903 |
| `execute_preflight_time_ms` |  96.50 |  193 |  130 |  63 |
| `execute_preflight_insn_mi/s` |  35.90 | -          |  36.81 |  34.99 |
| `trace_gen_time_ms   ` |  146 |  292 |  189 |  103 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,586 |  3,172 |  1,820 |  1,352 |
| `prover.main_trace_commit_time_ms` |  514.50 |  1,029 |  598 |  431 |
| `prover.rap_constraints_time_ms` |  941 |  1,882 |  1,084 |  798 |
| `prover.openings_time_ms` |  129 |  258 |  136 |  122 |
| `prover.rap_constraints.logup_gkr_time_ms` |  167.50 |  335 |  198 |  137 |
| `prover.rap_constraints.round0_time_ms` |  570 |  1,140 |  673 |  467 |
| `prover.rap_constraints.mle_rounds_time_ms` |  202.50 |  405 |  212 |  193 |
| `prover.openings.stacked_reduction_time_ms` |  94 |  188 |  101 |  87 |
| `prover.openings.stacked_reduction.round0_time_ms` |  50.50 |  101 |  51 |  50 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  43.50 |  87 |  50 |  37 |
| `prover.openings.whir_time_ms` |  34.50 |  69 |  35 |  34 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  955 |  955 |  955 |  955 |
| `execute_preflight_time_ms` |  20 |  20 |  20 |  20 |
| `trace_gen_time_ms   ` |  120 |  120 |  120 |  120 |
| `generate_blob_total_time_ms` |  9 |  9 |  9 |  9 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  834 |  834 |  834 |  834 |
| `prover.main_trace_commit_time_ms` |  405 |  405 |  405 |  405 |
| `prover.rap_constraints_time_ms` |  352 |  352 |  352 |  352 |
| `prover.openings_time_ms` |  76 |  76 |  76 |  76 |
| `prover.rap_constraints.logup_gkr_time_ms` |  138 |  138 |  138 |  138 |
| `prover.rap_constraints.round0_time_ms` |  136 |  136 |  136 |  136 |
| `prover.rap_constraints.mle_rounds_time_ms` |  77 |  77 |  77 |  77 |
| `prover.openings.stacked_reduction_time_ms` |  40 |  40 |  40 |  40 |
| `prover.openings.stacked_reduction.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.whir_time_ms` |  35 |  35 |  35 |  35 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  278 |  278 |  278 |  278 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  20 |  20 |  20 |  20 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  258 |  258 |  258 |  258 |
| `prover.main_trace_commit_time_ms` |  118 |  118 |  118 |  118 |
| `prover.rap_constraints_time_ms` |  108 |  108 |  108 |  108 |
| `prover.openings_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings.stacked_reduction_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  149 |  149 |  149 |  149 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  138 |  138 |  138 |  138 |
| `prover.main_trace_commit_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  133 |  133 |  133 |  133 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  123 |  123 |  123 |  123 |
| `prover.main_trace_commit_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints_time_ms` |  63 |  63 |  63 |  63 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/094e780c1ac157e5cd3af8ece725dcec137b06ec/kitchen_sink-094e780c1ac157e5cd3af8ece725dcec137b06ec.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 9.77 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 8.98 | app_proof.prover.0 |
| prover.gkr_input_evals | 8.98 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 8.98 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 8.98 | app_proof.prover.0 |
| prover.prove_whir_opening | 8.97 | app_proof.prover.0 |
| prover.openings | 8.97 | app_proof.prover.0 |
| prover.merkle_tree | 8.97 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 8.65 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 8.65 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 7.20 | app_proof.prover.0 |
| prover.stacked_commit | 7.20 | app_proof.prover.0 |
| prover.rs_code_matrix | 7.19 | app_proof.prover.0 |
| generate mem proving ctxs | 5.33 | app_proof.0 |
| set initial memory | 5.32 | app_proof.1 |
| tracegen.exp_bits_len | 1.86 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.86 | leaf.0 |
| tracegen.pow_checker | 1.86 | leaf.0 |
| tracegen.whir_folding | 1.73 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.73 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.73 | leaf.0 |
| tracegen.public_values | 1.64 | leaf.0 |
| tracegen.range_checker | 1.64 | leaf.0 |
| tracegen.proof_shape | 1.64 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 696 | 9 | 0 | 267,207 | 229,896 | 9 | 62 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 
| 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 691 | 322 | 3 | 
| 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 499 | 226 | 3 | 
| 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 467 | 218 | 3 | 
| 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 339 | 154 | 3 | 
| 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 266 | 129 | 3 | 
| 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 202 | 97 | 3 | 
| 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> |  | 113 | 326 | 3 | 
| 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> |  | 354 | 189 | 3 | 
| 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> |  | 258 | 141 | 3 | 
| 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 35 | 3 | 
| 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 976 | 537 | 3 | 
| 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | 81 | 222 | 3 | 
| 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 242 | 129 | 3 | 
| 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> |  | 178 | 97 | 3 | 
| 43 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, ShiftCoreAir<32, 8> |  | 163 | 2,139 | 3 | 
| 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> |  | 145 | 28 | 2 | 
| 45 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchLessThanCoreAir<32, 8> |  | 78 | 125 | 3 | 
| 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> |  | 76 | 55 | 3 | 
| 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> |  | 115 | 131 | 3 | 
| 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> |  | 145 | 97 | 3 | 
| 49 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> |  | 884 | 417 | 2 | 
| 50 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 52 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 53 | Sha2MainAir<Sha512Config> | 1 | 276 | 71 | 3 | 
| 54 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 55 | Sha2MainAir<Sha256Config> | 1 | 148 | 39 | 3 | 
| 56 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 57 | KeccakfOpAir |  | 310 | 52 | 2 | 
| 58 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 59 | XorinVmAir |  | 561 | 177 | 3 | 
| 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 60 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 
| 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 66 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 
| 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 71 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 72 | PhantomAir |  | 3 | 1 | 2 | 
| 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 74 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 527 | 296 | 3 | 
| 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> |  | 596 | 281 | 2 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 70 | 9 |  |  | 0 |  | 364 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 1,628 |  |  | 36 | 2,579,903 | 70.48 | 0 | 3,921 |  | 
| internal_for_leaf |  |  |  | 278 |  |  |  |  |  |  |  |  |  | 278 | 
| internal_recursive.0 |  |  |  | 149 |  |  |  |  |  |  |  |  |  | 149 | 
| internal_recursive.1 |  |  |  | 134 |  |  |  |  |  |  |  |  |  | 134 | 
| leaf |  |  | 955 |  |  |  |  |  |  |  |  |  |  | 955 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 10 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 | 0 | 
| app_proof | XorinVmAir | 0 | 7 | 
| app_proof | KeccakfOpAir | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 1 | 0 | 
| app_proof | XorinVmAir | 1 | 15 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 69 | 213 | 4 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 92 | 4 | 
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
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 65,536 | 44 | 2,883,584 | 
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
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 3 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 63 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 63 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 1 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | IS_EQ | 0 | 6 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | MUL | 0 | 151 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | BEQ | 0 | 151 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | SLTU | 0 | 440 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | ADD | 0 | 453 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | AND | 0 | 151 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | SUB | 0 | 151 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | XOR | 0 | 151 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | MUL | 0 | 315 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 11,476 | 
| app_proof | 57 | KeccakfOpAir | KECCAKF | 0 | 5,449 | 
| app_proof | 59 | XorinVmAir | XORIN | 0 | 5,440 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 0 | 18,688 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 0 | 37,368 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 0 | 23,174 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 0 | 2,616 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGE | 0 | 16 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 0 | 479 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 0 | 10 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 0 | 110,398 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 54,577 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 67,605 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 0 | 19,851 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 0 | 294,926 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 0 | 8,424 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 0 | 320,728 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 0 | 15,418 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SRL | 0 | 4,530 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 0 | 25,731 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 352,704 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 0 | 82,549 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 0 | 39,487 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 35,865 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | XOR | 0 | 302 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | MUL | 1 | 49 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | BEQ | 1 | 49 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | SLTU | 1 | 150 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | ADD | 1 | 147 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | AND | 1 | 49 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | SUB | 1 | 49 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | XOR | 1 | 49 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | MUL | 1 | 98 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | SHA256 | 1 | 8,624 | 
| app_proof | 57 | KeccakfOpAir | KECCAKF | 1 | 4,021 | 
| app_proof | 59 | XorinVmAir | XORIN | 1 | 4,018 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 1 | 13,183 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 1 | 26,371 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 1 | 16,908 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 1 | 788 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 1 | 147 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 1 | 2 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 1 | 79,084 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 39,055 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 48,562 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 1 | 6,451 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 1 | 200,043 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 1 | 2,704 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 1 | 210,501 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 1 | 4,998 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SRL | 1 | 1,470 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 1 | 18,107 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 1 | 247,703 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 1 | 58,310 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 1 | 22,194 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 1 | 25,920 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | XOR | 1 | 98 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 919 | 1,838 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 727 | 1,454 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 623 | 1,246 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 495 | 990 | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 4 | 208 | 832 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 350 | 22,400 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 286 | 18,304 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 21 | 172,032 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | prover | 0 | 8 | 296 | 2,368 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 474 | 948 | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 378 | 1,512 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 32 | 262,144 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 208 | 416 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 262 | 524 | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,128 | 4,512 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 208 | 1,664 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 326 | 652 | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 262 | 1,048 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 227 | 58,112 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 0 | 256 | 166 | 42,496 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 512 | 232 | 118,784 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 231 | 236,544 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 1,111 | 1,111 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 512 | 31 | 15,872 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 262,144 | 456 | 119,537,664 | 
| app_proof | 57 | KeccakfOpAir | prover | 0 | 8,192 | 561 | 4,595,712 | 
| app_proof | 58 | KeccakfPermAir | prover | 0 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 59 | XorinVmAir | prover | 0 | 8,192 | 914 | 7,487,488 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 655,360 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 65,536 | 28 | 1,835,008 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 32,768 | 18 | 589,824 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 131,072 | 26 | 3,407,872 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 32,768 | 53 | 1,736,704 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 32,768 | 37 | 1,212,416 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 524,288 | 36 | 18,874,368 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 631 | 2,524 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1 | 751 | 751 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 4,096 | 21 | 86,016 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 4,096 | 32 | 131,072 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 64 | 227 | 14,528 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | prover | 1 | 64 | 166 | 10,624 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | prover | 1 | 256 | 232 | 59,392 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 512 | 231 | 118,272 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 1 | 128 | 31 | 3,968 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | prover | 1 | 16,384 | 284 | 4,653,056 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 262,144 | 456 | 119,537,664 | 
| app_proof | 57 | KeccakfOpAir | prover | 1 | 4,096 | 561 | 2,297,856 | 
| app_proof | 58 | KeccakfPermAir | prover | 1 | 131,072 | 2,634 | 345,243,648 | 
| app_proof | 59 | XorinVmAir | prover | 1 | 4,096 | 914 | 3,743,744 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 16,384 | 20 | 327,680 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 32,768 | 28 | 917,504 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 32,768 | 18 | 589,824 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 131,072 | 32 | 4,194,304 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 131,072 | 26 | 3,407,872 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 524,288 | 41 | 21,495,808 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 8,192 | 53 | 434,176 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 32,768 | 37 | 1,212,416 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 524,288 | 36 | 18,874,368 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 4,096 | 300 | 1,228,800 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interactions | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 10,064 | 6,320 | 301,920 | 189,600 | 1,207,680 | 758,400 | 100,640 | 63,200 | 10,065 | 364,820 | 229,100 | 10,064 | 6,320 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 72 |  | 144 |  | 12 |  | 15 | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 5,679 | 1,893 | 22,716 | 7,572 | 1,893 | 631 | 2,108 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 2,253 |  | 9,012 |  | 751 |  | 1,192 | 21,605 |  | 596 |  | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 5,514 |  | 22,056 |  | 1,838 |  | 2,073 | 50,098 |  | 1,382 |  | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 4,362 |  | 17,448 |  | 1,454 |  | 1,497 | 36,178 |  | 998 |  | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 3,738 |  | 14,952 |  | 1,246 |  | 1,401 | 33,858 |  | 934 |  | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 2,970 |  | 11,880 |  | 990 |  | 1,017 | 24,578 |  | 678 |  | 
| app_proof | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4 |  | 2,496 |  | 9,984 |  | 832 |  | 405 | 11,745 |  | 324 |  | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 66,150 | 1,050 | 264,600 | 4,200 | 22,050 | 350 | 17,024 | 607,478 | 9,642 | 16,758 | 266 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 54,054 | 858 | 216,216 | 3,432 | 18,018 | 286 | 12,928 | 461,318 | 7,322 | 12,726 | 202 | 
| app_proof | 19 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,248 |  | 4,992 |  | 416 |  | 243 | 5,873 |  | 162 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 8,192 |  | 516,096 |  | 2,064,384 |  | 172,032 |  | 32,772 | 1,187,840 |  | 32,768 |  | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 21 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,572 |  | 6,288 |  | 524 |  | 534 | 12,905 |  | 356 |  | 
| app_proof | 22 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 12, 4, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 7 | 1 | 6,216 | 888 | 24,864 | 3,552 | 2,072 | 296 | 904 | 28,674 | 4,096 | 791 | 113 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 2,844 |  | 11,376 |  | 948 |  | 1,062 | 25,665 |  | 708 |  | 
| app_proof | 24 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 12, 12, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 3,402 | 1,134 | 13,608 | 4,536 | 1,134 | 378 | 1,032 | 28,058 | 9,352 | 774 | 258 | 
| app_proof | 25 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,248 |  | 4,992 |  | 416 |  | 243 | 5,873 |  | 162 |  | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 27 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,572 |  | 6,288 |  | 524 |  | 534 | 12,905 |  | 356 |  | 
| app_proof | 28 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 4,368 | 624 | 17,472 | 2,496 | 1,456 | 208 | 648 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 11,008 | 5,376 | 2,113,536 | 1,032,192 | 4,227,072 | 2,064,384 | 352,256 | 172,032 | 44,036 | 1,596,160 | 779,520 | 44,032 | 21,504 | 
| app_proof | 30 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 2,358 | 786 | 9,432 | 3,144 | 786 | 262 | 712 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 31 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,248 |  | 4,992 |  | 416 |  | 243 | 5,873 |  | 162 |  | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 33 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,572 |  | 6,288 |  | 524 |  | 534 | 12,905 |  | 356 |  | 
| app_proof | 34 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 4,368 | 624 | 17,472 | 2,496 | 1,456 | 208 | 648 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 36 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 2,358 | 786 | 9,432 | 3,144 | 786 | 262 | 712 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 37 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 1,248 |  | 4,992 |  | 416 |  | 243 | 5,873 |  | 162 |  | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 39 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,572 |  | 6,288 |  | 524 |  | 534 | 12,905 |  | 356 |  | 
| app_proof | 4 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 10,152 | 3,384 | 40,608 | 13,536 | 3,384 | 1,128 | 3,904 | 106,140 | 35,380 | 2,928 | 976 | 
| app_proof | 40 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 8, 4, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 7 | 1 | 4,368 | 624 | 17,472 | 2,496 | 1,456 | 208 | 648 | 20,554 | 2,936 | 567 | 81 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,956 |  | 7,824 |  | 652 |  | 726 | 17,545 |  | 484 |  | 
| app_proof | 42 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 2,358 | 786 | 9,432 | 3,144 | 786 | 262 | 712 | 19,358 | 6,452 | 534 | 178 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 151 | 105 | 102,831 | 71,505 | 411,324 | 286,020 | 34,277 | 23,835 | 22,040 | 793,694 | 551,906 | 21,895 | 15,225 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 0 | 151 | 105 | 75,198 | 52,290 | 300,792 | 209,160 | 25,066 | 17,430 | 11,552 | 416,005 | 289,275 | 11,476 | 7,980 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 0 | 440 | 72 | 306,240 | 50,112 | 1,224,960 | 200,448 | 102,080 | 16,704 | 50,715 | 1,834,250 | 300,150 | 50,600 | 8,280 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 906 | 118 | 627,858 | 81,774 | 2,511,432 | 327,096 | 209,286 | 27,258 | 131,515 | 4,762,163 | 620,237 | 131,370 | 17,110 | 
| app_proof | 5 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 24, 24, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 3,333 |  | 13,332 |  | 1,111 |  | 1,768 | 32,045 |  | 884 |  | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 315 | 197 | 29,295 | 18,321 | 117,180 | 73,284 | 9,765 | 6,107 | 6,004 | 216,957 | 135,683 | 5,985 | 3,743 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 37,748,736 |  | 75,497,472 |  | 6,291,456 |  | 2,097,153 | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | 0 | 11,476 | 4,908 | 19,555,104 | 8,363,232 | 39,110,208 | 16,726,464 | 3,259,184 | 1,393,872 | 1,698,596 | 61,568,740 | 26,331,420 | 1,698,448 | 726,384 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 195,092 | 67,052 | 533,771,712 | 183,454,272 | 1,067,543,424 | 366,908,544 | 88,961,952 | 30,575,712 | 5,657,697 | 205,090,465 | 70,488,415 | 5,657,668 | 1,944,508 | 
| app_proof | 57 | KeccakfOpAir | 0 | 5,449 | 2,743 | 9,170,667 | 4,616,469 | 36,682,668 | 18,465,876 | 3,056,889 | 1,538,823 | 1,689,500 | 61,233,138 | 30,824,462 | 1,689,190 | 850,330 | 
| app_proof | 58 | KeccakfPermAir | 0 | 130,776 | 296 | 2,066,783,904 | 4,677,984 | 4,133,567,808 | 9,355,968 | 344,463,984 | 779,664 | 261,554 | 9,481,260 | 21,460 | 261,552 | 592 | 
| app_proof | 59 | XorinVmAir | 0 | 5,440 | 2,752 | 14,916,480 | 7,545,984 | 59,665,920 | 30,183,936 | 4,972,160 | 2,515,328 | 3,052,401 | 110,629,200 | 55,965,360 | 3,051,840 | 1,543,872 | 
| app_proof | 6 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 5,679 | 1,893 | 22,716 | 7,572 | 1,893 | 631 | 2,108 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 18,688 | 14,080 | 1,121,280 | 844,800 | 4,485,120 | 3,379,200 | 373,760 | 281,600 | 224,268 | 8,129,280 | 6,124,800 | 224,256 | 168,960 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 37,368 | 28,168 | 3,138,912 | 2,366,112 | 12,555,648 | 9,464,448 | 1,046,304 | 788,704 | 597,904 | 21,673,440 | 16,337,440 | 597,888 | 450,688 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 25,790 | 6,978 | 1,392,660 | 376,812 | 5,570,640 | 1,507,248 | 464,220 | 125,604 | 257,910 | 9,348,875 | 2,529,525 | 257,900 | 69,780 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 110,903 | 20,169 | 10,646,688 | 1,936,224 | 42,586,752 | 7,744,896 | 3,548,896 | 645,408 | 1,441,752 | 52,263,039 | 9,504,641 | 1,441,739 | 262,197 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 122,182 | 8,890 | 9,530,196 | 693,420 | 38,120,784 | 2,773,680 | 3,176,732 | 231,140 | 1,344,013 | 48,720,073 | 3,544,887 | 1,344,002 | 97,790 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 643,929 | 404,647 | 79,203,267 | 49,771,581 | 316,813,068 | 199,086,324 | 26,401,089 | 16,590,527 | 10,946,810 | 396,821,247 | 249,363,713 | 10,946,793 | 6,878,999 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 19,948 | 12,820 | 3,171,732 | 2,038,380 | 12,686,928 | 8,153,520 | 1,057,244 | 679,460 | 478,776 | 17,354,760 | 11,153,400 | 478,752 | 307,680 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 25,731 | 7,037 | 2,856,141 | 781,107 | 11,424,564 | 3,124,428 | 952,047 | 260,369 | 463,176 | 16,789,478 | 4,591,642 | 463,158 | 126,666 | 
| app_proof | 7 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 2,253 |  | 9,012 |  | 751 |  | 1,192 | 21,605 |  | 596 |  | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 510,907 | 13,381 | 55,177,956 | 1,445,148 | 220,711,824 | 5,780,592 | 18,392,652 | 481,716 | 10,218,160 | 370,407,575 | 9,701,225 | 10,218,140 | 267,620 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 7,077,888 |  | 14,155,776 |  | 1,179,648 |  | 131,074 | 4,751,360 |  | 131,072 |  | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 19,200 | 13,568 | 17,280,000 | 12,211,200 | 69,120,000 | 48,844,800 | 5,760,000 | 4,070,400 | 19,201 | 696,000 | 491,840 | 19,200 | 13,568 | 
| app_proof | 74 | VariableRangeCheckerAir | 0 | 262,144 |  | 6,291,456 |  | 12,582,912 |  | 1,048,576 |  | 262,145 | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 5,679 | 1,893 | 22,716 | 7,572 | 1,893 | 631 | 2,108 | 57,312 | 19,103 | 1,581 | 527 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 16, 16, 4, 4>, FieldExpressionCoreAir> | 0 | 1 |  | 2,253 |  | 9,012 |  | 751 |  | 1,192 | 21,605 |  | 596 |  | 
| app_proof | 0 | ProgramAir | 1 | 10,064 | 6,320 | 301,920 | 189,600 | 1,207,680 | 758,400 | 100,640 | 63,200 | 10,065 | 364,820 | 229,100 | 10,064 | 6,320 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 72 |  | 144 |  | 12 |  | 15 | 363 |  | 10 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 4,352 | 3,840 | 274,176 | 241,920 | 1,096,704 | 967,680 | 91,392 | 80,640 | 17,412 | 631,040 | 556,800 | 17,408 | 15,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 5,848 | 2,344 | 1,122,816 | 450,048 | 2,245,632 | 900,096 | 187,136 | 75,008 | 23,396 | 847,960 | 339,880 | 23,392 | 9,376 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 49 | 15 | 33,369 | 10,215 | 133,476 | 40,860 | 11,123 | 3,405 | 7,250 | 257,557 | 78,843 | 7,105 | 2,175 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapBranchAdapterAir<2, 8, 4>, 2, 8, 4, 32>, BranchEqualCoreAir<32> | 1 | 49 | 15 | 24,402 | 7,470 | 97,608 | 29,880 | 8,134 | 2,490 | 3,800 | 134,995 | 41,325 | 3,724 | 1,140 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, LessThanCoreAir<32, 8> | 1 | 150 | 106 | 104,400 | 73,776 | 417,600 | 295,104 | 34,800 | 24,592 | 17,365 | 625,313 | 441,887 | 17,250 | 12,190 | 
| app_proof | 48 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 8, 8, 4, 4>, 2, 8, 8, 4, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 294 | 218 | 203,742 | 151,074 | 814,968 | 604,296 | 67,914 | 50,358 | 42,775 | 1,545,338 | 1,145,862 | 42,630 | 31,610 | 
| app_proof | 51 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 98 | 30 | 9,114 | 2,790 | 36,456 | 11,160 | 3,038 | 930 | 1,881 | 67,498 | 20,662 | 1,862 | 570 | 
| app_proof | 52 | RangeTupleCheckerAir<2> | 1 | 2,097,152 |  | 37,748,736 |  | 75,497,472 |  | 6,291,456 |  | 2,097,153 | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 55 | Sha2MainAir<Sha256Config> | 1 | 8,624 | 7,760 | 14,695,296 | 13,223,040 | 29,390,592 | 26,446,080 | 2,449,216 | 2,203,840 | 1,276,500 | 46,267,760 | 41,632,400 | 1,276,352 | 1,148,480 | 
| app_proof | 56 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 146,608 | 115,536 | 401,119,488 | 316,106,496 | 802,238,976 | 632,212,992 | 66,853,248 | 52,684,416 | 4,251,661 | 154,121,660 | 121,457,220 | 4,251,632 | 3,350,544 | 
| app_proof | 57 | KeccakfOpAir | 1 | 4,021 | 75 | 6,767,343 | 126,225 | 27,069,372 | 504,900 | 2,255,781 | 42,075 | 1,246,820 | 45,185,988 | 842,812 | 1,246,510 | 23,250 | 
| app_proof | 58 | KeccakfPermAir | 1 | 96,504 | 34,568 | 1,525,149,216 | 546,312,672 | 3,050,298,432 | 1,092,625,344 | 254,191,536 | 91,052,112 | 193,010 | 6,996,540 | 2,506,180 | 193,008 | 69,136 | 
| app_proof | 59 | XorinVmAir | 1 | 4,018 | 78 | 11,017,356 | 213,876 | 44,069,424 | 855,504 | 3,672,452 | 71,292 | 2,254,659 | 81,711,053 | 1,586,227 | 2,254,098 | 43,758 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 13,183 | 3,201 | 790,980 | 192,060 | 3,163,920 | 768,240 | 263,660 | 64,020 | 158,208 | 5,734,605 | 1,392,435 | 158,196 | 38,412 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 26,371 | 6,397 | 2,215,164 | 537,348 | 8,860,656 | 2,149,392 | 738,388 | 179,116 | 421,952 | 15,295,180 | 3,710,260 | 421,936 | 102,352 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 17,696 | 15,072 | 955,584 | 813,888 | 3,822,336 | 3,255,552 | 318,528 | 271,296 | 176,970 | 6,414,800 | 5,463,600 | 176,960 | 150,720 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 79,233 | 51,839 | 7,606,368 | 4,976,544 | 30,425,472 | 19,906,176 | 2,535,456 | 1,658,848 | 1,030,042 | 37,338,552 | 24,429,128 | 1,030,029 | 673,907 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 87,617 | 43,455 | 6,834,126 | 3,389,490 | 27,336,504 | 13,557,960 | 2,278,042 | 1,129,830 | 963,798 | 34,937,279 | 17,327,681 | 963,787 | 478,005 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 419,699 | 104,589 | 51,622,977 | 12,864,447 | 206,491,908 | 51,457,788 | 17,207,659 | 4,288,149 | 7,134,900 | 258,639,509 | 64,452,971 | 7,134,883 | 1,778,013 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 6,468 | 1,724 | 1,028,412 | 274,116 | 4,113,648 | 1,096,464 | 342,804 | 91,372 | 155,256 | 5,627,160 | 1,499,880 | 155,232 | 41,376 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 18,107 | 14,661 | 2,009,877 | 1,627,371 | 8,039,508 | 6,509,484 | 669,959 | 542,457 | 325,944 | 11,814,818 | 9,566,302 | 325,926 | 263,898 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 354,225 | 170,063 | 38,256,300 | 18,366,804 | 153,025,200 | 73,467,216 | 12,752,100 | 6,122,268 | 7,084,520 | 256,813,125 | 123,295,675 | 7,084,500 | 3,401,260 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 7,077,888 |  | 14,155,776 |  | 1,179,648 |  | 131,074 | 4,751,360 |  | 131,072 |  | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 10,200 | 6,184 | 9,180,000 | 5,565,600 | 36,720,000 | 22,262,400 | 3,060,000 | 1,855,200 | 10,201 | 369,750 | 224,170 | 10,200 | 6,184 | 
| app_proof | 74 | VariableRangeCheckerAir | 1 | 262,144 |  | 6,291,456 |  | 12,582,912 |  | 1,048,576 |  | 262,145 | 9,502,720 |  | 262,144 |  | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 20 | 278 | 20 | 6 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 10 | 149 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 133 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 120 | 955 | 120 | 23 | 9 | 20 | 5 | 5 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,979 | 258 | 117 | 0 | 0 | 108 | 36 | 35 | 51 | 20 | 0 | 31 | 17 | 13 | 3 | 9 | 118 | 108 | 0 | 3 | 19 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,767 | 138 | 45 | 0 | 0 | 67 | 25 | 24 | 28 | 13 | 0 | 25 | 17 | 8 | 1 | 6 | 45 | 67 | 0 | 2 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,863 | 123 | 31 | 0 | 0 | 63 | 25 | 24 | 24 | 13 | 0 | 28 | 20 | 7 | 1 | 5 | 31 | 63 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 265,003,892 | 834 | 405 | 0 | 1 | 352 | 136 | 135 | 77 | 138 | 0 | 76 | 35 | 40 | 22 | 18 | 405 | 352 | 0 | 6 | 137 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,586 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,382 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,358 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 40,826,179 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 567,917,156 | 1,820 | 598 | 0 | 185 | 1,084 | 673 | 672 | 212 | 198 | 55 | 136 | 34 | 101 | 51 | 50 | 598 | 1,084 | 0 | 3 | 141 | 0 | 0 | 
| app_proof | prover | 1 | 537,266,124 | 1,352 | 431 | 0 | 1 | 798 | 467 | 467 | 193 | 137 | 0 | 122 | 35 | 87 | 50 | 37 | 431 | 798 | 0 | 3 | 135 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 54,936,802 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 40,900,042 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_max_padded_height | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 189 | 2,251 | 189 | 178 | 9,069,025,194 | 1,018,123,350 | 10,087,148,544 | 2,097,152 | 2,097,152 | 0 | 3 | 63 | 1,540,000 | 34.99 | 
| app_proof | 1 | 103 | 1,628 | 103 | 41 | 6,675,792,954 | 2,876,399,838 | 9,552,192,792 | 2,097,152 | 2,097,152 | 0 | 2 | 130 | 1,039,903 | 36.81 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/094e780c1ac157e5cd3af8ece725dcec137b06ec

Max Segment Length: 4194304

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27169101120)
