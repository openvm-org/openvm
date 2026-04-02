| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  3.15 |  2.20 |  2.20 |
| app_proof |  2.37 |  1.42 |  1.42 |
| leaf |  0.41 |  0.41 |  0.41 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.11 |  0.11 |  0.11 |
| internal_recursive.1 |  0.10 |  0.10 |  0.10 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,182.50 |  2,365 |  1,413 |  952 |
| `execute_metered_time_ms` |  7 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  20.47 | -          |  20.47 |  20.47 |
| `execute_preflight_insns` |  77,381.50 |  154,763 |  121,000 |  33,763 |
| `execute_preflight_time_ms` |  69.50 |  139 |  120 |  19 |
| `execute_preflight_insn_mi/s` |  8.66 | -          |  11.49 |  5.83 |
| `trace_gen_time_ms   ` |  159 |  318 |  189 |  129 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  839.50 |  1,679 |  1,018 |  661 |
| `prover.main_trace_commit_time_ms` |  220.50 |  441 |  292 |  149 |
| `prover.rap_constraints_time_ms` |  557 |  1,114 |  653 |  461 |
| `prover.openings_time_ms` |  60.50 |  121 |  72 |  49 |
| `prover.rap_constraints.logup_gkr_time_ms` |  140 |  280 |  153 |  127 |
| `prover.rap_constraints.round0_time_ms` |  284 |  568 |  360 |  208 |
| `prover.rap_constraints.mle_rounds_time_ms` |  132 |  264 |  139 |  125 |
| `prover.openings.stacked_reduction_time_ms` |  48 |  96 |  60 |  36 |
| `prover.openings.stacked_reduction.round0_time_ms` |  14.50 |  29 |  15 |  14 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  33 |  66 |  44 |  22 |
| `prover.openings.whir_time_ms` |  12 |  24 |  12 |  12 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  410 |  410 |  410 |  410 |
| `execute_preflight_time_ms` |  17 |  17 |  17 |  17 |
| `trace_gen_time_ms   ` |  103 |  103 |  103 |  103 |
| `generate_blob_total_time_ms` |  11 |  11 |  11 |  11 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  306 |  306 |  306 |  306 |
| `prover.main_trace_commit_time_ms` |  124 |  124 |  124 |  124 |
| `prover.rap_constraints_time_ms` |  139 |  139 |  139 |  139 |
| `prover.openings_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  58 |  58 |  58 |  58 |
| `prover.rap_constraints.round0_time_ms` |  44 |  44 |  44 |  44 |
| `prover.rap_constraints.mle_rounds_time_ms` |  35 |  35 |  35 |  35 |
| `prover.openings.stacked_reduction_time_ms` |  30 |  30 |  30 |  30 |
| `prover.openings.stacked_reduction.round0_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.whir_time_ms` |  12 |  12 |  12 |  12 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  156 |  156 |  156 |  156 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  138 |  138 |  138 |  138 |
| `prover.main_trace_commit_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints_time_ms` |  64 |  64 |  64 |  64 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  8 |  8 |  8 |  8 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  113 |  113 |  113 |  113 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  103 |  103 |  103 |  103 |
| `prover.main_trace_commit_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints_time_ms` |  50 |  50 |  50 |  50 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  10 |  10 |  10 |  10 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  101 |  101 |  101 |  101 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  8 |  8 |  8 |  8 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  92 |  92 |  92 |  92 |
| `prover.main_trace_commit_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints_time_ms` |  48 |  48 |  48 |  48 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  7 |  7 |  7 |  7 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/9fae58d796a4663e68c675efcd207f71258bed21/kitchen_sink-9fae58d796a4663e68c675efcd207f71258bed21.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 10.34 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 9.28 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 9.28 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 9.17 | app_proof.prover.0 |
| prover.gkr_input_evals | 9.17 | app_proof.prover.0 |
| prover.openings | 9.11 | app_proof.prover.0 |
| prover.merkle_tree | 9.11 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.11 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 8.56 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 8.56 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 7.15 | app_proof.prover.0 |
| prover.stacked_commit | 7.15 | app_proof.prover.0 |
| prover.rs_code_matrix | 7.13 | app_proof.prover.0 |
| generate mem proving ctxs | 5.32 | app_proof.0 |
| set initial memory | 5.32 | app_proof.1 |
| tracegen.whir_final_poly_query_eval | 1.82 | leaf.0 |
| tracegen.exp_bits_len | 1.82 | leaf.0 |
| tracegen.pow_checker | 1.82 | leaf.0 |
| tracegen.whir_folding | 1.69 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.69 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.69 | leaf.0 |
| tracegen.range_checker | 1.58 | leaf.0 |
| tracegen.public_values | 1.58 | leaf.0 |
| tracegen.proof_shape | 1.58 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 617 | 3 | 0 | 264,807 | 227,731 | 3 | 63 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| AccessAdapterAir<16> | 5 | 5 | 2 | 
| AccessAdapterAir<2> | 5 | 5 | 2 | 
| AccessAdapterAir<32> | 5 | 5 | 2 | 
| AccessAdapterAir<4> | 5 | 5 | 2 | 
| AccessAdapterAir<8> | 5 | 5 | 2 | 
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| KeccakVmAir | 321 | 4,247 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| ProgramAir | 1 |  | 1 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| Rv32HintStoreAir | 18 | 17 | 3 | 
| Sha256VmAir | 50 | 634 | 4 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 22 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 28 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 76 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 25 | 3 | 
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 9 | 2 | 
| VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 61 | 76 | 3 | 
| VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 31 | 110 | 3 | 
| VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 61 | 7 | 2 | 
| VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, ShiftCoreAir<32, 8> | 79 | 2,118 | 3 | 
| VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 20 | 41 | 3 | 
| VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchLessThanCoreAir<32, 8> | 22 | 111 | 3 | 
| VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 25 | 208 | 3 | 
| VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 41 | 308 | 3 | 
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 18 | 18 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 25 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 25 | 64 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 24 | 11 | 2 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 4 | 2 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 5 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 415 | 268 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 832 | 501 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 182 | 108 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 428 | 239 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 246 | 162 | 3 | 
| VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 668 | 363 | 3 | 
| VmConnectorAir | 5 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 73 | 3 |  |  | 0 |  | 331 | 3 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 952 |  |  | 7 | 154,763 | 20.47 | 0 | 2,378 |  | 
| internal_for_leaf |  |  |  | 156 |  |  |  |  |  |  |  |  |  | 156 | 
| internal_recursive.0 |  |  |  | 113 |  |  |  |  |  |  |  |  |  | 113 | 
| internal_recursive.1 |  |  |  | 101 |  |  |  |  |  |  |  |  |  | 101 | 
| leaf |  |  | 410 |  |  |  |  |  |  |  |  |  |  | 410 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakVmAir | 0 | 21 | 
| app_proof | Sha256VmAir | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 100 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | KeccakVmAir | 1 | 0 | 
| app_proof | Sha256VmAir | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 107 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 17 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 0 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 345 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 152 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 464 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 224 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 7,680 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 729,088 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 266,240 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 1,152 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 458,752 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 20,250,624 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 999,424 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 22,528 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 3,072 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 102 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 5,312 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 66,048 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 38 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 2,048 | 63 | 309,248 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 2,528 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 4,512 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 210,944 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 6,684,672 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 664 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 73,728 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 32,768 | 89 | 4,620,288 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 180,224 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 385,024 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 88,064 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 17,039,360 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 393,216 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 10,240 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 11,392 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 2,592 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 606,208 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 592 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 690 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 304 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 928 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 448 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 512 | 32 | 30,720 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 5,832,704 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 32,768 | 25 | 2,129,920 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 2,304 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 3,670,016 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 162,004,992 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 3,997,696 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 47 | 90,880 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 204 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 10,624 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 132,096 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 76 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 4,947,968 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 5,056 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 9,024 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 421,888 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 140,509,184 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 1,328 | 
| leaf | 31 | EqBitsAir | 0 | prover | 32,768 | 16 | 1,179,648 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 1,360 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 3,648 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 53,248 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 36,962,304 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 360,448 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 770,048 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 176,128 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 34,078,720 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 12,416 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 786,432 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 20,480 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 22,784 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 5,184 | 
| leaf | 8 | Eq3bAir | 0 | prover | 262,144 | 25 | 9,699,328 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 1,184 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 229,376 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 1 | 949 | 3,621 | 
| app_proof | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 8,828 | 
| app_proof | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 1 | 625 | 2,337 | 
| app_proof | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 8,828 | 
| app_proof | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 1 | 625 | 2,337 | 
| app_proof | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 8,828 | 
| app_proof | 16 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 1 | 625 | 2,337 | 
| app_proof | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 2 | 757 | 5,314 | 
| app_proof | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 2 | 565 | 3,394 | 
| app_proof | 19 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 497 | 3,386 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 20 | 262,144 | 
| app_proof | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 369 | 2,106 | 
| app_proof | 21 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 4 | 166 | 1,064 | 
| app_proof | 22 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 64 | 287 | 64,960 | 
| app_proof | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 64 | 223 | 44,480 | 
| app_proof | 24 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 166 | 532 | 
| app_proof | 25 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 199 | 1,150 | 
| app_proof | 27 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | prover | 0 | 8 | 242 | 3,248 | 
| app_proof | 28 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 2 | 393 | 2,754 | 
| app_proof | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 4 | 297 | 3,588 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 32 | 393,216 | 
| app_proof | 30 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 166 | 532 | 
| app_proof | 31 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 199 | 1,150 | 
| app_proof | 33 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 166 | 2,128 | 
| app_proof | 34 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 199 | 2,300 | 
| app_proof | 36 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 166 | 532 | 
| app_proof | 37 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 199 | 1,150 | 
| app_proof | 39 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 166 | 2,128 | 
| app_proof | 40 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 199 | 2,300 | 
| app_proof | 42 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 2 | 166 | 532 | 
| app_proof | 43 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 199 | 1,150 | 
| app_proof | 45 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | prover | 0 | 8 | 166 | 2,128 | 
| app_proof | 46 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 2 | 263 | 1,790 | 
| app_proof | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | prover | 0 | 4 | 199 | 2,300 | 
| app_proof | 49 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 164 | 104,448 | 
| app_proof | 51 | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | prover | 0 | 256 | 124 | 52,224 | 
| app_proof | 52 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | prover | 0 | 512 | 169 | 150,016 | 
| app_proof | 53 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | prover | 0 | 1,024 | 168 | 421,888 | 
| app_proof | 56 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 16 | 31 | 1,712 | 
| app_proof | 57 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 14,680,064 | 
| app_proof | 58 | Sha256VmAir | prover | 0 | 262,144 | 470 | 175,636,480 | 
| app_proof | 59 | KeccakVmAir | prover | 0 | 131,072 | 3,163 | 582,877,184 | 
| app_proof | 6 | AccessAdapterAir<8> | prover | 0 | 262,144 | 17 | 9,699,328 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 1,024 | 20 | 69,632 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 2,048 | 28 | 188,416 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 1,024 | 18 | 59,392 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 4,096 | 32 | 344,064 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 4,096 | 26 | 286,720 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 131,072 | 41 | 14,286,848 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 8,192 | 53 | 1,220,608 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 1,024 | 37 | 111,616 | 
| app_proof | 7 | AccessAdapterAir<16> | prover | 0 | 131,072 | 25 | 5,898,240 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 32,768 | 36 | 3,801,088 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,245,184 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 8 | AccessAdapterAir<32> | prover | 0 | 8,192 | 41 | 499,712 | 
| app_proof | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,020 | 17,392 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 229,376 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 52 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 4,096 | 20 | 131,072 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 4,096 | 32 | 196,608 | 
| app_proof | 49 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 1 | 64 | 164 | 26,112 | 
| app_proof | 51 | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | prover | 1 | 64 | 124 | 13,056 | 
| app_proof | 52 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | prover | 1 | 256 | 169 | 75,008 | 
| app_proof | 53 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | prover | 1 | 512 | 168 | 210,944 | 
| app_proof | 57 | RangeTupleCheckerAir<2> | prover | 1 | 2,097,152 | 3 | 14,680,064 | 
| app_proof | 58 | Sha256VmAir | prover | 1 | 262,144 | 470 | 175,636,480 | 
| app_proof | 59 | KeccakVmAir | prover | 1 | 131,072 | 3,163 | 582,877,184 | 
| app_proof | 6 | AccessAdapterAir<8> | prover | 1 | 262,144 | 17 | 9,699,328 | 
| app_proof | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 256 | 20 | 17,408 | 
| app_proof | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 512 | 28 | 47,104 | 
| app_proof | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 256 | 18 | 14,848 | 
| app_proof | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 512 | 32 | 43,008 | 
| app_proof | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 2,048 | 26 | 143,360 | 
| app_proof | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 32,768 | 41 | 3,571,712 | 
| app_proof | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 4,096 | 53 | 610,304 | 
| app_proof | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 512 | 37 | 55,808 | 
| app_proof | 7 | AccessAdapterAir<16> | prover | 1 | 131,072 | 25 | 5,898,240 | 
| app_proof | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 8,192 | 36 | 950,272 | 
| app_proof | 71 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,703,936 | 
| app_proof | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 4,096 | 300 | 1,245,184 | 
| app_proof | 74 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 2,097,152 | 
| app_proof | 8 | AccessAdapterAir<32> | prover | 1 | 2,048 | 41 | 124,928 | 

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
| agg_keygen | ProofShapeAir<4, 8> | 77 | 85 | 4 | 
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
| internal_for_leaf | 0 | 18 | 156 | 18 | 6 | 1 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 9 | 113 | 9 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 8 | 101 | 8 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 103 | 410 | 103 | 25 | 11 | 17 | 5 | 5 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 53,472,003 | 138 | 40 | 0 | 4 | 64 | 19 | 19 | 29 | 15 | 0 | 32 | 8 | 23 | 1 | 21 | 41 | 64 | 0 | 1 | 14 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 23,651,975 | 103 | 19 | 0 | 4 | 50 | 17 | 16 | 20 | 13 | 0 | 32 | 10 | 21 | 1 | 20 | 19 | 50 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 92 | 14 | 0 | 4 | 48 | 16 | 16 | 18 | 13 | 0 | 28 | 7 | 21 | 0 | 20 | 14 | 48 | 0 | 1 | 12 | 0 | 0 | 
| leaf | 0 | prover | 407,935,872 | 306 | 123 | 0 | 9 | 139 | 44 | 43 | 35 | 58 | 0 | 43 | 12 | 30 | 6 | 23 | 124 | 139 | 0 | 2 | 57 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,522 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 38,468,419 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 816,542,084 | 1,018 | 291 | 0 | 162 | 653 | 360 | 359 | 139 | 153 | 61 | 72 | 12 | 60 | 15 | 44 | 292 | 653 | 0 | 1 | 91 | 0 | 0 | 
| app_proof | prover | 1 | 800,298,548 | 661 | 149 | 0 | 20 | 461 | 208 | 208 | 125 | 127 | 73 | 49 | 12 | 36 | 14 | 22 | 149 | 461 | 0 | 1 | 53 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 63,146,482 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 60,612,938 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 189 | 1,413 | 189 | 184 | 0 | 50 | 19 | 121,000 | 11.49 | 
| app_proof | 1 | 129 | 952 | 129 | 40 | 0 | 2 | 120 | 33,763 | 5.83 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/9fae58d796a4663e68c675efcd207f71258bed21

Max Segment Length: 4194304

Instance Type: g6e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23922415143)
