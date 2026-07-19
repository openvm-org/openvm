| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  0.92 |  0.92 |  0.92 |
| app_proof |  0.33 |  0.33 |  0.33 |
| leaf |  0.19 |  0.19 |  0.19 |
| internal_for_leaf |  0.17 |  0.17 |  0.17 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.12 |  0.12 |  0.12 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  285 |  285 |  285 |  285 |
| `compile_metered_time_ms` |  6 |  6 |  6 |  6 |
| `execute_metered_time_ms` |  47 | -          | -          | -          |
| `execute_metered_insns` |  592,827 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  12.57 | -          |  12.57 |  12.57 |
| `execute_preflight_insns` |  592,827 |  592,827 |  592,827 |  592,827 |
| `execute_preflight_time_ms` |  67 |  67 |  67 |  67 |
| `execute_preflight_insn_mi/s` |  11 | -          |  11 |  11 |
| `trace_gen_time_ms   ` |  68 |  68 |  68 |  68 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  148 |  148 |  148 |  148 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  95 |  95 |  95 |  95 |
| `prover.openings_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.round0_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  186 |  186 |  186 |  186 |
| `execute_preflight_time_ms` |  3 |  3 |  3 |  3 |
| `trace_gen_time_ms   ` |  30 |  30 |  30 |  30 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  155 |  155 |  155 |  155 |
| `prover.main_trace_commit_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints_time_ms` |  81 |  81 |  81 |  81 |
| `prover.openings_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.logup_gkr_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.round0_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  32 |  32 |  32 |  32 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  167 |  167 |  167 |  167 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  17 |  17 |  17 |  17 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  149 |  149 |  149 |  149 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  73 |  73 |  73 |  73 |
| `prover.openings_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.mle_rounds_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  33 |  33 |  33 |  33 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  120 |  120 |  120 |  120 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  109 |  109 |  109 |  109 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  25 |  25 |  25 |  25 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  115 |  115 |  115 |  115 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  104 |  104 |  104 |  104 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  27 |  27 |  27 |  27 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/c9acdfc8756fb9ed314fb6495f44f4df233270d5/pairing-c9acdfc8756fb9ed314fb6495f44f4df233270d5.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| frac_sumcheck.gkr_rounds | 1.33 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 1.33 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 1.29 | app_proof.prover.0 |
| prover.gkr_input_evals | 1.29 | app_proof.prover.0 |
| prover.stacked_commit | 1.02 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 0.98 | internal_for_leaf.0.prover |
| prover.openings | 0.98 | internal_for_leaf.0.prover |
| prover.merkle_tree | 0.98 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 0.98 | internal_for_leaf.0.prover |
| generate mem proving ctxs | 0.88 | app_proof.0 |
| set initial memory | 0.87 | app_proof.0 |
| prover.rap_constraints | 0.79 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 0.47 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 0.47 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 0.35 | app_proof.prover.0 |
| tracegen.pow_checker | 0.31 | leaf.0 |
| tracegen.exp_bits_len | 0.31 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.31 | leaf.0 |
| tracegen.whir_folding | 0.24 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.24 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.24 | leaf.0 |
| tracegen.range_checker | 0.24 | leaf.0 |
| tracegen.public_values | 0.24 | leaf.0 |
| tracegen.proof_shape | 0.24 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 131 | 267,239 | 229,966 | 21 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 11 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 14 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 15 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 17 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 19 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 21 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 7 | 2 | 
| 22 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 22 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 45 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 68 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> |  | 16 | 8 | 3 | 
| 26 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 27 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 28 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 31 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 29 | 13 | 3 | 
| 32 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 28 | 15 | 3 | 
| 33 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 27 | 13 | 3 | 
| 34 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 26 | 15 | 3 | 
| 35 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 27 | 16 | 3 | 
| 36 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 26 | 13 | 3 | 
| 37 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 25 | 15 | 3 | 
| 38 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 26 | 16 | 3 | 
| 39 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 11 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 40 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 13 | 3 | 
| 41 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 14 | 3 | 
| 42 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 36 | 3 | 
| 43 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 45 | 3 | 
| 44 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 46 | 3 | 
| 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 69 | 3 | 
| 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 23 | 3 | 
| 47 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 18 | 16 | 3 | 
| 48 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 7 | 2 | 
| 49 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 14 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 50 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 51 | PhantomAir |  | 3 | 1 | 2 | 
| 52 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 53 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 372 | 194 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 244 | 130 | 3 | 
| 8 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 

| group | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | new_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 60 |  |  |  | 305 |  |  |  |  |  |  | 
| app_proof |  |  |  | 285 |  | 47 | 592,827 | 12.57 | 0 | 341 |  | 
| internal_for_leaf |  |  | 167 |  |  |  |  |  |  |  | 167 | 
| internal_recursive.0 |  |  | 120 |  |  |  |  |  |  |  | 120 | 
| internal_recursive.1 |  |  | 115 |  |  |  |  |  |  |  | 115 | 
| leaf |  | 186 |  |  |  |  |  |  |  |  | 186 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 20 | 
| app_proof | Rv64HintStoreAir | 37 | Rv64HintStoreAir | 0 | 1,952 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 36 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 486,508 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 32 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 | 3,335,552 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 35 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 11,440 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 33 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 214,016 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 60,120 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 8 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 606,780 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 108,324 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 27 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1,501,968 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 28 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 283,104 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 29 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 224,080 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 46 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 2,664 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 30 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 360,624 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 17 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 60,480 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 16 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 12,769,140 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 22 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 95,040 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 41 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 10,860 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 39 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 11,256 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 31 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 132,064 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 18 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 37,344 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 60 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 26 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 12,074,820 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 44 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 7,104 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 138,048 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 50 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 2,173,584 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 51 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 2,813,664 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 |  | 20 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 70 | 217 | 4 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 90 | 4 | 
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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 102,400 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 180,224 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 9,863,168 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 729,088 | 
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
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
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
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 46 | 2,944 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 7,864,320 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 1,458,176 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 64 | 24 | 1,536 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 11 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 17 | 
| app_proof | 11 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 719 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 37 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | MULHU | 0 | 181 | 
| app_proof | 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 201 | 
| app_proof | 20 | Rv64HintStoreAir | HINT_BUFFER | 0 | 1 | 
| app_proof | 21 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | ANDI | 0 | 11,057 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | SLTIU | 0 | 260 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SLLI | 0 | 3,230 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SRLI | 0 | 1,634 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | ADDI | 0 | 75,808 | 
| app_proof | 26 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 4,127 | 
| app_proof | 27 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 7,513 | 
| app_proof | 28 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 3,767 | 
| app_proof | 28 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 1,835 | 
| app_proof | 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGE | 0 | 6 | 
| app_proof | 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 2,159 | 
| app_proof | 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 3,733 | 
| app_proof | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 14,064 | 
| app_proof | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 17,227 | 
| app_proof | 31 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | STORED | 0 | 201,247 | 
| app_proof | 32 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | LOADD | 0 | 212,819 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | STOREW | 0 | 1 | 
| app_proof | 35 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | LOADW | 0 | 1,584 | 
| app_proof | 39 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | STOREB | 0 | 778 | 
| app_proof | 40 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | LOADBU | 0 | 1,260 | 
| app_proof | 41 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | LOADB | 0 | 8 | 
| app_proof | 47 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | ADDW | 0 | 17 | 
| app_proof | 47 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | SUBW | 0 | 1,576 | 
| app_proof | 48 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 618 | 
| app_proof | 48 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 384 | 
| app_proof | 49 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | ADD | 0 | 7,301 | 
| app_proof | 49 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | SUB | 0 | 2,812 | 
| app_proof | 51 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 8,374 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 6,469 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 327,680 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 32 | 116 | 3,712 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 268 | 274,432 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 204 | 13,056 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | prover | 0 | 256 | 55 | 14,080 | 
| app_proof | 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 19 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 16,384 | 21 | 344,064 | 
| app_proof | 20 | Rv64HintStoreAir | prover | 0 | 128 | 27 | 3,456 | 
| app_proof | 21 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover | 0 | 16,384 | 36 | 589,824 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover | 0 | 512 | 29 | 14,848 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover | 0 | 8,192 | 53 | 434,176 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | prover | 0 | 131,072 | 25 | 3,276,800 | 
| app_proof | 26 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 8,192 | 17 | 139,264 | 
| app_proof | 27 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 8,192 | 24 | 196,608 | 
| app_proof | 28 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 8,192 | 18 | 147,456 | 
| app_proof | 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 8,192 | 32 | 262,144 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 16,384 | 33 | 540,672 | 
| app_proof | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 32,768 | 26 | 851,968 | 
| app_proof | 31 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover | 0 | 262,144 | 44 | 11,534,336 | 
| app_proof | 32 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover | 0 | 262,144 | 43 | 11,272,192 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover | 0 | 1 | 42 | 42 | 
| app_proof | 35 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover | 0 | 2,048 | 42 | 86,016 | 
| app_proof | 39 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 40 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover | 0 | 2,048 | 31 | 63,488 | 
| app_proof | 41 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover | 0 | 8 | 32 | 256 | 
| app_proof | 47 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | prover | 0 | 2,048 | 33 | 67,584 | 
| app_proof | 48 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 1,024 | 45 | 46,080 | 
| app_proof | 49 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover | 0 | 16,384 | 32 | 524,288 | 
| app_proof | 50 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 51 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 52 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 16,384 | 300 | 4,915,200 | 
| app_proof | 53 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16,384 | 513 | 8,404,992 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 8,192 | 385 | 3,153,920 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 19,082 | 13,686 | 477,050 | 342,150 | 763,280 | 547,440 | 190,820 | 136,860 | 691,723 | 496,117 | 19,082 | 13,686 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 11 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 18 | 14 | 5,220 | 4,060 | 8,352 | 6,496 | 2,088 | 1,624 | 33,278 | 25,882 | 918 | 714 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 719 | 305 | 481,730 | 204,350 | 770,768 | 326,960 | 192,692 | 81,740 | 5,082,432 | 2,155,968 | 140,205 | 59,475 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 37 | 27 | 18,870 | 13,770 | 30,192 | 22,032 | 7,548 | 5,508 | 175,704 | 128,216 | 4,847 | 3,537 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 181 | 75 | 24,888 | 10,312 | 39,820 | 16,500 | 9,955 | 4,125 | 262,450 | 108,750 | 7,240 | 3,000 | 
| app_proof | 18 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 201 | 55 | 21,608 | 5,912 | 34,572 | 9,460 | 8,643 | 2,365 | 225,874 | 61,806 | 6,231 | 1,705 | 
| app_proof | 19 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 10,638 | 5,746 | 558,495 | 301,665 | 893,592 | 482,664 | 223,398 | 120,666 | 1,542,510 | 833,170 | 42,552 | 22,984 | 
| app_proof | 20 | Rv64HintStoreAir | 0 | 96 | 32 | 12,960 | 4,320 | 10,368 | 3,456 | 2,592 | 864 | 59,160 | 19,720 | 1,632 | 544 | 
| app_proof | 21 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 11,057 | 5,327 | 995,130 | 479,430 | 1,592,208 | 767,088 | 398,052 | 191,772 | 8,016,325 | 3,862,075 | 221,140 | 106,540 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 260 | 252 | 18,850 | 18,270 | 30,160 | 29,232 | 7,540 | 7,308 | 131,950 | 127,890 | 3,640 | 3,528 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 4,864 | 3,328 | 644,480 | 440,960 | 1,031,168 | 705,536 | 257,792 | 176,384 | 3,350,080 | 2,292,160 | 92,416 | 63,232 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 | 75,808 | 55,264 | 4,738,000 | 3,454,000 | 7,580,800 | 5,526,400 | 1,895,200 | 1,381,600 | 43,968,640 | 32,053,120 | 1,212,928 | 884,224 | 
| app_proof | 26 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 4,127 | 4,065 | 175,398 | 172,762 | 280,636 | 276,420 | 70,159 | 69,105 | 2,094,453 | 2,062,987 | 57,778 | 56,910 | 
| app_proof | 27 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 7,513 | 679 | 450,780 | 40,740 | 721,248 | 65,184 | 180,312 | 16,296 | 4,085,194 | 369,206 | 112,695 | 10,185 | 
| app_proof | 28 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 5,602 | 2,590 | 252,090 | 116,550 | 403,344 | 186,480 | 100,836 | 46,620 | 2,436,870 | 1,126,650 | 67,224 | 31,080 | 
| app_proof | 29 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 5,898 | 2,294 | 471,840 | 183,520 | 754,944 | 293,632 | 188,736 | 73,408 | 2,993,235 | 1,164,205 | 82,572 | 32,116 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 11,832 | 4,552 | 1,952,280 | 751,080 | 1,561,824 | 600,864 | 390,456 | 150,216 | 1,715,640 | 660,040 | 47,328 | 18,208 | 
| app_proof | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 31,291 | 1,477 | 2,033,915 | 96,005 | 3,254,264 | 153,608 | 813,566 | 38,402 | 12,477,287 | 588,953 | 344,201 | 16,247 | 
| app_proof | 31 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 201,247 | 60,897 | 22,137,170 | 6,698,670 | 35,419,472 | 10,717,872 | 8,854,868 | 2,679,468 | 211,560,909 | 64,017,971 | 5,836,163 | 1,766,013 | 
| app_proof | 32 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 212,819 | 49,325 | 22,878,043 | 5,302,437 | 36,604,868 | 8,483,900 | 9,151,217 | 2,120,975 | 216,011,285 | 50,064,875 | 5,958,932 | 1,381,100 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 1 |  | 105 |  | 168 |  | 42 |  | 979 |  | 27 |  | 
| app_proof | 35 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 1,584 | 464 | 166,320 | 48,720 | 266,112 | 77,952 | 66,528 | 19,488 | 1,550,340 | 454,140 | 42,768 | 12,528 | 
| app_proof | 39 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 778 | 246 | 62,240 | 19,680 | 99,584 | 31,488 | 24,896 | 7,872 | 535,848 | 169,432 | 14,782 | 4,674 | 
| app_proof | 40 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 1,260 | 788 | 97,650 | 61,070 | 156,240 | 97,712 | 39,060 | 24,428 | 822,150 | 514,170 | 22,680 | 14,184 | 
| app_proof | 41 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 8 |  | 640 |  | 1,024 |  | 256 |  | 5,510 |  | 152 |  | 
| app_proof | 47 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 1,593 | 455 | 131,423 | 37,537 | 210,276 | 60,060 | 52,569 | 15,015 | 1,039,433 | 296,887 | 28,674 | 8,190 | 
| app_proof | 48 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 1,002 | 22 | 112,725 | 2,475 | 180,360 | 3,960 | 45,090 | 990 | 835,418 | 18,342 | 23,046 | 506 | 
| app_proof | 49 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 10,113 | 6,271 | 809,040 | 501,680 | 1,294,464 | 802,688 | 323,616 | 200,672 | 6,965,329 | 4,319,151 | 192,147 | 119,149 | 
| app_proof | 50 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 51 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 52 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 11,614 | 4,770 | 8,710,500 | 3,577,500 | 13,936,800 | 5,724,000 | 3,484,200 | 1,431,000 | 421,008 | 172,912 | 11,614 | 4,770 | 
| app_proof | 53 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 8,374 | 8,010 | 10,739,655 | 10,272,825 | 17,183,448 | 16,436,520 | 4,295,862 | 4,109,130 | 112,923,390 | 108,014,850 | 3,115,128 | 2,979,720 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 6,469 | 1,723 | 6,226,413 | 1,658,387 | 9,962,260 | 2,653,420 | 2,490,565 | 663,355 | 57,218,305 | 15,239,935 | 1,578,436 | 420,412 | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 6 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 17 | 167 | 16 | 5 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 120 | 10 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 10 | 115 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 30 | 186 | 30 | 7 | 0 | 3 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,560,317 | 149 | 33 | 0 | 0 | 73 | 26 | 25 | 33 | 14 | 0 | 42 | 33 | 9 | 2 | 6 | 33 | 33 | 73 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 109 | 20 | 0 | 0 | 55 | 21 | 20 | 23 | 10 | 0 | 33 | 25 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 104 | 14 | 0 | 0 | 54 | 22 | 21 | 21 | 11 | 0 | 34 | 27 | 7 | 1 | 5 | 15 | 14 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 44,715,645 | 155 | 32 | 0 | 0 | 81 | 34 | 33 | 24 | 22 | 0 | 41 | 32 | 9 | 2 | 6 | 32 | 32 | 81 | 0 | 3 | 22 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,803 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,299,011 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 7 | 0 | 7 | 7 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 52,920,380 | 148 | 20 | 0 | 0 | 95 | 40 | 39 | 18 | 36 | 0 | 31 | 22 | 9 | 3 | 6 | 20 | 20 | 95 | 0 | 1 | 35 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 28,770,144 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 2 | 68 | 285 | 68 | 0 | 910,167,782 | 346,528,610 | 1,256,696,392 | 2,097,152 | 1 | 5 | 67 | 592,827 | 11 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/c9acdfc8756fb9ed314fb6495f44f4df233270d5

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684794973)
