| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  0.81 |  0.81 |  0.81 |
| app_proof |  0.23 |  0.23 |  0.23 |
| leaf |  0.20 |  0.20 |  0.20 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  222 |  222 |  222 |  222 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insns` |  112,210 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  18.62 | -          |  18.62 |  18.62 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `execute_preflight_insns` |  112,210 |  112,210 |  112,210 |  112,210 |
| `execute_preflight_time_ms` |  7 |  7 |  7 |  7 |
| `execute_preflight_insn_mi/s` |  16.03 | -          |  15.44 |  15.44 |
| `postflight_time_ms  ` |  29 |  29 |  29 |  29 |
| `postflight_memory_chronology_time_ms` |  0 |  0 |  0 |  0 |
| `postflight_program_index_time_ms` |  0 |  0 |  0 |  0 |
| `trace_gen_time_ms   ` |  36 |  36 |  36 |  36 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  147 |  147 |  147 |  147 |
| `prover.main_trace_commit_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints_time_ms` |  107 |  107 |  107 |  107 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  72 |  72 |  72 |  72 |
| `prover.rap_constraints.mle_rounds_time_ms` |  16 |  16 |  16 |  16 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  195 |  195 |  195 |  195 |
| `execute_preflight_time_ms` |  6 |  6 |  6 |  6 |
| `trace_gen_time_ms   ` |  36 |  36 |  36 |  36 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  159 |  159 |  159 |  159 |
| `prover.main_trace_commit_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints_time_ms` |  86 |  86 |  86 |  86 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  31 |  31 |  31 |  31 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  163 |  163 |  163 |  163 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  15 |  15 |  15 |  15 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  147 |  147 |  147 |  147 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  73 |  73 |  73 |  73 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.mle_rounds_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  31 |  31 |  31 |  31 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  120 |  120 |  120 |  120 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  107 |  107 |  107 |  107 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  23 |  23 |  23 |  23 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  107 |  107 |  107 |  107 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  97 |  97 |  97 |  97 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb/ecrecover-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 1.02 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 0.99 | internal_for_leaf.0.prover |
| prover.merkle_tree | 0.99 | internal_for_leaf.0.prover |
| prover.openings | 0.99 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 0.99 | internal_for_leaf.0.prover |
| prover.batch_constraints.before_round0 | 0.94 | leaf.0.prover |
| frac_sumcheck.gkr_rounds | 0.94 | leaf.0.prover |
| frac_sumcheck.segment_tree | 0.92 | leaf.0.prover |
| prover.gkr_input_evals | 0.92 | leaf.0.prover |
| postflight | 0.91 | app_proof..0 |
| tracegen | 0.89 | app_proof..0 |
| generate mem proving ctxs | 0.89 | app_proof..0 |
| set initial memory | 0.88 | app_proof..0 |
| prover.rap_constraints | 0.80 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 0.47 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 0.47 | leaf.0.prover |
| prover.before_gkr_input_evals | 0.32 | leaf.0.prover |
| tracegen.whir_final_poly_query_eval | 0.32 | leaf.0 |
| tracegen.exp_bits_len | 0.32 | leaf.0 |
| tracegen.pow_checker | 0.32 | leaf.0 |
| tracegen.whir_folding | 0.25 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.25 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.25 | leaf.0 |
| tracegen.range_checker | 0.25 | leaf.0 |
| tracegen.public_values | 0.25 | leaf.0 |
| tracegen.proof_shape | 0.25 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 224 | 267,239 | 229,456 | 30 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 11 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 12 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 13 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 14 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 15 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 16 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 17 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 18 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 19 | KeccakfOpAir |  | 110 | 1 | 2 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 20 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 21 | XorinVmAir |  | 357 | 34 | 3 | 
| 22 | RevealAir |  | 17 | 3 | 2 | 
| 23 | HintStoreAir | 1 | 17 | 12 | 3 | 
| 24 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 25 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 26 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 27 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 28 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 29 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 8 | 3 | 
| 31 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 10 | 2 | 
| 32 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 33 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 34 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 29 | 8 | 3 | 
| 35 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 28 | 11 | 3 | 
| 36 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 27 | 8 | 3 | 
| 37 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 26 | 11 | 3 | 
| 38 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 27 | 12 | 3 | 
| 39 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 26 | 8 | 3 | 
| 4 | EcMulAir<32, 8> | 1 | 1,388 | 1,380 | 3 | 
| 40 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 25 | 11 | 3 | 
| 41 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 26 | 12 | 3 | 
| 42 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 43 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 44 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 45 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 46 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 47 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 48 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 49 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 5 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 262 | 3 | 
| 50 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 51 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 52 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 53 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 54 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 55 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 56 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 57 | PhantomAir |  | 3 | 1 | 2 | 
| 58 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 59 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 6 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 230 | 2 | 
| 7 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 8 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 9 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 61 |  |  |  | 313 |  | 
| app_proof | 0 |  |  |  | 4 |  |  | 
| internal_for_leaf |  |  |  | 163 |  |  | 163 | 
| internal_recursive.0 |  |  |  | 120 |  |  | 120 | 
| internal_recursive.1 |  |  |  | 107 |  |  | 107 | 
| leaf |  |  | 195 |  |  |  | 195 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | EcMulAir<32, 8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 0 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | RevealAir |  | 0 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 0 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 3 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 11 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 11 | 
| app_proof | XorinVmAir |  | 0 | 0 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 91 | 4 | 
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
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 9,863,168 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 47 | 3,008 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 11,520 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 262,144 | 60 | 15,728,640 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 729,088 | 
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
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 32,768 | 10 | 327,680 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 4,096 | 105 | 430,080 | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 32 | 253 | 8,096 | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 16 | 189 | 3,024 | 
| app_proof | 17 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 16 | 40 | 640 | 
| app_proof | 18 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 19 | KeccakfOpAir | prover |  | 0 | 8 | 258 | 2,064 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 1,024 | 38 | 38,912 | 
| app_proof | 20 | KeccakfPermAir | prover |  | 0 | 128 | 2,634 | 337,152 | 
| app_proof | 21 | XorinVmAir | prover |  | 0 | 8 | 543 | 4,344 | 
| app_proof | 23 | HintStoreAir | prover |  | 0 | 128 | 24 | 3,072 | 
| app_proof | 24 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 4,096 | 34 | 139,264 | 
| app_proof | 25 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 1,024 | 27 | 27,648 | 
| app_proof | 27 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 16,384 | 51 | 835,584 | 
| app_proof | 28 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 32,768 | 23 | 753,664 | 
| app_proof | 29 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 4,096 | 16 | 65,536 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 2,048 | 33 | 67,584 | 
| app_proof | 30 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 2,048 | 22 | 45,056 | 
| app_proof | 31 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 4,096 | 17 | 69,632 | 
| app_proof | 32 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 1,024 | 30 | 30,720 | 
| app_proof | 33 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 16,384 | 24 | 393,216 | 
| app_proof | 34 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 16,384 | 39 | 638,976 | 
| app_proof | 35 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 16,384 | 39 | 638,976 | 
| app_proof | 36 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 16 | 37 | 592 | 
| app_proof | 38 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 256 | 38 | 9,728 | 
| app_proof | 39 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | prover |  | 0 | 8 | 36 | 288 | 
| app_proof | 42 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 4,096 | 28 | 114,688 | 
| app_proof | 43 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 1,024 | 29 | 29,696 | 
| app_proof | 46 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 256 | 44 | 11,264 | 
| app_proof | 47 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 4,096 | 22 | 90,112 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2,048 | 529 | 1,083,392 | 
| app_proof | 51 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover |  | 0 | 2,048 | 58 | 118,784 | 
| app_proof | 52 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 128 | 33 | 4,224 | 
| app_proof | 53 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 512 | 28 | 14,336 | 
| app_proof | 54 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 16,384 | 42 | 688,128 | 
| app_proof | 55 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 4,096 | 29 | 118,784 | 
| app_proof | 56 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 57 | PhantomAir | prover |  | 0 | 16 | 6 | 96 | 
| app_proof | 58 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 2,048 | 300 | 614,400 | 
| app_proof | 59 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1,024 | 614 | 628,736 | 
| app_proof | 7 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 32 | 105 | 3,360 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 16 | 253 | 4,048 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 16 | 189 | 3,024 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 17,293 | 15,475 | 432,325 | 386,875 | 691,720 | 619,000 | 172,930 | 154,750 | 626,872 | 560,968 | 17,293 | 15,475 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 3,204 | 892 | 841,050 | 234,150 | 1,345,680 | 374,640 | 336,420 | 93,660 | 5,923,395 | 1,649,085 | 163,404 | 45,492 | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 21 | 11 | 13,283 | 6,957 | 21,252 | 11,132 | 5,313 | 2,783 | 148,444 | 77,756 | 4,095 | 2,145 | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 14 | 2 | 6,615 | 945 | 10,584 | 1,512 | 2,646 | 378 | 66,483 | 9,497 | 1,834 | 262 | 
| app_proof | 17 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 16 |  | 1,600 |  | 2,560 |  | 640 |  | 17,980 |  | 496 |  | 
| app_proof | 18 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 19 | KeccakfOpAir |  | 0 | 5 | 3 | 3,225 | 1,935 | 5,160 | 3,096 | 1,290 | 774 | 19,938 | 11,962 | 550 | 330 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 913 | 111 | 86,735 | 10,545 | 138,776 | 16,872 | 34,694 | 4,218 | 264,770 | 32,190 | 7,304 | 888 | 
| app_proof | 20 | KeccakfPermAir |  | 0 | 120 | 8 | 1,580,400 | 105,360 | 1,264,320 | 84,288 | 316,080 | 21,072 | 8,700 | 580 | 240 | 16 | 
| app_proof | 21 | XorinVmAir |  | 0 | 5 | 3 | 6,788 | 4,072 | 10,860 | 6,516 | 2,715 | 1,629 | 64,707 | 38,823 | 1,785 | 1,071 | 
| app_proof | 23 | HintStoreAir |  | 0 | 115 | 13 | 13,800 | 1,560 | 11,040 | 1,248 | 2,760 | 312 | 70,869 | 8,011 | 1,955 | 221 | 
| app_proof | 24 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 2,724 | 1,372 | 231,540 | 116,620 | 370,464 | 186,592 | 92,616 | 46,648 | 1,974,900 | 994,700 | 54,480 | 27,440 | 
| app_proof | 25 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 717 | 307 | 48,398 | 20,722 | 77,436 | 33,156 | 19,359 | 8,289 | 363,878 | 155,802 | 10,038 | 4,298 | 
| app_proof | 27 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 10,702 | 5,682 | 1,364,505 | 724,455 | 2,183,208 | 1,159,128 | 545,802 | 289,782 | 7,371,003 | 3,913,477 | 203,338 | 107,958 | 
| app_proof | 28 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 18,305 | 14,463 | 1,052,538 | 831,622 | 1,684,060 | 1,330,596 | 421,015 | 332,649 | 10,616,900 | 8,388,540 | 292,880 | 231,408 | 
| app_proof | 29 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 2,278 | 1,818 | 91,120 | 72,720 | 145,792 | 116,352 | 36,448 | 29,088 | 1,156,085 | 922,635 | 31,892 | 25,452 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 2,082 | 2,014 | 343,530 | 332,310 | 274,824 | 265,848 | 68,706 | 66,462 | 301,890 | 292,030 | 8,328 | 8,056 | 
| app_proof | 30 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 1,763 | 285 | 96,965 | 15,675 | 155,144 | 25,080 | 38,786 | 6,270 | 958,632 | 154,968 | 26,445 | 4,275 | 
| app_proof | 31 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 2,365 | 1,731 | 100,513 | 73,567 | 160,820 | 117,708 | 40,205 | 29,427 | 1,028,775 | 752,985 | 28,380 | 20,772 | 
| app_proof | 32 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 849 | 175 | 63,675 | 13,125 | 101,880 | 21,000 | 25,470 | 5,250 | 430,868 | 88,812 | 11,886 | 2,450 | 
| app_proof | 33 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 10,051 | 6,333 | 603,060 | 379,980 | 964,896 | 607,968 | 241,224 | 151,992 | 4,007,837 | 2,525,283 | 110,561 | 69,663 | 
| app_proof | 34 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 10,328 | 6,056 | 1,006,980 | 590,460 | 1,611,168 | 944,736 | 402,792 | 236,184 | 10,857,310 | 6,366,370 | 299,512 | 175,624 | 
| app_proof | 35 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 9,666 | 6,718 | 942,435 | 655,005 | 1,507,896 | 1,048,008 | 376,974 | 262,002 | 9,810,990 | 6,818,770 | 270,648 | 188,104 | 
| app_proof | 36 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 10 | 6 | 925 | 555 | 1,480 | 888 | 370 | 222 | 9,788 | 5,872 | 270 | 162 | 
| app_proof | 38 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 192 | 64 | 18,240 | 6,080 | 29,184 | 9,728 | 7,296 | 2,432 | 187,920 | 62,640 | 5,184 | 1,728 | 
| app_proof | 39 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 5 | 3 | 450 | 270 | 720 | 432 | 180 | 108 | 4,713 | 2,827 | 130 | 78 | 
| app_proof | 42 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 2,417 | 1,679 | 169,190 | 117,530 | 270,704 | 188,048 | 67,676 | 47,012 | 1,664,709 | 1,156,411 | 45,923 | 31,901 | 
| app_proof | 43 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 14,093 | 2,291 | 986,510 | 160,370 | 1,578,416 | 256,592 | 394,604 | 64,148 | 9,195,683 | 1,494,877 | 253,674 | 41,238 | 
| app_proof | 44 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 576 | 448 | 41,760 | 32,480 | 66,816 | 51,968 | 16,704 | 12,992 | 396,720 | 308,560 | 10,944 | 8,512 | 
| app_proof | 46 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 140 | 116 | 15,400 | 12,760 | 24,640 | 20,416 | 6,160 | 5,104 | 81,200 | 67,280 | 2,240 | 1,856 | 
| app_proof | 47 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 3,254 | 842 | 178,970 | 46,310 | 286,352 | 74,096 | 71,588 | 18,524 | 1,651,405 | 427,315 | 45,556 | 11,788 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1,271 | 777 | 1,680,898 | 1,027,582 | 2,689,436 | 1,644,132 | 672,359 | 411,033 | 21,378,220 | 13,069,140 | 589,744 | 360,528 | 
| app_proof | 51 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 0 | 1,365 | 683 | 197,925 | 99,035 | 316,680 | 158,456 | 79,170 | 39,614 | 1,187,550 | 594,210 | 32,760 | 16,392 | 
| app_proof | 52 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 86 | 42 | 7,095 | 3,465 | 11,352 | 5,544 | 2,838 | 1,386 | 56,115 | 27,405 | 1,548 | 756 | 
| app_proof | 53 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 267 | 245 | 18,690 | 17,150 | 29,904 | 27,440 | 7,476 | 6,860 | 164,539 | 150,981 | 4,539 | 4,165 | 
| app_proof | 54 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 11,943 | 4,441 | 1,254,015 | 466,305 | 2,006,424 | 746,088 | 501,606 | 186,522 | 9,957,477 | 3,702,683 | 274,689 | 102,143 | 
| app_proof | 55 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 2,765 | 1,331 | 200,463 | 96,497 | 320,740 | 154,396 | 80,185 | 38,599 | 1,904,394 | 916,726 | 52,535 | 25,289 | 
| app_proof | 56 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 57 | PhantomAir |  | 0 | 11 | 5 | 165 | 75 | 264 | 120 | 66 | 30 | 1,197 | 543 | 33 | 15 | 
| app_proof | 58 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 2,077 | 2,019 | 1,557,750 | 1,514,250 | 2,492,400 | 2,422,800 | 623,100 | 605,700 | 75,292 | 73,188 | 2,077 | 2,019 | 
| app_proof | 59 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 726 | 298 | 1,114,410 | 457,430 | 1,783,056 | 731,888 | 445,764 | 182,972 | 13,185,068 | 5,412,052 | 363,726 | 149,298 | 
| app_proof | 7 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 31 | 1 | 8,138 | 262 | 13,020 | 420 | 3,255 | 105 | 57,312 | 1,848 | 1,581 | 51 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 11 | 5 | 6,958 | 3,162 | 11,132 | 5,060 | 2,783 | 1,265 | 77,757 | 35,343 | 2,145 | 975 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 11 | 5 | 5,198 | 2,362 | 8,316 | 3,780 | 2,079 | 945 | 52,237 | 23,743 | 1,441 | 655 | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 15 | 163 | 15 | 5 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 12 | 120 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 36 | 195 | 36 | 6 | 0 | 6 | 7 | 7 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,560,317 | 147 | 33 | 0 | 0 | 73 | 26 | 25 | 33 | 13 | 0 | 40 | 31 | 9 | 2 | 6 | 33 | 33 | 73 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 107 | 20 | 0 | 0 | 55 | 21 | 20 | 23 | 11 | 0 | 31 | 23 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 97 | 15 | 0 | 0 | 54 | 21 | 20 | 21 | 11 | 0 | 27 | 20 | 7 | 1 | 5 | 15 | 15 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 44,071,357 | 159 | 31 | 0 | 0 | 86 | 36 | 34 | 24 | 26 | 0 | 40 | 31 | 9 | 2 | 6 | 31 | 31 | 86 | 0 | 3 | 25 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,803 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 16,495,299 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 7 | 0 | 7 | 7 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 14,231,316 | 147 | 12 | 0 | 0 | 107 | 72 | 71 | 16 | 18 | 0 | 27 | 18 | 9 | 1 | 7 | 12 | 12 | 107 | 0 | 1 | 17 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 6,360,594 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 222 | 6 | 112,210 | 18.62 | 0 | 234 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 0 | 2 | 36 | 222 | 3 | 0 | 0 | 29 | 0 | 0 | 0 | 217,919,416 | 74,773,669 | 292,693,085 | 2,097,152 | 2 | 0 | 0 | 3 | 32 | 7 | 112,210 | 15.44 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31537112787)
