| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  9.51 |  3.38 |  3.38 |
| app_proof |  7.32 |  1.70 |  1.70 |
| leaf |  1.63 |  1.11 |  1.11 |
| internal_for_leaf |  0.33 |  0.33 |  0.33 |
| internal_recursive.0 |  0.13 |  0.13 |  0.13 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,198.50 |  7,191 |  1,272 |  1,108 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  127 | -          | -          | -          |
| `execute_metered_insns` |  14,365,133 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  112.75 | -          |  112.75 |  112.75 |
| `set_initial_memory_time_ms` |  2.50 |  15 |  5 |  2 |
| `execute_preflight_insns` |  2,394,188.83 |  14,365,133 |  2,413,000 |  2,300,133 |
| `execute_preflight_time_ms` |  64.33 |  386 |  81 |  58 |
| `execute_preflight_insn_mi/s` |  37.22 | -          |  39.30 |  29.76 |
| `postflight_time_ms  ` |  28 |  168 |  38 |  25 |
| `postflight_memory_chronology_time_ms` |  3 |  18 |  6 |  2 |
| `postflight_program_index_time_ms` |  0 |  0 |  0 |  0 |
| `trace_gen_time_ms   ` |  13 |  78 |  18 |  12 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,076.67 |  6,460 |  1,112 |  999 |
| `prover.main_trace_commit_time_ms` |  333.17 |  1,999 |  368 |  263 |
| `prover.rap_constraints_time_ms` |  514.83 |  3,089 |  516 |  512 |
| `prover.openings_time_ms` |  228 |  1,368 |  231 |  224 |
| `prover.rap_constraints.logup_gkr_time_ms` |  105.67 |  634 |  106 |  104 |
| `prover.rap_constraints.round0_time_ms` |  283.67 |  1,702 |  285 |  283 |
| `prover.rap_constraints.mle_rounds_time_ms` |  124.67 |  748 |  125 |  124 |
| `prover.openings.stacked_reduction_time_ms` |  61.67 |  370 |  62 |  61 |
| `prover.openings.stacked_reduction.round0_time_ms` |  36.67 |  220 |  37 |  36 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24.50 |  147 |  25 |  24 |
| `prover.openings.whir_time_ms` |  165.50 |  993 |  168 |  162 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  813.50 |  1,627 |  1,113 |  514 |
| `execute_preflight_time_ms` |  5 |  10 |  5 |  5 |
| `trace_gen_time_ms   ` |  136.50 |  273 |  181 |  92 |
| `generate_blob_total_time_ms` |  16 |  32 |  21 |  11 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  676 |  1,352 |  931 |  421 |
| `prover.main_trace_commit_time_ms` |  269.50 |  539 |  391 |  148 |
| `prover.rap_constraints_time_ms` |  196 |  392 |  237 |  155 |
| `prover.openings_time_ms` |  210 |  420 |  303 |  117 |
| `prover.rap_constraints.logup_gkr_time_ms` |  38.50 |  77 |  41 |  36 |
| `prover.rap_constraints.round0_time_ms` |  97 |  194 |  122 |  72 |
| `prover.rap_constraints.mle_rounds_time_ms` |  58.50 |  117 |  72 |  45 |
| `prover.openings.stacked_reduction_time_ms` |  29.50 |  59 |  37 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  15 |  30 |  20 |  10 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  14 |  28 |  17 |  11 |
| `prover.openings.whir_time_ms` |  180 |  360 |  265 |  95 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  332 |  332 |  332 |  332 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  41 |  41 |  41 |  41 |
| `generate_blob_total_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  290 |  290 |  290 |  290 |
| `prover.main_trace_commit_time_ms` |  94 |  94 |  94 |  94 |
| `prover.rap_constraints_time_ms` |  113 |  113 |  113 |  113 |
| `prover.openings_time_ms` |  83 |  83 |  83 |  83 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  39 |  39 |  39 |  39 |
| `prover.rap_constraints.mle_rounds_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings.stacked_reduction_time_ms` |  15 |  15 |  15 |  15 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.whir_time_ms` |  67 |  67 |  67 |  67 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  125 |  125 |  125 |  125 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  111 |  111 |  111 |  111 |
| `prover.main_trace_commit_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints_time_ms` |  59 |  59 |  59 |  59 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  108 |  108 |  108 |  108 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  98 |  98 |  98 |  98 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ec7cb9c3272e9fec9aeee23432127eec0179fd48/keccak-ec7cb9c3272e9fec9aeee23432127eec0179fd48.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 9.51 | app_proof.prover..4 |
| prover.prove_whir_opening | 7.43 | leaf.0.prover |
| prover.openings | 7.43 | leaf.0.prover |
| prover.merkle_tree | 7.43 | leaf.0.prover |
| prover.rs_code_matrix | 7.42 | leaf.0.prover |
| prover.rap_constraints | 7.00 | app_proof.prover..4 |
| prover.batch_constraints.before_round0 | 6.25 | app_proof.prover..4 |
| frac_sumcheck.gkr_rounds | 6.25 | app_proof.prover..4 |
| frac_sumcheck.segment_tree | 6.20 | app_proof.prover..4 |
| prover.gkr_input_evals | 6.20 | app_proof.prover..4 |
| prover.batch_constraints.round0 | 5.89 | app_proof.prover..4 |
| prover.batch_constraints.fold_ple_evals | 5.89 | app_proof.prover..4 |
| postflight | 4.47 | app_proof..0 |
| generate mem proving ctxs | 4.30 | app_proof..0 |
| tracegen | 4.30 | app_proof..0 |
| set initial memory | 4.12 | app_proof..0 |
| prover.before_gkr_input_evals | 3.27 | app_proof.prover..4 |
| tracegen.pow_checker | 2.08 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.08 | leaf.0 |
| tracegen.exp_bits_len | 2.08 | leaf.0 |
| tracegen.whir_folding | 1.82 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.82 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.81 | leaf.0 |
| tracegen.proof_shape | 1.47 | leaf.0 |
| tracegen.range_checker | 1.47 | leaf.0 |
| tracegen.public_values | 1.47 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 124 | 267,407 | 229,134 | 0 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 9 | 11 | 3 | 
| 10 | KeccakfOpAir |  | 135 | 27 | 3 | 
| 11 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 12 | XorinVmAir |  | 408 | 87 | 3 | 
| 13 | RevealAir |  | 25 | 3 | 2 | 
| 14 | HintStoreAir | 1 | 18 | 15 | 3 | 
| 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 10 | 11 | 2 | 
| 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 12 | 4 | 3 | 
| 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 8 | 3 | 
| 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 28 | 9 | 3 | 
| 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 27 | 12 | 3 | 
| 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 26 | 9 | 3 | 
| 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 25 | 12 | 3 | 
| 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 26 | 13 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 25 | 9 | 3 | 
| 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 24 | 12 | 3 | 
| 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 25 | 13 | 3 | 
| 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 8 | 3 | 
| 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 11 | 3 | 
| 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 12 | 3 | 
| 36 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 39 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 4 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 40 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 41 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 47 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 48 | PhantomAir |  | 3 | 1 | 2 | 
| 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 50 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 7 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 66 |  |  |  | 322 |  | 
| app_proof | 0 |  |  |  | 4 |  |  | 
| internal_for_leaf |  |  |  | 332 |  |  | 332 | 
| internal_recursive.0 |  |  |  | 125 |  |  | 125 | 
| internal_recursive.1 |  |  |  | 109 |  |  | 109 | 
| leaf |  |  | 514 |  |  |  | 1,628 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 1 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 6 | 
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
| app_proof | XorinVmAir |  | 0 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 1 | 0 | 
| app_proof | HintStoreAir |  | 1 | 0 | 
| app_proof | KeccakfOpAir |  | 1 | 0 | 
| app_proof | KeccakfPermAir |  | 1 | 0 | 
| app_proof | PhantomAir |  | 1 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 1 | 0 | 
| app_proof | RevealAir |  | 1 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 1 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 1 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 0 | 
| app_proof | XorinVmAir |  | 1 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 2 | 0 | 
| app_proof | HintStoreAir |  | 2 | 0 | 
| app_proof | KeccakfOpAir |  | 2 | 0 | 
| app_proof | KeccakfPermAir |  | 2 | 0 | 
| app_proof | PhantomAir |  | 2 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 2 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 2 | 0 | 
| app_proof | RevealAir |  | 2 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 2 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 2 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 2 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 2 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 2 | 0 | 
| app_proof | XorinVmAir |  | 2 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 3 | 0 | 
| app_proof | HintStoreAir |  | 3 | 0 | 
| app_proof | KeccakfOpAir |  | 3 | 0 | 
| app_proof | KeccakfPermAir |  | 3 | 0 | 
| app_proof | PhantomAir |  | 3 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 3 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 3 | 0 | 
| app_proof | RevealAir |  | 3 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 3 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 3 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 3 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 3 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 3 | 0 | 
| app_proof | XorinVmAir |  | 3 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 4 | 0 | 
| app_proof | HintStoreAir |  | 4 | 0 | 
| app_proof | KeccakfOpAir |  | 4 | 0 | 
| app_proof | KeccakfPermAir |  | 4 | 0 | 
| app_proof | PhantomAir |  | 4 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 4 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 4 | 0 | 
| app_proof | RevealAir |  | 4 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 4 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 4 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 4 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 4 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 4 | 0 | 
| app_proof | XorinVmAir |  | 4 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 5 | 0 | 
| app_proof | HintStoreAir |  | 5 | 0 | 
| app_proof | KeccakfOpAir |  | 5 | 0 | 
| app_proof | KeccakfPermAir |  | 5 | 0 | 
| app_proof | PhantomAir |  | 5 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 5 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 5 | 0 | 
| app_proof | RevealAir |  | 5 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 5 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 5 | 0 | 
| app_proof | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 5 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 5 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 5 | 0 | 
| app_proof | XorinVmAir |  | 5 | 0 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 70 | 217 | 4 | 
| agg_keygen | 1 | VmPvsAir | 1 | 32 | 57 | 4 | 
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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 71 | 142 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 34 | 68 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 45 | 5,760 | 
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
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 34 | 34 | 
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
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 71 | 71 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 34 | 34 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 71 | 284 | 
| leaf | 0 | VerifierPvsAir | 1 | prover | 2 | 71 | 142 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 34 | 136 | 
| leaf | 1 | VmPvsAir | 1 | prover | 2 | 34 | 68 | 
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
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 256 | 46 | 11,776 | 
| leaf | 19 | ProofShapeAir<4, 8> | 1 | prover | 128 | 46 | 5,888 | 
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
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 7,864,320 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 131,072 | 60 | 7,864,320 | 
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
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| leaf | 5 | UnivariateSumcheckAir | 1 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 6 | MultilinearSumcheckAir | 1 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 7 | EqNsAir | 1 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 131,072 | 25 | 3,276,800 | 
| leaf | 8 | Eq3bAir | 1 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 9 | EqSharpUniAir | 1 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 0 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 0 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 41 | 21,495,808 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 1 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 1 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 1 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 1 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 1 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 1 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 1 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 1 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 1 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 1 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 1 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 1 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 1 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 1 | 524,288 | 41 | 21,495,808 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 1 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 1 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 1 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 1 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 1 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 1 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 1 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 1 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 1 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 1 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 2 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 2 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 2 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 2 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 2 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 2 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 2 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 2 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 2 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 2 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 2 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 2 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 2 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 2 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 2 | 524,288 | 41 | 21,495,808 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 2 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 2 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 2 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 2 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 2 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 2 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 2 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 2 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 2 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 2 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 3 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 3 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 3 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 3 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 3 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 3 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 3 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 3 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 3 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 3 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 3 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 3 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 3 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 3 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 3 | 524,288 | 41 | 21,495,808 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 3 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 3 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 3 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 3 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 3 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 3 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 3 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 3 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 3 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 3 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 3 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 4 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 4 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 4 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 4 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 4 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 4 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 4 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 4 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 4 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 4 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 4 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 4 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 4 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 4 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 4 | 524,288 | 41 | 21,495,808 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 4 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 4 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 4 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 4 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 4 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 4 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 4 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 4 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 4 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 4 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 4 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 4 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 5 | 4,096 | 11 | 45,056 | 
| app_proof | 1 | VmConnectorAir | prover |  | 5 | 2 | 7 | 14 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 5 | 16,384 | 285 | 4,669,440 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 5 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 5 | 16,384 | 597 | 9,781,248 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 5 | 262,144 | 35 | 9,175,040 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 5 | 65,536 | 52 | 3,407,872 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 5 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 5 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 5 | 65,536 | 16 | 1,048,576 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 5 | 131,072 | 23 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 5 | 65,536 | 31 | 2,031,616 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 5 | 262,144 | 25 | 6,553,600 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 5 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 5 | 262,144 | 41 | 10,747,904 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 5 | 16,384 | 40 | 655,360 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 5 | 256 | 33 | 8,448 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 5 | 65,536 | 30 | 1,966,080 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 5 | 32,768 | 30 | 983,040 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 5 | 16,384 | 23 | 376,832 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 5 | 16,384 | 29 | 475,136 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 5 | 32,768 | 43 | 1,409,024 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 5 | 131,072 | 30 | 3,932,160 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 5 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 5 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 5 | 16,384 | 41 | 671,744 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 5 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 0 | 10,919 | 5,465 | 7,779,788 | 3,893,812 | 12,447,660 | 6,230,100 | 3,111,915 | 1,557,525 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 0 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 0 | 10,918 | 5,466 | 16,295,115 | 8,158,005 | 26,072,184 | 13,052,808 | 6,518,046 | 3,263,202 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 152,859 | 109,285 | 13,375,163 | 9,562,437 | 21,400,260 | 15,299,900 | 5,350,065 | 3,824,975 | 110,822,775 | 79,231,625 | 3,057,180 | 2,185,700 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 54,593 | 10,943 | 7,097,090 | 1,422,590 | 11,355,344 | 2,276,144 | 2,838,836 | 569,036 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 567,762 | 480,814 | 34,065,720 | 28,848,840 | 54,505,152 | 46,158,144 | 13,626,288 | 11,539,536 | 329,301,960 | 278,872,120 | 9,084,192 | 7,693,024 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 46 | 18 | 4,485 | 1,755 | 7,176 | 2,808 | 1,794 | 702 | 16,675 | 6,525 | 460 | 180 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 54,596 | 10,940 | 2,183,840 | 437,600 | 3,494,144 | 700,160 | 873,536 | 175,040 | 23,749,260 | 4,758,900 | 655,152 | 131,280 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 98,266 | 32,806 | 5,650,295 | 1,886,345 | 9,040,472 | 3,018,152 | 2,260,118 | 754,538 | 53,432,138 | 17,838,262 | 1,473,990 | 492,090 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 65,510 | 26 | 2,947,950 | 1,170 | 4,716,720 | 1,872 | 1,179,180 | 468 | 28,496,850 | 11,310 | 786,120 | 312 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 65,511 | 25 | 5,077,103 | 1,937 | 8,123,364 | 3,100 | 2,030,841 | 775 | 33,246,833 | 12,687 | 917,154 | 350 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 218,369 | 43,775 | 13,648,063 | 2,735,937 | 21,836,900 | 4,377,500 | 5,459,225 | 1,094,375 | 87,074,639 | 17,455,281 | 2,402,059 | 481,525 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 589,612 | 458,964 | 60,435,230 | 47,043,810 | 96,696,368 | 75,270,096 | 24,174,092 | 18,817,524 | 598,456,180 | 465,848,460 | 16,509,136 | 12,850,992 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 272,960 | 251,328 | 27,978,400 | 25,761,120 | 44,765,440 | 41,217,792 | 11,191,360 | 10,304,448 | 267,159,600 | 245,987,280 | 7,369,920 | 6,785,856 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 10,918 | 5,466 | 1,091,800 | 546,600 | 1,746,880 | 874,560 | 436,720 | 218,640 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 222 | 34 | 36,630 | 5,610 | 29,304 | 4,488 | 7,326 | 1,122 | 32,190 | 4,930 | 888 | 136 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 43,676 | 21,860 | 3,275,700 | 1,639,500 | 5,241,120 | 2,623,200 | 1,310,280 | 655,800 | 30,081,845 | 15,056,075 | 829,844 | 415,340 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 32,757 | 11 | 2,456,775 | 825 | 3,930,840 | 1,320 | 982,710 | 330 | 21,373,943 | 7,177 | 589,626 | 198 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 10,919 | 5,465 | 627,843 | 314,237 | 1,004,548 | 502,780 | 251,137 | 125,695 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 10,918 | 5,466 | 791,555 | 396,285 | 1,266,488 | 634,056 | 316,622 | 158,514 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 21,837 | 10,931 | 2,347,478 | 1,175,082 | 3,755,964 | 1,880,132 | 938,991 | 470,033 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 109,181 | 21,891 | 8,188,575 | 1,641,825 | 13,101,720 | 2,626,920 | 3,275,430 | 656,730 | 75,198,414 | 15,077,426 | 2,074,439 | 415,929 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 221 | 35 | 165,750 | 26,250 | 265,200 | 42,000 | 66,300 | 10,500 | 8,012 | 1,268 | 221 | 35 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 10,919 | 5,465 | 1,119,198 | 560,162 | 1,790,716 | 896,260 | 447,679 | 224,065 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 1 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 1 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 1 | 10,919 | 5,465 | 7,779,788 | 3,893,812 | 12,447,660 | 6,230,100 | 3,111,915 | 1,557,525 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 1 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 1 | 10,919 | 5,465 | 16,296,608 | 8,156,512 | 26,074,572 | 13,050,420 | 6,518,643 | 3,262,605 | 161,492,010 | 80,827,350 | 4,454,952 | 2,229,720 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 152,860 | 109,284 | 13,375,250 | 9,562,350 | 21,400,400 | 15,299,760 | 5,350,100 | 3,824,940 | 110,823,500 | 79,230,900 | 3,057,200 | 2,185,680 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 54,592 | 10,944 | 7,096,960 | 1,422,720 | 11,355,136 | 2,276,352 | 2,838,784 | 569,088 | 37,600,240 | 7,537,680 | 1,037,248 | 207,936 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 567,770 | 480,806 | 34,066,200 | 28,848,360 | 54,505,920 | 46,157,376 | 13,626,480 | 11,539,344 | 329,306,600 | 278,867,480 | 9,084,320 | 7,692,896 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 1 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 1 | 54,594 | 10,942 | 2,183,760 | 437,680 | 3,494,016 | 700,288 | 873,504 | 175,072 | 23,748,390 | 4,759,770 | 655,128 | 131,304 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 1 | 98,267 | 32,805 | 5,650,353 | 1,886,287 | 9,040,564 | 3,018,060 | 2,260,141 | 754,515 | 53,432,682 | 17,837,718 | 1,474,005 | 492,075 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 1 | 65,512 | 24 | 2,948,040 | 1,080 | 4,716,864 | 1,728 | 1,179,216 | 432 | 28,497,720 | 10,440 | 786,144 | 288 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 65,512 | 24 | 5,077,180 | 1,860 | 8,123,488 | 2,976 | 2,030,872 | 744 | 33,247,340 | 12,180 | 917,168 | 336 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 218,370 | 43,774 | 13,648,125 | 2,735,875 | 21,837,000 | 4,377,400 | 5,459,250 | 1,094,350 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 589,592 | 458,984 | 60,433,180 | 47,045,860 | 96,693,088 | 75,273,376 | 24,173,272 | 18,818,344 | 598,435,880 | 465,868,760 | 16,508,576 | 12,851,552 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 272,962 | 251,326 | 27,978,605 | 25,760,915 | 44,765,768 | 41,217,464 | 11,191,442 | 10,304,366 | 267,161,558 | 245,985,322 | 7,369,974 | 6,785,802 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 10,919 | 5,465 | 1,091,900 | 546,500 | 1,747,040 | 874,400 | 436,760 | 218,600 | 10,291,158 | 5,150,762 | 283,894 | 142,090 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 1 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 43,675 | 21,861 | 3,275,625 | 1,639,575 | 5,241,000 | 2,623,320 | 1,310,250 | 655,830 | 30,081,157 | 15,056,763 | 829,825 | 415,359 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 1 | 32,757 | 11 | 2,456,775 | 825 | 3,930,840 | 1,320 | 982,710 | 330 | 21,373,943 | 7,177 | 589,626 | 198 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 1 | 10,918 | 5,466 | 627,785 | 314,295 | 1,004,456 | 502,872 | 251,114 | 125,718 | 5,540,885 | 2,773,995 | 152,852 | 76,524 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 10,919 | 5,465 | 791,628 | 396,212 | 1,266,604 | 633,940 | 316,651 | 158,485 | 6,728,834 | 3,367,806 | 185,623 | 92,905 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 21,837 | 10,931 | 2,347,478 | 1,175,082 | 3,755,964 | 1,880,132 | 938,991 | 470,033 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 109,188 | 21,884 | 8,189,100 | 1,641,300 | 13,102,560 | 2,626,080 | 3,275,640 | 656,520 | 75,203,235 | 15,072,605 | 2,074,572 | 415,796 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 1 | 10,918 | 5,466 | 1,119,095 | 560,265 | 1,790,552 | 896,424 | 447,638 | 224,106 | 12,269,103 | 6,142,417 | 338,458 | 169,446 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 2 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 2 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 2 | 10,918 | 5,466 | 7,779,075 | 3,894,525 | 12,446,520 | 6,231,240 | 3,111,630 | 1,557,810 | 53,429,963 | 26,749,237 | 1,473,930 | 737,910 | 
| app_proof | 11 | KeccakfPermAir |  | 2 | 262,032 | 112 | 3,450,961,440 | 1,475,040 | 2,760,769,152 | 1,180,032 | 690,192,288 | 295,008 | 18,997,320 | 8,120 | 524,064 | 224 | 
| app_proof | 12 | XorinVmAir |  | 2 | 10,918 | 5,466 | 16,295,115 | 8,158,005 | 26,072,184 | 13,052,808 | 6,518,046 | 3,263,202 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 2 | 152,859 | 109,285 | 13,375,163 | 9,562,437 | 21,400,260 | 15,299,900 | 5,350,065 | 3,824,975 | 110,822,775 | 79,231,625 | 3,057,180 | 2,185,700 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 2 | 54,593 | 10,943 | 7,097,090 | 1,422,590 | 11,355,344 | 2,276,144 | 2,838,836 | 569,036 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 2 | 567,761 | 480,815 | 34,065,660 | 28,848,900 | 54,505,056 | 46,158,240 | 13,626,264 | 11,539,560 | 329,301,380 | 278,872,700 | 9,084,176 | 7,693,040 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 2 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 2 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 23,747,955 | 4,760,205 | 655,116 | 131,316 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 2 | 98,267 | 32,805 | 5,650,353 | 1,886,287 | 9,040,564 | 3,018,060 | 2,260,141 | 754,515 | 53,432,682 | 17,837,718 | 1,474,005 | 492,075 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 2 | 65,510 | 26 | 2,947,950 | 1,170 | 4,716,720 | 1,872 | 1,179,180 | 468 | 28,496,850 | 11,310 | 786,120 | 312 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 2 | 65,510 | 26 | 5,077,025 | 2,015 | 8,123,240 | 3,224 | 2,030,810 | 806 | 33,246,325 | 13,195 | 917,140 | 364 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 2 | 218,370 | 43,774 | 13,648,125 | 2,735,875 | 21,837,000 | 4,377,400 | 5,459,250 | 1,094,350 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 2 | 589,616 | 458,960 | 60,435,640 | 47,043,400 | 96,697,024 | 75,269,440 | 24,174,256 | 18,817,360 | 598,460,240 | 465,844,400 | 16,509,248 | 12,850,880 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 2 | 272,963 | 251,325 | 27,978,708 | 25,760,812 | 44,765,932 | 41,217,300 | 11,191,483 | 10,304,325 | 267,162,537 | 245,984,343 | 7,370,001 | 6,785,775 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 2 | 10,918 | 5,466 | 1,091,800 | 546,600 | 1,746,880 | 874,560 | 436,720 | 218,640 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 2 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 2 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 2 | 32,755 | 13 | 2,456,625 | 975 | 3,930,600 | 1,560 | 982,650 | 390 | 21,372,638 | 8,482 | 589,590 | 234 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 2 | 10,919 | 5,465 | 627,843 | 314,237 | 1,004,548 | 502,780 | 251,137 | 125,695 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 2 | 10,918 | 5,466 | 791,555 | 396,285 | 1,266,488 | 634,056 | 316,622 | 158,514 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 2 | 21,837 | 10,931 | 2,347,478 | 1,175,082 | 3,755,964 | 1,880,132 | 938,991 | 470,033 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 2 | 109,182 | 21,890 | 8,188,650 | 1,641,750 | 13,101,840 | 2,626,800 | 3,275,460 | 656,700 | 75,199,103 | 15,076,737 | 2,074,458 | 415,910 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 2 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 2 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 2 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 2 | 10,919 | 5,465 | 1,119,198 | 560,162 | 1,790,716 | 896,260 | 447,679 | 224,065 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 2 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 3 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 3 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 3 | 10,919 | 5,465 | 7,779,788 | 3,893,812 | 12,447,660 | 6,230,100 | 3,111,915 | 1,557,525 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 3 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 3 | 10,919 | 5,465 | 16,296,608 | 8,156,512 | 26,074,572 | 13,050,420 | 6,518,643 | 3,262,605 | 161,492,010 | 80,827,350 | 4,454,952 | 2,229,720 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 3 | 152,863 | 109,281 | 13,375,513 | 9,562,087 | 21,400,820 | 15,299,340 | 5,350,205 | 3,824,835 | 110,825,675 | 79,228,725 | 3,057,260 | 2,185,620 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 3 | 54,593 | 10,943 | 7,097,090 | 1,422,590 | 11,355,344 | 2,276,144 | 2,838,836 | 569,036 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 3 | 567,768 | 480,808 | 34,066,080 | 28,848,480 | 54,505,728 | 46,157,568 | 13,626,432 | 11,539,392 | 329,305,440 | 278,868,640 | 9,084,288 | 7,692,928 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 3 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 3 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 23,747,955 | 4,760,205 | 655,116 | 131,316 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 3 | 98,268 | 32,804 | 5,650,410 | 1,886,230 | 9,040,656 | 3,017,968 | 2,260,164 | 754,492 | 53,433,225 | 17,837,175 | 1,474,020 | 492,060 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 3 | 65,514 | 22 | 2,948,130 | 990 | 4,717,008 | 1,584 | 1,179,252 | 396 | 28,498,590 | 9,570 | 786,168 | 264 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 3 | 65,513 | 23 | 5,077,258 | 1,782 | 8,123,612 | 2,852 | 2,030,903 | 713 | 33,247,848 | 11,672 | 917,182 | 322 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 3 | 218,372 | 43,772 | 13,648,250 | 2,735,750 | 21,837,200 | 4,377,200 | 5,459,300 | 1,094,300 | 87,075,835 | 17,454,085 | 2,402,092 | 481,492 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 3 | 589,583 | 458,993 | 60,432,258 | 47,046,782 | 96,691,612 | 75,274,852 | 24,172,903 | 18,818,713 | 598,426,745 | 465,877,895 | 16,508,324 | 12,851,804 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 3 | 272,962 | 251,326 | 27,978,605 | 25,760,915 | 44,765,768 | 41,217,464 | 11,191,442 | 10,304,366 | 267,161,558 | 245,985,322 | 7,369,974 | 6,785,802 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 3 | 10,919 | 5,465 | 1,091,900 | 546,500 | 1,747,040 | 874,400 | 436,760 | 218,600 | 10,291,158 | 5,150,762 | 283,894 | 142,090 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 3 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 3 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 3 | 32,756 | 12 | 2,456,700 | 900 | 3,930,720 | 1,440 | 982,680 | 360 | 21,373,290 | 7,830 | 589,608 | 216 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 3 | 10,919 | 5,465 | 627,843 | 314,237 | 1,004,548 | 502,780 | 251,137 | 125,695 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 3 | 10,919 | 5,465 | 791,628 | 396,212 | 1,266,604 | 633,940 | 316,651 | 158,485 | 6,728,834 | 3,367,806 | 185,623 | 92,905 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 3 | 21,837 | 10,931 | 2,347,478 | 1,175,082 | 3,755,964 | 1,880,132 | 938,991 | 470,033 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 3 | 109,190 | 21,882 | 8,189,250 | 1,641,150 | 13,102,800 | 2,625,840 | 3,275,700 | 656,460 | 75,204,613 | 15,071,227 | 2,074,610 | 415,758 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 3 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 3 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 3 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 3 | 10,919 | 5,465 | 1,119,198 | 560,162 | 1,790,716 | 896,260 | 447,679 | 224,065 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 3 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 4 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 4 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 4 | 10,918 | 5,466 | 7,779,075 | 3,894,525 | 12,446,520 | 6,231,240 | 3,111,630 | 1,557,810 | 53,429,963 | 26,749,237 | 1,473,930 | 737,910 | 
| app_proof | 11 | KeccakfPermAir |  | 4 | 262,032 | 112 | 3,450,961,440 | 1,475,040 | 2,760,769,152 | 1,180,032 | 690,192,288 | 295,008 | 18,997,320 | 8,120 | 524,064 | 224 | 
| app_proof | 12 | XorinVmAir |  | 4 | 10,918 | 5,466 | 16,295,115 | 8,158,005 | 26,072,184 | 13,052,808 | 6,518,046 | 3,263,202 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 4 | 152,855 | 109,289 | 13,374,813 | 9,562,787 | 21,399,700 | 15,300,460 | 5,349,925 | 3,825,115 | 110,819,875 | 79,234,525 | 3,057,100 | 2,185,780 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 4 | 54,592 | 10,944 | 7,096,960 | 1,422,720 | 11,355,136 | 2,276,352 | 2,838,784 | 569,088 | 37,600,240 | 7,537,680 | 1,037,248 | 207,936 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 4 | 567,765 | 480,811 | 34,065,900 | 28,848,660 | 54,505,440 | 46,157,856 | 13,626,360 | 11,539,464 | 329,303,700 | 278,870,380 | 9,084,240 | 7,692,976 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 4 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 4 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 23,747,955 | 4,760,205 | 655,116 | 131,316 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 4 | 98,266 | 32,806 | 5,650,295 | 1,886,345 | 9,040,472 | 3,018,152 | 2,260,118 | 754,538 | 53,432,138 | 17,838,262 | 1,473,990 | 492,090 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 4 | 65,509 | 27 | 2,947,905 | 1,215 | 4,716,648 | 1,944 | 1,179,162 | 486 | 28,496,415 | 11,745 | 786,108 | 324 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 4 | 65,511 | 25 | 5,077,103 | 1,937 | 8,123,364 | 3,100 | 2,030,841 | 775 | 33,246,833 | 12,687 | 917,154 | 350 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 4 | 218,370 | 43,774 | 13,648,125 | 2,735,875 | 21,837,000 | 4,377,400 | 5,459,250 | 1,094,350 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 4 | 589,619 | 458,957 | 60,435,948 | 47,043,092 | 96,697,516 | 75,268,948 | 24,174,379 | 18,817,237 | 598,463,285 | 465,841,355 | 16,509,332 | 12,850,796 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 4 | 272,964 | 251,324 | 27,978,810 | 25,760,710 | 44,766,096 | 41,217,136 | 11,191,524 | 10,304,284 | 267,163,515 | 245,983,365 | 7,370,028 | 6,785,748 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 4 | 10,918 | 5,466 | 1,091,800 | 546,600 | 1,746,880 | 874,560 | 436,720 | 218,640 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 4 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 4 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 4 | 32,755 | 13 | 2,456,625 | 975 | 3,930,600 | 1,560 | 982,650 | 390 | 21,372,638 | 8,482 | 589,590 | 234 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 4 | 10,918 | 5,466 | 627,785 | 314,295 | 1,004,456 | 502,872 | 251,114 | 125,718 | 5,540,885 | 2,773,995 | 152,852 | 76,524 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 4 | 10,918 | 5,466 | 791,555 | 396,285 | 1,266,488 | 634,056 | 316,622 | 158,514 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 4 | 21,837 | 10,931 | 2,347,478 | 1,175,082 | 3,755,964 | 1,880,132 | 938,991 | 470,033 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 4 | 109,182 | 21,890 | 8,188,650 | 1,641,750 | 13,101,840 | 2,626,800 | 3,275,460 | 656,700 | 75,199,103 | 15,076,737 | 2,074,458 | 415,910 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 4 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 4 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 4 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 4 | 10,918 | 5,466 | 1,119,095 | 560,265 | 1,790,552 | 896,424 | 447,638 | 224,106 | 12,269,103 | 6,142,417 | 338,458 | 169,446 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 4 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 5 | 2,084 | 2,012 | 57,310 | 55,330 | 91,696 | 88,528 | 22,924 | 22,132 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 5 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | KeccakfOpAir |  | 5 | 10,408 | 5,976 | 7,415,700 | 4,257,900 | 11,865,120 | 6,812,640 | 2,966,280 | 1,703,160 | 50,934,150 | 29,245,050 | 1,405,080 | 806,760 | 
| app_proof | 11 | KeccakfPermAir |  | 5 | 249,792 | 12,352 | 3,289,760,640 | 162,675,840 | 2,631,808,512 | 130,140,672 | 657,952,128 | 32,535,168 | 18,109,920 | 895,520 | 499,584 | 24,704 | 
| app_proof | 12 | XorinVmAir |  | 5 | 10,408 | 5,976 | 15,533,940 | 8,919,180 | 24,854,304 | 14,270,688 | 6,213,576 | 3,567,672 | 153,934,320 | 88,385,040 | 4,246,464 | 2,438,208 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 5 | 145,711 | 116,433 | 12,749,713 | 10,187,887 | 20,399,540 | 16,300,620 | 5,099,885 | 4,075,155 | 105,640,475 | 84,413,925 | 2,914,220 | 2,328,660 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 5 | 52,040 | 13,496 | 6,765,200 | 1,754,480 | 10,824,320 | 2,807,168 | 2,706,080 | 701,792 | 35,842,550 | 9,295,370 | 988,760 | 256,424 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 5 | 541,202 | 507,374 | 32,472,120 | 30,442,440 | 51,955,392 | 48,707,904 | 12,988,848 | 12,176,976 | 313,897,160 | 294,276,920 | 8,659,232 | 8,117,984 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 5 | 44 | 20 | 4,290 | 1,950 | 6,864 | 3,120 | 1,716 | 780 | 15,950 | 7,250 | 440 | 200 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 5 | 52,037 | 13,499 | 2,081,480 | 539,960 | 3,330,368 | 863,936 | 832,592 | 215,984 | 22,636,095 | 5,872,065 | 624,444 | 161,988 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 5 | 93,671 | 37,401 | 5,386,083 | 2,150,557 | 8,617,732 | 3,440,892 | 2,154,433 | 860,223 | 50,933,607 | 20,336,793 | 1,405,065 | 561,015 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 5 | 62,447 | 3,089 | 2,810,115 | 139,005 | 4,496,184 | 222,408 | 1,124,046 | 55,602 | 27,164,445 | 1,343,715 | 749,364 | 37,068 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 5 | 62,446 | 3,090 | 4,839,565 | 239,475 | 7,743,304 | 383,160 | 1,935,826 | 95,790 | 31,691,345 | 1,568,175 | 874,244 | 43,260 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 5 | 208,158 | 53,986 | 13,009,875 | 3,374,125 | 20,815,800 | 5,398,600 | 5,203,950 | 1,349,650 | 83,003,003 | 21,526,917 | 2,289,738 | 593,846 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 5 | 562,019 | 486,557 | 57,606,948 | 49,872,092 | 92,171,116 | 79,795,348 | 23,042,779 | 19,948,837 | 570,449,285 | 493,855,355 | 15,736,532 | 13,623,596 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 5 | 260,205 | 1,939 | 26,671,013 | 198,747 | 42,673,620 | 317,996 | 10,668,405 | 79,499 | 254,675,644 | 1,897,796 | 7,025,535 | 52,353 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 5 | 10,408 | 5,976 | 1,040,800 | 597,600 | 1,665,280 | 956,160 | 416,320 | 239,040 | 9,809,540 | 5,632,380 | 270,608 | 155,376 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 5 | 222 | 34 | 36,630 | 5,610 | 29,304 | 4,488 | 7,326 | 1,122 | 32,190 | 4,930 | 888 | 136 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 5 | 41,631 | 23,905 | 3,122,325 | 1,792,875 | 4,995,720 | 2,868,600 | 1,248,930 | 717,150 | 28,673,352 | 16,464,568 | 790,989 | 454,195 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 5 | 31,223 | 1,545 | 2,341,725 | 115,875 | 3,746,760 | 185,400 | 936,690 | 46,350 | 20,373,008 | 1,008,112 | 562,014 | 27,810 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 5 | 10,408 | 5,976 | 598,460 | 343,620 | 957,536 | 549,792 | 239,384 | 137,448 | 5,282,060 | 3,032,820 | 145,712 | 83,664 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 5 | 10,408 | 5,976 | 754,580 | 433,260 | 1,207,328 | 693,216 | 301,832 | 173,304 | 6,413,930 | 3,682,710 | 176,936 | 101,592 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 5 | 20,816 | 11,952 | 2,237,720 | 1,284,840 | 3,580,352 | 2,055,744 | 895,088 | 513,936 | 17,355,340 | 9,964,980 | 478,768 | 274,896 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 5 | 104,078 | 26,994 | 7,805,850 | 2,024,550 | 12,489,360 | 3,239,280 | 3,122,340 | 809,820 | 71,683,723 | 18,592,117 | 1,977,482 | 512,886 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 5 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 5 | 310 | 202 | 232,500 | 151,500 | 372,000 | 242,400 | 93,000 | 60,600 | 11,238 | 7,322 | 310 | 202 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 5 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 5 | 10,408 | 5,976 | 1,066,820 | 612,540 | 1,706,912 | 980,064 | 426,728 | 245,016 | 11,695,990 | 6,715,530 | 322,648 | 185,256 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 5 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 41 | 332 | 41 | 13 | 3 | 2 | 3 | 3 | 
| internal_recursive.0 | 1 | 14 | 125 | 13 | 2 | 0 | 2 | 2 | 2 | 
| internal_recursive.1 | 1 | 10 | 108 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 181 | 1,113 | 181 | 58 | 21 | 5 | 10 | 10 | 
| leaf | 1 | 92 | 514 | 92 | 29 | 11 | 5 | 11 | 11 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 81,415,164 | 290 | 93 | 0 | 0 | 113 | 39 | 38 | 56 | 17 | 0 | 83 | 67 | 15 | 5 | 10 | 94 | 93 | 113 | 0 | 1 | 16 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,947 | 111 | 21 | 0 | 0 | 59 | 21 | 21 | 24 | 12 | 0 | 30 | 21 | 8 | 1 | 6 | 21 | 21 | 59 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,867 | 98 | 15 | 0 | 0 | 53 | 20 | 19 | 21 | 10 | 0 | 29 | 22 | 7 | 1 | 5 | 15 | 15 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 480,510,966 | 931 | 390 | 0 | 67 | 237 | 122 | 121 | 72 | 41 | 0 | 303 | 265 | 37 | 20 | 17 | 391 | 390 | 237 | 0 | 3 | 41 | 0 | 0 | 
| leaf | 1 | prover | 244,187,964 | 421 | 148 | 0 | 0 | 155 | 72 | 71 | 45 | 36 | 0 | 117 | 95 | 22 | 10 | 11 | 148 | 148 | 155 | 0 | 3 | 35 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 7,020,873 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,377 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,361 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 31,478,993 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 19,147,529 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 7 | 0 | 7 | 7 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 836,978,382 | 1,112 | 368 | 0 | 0 | 516 | 285 | 285 | 124 | 106 | 0 | 227 | 165 | 61 | 36 | 24 | 368 | 368 | 516 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 1 | 836,978,382 | 1,091 | 347 | 0 | 0 | 515 | 283 | 283 | 125 | 106 | 0 | 228 | 166 | 62 | 37 | 24 | 347 | 347 | 515 | 0 | 1 | 105 | 0 | 0 | 
| app_proof | prover |  | 2 | 836,978,382 | 1,086 | 340 | 0 | 0 | 516 | 284 | 283 | 125 | 106 | 0 | 229 | 166 | 62 | 37 | 25 | 340 | 340 | 515 | 0 | 1 | 105 | 0 | 0 | 
| app_proof | prover |  | 3 | 836,978,382 | 1,085 | 339 | 0 | 0 | 515 | 284 | 283 | 125 | 106 | 0 | 229 | 166 | 62 | 37 | 25 | 340 | 339 | 515 | 0 | 1 | 105 | 0 | 0 | 
| app_proof | prover |  | 4 | 836,978,382 | 1,087 | 341 | 0 | 0 | 515 | 283 | 283 | 125 | 106 | 0 | 231 | 168 | 62 | 37 | 25 | 341 | 341 | 515 | 0 | 1 | 105 | 0 | 0 | 
| app_proof | prover |  | 5 | 826,230,478 | 999 | 262 | 0 | 0 | 512 | 283 | 283 | 124 | 104 | 0 | 224 | 162 | 61 | 36 | 24 | 263 | 262 | 512 | 0 | 1 | 102 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 91,510,674 | 2,013,265,921 | 
| app_proof | prover |  | 1 | 0 | 91,510,674 | 2,013,265,921 | 
| app_proof | prover |  | 2 | 0 | 91,510,674 | 2,013,265,921 | 
| app_proof | prover |  | 3 | 0 | 91,510,674 | 2,013,265,921 | 
| app_proof | prover |  | 4 | 0 | 91,510,674 | 2,013,265,921 | 
| app_proof | prover |  | 5 | 0 | 84,432,786 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 1,108 | 127 | 14,365,133 | 112.75 | 0 | 7,328 | 

| group | program | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- |
| app_proof |  | memory | 0 | 1 | 
| app_proof |  | memory | 1 | 1 | 
| app_proof |  | memory | 2 | 1 | 
| app_proof |  | memory | 3 | 1 | 
| app_proof |  | memory | 4 | 1 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 5 | 2 | 18 | 1,272 | 3 | 5 | 0 | 38 | 0 | 6 | 0 | 9,387,610,620 | 656,129,964 | 10,043,740,584 | 2,097,152 | 2 | 0 | 0 | 3 | 15 | 81 | 2,413,000 | 29.76 | 0 | 0 | 
| app_proof |  | 1 | 2 | 2 | 12 | 1,207 | 3 | 2 | 0 | 26 | 0 | 2 | 0 | 9,387,918,048 | 656,744,136 | 10,044,662,184 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 63 | 2,413,000 | 38.14 | 0 | 0 | 
| app_proof |  | 2 | 2 | 2 | 12 | 1,202 | 3 | 2 | 0 | 27 | 0 | 4 | 0 | 9,387,154,116 | 657,508,068 | 10,044,662,184 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 62 | 2,413,000 | 38.67 | 0 | 0 | 
| app_proof |  | 3 | 2 | 2 | 12 | 1,200 | 3 | 2 | 0 | 26 | 0 | 2 | 0 | 9,387,917,184 | 656,745,000 | 10,044,662,184 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 61 | 2,413,000 | 39.20 | 0 | 0 | 
| app_proof |  | 4 | 2 | 2 | 12 | 1,202 | 3 | 2 | 0 | 26 | 0 | 2 | 0 | 9,387,154,044 | 657,508,140 | 10,044,662,184 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 61 | 2,413,000 | 39.16 | 0 | 0 | 
| app_proof |  | 5 | 2 | 2 | 12 | 1,108 | 3 | 2 | 0 | 25 | 0 | 2 | 0 | 8,951,700,864 | 963,986,472 | 9,915,687,336 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 58 | 2,300,133 | 39.30 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ec7cb9c3272e9fec9aeee23432127eec0179fd48

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31752739045)
