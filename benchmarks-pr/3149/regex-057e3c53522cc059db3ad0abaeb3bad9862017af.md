| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.38 |  1.38 |  1.38 |
| app_proof |  0.77 |  0.77 |  0.77 |
| leaf |  0.22 |  0.22 |  0.22 |
| internal_for_leaf |  0.17 |  0.17 |  0.17 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  747 |  747 |  747 |  747 |
| `compile_metered_time_ms` |  5 |  5 |  5 |  5 |
| `execute_metered_time_ms` |  22 | -          | -          | -          |
| `execute_metered_insns` |  4,090,656 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  180.79 | -          |  180.79 |  180.79 |
| `set_initial_memory_time_ms` |  5 |  5 |  5 |  5 |
| `execute_preflight_insns` |  4,090,656 |  4,090,656 |  4,090,656 |  4,090,656 |
| `execute_preflight_time_ms` |  148 |  148 |  148 |  148 |
| `execute_preflight_insn_mi/s` |  27.64 | -          |  27.61 |  27.61 |
| `postflight_time_ms  ` |  65 |  65 |  65 |  65 |
| `postflight_memory_chronology_time_ms` |  11 |  11 |  11 |  11 |
| `postflight_program_index_time_ms` |  1 |  1 |  1 |  1 |
| `trace_gen_time_ms   ` |  21 |  21 |  21 |  21 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  485 |  485 |  485 |  485 |
| `prover.main_trace_commit_time_ms` |  66 |  66 |  66 |  66 |
| `prover.rap_constraints_time_ms` |  347 |  347 |  347 |  347 |
| `prover.openings_time_ms` |  71 |  71 |  71 |  71 |
| `prover.rap_constraints.logup_gkr_time_ms` |  212 |  212 |  212 |  212 |
| `prover.rap_constraints.round0_time_ms` |  103 |  103 |  103 |  103 |
| `prover.rap_constraints.mle_rounds_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  50 |  50 |  50 |  50 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  217 |  217 |  217 |  217 |
| `execute_preflight_time_ms` |  5 |  5 |  5 |  5 |
| `trace_gen_time_ms   ` |  40 |  40 |  40 |  40 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  176 |  176 |  176 |  176 |
| `prover.main_trace_commit_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints_time_ms` |  85 |  85 |  85 |  85 |
| `prover.openings_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.mle_rounds_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  37 |  37 |  37 |  37 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  165 |  165 |  165 |  165 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  15 |  15 |  15 |  15 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  149 |  149 |  149 |  149 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  74 |  74 |  74 |  74 |
| `prover.openings_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.mle_rounds_time_ms` |  34 |  34 |  34 |  34 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  32 |  32 |  32 |  32 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  123 |  123 |  123 |  123 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  112 |  112 |  112 |  112 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  27 |  27 |  27 |  27 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  107 |  107 |  107 |  107 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  97 |  97 |  97 |  97 |
| `prover.main_trace_commit_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/057e3c53522cc059db3ad0abaeb3bad9862017af/regex-057e3c53522cc059db3ad0abaeb3bad9862017af.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| frac_sumcheck.gkr_rounds | 4.90 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 4.90 | app_proof.prover..0 |
| prover.gkr_input_evals | 4.89 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 4.89 | app_proof.prover..0 |
| postflight | 4.71 | app_proof..0 |
| tracegen | 4.45 | app_proof..0 |
| generate mem proving ctxs | 4.45 | app_proof..0 |
| set initial memory | 4.13 | app_proof..0 |
| prover.stacked_commit | 2.49 | app_proof.prover..0 |
| prover.openings | 1.77 | app_proof.prover..0 |
| prover.merkle_tree | 1.77 | app_proof.prover..0 |
| prover.prove_whir_opening | 1.77 | app_proof.prover..0 |
| prover.rs_code_matrix | 1.77 | app_proof.prover..0 |
| prover.rap_constraints | 1.68 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 1.40 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 1.40 | app_proof.prover..0 |
| prover.before_gkr_input_evals | 0.94 | app_proof.prover..0 |
| tracegen.exp_bits_len | 0.40 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.40 | leaf.0 |
| tracegen.pow_checker | 0.40 | leaf.0 |
| tracegen.whir_folding | 0.34 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.34 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.34 | leaf.0 |
| tracegen.range_checker | 0.31 | leaf.0 |
| tracegen.public_values | 0.31 | leaf.0 |
| tracegen.proof_shape | 0.31 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 119 | 267,351 | 230,250 | 1 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | KeccakfOpAir |  | 111 | 1 | 2 | 
| 11 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 12 | XorinVmAir |  | 359 | 34 | 3 | 
| 13 | RevealAir |  | 25 | 3 | 2 | 
| 14 | HintStoreAir | 1 | 18 | 12 | 3 | 
| 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 11 | 3 | 
| 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 15 | 2 | 
| 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 27 | 7 | 3 | 
| 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 26 | 10 | 3 | 
| 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 25 | 7 | 3 | 
| 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 24 | 10 | 3 | 
| 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 25 | 11 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 24 | 7 | 3 | 
| 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 23 | 10 | 3 | 
| 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 24 | 11 | 3 | 
| 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
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
| agg_keygen |  | 58 |  |  |  | 296 |  | 
| app_proof | 3 |  |  |  | 8 |  |  | 
| internal_for_leaf |  |  |  | 165 |  |  | 165 | 
| internal_recursive.0 |  |  |  | 123 |  |  | 123 | 
| internal_recursive.1 |  |  |  | 107 |  |  | 107 | 
| leaf |  |  | 217 |  |  |  | 217 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 1 | 
| app_proof | KeccakfOpAir |  | 0 | 0 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 2 | 
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
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 4 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 0 | 
| app_proof | XorinVmAir |  | 0 | 0 | 

| group | air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 1 | 70 | 218 | 4 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 127 | 4 | 
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

| group | air_id | air_name | bus_index | bus_name | program | segment | metered_bus_interaction_memory_unpadded_bytes | metered_bus_interaction_memory_padding_bytes | metered_bus_interaction_cells_unpadded | metered_bus_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 2 | Program |  | 0 | 2,936,832 | 1,257,472 | 91,776 | 39,296 | 
| app_proof | 1 | VmConnectorAir | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange |  | 0 | 128 |  | 4 |  | 
| app_proof | 10 | KeccakfOpAir | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 10 | KeccakfOpAir | 1 | Memory |  | 0 | 1,664 |  | 52 |  | 
| app_proof | 10 | KeccakfOpAir | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 10 | KeccakfOpAir | 3 | VariableRange |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 10 | KeccakfOpAir | 7 | KeccakfState |  | 0 | 64 |  | 2 |  | 
| app_proof | 11 | KeccakfPermAir | 7 | KeccakfState |  | 0 | 1,536 | 512 | 48 | 16 | 
| app_proof | 12 | XorinVmAir | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 12 | XorinVmAir | 1 | Memory |  | 0 | 3,456 |  | 108 |  | 
| app_proof | 12 | XorinVmAir | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 12 | XorinVmAir | 3 | VariableRange |  | 0 | 3,584 |  | 112 |  | 
| app_proof | 12 | XorinVmAir | 6 | BitwiseLookup |  | 0 | 4,352 |  | 136 |  | 
| app_proof | 13 | RevealAir | 0 | Execution |  | 0 | 256 |  | 8 |  | 
| app_proof | 13 | RevealAir | 1 | Memory |  | 0 | 1,024 |  | 32 |  | 
| app_proof | 13 | RevealAir | 2 | Program |  | 0 | 128 |  | 4 |  | 
| app_proof | 13 | RevealAir | 3 | VariableRange |  | 0 | 1,280 |  | 40 |  | 
| app_proof | 13 | RevealAir | 6 | BitwiseLookup |  | 0 | 512 |  | 16 |  | 
| app_proof | 14 | HintStoreAir | 0 | Execution |  | 0 | 408,576 | 115,712 | 12,768 | 3,616 | 
| app_proof | 14 | HintStoreAir | 1 | Memory |  | 0 | 1,225,728 | 347,136 | 38,304 | 10,848 | 
| app_proof | 14 | HintStoreAir | 2 | Program |  | 0 | 204,288 | 57,856 | 6,384 | 1,808 | 
| app_proof | 14 | HintStoreAir | 3 | VariableRange |  | 0 | 1,838,592 | 520,704 | 57,456 | 16,272 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | Execution |  | 0 | 4,696,896 | 3,691,712 | 146,778 | 115,366 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | Memory |  | 0 | 9,393,792 | 7,383,424 | 293,556 | 230,732 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 2 | Program |  | 0 | 2,348,448 | 1,845,856 | 73,389 | 57,683 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 3 | VariableRange |  | 0 | 9,393,792 | 7,383,424 | 293,556 | 230,732 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 21,136,032 | 16,612,704 | 660,501 | 519,147 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | Execution |  | 0 | 614,208 | 434,368 | 19,194 | 13,574 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 1 | Memory |  | 0 | 1,228,416 | 868,736 | 38,388 | 27,148 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 2 | Program |  | 0 | 307,104 | 217,184 | 9,597 | 6,787 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 2,149,728 | 1,520,288 | 67,179 | 47,509 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 0 | Execution |  | 0 | 5,824 | 2,368 | 182 | 74 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 1 | Memory |  | 0 | 11,648 | 4,736 | 364 | 148 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 2 | Program |  | 0 | 2,912 | 1,184 | 91 | 37 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 37,856 | 15,392 | 1,183 | 481 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | Execution |  | 0 | 22,279,616 | 11,274,816 | 696,238 | 352,338 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | Memory |  | 0 | 44,559,232 | 22,549,632 | 1,392,476 | 704,676 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 2 | Program |  | 0 | 11,139,808 | 5,637,408 | 348,119 | 176,169 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 133,677,696 | 67,648,896 | 4,177,428 | 2,114,028 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | Execution |  | 0 | 42,821,312 | 24,287,552 | 1,338,166 | 758,986 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | Memory |  | 0 | 85,642,624 | 48,575,104 | 2,676,332 | 1,517,972 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 2 | Program |  | 0 | 21,410,656 | 12,143,776 | 669,083 | 379,493 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 192,695,904 | 109,293,984 | 6,021,747 | 3,415,437 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory |  | 0 | 3,329,536 | 864,768 | 104,048 | 27,024 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle |  | 0 | 1,664,768 | 432,384 | 52,024 | 13,512 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression |  | 0 | 1,664,768 | 432,384 | 52,024 | 13,512 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 0 | Execution |  | 0 | 5,987,904 | 2,400,704 | 187,122 | 75,022 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 1 | Memory |  | 0 | 5,987,904 | 2,400,704 | 187,122 | 75,022 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 2 | Program |  | 0 | 2,993,952 | 1,200,352 | 93,561 | 37,511 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 3 | VariableRange |  | 0 | 26,945,568 | 10,803,168 | 842,049 | 337,599 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 0 | Execution |  | 0 | 8,570,112 | 8,207,104 | 267,816 | 256,472 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 1 | Memory |  | 0 | 17,140,224 | 16,414,208 | 535,632 | 512,944 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 2 | Program |  | 0 | 4,285,056 | 4,103,552 | 133,908 | 128,236 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 3 | VariableRange |  | 0 | 34,280,448 | 32,828,416 | 1,071,264 | 1,025,888 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 0 | Execution |  | 0 | 4,341,888 | 4,046,720 | 135,684 | 126,460 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 1 | Memory |  | 0 | 4,341,888 | 4,046,720 | 135,684 | 126,460 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 2 | Program |  | 0 | 2,170,944 | 2,023,360 | 67,842 | 63,230 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 3 | VariableRange |  | 0 | 15,196,608 | 14,163,520 | 474,894 | 442,610 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 11,795,200 | 4,982,016 | 368,600 | 155,688 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 23,590,400 | 9,964,032 | 737,200 | 311,376 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | Program |  | 0 | 5,897,600 | 2,491,008 | 184,300 | 77,844 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 41,283,200 | 17,437,056 | 1,290,100 | 544,908 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 0 | Execution |  | 0 | 15,056,576 | 1,720,640 | 470,518 | 53,770 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 1 | Memory |  | 0 | 30,113,152 | 3,441,280 | 941,036 | 107,540 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 2 | Program |  | 0 | 7,528,288 | 860,320 | 235,259 | 26,885 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 3 | VariableRange |  | 0 | 30,113,152 | 3,441,280 | 941,036 | 107,540 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | Execution |  | 0 | 41,203,648 | 25,905,216 | 1,287,614 | 809,538 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 1 | Memory |  | 0 | 164,814,592 | 103,620,864 | 5,150,456 | 3,238,152 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 2 | Program |  | 0 | 20,601,824 | 12,952,608 | 643,807 | 404,769 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 3 | VariableRange |  | 0 | 206,018,240 | 129,526,080 | 6,438,070 | 4,047,690 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 6 | BitwiseLookup |  | 0 | 123,610,944 | 77,715,648 | 3,862,842 | 2,428,614 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | Execution |  | 0 | 60,472,768 | 6,636,096 | 1,889,774 | 207,378 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 1 | Memory |  | 0 | 241,891,072 | 26,544,384 | 7,559,096 | 829,512 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 2 | Program |  | 0 | 30,236,384 | 3,318,048 | 944,887 | 103,689 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 3 | VariableRange |  | 0 | 302,363,840 | 33,180,480 | 9,448,870 | 1,036,890 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 6 | BitwiseLookup |  | 0 | 151,181,920 | 16,590,240 | 4,724,435 | 518,445 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | Execution |  | 0 | 7,517,184 | 871,424 | 234,912 | 27,232 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 1 | Memory |  | 0 | 30,068,736 | 3,485,696 | 939,648 | 108,928 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 2 | Program |  | 0 | 3,758,592 | 435,712 | 117,456 | 13,616 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 3 | VariableRange |  | 0 | 37,585,920 | 4,357,120 | 1,174,560 | 136,160 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 6 | BitwiseLookup |  | 0 | 15,034,368 | 1,742,848 | 469,824 | 54,464 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | 0 | Execution |  | 0 | 5,640,960 | 2,747,648 | 176,280 | 85,864 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | 1 | Memory |  | 0 | 22,563,840 | 10,990,592 | 705,120 | 343,456 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | 2 | Program |  | 0 | 2,820,480 | 1,373,824 | 88,140 | 42,932 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | 3 | VariableRange |  | 0 | 28,204,800 | 13,738,240 | 881,400 | 429,320 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | 6 | BitwiseLookup |  | 0 | 8,461,440 | 4,121,472 | 264,420 | 128,796 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | Execution |  | 0 | 5,416,448 | 2,972,160 | 169,264 | 92,880 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 1 | Memory |  | 0 | 21,665,792 | 11,888,640 | 677,056 | 371,520 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 2 | Program |  | 0 | 2,708,224 | 1,486,080 | 84,632 | 46,440 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 3 | VariableRange |  | 0 | 29,790,464 | 16,346,880 | 930,952 | 510,840 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 6 | BitwiseLookup |  | 0 | 8,124,672 | 4,458,240 | 253,896 | 139,320 | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle |  | 0 | 5,097,600 | 1,193,856 | 159,300 | 37,308 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression |  | 0 | 1,699,200 | 397,952 | 53,100 | 12,436 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | 0 | Execution |  | 0 | 644,992 | 403,584 | 20,156 | 12,612 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | 1 | Memory |  | 0 | 2,579,968 | 1,614,336 | 80,624 | 50,448 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | 2 | Program |  | 0 | 322,496 | 201,792 | 10,078 | 6,306 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | 3 | VariableRange |  | 0 | 3,224,960 | 2,017,920 | 100,780 | 63,060 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | 6 | BitwiseLookup |  | 0 | 967,488 | 605,376 | 30,234 | 18,918 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | 0 | Execution |  | 0 | 5,440 | 2,752 | 170 | 86 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | 1 | Memory |  | 0 | 21,760 | 11,008 | 680 | 344 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | 2 | Program |  | 0 | 2,720 | 1,376 | 85 | 43 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | 3 | VariableRange |  | 0 | 27,200 | 13,760 | 850 | 430 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | 6 | BitwiseLookup |  | 0 | 5,440 | 2,752 | 170 | 86 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | 0 | Execution |  | 0 | 1,728 | 320 | 54 | 10 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | 1 | Memory |  | 0 | 6,912 | 1,280 | 216 | 40 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | 2 | Program |  | 0 | 864 | 160 | 27 | 5 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | 3 | VariableRange |  | 0 | 9,504 | 1,760 | 297 | 55 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | 6 | BitwiseLookup |  | 0 | 1,728 | 320 | 54 | 10 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 0 | Execution |  | 0 | 137,984 | 124,160 | 4,312 | 3,880 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 1 | Memory |  | 0 | 413,952 | 372,480 | 12,936 | 11,640 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 2 | Program |  | 0 | 68,992 | 62,080 | 2,156 | 1,940 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 3 | VariableRange |  | 0 | 551,936 | 496,640 | 17,248 | 15,520 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 6 | BitwiseLookup |  | 0 | 137,984 | 124,160 | 4,312 | 3,880 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 0 | Execution |  | 0 | 1,611,904 | 485,248 | 50,372 | 15,164 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 1 | Memory |  | 0 | 4,835,712 | 1,455,744 | 151,116 | 45,492 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 2 | Program |  | 0 | 805,952 | 242,624 | 25,186 | 7,582 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 3 | VariableRange |  | 0 | 6,447,616 | 1,940,992 | 201,488 | 60,656 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 6 | BitwiseLookup |  | 0 | 805,952 | 242,624 | 25,186 | 7,582 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | Execution |  | 0 | 54,656 | 10,880 | 1,708 | 340 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 1 | Memory |  | 0 | 163,968 | 32,640 | 5,124 | 1,020 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 2 | Program |  | 0 | 27,328 | 5,440 | 854 | 170 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 3 | VariableRange |  | 0 | 245,952 | 48,960 | 7,686 | 1,530 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 6 | BitwiseLookup |  | 0 | 27,328 | 5,440 | 854 | 170 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | Execution |  | 0 | 6,208 | 1,984 | 194 | 62 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | Memory |  | 0 | 12,416 | 3,968 | 388 | 124 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 2 | Program |  | 0 | 3,104 | 992 | 97 | 31 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 3 | VariableRange |  | 0 | 27,936 | 8,928 | 873 | 279 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | Execution |  | 0 | 25,856 | 6,912 | 808 | 216 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 1 | Memory |  | 0 | 51,712 | 13,824 | 1,616 | 432 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 2 | Program |  | 0 | 12,928 | 3,456 | 404 | 108 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 90,496 | 24,192 | 2,828 | 756 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | Execution |  | 0 | 43,392 | 22,144 | 1,356 | 692 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | Memory |  | 0 | 130,176 | 66,432 | 4,068 | 2,076 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 2 | Program |  | 0 | 21,696 | 11,072 | 678 | 346 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 3 | VariableRange |  | 0 | 325,440 | 166,080 | 10,170 | 5,190 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 588,480 | 460,096 | 18,390 | 14,378 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 1,765,440 | 1,380,288 | 55,170 | 43,134 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 2 | Program |  | 0 | 294,240 | 230,048 | 9,195 | 7,189 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 2,648,160 | 2,070,432 | 82,755 | 64,701 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | Execution |  | 0 | 324,416 | 199,872 | 10,138 | 6,246 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 1 | Memory |  | 0 | 973,248 | 599,616 | 30,414 | 18,738 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 2 | Program |  | 0 | 162,208 | 99,936 | 5,069 | 3,123 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 1,297,664 | 799,488 | 40,552 | 24,984 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | Execution |  | 0 | 2,278,656 | 1,915,648 | 71,208 | 59,864 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | Memory |  | 0 | 6,835,968 | 5,746,944 | 213,624 | 179,592 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 2 | Program |  | 0 | 1,139,328 | 957,824 | 35,604 | 29,932 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 3 | VariableRange |  | 0 | 6,835,968 | 5,746,944 | 213,624 | 179,592 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 9,114,624 | 7,662,592 | 284,832 | 239,456 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | Execution |  | 0 | 19,308,864 | 14,245,568 | 603,402 | 445,174 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 1 | Memory |  | 0 | 57,926,592 | 42,736,704 | 1,810,206 | 1,335,522 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 2 | Program |  | 0 | 9,654,432 | 7,122,784 | 301,701 | 222,587 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 96,544,320 | 71,227,840 | 3,017,010 | 2,225,870 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup |  | 0 | 4,194,304 |  | 131,072 |  | 
| app_proof | 48 | PhantomAir | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 48 | PhantomAir | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression |  | 0 | 1,698,752 | 398,400 | 53,086 | 12,450 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 0 | Execution |  | 0 | 2,432 | 1,664 | 76 | 52 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 1 | Memory |  | 0 | 7,296 | 4,992 | 228 | 156 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 2 | Program |  | 0 | 1,216 | 832 | 38 | 26 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 3 | VariableRange |  | 0 | 7,296 | 4,992 | 228 | 156 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 12,160 | 8,320 | 380 | 260 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | 8 | RangeTuple |  | 0 | 19,456 | 13,312 | 608 | 416 | 
| app_proof | 50 | VariableRangeCheckerAir | 3 | VariableRange |  | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 0 | Execution |  | 0 | 10,176 | 6,208 | 318 | 194 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 1 | Memory |  | 0 | 30,528 | 18,624 | 954 | 582 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 2 | Program |  | 0 | 5,088 | 3,104 | 159 | 97 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 3 | VariableRange |  | 0 | 30,528 | 18,624 | 954 | 582 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 45,792 | 27,936 | 1,431 | 873 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | 8 | RangeTuple |  | 0 | 81,408 | 49,664 | 2,544 | 1,552 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | Execution |  | 0 | 335,232 | 189,056 | 10,476 | 5,908 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | Memory |  | 0 | 1,005,696 | 567,168 | 31,428 | 17,724 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 2 | Program |  | 0 | 167,616 | 94,528 | 5,238 | 2,954 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 3 | VariableRange |  | 0 | 1,005,696 | 567,168 | 31,428 | 17,724 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 1,340,928 | 756,224 | 41,904 | 23,632 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 8 | RangeTuple |  | 0 | 1,340,928 | 756,224 | 41,904 | 23,632 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 8 | RangeTuple |  | 0 | 33,554,432 |  | 1,048,576 |  | 

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
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 38 | 622,592 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 45 | 2,880 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 11,520 | 
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
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 1,458,176 | 
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
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 38 | 311,296 | 
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
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 38 | 311,296 | 
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
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 39,452,672 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 38 | 1,245,184 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 46 | 2,944 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 8,192 | 16 | 131,072 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 89 | 5,832,704 | 
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

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 131,072 | 10 | 1,310,720 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 0 | 1 | 258 | 258 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 0 | 32 | 2,634 | 84,288 | 
| app_proof | 12 | XorinVmAir | prover |  | 0 | 1 | 543 | 543 | 
| app_proof | 13 | RevealAir | prover |  | 0 | 4 | 34 | 136 | 
| app_proof | 14 | HintStoreAir | prover |  | 0 | 8,192 | 24 | 196,608 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 131,072 | 34 | 4,456,448 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 16,384 | 27 | 442,368 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | prover |  | 0 | 128 | 49 | 6,272 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 524,288 | 51 | 26,738,688 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 32,768 | 38 | 1,245,184 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 131,072 | 16 | 2,097,152 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 262,144 | 23 | 6,029,312 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 262,144 | 30 | 7,864,320 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 1,048,576 | 38 | 39,845,888 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 1,048,576 | 38 | 39,845,888 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 131,072 | 36 | 4,718,592 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> | prover |  | 0 | 131,072 | 36 | 4,718,592 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 131,072 | 37 | 4,849,664 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 65,536 | 33 | 2,162,688 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> | prover |  | 0 | 16,384 | 35 | 573,440 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> | prover |  | 0 | 128 | 35 | 4,480 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> | prover |  | 0 | 32 | 36 | 1,152 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 4,096 | 28 | 114,688 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 32,768 | 28 | 917,504 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 1,024 | 29 | 29,696 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 128 | 44 | 5,632 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 512 | 22 | 11,264 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover |  | 0 | 1,024 | 58 | 59,392 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 16,384 | 33 | 540,672 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 8,192 | 28 | 229,376 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 65,536 | 42 | 2,752,512 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 524,288 | 29 | 15,204,352 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | PhantomAir | prover |  | 0 | 1 | 7 | 7 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 16,384 | 300 | 4,915,200 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> | prover |  | 0 | 64 | 84 | 5,376 | 
| app_proof | 50 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> | prover |  | 0 | 256 | 52 | 13,312 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 8,192 | 40 | 327,680 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_cells_unpadded | metered_interaction_cells_padding | metered_constraint_eval_cells_unpadded | metered_constraint_eval_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 91,776 | 39,296 | 3,671,040 | 1,571,840 | 917,760 | 392,960 | 91,776 | 39,296 |  |  | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 10 | KeccakfOpAir |  | 0 | 1 |  | 1,032 |  | 258 |  | 111 |  | 2 |  | 
| app_proof | 11 | KeccakfPermAir |  | 0 | 24 | 8 | 252,864 | 84,288 | 63,216 | 21,072 | 48 | 16 | 64,104 | 21,368 | 
| app_proof | 12 | XorinVmAir |  | 0 | 1 |  | 2,172 |  | 543 |  | 359 |  | 4 |  | 
| app_proof | 13 | RevealAir |  | 0 | 4 |  | 544 |  | 136 |  | 100 |  | 8 |  | 
| app_proof | 14 | HintStoreAir |  | 0 | 6,384 | 1,808 | 612,864 | 173,568 | 153,216 | 43,392 | 114,912 | 32,544 | 44,688 | 12,656 | 
| app_proof | 15 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 73,389 | 57,683 | 9,980,904 | 7,844,888 | 2,495,226 | 1,961,222 | 1,467,780 | 1,153,660 | 293,556 | 230,732 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 9,597 | 6,787 | 1,036,476 | 732,996 | 259,119 | 183,249 | 134,358 | 95,018 | 76,776 | 54,296 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 91 | 37 | 17,836 | 7,252 | 4,459 | 1,813 | 1,820 | 740 | 2,366 | 962 | 
| app_proof | 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 348,119 | 176,169 | 71,016,276 | 35,938,476 | 17,754,069 | 8,984,619 | 6,614,261 | 3,347,211 | 7,658,618 | 3,875,718 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 669,083 | 379,493 | 61,555,636 | 34,913,356 | 15,388,909 | 8,728,339 | 10,705,328 | 6,071,888 | 3,345,415 | 1,897,465 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 26,012 | 6,756 | 3,953,824 | 1,026,912 | 988,456 | 256,728 | 208,096 | 54,048 | 104,048 | 27,024 | 
| app_proof | 20 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 93,561 | 37,511 | 5,987,904 | 2,400,704 | 1,496,976 | 600,176 | 1,309,854 | 525,154 | 374,244 | 150,044 | 
| app_proof | 21 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 133,908 | 128,236 | 12,319,536 | 11,797,712 | 3,079,884 | 2,949,428 | 2,008,620 | 1,923,540 | 669,540 | 641,180 | 
| app_proof | 22 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 67,842 | 63,230 | 4,884,624 | 4,552,560 | 1,221,156 | 1,138,140 | 814,104 | 758,760 | 678,420 | 632,300 | 
| app_proof | 23 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 184,300 | 77,844 | 22,116,000 | 9,341,280 | 5,529,000 | 2,335,320 | 2,580,200 | 1,089,816 | 1,474,400 | 622,752 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 235,259 | 26,885 | 22,584,864 | 2,580,960 | 5,646,216 | 645,240 | 2,587,849 | 295,735 | 1,646,813 | 188,195 | 
| app_proof | 25 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 643,807 | 404,769 | 97,858,664 | 61,524,888 | 24,464,666 | 15,381,222 | 17,382,789 | 10,928,763 | 3,862,842 | 2,428,614 | 
| app_proof | 26 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 944,887 | 103,689 | 143,622,824 | 15,760,728 | 35,905,706 | 3,940,182 | 24,567,062 | 2,695,914 | 5,669,322 | 622,134 | 
| app_proof | 27 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 117,456 | 13,616 | 16,913,664 | 1,960,704 | 4,228,416 | 490,176 | 2,936,400 | 340,400 | 704,736 | 81,696 | 
| app_proof | 28 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 0 | 88,140 | 42,932 | 12,692,160 | 6,182,208 | 3,173,040 | 1,545,552 | 2,115,360 | 1,030,368 | 528,840 | 257,592 | 
| app_proof | 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 84,632 | 46,440 | 12,525,536 | 6,873,120 | 3,131,384 | 1,718,280 | 2,115,800 | 1,161,000 | 507,792 | 278,640 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 53,100 | 12,436 | 7,009,200 | 1,641,552 | 1,752,300 | 410,388 | 212,400 | 49,744 | 477,900 | 111,924 | 
| app_proof | 30 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 10,078 | 6,306 | 1,410,920 | 882,840 | 352,730 | 220,710 | 241,872 | 151,344 | 60,468 | 37,836 | 
| app_proof | 31 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 0 | 85 | 43 | 11,900 | 6,020 | 2,975 | 1,505 | 1,955 | 989 | 510 | 258 | 
| app_proof | 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 0 | 27 | 5 | 3,888 | 720 | 972 | 180 | 648 | 120 | 162 | 30 | 
| app_proof | 33 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 2,156 | 1,940 | 241,472 | 217,280 | 60,368 | 54,320 | 40,964 | 36,860 | 12,936 | 11,640 | 
| app_proof | 34 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 25,186 | 7,582 | 2,820,832 | 849,184 | 705,208 | 212,296 | 453,348 | 136,476 | 151,116 | 45,492 | 
| app_proof | 35 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 854 | 170 | 99,064 | 19,720 | 24,766 | 4,930 | 16,226 | 3,230 | 5,124 | 1,020 | 
| app_proof | 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 97 | 31 | 17,072 | 5,456 | 4,268 | 1,364 | 1,552 | 496 | 2,134 | 682 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 404 | 108 | 35,552 | 9,504 | 8,888 | 2,376 | 5,656 | 1,512 | 2,020 | 540 | 
| app_proof | 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 0 | 678 | 346 | 157,296 | 80,272 | 39,324 | 20,068 | 16,272 | 8,304 | 14,916 | 7,612 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 9,195 | 7,189 | 1,213,740 | 948,948 | 303,435 | 237,237 | 165,510 | 129,402 | 73,560 | 57,512 | 
| app_proof | 44 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 5,069 | 3,123 | 567,728 | 349,776 | 141,932 | 87,444 | 86,173 | 53,091 | 40,552 | 24,984 | 
| app_proof | 45 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 35,604 | 29,932 | 5,981,472 | 5,028,576 | 1,495,368 | 1,257,144 | 818,892 | 688,436 | 142,416 | 119,728 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 301,701 | 222,587 | 34,997,316 | 25,820,092 | 8,749,329 | 6,455,023 | 5,732,319 | 4,229,153 | 2,413,608 | 1,780,696 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 48 | PhantomAir |  | 0 | 1 |  | 28 |  | 7 |  | 3 |  | 2 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 53,086 | 12,450 | 63,703,200 | 14,940,000 | 15,925,800 | 3,735,000 | 53,086 | 12,450 | 1,751,838 | 410,850 | 
| app_proof | 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 38 | 26 | 12,768 | 8,736 | 3,192 | 2,184 | 1,558 | 1,066 | 1,216 | 832 | 
| app_proof | 50 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 159 | 97 | 33,072 | 20,176 | 8,268 | 5,044 | 6,360 | 3,880 | 795 | 485 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 5,238 | 2,954 | 838,080 | 472,640 | 209,520 | 118,160 | 162,378 | 91,574 | 10,476 | 5,908 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 12,582,912 |  | 3,145,728 |  | 1,048,576 |  | 5,242,880 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 5 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 15 | 165 | 15 | 5 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 11 | 123 | 9 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 40 | 217 | 39 | 8 | 1 | 5 | 10 | 10 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 27,294,269 | 149 | 33 | 0 | 0 | 74 | 26 | 25 | 34 | 14 | 0 | 41 | 32 | 9 | 2 | 7 | 33 | 33 | 74 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,386,961 | 112 | 19 | 0 | 0 | 56 | 20 | 20 | 23 | 11 | 0 | 35 | 27 | 7 | 1 | 6 | 20 | 19 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,759,057 | 97 | 14 | 0 | 0 | 55 | 21 | 20 | 21 | 11 | 0 | 26 | 19 | 7 | 1 | 5 | 14 | 14 | 55 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 65,398,397 | 176 | 42 | 0 | 0 | 85 | 40 | 39 | 26 | 18 | 0 | 48 | 37 | 10 | 3 | 7 | 42 | 42 | 85 | 0 | 3 | 18 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,449,923 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 6,834,115 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 5 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 210,461,308 | 485 | 65 | 0 | 0 | 347 | 103 | 102 | 31 | 212 | 83 | 71 | 50 | 21 | 9 | 11 | 66 | 65 | 347 | 0 | 1 | 127 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 124,308,807 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 747 | 22 | 4,090,656 | 180.79 | 0 | 784 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_whir_memory_bytes | metered_secondary_peak_memory_bytes | metered_rs_code_matrix_memory_bytes | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_gkr_memory_bytes | metered_batch_constraint_memory_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 5 | 2 | 21 | 747 | 6 | 5 | 0 | 65 | 1 | 11 | 0 | 125,829,120 | 4,533,102,816 | 1,811,939,328 | 3,988,816,580 | 1,445,113,868 | 5,433,930,448 | 4,533,102,816 | 850,996,596 | 3 | 0 | 2 | 6 | 14 | 148 | 4,090,656 | 27.61 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/057e3c53522cc059db3ad0abaeb3bad9862017af

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33812815335)
