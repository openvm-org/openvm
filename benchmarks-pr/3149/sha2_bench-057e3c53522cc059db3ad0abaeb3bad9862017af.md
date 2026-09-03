| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.45 |  4.12 |  4.12 |
| app_proof |  4.49 |  3.16 |  3.16 |
| leaf |  0.53 |  0.53 |  0.53 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,215 |  4,430 |  2,930 |  1,500 |
| `compile_metered_time_ms` |  0 |  0 |  0 |  0 |
| `execute_metered_time_ms` |  61 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  181.07 | -          |  181.07 |  181.07 |
| `set_initial_memory_time_ms` |  4 |  8 |  5 |  3 |
| `execute_preflight_insns` |  5,583,980.50 |  11,167,961 |  7,136,000 |  4,031,961 |
| `execute_preflight_time_ms` |  259.50 |  519 |  352 |  167 |
| `execute_preflight_insn_mi/s` |  21.52 | -          |  24.09 |  20.25 |
| `postflight_time_ms  ` |  135 |  270 |  194 |  76 |
| `postflight_memory_chronology_time_ms` |  12.50 |  25 |  20 |  5 |
| `postflight_program_index_time_ms` |  1.50 |  3 |  2 |  1 |
| `trace_gen_time_ms   ` |  53.50 |  107 |  75 |  32 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,725 |  3,450 |  2,255 |  1,195 |
| `prover.main_trace_commit_time_ms` |  384.50 |  769 |  560 |  209 |
| `prover.rap_constraints_time_ms` |  1,076.50 |  2,153 |  1,360 |  793 |
| `prover.openings_time_ms` |  263.50 |  527 |  334 |  193 |
| `prover.rap_constraints.logup_gkr_time_ms` |  249 |  498 |  273 |  225 |
| `prover.rap_constraints.round0_time_ms` |  645 |  1,290 |  849 |  441 |
| `prover.rap_constraints.mle_rounds_time_ms` |  180.50 |  361 |  236 |  125 |
| `prover.openings.stacked_reduction_time_ms` |  71 |  142 |  92 |  50 |
| `prover.openings.stacked_reduction.round0_time_ms` |  43 |  86 |  56 |  30 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  27.50 |  55 |  35 |  20 |
| `prover.openings.whir_time_ms` |  191.50 |  383 |  241 |  142 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  534 |  534 |  534 |  534 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  117 |  117 |  117 |  117 |
| `generate_blob_total_time_ms` |  12 |  12 |  12 |  12 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  416 |  416 |  416 |  416 |
| `prover.main_trace_commit_time_ms` |  142 |  142 |  142 |  142 |
| `prover.rap_constraints_time_ms` |  156 |  156 |  156 |  156 |
| `prover.openings_time_ms` |  116 |  116 |  116 |  116 |
| `prover.rap_constraints.logup_gkr_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.round0_time_ms` |  83 |  83 |  83 |  83 |
| `prover.rap_constraints.mle_rounds_time_ms` |  44 |  44 |  44 |  44 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  94 |  94 |  94 |  94 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  197 |  197 |  197 |  197 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  21 |  21 |  21 |  21 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  175 |  175 |  175 |  175 |
| `prover.main_trace_commit_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints_time_ms` |  81 |  81 |  81 |  81 |
| `prover.openings_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  37 |  37 |  37 |  37 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  119 |  119 |  119 |  119 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  107 |  107 |  107 |  107 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  57 |  57 |  57 |  57 |
| `prover.openings_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  114 |  114 |  114 |  114 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  103 |  103 |  103 |  103 |
| `prover.main_trace_commit_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  26 |  26 |  26 |  26 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/057e3c53522cc059db3ad0abaeb3bad9862017af/sha2_bench-057e3c53522cc059db3ad0abaeb3bad9862017af.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 14.47 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 13.38 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 13.38 | app_proof.prover..0 |
| prover.gkr_input_evals | 13.07 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 13.07 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 10.65 | app_proof.prover..0 |
| prover.rap_constraints | 10.65 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 10.65 | app_proof.prover..0 |
| prover.prove_whir_opening | 9.77 | app_proof.prover..0 |
| prover.openings | 9.77 | app_proof.prover..0 |
| prover.merkle_tree | 9.77 | app_proof.prover..0 |
| prover.rs_code_matrix | 9.76 | app_proof.prover..0 |
| tracegen | 5.42 | app_proof..0 |
| postflight | 5.16 | app_proof..0 |
| prover.before_gkr_input_evals | 4.91 | app_proof.prover..0 |
| generate mem proving ctxs | 4.64 | app_proof..0 |
| set initial memory | 4.12 | app_proof..1 |
| tracegen.pow_checker | 1.10 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.10 | leaf.0 |
| tracegen.exp_bits_len | 1.10 | leaf.0 |
| tracegen.whir_folding | 0.97 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.97 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.96 | leaf.0 |
| tracegen.proof_shape | 0.79 | leaf.0 |
| tracegen.public_values | 0.79 | leaf.0 |
| tracegen.range_checker | 0.79 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 119 | 267,351 | 230,428 | 1 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 1 | 152 | 4 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 1 | 88 | 4 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 14 | RevealAir |  | 25 | 3 | 2 | 
| 15 | HintStoreAir | 1 | 18 | 12 | 3 | 
| 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 11 | 3 | 
| 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 15 | 2 | 
| 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 27 | 7 | 3 | 
| 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 26 | 10 | 3 | 
| 28 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 25 | 7 | 3 | 
| 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 24 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 25 | 11 | 3 | 
| 31 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 24 | 7 | 3 | 
| 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 23 | 10 | 3 | 
| 33 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 24 | 11 | 3 | 
| 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 35 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 37 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 4 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 40 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 41 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 42 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 44 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 48 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 49 | PhantomAir |  | 3 | 1 | 2 | 
| 5 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 51 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 6 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 7 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 59 |  |  |  | 299 |  | 
| app_proof | 0 |  |  |  | 1 |  |  | 
| internal_for_leaf |  |  |  | 197 |  |  | 197 | 
| internal_recursive.0 |  |  |  | 119 |  |  | 120 | 
| internal_recursive.1 |  |  |  | 114 |  |  | 114 | 
| leaf |  |  | 534 |  |  |  | 534 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 44 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | RevealAir |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> |  | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> |  | 0 | 3 | 
| app_proof | Sha2MainAir<Sha512Config> |  | 0 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 0 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 1 | 
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
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 5 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 4 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 4 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 1 | 0 | 
| app_proof | HintStoreAir |  | 1 | 0 | 
| app_proof | PhantomAir |  | 1 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 26 | 
| app_proof | RangeTupleCheckerAir<2> |  | 1 | 0 | 
| app_proof | RevealAir |  | 1 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 1 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> |  | 1 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> |  | 1 | 0 | 
| app_proof | Sha2MainAir<Sha512Config> |  | 1 | 0 | 
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
| app_proof | 0 | ProgramAir | 2 | Program |  | 0 | 295,808 | 228,480 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange |  | 0 | 128 |  | 4 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 0 | Execution |  | 0 | 6,700,032 | 1,688,576 | 209,376 | 52,768 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 1 | Memory |  | 0 | 127,300,608 | 32,082,944 | 3,978,144 | 1,002,592 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 2 | Program |  | 0 | 3,350,016 | 844,288 | 104,688 | 26,384 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 3 | VariableRange |  | 0 | 147,400,704 | 37,148,672 | 4,606,272 | 1,160,896 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 7 | Sha2Block |  | 0 | 10,050,048 | 2,532,864 | 314,064 | 79,152 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 3 | VariableRange |  | 0 | 911,204,352 | 162,537,472 | 28,475,136 | 5,079,296 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 6 | BitwiseLookup |  | 0 | 455,602,176 | 81,268,736 | 14,237,568 | 2,539,648 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 7 | Sha2Block |  | 0 | 170,850,816 | 30,475,776 | 5,339,088 | 952,368 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 8 | Sha2SubAir |  | 0 | 113,900,544 | 20,317,184 | 3,559,392 | 634,912 | 
| app_proof | 15 | HintStoreAir | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 15 | HintStoreAir | 1 | Memory |  | 0 | 384 |  | 12 |  | 
| app_proof | 15 | HintStoreAir | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 15 | HintStoreAir | 3 | VariableRange |  | 0 | 576 |  | 18 |  | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | Execution |  | 0 | 33,500,736 | 53,696 | 1,046,898 | 1,678 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | Memory |  | 0 | 67,001,472 | 107,392 | 2,093,796 | 3,356 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 2 | Program |  | 0 | 16,750,368 | 26,848 | 523,449 | 839 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 3 | VariableRange |  | 0 | 67,001,472 | 107,392 | 2,093,796 | 3,356 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 150,753,312 | 241,632 | 4,711,041 | 7,551 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 1 | Memory |  | 0 | 256 |  | 8 |  | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 448 |  | 14 |  | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | Execution |  | 0 | 13,400,448 | 3,376,768 | 418,764 | 105,524 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | Memory |  | 0 | 26,800,896 | 6,753,536 | 837,528 | 211,048 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 2 | Program |  | 0 | 6,700,224 | 1,688,384 | 209,382 | 52,762 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 80,402,688 | 20,260,608 | 2,512,584 | 633,144 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory |  | 0 | 37,760 | 27,776 | 1,180 | 868 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle |  | 0 | 18,880 | 13,888 | 590 | 434 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression |  | 0 | 18,880 | 13,888 | 590 | 434 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | Execution |  | 0 | 67,324,480 | 66,893,248 | 2,103,890 | 2,090,414 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | Memory |  | 0 | 134,648,960 | 133,786,496 | 4,207,780 | 4,180,828 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 2 | Program |  | 0 | 33,662,240 | 33,446,624 | 1,051,945 | 1,045,207 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 302,960,160 | 301,019,616 | 9,467,505 | 9,406,863 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 0 | Execution |  | 0 | 6,700,928 | 1,687,680 | 209,404 | 52,740 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 1 | Memory |  | 0 | 6,700,928 | 1,687,680 | 209,404 | 52,740 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 2 | Program |  | 0 | 3,350,464 | 843,840 | 104,702 | 26,370 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 3 | VariableRange |  | 0 | 30,154,176 | 7,594,560 | 942,318 | 237,330 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 0 | Execution |  | 0 | 20,100,544 | 13,453,888 | 628,142 | 420,434 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 1 | Memory |  | 0 | 40,201,088 | 26,907,776 | 1,256,284 | 840,868 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 2 | Program |  | 0 | 10,050,272 | 6,726,944 | 314,071 | 210,217 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 3 | VariableRange |  | 0 | 80,402,176 | 53,815,552 | 2,512,568 | 1,681,736 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 0 | Execution |  | 0 | 27,009,984 | 6,544,448 | 844,062 | 204,514 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 1 | Memory |  | 0 | 27,009,984 | 6,544,448 | 844,062 | 204,514 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 2 | Program |  | 0 | 13,504,992 | 3,272,224 | 422,031 | 102,257 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 3 | VariableRange |  | 0 | 94,534,944 | 22,905,568 | 2,954,217 | 715,799 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 26,905,408 | 6,649,024 | 840,794 | 207,782 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 53,810,816 | 13,298,048 | 1,681,588 | 415,564 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | Program |  | 0 | 13,452,704 | 3,324,512 | 420,397 | 103,891 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 94,168,928 | 23,271,584 | 2,942,779 | 727,237 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 0 | Execution |  | 0 | 40,309,568 | 26,799,296 | 1,259,674 | 837,478 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 1 | Memory |  | 0 | 80,619,136 | 53,598,592 | 2,519,348 | 1,674,956 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 2 | Program |  | 0 | 20,154,784 | 13,399,648 | 629,837 | 418,739 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 3 | VariableRange |  | 0 | 80,619,136 | 53,598,592 | 2,519,348 | 1,674,956 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | Execution |  | 0 | 67,139,776 | 67,077,952 | 2,098,118 | 2,096,186 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 1 | Memory |  | 0 | 268,559,104 | 268,311,808 | 8,392,472 | 8,384,744 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 2 | Program |  | 0 | 33,569,888 | 33,538,976 | 1,049,059 | 1,048,093 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 3 | VariableRange |  | 0 | 335,698,880 | 335,389,760 | 10,490,590 | 10,480,930 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 6 | BitwiseLookup |  | 0 | 201,419,328 | 201,233,856 | 6,294,354 | 6,288,558 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | Execution |  | 0 | 67,105,664 | 3,200 | 2,097,052 | 100 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 1 | Memory |  | 0 | 268,422,656 | 12,800 | 8,388,208 | 400 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 2 | Program |  | 0 | 33,552,832 | 1,600 | 1,048,526 | 50 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 3 | VariableRange |  | 0 | 335,528,320 | 16,000 | 10,485,260 | 500 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 6 | BitwiseLookup |  | 0 | 167,764,160 | 8,000 | 5,242,630 | 250 | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle |  | 0 | 72,768 | 25,536 | 2,274 | 798 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression |  | 0 | 24,256 | 8,512 | 758 | 266 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | Execution |  | 0 | 6,700,032 | 1,688,576 | 209,376 | 52,768 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 1 | Memory |  | 0 | 26,800,128 | 6,754,304 | 837,504 | 211,072 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 2 | Program |  | 0 | 3,350,016 | 844,288 | 104,688 | 26,384 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 3 | VariableRange |  | 0 | 36,850,176 | 9,287,168 | 1,151,568 | 290,224 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 6 | BitwiseLookup |  | 0 | 10,050,048 | 2,532,864 | 314,064 | 79,152 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | Memory |  | 0 | 128 |  | 4 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 3 | VariableRange |  | 0 | 288 |  | 9 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 1 | Memory |  | 0 | 128 |  | 4 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 224 |  | 7 |  | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | Execution |  | 0 | 6,700,096 | 1,688,512 | 209,378 | 52,766 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 1 | Memory |  | 0 | 20,100,288 | 5,065,536 | 628,134 | 158,298 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 2 | Program |  | 0 | 3,350,048 | 844,256 | 104,689 | 26,383 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 26,800,384 | 6,754,048 | 837,512 | 211,064 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | Memory |  | 0 | 576 | 192 | 18 | 6 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 3 | VariableRange |  | 0 | 576 | 192 | 18 | 6 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 768 | 256 | 24 | 8 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | Execution |  | 0 | 67,105,600 | 3,264 | 2,097,050 | 102 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 1 | Memory |  | 0 | 201,316,800 | 9,792 | 6,291,150 | 306 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 2 | Program |  | 0 | 33,552,800 | 1,632 | 1,048,525 | 51 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 335,528,000 | 16,320 | 10,485,250 | 510 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup |  | 0 | 4,194,304 |  | 131,072 |  | 
| app_proof | 49 | PhantomAir | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 49 | PhantomAir | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression |  | 0 | 24,288 | 8,480 | 759 | 265 | 
| app_proof | 51 | VariableRangeCheckerAir | 3 | VariableRange |  | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | Memory |  | 0 | 192 |  | 6 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 3 | VariableRange |  | 0 | 192 |  | 6 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 256 |  | 8 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 9 | RangeTuple |  | 0 | 256 |  | 8 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 9 | RangeTuple |  | 0 | 33,554,432 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir | 2 | Program |  | 1 | 295,808 | 228,480 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 0 | Execution |  | 1 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program |  | 1 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange |  | 1 | 128 |  | 4 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 0 | Execution |  | 1 | 3,785,792 | 408,512 | 118,306 | 12,766 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 1 | Memory |  | 1 | 71,930,048 | 7,761,728 | 2,247,814 | 242,554 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 2 | Program |  | 1 | 1,892,896 | 204,256 | 59,153 | 6,383 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 3 | VariableRange |  | 1 | 83,287,424 | 8,987,264 | 2,602,732 | 280,852 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 7 | Sha2Block |  | 1 | 5,678,688 | 612,768 | 177,459 | 19,149 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 3 | VariableRange |  | 1 | 514,867,712 | 22,003,200 | 16,089,616 | 687,600 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 6 | BitwiseLookup |  | 1 | 257,433,856 | 11,001,600 | 8,044,808 | 343,800 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 7 | Sha2Block |  | 1 | 96,537,696 | 4,125,600 | 3,016,803 | 128,925 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 8 | Sha2SubAir |  | 1 | 64,358,464 | 2,750,400 | 2,011,202 | 85,950 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | Execution |  | 1 | 18,929,280 | 14,625,152 | 591,540 | 457,036 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | Memory |  | 1 | 37,858,560 | 29,250,304 | 1,183,080 | 914,072 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 2 | Program |  | 1 | 9,464,640 | 7,312,576 | 295,770 | 228,518 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 3 | VariableRange |  | 1 | 37,858,560 | 29,250,304 | 1,183,080 | 914,072 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 6 | BitwiseLookup |  | 1 | 85,181,760 | 65,813,184 | 2,661,930 | 2,056,662 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | Execution |  | 1 | 7,573,888 | 814,720 | 236,684 | 25,460 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | Memory |  | 1 | 15,147,776 | 1,629,440 | 473,368 | 50,920 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 2 | Program |  | 1 | 3,786,944 | 407,360 | 118,342 | 12,730 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 3 | VariableRange |  | 1 | 45,443,328 | 4,888,320 | 1,420,104 | 152,760 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory |  | 1 | 39,680 | 25,856 | 1,240 | 808 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle |  | 1 | 19,840 | 12,928 | 620 | 404 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression |  | 1 | 19,840 | 12,928 | 620 | 404 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | Execution |  | 1 | 38,039,552 | 29,069,312 | 1,188,736 | 908,416 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | Memory |  | 1 | 76,079,104 | 58,138,624 | 2,377,472 | 1,816,832 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 2 | Program |  | 1 | 19,019,776 | 14,534,656 | 594,368 | 454,208 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 3 | VariableRange |  | 1 | 171,177,984 | 130,811,904 | 5,349,312 | 4,087,872 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 0 | Execution |  | 1 | 3,785,984 | 408,320 | 118,312 | 12,760 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 1 | Memory |  | 1 | 3,785,984 | 408,320 | 118,312 | 12,760 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 2 | Program |  | 1 | 1,892,992 | 204,160 | 59,156 | 6,380 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 3 | VariableRange |  | 1 | 17,036,928 | 1,837,440 | 532,404 | 57,420 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 0 | Execution |  | 1 | 11,357,760 | 5,419,456 | 354,930 | 169,358 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 1 | Memory |  | 1 | 22,715,520 | 10,838,912 | 709,860 | 338,716 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 2 | Program |  | 1 | 5,678,880 | 2,709,728 | 177,465 | 84,679 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 3 | VariableRange |  | 1 | 45,431,040 | 21,677,824 | 1,419,720 | 677,432 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 0 | Execution |  | 1 | 15,261,760 | 1,515,456 | 476,930 | 47,358 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 1 | Memory |  | 1 | 15,261,760 | 1,515,456 | 476,930 | 47,358 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 2 | Program |  | 1 | 7,630,880 | 757,728 | 238,465 | 23,679 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 3 | VariableRange |  | 1 | 53,416,160 | 5,304,096 | 1,669,255 | 165,753 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | Execution |  | 1 | 15,202,496 | 1,574,720 | 475,078 | 49,210 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | Memory |  | 1 | 30,404,992 | 3,149,440 | 950,156 | 98,420 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | Program |  | 1 | 7,601,248 | 787,360 | 237,539 | 24,605 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | VariableRange |  | 1 | 53,208,736 | 5,511,520 | 1,662,773 | 172,235 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 0 | Execution |  | 1 | 22,775,104 | 10,779,328 | 711,722 | 336,854 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 1 | Memory |  | 1 | 45,550,208 | 21,558,656 | 1,423,444 | 673,708 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 2 | Program |  | 1 | 11,387,552 | 5,389,664 | 355,861 | 168,427 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 3 | VariableRange |  | 1 | 45,550,208 | 21,558,656 | 1,423,444 | 673,708 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | Execution |  | 1 | 37,918,976 | 29,189,888 | 1,184,968 | 912,184 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 1 | Memory |  | 1 | 151,675,904 | 116,759,552 | 4,739,872 | 3,648,736 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 2 | Program |  | 1 | 18,959,488 | 14,594,944 | 592,484 | 456,092 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 3 | VariableRange |  | 1 | 189,594,880 | 145,949,440 | 5,924,840 | 4,560,920 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 6 | BitwiseLookup |  | 1 | 113,756,928 | 87,569,664 | 3,554,904 | 2,736,552 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | Execution |  | 1 | 37,919,872 | 29,188,992 | 1,184,996 | 912,156 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 1 | Memory |  | 1 | 151,679,488 | 116,755,968 | 4,739,984 | 3,648,624 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 2 | Program |  | 1 | 18,959,936 | 14,594,496 | 592,498 | 456,078 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 3 | VariableRange |  | 1 | 189,599,360 | 145,944,960 | 5,924,980 | 4,560,780 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 6 | BitwiseLookup |  | 1 | 94,799,680 | 72,972,480 | 2,962,490 | 2,280,390 | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle |  | 1 | 73,152 | 25,152 | 2,286 | 786 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression |  | 1 | 24,384 | 8,384 | 762 | 262 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | Execution |  | 1 | 3,786,304 | 408,000 | 118,322 | 12,750 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 1 | Memory |  | 1 | 15,145,216 | 1,632,000 | 473,288 | 51,000 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 2 | Program |  | 1 | 1,893,152 | 204,000 | 59,161 | 6,375 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 3 | VariableRange |  | 1 | 20,824,672 | 2,244,000 | 650,771 | 70,125 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 6 | BitwiseLookup |  | 1 | 5,679,456 | 612,000 | 177,483 | 19,125 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 0 | Execution |  | 1 | 2,560 | 1,536 | 80 | 48 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 1 | Memory |  | 1 | 7,680 | 4,608 | 240 | 144 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 2 | Program |  | 1 | 1,280 | 768 | 40 | 24 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 3 | VariableRange |  | 1 | 10,240 | 6,144 | 320 | 192 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 6 | BitwiseLookup |  | 1 | 2,560 | 1,536 | 80 | 48 | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | Execution |  | 1 | 512 |  | 16 |  | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 1 | Memory |  | 1 | 1,536 |  | 48 |  | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 2 | Program |  | 1 | 256 |  | 8 |  | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 3 | VariableRange |  | 1 | 2,304 |  | 72 |  | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 6 | BitwiseLookup |  | 1 | 256 |  | 8 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | Execution |  | 1 | 192 | 64 | 6 | 2 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | Memory |  | 1 | 384 | 128 | 12 | 4 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 2 | Program |  | 1 | 96 | 32 | 3 | 1 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 3 | VariableRange |  | 1 | 864 | 288 | 27 | 9 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | Execution |  | 1 | 896 | 128 | 28 | 4 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | Memory |  | 1 | 2,688 | 384 | 84 | 12 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 2 | Program |  | 1 | 448 | 64 | 14 | 2 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 3 | VariableRange |  | 1 | 6,720 | 960 | 210 | 30 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | Execution |  | 1 | 3,785,792 | 408,512 | 118,306 | 12,766 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 1 | Memory |  | 1 | 11,357,376 | 1,225,536 | 354,918 | 38,298 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 2 | Program |  | 1 | 1,892,896 | 204,256 | 59,153 | 6,383 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 3 | VariableRange |  | 1 | 15,143,168 | 1,634,048 | 473,224 | 51,064 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | Execution |  | 1 | 1,152 | 896 | 36 | 28 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | Memory |  | 1 | 3,456 | 2,688 | 108 | 84 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 2 | Program |  | 1 | 576 | 448 | 18 | 14 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 3 | VariableRange |  | 1 | 3,456 | 2,688 | 108 | 84 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 6 | BitwiseLookup |  | 1 | 4,608 | 3,584 | 144 | 112 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | Execution |  | 1 | 37,917,568 | 29,191,296 | 1,184,924 | 912,228 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 1 | Memory |  | 1 | 113,752,704 | 87,573,888 | 3,554,772 | 2,736,684 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 2 | Program |  | 1 | 18,958,784 | 14,595,648 | 592,462 | 456,114 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 3 | VariableRange |  | 1 | 189,587,840 | 145,956,480 | 5,924,620 | 4,561,140 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup |  | 1 | 4,194,304 |  | 131,072 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression |  | 1 | 43,328 | 22,208 | 1,354 | 694 | 
| app_proof | 51 | VariableRangeCheckerAir | 3 | VariableRange |  | 1 | 8,388,608 |  | 262,144 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 9 | RangeTuple |  | 1 | 33,554,432 |  | 1,048,576 |  | 

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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 19,726,336 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 38 | 622,592 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 71 | 142 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 544 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 256 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 102,400 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 1,280 | 
| leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 360,448 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 524,288 | 301 | 157,810,688 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 38 | 2,490,368 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 512 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 52 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 2,944 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 46,080 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 20 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 1,728 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 3,648 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 143,360 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 816 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 368 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 1,216 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 32,768 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 524,288 | 89 | 46,661,632 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 229,376 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 507,904 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 69,632 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 23,592,960 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 3,072 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 8,448 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 2,624 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 819,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 544 | 

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover |  | 0 | 131,072 | 131 | 17,170,432 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 15 | HintStoreAir | prover |  | 0 | 2 | 24 | 48 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 524,288 | 34 | 17,825,792 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 2 | 27 | 54 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 262,144 | 51 | 13,369,344 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 512 | 38 | 19,456 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 2,097,152 | 23 | 48,234,496 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 131,072 | 16 | 2,097,152 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 524,288 | 23 | 12,058,624 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 524,288 | 18 | 9,437,184 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 524,288 | 30 | 15,728,640 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 2,097,152 | 38 | 79,691,776 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 1,048,576 | 38 | 39,845,888 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 1,024 | 33 | 33,792 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 131,072 | 37 | 4,849,664 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 1 | 44 | 44 | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1 | 22 | 22 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 131,072 | 28 | 3,670,016 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 4 | 42 | 168 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 1,048,576 | 29 | 30,408,704 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | PhantomAir | prover |  | 0 | 1 | 7 | 7 | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 256 | 300 | 76,800 | 
| app_proof | 51 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 1 | 40 | 40 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover |  | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover |  | 1 | 65,536 | 131 | 8,585,216 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 1 | 524,288 | 34 | 17,825,792 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 1 | 131,072 | 51 | 6,684,672 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 1 | 512 | 38 | 19,456 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 1 | 65,536 | 16 | 1,048,576 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 1 | 262,144 | 23 | 6,029,312 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 1 | 262,144 | 30 | 7,864,320 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 1 | 524,288 | 24 | 12,582,912 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 1 | 1,048,576 | 38 | 39,845,888 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 1 | 1,048,576 | 38 | 39,845,888 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 1 | 512 | 33 | 16,896 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 1 | 65,536 | 37 | 2,424,832 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 1 | 64 | 28 | 1,792 | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 1 | 8 | 29 | 232 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 1 | 4 | 44 | 176 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover |  | 1 | 16 | 58 | 928 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 1 | 65,536 | 28 | 1,835,008 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 1 | 32 | 42 | 1,344 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 29 | 30,408,704 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | prover |  | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 1 | 512 | 300 | 153,600 | 
| app_proof | 51 | VariableRangeCheckerAir | prover |  | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 1 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_cells_unpadded | metered_interaction_cells_padding | metered_constraint_eval_cells_unpadded | metered_constraint_eval_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 9,244 | 7,140 | 369,760 | 285,600 | 92,440 | 71,400 | 9,244 | 7,140 |  |  | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> |  | 0 | 104,688 | 26,384 | 54,856,512 | 13,825,216 | 13,714,128 | 3,456,304 | 9,212,544 | 2,321,792 | 314,064 | 79,152 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 1,779,696 | 317,456 | 3,246,165,504 | 579,039,744 | 811,541,376 | 144,759,936 | 51,611,184 | 9,206,224 | 1,302,737,472 | 232,377,792 | 
| app_proof | 15 | HintStoreAir |  | 0 | 2 |  | 192 |  | 48 |  | 36 |  | 14 |  | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 523,449 | 839 | 71,189,064 | 114,104 | 17,797,266 | 28,526 | 10,468,980 | 16,780 | 2,093,796 | 3,356 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 2 |  | 216 |  | 54 |  | 28 |  | 16 |  | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 209,382 | 52,762 | 42,713,928 | 10,763,448 | 10,678,482 | 2,690,862 | 3,978,258 | 1,002,478 | 4,606,404 | 1,160,764 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 295 | 217 | 44,840 | 32,984 | 11,210 | 8,246 | 2,360 | 1,736 | 1,180 | 868 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 1,051,945 | 1,045,207 | 96,778,940 | 96,159,044 | 24,194,735 | 24,039,761 | 16,831,120 | 16,723,312 | 5,259,725 | 5,226,035 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 104,702 | 26,370 | 6,700,928 | 1,687,680 | 1,675,232 | 421,920 | 1,465,828 | 369,180 | 418,808 | 105,480 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 314,071 | 210,217 | 28,894,532 | 19,339,964 | 7,223,633 | 4,834,991 | 4,711,065 | 3,153,255 | 1,570,355 | 1,051,085 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 422,031 | 102,257 | 30,386,232 | 7,362,504 | 7,596,558 | 1,840,626 | 5,064,372 | 1,227,084 | 4,220,310 | 1,022,570 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 420,397 | 103,891 | 50,447,640 | 12,466,920 | 12,611,910 | 3,116,730 | 5,885,558 | 1,454,474 | 3,363,176 | 831,128 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 629,837 | 418,739 | 60,464,352 | 40,198,944 | 15,116,088 | 10,049,736 | 6,928,207 | 4,606,129 | 4,408,859 | 2,931,173 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 1,049,059 | 1,048,093 | 159,456,968 | 159,310,136 | 39,864,242 | 39,827,534 | 28,324,593 | 28,298,511 | 6,294,354 | 6,288,558 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 1,048,526 | 50 | 159,375,952 | 7,600 | 39,843,988 | 1,900 | 27,261,676 | 1,300 | 6,291,156 | 300 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 758 | 266 | 100,056 | 35,112 | 25,014 | 8,778 | 3,032 | 1,064 | 6,822 | 2,394 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 104,688 | 26,384 | 15,493,824 | 3,904,832 | 3,873,456 | 976,208 | 2,617,200 | 659,600 | 628,128 | 158,304 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 1 |  | 176 |  | 44 |  | 16 |  | 22 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 1 |  | 88 |  | 22 |  | 14 |  | 5 |  | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 104,689 | 26,383 | 11,725,168 | 2,954,896 | 2,931,292 | 738,724 | 1,779,713 | 448,511 | 837,512 | 211,064 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 3 | 1 | 504 | 168 | 126 | 42 | 69 | 23 | 12 | 4 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 1,048,525 | 51 | 121,628,900 | 5,916 | 30,407,225 | 1,479 | 19,921,975 | 969 | 8,388,200 | 408 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 49 | PhantomAir |  | 0 | 1 |  | 28 |  | 7 |  | 3 |  | 2 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 759 | 265 | 910,800 | 318,000 | 227,700 | 79,500 | 759 | 265 | 25,047 | 8,745 | 
| app_proof | 51 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 1 |  | 160 |  | 40 |  | 31 |  | 2 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 12,582,912 |  | 3,145,728 |  | 1,048,576 |  | 5,242,880 |  | 
| app_proof | 0 | ProgramAir |  | 1 | 9,244 | 7,140 | 369,760 | 285,600 | 92,440 | 71,400 | 9,244 | 7,140 |  |  | 
| app_proof | 1 | VmConnectorAir |  | 1 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> |  | 1 | 59,153 | 6,383 | 30,996,172 | 3,344,692 | 7,749,043 | 836,173 | 5,205,464 | 561,704 | 177,459 | 19,149 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> |  | 1 | 1,005,601 | 42,975 | 1,834,216,224 | 78,386,400 | 458,554,056 | 19,596,600 | 29,162,429 | 1,246,275 | 736,099,932 | 31,457,700 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 295,770 | 228,518 | 40,224,720 | 31,078,448 | 10,056,180 | 7,769,612 | 5,915,400 | 4,570,360 | 1,183,080 | 914,072 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 118,342 | 12,730 | 24,141,768 | 2,596,920 | 6,035,442 | 649,230 | 2,248,498 | 241,870 | 2,603,524 | 280,060 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 1 | 310 | 202 | 47,120 | 30,704 | 11,780 | 7,676 | 2,480 | 1,616 | 1,240 | 808 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 594,368 | 454,208 | 54,681,856 | 41,787,136 | 13,670,464 | 10,446,784 | 9,509,888 | 7,267,328 | 2,971,840 | 2,271,040 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 1 | 59,156 | 6,380 | 3,785,984 | 408,320 | 946,496 | 102,080 | 828,184 | 89,320 | 236,624 | 25,520 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 1 | 177,465 | 84,679 | 16,326,780 | 7,790,468 | 4,081,695 | 1,947,617 | 2,661,975 | 1,270,185 | 887,325 | 423,395 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 1 | 238,465 | 23,679 | 17,169,480 | 1,704,888 | 4,292,370 | 426,222 | 2,861,580 | 284,148 | 2,384,650 | 236,790 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 237,539 | 24,605 | 28,504,680 | 2,952,600 | 7,126,170 | 738,150 | 3,325,546 | 344,470 | 1,900,312 | 196,840 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 355,861 | 168,427 | 34,162,656 | 16,168,992 | 8,540,664 | 4,042,248 | 3,914,471 | 1,852,697 | 2,491,027 | 1,178,989 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 592,484 | 456,092 | 90,057,568 | 69,325,984 | 22,514,392 | 17,331,496 | 15,997,068 | 12,314,484 | 3,554,904 | 2,736,552 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 592,498 | 456,078 | 90,059,696 | 69,323,856 | 22,514,924 | 17,330,964 | 15,404,948 | 11,858,028 | 3,554,988 | 2,736,468 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 1 | 762 | 262 | 100,584 | 34,584 | 25,146 | 8,646 | 3,048 | 1,048 | 6,858 | 2,358 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 59,161 | 6,375 | 8,755,828 | 943,500 | 2,188,957 | 235,875 | 1,479,025 | 159,375 | 354,966 | 38,250 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 40 | 24 | 4,480 | 2,688 | 1,120 | 672 | 760 | 456 | 240 | 144 | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 1 | 8 |  | 928 |  | 232 |  | 152 |  | 48 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 1 | 3 | 1 | 528 | 176 | 132 | 44 | 48 | 16 | 66 | 22 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 1 | 14 | 2 | 3,248 | 464 | 812 | 116 | 336 | 48 | 308 | 44 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 59,153 | 6,383 | 6,625,136 | 714,896 | 1,656,284 | 178,724 | 1,005,601 | 108,511 | 473,224 | 51,064 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 18 | 14 | 3,024 | 2,352 | 756 | 588 | 414 | 322 | 72 | 56 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 592,462 | 456,114 | 68,725,592 | 52,909,224 | 17,181,398 | 13,227,306 | 11,256,778 | 8,666,166 | 4,739,696 | 3,648,912 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> |  | 1 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 1,354 | 694 | 1,624,800 | 832,800 | 406,200 | 208,200 | 1,354 | 694 | 44,682 | 22,902 | 
| app_proof | 51 | VariableRangeCheckerAir |  | 1 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 1 | 1,048,576 |  | 12,582,912 |  | 3,145,728 |  | 1,048,576 |  | 5,242,880 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 0 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 21 | 197 | 21 | 7 | 1 | 2 | 3 | 3 | 
| internal_recursive.0 | 1 | 12 | 119 | 12 | 2 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 114 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 117 | 534 | 117 | 33 | 12 | 2 | 10 | 10 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,627,133 | 175 | 45 | 0 | 0 | 81 | 29 | 28 | 38 | 14 | 0 | 48 | 37 | 10 | 2 | 7 | 45 | 45 | 81 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,386,961 | 107 | 19 | 0 | 0 | 57 | 22 | 21 | 23 | 11 | 0 | 29 | 21 | 7 | 1 | 6 | 20 | 19 | 57 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,759,057 | 103 | 14 | 0 | 0 | 54 | 20 | 19 | 22 | 11 | 0 | 33 | 26 | 7 | 1 | 5 | 14 | 14 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 237,994,808 | 416 | 142 | 0 | 0 | 156 | 83 | 82 | 44 | 28 | 0 | 116 | 94 | 21 | 10 | 11 | 142 | 142 | 156 | 0 | 3 | 27 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 15,094,533 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 5 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,281,523,083 | 2,255 | 560 | 0 | 0 | 1,360 | 849 | 848 | 236 | 273 | 1 | 334 | 241 | 92 | 56 | 35 | 560 | 560 | 1,360 | 0 | 1 | 272 | 0 | 0 | 
| app_proof | prover |  | 1 | 687,699,844 | 1,195 | 208 | 0 | 0 | 793 | 441 | 441 | 125 | 225 | 0 | 193 | 142 | 50 | 30 | 20 | 209 | 208 | 793 | 0 | 1 | 224 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 267,018,726 | 2,013,265,921 | 
| app_proof | prover |  | 1 | 0 | 163,079,170 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 1,500 | 61 | 11,167,961 | 181.07 | 0 | 4,496 | 

| group | program | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- |
| app_proof |  | memory | 0 | 1 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_whir_memory_bytes | metered_secondary_peak_memory_bytes | metered_rs_code_matrix_memory_bytes | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_gkr_memory_bytes | metered_batch_constraint_memory_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 5 | 2 | 75 | 2,930 | 6 | 5 | 0 | 194 | 2 | 20 | 0 | 125,829,120 | 10,960,542,912 | 10,267,656,192 | 12,915,749,568 | 3,171,807,276 | 16,087,556,844 | 9,383,484,608 | 10,960,542,912 | 2 | 0 | 2 | 5 | 69 | 352 | 7,136,000 | 20.25 | 0 | 0 | 
| app_proof |  | 1 | 3 | 2 | 32 | 1,500 | 3 | 3 | 0 | 76 | 1 | 5 | 0 | 125,829,120 | 6,812,188,528 | 5,519,704,064 | 7,305,829,524 | 2,259,069,164 | 9,564,898,688 | 6,057,508,928 | 6,812,188,528 | 2 | 0 | 0 | 2 | 29 | 167 | 4,031,961 | 24.09 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/057e3c53522cc059db3ad0abaeb3bad9862017af

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33812815335)
