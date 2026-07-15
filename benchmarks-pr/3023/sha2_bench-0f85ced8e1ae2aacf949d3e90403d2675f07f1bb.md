| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.13 |  3.55 |  3.55 |
| app_proof |  4.18 |  2.60 |  2.60 |
| leaf |  0.52 |  0.52 |  0.52 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,063.50 |  4,127 |  2,541 |  1,586 |
| `compile_metered_time_ms` |  6 |  6 |  6 |  6 |
| `execute_metered_time_ms` |  56 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  198.23 | -          |  198.23 |  198.23 |
| `execute_preflight_insns` |  5,583,980.50 |  11,167,961 |  7,147,000 |  4,020,961 |
| `execute_preflight_time_ms` |  342 |  684 |  476 |  208 |
| `execute_preflight_insn_mi/s` |  35.91 | -          |  36.38 |  35.45 |
| `trace_gen_time_ms   ` |  71.50 |  143 |  99 |  44 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,648.50 |  3,297 |  2,232 |  1,065 |
| `prover.main_trace_commit_time_ms` |  405 |  810 |  596 |  214 |
| `prover.rap_constraints_time_ms` |  968.50 |  1,937 |  1,280 |  657 |
| `prover.openings_time_ms` |  273.50 |  547 |  355 |  192 |
| `prover.rap_constraints.logup_gkr_time_ms` |  265.50 |  531 |  347 |  184 |
| `prover.rap_constraints.round0_time_ms` |  521.50 |  1,043 |  697 |  346 |
| `prover.rap_constraints.mle_rounds_time_ms` |  181 |  362 |  236 |  126 |
| `prover.openings.stacked_reduction_time_ms` |  74.50 |  149 |  97 |  52 |
| `prover.openings.stacked_reduction.round0_time_ms` |  45 |  90 |  60 |  30 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  29 |  58 |  37 |  21 |
| `prover.openings.whir_time_ms` |  198.50 |  397 |  257 |  140 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  523 |  523 |  523 |  523 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  115 |  115 |  115 |  115 |
| `generate_blob_total_time_ms` |  13 |  13 |  13 |  13 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  407 |  407 |  407 |  407 |
| `prover.main_trace_commit_time_ms` |  146 |  146 |  146 |  146 |
| `prover.rap_constraints_time_ms` |  142 |  142 |  142 |  142 |
| `prover.openings_time_ms` |  119 |  119 |  119 |  119 |
| `prover.rap_constraints.logup_gkr_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.round0_time_ms` |  71 |  71 |  71 |  71 |
| `prover.rap_constraints.mle_rounds_time_ms` |  44 |  44 |  44 |  44 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  97 |  97 |  97 |  97 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  197 |  197 |  197 |  197 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  20 |  20 |  20 |  20 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  176 |  176 |  176 |  176 |
| `prover.main_trace_commit_time_ms` |  47 |  47 |  47 |  47 |
| `prover.rap_constraints_time_ms` |  79 |  79 |  79 |  79 |
| `prover.openings_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  39 |  39 |  39 |  39 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  119 |  119 |  119 |  119 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  108 |  108 |  108 |  108 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  24 |  24 |  24 |  24 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  112 |  112 |  112 |  112 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  103 |  103 |  103 |  103 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  26 |  26 |  26 |  26 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/0f85ced8e1ae2aacf949d3e90403d2675f07f1bb/sha2_bench-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.batch_constraints.fold_ple_evals | 16.58 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 16.58 | app_proof.prover.0 |
| prover.rap_constraints | 16.58 | app_proof.prover.0 |
| prover.stacked_commit | 15.08 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 14.28 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 14.28 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 13.41 | app_proof.prover.0 |
| prover.gkr_input_evals | 13.41 | app_proof.prover.0 |
| prover.openings | 10.17 | app_proof.prover.0 |
| prover.prove_whir_opening | 10.17 | app_proof.prover.0 |
| prover.merkle_tree | 10.17 | app_proof.prover.0 |
| prover.rs_code_matrix | 10.16 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 5.12 | app_proof.prover.0 |
| tracegen.pow_checker | 1.10 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.10 | leaf.0 |
| tracegen.exp_bits_len | 1.10 | leaf.0 |
| tracegen.whir_folding | 0.97 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.97 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.97 | leaf.0 |
| generate mem proving ctxs | 0.87 | app_proof.1 |
| set initial memory | 0.87 | app_proof.1 |
| tracegen.proof_shape | 0.80 | leaf.0 |
| tracegen.public_values | 0.80 | leaf.0 |
| tracegen.range_checker | 0.80 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 110 | 267,239 | 229,016 | 23 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 1 | 149 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 1 | 85 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 14 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 7 | 2 | 
| 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 45 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 99 | 3 | 
| 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 8 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> |  | 17 | 11 | 3 | 
| 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> |  | 17 | 13 | 3 | 
| 27 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> |  | 17 | 11 | 3 | 
| 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> |  | 17 | 13 | 3 | 
| 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> |  | 18 | 14 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 30 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> |  | 17 | 13 | 3 | 
| 31 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> |  | 17 | 15 | 3 | 
| 32 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 18 | 16 | 3 | 
| 33 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> |  | 19 | 16 | 3 | 
| 34 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> |  | 18 | 17 | 3 | 
| 35 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 18 | 3 | 
| 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 30 | 3 | 
| 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 70 | 3 | 
| 38 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 7 | 3 | 
| 39 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 31 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 40 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 103 | 3 | 
| 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 46 | 3 | 
| 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 132 | 3 | 
| 43 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 23 | 3 | 
| 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> |  | 18 | 11 | 3 | 
| 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 7 | 2 | 
| 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> |  | 19 | 14 | 3 | 
| 47 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 48 | PhantomAir |  | 3 | 1 | 2 | 
| 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 50 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | new_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 57 |  |  |  | 284 |  |  |  |  |  |  | 
| app_proof |  |  |  | 1,586 |  | 56 | 11,167,961 | 198.23 | 0 | 4,192 |  | 
| internal_for_leaf |  |  | 197 |  |  |  |  |  |  |  | 197 | 
| internal_recursive.0 |  |  | 119 |  |  |  |  |  |  |  | 119 | 
| internal_recursive.1 |  |  | 112 |  |  |  |  |  |  |  | 113 | 
| leaf |  | 523 |  |  |  |  |  |  |  |  | 523 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 20 | 
| app_proof | Rv64HintStoreAir | 40 | Rv64HintStoreAir | 0 | 104 | 
| app_proof | Sha2MainAir<Sha256Config> | 42 | Sha2MainAir<Sha256Config> | 0 | 28,518,928 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 39 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 23,067,176 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 35 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | 46,356,948 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 38 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 96 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 36 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 9,226,976 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 180 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 63,008,340 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 16 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | 48 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 17 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 48 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 6,710,400 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 30,278,688 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 31 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 20,210,112 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 32 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 16,907,320 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 33 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 15,098,640 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 54,607,644 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 5,452,200 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 46 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 56 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 34 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 3,355,616 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 54,635,256 | 
| app_proof | Sha2MainAir<Sha256Config> | 42 | Sha2MainAir<Sha256Config> | 1 | 16,045,824 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 39 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | 12,978,460 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 35 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | 26,080,824 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 36 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | 5,192,880 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 1,080 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 1 | 35,450,880 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 840 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 17 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | 144 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 1 | 3,775,488 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 17,034,816 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 31 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 11,370,816 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 32 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 9,512,520 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 33 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 8,495,088 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 1 | 30,725,604 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 19 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 1 | 416 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 1 | 3,067,948 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 34 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 1,887,840 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 21 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 1 | 2,080 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 1 | 30,724,980 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 |  | 6 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 7 | 7 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 |  | 6 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 0 |  | 10 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 3 | 5 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 2 | 3 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 1 | 2 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 1 | 2 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 4 | 8 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 5 | 9 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 |  | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | 16 | 16 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 |  | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 1 |  | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 1 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 1 | 2 | 2 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 1 | 2 | 2 | 

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
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
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

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 104,849 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_BUFFER | 0 | 1 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_STORED | 0 | 1 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | ANDI | 0 | 524,254 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | SLTIU | 0 | 2 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SLLI | 0 | 104,851 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SRLI | 0 | 104,853 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | ADDI | 0 | 1,053,567 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 104,863 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 314,555 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 316,188 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 106,495 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 104,852 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 316,192 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 421,040 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 209,766 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | STORED | 0 | 1,050,678 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | LOADD | 0 | 1,050,147 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | LOADW | 0 | 104,850 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | SRLIW | 0 | 1 | 
| app_proof | 38 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | ADDIW | 0 | 1 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 0 | 104,850 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 2 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 1 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 0 | 735,588 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 0 | 314,551 | 
| app_proof | 48 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 1 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 1 | 58,992 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | ANDI | 1 | 294,965 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SLLI | 1 | 58,998 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SRLI | 1 | 59,022 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | ADDI | 1 | 592,746 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 1 | 58,995 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 1 | 176,981 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 1 | 177,899 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 1 | 59,914 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 1 | 58,992 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLT | 1 | 1 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 1 | 177,899 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 236,893 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 117,999 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | STORED | 1 | 590,865 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | LOADD | 1 | 590,877 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | LOADW | 1 | 58,999 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | STOREB | 1 | 40 | 
| app_proof | 35 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | LOADB | 1 | 8 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | SLLIW | 1 | 1 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | SRLIW | 1 | 2 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 1 | 7 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 1 | 7 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 1 | 58,992 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 1 | 4 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 1 | 14 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 1 | 413,869 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 1 | 176,979 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 0 | 131,072 | 150 | 19,660,800 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 14 | Rv64HintStoreAir | prover | 0 | 2 | 27 | 54 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover | 0 | 524,288 | 36 | 18,874,368 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover | 0 | 2 | 30 | 60 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover | 0 | 262,144 | 54 | 14,155,776 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover | 0 | 2,097,152 | 25 | 52,428,800 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 131,072 | 17 | 2,228,224 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 524,288 | 24 | 12,582,912 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 524,288 | 18 | 9,437,184 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 524,288 | 32 | 16,777,216 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 1,048,576 | 26 | 27,262,976 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | prover | 0 | 2,097,152 | 30 | 62,914,560 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | prover | 0 | 2,097,152 | 30 | 62,914,560 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | prover | 0 | 131,072 | 31 | 4,063,232 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 1,024 | 33 | 33,792 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover | 0 | 1 | 47 | 47 | 
| app_proof | 38 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover | 0 | 1 | 24 | 24 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | prover | 0 | 131,072 | 31 | 4,063,232 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 4 | 45 | 180 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | prover | 0 | 2,097,152 | 32 | 67,108,864 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 1 | 43 | 43 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 1 | 65,536 | 150 | 9,830,400 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover | 1 | 524,288 | 36 | 18,874,368 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover | 1 | 131,072 | 54 | 7,077,888 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover | 1 | 1,048,576 | 25 | 26,214,400 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 17 | 1,114,112 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 262,144 | 24 | 6,291,456 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 262,144 | 32 | 8,388,608 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 524,288 | 26 | 13,631,488 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | prover | 1 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | prover | 1 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | prover | 1 | 65,536 | 31 | 2,031,616 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 1,024 | 33 | 33,792 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | prover | 1 | 64 | 36 | 2,304 | 
| app_proof | 35 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | prover | 1 | 8 | 35 | 280 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover | 1 | 4 | 47 | 188 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 1 | 16 | 64 | 1,024 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | prover | 1 | 65,536 | 31 | 2,031,616 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 1 | 32 | 45 | 1,440 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | prover | 1 | 1,048,576 | 32 | 33,554,432 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 50 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 0 | 104,849 | 26,223 | 78,636,750 | 19,667,250 | 62,909,400 | 15,733,800 | 15,727,350 | 3,933,450 | 323,065,982 | 80,799,618 | 8,912,165 | 2,228,955 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 1,782,433 | 314,719 | 4,063,947,240 | 717,559,320 | 3,251,157,792 | 574,047,456 | 812,789,448 | 143,511,864 | 1,873,782,692 | 330,848,348 | 51,690,557 | 9,126,851 | 
| app_proof | 14 | Rv64HintStoreAir | 0 | 2 |  | 270 |  | 216 |  | 54 |  | 1,233 |  | 34 |  | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 524,254 | 34 | 47,182,860 | 3,060 | 75,492,576 | 4,896 | 18,873,144 | 1,224 | 380,084,150 | 24,650 | 10,485,080 | 680 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 2 |  | 150 |  | 240 |  | 60 |  | 1,015 |  | 28 |  | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 209,704 | 52,440 | 28,310,040 | 7,079,400 | 45,296,064 | 11,327,040 | 11,324,016 | 2,831,760 | 144,433,630 | 36,118,050 | 3,984,376 | 996,360 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | 1,053,567 | 1,043,585 | 65,847,938 | 65,224,062 | 105,356,700 | 104,358,500 | 26,339,175 | 26,089,625 | 611,068,860 | 605,279,300 | 16,857,072 | 16,697,360 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 590 | 434 | 30,975 | 22,785 | 49,560 | 36,456 | 12,390 | 9,114 | 85,550 | 62,930 | 2,360 | 1,736 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 104,863 | 26,209 | 4,456,678 | 1,113,882 | 7,130,684 | 1,782,212 | 1,782,671 | 445,553 | 53,217,973 | 13,301,067 | 1,468,082 | 366,926 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 314,555 | 209,733 | 18,873,300 | 12,583,980 | 30,197,280 | 20,134,368 | 7,549,320 | 5,033,592 | 171,039,282 | 114,042,318 | 4,718,325 | 3,145,995 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 422,683 | 101,605 | 19,020,735 | 4,572,225 | 30,433,176 | 7,315,560 | 7,608,294 | 1,828,890 | 183,867,105 | 44,198,175 | 5,072,196 | 1,219,260 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 421,044 | 103,244 | 33,683,520 | 8,259,520 | 53,893,632 | 13,215,232 | 13,473,408 | 3,303,808 | 213,679,830 | 52,396,330 | 5,894,616 | 1,445,416 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 630,806 | 417,770 | 41,002,390 | 27,155,050 | 65,603,824 | 43,448,080 | 16,400,956 | 10,862,020 | 251,533,893 | 166,585,787 | 6,938,866 | 4,595,470 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 1,050,678 | 1,046,474 | 78,800,850 | 78,485,550 | 126,081,360 | 125,576,880 | 31,520,340 | 31,394,220 | 647,480,318 | 644,889,602 | 17,861,526 | 17,790,058 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 1,050,147 | 1,047,005 | 78,761,025 | 78,525,375 | 126,017,640 | 125,640,600 | 31,504,410 | 31,410,150 | 647,153,089 | 645,216,831 | 17,852,499 | 17,799,085 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 104,850 | 26,222 | 8,125,875 | 2,032,205 | 13,001,400 | 3,251,528 | 3,250,350 | 812,882 | 68,414,625 | 17,109,855 | 1,887,300 | 471,996 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 746 | 278 | 123,090 | 45,870 | 98,472 | 36,696 | 24,618 | 9,174 | 108,170 | 40,310 | 2,984 | 1,112 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 1 |  | 118 |  | 188 |  | 47 |  | 580 |  | 16 |  | 
| app_proof | 38 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | 1 |  | 60 |  | 96 |  | 24 |  | 508 |  | 14 |  | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 104,850 | 26,222 | 8,125,875 | 2,032,205 | 13,001,400 | 3,251,528 | 3,250,350 | 812,882 | 68,414,625 | 17,109,855 | 1,887,300 | 471,996 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 3 | 1 | 338 | 112 | 540 | 180 | 135 | 45 | 2,502 | 833 | 69 | 23 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 1,050,139 | 1,047,013 | 84,011,120 | 83,761,040 | 134,417,792 | 134,017,664 | 33,604,448 | 33,504,416 | 723,283,237 | 721,130,203 | 19,952,641 | 19,893,247 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 1,336 | 712 | 1,002,000 | 534,000 | 1,603,200 | 854,400 | 400,800 | 213,600 | 48,430 | 25,810 | 1,336 | 712 | 
| app_proof | 50 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 |  | 108 |  | 172 |  | 43 |  | 1,124 |  | 31 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir | 1 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 1 | 58,992 | 6,544 | 44,244,000 | 4,908,000 | 35,395,200 | 3,926,400 | 8,848,800 | 981,600 | 181,769,100 | 20,163,700 | 5,014,320 | 556,240 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 1,002,864 | 45,712 | 2,286,529,920 | 104,223,360 | 1,829,223,936 | 83,378,688 | 457,305,984 | 20,844,672 | 1,054,260,780 | 48,054,740 | 29,083,056 | 1,325,648 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | 294,965 | 229,323 | 26,546,850 | 20,639,070 | 42,474,960 | 33,022,512 | 10,618,740 | 8,255,628 | 213,849,625 | 166,259,175 | 5,899,300 | 4,586,460 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | 118,020 | 13,052 | 15,932,700 | 1,762,020 | 25,492,320 | 2,819,232 | 6,373,080 | 704,808 | 81,286,275 | 8,989,565 | 2,242,380 | 247,988 | 
| app_proof | 19 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | 592,746 | 455,830 | 37,046,625 | 28,489,375 | 59,274,600 | 45,583,000 | 14,818,650 | 11,395,750 | 343,792,680 | 264,381,400 | 9,483,936 | 7,293,280 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 620 | 404 | 32,550 | 21,210 | 52,080 | 33,936 | 13,020 | 8,484 | 89,900 | 58,580 | 2,480 | 1,616 | 
| app_proof | 20 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 58,995 | 6,541 | 2,507,288 | 277,992 | 4,011,660 | 444,788 | 1,002,915 | 111,197 | 29,939,963 | 3,319,557 | 825,930 | 91,574 | 
| app_proof | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 176,981 | 85,163 | 10,618,860 | 5,109,780 | 16,990,176 | 8,175,648 | 4,247,544 | 2,043,912 | 96,233,419 | 46,307,381 | 2,654,715 | 1,277,445 | 
| app_proof | 22 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 237,813 | 24,331 | 10,701,585 | 1,094,895 | 17,122,536 | 1,751,832 | 4,280,634 | 437,958 | 103,448,655 | 10,583,985 | 2,853,756 | 291,972 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 236,892 | 25,252 | 18,951,360 | 2,020,160 | 30,322,176 | 3,232,256 | 7,580,544 | 808,064 | 120,222,690 | 12,815,390 | 3,316,488 | 353,528 | 
| app_proof | 24 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 354,892 | 169,396 | 23,067,980 | 11,010,740 | 36,908,768 | 17,617,184 | 9,227,192 | 4,404,296 | 141,513,185 | 67,546,655 | 3,903,812 | 1,863,356 | 
| app_proof | 25 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 1 | 590,865 | 457,711 | 44,314,875 | 34,328,325 | 70,903,800 | 54,925,320 | 17,725,950 | 13,731,330 | 364,120,557 | 282,064,403 | 10,044,705 | 7,781,087 | 
| app_proof | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 1 | 590,877 | 457,699 | 44,315,775 | 34,327,425 | 70,905,240 | 54,923,880 | 17,726,310 | 13,730,970 | 364,127,952 | 282,057,008 | 10,044,909 | 7,780,883 | 
| app_proof | 29 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 1 | 58,999 | 6,537 | 4,572,423 | 506,617 | 7,315,876 | 810,588 | 1,828,969 | 202,647 | 38,496,848 | 4,265,392 | 1,061,982 | 117,666 | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 750 | 274 | 123,750 | 45,210 | 99,000 | 36,168 | 24,750 | 9,042 | 108,750 | 39,730 | 3,000 | 1,096 | 
| app_proof | 33 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 1 | 40 | 24 | 3,600 | 2,160 | 5,760 | 3,456 | 1,440 | 864 | 27,550 | 16,530 | 760 | 456 | 
| app_proof | 35 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 1 | 8 |  | 700 |  | 1,120 |  | 280 |  | 5,510 |  | 152 |  | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | 3 | 1 | 353 | 117 | 564 | 188 | 141 | 47 | 1,740 | 580 | 48 | 16 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 14 | 2 | 2,240 | 320 | 3,584 | 512 | 896 | 128 | 12,180 | 1,740 | 336 | 48 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16> | 1 | 58,992 | 6,544 | 4,571,880 | 507,160 | 7,315,008 | 811,456 | 1,828,752 | 202,864 | 38,492,280 | 4,269,960 | 1,061,856 | 117,792 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 18 | 14 | 2,025 | 1,575 | 3,240 | 2,520 | 810 | 630 | 15,008 | 11,672 | 414 | 322 | 
| app_proof | 46 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16> | 1 | 590,848 | 457,728 | 47,267,840 | 36,618,240 | 75,628,544 | 58,589,184 | 18,907,136 | 14,647,296 | 406,946,560 | 315,260,160 | 11,226,112 | 8,696,832 | 
| app_proof | 47 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 1,370 | 678 | 1,027,500 | 508,500 | 1,644,000 | 813,600 | 411,000 | 203,400 | 49,663 | 24,577 | 1,370 | 678 | 
| app_proof | 50 | VariableRangeCheckerAir | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 6 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 20 | 197 | 20 | 6 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 11 | 119 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 112 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 115 | 523 | 115 | 34 | 13 | 2 | 9 | 9 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,749 | 176 | 46 | 0 | 0 | 79 | 29 | 28 | 36 | 13 | 0 | 49 | 39 | 10 | 2 | 7 | 47 | 46 | 79 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 108 | 20 | 0 | 0 | 55 | 20 | 20 | 23 | 11 | 0 | 32 | 24 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 103 | 15 | 0 | 0 | 54 | 21 | 20 | 21 | 11 | 0 | 33 | 26 | 7 | 1 | 5 | 15 | 15 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 237,929,272 | 407 | 145 | 0 | 0 | 142 | 71 | 69 | 44 | 26 | 0 | 119 | 97 | 21 | 10 | 11 | 146 | 145 | 142 | 0 | 3 | 25 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 15,094,533 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 6 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,336,444,330 | 2,232 | 596 | 0 | 0 | 1,280 | 697 | 696 | 236 | 347 | 0 | 355 | 257 | 97 | 60 | 37 | 596 | 596 | 1,280 | 0 | 1 | 345 | 0 | 0 | 
| app_proof | prover | 1 | 680,499,328 | 1,065 | 214 | 0 | 0 | 657 | 346 | 345 | 126 | 184 | 0 | 192 | 140 | 52 | 30 | 21 | 214 | 214 | 657 | 0 | 1 | 183 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 273,179,108 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 142,568,194 | 2,013,265,921 | 

| group | reason | segment | segmentation_trigger |
| --- | --- | --- | --- |
| app_proof | memory | 0 | 1 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 2 | 99 | 2,541 | 99 | 0 | 12,490,827,132 | 3,552,956,028 | 16,043,783,160 | 2,097,152 | 0 | 3 | 208 | 7,147,000 | 35.45 | 
| app_proof | 1 | 0 | 1 | 44 | 1,586 | 43 | 0 | 7,058,879,292 | 1,113,563,844 | 8,172,443,136 | 2,097,152 | 0 | 2 | 476 | 4,020,961 | 36.38 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/0f85ced8e1ae2aacf949d3e90403d2675f07f1bb

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29435532203)
