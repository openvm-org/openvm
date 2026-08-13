| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  5.20 |  3.99 |  3.99 |
| app_proof |  4.25 |  3.05 |  3.05 |
| leaf |  0.52 |  0.52 |  0.52 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,099 |  4,198 |  2,828 |  1,370 |
| `compile_metered_time_ms` |  4 |  4 |  4 |  4 |
| `execute_metered_time_ms` |  55 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  202.20 | -          |  202.20 |  202.20 |
| `set_initial_memory_time_ms` |  3.50 |  7 |  4 |  3 |
| `execute_preflight_insns` |  5,583,980.50 |  11,167,961 |  7,136,000 |  4,031,961 |
| `execute_preflight_time_ms` |  254 |  508 |  345 |  163 |
| `execute_preflight_insn_mi/s` |  21.98 | -          |  24.59 |  20.66 |
| `postflight_time_ms  ` |  96.50 |  193 |  140 |  53 |
| `postflight_memory_chronology_time_ms` |  12.50 |  25 |  21 |  4 |
| `postflight_program_index_time_ms` |  1.50 |  3 |  2 |  1 |
| `trace_gen_time_ms   ` |  41.50 |  83 |  58 |  25 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,669.50 |  3,339 |  2,236 |  1,103 |
| `prover.main_trace_commit_time_ms` |  401.50 |  803 |  582 |  221 |
| `prover.rap_constraints_time_ms` |  991.50 |  1,983 |  1,302 |  681 |
| `prover.openings_time_ms` |  275 |  550 |  350 |  200 |
| `prover.rap_constraints.logup_gkr_time_ms` |  278.50 |  557 |  359 |  198 |
| `prover.rap_constraints.round0_time_ms` |  529 |  1,058 |  706 |  352 |
| `prover.rap_constraints.mle_rounds_time_ms` |  183 |  366 |  237 |  129 |
| `prover.openings.stacked_reduction_time_ms` |  73.50 |  147 |  95 |  52 |
| `prover.openings.stacked_reduction.round0_time_ms` |  44.50 |  89 |  58 |  31 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  28.50 |  57 |  36 |  21 |
| `prover.openings.whir_time_ms` |  201 |  402 |  255 |  147 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  523 |  523 |  523 |  523 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  112 |  112 |  112 |  112 |
| `generate_blob_total_time_ms` |  12 |  12 |  12 |  12 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  411 |  411 |  411 |  411 |
| `prover.main_trace_commit_time_ms` |  147 |  147 |  147 |  147 |
| `prover.rap_constraints_time_ms` |  143 |  143 |  143 |  143 |
| `prover.openings_time_ms` |  120 |  120 |  120 |  120 |
| `prover.rap_constraints.logup_gkr_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.round0_time_ms` |  72 |  72 |  72 |  72 |
| `prover.rap_constraints.mle_rounds_time_ms` |  44 |  44 |  44 |  44 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  98 |  98 |  98 |  98 |

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
| `prover.rap_constraints.round0_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  39 |  39 |  39 |  39 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  120 |  120 |  120 |  120 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  109 |  109 |  109 |  109 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  25 |  25 |  25 |  25 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  107 |  107 |  107 |  107 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  97 |  97 |  97 |  97 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/ec7cb9c3272e9fec9aeee23432127eec0179fd48/sha2_bench-ec7cb9c3272e9fec9aeee23432127eec0179fd48.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 16.44 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 16.44 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 16.44 | app_proof.prover..0 |
| prover.stacked_commit | 14.66 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 14.10 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 14.10 | app_proof.prover..0 |
| prover.gkr_input_evals | 13.27 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 13.27 | app_proof.prover..0 |
| prover.openings | 9.89 | app_proof.prover..0 |
| prover.prove_whir_opening | 9.89 | app_proof.prover..0 |
| prover.merkle_tree | 9.89 | app_proof.prover..0 |
| prover.rs_code_matrix | 9.88 | app_proof.prover..0 |
| tracegen | 5.49 | app_proof..0 |
| postflight | 5.16 | app_proof..0 |
| prover.before_gkr_input_evals | 4.98 | app_proof.prover..0 |
| generate mem proving ctxs | 4.64 | app_proof..0 |
| set initial memory | 4.12 | app_proof..1 |
| tracegen.pow_checker | 1.10 | leaf.0 |
| tracegen.exp_bits_len | 1.10 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.10 | leaf.0 |
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
| 118 | 267,407 | 229,324 | 1 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 9 | 11 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 1 | 181 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 1 | 101 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 14 | RevealAir |  | 25 | 3 | 2 | 
| 15 | HintStoreAir | 1 | 18 | 15 | 3 | 
| 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 18 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 10 | 11 | 2 | 
| 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 12 | 4 | 3 | 
| 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 8 | 3 | 
| 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 28 | 9 | 3 | 
| 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 27 | 12 | 3 | 
| 28 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 26 | 9 | 3 | 
| 29 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 25 | 12 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 26 | 13 | 3 | 
| 31 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 25 | 9 | 3 | 
| 32 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 24 | 12 | 3 | 
| 33 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 25 | 13 | 3 | 
| 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 8 | 3 | 
| 35 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 11 | 3 | 
| 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 12 | 3 | 
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
| agg_keygen |  | 60 |  |  |  | 297 |  | 
| app_proof | 0 |  |  |  | 6 |  |  | 
| internal_for_leaf |  |  |  | 197 |  |  | 197 | 
| internal_recursive.0 |  |  |  | 120 |  |  | 121 | 
| internal_recursive.1 |  |  |  | 107 |  |  | 107 | 
| leaf |  |  | 523 |  |  |  | 524 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 5 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 0 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | RevealAir |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 24 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> |  | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> |  | 0 | 2 | 
| app_proof | Sha2MainAir<Sha512Config> |  | 0 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 0 | 1 | 
| app_proof | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 2 | 
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
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 3 | 
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
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 19 | 
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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 34 | 34 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 71 | 142 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 34 | 68 | 
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

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 16,384 | 11 | 180,224 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 7 | 14 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover |  | 0 | 131,072 | 151 | 19,791,872 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 15 | HintStoreAir | prover |  | 0 | 2 | 27 | 54 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 524,288 | 35 | 18,350,080 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 2 | 28 | 56 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 262,144 | 52 | 13,631,488 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 512 | 39 | 19,968 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 2,097,152 | 24 | 50,331,648 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 131,072 | 16 | 2,097,152 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 524,288 | 23 | 12,058,624 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 524,288 | 18 | 9,437,184 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 524,288 | 31 | 16,252,928 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 1,048,576 | 25 | 26,214,400 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 2,097,152 | 41 | 85,983,232 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 1,024 | 33 | 33,792 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 131,072 | 40 | 5,242,880 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 1 | 45 | 45 | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1 | 23 | 23 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 4 | 43 | 172 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 49 | PhantomAir | prover |  | 0 | 1 | 7 | 7 | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 256 | 300 | 76,800 | 
| app_proof | 51 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 1 | 41 | 41 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 1 | 16,384 | 11 | 180,224 | 
| app_proof | 1 | VmConnectorAir | prover |  | 1 | 2 | 7 | 14 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover |  | 1 | 65,536 | 151 | 9,895,936 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 1 | 524,288 | 35 | 18,350,080 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 1 | 131,072 | 52 | 6,815,744 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 1 | 512 | 39 | 19,968 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 24 | 25,165,824 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 1 | 65,536 | 16 | 1,048,576 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 1 | 262,144 | 23 | 6,029,312 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 1 | 262,144 | 31 | 8,126,464 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 1 | 524,288 | 25 | 13,107,200 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 1 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 1 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 1 | 512 | 33 | 16,896 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 1 | 65,536 | 40 | 2,621,440 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 1 | 64 | 30 | 1,920 | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 1 | 8 | 31 | 248 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 1 | 4 | 45 | 180 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover |  | 1 | 16 | 59 | 944 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 1 | 65,536 | 29 | 1,900,544 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 1 | 32 | 43 | 1,376 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> | prover |  | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 1 | 512 | 300 | 153,600 | 
| app_proof | 51 | VariableRangeCheckerAir | prover |  | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 1 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 9,244 | 7,140 | 254,210 | 196,350 | 406,736 | 314,160 | 101,684 | 78,540 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> |  | 0 | 104,688 | 26,384 | 79,039,440 | 19,919,920 | 63,231,552 | 15,935,936 | 15,807,888 | 3,983,984 | 383,288,940 | 96,598,420 | 10,573,488 | 2,664,784 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 1,779,696 | 317,456 | 4,057,706,880 | 723,799,680 | 3,246,165,504 | 579,039,744 | 811,541,376 | 144,759,936 | 1,870,905,420 | 333,725,620 | 51,611,184 | 9,206,224 | 
| app_proof | 15 | HintStoreAir |  | 0 | 2 |  | 270 |  | 216 |  | 54 |  | 1,305 |  | 36 |  | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 523,449 | 839 | 45,801,788 | 73,412 | 73,282,860 | 117,460 | 18,320,715 | 29,365 | 379,500,525 | 608,275 | 10,468,980 | 16,780 | 
| app_proof | 17 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 2 |  | 140 |  | 224 |  | 56 |  | 1,015 |  | 28 |  | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 209,382 | 52,762 | 27,219,660 | 6,859,060 | 43,551,456 | 10,974,496 | 10,887,864 | 2,743,624 | 144,211,853 | 36,339,827 | 3,978,258 | 1,002,478 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 295 | 217 | 28,763 | 21,157 | 46,020 | 33,852 | 11,505 | 8,463 | 106,938 | 78,662 | 2,950 | 2,170 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 1,051,945 | 1,045,207 | 63,116,700 | 62,712,420 | 100,986,720 | 100,339,872 | 25,246,680 | 25,084,968 | 610,128,100 | 606,220,060 | 16,831,120 | 16,723,312 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 104,702 | 26,370 | 4,188,080 | 1,054,800 | 6,700,928 | 1,687,680 | 1,675,232 | 421,920 | 45,545,370 | 11,470,950 | 1,256,424 | 316,440 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 314,071 | 210,217 | 18,059,083 | 12,087,477 | 28,894,532 | 19,339,964 | 7,223,633 | 4,834,991 | 170,776,107 | 114,305,493 | 4,711,065 | 3,153,255 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 422,031 | 102,257 | 18,991,395 | 4,601,565 | 30,386,232 | 7,362,504 | 7,596,558 | 1,840,626 | 183,583,485 | 44,481,795 | 5,064,372 | 1,227,084 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 420,397 | 103,891 | 32,580,768 | 8,051,552 | 52,129,228 | 12,882,484 | 13,032,307 | 3,220,621 | 213,351,478 | 52,724,682 | 5,885,558 | 1,454,474 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 629,837 | 418,739 | 39,364,813 | 26,171,187 | 62,983,700 | 41,873,900 | 15,745,925 | 10,468,475 | 251,147,504 | 166,972,176 | 6,928,207 | 4,606,129 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 1,049,059 | 1,048,093 | 107,528,548 | 107,429,532 | 172,045,676 | 171,887,252 | 43,011,419 | 42,971,813 | 1,064,794,885 | 1,063,814,395 | 29,373,652 | 29,346,604 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 1,048,526 | 50 | 107,473,915 | 5,125 | 171,958,264 | 8,200 | 42,989,566 | 2,050 | 1,026,244,823 | 48,937 | 28,310,202 | 1,350 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 758 | 266 | 125,070 | 43,890 | 100,056 | 35,112 | 25,014 | 8,778 | 109,910 | 38,570 | 3,032 | 1,064 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 104,688 | 26,384 | 10,468,800 | 2,638,400 | 16,750,080 | 4,221,440 | 4,187,520 | 1,055,360 | 98,668,440 | 24,866,920 | 2,721,888 | 685,984 | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 1 |  | 113 |  | 180 |  | 45 |  | 580 |  | 16 |  | 
| app_proof | 39 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 1 |  | 58 |  | 92 |  | 23 |  | 508 |  | 14 |  | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 104,689 | 26,383 | 7,589,953 | 1,912,767 | 12,143,924 | 3,060,428 | 3,035,981 | 765,107 | 64,514,597 | 16,258,523 | 1,779,713 | 448,511 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 3 | 1 | 323 | 107 | 516 | 172 | 129 | 43 | 2,502 | 833 | 69 | 23 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 1,048,525 | 51 | 78,639,375 | 3,825 | 125,823,000 | 6,120 | 31,455,750 | 1,530 | 722,171,594 | 35,126 | 19,921,975 | 969 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 49 | PhantomAir |  | 0 | 1 |  | 18 |  | 28 |  | 7 |  | 109 |  | 3 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 759 | 265 | 569,250 | 198,750 | 910,800 | 318,000 | 227,700 | 79,500 | 27,514 | 9,606 | 759 | 265 | 
| app_proof | 51 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 1 |  | 103 |  | 164 |  | 41 |  | 1,124 |  | 31 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 1 | 9,244 | 7,140 | 254,210 | 196,350 | 406,736 | 314,160 | 101,684 | 78,540 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir |  | 1 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> |  | 1 | 59,153 | 6,383 | 44,660,515 | 4,819,165 | 35,728,412 | 3,855,332 | 8,932,103 | 963,833 | 216,573,922 | 23,369,758 | 5,974,453 | 644,683 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> |  | 1 | 1,005,601 | 42,975 | 2,292,770,280 | 97,983,000 | 1,834,216,224 | 78,386,400 | 458,554,056 | 19,596,600 | 1,057,138,052 | 45,177,468 | 29,162,429 | 1,246,275 | 
| app_proof | 16 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 295,770 | 228,518 | 25,879,875 | 19,995,325 | 41,407,800 | 31,992,520 | 10,351,950 | 7,998,130 | 214,433,250 | 165,675,550 | 5,915,400 | 4,570,360 | 
| app_proof | 19 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 118,342 | 12,730 | 15,384,460 | 1,654,900 | 24,615,136 | 2,647,840 | 6,153,784 | 661,960 | 81,508,053 | 8,767,787 | 2,248,498 | 241,870 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 1 | 310 | 202 | 30,225 | 19,695 | 48,360 | 31,512 | 12,090 | 7,878 | 112,375 | 73,225 | 3,100 | 2,020 | 
| app_proof | 20 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 594,368 | 454,208 | 35,662,080 | 27,252,480 | 57,059,328 | 43,603,968 | 14,264,832 | 10,900,992 | 344,733,440 | 263,440,640 | 9,509,888 | 7,267,328 | 
| app_proof | 21 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 1 | 59,156 | 6,380 | 2,366,240 | 255,200 | 3,785,984 | 408,320 | 946,496 | 102,080 | 25,732,860 | 2,775,300 | 709,872 | 76,560 | 
| app_proof | 22 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 1 | 177,465 | 84,679 | 10,204,238 | 4,869,042 | 16,326,780 | 7,790,468 | 4,081,695 | 1,947,617 | 96,496,594 | 46,044,206 | 2,661,975 | 1,270,185 | 
| app_proof | 23 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 1 | 238,465 | 23,679 | 10,730,925 | 1,065,555 | 17,169,480 | 1,704,888 | 4,292,370 | 426,222 | 103,732,275 | 10,300,365 | 2,861,580 | 284,148 | 
| app_proof | 24 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 237,539 | 24,605 | 18,409,273 | 1,906,887 | 29,454,836 | 3,051,020 | 7,363,709 | 762,755 | 120,551,043 | 12,487,037 | 3,325,546 | 344,470 | 
| app_proof | 25 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 355,861 | 168,427 | 22,241,313 | 10,526,687 | 35,586,100 | 16,842,700 | 8,896,525 | 4,210,675 | 141,899,574 | 67,160,266 | 3,914,471 | 1,852,697 | 
| app_proof | 26 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 592,484 | 456,092 | 60,729,610 | 46,749,430 | 97,167,376 | 74,799,088 | 24,291,844 | 18,699,772 | 601,371,260 | 462,933,380 | 16,589,552 | 12,770,576 | 
| app_proof | 27 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 592,498 | 456,078 | 60,731,045 | 46,747,995 | 97,169,672 | 74,796,792 | 24,292,418 | 18,699,198 | 579,907,418 | 446,386,342 | 15,997,446 | 12,314,106 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 1 | 762 | 262 | 125,730 | 43,230 | 100,584 | 34,584 | 25,146 | 8,646 | 110,490 | 37,990 | 3,048 | 1,048 | 
| app_proof | 30 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 59,161 | 6,375 | 5,916,100 | 637,500 | 9,465,760 | 1,020,000 | 2,366,440 | 255,000 | 55,759,243 | 6,008,437 | 1,538,186 | 165,750 | 
| app_proof | 34 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 40 | 24 | 3,000 | 1,800 | 4,800 | 2,880 | 1,200 | 720 | 27,550 | 16,530 | 760 | 456 | 
| app_proof | 36 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 1 | 8 |  | 620 |  | 992 |  | 248 |  | 5,510 |  | 152 |  | 
| app_proof | 38 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 1 | 3 | 1 | 338 | 112 | 540 | 180 | 135 | 45 | 1,740 | 580 | 48 | 16 | 
| app_proof | 43 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 1 | 14 | 2 | 2,065 | 295 | 3,304 | 472 | 826 | 118 | 12,180 | 1,740 | 336 | 48 | 
| app_proof | 45 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 59,153 | 6,383 | 4,288,593 | 462,767 | 6,861,748 | 740,428 | 1,715,437 | 185,107 | 36,453,037 | 3,933,523 | 1,005,601 | 108,511 | 
| app_proof | 46 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 18 | 14 | 1,935 | 1,505 | 3,096 | 2,408 | 774 | 602 | 15,008 | 11,672 | 414 | 322 | 
| app_proof | 47 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 592,462 | 456,114 | 44,434,650 | 34,208,550 | 71,095,440 | 54,733,680 | 17,773,860 | 13,683,420 | 408,058,203 | 314,148,517 | 11,256,778 | 8,666,166 | 
| app_proof | 48 | BitwiseOperationLookupAir<8> |  | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 50 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 1,354 | 694 | 1,015,500 | 520,500 | 1,624,800 | 832,800 | 406,200 | 208,200 | 49,083 | 25,157 | 1,354 | 694 | 
| app_proof | 51 | VariableRangeCheckerAir |  | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 4 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 20 | 197 | 20 | 6 | 1 | 2 | 3 | 3 | 
| internal_recursive.0 | 1 | 11 | 120 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 112 | 523 | 112 | 32 | 12 | 2 | 10 | 10 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,751 | 176 | 46 | 0 | 0 | 79 | 28 | 27 | 37 | 13 | 0 | 49 | 39 | 10 | 2 | 7 | 47 | 46 | 79 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,771 | 109 | 20 | 0 | 0 | 55 | 20 | 20 | 23 | 11 | 0 | 32 | 25 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,867 | 97 | 15 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 28 | 21 | 7 | 1 | 5 | 15 | 15 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 237,929,276 | 411 | 146 | 0 | 0 | 143 | 72 | 71 | 44 | 26 | 0 | 120 | 98 | 22 | 10 | 11 | 147 | 146 | 143 | 0 | 3 | 25 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,829 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,385 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,361 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 15,094,537 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 7 | 0 | 7 | 7 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,299,627,932 | 2,236 | 582 | 0 | 0 | 1,302 | 706 | 705 | 237 | 359 | 1 | 350 | 255 | 95 | 58 | 36 | 582 | 582 | 1,302 | 0 | 1 | 357 | 0 | 0 | 
| app_proof | prover |  | 1 | 699,120,202 | 1,103 | 221 | 0 | 0 | 681 | 352 | 352 | 129 | 198 | 0 | 200 | 147 | 52 | 31 | 21 | 221 | 221 | 681 | 0 | 1 | 197 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 271,738,350 | 2,013,265,921 | 
| app_proof | prover |  | 1 | 0 | 165,963,786 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 1,370 | 55 | 11,167,961 | 202.20 | 0 | 4,262 | 

| group | program | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- |
| app_proof |  | memory | 0 | 1 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 4 | 2 | 58 | 2,828 | 3 | 4 | 0 | 140 | 2 | 21 | 0 | 12,689,983,656 | 2,908,316,328 | 15,598,299,984 | 2,097,152 | 2 | 0 | 0 | 3 | 54 | 345 | 7,136,000 | 20.66 | 0 | 0 | 
| app_proof |  | 1 | 3 | 2 | 25 | 1,370 | 3 | 3 | 0 | 53 | 1 | 4 | 0 | 7,202,397,456 | 1,614,408,451 | 8,816,805,907 | 2,097,152 | 2 | 0 | 0 | 2 | 22 | 163 | 4,031,961 | 24.59 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 7 | 0 | 7 | 7 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/ec7cb9c3272e9fec9aeee23432127eec0179fd48

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31752739045)
