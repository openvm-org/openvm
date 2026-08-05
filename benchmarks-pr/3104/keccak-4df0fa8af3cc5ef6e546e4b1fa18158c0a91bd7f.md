| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  10.76 |  3.49 |  3.49 |
| app_proof |  8.68 |  1.91 |  1.91 |
| leaf |  1.51 |  1.01 |  1.01 |
| internal_for_leaf |  0.33 |  0.33 |  0.33 |
| internal_recursive.0 |  0.13 |  0.13 |  0.13 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,425.83 |  8,555 |  1,468 |  1,383 |
| `compile_metered_time_ms` |  2 |  2 |  2 |  2 |
| `execute_metered_time_ms` |  128 | -          | -          | -          |
| `execute_metered_insns` |  14,365,133 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  111.44 | -          |  111.44 |  111.44 |
| `set_initial_memory_time_ms` |  7.17 |  43 |  38 |  1 |
| `execute_preflight_insns` |  2,394,188.83 |  14,365,133 |  2,413,000 |  2,300,133 |
| `execute_preflight_time_ms` |  61.50 |  369 |  76 |  57 |
| `execute_preflight_insn_mi/s` |  38.93 | -          |  42.31 |  31.69 |
| `postflight_time_ms  ` |  223.83 |  1,343 |  232 |  200 |
| `postflight_memory_chronology_time_ms` |  2.67 |  16 |  6 |  2 |
| `postflight_program_index_time_ms` |  0.17 |  1 |  1 |  0 |
| `trace_gen_time_ms   ` |  12.83 |  77 |  18 |  11 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,107.50 |  6,645 |  1,157 |  1,072 |
| `prover.main_trace_commit_time_ms` |  341.50 |  2,049 |  358 |  334 |
| `prover.rap_constraints_time_ms` |  513.83 |  3,083 |  516 |  511 |
| `prover.openings_time_ms` |  250.83 |  1,505 |  302 |  225 |
| `prover.rap_constraints.logup_gkr_time_ms` |  104.83 |  629 |  106 |  103 |
| `prover.rap_constraints.round0_time_ms` |  283.83 |  1,703 |  285 |  283 |
| `prover.rap_constraints.mle_rounds_time_ms` |  124 |  744 |  124 |  124 |
| `prover.openings.stacked_reduction_time_ms` |  61.17 |  367 |  62 |  61 |
| `prover.openings.stacked_reduction.round0_time_ms` |  36.33 |  218 |  37 |  36 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  144 |  24 |  24 |
| `prover.openings.whir_time_ms` |  189.33 |  1,136 |  240 |  164 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  754 |  1,508 |  1,010 |  498 |
| `execute_preflight_time_ms` |  4.50 |  9 |  5 |  4 |
| `trace_gen_time_ms   ` |  136 |  272 |  182 |  90 |
| `generate_blob_total_time_ms` |  15.50 |  31 |  21 |  10 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  617.50 |  1,235 |  827 |  408 |
| `prover.main_trace_commit_time_ms` |  266 |  532 |  385 |  147 |
| `prover.rap_constraints_time_ms` |  187 |  374 |  232 |  142 |
| `prover.openings_time_ms` |  163 |  326 |  209 |  117 |
| `prover.rap_constraints.logup_gkr_time_ms` |  33.50 |  67 |  41 |  26 |
| `prover.rap_constraints.round0_time_ms` |  95.50 |  191 |  120 |  71 |
| `prover.rap_constraints.mle_rounds_time_ms` |  57.50 |  115 |  70 |  45 |
| `prover.openings.stacked_reduction_time_ms` |  29.50 |  59 |  37 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  15 |  30 |  20 |  10 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  14 |  28 |  17 |  11 |
| `prover.openings.whir_time_ms` |  133.50 |  267 |  172 |  95 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  329 |  329 |  329 |  329 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  39 |  39 |  39 |  39 |
| `generate_blob_total_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  289 |  289 |  289 |  289 |
| `prover.main_trace_commit_time_ms` |  93 |  93 |  93 |  93 |
| `prover.rap_constraints_time_ms` |  113 |  113 |  113 |  113 |
| `prover.openings_time_ms` |  82 |  82 |  82 |  82 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  39 |  39 |  39 |  39 |
| `prover.rap_constraints.mle_rounds_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings.stacked_reduction_time_ms` |  15 |  15 |  15 |  15 |
| `prover.openings.stacked_reduction.round0_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.whir_time_ms` |  66 |  66 |  66 |  66 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  131 |  131 |  131 |  131 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  16 |  16 |  16 |  16 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  115 |  115 |  115 |  115 |
| `prover.main_trace_commit_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints_time_ms` |  59 |  59 |  59 |  59 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  25 |  25 |  25 |  25 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  108 |  108 |  108 |  108 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
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
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f/keccak-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 9.47 | app_proof.prover..4 |
| prover.prove_whir_opening | 7.36 | leaf.0.prover |
| prover.openings | 7.36 | leaf.0.prover |
| prover.merkle_tree | 7.36 | leaf.0.prover |
| prover.rs_code_matrix | 7.36 | leaf.0.prover |
| prover.rap_constraints | 6.98 | app_proof.prover..4 |
| postflight | 6.47 | app_proof..0 |
| tracegen | 6.30 | app_proof..0 |
| generate mem proving ctxs | 6.30 | app_proof..0 |
| frac_sumcheck.gkr_rounds | 6.24 | app_proof.prover..4 |
| prover.batch_constraints.before_round0 | 6.24 | app_proof.prover..4 |
| frac_sumcheck.segment_tree | 6.20 | app_proof.prover..4 |
| prover.gkr_input_evals | 6.20 | app_proof.prover..4 |
| set initial memory | 6.12 | app_proof..5 |
| prover.batch_constraints.round0 | 5.87 | app_proof.prover..4 |
| prover.batch_constraints.fold_ple_evals | 5.87 | app_proof.prover..4 |
| prover.before_gkr_input_evals | 3.26 | app_proof.prover..4 |
| tracegen.pow_checker | 2.06 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 2.06 | leaf.0 |
| tracegen.exp_bits_len | 2.06 | leaf.0 |
| tracegen.whir_folding | 1.81 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.80 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.80 | leaf.0 |
| tracegen.proof_shape | 1.45 | leaf.0 |
| tracegen.public_values | 1.45 | leaf.0 |
| tracegen.range_checker | 1.45 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 125 | 267,335 | 227,944 | 217 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | KeccakfOpAir |  | 135 | 27 | 3 | 
| 11 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 12 | XorinVmAir |  | 408 | 87 | 3 | 
| 13 | Rv64HintStoreAir | 1 | 18 | 15 | 3 | 
| 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 4 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 10 | 11 | 2 | 
| 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 8 | 3 | 
| 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 10 | 2 | 
| 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 28 | 10 | 3 | 
| 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 27 | 12 | 3 | 
| 26 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 26 | 10 | 3 | 
| 27 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 25 | 12 | 3 | 
| 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 26 | 13 | 3 | 
| 29 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 25 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 24 | 12 | 3 | 
| 31 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 25 | 13 | 3 | 
| 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 9 | 3 | 
| 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 11 | 3 | 
| 34 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 12 | 3 | 
| 35 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 38 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 39 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 40 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 46 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 47 | PhantomAir |  | 3 | 1 | 2 | 
| 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 49 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 61 |  |  |  | 296 |  | 
| app_proof | 0 |  |  |  | 2 |  |  | 
| internal_for_leaf |  |  |  | 329 |  |  | 329 | 
| internal_recursive.0 |  |  |  | 131 |  |  | 131 | 
| internal_recursive.1 |  |  |  | 108 |  |  | 108 | 
| leaf |  |  | 498 |  |  |  | 1,509 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 0 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 5 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | Rv64HintStoreAir |  | 0 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 3 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 0 | 
| app_proof | XorinVmAir |  | 0 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 1 | 0 | 
| app_proof | KeccakfOpAir |  | 1 | 0 | 
| app_proof | KeccakfPermAir |  | 1 | 0 | 
| app_proof | PhantomAir |  | 1 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 1 | 0 | 
| app_proof | Rv64HintStoreAir |  | 1 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 0 | 
| app_proof | XorinVmAir |  | 1 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 2 | 0 | 
| app_proof | KeccakfOpAir |  | 2 | 0 | 
| app_proof | KeccakfPermAir |  | 2 | 0 | 
| app_proof | PhantomAir |  | 2 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 2 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 2 | 0 | 
| app_proof | Rv64HintStoreAir |  | 2 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 2 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 2 | 0 | 
| app_proof | XorinVmAir |  | 2 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 3 | 0 | 
| app_proof | KeccakfOpAir |  | 3 | 0 | 
| app_proof | KeccakfPermAir |  | 3 | 0 | 
| app_proof | PhantomAir |  | 3 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 3 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 3 | 0 | 
| app_proof | Rv64HintStoreAir |  | 3 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 3 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 3 | 0 | 
| app_proof | XorinVmAir |  | 3 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 4 | 0 | 
| app_proof | KeccakfOpAir |  | 4 | 0 | 
| app_proof | KeccakfPermAir |  | 4 | 0 | 
| app_proof | PhantomAir |  | 4 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 4 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 4 | 0 | 
| app_proof | Rv64HintStoreAir |  | 4 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 4 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 4 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 4 | 0 | 
| app_proof | XorinVmAir |  | 4 | 0 | 
| app_proof | BitwiseOperationLookupAir<8> |  | 5 | 0 | 
| app_proof | KeccakfOpAir |  | 5 | 0 | 
| app_proof | KeccakfPermAir |  | 5 | 0 | 
| app_proof | PhantomAir |  | 5 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 5 | 6 | 
| app_proof | RangeTupleCheckerAir<2> |  | 5 | 0 | 
| app_proof | Rv64HintStoreAir |  | 5 | 0 | 
| app_proof | VariableRangeCheckerAir |  | 5 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 5 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 5 | 0 | 
| app_proof | XorinVmAir |  | 5 | 0 | 

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
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 71 | 142 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 64 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 71 | 284 | 
| leaf | 0 | VerifierPvsAir | 1 | prover | 2 | 71 | 142 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 1 | VmPvsAir | 1 | prover | 2 | 32 | 64 | 
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
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 65,536 | 60 | 3,932,160 | 
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
| app_proof | 0 | ProgramAir | prover |  | 0 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 0 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 0 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 0 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 0 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 40 | 20,971,520 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 1 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 1 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 1 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 1 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 1 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 1 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 1 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 1 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 1 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 1 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 1 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 1 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 1 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 1 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 1 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 1 | 524,288 | 40 | 20,971,520 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 1 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 1 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 1 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 1 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 1 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 1 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 1 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 1 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 1 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 1 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 2 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 2 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 2 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 2 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 2 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 2 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 2 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 2 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 2 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 2 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 2 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 2 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 2 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 2 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 2 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 2 | 524,288 | 40 | 20,971,520 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 2 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 2 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 2 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 2 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 2 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 2 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 2 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 2 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 2 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 2 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 3 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 3 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 3 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 3 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 3 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 3 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 3 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 3 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 3 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 3 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 3 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 3 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 3 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 3 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 3 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 3 | 524,288 | 40 | 20,971,520 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 3 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 3 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 3 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 3 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 3 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 3 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 3 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 3 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 3 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 3 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 3 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 4 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 4 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 4 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 4 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 4 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 4 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 4 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 4 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 4 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 4 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 4 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 4 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 4 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 4 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 4 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 4 | 524,288 | 40 | 20,971,520 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 4 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 4 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 4 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 4 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 4 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 4 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 4 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 4 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 4 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 4 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 4 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 4 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 4 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover |  | 5 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover |  | 5 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover |  | 5 | 16,384 | 284 | 4,653,056 | 
| app_proof | 11 | KeccakfPermAir | prover |  | 5 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 12 | XorinVmAir | prover |  | 5 | 16,384 | 596 | 9,764,864 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 5 | 262,144 | 34 | 8,912,896 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 5 | 65,536 | 51 | 3,342,336 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 5 | 1,048,576 | 23 | 24,117,248 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 5 | 65,536 | 16 | 1,048,576 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 5 | 64 | 39 | 2,496 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 5 | 131,072 | 22 | 2,883,584 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 5 | 65,536 | 17 | 1,114,112 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 5 | 65,536 | 30 | 1,966,080 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 5 | 262,144 | 24 | 6,291,456 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 5 | 1,048,576 | 41 | 42,991,616 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 5 | 262,144 | 40 | 10,485,760 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 5 | 16,384 | 39 | 638,976 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 5 | 256 | 33 | 8,448 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 5 | 65,536 | 30 | 1,966,080 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 5 | 32,768 | 29 | 950,272 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 5 | 16,384 | 22 | 360,448 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 5 | 16,384 | 28 | 458,752 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 5 | 32,768 | 42 | 1,376,256 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 5 | 131,072 | 29 | 3,801,088 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover |  | 5 | 65,536 | 18 | 1,179,648 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 5 | 256 | 300 | 76,800 | 
| app_proof | 49 | VariableRangeCheckerAir | prover |  | 5 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 5 | 16,384 | 40 | 655,360 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover |  | 5 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 0 | 10,919 | 5,465 | 7,752,490 | 3,880,150 | 12,403,984 | 6,208,240 | 3,100,996 | 1,552,060 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 0 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 0 | 10,918 | 5,466 | 16,267,820 | 8,144,340 | 26,028,512 | 13,030,944 | 6,507,128 | 3,257,736 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 152,859 | 109,285 | 12,993,015 | 9,289,225 | 20,788,824 | 14,862,760 | 5,197,206 | 3,715,690 | 110,822,775 | 79,231,625 | 3,057,180 | 2,185,700 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 54,593 | 10,943 | 6,960,608 | 1,395,232 | 11,136,972 | 2,232,372 | 2,784,243 | 558,093 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 567,762 | 480,814 | 32,646,315 | 27,646,805 | 52,234,104 | 44,234,888 | 13,058,526 | 11,058,722 | 329,301,960 | 278,872,120 | 9,084,192 | 7,693,024 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 0 | 54,596 | 10,940 | 2,183,840 | 437,600 | 3,494,144 | 700,160 | 873,536 | 175,040 | 27,707,470 | 5,552,050 | 764,344 | 153,160 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 46 | 18 | 4,485 | 1,755 | 7,176 | 2,808 | 1,794 | 702 | 16,675 | 6,525 | 460 | 180 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 0 | 98,266 | 32,806 | 5,404,630 | 1,804,330 | 8,647,408 | 2,886,928 | 2,161,852 | 721,732 | 53,432,138 | 17,838,262 | 1,473,990 | 492,090 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 0 | 65,510 | 26 | 2,784,175 | 1,105 | 4,454,680 | 1,768 | 1,113,670 | 442 | 28,496,850 | 11,310 | 786,120 | 312 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 65,511 | 25 | 4,913,325 | 1,875 | 7,861,320 | 3,000 | 1,965,330 | 750 | 33,246,833 | 12,687 | 917,154 | 350 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 218,369 | 43,775 | 13,102,140 | 2,626,500 | 20,963,424 | 4,202,400 | 5,240,856 | 1,050,600 | 87,074,639 | 17,455,281 | 2,402,059 | 481,525 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 589,612 | 458,964 | 60,435,230 | 47,043,810 | 96,696,368 | 75,270,096 | 24,174,092 | 18,817,524 | 598,456,180 | 465,848,460 | 16,509,136 | 12,850,992 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 272,960 | 251,328 | 27,296,000 | 25,132,800 | 43,673,600 | 40,212,480 | 10,918,400 | 10,053,120 | 267,159,600 | 245,987,280 | 7,369,920 | 6,785,856 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 10,918 | 5,466 | 1,064,505 | 532,935 | 1,703,208 | 852,696 | 425,802 | 213,174 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 222 | 34 | 36,630 | 5,610 | 29,304 | 4,488 | 7,326 | 1,122 | 32,190 | 4,930 | 888 | 136 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 43,676 | 21,860 | 3,275,700 | 1,639,500 | 5,241,120 | 2,623,200 | 1,310,280 | 655,800 | 30,081,845 | 15,056,075 | 829,844 | 415,340 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 32,757 | 11 | 2,374,883 | 797 | 3,799,812 | 1,276 | 949,953 | 319 | 21,373,943 | 7,177 | 589,626 | 198 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 10,919 | 5,465 | 600,545 | 300,575 | 960,872 | 480,920 | 240,218 | 120,230 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 10,918 | 5,466 | 764,260 | 382,620 | 1,222,816 | 612,192 | 305,704 | 153,048 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 21,837 | 10,931 | 2,292,885 | 1,147,755 | 3,668,616 | 1,836,408 | 917,154 | 459,102 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 109,181 | 21,891 | 7,915,623 | 1,587,097 | 12,664,996 | 2,539,356 | 3,166,249 | 634,839 | 75,198,414 | 15,077,426 | 2,074,439 | 415,929 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 221 | 35 | 165,750 | 26,250 | 265,200 | 42,000 | 66,300 | 10,500 | 8,012 | 1,268 | 221 | 35 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 10,919 | 5,465 | 1,091,900 | 546,500 | 1,747,040 | 874,400 | 436,760 | 218,600 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 1 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 1 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 1 | 10,919 | 5,465 | 7,752,490 | 3,880,150 | 12,403,984 | 6,208,240 | 3,100,996 | 1,552,060 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 1 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 1 | 10,919 | 5,465 | 16,269,310 | 8,142,850 | 26,030,896 | 13,028,560 | 6,507,724 | 3,257,140 | 161,492,010 | 80,827,350 | 4,454,952 | 2,229,720 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 1 | 152,860 | 109,284 | 12,993,100 | 9,289,140 | 20,788,960 | 14,862,624 | 5,197,240 | 3,715,656 | 110,823,500 | 79,230,900 | 3,057,200 | 2,185,680 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 1 | 54,592 | 10,944 | 6,960,480 | 1,395,360 | 11,136,768 | 2,232,576 | 2,784,192 | 558,144 | 37,600,240 | 7,537,680 | 1,037,248 | 207,936 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 1 | 567,770 | 480,806 | 32,646,775 | 27,646,345 | 52,234,840 | 44,234,152 | 13,058,710 | 11,058,538 | 329,306,600 | 278,867,480 | 9,084,320 | 7,692,896 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 1 | 54,594 | 10,942 | 2,183,760 | 437,680 | 3,494,016 | 700,288 | 873,504 | 175,072 | 27,706,455 | 5,553,065 | 764,316 | 153,188 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 1 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 1 | 98,267 | 32,805 | 5,404,685 | 1,804,275 | 8,647,496 | 2,886,840 | 2,161,874 | 721,710 | 53,432,682 | 17,837,718 | 1,474,005 | 492,075 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 1 | 65,512 | 24 | 2,784,260 | 1,020 | 4,454,816 | 1,632 | 1,113,704 | 408 | 28,497,720 | 10,440 | 786,144 | 288 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 1 | 65,512 | 24 | 4,913,400 | 1,800 | 7,861,440 | 2,880 | 1,965,360 | 720 | 33,247,340 | 12,180 | 917,168 | 336 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 1 | 218,370 | 43,774 | 13,102,200 | 2,626,440 | 20,963,520 | 4,202,304 | 5,240,880 | 1,050,576 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 1 | 589,592 | 458,984 | 60,433,180 | 47,045,860 | 96,693,088 | 75,273,376 | 24,173,272 | 18,818,344 | 598,435,880 | 465,868,760 | 16,508,576 | 12,851,552 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 1 | 272,962 | 251,326 | 27,296,200 | 25,132,600 | 43,673,920 | 40,212,160 | 10,918,480 | 10,053,040 | 267,161,558 | 245,985,322 | 7,369,974 | 6,785,802 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 1 | 10,919 | 5,465 | 1,064,603 | 532,837 | 1,703,364 | 852,540 | 425,841 | 213,135 | 10,291,158 | 5,150,762 | 283,894 | 142,090 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 1 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 1 | 43,675 | 21,861 | 3,275,625 | 1,639,575 | 5,241,000 | 2,623,320 | 1,310,250 | 655,830 | 30,081,157 | 15,056,763 | 829,825 | 415,359 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 1 | 32,757 | 11 | 2,374,883 | 797 | 3,799,812 | 1,276 | 949,953 | 319 | 21,373,943 | 7,177 | 589,626 | 198 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 1 | 10,918 | 5,466 | 600,490 | 300,630 | 960,784 | 481,008 | 240,196 | 120,252 | 5,540,885 | 2,773,995 | 152,852 | 76,524 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 1 | 10,919 | 5,465 | 764,330 | 382,550 | 1,222,928 | 612,080 | 305,732 | 153,020 | 6,728,834 | 3,367,806 | 185,623 | 92,905 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 1 | 21,837 | 10,931 | 2,292,885 | 1,147,755 | 3,668,616 | 1,836,408 | 917,154 | 459,102 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 1 | 109,188 | 21,884 | 7,916,130 | 1,586,590 | 12,665,808 | 2,538,544 | 3,166,452 | 634,636 | 75,203,235 | 15,072,605 | 2,074,572 | 415,796 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 1 | 10,918 | 5,466 | 1,091,800 | 546,600 | 1,746,880 | 874,560 | 436,720 | 218,640 | 12,269,103 | 6,142,417 | 338,458 | 169,446 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 2 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 2 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 2 | 10,918 | 5,466 | 7,751,780 | 3,880,860 | 12,402,848 | 6,209,376 | 3,100,712 | 1,552,344 | 53,429,963 | 26,749,237 | 1,473,930 | 737,910 | 
| app_proof | 11 | KeccakfPermAir |  | 2 | 262,032 | 112 | 3,450,961,440 | 1,475,040 | 2,760,769,152 | 1,180,032 | 690,192,288 | 295,008 | 18,997,320 | 8,120 | 524,064 | 224 | 
| app_proof | 12 | XorinVmAir |  | 2 | 10,918 | 5,466 | 16,267,820 | 8,144,340 | 26,028,512 | 13,030,944 | 6,507,128 | 3,257,736 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 2 | 152,859 | 109,285 | 12,993,015 | 9,289,225 | 20,788,824 | 14,862,760 | 5,197,206 | 3,715,690 | 110,822,775 | 79,231,625 | 3,057,180 | 2,185,700 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 2 | 54,593 | 10,943 | 6,960,608 | 1,395,232 | 11,136,972 | 2,232,372 | 2,784,243 | 558,093 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 2 | 567,761 | 480,815 | 32,646,258 | 27,646,862 | 52,234,012 | 44,234,980 | 13,058,503 | 11,058,745 | 329,301,380 | 278,872,700 | 9,084,176 | 7,693,040 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 2 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 27,705,948 | 5,553,572 | 764,302 | 153,202 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 2 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 2 | 98,267 | 32,805 | 5,404,685 | 1,804,275 | 8,647,496 | 2,886,840 | 2,161,874 | 721,710 | 53,432,682 | 17,837,718 | 1,474,005 | 492,075 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 2 | 65,510 | 26 | 2,784,175 | 1,105 | 4,454,680 | 1,768 | 1,113,670 | 442 | 28,496,850 | 11,310 | 786,120 | 312 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 2 | 65,510 | 26 | 4,913,250 | 1,950 | 7,861,200 | 3,120 | 1,965,300 | 780 | 33,246,325 | 13,195 | 917,140 | 364 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 2 | 218,370 | 43,774 | 13,102,200 | 2,626,440 | 20,963,520 | 4,202,304 | 5,240,880 | 1,050,576 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 2 | 589,616 | 458,960 | 60,435,640 | 47,043,400 | 96,697,024 | 75,269,440 | 24,174,256 | 18,817,360 | 598,460,240 | 465,844,400 | 16,509,248 | 12,850,880 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 2 | 272,963 | 251,325 | 27,296,300 | 25,132,500 | 43,674,080 | 40,212,000 | 10,918,520 | 10,053,000 | 267,162,537 | 245,984,343 | 7,370,001 | 6,785,775 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 2 | 10,918 | 5,466 | 1,064,505 | 532,935 | 1,703,208 | 852,696 | 425,802 | 213,174 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 2 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 2 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 2 | 32,755 | 13 | 2,374,738 | 942 | 3,799,580 | 1,508 | 949,895 | 377 | 21,372,638 | 8,482 | 589,590 | 234 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 2 | 10,919 | 5,465 | 600,545 | 300,575 | 960,872 | 480,920 | 240,218 | 120,230 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 2 | 10,918 | 5,466 | 764,260 | 382,620 | 1,222,816 | 612,192 | 305,704 | 153,048 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 2 | 21,837 | 10,931 | 2,292,885 | 1,147,755 | 3,668,616 | 1,836,408 | 917,154 | 459,102 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 2 | 109,182 | 21,890 | 7,915,695 | 1,587,025 | 12,665,112 | 2,539,240 | 3,166,278 | 634,810 | 75,199,103 | 15,076,737 | 2,074,458 | 415,910 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 2 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 2 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 2 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 2 | 10,919 | 5,465 | 1,091,900 | 546,500 | 1,747,040 | 874,400 | 436,760 | 218,600 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 2 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 3 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 3 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 3 | 10,919 | 5,465 | 7,752,490 | 3,880,150 | 12,403,984 | 6,208,240 | 3,100,996 | 1,552,060 | 53,434,857 | 26,744,343 | 1,474,065 | 737,775 | 
| app_proof | 11 | KeccakfPermAir |  | 3 | 262,056 | 88 | 3,451,277,520 | 1,158,960 | 2,761,022,016 | 927,168 | 690,255,504 | 231,792 | 18,999,060 | 6,380 | 524,112 | 176 | 
| app_proof | 12 | XorinVmAir |  | 3 | 10,919 | 5,465 | 16,269,310 | 8,142,850 | 26,030,896 | 13,028,560 | 6,507,724 | 3,257,140 | 161,492,010 | 80,827,350 | 4,454,952 | 2,229,720 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 3 | 152,863 | 109,281 | 12,993,355 | 9,288,885 | 20,789,368 | 14,862,216 | 5,197,342 | 3,715,554 | 110,825,675 | 79,228,725 | 3,057,260 | 2,185,620 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 3 | 54,593 | 10,943 | 6,960,608 | 1,395,232 | 11,136,972 | 2,232,372 | 2,784,243 | 558,093 | 37,600,929 | 7,536,991 | 1,037,267 | 207,917 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 3 | 567,768 | 480,808 | 32,646,660 | 27,646,460 | 52,234,656 | 44,234,336 | 13,058,664 | 11,058,584 | 329,305,440 | 278,868,640 | 9,084,288 | 7,692,928 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 3 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 27,705,948 | 5,553,572 | 764,302 | 153,202 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 3 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 3 | 98,268 | 32,804 | 5,404,740 | 1,804,220 | 8,647,584 | 2,886,752 | 2,161,896 | 721,688 | 53,433,225 | 17,837,175 | 1,474,020 | 492,060 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 3 | 65,514 | 22 | 2,784,345 | 935 | 4,454,952 | 1,496 | 1,113,738 | 374 | 28,498,590 | 9,570 | 786,168 | 264 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 3 | 65,513 | 23 | 4,913,475 | 1,725 | 7,861,560 | 2,760 | 1,965,390 | 690 | 33,247,848 | 11,672 | 917,182 | 322 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 3 | 218,372 | 43,772 | 13,102,320 | 2,626,320 | 20,963,712 | 4,202,112 | 5,240,928 | 1,050,528 | 87,075,835 | 17,454,085 | 2,402,092 | 481,492 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 3 | 589,583 | 458,993 | 60,432,258 | 47,046,782 | 96,691,612 | 75,274,852 | 24,172,903 | 18,818,713 | 598,426,745 | 465,877,895 | 16,508,324 | 12,851,804 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 3 | 272,962 | 251,326 | 27,296,200 | 25,132,600 | 43,673,920 | 40,212,160 | 10,918,480 | 10,053,040 | 267,161,558 | 245,985,322 | 7,369,974 | 6,785,802 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 3 | 10,919 | 5,465 | 1,064,603 | 532,837 | 1,703,364 | 852,540 | 425,841 | 213,135 | 10,291,158 | 5,150,762 | 283,894 | 142,090 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 3 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 3 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 3 | 32,756 | 12 | 2,374,810 | 870 | 3,799,696 | 1,392 | 949,924 | 348 | 21,373,290 | 7,830 | 589,608 | 216 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 3 | 10,919 | 5,465 | 600,545 | 300,575 | 960,872 | 480,920 | 240,218 | 120,230 | 5,541,393 | 2,773,487 | 152,866 | 76,510 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 3 | 10,919 | 5,465 | 764,330 | 382,550 | 1,222,928 | 612,080 | 305,732 | 153,020 | 6,728,834 | 3,367,806 | 185,623 | 92,905 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 3 | 21,837 | 10,931 | 2,292,885 | 1,147,755 | 3,668,616 | 1,836,408 | 917,154 | 459,102 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 3 | 109,190 | 21,882 | 7,916,275 | 1,586,445 | 12,666,040 | 2,538,312 | 3,166,510 | 634,578 | 75,204,613 | 15,071,227 | 2,074,610 | 415,758 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 3 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 3 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 3 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 3 | 10,919 | 5,465 | 1,091,900 | 546,500 | 1,747,040 | 874,400 | 436,760 | 218,600 | 12,270,227 | 6,141,293 | 338,489 | 169,415 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 3 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 4 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 4 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 4 | 10,918 | 5,466 | 7,751,780 | 3,880,860 | 12,402,848 | 6,209,376 | 3,100,712 | 1,552,344 | 53,429,963 | 26,749,237 | 1,473,930 | 737,910 | 
| app_proof | 11 | KeccakfPermAir |  | 4 | 262,032 | 112 | 3,450,961,440 | 1,475,040 | 2,760,769,152 | 1,180,032 | 690,192,288 | 295,008 | 18,997,320 | 8,120 | 524,064 | 224 | 
| app_proof | 12 | XorinVmAir |  | 4 | 10,918 | 5,466 | 16,267,820 | 8,144,340 | 26,028,512 | 13,030,944 | 6,507,128 | 3,257,736 | 161,477,220 | 80,842,140 | 4,454,544 | 2,230,128 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 4 | 152,855 | 109,289 | 12,992,675 | 9,289,565 | 20,788,280 | 14,863,304 | 5,197,070 | 3,715,826 | 110,819,875 | 79,234,525 | 3,057,100 | 2,185,780 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 4 | 54,592 | 10,944 | 6,960,480 | 1,395,360 | 11,136,768 | 2,232,576 | 2,784,192 | 558,144 | 37,600,240 | 7,537,680 | 1,037,248 | 207,936 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 4 | 567,765 | 480,811 | 32,646,488 | 27,646,632 | 52,234,380 | 44,234,612 | 13,058,595 | 11,058,653 | 329,303,700 | 278,870,380 | 9,084,240 | 7,692,976 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 4 | 54,593 | 10,943 | 2,183,720 | 437,720 | 3,493,952 | 700,352 | 873,488 | 175,088 | 27,705,948 | 5,553,572 | 764,302 | 153,202 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 4 | 43 | 21 | 4,193 | 2,047 | 6,708 | 3,276 | 1,677 | 819 | 15,588 | 7,612 | 430 | 210 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 4 | 98,266 | 32,806 | 5,404,630 | 1,804,330 | 8,647,408 | 2,886,928 | 2,161,852 | 721,732 | 53,432,138 | 17,838,262 | 1,473,990 | 492,090 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 4 | 65,509 | 27 | 2,784,133 | 1,147 | 4,454,612 | 1,836 | 1,113,653 | 459 | 28,496,415 | 11,745 | 786,108 | 324 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 4 | 65,511 | 25 | 4,913,325 | 1,875 | 7,861,320 | 3,000 | 1,965,330 | 750 | 33,246,833 | 12,687 | 917,154 | 350 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 4 | 218,370 | 43,774 | 13,102,200 | 2,626,440 | 20,963,520 | 4,202,304 | 5,240,880 | 1,050,576 | 87,075,038 | 17,454,882 | 2,402,070 | 481,514 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 4 | 589,619 | 458,957 | 60,435,948 | 47,043,092 | 96,697,516 | 75,268,948 | 24,174,379 | 18,817,237 | 598,463,285 | 465,841,355 | 16,509,332 | 12,850,796 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 4 | 272,964 | 251,324 | 27,296,400 | 25,132,400 | 43,674,240 | 40,211,840 | 10,918,560 | 10,052,960 | 267,163,515 | 245,983,365 | 7,370,028 | 6,785,748 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 4 | 10,918 | 5,466 | 1,064,505 | 532,935 | 1,703,208 | 852,696 | 425,802 | 213,174 | 10,290,215 | 5,151,705 | 283,868 | 142,116 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 4 | 220 | 36 | 36,300 | 5,940 | 29,040 | 4,752 | 7,260 | 1,188 | 31,900 | 5,220 | 880 | 144 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 4 | 43,674 | 21,862 | 3,275,550 | 1,639,650 | 5,240,880 | 2,623,440 | 1,310,220 | 655,860 | 30,080,468 | 15,057,452 | 829,806 | 415,378 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 4 | 32,755 | 13 | 2,374,738 | 942 | 3,799,580 | 1,508 | 949,895 | 377 | 21,372,638 | 8,482 | 589,590 | 234 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 4 | 10,918 | 5,466 | 600,490 | 300,630 | 960,784 | 481,008 | 240,196 | 120,252 | 5,540,885 | 2,773,995 | 152,852 | 76,524 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 4 | 10,918 | 5,466 | 764,260 | 382,620 | 1,222,816 | 612,192 | 305,704 | 153,048 | 6,728,218 | 3,368,422 | 185,606 | 92,922 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 4 | 21,837 | 10,931 | 2,292,885 | 1,147,755 | 3,668,616 | 1,836,408 | 917,154 | 459,102 | 18,206,599 | 9,113,721 | 502,251 | 251,413 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 4 | 109,182 | 21,890 | 7,915,695 | 1,587,025 | 12,665,112 | 2,539,240 | 3,166,278 | 634,810 | 75,199,103 | 15,076,737 | 2,074,458 | 415,910 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 4 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 4 | 306 | 206 | 229,500 | 154,500 | 367,200 | 247,200 | 91,800 | 61,800 | 11,093 | 7,467 | 306 | 206 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 4 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 4 | 10,918 | 5,466 | 1,091,800 | 546,600 | 1,746,880 | 874,560 | 436,720 | 218,640 | 12,269,103 | 6,142,417 | 338,458 | 169,446 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 4 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir |  | 5 | 2,084 | 2,012 | 52,100 | 50,300 | 83,360 | 80,480 | 20,840 | 20,120 | 75,545 | 72,935 | 2,084 | 2,012 | 
| app_proof | 1 | VmConnectorAir |  | 5 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir |  | 5 | 10,408 | 5,976 | 7,389,680 | 4,242,960 | 11,823,488 | 6,788,736 | 2,955,872 | 1,697,184 | 50,934,150 | 29,245,050 | 1,405,080 | 806,760 | 
| app_proof | 11 | KeccakfPermAir |  | 5 | 249,792 | 12,352 | 3,289,760,640 | 162,675,840 | 2,631,808,512 | 130,140,672 | 657,952,128 | 32,535,168 | 18,109,920 | 895,520 | 499,584 | 24,704 | 
| app_proof | 12 | XorinVmAir |  | 5 | 10,408 | 5,976 | 15,507,920 | 8,904,240 | 24,812,672 | 14,246,784 | 6,203,168 | 3,561,696 | 153,934,320 | 88,385,040 | 4,246,464 | 2,438,208 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 5 | 145,711 | 116,433 | 12,385,435 | 9,896,805 | 19,816,696 | 15,834,888 | 4,954,174 | 3,958,722 | 105,640,475 | 84,413,925 | 2,914,220 | 2,328,660 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 5 | 52,040 | 13,496 | 6,635,100 | 1,720,740 | 10,616,160 | 2,753,184 | 2,654,040 | 688,296 | 35,842,550 | 9,295,370 | 988,760 | 256,424 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 5 | 541,202 | 507,374 | 31,119,115 | 29,174,005 | 49,790,584 | 46,678,408 | 12,447,646 | 11,669,602 | 313,897,160 | 294,276,920 | 8,659,232 | 8,117,984 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 5 | 52,037 | 13,499 | 2,081,480 | 539,960 | 3,330,368 | 863,936 | 832,592 | 215,984 | 26,408,778 | 6,850,742 | 728,518 | 188,986 | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 5 | 44 | 20 | 4,290 | 1,950 | 6,864 | 3,120 | 1,716 | 780 | 15,950 | 7,250 | 440 | 200 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 5 | 93,671 | 37,401 | 5,151,905 | 2,057,055 | 8,243,048 | 3,291,288 | 2,060,762 | 822,822 | 50,933,607 | 20,336,793 | 1,405,065 | 561,015 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 5 | 62,447 | 3,089 | 2,653,998 | 131,282 | 4,246,396 | 210,052 | 1,061,599 | 52,513 | 27,164,445 | 1,343,715 | 749,364 | 37,068 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 5 | 62,446 | 3,090 | 4,683,450 | 231,750 | 7,493,520 | 370,800 | 1,873,380 | 92,700 | 31,691,345 | 1,568,175 | 874,244 | 43,260 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 5 | 208,158 | 53,986 | 12,489,480 | 3,239,160 | 19,983,168 | 5,182,656 | 4,995,792 | 1,295,664 | 83,003,003 | 21,526,917 | 2,289,738 | 593,846 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 5 | 562,019 | 486,557 | 57,606,948 | 49,872,092 | 92,171,116 | 79,795,348 | 23,042,779 | 19,948,837 | 570,449,285 | 493,855,355 | 15,736,532 | 13,623,596 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 5 | 260,205 | 1,939 | 26,020,500 | 193,900 | 41,632,800 | 310,240 | 10,408,200 | 77,560 | 254,675,644 | 1,897,796 | 7,025,535 | 52,353 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 5 | 10,408 | 5,976 | 1,014,780 | 582,660 | 1,623,648 | 932,256 | 405,912 | 233,064 | 9,809,540 | 5,632,380 | 270,608 | 155,376 | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 5 | 222 | 34 | 36,630 | 5,610 | 29,304 | 4,488 | 7,326 | 1,122 | 32,190 | 4,930 | 888 | 136 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 5 | 41,631 | 23,905 | 3,122,325 | 1,792,875 | 4,995,720 | 2,868,600 | 1,248,930 | 717,150 | 28,673,352 | 16,464,568 | 790,989 | 454,195 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 5 | 31,223 | 1,545 | 2,263,668 | 112,012 | 3,621,868 | 179,220 | 905,467 | 44,805 | 20,373,008 | 1,008,112 | 562,014 | 27,810 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 5 | 10,408 | 5,976 | 572,440 | 328,680 | 915,904 | 525,888 | 228,976 | 131,472 | 5,282,060 | 3,032,820 | 145,712 | 83,664 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 5 | 10,408 | 5,976 | 728,560 | 418,320 | 1,165,696 | 669,312 | 291,424 | 167,328 | 6,413,930 | 3,682,710 | 176,936 | 101,592 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 5 | 20,816 | 11,952 | 2,185,680 | 1,254,960 | 3,497,088 | 2,007,936 | 874,272 | 501,984 | 17,355,340 | 9,964,980 | 478,768 | 274,896 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 5 | 104,078 | 26,994 | 7,545,655 | 1,957,065 | 12,073,048 | 3,131,304 | 3,018,262 | 782,826 | 71,683,723 | 18,592,117 | 1,977,482 | 512,886 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> |  | 5 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 5 | 310 | 202 | 232,500 | 151,500 | 372,000 | 242,400 | 93,000 | 60,600 | 11,238 | 7,322 | 310 | 202 | 
| app_proof | 49 | VariableRangeCheckerAir |  | 5 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 5 | 10,408 | 5,976 | 1,040,800 | 597,600 | 1,665,280 | 956,160 | 416,320 | 239,040 | 11,695,990 | 6,715,530 | 322,648 | 185,256 | 
| app_proof | 9 | RangeTupleCheckerAir<2> |  | 5 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 39 | 329 | 39 | 13 | 3 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 16 | 131 | 13 | 2 | 0 | 2 | 2 | 2 | 
| internal_recursive.1 | 1 | 10 | 108 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 182 | 1,010 | 182 | 57 | 21 | 4 | 11 | 11 | 
| leaf | 1 | 90 | 498 | 90 | 28 | 10 | 5 | 10 | 10 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 81,415,160 | 289 | 93 | 0 | 0 | 113 | 39 | 38 | 56 | 17 | 0 | 82 | 66 | 15 | 5 | 10 | 93 | 93 | 113 | 0 | 1 | 17 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 16,836,945 | 115 | 21 | 0 | 0 | 59 | 21 | 21 | 24 | 12 | 0 | 33 | 25 | 8 | 1 | 6 | 21 | 21 | 59 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 97 | 15 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 28 | 20 | 7 | 1 | 5 | 15 | 15 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 476,578,798 | 827 | 385 | 0 | 0 | 232 | 120 | 118 | 70 | 41 | 0 | 209 | 172 | 37 | 20 | 17 | 385 | 385 | 232 | 0 | 3 | 40 | 0 | 0 | 
| leaf | 1 | prover | 240,255,800 | 408 | 147 | 0 | 0 | 142 | 71 | 69 | 45 | 26 | 0 | 117 | 95 | 22 | 10 | 11 | 147 | 147 | 142 | 0 | 3 | 25 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 7,020,869 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,281,375 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 28,071,113 | 2,013,265,921 | 
| leaf | 1 | prover | 0 | 15,739,653 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 7 | 0 | 7 | 7 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 834,254,540 | 1,102 | 358 | 0 | 0 | 514 | 284 | 283 | 124 | 105 | 0 | 229 | 167 | 61 | 37 | 24 | 358 | 358 | 514 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 1 | 834,254,540 | 1,083 | 343 | 0 | 0 | 514 | 284 | 283 | 124 | 105 | 0 | 225 | 164 | 61 | 36 | 24 | 343 | 343 | 514 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 2 | 834,254,540 | 1,080 | 338 | 0 | 0 | 515 | 284 | 284 | 124 | 106 | 0 | 225 | 164 | 61 | 36 | 24 | 338 | 338 | 515 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 3 | 834,254,540 | 1,157 | 340 | 0 | 53 | 513 | 283 | 282 | 124 | 105 | 0 | 302 | 240 | 62 | 37 | 24 | 340 | 340 | 513 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 4 | 834,254,540 | 1,151 | 336 | 0 | 50 | 516 | 285 | 284 | 124 | 105 | 0 | 298 | 237 | 61 | 36 | 24 | 336 | 336 | 516 | 0 | 1 | 104 | 0 | 0 | 
| app_proof | prover |  | 5 | 823,768,780 | 1,072 | 334 | 0 | 0 | 511 | 283 | 283 | 124 | 103 | 0 | 226 | 164 | 61 | 36 | 24 | 334 | 334 | 511 | 0 | 1 | 102 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 91,641,738 | 2,013,265,921 | 
| app_proof | prover |  | 1 | 0 | 91,641,738 | 2,013,265,921 | 
| app_proof | prover |  | 2 | 0 | 91,641,738 | 2,013,265,921 | 
| app_proof | prover |  | 3 | 0 | 91,641,738 | 2,013,265,921 | 
| app_proof | prover |  | 4 | 0 | 91,641,738 | 2,013,265,921 | 
| app_proof | prover |  | 5 | 0 | 84,563,850 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 1,383 | 128 | 14,365,133 | 111.44 | 0 | 8,692 | 

| group | program | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- |
| app_proof |  | memory | 0 | 1 | 
| app_proof |  | memory | 1 | 1 | 
| app_proof |  | memory | 2 | 1 | 
| app_proof |  | memory | 3 | 1 | 
| app_proof |  | memory | 4 | 1 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 38 | 2 | 18 | 1,452 | 3 | 38 | 0 | 200 | 0 | 6 | 0 | 9,366,884,196 | 644,170,284 | 10,011,054,480 | 2,097,152 | 2 | 0 | 0 | 3 | 14 | 76 | 2,413,000 | 31.69 | 0 | 0 | 
| app_proof |  | 1 | 1 | 2 | 12 | 1,395 | 2 | 1 | 0 | 227 | 0 | 2 | 0 | 9,367,191,348 | 644,784,732 | 10,011,976,080 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 59 | 2,413,000 | 40.44 | 0 | 0 | 
| app_proof |  | 2 | 1 | 2 | 12 | 1,393 | 2 | 1 | 0 | 232 | 1 | 2 | 0 | 9,366,427,680 | 645,548,400 | 10,011,976,080 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 57 | 2,413,000 | 42.31 | 0 | 0 | 
| app_proof |  | 3 | 1 | 2 | 12 | 1,468 | 2 | 1 | 0 | 228 | 0 | 2 | 0 | 9,367,190,352 | 644,785,728 | 10,011,976,080 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 58 | 2,413,000 | 40.98 | 0 | 0 | 
| app_proof |  | 4 | 1 | 2 | 12 | 1,464 | 2 | 1 | 0 | 226 | 0 | 2 | 0 | 9,366,427,644 | 645,548,436 | 10,011,976,080 | 2,097,152 | 2 | 0 | 0 | 2 | 9 | 61 | 2,413,000 | 39.39 | 0 | 0 | 
| app_proof |  | 5 | 1 | 2 | 11 | 1,383 | 2 | 1 | 0 | 230 | 0 | 2 | 0 | 8,931,942,492 | 954,204,468 | 9,886,146,960 | 2,097,152 | 2 | 0 | 0 | 2 | 8 | 58 | 2,300,133 | 39.35 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 7 | 0 | 7 | 7 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31022383256)
