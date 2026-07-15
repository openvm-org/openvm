| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.10 |  1.10 |  1.10 |
| app_proof |  0.52 |  0.52 |  0.52 |
| leaf |  0.20 |  0.20 |  0.20 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  501 |  501 |  501 |  501 |
| `compile_metered_time_ms` |  8 |  8 |  8 |  8 |
| `execute_metered_time_ms` |  20 | -          | -          | -          |
| `execute_metered_insns` |  4,090,656 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  204.20 | -          |  204.20 |  204.20 |
| `execute_preflight_insns` |  4,090,656 |  4,090,656 |  4,090,656 |  4,090,656 |
| `execute_preflight_time_ms` |  123 |  123 |  123 |  123 |
| `execute_preflight_insn_mi/s` |  36.71 | -          |  36.71 |  36.71 |
| `trace_gen_time_ms   ` |  34 |  34 |  34 |  34 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  342 |  342 |  342 |  342 |
| `prover.main_trace_commit_time_ms` |  64 |  64 |  64 |  64 |
| `prover.rap_constraints_time_ms` |  207 |  207 |  207 |  207 |
| `prover.openings_time_ms` |  70 |  70 |  70 |  70 |
| `prover.rap_constraints.logup_gkr_time_ms` |  100 |  100 |  100 |  100 |
| `prover.rap_constraints.round0_time_ms` |  78 |  78 |  78 |  78 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction.round0_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.whir_time_ms` |  49 |  49 |  49 |  49 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  197 |  197 |  197 |  197 |
| `execute_preflight_time_ms` |  5 |  5 |  5 |  5 |
| `trace_gen_time_ms   ` |  41 |  41 |  41 |  41 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  155 |  155 |  155 |  155 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  76 |  76 |  76 |  76 |
| `prover.openings_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  37 |  37 |  37 |  37 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  157 |  157 |  157 |  157 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  14 |  14 |  14 |  14 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  143 |  143 |  143 |  143 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  72 |  72 |  72 |  72 |
| `prover.openings_time_ms` |  37 |  37 |  37 |  37 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  28 |  28 |  28 |  28 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  121 |  121 |  121 |  121 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  110 |  110 |  110 |  110 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  25 |  25 |  25 |  25 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  106 |  106 |  106 |  106 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  96 |  96 |  96 |  96 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/3cdc14dd3e021e96f3b8f0eacb02000da4669c1e/regex-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.batch_constraints.before_round0 | 4.17 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 4.17 | app_proof.prover.0 |
| prover.gkr_input_evals | 4.07 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 4.07 | app_proof.prover.0 |
| prover.stacked_commit | 2.37 | app_proof.prover.0 |
| prover.openings | 1.69 | app_proof.prover.0 |
| prover.merkle_tree | 1.69 | app_proof.prover.0 |
| prover.prove_whir_opening | 1.69 | app_proof.prover.0 |
| prover.rs_code_matrix | 1.69 | app_proof.prover.0 |
| prover.rap_constraints | 1.54 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 1.13 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 1.04 | app_proof.prover.0 |
| generate mem proving ctxs | 0.90 | app_proof.0 |
| prover.before_gkr_input_evals | 0.90 | app_proof.prover.0 |
| set initial memory | 0.87 | app_proof.0 |
| tracegen.exp_bits_len | 0.33 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.33 | leaf.0 |
| tracegen.pow_checker | 0.33 | leaf.0 |
| tracegen.whir_folding | 0.26 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.26 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.26 | leaf.0 |
| tracegen.proof_shape | 0.24 | leaf.0 |
| tracegen.public_values | 0.24 | leaf.0 |
| tracegen.range_checker | 0.24 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 121 | 267,239 | 230,727 | 24 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | KeccakfOpAir |  | 110 | 27 | 2 | 
| 11 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 12 | XorinVmAir |  | 357 | 92 | 3 | 
| 13 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 7 | 2 | 
| 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 45 | 3 | 
| 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 99 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 8 | 3 | 
| 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 24 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> |  | 17 | 11 | 3 | 
| 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> |  | 17 | 13 | 3 | 
| 26 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> |  | 17 | 11 | 3 | 
| 27 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> |  | 17 | 13 | 3 | 
| 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> |  | 18 | 14 | 3 | 
| 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> |  | 17 | 13 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 30 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> |  | 17 | 15 | 3 | 
| 31 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 18 | 16 | 3 | 
| 32 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> |  | 19 | 16 | 3 | 
| 33 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> |  | 18 | 17 | 3 | 
| 34 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 18 | 3 | 
| 35 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 30 | 3 | 
| 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 70 | 3 | 
| 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 7 | 3 | 
| 38 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 31 | 3 | 
| 39 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 103 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 40 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 46 | 3 | 
| 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 132 | 3 | 
| 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 23 | 3 | 
| 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 11 | 3 | 
| 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 7 | 2 | 
| 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 14 | 3 | 
| 46 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 47 | PhantomAir |  | 3 | 1 | 2 | 
| 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 49 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | new_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 59 |  |  |  | 287 |  |  |  |  |  |  | 
| app_proof |  |  |  | 501 |  | 20 | 4,090,656 | 204.20 | 0 | 537 |  | 
| internal_for_leaf |  |  | 157 |  |  |  |  |  |  |  | 158 | 
| internal_recursive.0 |  |  | 121 |  |  |  |  |  |  |  | 121 | 
| internal_recursive.1 |  |  | 106 |  |  |  |  |  |  |  | 106 | 
| leaf |  | 197 |  |  |  |  |  |  |  |  | 197 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 43 | KeccakfOpAir | 0 | 320 | 
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 20 | 
| app_proof | Rv64HintStoreAir | 40 | Rv64HintStoreAir | 0 | 127,936 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 39 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 3,229,116 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 35 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | 29,439,652 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 38 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 460,656 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 36 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 15,317,236 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 37 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 0 | 4,004 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 2,136,240 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 8 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 18,102,060 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 11 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 551,700 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 40,680 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 16 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | 19,392 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 17 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 4,656 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 10 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 324,416 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 30 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 11,292,432 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 31 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 8,846,400 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 32 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 2,713,680 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 33 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 6,427,584 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | 20 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | 0 | 1,309,672 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | 23 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | 0 | 4,420 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | 26 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | 0 | 4,583,280 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 49,134,124 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 19 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 0 | 44,408 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | 22 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | 0 | 1,404 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 4,400,864 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 48 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 0 | 2,280 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 47 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 9,540 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 45 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 293,328 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 34 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 2,993,952 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 21 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 0 | 112,112 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | 24 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | 0 | 524,056 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | 27 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | 0 | 6,107,712 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 33,478,172 | 
| app_proof | XorinVmAir | 41 | XorinVmAir | 0 | 656 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 |  | 0 | 
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 |  | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 |  | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 4 | 4 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 0 | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 2 | 3 | 
| app_proof | XorinVmAir | 0 |  | 0 | 

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
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 303,104 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
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
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 3,932,160 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 262,144 | 
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

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | KeccakfOpAir | KECCAKF | 0 | 1 | 
| app_proof | 12 | XorinVmAir | XORIN | 0 | 1 | 
| app_proof | 13 | Rv64HintStoreAir | HINT_BUFFER | 0 | 7 | 
| app_proof | 13 | Rv64HintStoreAir | HINT_STORED | 0 | 1 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | ANDI | 0 | 73,203 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | ORI | 0 | 35 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | XORI | 0 | 151 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | SLTI | 0 | 93 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | SLTIU | 0 | 9,504 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | SRAI | 0 | 91 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SLLI | 0 | 295,797 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | SRLI | 0 | 52,322 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | ADDI | 0 | 669,083 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 93,561 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 133,908 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 62,035 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 5,807 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGE | 0 | 357 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 121,565 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLT | 0 | 5,201 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 57,177 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 148,200 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 87,059 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | STORED | 0 | 643,811 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | LOADD | 0 | 944,887 | 
| app_proof | 26 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | STOREW | 0 | 117,456 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | LOADWU | 0 | 88,140 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | LOADW | 0 | 84,632 | 
| app_proof | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | STOREH | 0 | 10,078 | 
| app_proof | 30 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | LOADHU | 0 | 85 | 
| app_proof | 31 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | LOADH | 0 | 27 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | STOREB | 0 | 2,156 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | LOADBU | 0 | 25,186 | 
| app_proof | 34 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | LOADB | 0 | 854 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | SLLIW | 0 | 1 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | SRLIW | 0 | 96 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | ADDIW | 0 | 404 | 
| app_proof | 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 0 | 335 | 
| app_proof | 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 0 | 343 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | SLT | 0 | 1 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 9,194 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | ADDW | 0 | 34 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | SUBW | 0 | 5,035 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 13,335 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 12,917 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | XOR | 0 | 9,352 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | ADD | 0 | 244,714 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | SUB | 0 | 56,987 | 
| app_proof | 47 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | REMU | 0 | 38 | 
| app_proof | 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | MULHU | 0 | 159 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 5,238 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 131,072 | 10 | 1,310,720 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | KeccakfOpAir | prover | 0 | 1 | 284 | 284 | 
| app_proof | 11 | KeccakfPermAir | prover | 0 | 32 | 2,634 | 84,288 | 
| app_proof | 12 | XorinVmAir | prover | 0 | 1 | 669 | 669 | 
| app_proof | 13 | Rv64HintStoreAir | prover | 0 | 8,192 | 27 | 221,184 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover | 0 | 131,072 | 36 | 4,718,592 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover | 0 | 16,384 | 30 | 491,520 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | prover | 0 | 128 | 51 | 6,528 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover | 0 | 524,288 | 54 | 28,311,552 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover | 0 | 1,048,576 | 25 | 26,214,400 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 131,072 | 17 | 2,228,224 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 65,536 | 21 | 1,376,256 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 262,144 | 32 | 8,388,608 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | prover | 0 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | prover | 0 | 1,048,576 | 30 | 31,457,280 | 
| app_proof | 26 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | prover | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | prover | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | prover | 0 | 131,072 | 31 | 4,063,232 | 
| app_proof | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | prover | 0 | 16,384 | 31 | 507,904 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 65,536 | 33 | 2,162,688 | 
| app_proof | 30 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | prover | 0 | 128 | 31 | 3,968 | 
| app_proof | 31 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | prover | 0 | 32 | 32 | 1,024 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | prover | 0 | 4,096 | 36 | 147,456 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | prover | 0 | 32,768 | 34 | 1,114,112 | 
| app_proof | 34 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | prover | 0 | 1,024 | 35 | 35,840 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover | 0 | 128 | 47 | 6,016 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover | 0 | 512 | 24 | 12,288 | 
| app_proof | 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 0 | 1,024 | 64 | 65,536 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 16,384 | 36 | 589,824 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover | 0 | 8,192 | 31 | 253,952 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 65,536 | 45 | 2,949,120 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover | 0 | 524,288 | 32 | 16,777,216 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 47 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 16,384 | 300 | 4,915,200 | 
| app_proof | 49 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | prover | 0 | 64 | 87 | 5,568 | 
| app_proof | 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | prover | 0 | 256 | 55 | 14,080 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 8,192 | 43 | 352,256 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 91,776 | 39,296 | 2,294,400 | 982,400 | 3,671,040 | 1,571,840 | 917,760 | 392,960 | 3,326,880 | 1,424,480 | 91,776 | 39,296 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | KeccakfOpAir | 0 | 1 |  | 710 |  | 1,136 |  | 284 |  | 3,988 |  | 110 |  | 
| app_proof | 11 | KeccakfPermAir | 0 | 24 | 8 | 316,080 | 105,360 | 252,864 | 84,288 | 63,216 | 21,072 | 1,740 | 580 | 48 | 16 | 
| app_proof | 12 | XorinVmAir | 0 | 1 |  | 1,673 |  | 2,676 |  | 669 |  | 12,942 |  | 357 |  | 
| app_proof | 13 | Rv64HintStoreAir | 0 | 6,384 | 1,808 | 861,840 | 244,080 | 689,472 | 195,264 | 172,368 | 48,816 | 3,934,140 | 1,114,180 | 108,528 | 30,736 | 
| app_proof | 14 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | 73,389 | 57,683 | 6,605,010 | 5,191,470 | 10,568,016 | 8,306,352 | 2,642,004 | 2,076,588 | 53,207,025 | 41,820,175 | 1,467,780 | 1,153,660 | 
| app_proof | 15 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | 9,597 | 6,787 | 719,775 | 509,025 | 1,151,640 | 814,440 | 287,910 | 203,610 | 4,870,478 | 3,444,402 | 134,358 | 95,018 | 
| app_proof | 16 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> | 0 | 91 | 37 | 11,603 | 4,717 | 18,564 | 7,548 | 4,641 | 1,887 | 65,975 | 26,825 | 1,820 | 740 | 
| app_proof | 17 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | 348,119 | 176,169 | 46,996,065 | 23,782,815 | 75,193,704 | 38,052,504 | 18,798,426 | 9,513,126 | 239,766,962 | 121,336,398 | 6,614,261 | 3,347,211 | 
| app_proof | 18 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | 669,083 | 379,493 | 41,817,688 | 23,718,312 | 66,908,300 | 37,949,300 | 16,727,075 | 9,487,325 | 388,068,140 | 220,105,940 | 10,705,328 | 6,071,888 | 
| app_proof | 19 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 93,561 | 37,511 | 3,976,343 | 1,594,217 | 6,362,148 | 2,550,748 | 1,590,537 | 637,687 | 47,482,208 | 19,036,832 | 1,309,854 | 525,154 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 52,020 | 13,516 | 2,731,050 | 709,590 | 4,369,680 | 1,135,344 | 1,092,420 | 283,836 | 7,542,900 | 1,959,820 | 208,080 | 54,064 | 
| app_proof | 20 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 133,908 | 128,236 | 8,034,480 | 7,694,160 | 12,855,168 | 12,310,656 | 3,213,792 | 3,077,664 | 72,812,475 | 69,728,325 | 2,008,620 | 1,923,540 | 
| app_proof | 21 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 67,842 | 63,230 | 3,052,890 | 2,845,350 | 4,884,624 | 4,552,560 | 1,221,156 | 1,138,140 | 29,511,270 | 27,505,050 | 814,104 | 758,760 | 
| app_proof | 22 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 184,300 | 77,844 | 14,744,000 | 6,227,520 | 23,590,400 | 9,964,032 | 5,897,600 | 2,491,008 | 93,532,250 | 39,505,830 | 2,580,200 | 1,089,816 | 
| app_proof | 23 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 235,259 | 26,885 | 15,291,835 | 1,747,525 | 24,466,936 | 2,796,040 | 6,116,734 | 699,010 | 93,809,527 | 10,720,393 | 2,587,849 | 295,735 | 
| app_proof | 24 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<8, 1> | 0 | 643,811 | 404,765 | 48,285,825 | 30,357,375 | 77,257,320 | 48,571,800 | 19,314,330 | 12,142,950 | 396,748,529 | 249,436,431 | 10,944,787 | 6,881,005 | 
| app_proof | 25 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<8, 1> | 0 | 944,887 | 103,689 | 70,866,525 | 7,776,675 | 113,386,440 | 12,442,680 | 28,346,610 | 3,110,670 | 582,286,614 | 63,898,346 | 16,063,079 | 1,762,713 | 
| app_proof | 26 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<4, 1> | 0 | 117,456 | 13,616 | 8,809,200 | 1,021,200 | 14,094,720 | 1,633,920 | 3,523,680 | 408,480 | 72,382,260 | 8,390,860 | 1,996,752 | 231,472 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<4, 1> | 0 | 88,140 | 42,932 | 6,610,500 | 3,219,900 | 10,576,800 | 5,151,840 | 2,644,200 | 1,287,960 | 54,316,275 | 26,456,845 | 1,498,380 | 729,844 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<4, 1> | 0 | 84,632 | 46,440 | 6,558,980 | 3,599,100 | 10,494,368 | 5,758,560 | 2,623,592 | 1,439,640 | 55,222,380 | 30,302,100 | 1,523,376 | 835,920 | 
| app_proof | 29 | VmAirWrapper<Rv64StoreAdapterAir, StoreCoreAir<2, 2> | 0 | 10,078 | 6,306 | 781,045 | 488,715 | 1,249,672 | 781,944 | 312,418 | 195,486 | 6,210,568 | 3,886,072 | 171,326 | 107,202 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 53,080 | 12,456 | 8,758,200 | 2,055,240 | 7,006,560 | 1,644,192 | 1,751,640 | 411,048 | 7,696,600 | 1,806,120 | 212,320 | 49,824 | 
| app_proof | 30 | VmAirWrapper<Rv64LoadAdapterAir, LoadCoreAir<2, 2> | 0 | 85 | 43 | 6,588 | 3,332 | 10,540 | 5,332 | 2,635 | 1,333 | 52,382 | 26,498 | 1,445 | 731 | 
| app_proof | 31 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendCoreAir<2, 2> | 0 | 27 | 5 | 2,160 | 400 | 3,456 | 640 | 864 | 160 | 17,618 | 3,262 | 486 | 90 | 
| app_proof | 32 | VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir> | 0 | 2,156 | 1,940 | 194,040 | 174,600 | 310,464 | 279,360 | 77,616 | 69,840 | 1,484,945 | 1,336,175 | 40,964 | 36,860 | 
| app_proof | 33 | VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir> | 0 | 25,186 | 7,582 | 2,140,810 | 644,470 | 3,425,296 | 1,031,152 | 856,324 | 257,788 | 16,433,865 | 4,947,255 | 453,348 | 136,476 | 
| app_proof | 34 | VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendByteCoreAir> | 0 | 854 | 170 | 74,725 | 14,875 | 119,560 | 23,800 | 29,890 | 5,950 | 588,193 | 117,087 | 16,226 | 3,230 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | 97 | 31 | 11,398 | 3,642 | 18,236 | 5,828 | 4,559 | 1,457 | 56,260 | 17,980 | 1,552 | 496 | 
| app_proof | 37 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | 404 | 108 | 24,240 | 6,480 | 38,784 | 10,368 | 9,696 | 2,592 | 205,030 | 54,810 | 5,656 | 1,512 | 
| app_proof | 41 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 678 | 346 | 108,480 | 55,360 | 173,568 | 88,576 | 43,392 | 22,144 | 589,860 | 301,020 | 16,272 | 8,304 | 
| app_proof | 42 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 9,195 | 7,189 | 827,550 | 647,010 | 1,324,080 | 1,035,216 | 331,020 | 258,804 | 5,999,738 | 4,690,822 | 165,510 | 129,402 | 
| app_proof | 43 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 5,069 | 3,123 | 392,848 | 242,032 | 628,556 | 387,252 | 157,139 | 96,813 | 3,123,772 | 1,924,548 | 86,173 | 53,091 | 
| app_proof | 44 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 35,604 | 29,932 | 4,005,450 | 3,367,350 | 6,408,720 | 5,387,760 | 1,602,180 | 1,346,940 | 29,684,835 | 24,955,805 | 818,892 | 688,436 | 
| app_proof | 45 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 301,701 | 222,587 | 24,136,080 | 17,806,960 | 38,617,728 | 28,491,136 | 9,654,432 | 7,122,784 | 207,796,564 | 153,306,796 | 5,732,319 | 4,229,153 | 
| app_proof | 46 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 47 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 48 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 105,100 | 25,972 | 78,825,000 | 19,479,000 | 126,120,000 | 31,166,400 | 31,530,000 | 7,791,600 | 3,809,875 | 941,485 | 105,100 | 25,972 | 
| app_proof | 49 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 0 | 38 | 26 | 8,265 | 5,655 | 13,224 | 9,048 | 3,306 | 2,262 | 56,478 | 38,642 | 1,558 | 1,066 | 
| app_proof | 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 0 | 159 | 97 | 21,863 | 13,337 | 34,980 | 21,340 | 8,745 | 5,335 | 230,550 | 140,650 | 6,360 | 3,880 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 5,238 | 2,954 | 563,085 | 317,555 | 900,936 | 508,088 | 225,234 | 127,022 | 5,886,203 | 3,319,557 | 162,378 | 91,574 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 8 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 14 | 157 | 14 | 5 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 121 | 10 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 106 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 41 | 197 | 40 | 8 | 1 | 5 | 10 | 10 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,548,797 | 143 | 33 | 0 | 0 | 72 | 25 | 24 | 33 | 13 | 0 | 37 | 28 | 9 | 2 | 6 | 33 | 33 | 72 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 110 | 20 | 0 | 0 | 56 | 20 | 20 | 23 | 12 | 0 | 33 | 25 | 7 | 1 | 6 | 20 | 20 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 96 | 14 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 27 | 20 | 7 | 1 | 5 | 15 | 14 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 45,770,365 | 155 | 32 | 0 | 0 | 76 | 34 | 33 | 23 | 18 | 0 | 46 | 37 | 9 | 2 | 6 | 33 | 32 | 76 | 0 | 3 | 16 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,343,427 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 6,744,003 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 6 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 198,949,451 | 342 | 64 | 0 | 0 | 207 | 78 | 77 | 28 | 100 | 1 | 70 | 49 | 20 | 9 | 11 | 64 | 64 | 207 | 0 | 1 | 98 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 101,378,336 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 2 | 34 | 501 | 34 | 0 | 3,211,887,125 | 1,402,755,551 | 4,614,642,676 | 2,097,152 | 3 | 6 | 123 | 4,090,656 | 36.71 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/3cdc14dd3e021e96f3b8f0eacb02000da4669c1e

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29438095212)
