| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  10.16 |  7.11 |  7.11 |
| app_proof |  8.45 |  5.39 |  5.39 |
| leaf |  1.02 |  1.02 |  1.02 |
| internal_for_leaf |  0.36 |  0.36 |  0.36 |
| internal_recursive.0 |  0.19 |  0.19 |  0.19 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,178 |  8,356 |  5,303 |  3,053 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  91 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  122.52 | -          |  122.52 |  122.52 |
| `execute_preflight_insns` |  5,583,980.50 |  11,167,961 |  7,134,000 |  4,033,961 |
| `execute_preflight_time_ms` |  251 |  502 |  287 |  215 |
| `execute_preflight_insn_mi/s` |  35.62 | -          |  37.62 |  33.62 |
| `trace_gen_time_ms   ` |  123.50 |  247 |  176 |  71 |
| `set_initial_memory_time_ms` |  87 |  174 |  133 |  41 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  3,714 |  7,428 |  4,776 |  2,652 |
| `prover.main_trace_commit_time_ms` |  978 |  1,956 |  1,362 |  594 |
| `prover.rap_constraints_time_ms` |  2,019 |  4,038 |  2,503 |  1,535 |
| `prover.openings_time_ms` |  716 |  1,432 |  910 |  522 |
| `prover.rap_constraints.logup_gkr_time_ms` |  567 |  1,134 |  615 |  519 |
| `prover.rap_constraints.round0_time_ms` |  1,059.50 |  2,119 |  1,379 |  740 |
| `prover.rap_constraints.mle_rounds_time_ms` |  392 |  784 |  509 |  275 |
| `prover.openings.stacked_reduction_time_ms` |  170 |  340 |  221 |  119 |
| `prover.openings.stacked_reduction.round0_time_ms` |  105.50 |  211 |  139 |  72 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  64 |  128 |  82 |  46 |
| `prover.openings.whir_time_ms` |  545 |  1,090 |  688 |  402 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,022 |  1,022 |  1,022 |  1,022 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  112 |  112 |  112 |  112 |
| `generate_blob_total_time_ms` |  17 |  17 |  17 |  17 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  909 |  909 |  909 |  909 |
| `prover.main_trace_commit_time_ms` |  381 |  381 |  381 |  381 |
| `prover.rap_constraints_time_ms` |  232 |  232 |  232 |  232 |
| `prover.openings_time_ms` |  295 |  295 |  295 |  295 |
| `prover.rap_constraints.logup_gkr_time_ms` |  52 |  52 |  52 |  52 |
| `prover.rap_constraints.round0_time_ms` |  113 |  113 |  113 |  113 |
| `prover.rap_constraints.mle_rounds_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings.stacked_reduction_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.stacked_reduction.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.whir_time_ms` |  257 |  257 |  257 |  257 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  356 |  356 |  356 |  356 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  22 |  22 |  22 |  22 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  334 |  334 |  334 |  334 |
| `prover.main_trace_commit_time_ms` |  119 |  119 |  119 |  119 |
| `prover.rap_constraints_time_ms` |  109 |  109 |  109 |  109 |
| `prover.openings_time_ms` |  104 |  104 |  104 |  104 |
| `prover.rap_constraints.logup_gkr_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  52 |  52 |  52 |  52 |
| `prover.openings.stacked_reduction_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  90 |  90 |  90 |  90 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  185 |  185 |  185 |  185 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  173 |  173 |  173 |  173 |
| `prover.main_trace_commit_time_ms` |  47 |  47 |  47 |  47 |
| `prover.rap_constraints_time_ms` |  68 |  68 |  68 |  68 |
| `prover.openings_time_ms` |  57 |  57 |  57 |  57 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  48 |  48 |  48 |  48 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  149 |  149 |  149 |  149 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  139 |  139 |  139 |  139 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  43 |  43 |  43 |  43 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  36 |  36 |  36 |  36 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/f25fa657ef952641bd5dc1983b3ce3cafb5a5af6/sha2_bench-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 14.71 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 12.65 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 12.65 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 12.31 | app_proof.prover.0 |
| prover.gkr_input_evals | 12.31 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 10.74 | app_proof.prover.0 |
| prover.rap_constraints | 10.74 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 10.74 | app_proof.prover.0 |
| prover.merkle_tree | 9.92 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.92 | app_proof.prover.0 |
| prover.openings | 9.92 | app_proof.prover.0 |
| prover.rs_code_matrix | 9.91 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 4.99 | app_proof.prover.0 |
| tracegen.whir_final_poly_query_eval | 1.10 | leaf.0 |
| tracegen.pow_checker | 1.10 | leaf.0 |
| tracegen.exp_bits_len | 1.10 | leaf.0 |
| tracegen.whir_folding | 0.97 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.97 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.97 | leaf.0 |
| generate mem proving ctxs | 0.87 | app_proof.1 |
| set initial memory | 0.87 | app_proof.1 |
| tracegen.public_values | 0.80 | leaf.0 |
| tracegen.proof_shape | 0.80 | leaf.0 |
| tracegen.range_checker | 0.80 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 138 | 267,239 | 228,876 | 63 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 1 | 149 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 1 | 85 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 14 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> |  | 16 | 8 | 3 | 
| 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 23 | 3 | 
| 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 32 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 27 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> |  | 19 | 16 | 3 | 
| 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 35 | 3 | 
| 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> |  | 20 | 21 | 3 | 
| 31 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 32 | PhantomAir |  | 3 | 1 | 2 | 
| 33 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 34 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | new_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 70 |  |  |  | 344 |  |  |  |  |  |  | 
| app_proof |  |  |  | 3,053 |  | 91 | 11,167,961 | 122.52 | 0 | 8,453 |  | 
| internal_for_leaf |  |  | 356 |  |  |  |  |  |  |  | 356 | 
| internal_recursive.0 |  |  | 185 |  |  |  |  |  |  |  | 185 | 
| internal_recursive.1 |  |  | 149 |  |  |  |  |  |  |  | 149 | 
| leaf |  | 1,022 |  |  |  |  |  |  |  |  | 1,022 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 20 | 
| app_proof | Rv64HintStoreAir | 24 | Rv64HintStoreAir | 0 | 104 | 
| app_proof | Sha2MainAir<Sha256Config> | 26 | Sha2MainAir<Sha256Config> | 0 | 28,466,976 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 33,491,456 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 67,086,656 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 11 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 128 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 13,396,608 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 7,116,880 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 14 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 68 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 30,223,680 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 20,173,296 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 20 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 16,876,560 | 
| app_proof | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 23 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 0 | 46,272,644 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 15,071,184 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 5,860,904 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 117,432,224 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 30 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 56 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 3,349,504 | 
| app_proof | Sha2MainAir<Sha256Config> | 26 | Sha2MainAir<Sha256Config> | 1 | 16,097,776 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 18,939,904 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 1 | 37,936,512 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 7,578,624 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 1 | 4,024,444 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 14 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 1 | 204 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 17,089,824 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 11,407,632 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 20 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 9,543,280 | 
| app_proof | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 23 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 1 | 26,165,128 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 8,522,544 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 3,315,088 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 66,393,768 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 1,893,952 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 |  | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 3 | 6 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 |  | 11 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 3 | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 2 | 4 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 1 | 3 | 
| app_proof | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 0 |  | 93 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 12 | 26 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 2 | 2 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 |  | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 1 |  | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 2 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 1 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 1 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 1 |  | 46 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 5 | 5 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 0 | 0 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 88 | 4 | 
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
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 44 | 5,632 | 
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
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 104,658 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_BUFFER | 0 | 1 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_STORED | 0 | 1 | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | ADDI | 0 | 1,051,651 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 104,672 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 313,983 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 315,613 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 106,301 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 104,661 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 315,616 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 420,274 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 209,386 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 0 | 104,659 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 0 | 1,048,235 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 0 | 1,048,769 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SRLW | 0 | 1 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 0 | 104,660 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 0 | 104,662 | 
| app_proof | 27 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 2 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | ADDW | 0 | 1 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 0 | 104,659 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 523,303 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 1 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 0 | 734,251 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 0 | 313,978 | 
| app_proof | 32 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 1 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 1 | 59,183 | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | ADDI | 1 | 594,662 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 1 | 59,186 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 1 | 177,553 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 1 | 178,474 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 1 | 60,108 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 1 | 59,183 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLT | 1 | 1 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 1 | 178,475 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 237,659 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 118,379 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADB | 1 | 8 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 1 | 59,190 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 1 | 592,789 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREB | 1 | 40 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 1 | 592,774 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SLLW | 1 | 1 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SRLW | 1 | 2 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 1 | 59,196 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 1 | 59,220 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 1 | 59,183 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 1 | 295,922 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 1 | 14 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 1 | 415,206 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 1 | 177,552 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 0 | 131,072 | 150 | 19,660,800 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 14 | Rv64HintStoreAir | prover | 0 | 2 | 27 | 54 | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | prover | 0 | 2,097,152 | 25 | 52,428,800 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 131,072 | 17 | 2,228,224 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 524,288 | 24 | 12,582,912 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 524,288 | 18 | 9,437,184 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 524,288 | 32 | 16,777,216 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 1,048,576 | 26 | 27,262,976 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 131,072 | 46 | 6,029,312 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 2,097,152 | 54 | 113,246,208 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | prover | 0 | 1 | 59 | 59 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 0 | 262,144 | 66 | 17,301,504 | 
| app_proof | 27 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 2 | 38 | 76 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | prover | 0 | 131,072 | 33 | 4,325,376 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 524,288 | 46 | 24,117,248 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | prover | 0 | 1,048,576 | 34 | 35,651,584 | 
| app_proof | 31 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 32 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 33 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 34 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 1 | 43 | 43 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 1 | 65,536 | 150 | 9,830,400 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 1,048,576 | 456 | 478,150,656 | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | prover | 1 | 1,048,576 | 25 | 26,214,400 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 65,536 | 17 | 1,114,112 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 262,144 | 24 | 6,291,456 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 262,144 | 32 | 8,388,608 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 524,288 | 26 | 13,631,488 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 65,536 | 46 | 3,014,656 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 2,097,152 | 54 | 113,246,208 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | prover | 1 | 4 | 59 | 236 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 1 | 131,072 | 66 | 8,650,752 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | prover | 1 | 65,536 | 33 | 2,162,688 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 1 | 524,288 | 46 | 24,117,248 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | prover | 1 | 1,048,576 | 34 | 35,651,584 | 
| app_proof | 31 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 33 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 34 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- | --- |
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | height | 0 | 1 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 0 | 104,658 | 26,414 | 78,493,500 | 19,810,500 | 62,794,800 | 15,848,400 | 15,698,700 | 3,962,100 | 322,477,463 | 81,388,137 | 8,895,930 | 2,245,190 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 1,779,186 | 317,966 | 4,056,544,080 | 724,962,480 | 3,245,235,264 | 579,969,984 | 811,308,816 | 144,992,496 | 1,870,369,283 | 334,261,757 | 51,596,394 | 9,221,014 | 
| app_proof | 14 | Rv64HintStoreAir | 0 | 2 |  | 270 |  | 216 |  | 54 |  | 1,233 |  | 34 |  | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 0 | 1,051,651 | 1,045,501 | 65,728,188 | 65,343,812 | 105,165,100 | 104,550,100 | 26,291,275 | 26,137,525 | 609,957,580 | 606,390,580 | 16,826,416 | 16,728,016 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 104,672 | 26,400 | 4,448,560 | 1,122,000 | 7,117,696 | 1,795,200 | 1,779,424 | 448,800 | 53,121,040 | 13,398,000 | 1,465,408 | 369,600 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 313,983 | 210,305 | 18,838,980 | 12,618,300 | 30,142,368 | 20,189,280 | 7,535,592 | 5,047,320 | 170,728,257 | 114,353,343 | 4,709,745 | 3,154,575 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 421,914 | 102,374 | 18,986,130 | 4,606,830 | 30,377,808 | 7,370,928 | 7,594,452 | 1,842,732 | 183,532,590 | 44,532,690 | 5,062,968 | 1,228,488 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 420,277 | 104,011 | 33,622,160 | 8,320,880 | 53,795,456 | 13,313,408 | 13,448,864 | 3,328,352 | 213,290,578 | 52,785,582 | 5,883,878 | 1,456,154 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 590 | 434 | 30,975 | 22,785 | 49,560 | 36,456 | 12,390 | 9,114 | 85,550 | 62,930 | 2,360 | 1,736 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 629,660 | 418,916 | 40,927,900 | 27,229,540 | 65,484,640 | 43,567,264 | 16,371,160 | 10,891,816 | 251,076,925 | 167,042,755 | 6,926,260 | 4,608,076 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 104,659 | 26,413 | 12,035,785 | 3,037,495 | 19,257,256 | 4,859,992 | 4,814,314 | 1,214,998 | 83,465,553 | 21,064,367 | 2,302,498 | 581,086 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 2,097,004 | 148 | 283,095,540 | 19,980 | 452,952,864 | 31,968 | 113,238,216 | 7,992 | 1,900,409,875 | 134,125 | 52,425,100 | 3,700 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 1 |  | 148 |  | 236 |  | 59 |  | 798 |  | 22 |  | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 209,322 | 52,822 | 34,538,130 | 8,715,630 | 55,261,008 | 13,945,008 | 13,815,252 | 3,486,252 | 189,698,063 | 47,869,937 | 5,233,050 | 1,320,550 | 
| app_proof | 27 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 2 |  | 190 |  | 304 |  | 76 |  | 1,378 |  | 38 |  | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 104,660 | 26,412 | 8,634,450 | 2,178,990 | 13,815,120 | 3,486,384 | 3,453,780 | 871,596 | 72,084,575 | 18,191,265 | 1,988,540 | 501,828 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 523,304 | 984 | 60,179,960 | 113,160 | 96,287,936 | 181,056 | 24,071,984 | 45,264 | 455,274,480 | 856,080 | 12,559,296 | 23,616 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 746 | 278 | 119,360 | 44,480 | 95,488 | 35,584 | 23,872 | 8,896 | 108,170 | 40,310 | 2,984 | 1,112 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 1,048,229 | 347 | 89,099,465 | 29,495 | 142,559,144 | 47,192 | 35,639,786 | 11,798 | 759,966,025 | 251,575 | 20,964,580 | 6,940 | 
| app_proof | 31 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 32 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 33 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 1,336 | 712 | 1,002,000 | 534,000 | 1,603,200 | 854,400 | 400,800 | 213,600 | 48,430 | 25,810 | 1,336 | 712 | 
| app_proof | 34 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 |  | 108 |  | 172 |  | 43 |  | 1,124 |  | 31 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir | 1 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 1 | 59,183 | 6,353 | 44,387,250 | 4,764,750 | 35,509,800 | 3,811,800 | 8,877,450 | 952,950 | 182,357,619 | 19,575,181 | 5,030,555 | 540,005 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 1,006,111 | 42,465 | 2,293,933,080 | 96,820,200 | 1,835,146,464 | 77,456,160 | 458,786,616 | 19,364,040 | 1,057,674,189 | 44,641,331 | 29,177,219 | 1,231,485 | 
| app_proof | 15 | VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<4, 16> | 1 | 594,662 | 453,914 | 37,166,375 | 28,369,625 | 59,466,200 | 45,391,400 | 14,866,550 | 11,347,850 | 344,903,960 | 263,270,120 | 9,514,592 | 7,262,624 | 
| app_proof | 16 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 59,186 | 6,350 | 2,515,405 | 269,875 | 4,024,648 | 431,800 | 1,006,162 | 107,950 | 30,036,895 | 3,222,625 | 828,604 | 88,900 | 
| app_proof | 17 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 177,553 | 84,591 | 10,653,180 | 5,075,460 | 17,045,088 | 8,120,736 | 4,261,272 | 2,030,184 | 96,544,444 | 45,996,356 | 2,663,295 | 1,268,865 | 
| app_proof | 18 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 238,582 | 23,562 | 10,736,190 | 1,060,290 | 17,177,904 | 1,696,464 | 4,294,476 | 424,116 | 103,783,170 | 10,249,470 | 2,862,984 | 282,744 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 237,659 | 24,485 | 19,012,720 | 1,958,800 | 30,420,352 | 3,134,080 | 7,605,088 | 783,520 | 120,611,943 | 12,426,137 | 3,327,226 | 342,790 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 620 | 404 | 32,550 | 21,210 | 52,080 | 33,936 | 13,020 | 8,484 | 89,900 | 58,580 | 2,480 | 1,616 | 
| app_proof | 20 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 356,038 | 168,250 | 23,142,470 | 10,936,250 | 37,027,952 | 17,498,000 | 9,256,988 | 4,374,500 | 141,970,153 | 67,089,687 | 3,916,418 | 1,850,750 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 59,198 | 6,338 | 6,807,770 | 728,870 | 10,892,432 | 1,166,192 | 2,723,108 | 291,548 | 47,210,405 | 5,054,555 | 1,302,356 | 139,436 | 
| app_proof | 22 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 1,185,603 | 911,549 | 160,056,405 | 123,059,115 | 256,090,248 | 196,894,584 | 64,022,562 | 49,223,646 | 1,074,452,719 | 826,091,281 | 29,640,075 | 22,788,725 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 1 | 3 | 1 | 443 | 147 | 708 | 236 | 177 | 59 | 2,393 | 797 | 66 | 22 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 1 | 118,416 | 12,656 | 19,538,640 | 2,088,240 | 31,261,824 | 3,341,184 | 7,815,456 | 835,296 | 107,314,500 | 11,469,500 | 2,960,400 | 316,400 | 
| app_proof | 28 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 1 | 59,183 | 6,353 | 4,882,598 | 524,122 | 7,812,156 | 838,596 | 1,953,039 | 209,649 | 40,762,292 | 4,375,628 | 1,124,477 | 120,707 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | 295,936 | 228,352 | 34,032,640 | 26,260,480 | 54,452,224 | 42,016,768 | 13,613,056 | 10,504,192 | 257,464,320 | 198,666,240 | 7,102,464 | 5,480,448 | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 750 | 274 | 120,000 | 43,840 | 96,000 | 35,072 | 24,000 | 8,768 | 108,750 | 39,730 | 3,000 | 1,096 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 1 | 592,758 | 455,818 | 50,384,430 | 38,744,530 | 80,615,088 | 61,991,248 | 20,153,772 | 15,497,812 | 429,749,550 | 330,468,050 | 11,855,160 | 9,116,360 | 
| app_proof | 31 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 33 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 1,370 | 678 | 1,027,500 | 508,500 | 1,644,000 | 813,600 | 411,000 | 203,400 | 49,663 | 24,577 | 1,370 | 678 | 
| app_proof | 34 | VariableRangeCheckerAir | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 22 | 356 | 21 | 5 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 11 | 185 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 149 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 112 | 1,022 | 112 | 34 | 17 | 2 | 7 | 7 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,749 | 334 | 119 | 0 | 0 | 109 | 36 | 35 | 52 | 20 | 0 | 104 | 90 | 13 | 3 | 9 | 119 | 119 | 109 | 0 | 3 | 19 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 173 | 47 | 0 | 0 | 68 | 25 | 25 | 28 | 14 | 0 | 57 | 48 | 8 | 1 | 6 | 47 | 47 | 68 | 0 | 3 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 139 | 32 | 0 | 0 | 62 | 23 | 23 | 24 | 13 | 0 | 43 | 36 | 7 | 1 | 5 | 33 | 32 | 62 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 237,929,016 | 909 | 380 | 0 | 0 | 232 | 113 | 112 | 67 | 52 | 0 | 295 | 257 | 38 | 20 | 17 | 381 | 380 | 232 | 0 | 12 | 51 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 15,094,533 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 9 | 0 | 9 | 9 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,303,019,770 | 4,776 | 1,362 | 0 | 0 | 2,503 | 1,379 | 1,378 | 509 | 615 | 1 | 910 | 688 | 221 | 139 | 82 | 1,362 | 1,362 | 2,503 | 0 | 7 | 613 | 0 | 0 | 
| app_proof | prover | 1 | 740,851,960 | 2,652 | 594 | 0 | 0 | 1,535 | 740 | 739 | 275 | 519 | 0 | 522 | 402 | 119 | 72 | 46 | 594 | 594 | 1,535 | 0 | 8 | 518 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 239,755,658 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 163,602,786 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 133 | 1 | 176 | 5,303 | 176 | 133 | 13,211,583,828 | 2,431,104,612 | 15,642,688,440 | 2,097,152 | 0 | 4 | 215 | 7,134,000 | 33.62 | 
| app_proof | 1 | 41 | 1 | 71 | 3,053 | 71 | 41 | 7,501,802,352 | 1,396,518,993 | 8,898,321,345 | 2,097,152 | 0 | 2 | 287 | 4,033,961 | 37.62 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 9 | 0 | 9 | 9 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/f25fa657ef952641bd5dc1983b3ce3cafb5a5af6

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28953253713)
