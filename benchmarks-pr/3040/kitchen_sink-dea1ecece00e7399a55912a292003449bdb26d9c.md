| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  2.83 |  2.83 |  2.83 |
| app_proof |  1.95 |  1.95 |  1.95 |
| leaf |  0.46 |  0.46 |  0.46 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,926 |  1,926 |  1,926 |  1,926 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  22 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  89.29 | -          |  89.29 |  89.29 |
| `execute_preflight_insns` |  1,979,971 |  1,979,971 |  1,979,971 |  1,979,971 |
| `execute_preflight_time_ms` |  78 |  78 |  78 |  78 |
| `execute_preflight_insn_mi/s` |  31.13 | -          |  31.13 |  31.13 |
| `trace_gen_time_ms   ` |  49 |  49 |  49 |  49 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,797 |  1,797 |  1,797 |  1,797 |
| `prover.main_trace_commit_time_ms` |  521 |  521 |  521 |  521 |
| `prover.rap_constraints_time_ms` |  931 |  931 |  931 |  931 |
| `prover.openings_time_ms` |  344 |  344 |  344 |  344 |
| `prover.rap_constraints.logup_gkr_time_ms` |  111 |  111 |  111 |  111 |
| `prover.rap_constraints.round0_time_ms` |  628 |  628 |  628 |  628 |
| `prover.rap_constraints.mle_rounds_time_ms` |  191 |  191 |  191 |  191 |
| `prover.openings.stacked_reduction_time_ms` |  87 |  87 |  87 |  87 |
| `prover.openings.stacked_reduction.round0_time_ms` |  48 |  48 |  48 |  48 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.whir_time_ms` |  257 |  257 |  257 |  257 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  463 |  463 |  463 |  463 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  111 |  111 |  111 |  111 |
| `generate_blob_total_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  352 |  352 |  352 |  352 |
| `prover.main_trace_commit_time_ms` |  98 |  98 |  98 |  98 |
| `prover.rap_constraints_time_ms` |  163 |  163 |  163 |  163 |
| `prover.openings_time_ms` |  89 |  89 |  89 |  89 |
| `prover.rap_constraints.logup_gkr_time_ms` |  58 |  58 |  58 |  58 |
| `prover.rap_constraints.round0_time_ms` |  66 |  66 |  66 |  66 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  72 |  72 |  72 |  72 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  195 |  195 |  195 |  195 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  176 |  176 |  176 |  176 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  79 |  79 |  79 |  79 |
| `prover.openings_time_ms` |  50 |  50 |  50 |  50 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  39 |  39 |  39 |  39 |

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
| `prover.rap_constraints_time_ms` |  57 |  57 |  57 |  57 |
| `prover.openings_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  26 |  26 |  26 |  26 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  105 |  105 |  105 |  105 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  95 |  95 |  95 |  95 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  27 |  27 |  27 |  27 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/dea1ecece00e7399a55912a292003449bdb26d9c/kitchen_sink-dea1ecece00e7399a55912a292003449bdb26d9c.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 12.08 | app_proof.prover.0 |
| prover.rap_constraints | 9.00 | app_proof.prover.0 |
| prover.openings | 8.18 | app_proof.prover.0 |
| prover.merkle_tree | 8.18 | app_proof.prover.0 |
| prover.prove_whir_opening | 8.18 | app_proof.prover.0 |
| prover.rs_code_matrix | 8.17 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 7.12 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 7.12 | app_proof.prover.0 |
| prover.gkr_input_evals | 7.07 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 7.07 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 7.03 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 7.03 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 4.15 | app_proof.prover.0 |
| generate mem proving ctxs | 0.91 | app_proof.0 |
| set initial memory | 0.91 | app_proof.0 |
| tracegen.whir_final_poly_query_eval | 0.82 | leaf.0 |
| tracegen.pow_checker | 0.82 | leaf.0 |
| tracegen.exp_bits_len | 0.82 | leaf.0 |
| tracegen.whir_folding | 0.75 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.75 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.75 | leaf.0 |
| tracegen.public_values | 0.66 | leaf.0 |
| tracegen.proof_shape | 0.66 | leaf.0 |
| tracegen.range_checker | 0.66 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 508 | 267,239 | 228,279 | 24 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 548 | 286 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 356 | 190 | 3 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 372 | 194 | 3 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 244 | 130 | 3 | 
| 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 219 | 117 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 155 | 85 | 3 | 
| 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 67 | 170 | 3 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 283 | 171 | 3 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 187 | 123 | 3 | 
| 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 881 | 513 | 3 | 
| 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 100 | 355 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 99 | 660 | 3 | 
| 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 130 | 16 | 2 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 48 | 69 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 45 | 31 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 69 | 71 | 3 | 
| 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 98 | 19 | 2 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 741 | 381 | 2 | 
| 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 82 | 50 | 3 | 
| 51 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 52 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 54 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 149 | 39 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 85 | 23 | 3 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 110 | 27 | 2 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 357 | 92 | 3 | 
| 64 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 65 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> |  | 16 | 8 | 3 | 
| 66 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 67 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 68 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 70 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 71 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 29 | 13 | 3 | 
| 72 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 28 | 15 | 3 | 
| 73 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 27 | 13 | 3 | 
| 74 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 26 | 15 | 3 | 
| 75 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 27 | 16 | 3 | 
| 76 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 26 | 13 | 3 | 
| 77 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 25 | 15 | 3 | 
| 78 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 26 | 16 | 3 | 
| 79 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 11 | 3 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 80 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 13 | 3 | 
| 81 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 14 | 3 | 
| 82 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 83 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 84 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 85 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 86 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 87 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 18 | 16 | 3 | 
| 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 89 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 14 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 90 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 91 | PhantomAir |  | 3 | 1 | 2 | 
| 92 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 93 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 

| group | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | new_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 58 |  |  |  | 308 |  |  |  |  |  |  | 
| app_proof |  |  |  | 1,926 |  | 22 | 1,979,971 | 89.29 | 0 | 1,954 |  | 
| internal_for_leaf |  |  | 195 |  |  |  |  |  |  |  | 195 | 
| internal_recursive.0 |  |  | 123 |  |  |  |  |  |  |  | 123 | 
| internal_recursive.1 |  |  | 105 |  |  |  |  |  |  |  | 105 | 
| leaf |  | 463 |  |  |  |  |  |  |  |  | 463 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 36 | KeccakfOpAir | 0 | 3,030,400 | 
| app_proof | Sha2MainAir<Sha256Config> | 38 | Sha2MainAir<Sha256Config> | 0 | 5,467,200 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 9,212,160 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 32 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 | 14,064,116 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 8 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 15,382,860 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 11 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 21,568 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 3,552,064 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 1,488,588 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 14 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 27,200 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 27 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,552,400 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 28 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 4,946,832 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 29 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 4,152,600 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 57 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 60 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 63 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 66 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 69 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 72 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 78 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 81 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 592 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 75 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | 1,372 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 30 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 4,032,720 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 17 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 340,560 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 16 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 76,800 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 25 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 17,412,600 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 22 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 1,364,460 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 42 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 11,480 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 31 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 1,022,720 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 18 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 443,568 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 24 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 97,080 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 26 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 18,419,880 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 93 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | 972 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 87 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 89 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 91 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 84 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 960 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 85 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 960 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 92 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 | 76,800 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 52 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 37,600 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 55 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 56 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 58 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 59 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 61 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 62 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 64 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 65 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 67 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 68 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 70 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 71 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 76 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 77 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 79 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 12,096 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 80 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 12,096 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 73 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 792 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 74 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 528 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 82 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 672 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 83 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 672 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 86 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 88 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 90 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 50 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | 27,200 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 47 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 0 | 76,800 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 49 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | 56,640 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 53 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 | 76,800 | 
| app_proof | XorinVmAir | 34 | XorinVmAir | 0 | 6,204,448 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 |  | 2 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 |  | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 |  | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 |  | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 |  | 1 | 
| app_proof | XorinVmAir | 0 |  | 5 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 94 | 4 | 
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
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 32,768 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 32 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 272 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 128 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 50 | 6,400 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 23,040 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 32,768 | 63 | 2,064,384 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 524,288 | 60 | 31,457,280 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 32,768 | 16 | 524,288 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 184 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 262,144 | 89 | 23,330,816 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 114,688 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 253,952 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 34,816 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 11,796,480 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 128 | 29 | 3,712 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 64 | 24 | 1,536 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 4,224 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 1,312 | 
| leaf | 8 | Eq3bAir | 0 | prover | 524,288 | 25 | 13,107,200 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | Fp2MulDiv | 0 | 2 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | Fp2AddSub | 0 | 2 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 3 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 63 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 63 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 1 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | IS_EQ | 0 | 6 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 1 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 6 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 1 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 6 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 1 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 2 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 6 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 2 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 3 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | SLL | 0 | 200 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | SRL | 0 | 200 | 
| app_proof | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | MUL | 0 | 200 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | BEQ | 0 | 200 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | SLTU | 0 | 295 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | AND | 0 | 200 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | XOR | 0 | 200 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | ADD | 0 | 200 | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | SUB | 0 | 200 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 205 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 20,100 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 61 | KeccakfOpAir | KECCAKF | 0 | 9,470 | 
| app_proof | 63 | XorinVmAir | XORIN | 0 | 9,458 | 
| app_proof | 65 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | ADDI | 0 | 319,639 | 
| app_proof | 66 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 31,960 | 
| app_proof | 67 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 84,015 | 
| app_proof | 68 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 82,058 | 
| app_proof | 68 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 21,757 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGE | 0 | 18 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 30,573 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 72,468 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 70 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 123,385 | 
| app_proof | 70 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 54,790 | 
| app_proof | 71 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | STORED | 0 | 306,998 | 
| app_proof | 72 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | LOADD | 0 | 290,210 | 
| app_proof | 73 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | STOREW | 0 | 1,618 | 
| app_proof | 75 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | LOADW | 0 | 22,741 | 
| app_proof | 79 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | STOREB | 0 | 9,241 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 80 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | LOADBU | 0 | 7,095 | 
| app_proof | 81 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | LOADB | 0 | 1,600 | 
| app_proof | 83 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SRLW | 0 | 400 | 
| app_proof | 85 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 0 | 28,161 | 
| app_proof | 85 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 0 | 27,340 | 
| app_proof | 86 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 337 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | ADDW | 0 | 758 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | SUBW | 0 | 21,133 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 126,677 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 16,863 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | XOR | 0 | 400 | 
| app_proof | 89 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | ADD | 0 | 174,698 | 
| app_proof | 89 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | SUB | 0 | 81,683 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 757 | 1,514 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 2 | 565 | 1,130 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 513 | 1,026 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2 | 385 | 770 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 4 | 116 | 464 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 292 | 18,688 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 64 | 228 | 14,592 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 8,192 | 21 | 172,032 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover | 0 | 8 | 160 | 1,280 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 2 | 390 | 780 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover | 0 | 4 | 294 | 1,176 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 8,192 | 33 | 270,336 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 2 | 116 | 232 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 204 | 408 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 4 | 1,004 | 4,016 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 8 | 116 | 928 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 2 | 268 | 536 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 4 | 204 | 816 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | prover | 0 | 512 | 190 | 97,280 | 
| app_proof | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover | 0 | 256 | 169 | 43,264 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover | 0 | 256 | 90 | 23,040 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover | 0 | 512 | 126 | 64,512 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | prover | 0 | 512 | 171 | 87,552 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover | 0 | 1 | 949 | 949 | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | prover | 0 | 512 | 122 | 62,464 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover | 0 | 32,768 | 150 | 4,915,200 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 62 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover | 0 | 16,384 | 669 | 10,960,896 | 
| app_proof | 65 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | prover | 0 | 524,288 | 25 | 13,107,200 | 
| app_proof | 66 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 32,768 | 17 | 557,056 | 
| app_proof | 67 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 24 | 3,145,728 | 
| app_proof | 68 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 70 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 71 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover | 0 | 524,288 | 44 | 23,068,672 | 
| app_proof | 72 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover | 0 | 524,288 | 43 | 22,544,384 | 
| app_proof | 73 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover | 0 | 2,048 | 42 | 86,016 | 
| app_proof | 75 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover | 0 | 32,768 | 42 | 1,376,256 | 
| app_proof | 79 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover | 0 | 16,384 | 32 | 524,288 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 80 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover | 0 | 8,192 | 31 | 253,952 | 
| app_proof | 81 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover | 0 | 2,048 | 32 | 65,536 | 
| app_proof | 83 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | prover | 0 | 512 | 59 | 30,208 | 
| app_proof | 85 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 0 | 65,536 | 66 | 4,325,376 | 
| app_proof | 86 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 512 | 38 | 19,456 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | prover | 0 | 32,768 | 33 | 1,081,344 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 262,144 | 46 | 12,058,624 | 
| app_proof | 89 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover | 0 | 262,144 | 32 | 8,388,608 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 90 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 92 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 93 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 6,504 | 1,688 | 162,600 | 42,200 | 260,160 | 67,520 | 65,040 | 16,880 | 235,770 | 61,190 | 6,504 | 1,688 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,103 | 1,367 | 6,564 | 2,188 | 1,641 | 547 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 1 |  | 1,603 |  | 2,564 |  | 641 |  | 18,162 |  | 501 |  | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 2 |  | 3,785 |  | 6,056 |  | 1,514 |  | 39,730 |  | 1,096 |  | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 2 |  | 2,825 |  | 4,520 |  | 1,130 |  | 25,810 |  | 712 |  | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 2 |  | 2,565 |  | 4,104 |  | 1,026 |  | 26,970 |  | 744 |  | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 2 |  | 1,925 |  | 3,080 |  | 770 |  | 17,690 |  | 488 |  | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 4 |  | 1,160 |  | 1,856 |  | 464 |  | 7,395 |  | 204 |  | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 45,990 | 730 | 73,584 | 1,168 | 18,396 | 292 | 500,142 | 7,938 | 13,797 | 219 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 63 | 1 | 35,910 | 570 | 57,456 | 912 | 14,364 | 228 | 353,982 | 5,618 | 9,765 | 155 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 3,698 |  | 102 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 4,118 | 4,074 | 216,195 | 213,885 | 345,912 | 342,216 | 86,478 | 85,554 | 597,110 | 590,730 | 16,472 | 16,296 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,020 |  | 1,632 |  | 408 |  | 9,498 |  | 262 |  | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | 7 | 1 | 2,800 | 400 | 4,480 | 640 | 1,120 | 160 | 17,002 | 2,428 | 469 | 67 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 2 |  | 1,950 |  | 3,120 |  | 780 |  | 20,518 |  | 566 |  | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 3 | 1 | 2,205 | 735 | 3,528 | 1,176 | 882 | 294 | 20,337 | 6,778 | 561 | 187 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 3,698 |  | 102 |  | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,020 |  | 1,632 |  | 408 |  | 9,498 |  | 262 |  | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 4,344 | 3,848 | 716,760 | 634,920 | 573,408 | 507,936 | 143,352 | 126,984 | 629,880 | 557,960 | 17,376 | 15,392 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,530 | 510 | 2,448 | 816 | 612 | 204 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 3,698 |  | 102 |  | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,020 |  | 1,632 |  | 408 |  | 9,498 |  | 262 |  | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,530 | 510 | 2,448 | 816 | 612 | 204 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 3,698 |  | 102 |  | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,020 |  | 1,632 |  | 408 |  | 9,498 |  | 262 |  | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | 3 | 1 | 7,530 | 2,510 | 12,048 | 4,016 | 3,012 | 1,004 | 95,809 | 31,936 | 2,643 | 881 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 |  | 2,144 |  | 536 |  | 14,138 |  | 390 |  | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 3 | 1 | 1,530 | 510 | 2,448 | 816 | 612 | 204 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 | 400 | 112 | 190,000 | 53,200 | 304,000 | 85,120 | 76,000 | 21,280 | 1,435,500 | 401,940 | 39,600 | 11,088 | 
| app_proof | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 200 | 56 | 84,500 | 23,660 | 135,200 | 37,856 | 33,800 | 9,464 | 942,500 | 263,900 | 26,000 | 7,280 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | 200 | 56 | 45,000 | 12,600 | 72,000 | 20,160 | 18,000 | 5,040 | 326,250 | 91,350 | 9,000 | 2,520 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | 295 | 217 | 92,925 | 68,355 | 148,680 | 109,368 | 37,170 | 27,342 | 737,869 | 542,771 | 20,355 | 14,973 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 | 400 | 112 | 171,000 | 47,880 | 273,600 | 76,608 | 68,400 | 19,152 | 1,421,000 | 397,880 | 39,200 | 10,976 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 1 |  | 2,373 |  | 3,796 |  | 949 |  | 26,862 |  | 741 |  | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 0 | 400 | 112 | 122,000 | 34,160 | 195,200 | 54,656 | 48,800 | 13,664 | 1,189,000 | 332,920 | 32,800 | 9,184 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 205 | 51 | 22,038 | 5,482 | 35,260 | 8,772 | 8,815 | 2,193 | 230,369 | 57,311 | 6,355 | 1,581 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 0 | 20,100 | 12,668 | 15,075,000 | 9,501,000 | 12,060,000 | 7,600,800 | 3,015,000 | 1,900,200 | 61,933,125 | 39,033,275 | 1,708,500 | 1,076,780 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,103 | 1,367 | 6,564 | 2,188 | 1,641 | 547 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 341,700 | 182,588 | 779,076,000 | 416,300,640 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 359,212,125 | 191,945,635 | 9,909,300 | 5,295,052 | 
| app_proof | 61 | KeccakfOpAir | 0 | 9,470 | 6,914 | 6,723,700 | 4,908,940 | 10,757,920 | 7,854,304 | 2,689,480 | 1,963,576 | 37,761,625 | 27,569,575 | 1,041,700 | 760,540 | 
| app_proof | 62 | KeccakfPermAir | 0 | 227,280 | 34,864 | 2,993,277,600 | 459,158,880 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 16,477,800 | 2,527,640 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir | 0 | 9,458 | 6,926 | 15,818,505 | 11,583,735 | 25,309,608 | 18,533,976 | 6,327,402 | 4,633,494 | 122,398,343 | 89,631,097 | 3,376,506 | 2,472,582 | 
| app_proof | 65 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16> | 0 | 319,639 | 204,649 | 19,977,438 | 12,790,562 | 31,963,900 | 20,464,900 | 7,990,975 | 5,116,225 | 185,390,620 | 118,696,420 | 5,114,224 | 3,274,384 | 
| app_proof | 66 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 31,960 | 808 | 1,358,300 | 34,340 | 2,173,280 | 54,944 | 543,320 | 13,736 | 16,219,700 | 410,060 | 447,440 | 11,312 | 
| app_proof | 67 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 84,015 | 47,057 | 5,040,900 | 2,823,420 | 8,065,440 | 4,517,472 | 2,016,360 | 1,129,368 | 45,683,157 | 25,587,243 | 1,260,225 | 705,855 | 
| app_proof | 68 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 103,815 | 27,257 | 4,671,675 | 1,226,565 | 7,474,680 | 1,962,504 | 1,868,670 | 490,626 | 45,159,525 | 11,856,795 | 1,245,780 | 327,084 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 103,059 | 28,013 | 8,244,720 | 2,241,040 | 13,191,552 | 3,585,664 | 3,297,888 | 896,416 | 52,302,443 | 14,216,597 | 1,442,826 | 392,182 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 1 |  | 1,603 |  | 2,564 |  | 641 |  | 18,162 |  | 501 |  | 
| app_proof | 70 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 178,175 | 83,969 | 11,581,375 | 5,457,985 | 18,530,200 | 8,732,776 | 4,632,550 | 2,183,194 | 71,047,282 | 33,482,638 | 1,959,925 | 923,659 | 
| app_proof | 71 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | 306,998 | 217,290 | 33,769,780 | 23,901,900 | 54,031,648 | 38,243,040 | 13,507,912 | 9,560,760 | 322,731,648 | 228,426,112 | 8,902,942 | 6,301,410 | 
| app_proof | 72 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | 290,210 | 234,078 | 31,197,575 | 25,163,385 | 49,916,120 | 40,261,416 | 12,479,030 | 10,065,354 | 294,563,150 | 237,589,170 | 8,125,880 | 6,554,184 | 
| app_proof | 73 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | 1,618 | 430 | 169,890 | 45,150 | 271,824 | 72,240 | 67,956 | 18,060 | 1,583,618 | 420,862 | 43,686 | 11,610 | 
| app_proof | 75 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | 22,741 | 10,027 | 2,387,805 | 1,052,835 | 3,820,488 | 1,684,536 | 955,122 | 421,134 | 22,257,754 | 9,813,926 | 614,007 | 270,729 | 
| app_proof | 79 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | 0 | 9,241 | 7,143 | 739,280 | 571,440 | 1,182,848 | 914,304 | 295,712 | 228,576 | 6,364,739 | 4,919,741 | 175,579 | 135,717 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,103 | 1,367 | 6,564 | 2,188 | 1,641 | 547 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 80 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | 0 | 7,095 | 1,097 | 549,863 | 85,017 | 879,780 | 136,028 | 219,945 | 34,007 | 4,629,488 | 715,792 | 127,710 | 19,746 | 
| app_proof | 81 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | 1,600 | 448 | 128,000 | 35,840 | 204,800 | 57,344 | 51,200 | 14,336 | 1,102,000 | 308,560 | 30,400 | 8,512 | 
| app_proof | 83 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 400 | 112 | 59,000 | 16,520 | 94,400 | 26,432 | 23,600 | 6,608 | 319,000 | 89,320 | 8,800 | 2,464 | 
| app_proof | 85 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 55,501 | 10,035 | 9,157,665 | 1,655,775 | 14,652,264 | 2,649,240 | 3,663,066 | 662,310 | 50,297,782 | 9,094,218 | 1,387,525 | 250,875 | 
| app_proof | 86 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 337 | 175 | 32,015 | 16,625 | 51,224 | 26,600 | 12,806 | 6,650 | 232,109 | 120,531 | 6,403 | 3,325 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | 21,891 | 10,877 | 1,806,008 | 897,352 | 2,889,612 | 1,435,764 | 722,403 | 358,941 | 14,283,878 | 7,097,242 | 394,038 | 195,786 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 143,940 | 118,204 | 16,553,100 | 13,593,460 | 26,484,960 | 21,749,536 | 6,621,240 | 5,437,384 | 125,227,800 | 102,837,480 | 3,454,560 | 2,836,896 | 
| app_proof | 89 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | 256,381 | 5,763 | 20,510,480 | 461,040 | 32,816,768 | 737,664 | 8,204,192 | 184,416 | 176,582,414 | 3,969,266 | 4,871,239 | 109,497 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 1 |  | 1,603 |  | 2,564 |  | 641 |  | 18,162 |  | 501 |  | 
| app_proof | 90 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 92 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,444 | 3,748 | 3,333,000 | 2,811,000 | 5,332,800 | 4,497,600 | 1,333,200 | 1,124,400 | 161,095 | 135,865 | 4,444 | 3,748 | 
| app_proof | 93 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 195 | 17 | 6 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 123 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 105 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 111 | 463 | 110 | 16 | 6 | 18 | 22 | 22 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,981 | 176 | 46 | 0 | 0 | 79 | 28 | 27 | 36 | 13 | 0 | 50 | 39 | 10 | 2 | 7 | 46 | 46 | 79 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 112 | 20 | 0 | 0 | 57 | 21 | 20 | 23 | 12 | 0 | 34 | 26 | 7 | 1 | 6 | 20 | 20 | 57 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 95 | 14 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 27 | 19 | 7 | 1 | 5 | 15 | 14 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 167,484,221 | 352 | 98 | 0 | 0 | 163 | 66 | 64 | 39 | 58 | 0 | 89 | 72 | 17 | 7 | 9 | 98 | 98 | 163 | 0 | 3 | 57 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,643 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 6 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,064,822,140 | 1,797 | 520 | 0 | 0 | 931 | 628 | 627 | 191 | 111 | 1 | 344 | 257 | 87 | 48 | 39 | 521 | 520 | 931 | 0 | 1 | 109 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 90,960,126 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 2 | 49 | 1,926 | 49 | 0 | 10,130,125,596 | 2,662,485,684 | 12,792,611,280 | 2,097,152 | 0 | 4 | 78 | 1,979,971 | 31.13 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/dea1ecece00e7399a55912a292003449bdb26d9c

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29652392496)
