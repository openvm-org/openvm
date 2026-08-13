| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  3.08 |  3.08 |  3.08 |
| app_proof |  2.19 |  2.19 |  2.19 |
| leaf |  0.46 |  0.46 |  0.46 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.13 |  0.13 |  0.13 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,169 |  2,169 |  2,169 |  2,169 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  21 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  90.55 | -          |  90.55 |  90.55 |
| `set_initial_memory_time_ms` |  55 |  55 |  55 |  55 |
| `execute_preflight_insns` |  1,979,971 |  1,979,971 |  1,979,971 |  1,979,971 |
| `execute_preflight_time_ms` |  75 |  75 |  75 |  75 |
| `execute_preflight_insn_mi/s` |  26.40 | -          |  26.36 |  26.36 |
| `postflight_time_ms  ` |  67 |  67 |  67 |  67 |
| `postflight_memory_chronology_time_ms` |  6 |  6 |  6 |  6 |
| `postflight_program_index_time_ms` |  0 |  0 |  0 |  0 |
| `trace_gen_time_ms   ` |  68 |  68 |  68 |  68 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,887 |  1,887 |  1,887 |  1,887 |
| `prover.main_trace_commit_time_ms` |  512 |  512 |  512 |  512 |
| `prover.rap_constraints_time_ms` |  931 |  931 |  931 |  931 |
| `prover.openings_time_ms` |  443 |  443 |  443 |  443 |
| `prover.rap_constraints.logup_gkr_time_ms` |  111 |  111 |  111 |  111 |
| `prover.rap_constraints.round0_time_ms` |  628 |  628 |  628 |  628 |
| `prover.rap_constraints.mle_rounds_time_ms` |  191 |  191 |  191 |  191 |
| `prover.openings.stacked_reduction_time_ms` |  86 |  86 |  86 |  86 |
| `prover.openings.stacked_reduction.round0_time_ms` |  47 |  47 |  47 |  47 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.whir_time_ms` |  357 |  357 |  357 |  357 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  464 |  464 |  464 |  464 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  110 |  110 |  110 |  110 |
| `generate_blob_total_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  354 |  354 |  354 |  354 |
| `prover.main_trace_commit_time_ms` |  98 |  98 |  98 |  98 |
| `prover.rap_constraints_time_ms` |  164 |  164 |  164 |  164 |
| `prover.openings_time_ms` |  91 |  91 |  91 |  91 |
| `prover.rap_constraints.logup_gkr_time_ms` |  58 |  58 |  58 |  58 |
| `prover.rap_constraints.round0_time_ms` |  66 |  66 |  66 |  66 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  74 |  74 |  74 |  74 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  196 |  196 |  196 |  196 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  17 |  17 |  17 |  17 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  178 |  178 |  178 |  178 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  79 |  79 |  79 |  79 |
| `prover.openings_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  40 |  40 |  40 |  40 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  126 |  126 |  126 |  126 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  114 |  114 |  114 |  114 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  37 |  37 |  37 |  37 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  29 |  29 |  29 |  29 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  107 |  107 |  107 |  107 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
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

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/983214f682bdd8418c6760a2c5f148acedb08fd9/kitchen_sink-983214f682bdd8418c6760a2c5f148acedb08fd9.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 11.94 | app_proof.prover..0 |
| prover.rap_constraints | 8.93 | app_proof.prover..0 |
| prover.openings | 8.09 | app_proof.prover..0 |
| prover.merkle_tree | 8.09 | app_proof.prover..0 |
| prover.prove_whir_opening | 8.09 | app_proof.prover..0 |
| prover.rs_code_matrix | 8.08 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 7.00 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 7.00 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 6.98 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 6.98 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 6.97 | app_proof.prover..0 |
| prover.gkr_input_evals | 6.97 | app_proof.prover..0 |
| postflight | 4.49 | app_proof..0 |
| generate mem proving ctxs | 4.32 | app_proof..0 |
| tracegen | 4.32 | app_proof..0 |
| set initial memory | 4.16 | app_proof..0 |
| prover.before_gkr_input_evals | 4.11 | app_proof.prover..0 |
| tracegen.pow_checker | 0.82 | leaf.0 |
| tracegen.exp_bits_len | 0.82 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.82 | leaf.0 |
| tracegen.whir_folding | 0.75 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.75 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.75 | leaf.0 |
| tracegen.proof_shape | 0.66 | leaf.0 |
| tracegen.range_checker | 0.66 | leaf.0 |
| tracegen.public_values | 0.66 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 551 | 267,407 | 229,532 | 1 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 9 | 11 | 3 | 
| 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 480 | 280 | 3 | 
| 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 101 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 525 | 257 | 3 | 
| 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 584 | 286 | 3 | 
| 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 392 | 190 | 3 | 
| 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 396 | 194 | 3 | 
| 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 268 | 130 | 3 | 
| 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 231 | 117 | 3 | 
| 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 167 | 85 | 3 | 
| 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 10 | 11 | 2 | 
| 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 79 | 169 | 3 | 
| 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 301 | 171 | 3 | 
| 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 205 | 123 | 3 | 
| 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 905 | 513 | 3 | 
| 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 59 | 117 | 3 | 
| 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 207 | 117 | 3 | 
| 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 143 | 85 | 3 | 
| 43 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 112 | 322 | 3 | 
| 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 111 | 597 | 3 | 
| 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 142 | 16 | 3 | 
| 46 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 56 | 69 | 3 | 
| 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 53 | 31 | 3 | 
| 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 81 | 71 | 3 | 
| 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 110 | 19 | 3 | 
| 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 777 | 381 | 3 | 
| 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 94 | 50 | 3 | 
| 51 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 52 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 53 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 54 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 181 | 39 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 101 | 23 | 3 | 
| 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 480 | 280 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 135 | 27 | 3 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 408 | 87 | 3 | 
| 64 | RevealAir |  | 25 | 3 | 2 | 
| 65 | HintStoreAir | 1 | 18 | 15 | 3 | 
| 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 68 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 525 | 257 | 3 | 
| 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 12 | 4 | 3 | 
| 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 8 | 3 | 
| 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 28 | 9 | 3 | 
| 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 27 | 12 | 3 | 
| 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 26 | 9 | 3 | 
| 79 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 25 | 12 | 3 | 
| 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 480 | 280 | 3 | 
| 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 26 | 13 | 3 | 
| 81 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 25 | 9 | 3 | 
| 82 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 24 | 12 | 3 | 
| 83 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 25 | 13 | 3 | 
| 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 8 | 3 | 
| 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 11 | 3 | 
| 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 12 | 3 | 
| 87 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 525 | 257 | 3 | 
| 90 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 91 | VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 92 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 93 | VmAirWrapper<BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 98 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 99 | PhantomAir |  | 3 | 1 | 2 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 59 |  |  |  | 314 |  | 
| app_proof | 0 |  |  |  | 4 |  |  | 
| internal_for_leaf |  |  |  | 196 |  |  | 196 | 
| internal_recursive.0 |  |  |  | 126 |  |  | 126 | 
| internal_recursive.1 |  |  |  | 107 |  |  | 107 | 
| leaf |  |  | 464 |  |  |  | 464 | 

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
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 5 | 
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
| app_proof | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 0 | 
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
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 8 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 5 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 1 | 
| app_proof | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | XorinVmAir |  | 0 | 0 | 

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
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 71 | 71 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 34 | 34 | 
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

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 8,192 | 11 | 90,112 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 7 | 14 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 548 | 2,192 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 101 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 642 | 642 | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 758 | 1,516 | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 566 | 1,132 | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 514 | 1,028 | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 386 | 772 | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 4 | 116 | 464 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 293 | 18,752 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 229 | 14,656 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 116 | 232 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 4,096 | 39 | 159,744 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 205 | 410 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover |  | 0 | 8 | 160 | 1,280 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 391 | 782 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 295 | 1,180 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 116 | 232 | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 205 | 410 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 116 | 928 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 8,192 | 33 | 270,336 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 205 | 820 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 116 | 232 | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 205 | 410 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 116 | 928 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 205 | 820 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 116 | 232 | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 205 | 410 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 1,005 | 4,020 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 116 | 928 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 269 | 538 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 205 | 820 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | prover |  | 0 | 512 | 188 | 96,256 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover |  | 0 | 256 | 170 | 43,520 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover |  | 0 | 256 | 91 | 23,296 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover |  | 0 | 512 | 127 | 65,024 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | prover |  | 0 | 512 | 172 | 88,064 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 950 | 950 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | prover |  | 0 | 512 | 123 | 62,976 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 256 | 41 | 10,496 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover |  | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover |  | 0 | 32,768 | 151 | 4,947,968 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 548 | 2,192 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover |  | 0 | 16,384 | 285 | 4,669,440 | 
| app_proof | 62 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover |  | 0 | 16,384 | 597 | 9,781,248 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 131,072 | 35 | 4,587,520 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 256 | 28 | 7,168 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 52 | 3,407,872 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 642 | 642 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 524,288 | 24 | 12,582,912 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 32,768 | 16 | 524,288 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 131,072 | 23 | 3,014,656 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 131,072 | 31 | 4,063,232 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 25 | 6,553,600 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 524,288 | 41 | 21,495,808 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 41 | 21,495,808 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 2,048 | 39 | 79,872 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 548 | 2,192 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 32,768 | 40 | 1,310,720 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 16,384 | 30 | 491,520 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 8,192 | 30 | 245,760 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 2,048 | 31 | 63,488 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 512 | 45 | 23,040 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1,024 | 23 | 23,552 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 642 | 642 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 128 | 34 | 4,352 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 32,768 | 29 | 950,272 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 43 | 1,409,024 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 262,144 | 30 | 7,864,320 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 6,504 | 1,688 | 178,860 | 46,420 | 286,176 | 74,272 | 71,544 | 18,568 | 235,770 | 61,190 | 6,504 | 1,688 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 70 |  | 56 |  | 14 |  | 653 |  | 18 |  | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 4,110 | 1,370 | 6,576 | 2,192 | 1,644 | 548 | 52,200 | 17,400 | 1,440 | 480 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 4,456 | 3,736 | 3,342,000 | 2,802,000 | 5,347,200 | 4,483,200 | 1,336,800 | 1,120,800 | 161,530 | 135,430 | 4,456 | 3,736 | 
| app_proof | 101 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,605 |  | 2,568 |  | 642 |  | 19,032 |  | 525 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 3,790 |  | 6,064 |  | 1,516 |  | 42,340 |  | 1,168 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,830 |  | 4,528 |  | 1,132 |  | 28,420 |  | 784 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,570 |  | 4,112 |  | 1,028 |  | 28,710 |  | 792 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,930 |  | 3,088 |  | 772 |  | 19,430 |  | 536 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 4 |  | 1,160 |  | 1,856 |  | 464 |  | 8,555 |  | 236 |  | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 46,148 | 732 | 73,836 | 1,172 | 18,459 | 293 | 527,547 | 8,373 | 14,553 | 231 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 36,068 | 572 | 57,708 | 916 | 14,427 | 229 | 381,387 | 6,053 | 10,521 | 167 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 4,278 |  | 118 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 2,059 | 2,037 | 200,753 | 198,607 | 321,204 | 317,772 | 80,301 | 79,443 | 746,388 | 738,412 | 20,590 | 20,370 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,025 |  | 1,640 |  | 410 |  | 10,368 |  | 286 |  | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 7 | 1 | 2,800 | 400 | 4,480 | 640 | 1,120 | 160 | 20,047 | 2,863 | 553 | 79 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,955 |  | 3,128 |  | 782 |  | 21,823 |  | 602 |  | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,213 | 737 | 3,540 | 1,180 | 885 | 295 | 22,294 | 7,431 | 615 | 205 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 4,278 |  | 118 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,025 |  | 1,640 |  | 410 |  | 10,368 |  | 286 |  | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 14,972 | 2,138 | 413 | 59 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 4,356 | 3,836 | 718,740 | 632,940 | 574,992 | 506,352 | 143,748 | 126,588 | 631,620 | 556,220 | 17,424 | 15,344 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,538 | 512 | 2,460 | 820 | 615 | 205 | 15,552 | 5,183 | 429 | 143 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 4,278 |  | 118 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,025 |  | 1,640 |  | 410 |  | 10,368 |  | 286 |  | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 14,972 | 2,138 | 413 | 59 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,538 | 512 | 2,460 | 820 | 615 | 205 | 15,552 | 5,183 | 429 | 143 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 580 |  | 928 |  | 232 |  | 4,278 |  | 118 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,025 |  | 1,640 |  | 410 |  | 10,368 |  | 286 |  | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 7,538 | 2,512 | 12,060 | 4,020 | 3,015 | 1,005 | 98,419 | 32,806 | 2,715 | 905 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,030 | 290 | 3,248 | 464 | 812 | 116 | 14,972 | 2,138 | 413 | 59 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,345 |  | 2,152 |  | 538 |  | 15,008 |  | 414 |  | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,538 | 512 | 2,460 | 820 | 615 | 205 | 15,552 | 5,183 | 429 | 143 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 400 | 112 | 188,000 | 52,640 | 300,800 | 84,224 | 75,200 | 21,056 | 1,609,500 | 450,660 | 44,400 | 12,432 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 200 | 56 | 85,000 | 23,800 | 136,000 | 38,080 | 34,000 | 9,520 | 1,029,500 | 288,260 | 28,400 | 7,952 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 200 | 56 | 45,500 | 12,740 | 72,800 | 20,384 | 18,200 | 5,096 | 384,250 | 107,590 | 10,600 | 2,968 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 295 | 217 | 93,663 | 68,897 | 149,860 | 110,236 | 37,465 | 27,559 | 866,194 | 637,166 | 23,895 | 17,577 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 400 | 112 | 172,000 | 48,160 | 275,200 | 77,056 | 68,800 | 19,264 | 1,595,000 | 446,600 | 44,000 | 12,320 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,375 |  | 3,800 |  | 950 |  | 28,167 |  | 777 |  | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 400 | 112 | 123,000 | 34,440 | 196,800 | 55,104 | 49,200 | 13,776 | 1,363,000 | 381,640 | 37,600 | 10,528 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 205 | 51 | 21,013 | 5,227 | 33,620 | 8,364 | 8,405 | 2,091 | 230,369 | 57,311 | 6,355 | 1,581 | 
| app_proof | 56 | RangeTupleCheckerAir<2> |  | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> |  | 0 | 20,100 | 12,668 | 15,175,500 | 9,564,340 | 12,140,400 | 7,651,472 | 3,035,100 | 1,912,868 | 73,591,125 | 46,380,715 | 2,030,100 | 1,279,468 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 4,110 | 1,370 | 6,576 | 2,192 | 1,644 | 548 | 52,200 | 17,400 | 1,440 | 480 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 341,700 | 182,588 | 779,076,000 | 416,300,640 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 359,212,125 | 191,945,635 | 9,909,300 | 5,295,052 | 
| app_proof | 61 | KeccakfOpAir |  | 0 | 9,470 | 6,914 | 6,747,375 | 4,926,225 | 10,795,800 | 7,881,960 | 2,698,950 | 1,970,490 | 46,343,813 | 33,835,387 | 1,278,450 | 933,390 | 
| app_proof | 62 | KeccakfPermAir |  | 0 | 227,280 | 34,864 | 2,993,277,600 | 459,158,880 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 16,477,800 | 2,527,640 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir |  | 0 | 9,458 | 6,926 | 14,116,065 | 10,337,055 | 22,585,704 | 16,539,288 | 5,646,426 | 4,134,822 | 139,883,820 | 102,435,540 | 3,858,864 | 2,825,808 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 126,459 | 4,613 | 11,065,163 | 403,637 | 17,704,260 | 645,820 | 4,426,065 | 161,455 | 91,682,775 | 3,344,425 | 2,529,180 | 92,260 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 233 | 23 | 16,310 | 1,610 | 26,096 | 2,576 | 6,524 | 644 | 118,248 | 11,672 | 3,262 | 322 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 55,501 | 10,035 | 7,215,130 | 1,304,550 | 11,544,208 | 2,087,280 | 2,886,052 | 521,820 | 38,226,314 | 6,911,606 | 1,054,519 | 190,665 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,605 |  | 2,568 |  | 642 |  | 19,032 |  | 525 |  | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 319,639 | 204,649 | 19,178,340 | 12,278,940 | 30,685,344 | 19,646,304 | 7,671,336 | 4,911,576 | 185,390,620 | 118,696,420 | 5,114,224 | 3,274,384 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 31,960 | 808 | 1,278,400 | 32,320 | 2,045,440 | 51,712 | 511,360 | 12,928 | 13,902,600 | 351,480 | 383,520 | 9,696 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 84,015 | 47,057 | 4,830,863 | 2,705,777 | 7,729,380 | 4,329,244 | 1,932,345 | 1,082,311 | 45,683,157 | 25,587,243 | 1,260,225 | 705,855 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 103,815 | 27,257 | 4,671,675 | 1,226,565 | 7,474,680 | 1,962,504 | 1,868,670 | 490,626 | 45,159,525 | 11,856,795 | 1,245,780 | 327,084 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 103,059 | 28,013 | 7,987,073 | 2,171,007 | 12,779,316 | 3,473,612 | 3,194,829 | 868,403 | 52,302,443 | 14,216,597 | 1,442,826 | 392,182 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 178,175 | 83,969 | 11,135,938 | 5,248,062 | 17,817,500 | 8,396,900 | 4,454,375 | 2,099,225 | 71,047,282 | 33,482,638 | 1,959,925 | 923,659 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 306,998 | 217,290 | 31,467,295 | 22,272,225 | 50,347,672 | 35,635,560 | 12,586,918 | 8,908,890 | 311,602,970 | 220,549,350 | 8,595,944 | 6,084,120 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 290,210 | 234,078 | 29,746,525 | 23,992,995 | 47,594,440 | 38,388,792 | 11,898,610 | 9,597,198 | 284,043,038 | 229,103,842 | 7,835,670 | 6,320,106 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 1,618 | 430 | 157,755 | 41,925 | 252,408 | 67,080 | 63,102 | 16,770 | 1,524,965 | 405,275 | 42,068 | 11,180 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 4,110 | 1,370 | 6,576 | 2,192 | 1,644 | 548 | 52,200 | 17,400 | 1,440 | 480 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 22,741 | 10,027 | 2,274,100 | 1,002,700 | 3,638,560 | 1,604,320 | 909,640 | 401,080 | 21,433,393 | 9,450,447 | 591,266 | 260,702 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 9,241 | 7,143 | 693,075 | 535,725 | 1,108,920 | 857,160 | 277,230 | 214,290 | 6,364,739 | 4,919,741 | 175,579 | 135,717 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 7,095 | 1,097 | 532,125 | 82,275 | 851,400 | 131,640 | 212,850 | 32,910 | 4,629,488 | 715,792 | 127,710 | 19,746 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 1,600 | 448 | 124,000 | 34,720 | 198,400 | 55,552 | 49,600 | 13,888 | 1,102,000 | 308,560 | 30,400 | 8,512 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 400 | 112 | 45,000 | 12,600 | 72,000 | 20,160 | 18,000 | 5,040 | 232,000 | 64,960 | 6,400 | 1,792 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 758 | 266 | 43,585 | 15,295 | 69,736 | 24,472 | 17,434 | 6,118 | 384,685 | 134,995 | 10,612 | 3,724 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,605 |  | 2,568 |  | 642 |  | 19,032 |  | 525 |  | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 104 | 24 | 8,840 | 2,040 | 14,144 | 3,264 | 3,536 | 816 | 67,860 | 15,660 | 1,872 | 432 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 21,133 | 11,635 | 1,532,143 | 843,537 | 2,451,428 | 1,349,660 | 612,857 | 337,415 | 13,023,212 | 7,170,068 | 359,261 | 197,795 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 17,481 | 15,287 | 1,879,208 | 1,643,352 | 3,006,732 | 2,629,364 | 751,683 | 657,341 | 14,574,784 | 12,745,536 | 402,063 | 351,601 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 256,381 | 5,763 | 19,228,575 | 432,225 | 30,765,720 | 691,560 | 7,691,430 | 172,890 | 176,582,414 | 3,969,266 | 4,871,239 | 109,497 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 17 | 196 | 17 | 5 | 0 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 11 | 126 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 110 | 464 | 109 | 17 | 6 | 18 | 21 | 21 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,983 | 178 | 46 | 0 | 0 | 79 | 29 | 27 | 36 | 13 | 0 | 51 | 40 | 10 | 2 | 7 | 46 | 46 | 79 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,771 | 114 | 20 | 0 | 0 | 56 | 21 | 20 | 23 | 11 | 0 | 37 | 29 | 7 | 1 | 6 | 20 | 20 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,867 | 96 | 15 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 27 | 20 | 7 | 1 | 5 | 15 | 15 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 167,484,223 | 354 | 98 | 0 | 0 | 164 | 66 | 65 | 39 | 58 | 0 | 91 | 74 | 17 | 7 | 9 | 98 | 98 | 164 | 0 | 3 | 57 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,589 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,385 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,361 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,645 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 6 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,052,244,290 | 1,887 | 512 | 0 | 64 | 931 | 628 | 626 | 191 | 111 | 1 | 443 | 357 | 86 | 47 | 39 | 512 | 512 | 931 | 0 | 1 | 109 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 88,286,726 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 2,169 | 21 | 1,979,971 | 90.55 | 0 | 2,196 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 55 | 2 | 68 | 2,169 | 3 | 55 | 0 | 67 | 0 | 6 | 0 | 10,060,812,264 | 2,580,864,816 | 12,641,677,080 | 2,097,152 | 2 | 0 | 0 | 3 | 64 | 75 | 1,979,971 | 26.36 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/983214f682bdd8418c6760a2c5f148acedb08fd9

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31752015199)
