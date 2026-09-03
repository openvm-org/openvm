| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  3.18 |  3.18 |  3.18 |
| app_proof |  2.28 |  2.28 |  2.28 |
| leaf |  0.48 |  0.48 |  0.48 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,252 |  2,252 |  2,252 |  2,252 |
| `compile_metered_time_ms` |  0 |  0 |  0 |  0 |
| `execute_metered_time_ms` |  28 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  70.60 | -          |  70.60 |  70.60 |
| `set_initial_memory_time_ms` |  53 |  53 |  53 |  53 |
| `execute_preflight_insns` |  1,979,971 |  1,979,971 |  1,979,971 |  1,979,971 |
| `execute_preflight_time_ms` |  78 |  78 |  78 |  78 |
| `execute_preflight_insn_mi/s` |  25.38 | -          |  25.26 |  25.26 |
| `postflight_time_ms  ` |  72 |  72 |  72 |  72 |
| `postflight_memory_chronology_time_ms` |  7 |  7 |  7 |  7 |
| `postflight_program_index_time_ms` |  0 |  0 |  0 |  0 |
| `trace_gen_time_ms   ` |  70 |  70 |  70 |  70 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,962 |  1,962 |  1,962 |  1,962 |
| `prover.main_trace_commit_time_ms` |  495 |  495 |  495 |  495 |
| `prover.rap_constraints_time_ms` |  1,037 |  1,037 |  1,037 |  1,037 |
| `prover.openings_time_ms` |  430 |  430 |  430 |  430 |
| `prover.rap_constraints.logup_gkr_time_ms` |  127 |  127 |  127 |  127 |
| `prover.rap_constraints.round0_time_ms` |  712 |  712 |  712 |  712 |
| `prover.rap_constraints.mle_rounds_time_ms` |  196 |  196 |  196 |  196 |
| `prover.openings.stacked_reduction_time_ms` |  85 |  85 |  85 |  85 |
| `prover.openings.stacked_reduction.round0_time_ms` |  46 |  46 |  46 |  46 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.whir_time_ms` |  344 |  344 |  344 |  344 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  478 |  478 |  478 |  478 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  107 |  107 |  107 |  107 |
| `generate_blob_total_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  371 |  371 |  371 |  371 |
| `prover.main_trace_commit_time_ms` |  97 |  97 |  97 |  97 |
| `prover.rap_constraints_time_ms` |  185 |  185 |  185 |  185 |
| `prover.openings_time_ms` |  88 |  88 |  88 |  88 |
| `prover.rap_constraints.logup_gkr_time_ms` |  67 |  67 |  67 |  67 |
| `prover.rap_constraints.round0_time_ms` |  78 |  78 |  78 |  78 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  70 |  70 |  70 |  70 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  198 |  198 |  198 |  198 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  179 |  179 |  179 |  179 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  81 |  81 |  81 |  81 |
| `prover.openings_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  40 |  40 |  40 |  40 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  120 |  120 |  120 |  120 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  108 |  108 |  108 |  108 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  31 |  31 |  31 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  23 |  23 |  23 |  23 |

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
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/5bdc769320d126793f07740a98edbf95ce3ed8f6/kitchen_sink-5bdc769320d126793f07740a98edbf95ce3ed8f6.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 11.87 | app_proof.prover..0 |
| prover.rap_constraints | 8.89 | app_proof.prover..0 |
| prover.openings | 8.04 | app_proof.prover..0 |
| prover.prove_whir_opening | 8.04 | app_proof.prover..0 |
| prover.merkle_tree | 8.04 | app_proof.prover..0 |
| prover.rs_code_matrix | 8.03 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 6.96 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 6.96 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 6.90 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 6.90 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 6.86 | app_proof.prover..0 |
| prover.gkr_input_evals | 6.86 | app_proof.prover..0 |
| postflight | 4.48 | app_proof..0 |
| tracegen | 4.32 | app_proof..0 |
| generate mem proving ctxs | 4.32 | app_proof..0 |
| set initial memory | 4.16 | app_proof..0 |
| prover.before_gkr_input_evals | 4.08 | app_proof.prover..0 |
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
| 524 | 267,351 | 230,170 | 0 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 101 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
| 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 551 | 247 | 3 | 
| 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 359 | 151 | 3 | 
| 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 375 | 167 | 3 | 
| 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 247 | 103 | 3 | 
| 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 222 | 102 | 3 | 
| 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 158 | 70 | 3 | 
| 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 69 | 155 | 3 | 
| 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 286 | 150 | 3 | 
| 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 190 | 102 | 3 | 
| 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 883 | 487 | 3 | 
| 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 53 | 107 | 3 | 
| 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 198 | 102 | 3 | 
| 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 134 | 70 | 3 | 
| 43 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 103 | 307 | 3 | 
| 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 102 | 582 | 3 | 
| 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 133 | 1 | 2 | 
| 46 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 50 | 59 | 3 | 
| 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 47 | 21 | 3 | 
| 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 72 | 56 | 3 | 
| 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 101 | 4 | 2 | 
| 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 744 | 342 | 2 | 
| 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 85 | 35 | 3 | 
| 51 | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 52 | VmAirWrapper<MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 53 | VmAirWrapper<MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 54 | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 152 | 4 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 88 | 4 | 3 | 
| 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 111 | 1 | 2 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 359 | 34 | 3 | 
| 64 | RevealAir |  | 25 | 3 | 2 | 
| 65 | HintStoreAir | 1 | 18 | 12 | 3 | 
| 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 68 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
| 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 14 | 4 | 3 | 
| 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 15 | 11 | 3 | 
| 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 12 | 15 | 2 | 
| 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 27 | 7 | 3 | 
| 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 26 | 10 | 3 | 
| 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 25 | 7 | 3 | 
| 79 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 24 | 10 | 3 | 
| 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 466 | 262 | 3 | 
| 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 25 | 11 | 3 | 
| 81 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 24 | 7 | 3 | 
| 82 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 23 | 10 | 3 | 
| 83 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 24 | 11 | 3 | 
| 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 7 | 3 | 
| 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 87 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 504 | 230 | 2 | 
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
| agg_keygen |  | 61 |  |  |  | 327 |  | 
| app_proof | 0 |  |  |  | 0 |  |  | 
| internal_for_leaf |  |  |  | 198 |  |  | 198 | 
| internal_recursive.0 |  |  |  | 120 |  |  | 120 | 
| internal_recursive.1 |  |  |  | 108 |  |  | 108 | 
| leaf |  |  | 478 |  |  |  | 478 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | HintStoreAir |  | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 8 | 
| app_proof | KeccakfPermAir |  | 0 | 0 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 5 | 
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
| app_proof | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 4 | 
| app_proof | VmAirWrapper<MultWAdapterAir, DivRemCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 0 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 7 | 
| app_proof | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 
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
| app_proof | 0 | ProgramAir | 2 | Program |  | 0 | 208,128 | 54,016 | 6,504 | 1,688 | 
| app_proof | 1 | VmConnectorAir | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 1 | VmConnectorAir | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 1 | VmConnectorAir | 3 | VariableRange |  | 0 | 128 |  | 4 |  | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | Poseidon2Compression |  | 0 | 142,592 | 119,552 | 4,456 | 3,736 | 
| app_proof | 101 | VariableRangeCheckerAir | 3 | VariableRange |  | 0 | 8,388,608 |  | 262,144 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 |  | 156 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 30,080 |  | 940 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 |  | 156 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 17,792 |  | 556 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 |  | 108 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 20,352 |  | 636 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 |  | 108 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 12,160 |  | 380 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 256 |  | 8 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 2,816 |  | 88 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 128 |  | 4 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 3,584 |  | 112 |  | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 4,032 | 64 | 126 | 2 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 60,480 | 960 | 1,890 | 30 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 2,016 | 32 | 63 | 1 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 381,024 | 6,048 | 11,907 | 189 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 4,032 | 64 | 126 | 2 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 60,480 | 960 | 1,890 | 30 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 2,016 | 32 | 63 | 1 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 252,000 | 4,000 | 7,875 | 125 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | Memory |  | 0 | 263,552 | 260,736 | 8,236 | 8,148 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 4 | MemoryMerkle |  | 0 | 131,776 | 130,368 | 4,118 | 4,074 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 5 | Poseidon2Compression |  | 0 | 131,776 | 130,368 | 4,118 | 4,074 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 1 | Memory |  | 0 | 6,720 | 960 | 210 | 30 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 3 | VariableRange |  | 0 | 8,064 | 1,152 | 252 | 36 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,688 |  | 84 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 15,424 |  | 482 |  | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,032 | 1,344 | 126 | 42 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 13,920 | 4,640 | 435 | 145 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 4 | MemoryMerkle |  | 0 | 418,176 | 368,256 | 13,068 | 11,508 | 
| app_proof | 3 | MemoryMerkleAir<8> | 5 | Poseidon2Compression |  | 0 | 139,392 | 122,752 | 4,356 | 3,836 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 1,408 |  | 44 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 1,792 |  | 56 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 6,464 |  | 202 |  | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 4,992 | 1,664 | 156 | 52 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 79,488 | 26,496 | 2,484 | 828 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | Execution |  | 0 | 448 | 64 | 14 | 2 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 1 | Memory |  | 0 | 4,928 | 704 | 154 | 22 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 2 | Program |  | 0 | 224 | 32 | 7 | 1 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 3 | VariableRange |  | 0 | 6,272 | 896 | 196 | 28 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 128 |  | 4 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,920 |  | 60 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 64 |  | 2 |  | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 10,560 |  | 330 |  | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,880 | 960 | 90 | 30 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 9,696 | 3,232 | 303 | 101 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 3 | VariableRange |  | 0 | 883,200 | 247,296 | 27,600 | 7,728 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | Execution |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | Memory |  | 0 | 192,000 | 53,760 | 6,000 | 1,680 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 10 | RangeTuple |  | 0 | 204,800 | 57,344 | 6,400 | 1,792 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 2 | Program |  | 0 | 6,400 | 1,792 | 200 | 56 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 3 | VariableRange |  | 0 | 230,400 | 64,512 | 7,200 | 2,016 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 6 | BitwiseLookup |  | 0 | 204,800 | 57,344 | 6,400 | 1,792 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | Execution |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 1 | Memory |  | 0 | 128,000 | 35,840 | 4,000 | 1,120 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 2 | Program |  | 0 | 6,400 | 1,792 | 200 | 56 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 3 | VariableRange |  | 0 | 153,600 | 43,008 | 4,800 | 1,344 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | Execution |  | 0 | 18,880 | 13,888 | 590 | 434 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 1 | Memory |  | 0 | 283,200 | 208,320 | 8,850 | 6,510 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 2 | Program |  | 0 | 9,440 | 6,944 | 295 | 217 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 3 | VariableRange |  | 0 | 368,160 | 270,816 | 11,505 | 8,463 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 3 | VariableRange |  | 0 | 460,800 | 129,024 | 14,400 | 4,032 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 6 | BitwiseLookup |  | 0 | 409,600 | 114,688 | 12,800 | 3,584 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 2,496 |  | 78 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 21,216 |  | 663 |  | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 1 | Memory |  | 0 | 384,000 | 107,520 | 12,000 | 3,360 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | 3 | VariableRange |  | 0 | 665,600 | 186,368 | 20,800 | 5,824 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | Execution |  | 0 | 13,120 | 3,264 | 410 | 102 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 1 | Memory |  | 0 | 39,360 | 9,792 | 1,230 | 306 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 10 | RangeTuple |  | 0 | 52,480 | 13,056 | 1,640 | 408 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 2 | Program |  | 0 | 6,560 | 1,632 | 205 | 51 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 3 | VariableRange |  | 0 | 39,360 | 9,792 | 1,230 | 306 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 52,480 | 13,056 | 1,640 | 408 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | 10 | RangeTuple |  | 0 | 67,108,864 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 0 | Execution |  | 0 | 1,286,400 | 810,752 | 40,200 | 25,336 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 1 | Memory |  | 0 | 24,441,600 | 15,404,288 | 763,800 | 481,384 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 2 | Program |  | 0 | 643,200 | 405,376 | 20,100 | 12,668 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 3 | VariableRange |  | 0 | 28,300,800 | 17,836,544 | 884,400 | 557,392 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 8 | Sha2Block |  | 0 | 1,929,600 | 1,216,128 | 60,300 | 38,004 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 3 | VariableRange |  | 0 | 174,950,400 | 93,485,056 | 5,467,200 | 2,921,408 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 6 | BitwiseLookup |  | 0 | 87,475,200 | 46,742,528 | 2,733,600 | 1,460,704 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 8 | Sha2Block |  | 0 | 32,803,200 | 17,528,448 | 1,025,100 | 547,764 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 9 | Sha2SubAir |  | 0 | 21,868,800 | 11,685,632 | 683,400 | 365,176 | 
| app_proof | 61 | KeccakfOpAir | 0 | Execution |  | 0 | 606,080 | 442,496 | 18,940 | 13,828 | 
| app_proof | 61 | KeccakfOpAir | 1 | Memory |  | 0 | 15,758,080 | 11,504,896 | 492,440 | 359,528 | 
| app_proof | 61 | KeccakfOpAir | 2 | Program |  | 0 | 303,040 | 221,248 | 9,470 | 6,914 | 
| app_proof | 61 | KeccakfOpAir | 3 | VariableRange |  | 0 | 16,364,160 | 11,947,392 | 511,380 | 373,356 | 
| app_proof | 61 | KeccakfOpAir | 7 | KeccakfState |  | 0 | 606,080 | 442,496 | 18,940 | 13,828 | 
| app_proof | 62 | KeccakfPermAir | 7 | KeccakfState |  | 0 | 14,545,920 | 2,231,296 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir | 0 | Execution |  | 0 | 605,312 | 443,264 | 18,916 | 13,852 | 
| app_proof | 63 | XorinVmAir | 1 | Memory |  | 0 | 32,686,848 | 23,936,256 | 1,021,464 | 748,008 | 
| app_proof | 63 | XorinVmAir | 2 | Program |  | 0 | 302,656 | 221,632 | 9,458 | 6,926 | 
| app_proof | 63 | XorinVmAir | 3 | VariableRange |  | 0 | 33,897,472 | 24,822,784 | 1,059,296 | 775,712 | 
| app_proof | 63 | XorinVmAir | 6 | BitwiseLookup |  | 0 | 41,161,216 | 30,141,952 | 1,286,288 | 941,936 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 0 | Execution |  | 0 | 8,093,376 | 295,232 | 252,918 | 9,226 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 1 | Memory |  | 0 | 16,186,752 | 590,464 | 505,836 | 18,452 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 2 | Program |  | 0 | 4,046,688 | 147,616 | 126,459 | 4,613 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 3 | VariableRange |  | 0 | 16,186,752 | 590,464 | 505,836 | 18,452 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 36,420,192 | 1,328,544 | 1,138,131 | 41,517 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 0 | Execution |  | 0 | 14,912 | 1,472 | 466 | 46 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 1 | Memory |  | 0 | 29,824 | 2,944 | 932 | 92 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 2 | Program |  | 0 | 7,456 | 736 | 233 | 23 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 52,192 | 5,152 | 1,631 | 161 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 0 | Execution |  | 0 | 3,552,064 | 642,240 | 111,002 | 20,070 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 1 | Memory |  | 0 | 7,104,128 | 1,284,480 | 222,004 | 40,140 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 2 | Program |  | 0 | 1,776,032 | 321,120 | 55,501 | 10,035 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | 3 | VariableRange |  | 0 | 21,312,384 | 3,853,440 | 666,012 | 120,420 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 0 | Execution |  | 0 | 20,456,896 | 13,097,536 | 639,278 | 409,298 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 1 | Memory |  | 0 | 40,913,792 | 26,195,072 | 1,278,556 | 818,596 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 2 | Program |  | 0 | 10,228,448 | 6,548,768 | 319,639 | 204,649 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 92,056,032 | 58,938,912 | 2,876,751 | 1,841,841 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 0 | Execution |  | 0 | 2,045,440 | 51,712 | 63,920 | 1,616 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 1 | Memory |  | 0 | 2,045,440 | 51,712 | 63,920 | 1,616 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 2 | Program |  | 0 | 1,022,720 | 25,856 | 31,960 | 808 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | 3 | VariableRange |  | 0 | 9,204,480 | 232,704 | 287,640 | 7,272 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 0 | Execution |  | 0 | 5,376,960 | 3,011,648 | 168,030 | 94,114 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 1 | Memory |  | 0 | 10,753,920 | 6,023,296 | 336,060 | 188,228 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 2 | Program |  | 0 | 2,688,480 | 1,505,824 | 84,015 | 47,057 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | 3 | VariableRange |  | 0 | 21,507,840 | 12,046,592 | 672,120 | 376,456 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 0 | Execution |  | 0 | 6,644,160 | 1,744,448 | 207,630 | 54,514 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 1 | Memory |  | 0 | 6,644,160 | 1,744,448 | 207,630 | 54,514 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 2 | Program |  | 0 | 3,322,080 | 872,224 | 103,815 | 27,257 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | 3 | VariableRange |  | 0 | 23,254,560 | 6,105,568 | 726,705 | 190,799 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 6,595,776 | 1,792,832 | 206,118 | 56,026 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 13,191,552 | 3,585,664 | 412,236 | 112,052 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 2 | Program |  | 0 | 3,297,888 | 896,416 | 103,059 | 28,013 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 23,085,216 | 6,274,912 | 721,413 | 196,091 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 0 | Execution |  | 0 | 11,403,200 | 5,374,016 | 356,350 | 167,938 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 1 | Memory |  | 0 | 22,806,400 | 10,748,032 | 712,700 | 335,876 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 2 | Program |  | 0 | 5,701,600 | 2,687,008 | 178,175 | 83,969 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | 3 | VariableRange |  | 0 | 22,806,400 | 10,748,032 | 712,700 | 335,876 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 0 | Execution |  | 0 | 19,647,872 | 13,906,560 | 613,996 | 434,580 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 1 | Memory |  | 0 | 78,591,488 | 55,626,240 | 2,455,984 | 1,738,320 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 2 | Program |  | 0 | 9,823,936 | 6,953,280 | 306,998 | 217,290 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 3 | VariableRange |  | 0 | 98,239,360 | 69,532,800 | 3,069,980 | 2,172,900 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | 6 | BitwiseLookup |  | 0 | 58,943,616 | 41,719,680 | 1,841,988 | 1,303,740 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 0 | Execution |  | 0 | 18,573,440 | 14,980,992 | 580,420 | 468,156 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 1 | Memory |  | 0 | 74,293,760 | 59,923,968 | 2,321,680 | 1,872,624 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 2 | Program |  | 0 | 9,286,720 | 7,490,496 | 290,210 | 234,078 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 3 | VariableRange |  | 0 | 92,867,200 | 74,904,960 | 2,902,100 | 2,340,780 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | 6 | BitwiseLookup |  | 0 | 46,433,600 | 37,452,480 | 1,451,050 | 1,170,390 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 0 | Execution |  | 0 | 103,552 | 27,520 | 3,236 | 860 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 1 | Memory |  | 0 | 414,208 | 110,080 | 12,944 | 3,440 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 2 | Program |  | 0 | 51,776 | 13,760 | 1,618 | 430 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 3 | VariableRange |  | 0 | 517,760 | 137,600 | 16,180 | 4,300 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | 6 | BitwiseLookup |  | 0 | 207,104 | 55,040 | 6,472 | 1,720 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 192 | 64 | 6 | 2 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 3,456 | 1,152 | 108 | 36 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 96 | 32 | 3 | 1 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 40,992 | 13,664 | 1,281 | 427 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 0 | Execution |  | 0 | 1,455,424 | 641,728 | 45,482 | 20,054 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 1 | Memory |  | 0 | 5,821,696 | 2,566,912 | 181,928 | 80,216 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 2 | Program |  | 0 | 727,712 | 320,864 | 22,741 | 10,027 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 3 | VariableRange |  | 0 | 8,004,832 | 3,529,504 | 250,151 | 110,297 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | 6 | BitwiseLookup |  | 0 | 2,183,136 | 962,592 | 68,223 | 30,081 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 0 | Execution |  | 0 | 591,424 | 457,152 | 18,482 | 14,286 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 1 | Memory |  | 0 | 1,774,272 | 1,371,456 | 55,446 | 42,858 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 2 | Program |  | 0 | 295,712 | 228,576 | 9,241 | 7,143 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 3 | VariableRange |  | 0 | 2,365,696 | 1,828,608 | 73,928 | 57,144 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | 6 | BitwiseLookup |  | 0 | 591,424 | 457,152 | 18,482 | 14,286 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 0 | Execution |  | 0 | 454,080 | 70,208 | 14,190 | 2,194 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 1 | Memory |  | 0 | 1,362,240 | 210,624 | 42,570 | 6,582 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 2 | Program |  | 0 | 227,040 | 35,104 | 7,095 | 1,097 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 3 | VariableRange |  | 0 | 1,816,320 | 280,832 | 56,760 | 8,776 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | 6 | BitwiseLookup |  | 0 | 227,040 | 35,104 | 7,095 | 1,097 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 0 | Execution |  | 0 | 102,400 | 28,672 | 3,200 | 896 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 1 | Memory |  | 0 | 307,200 | 86,016 | 9,600 | 2,688 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 2 | Program |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 3 | VariableRange |  | 0 | 460,800 | 129,024 | 14,400 | 4,032 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | 6 | BitwiseLookup |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 0 | Execution |  | 0 | 25,600 | 7,168 | 800 | 224 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 1 | Memory |  | 0 | 51,200 | 14,336 | 1,600 | 448 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 2 | Program |  | 0 | 12,800 | 3,584 | 400 | 112 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | 3 | VariableRange |  | 0 | 115,200 | 32,256 | 3,600 | 1,008 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 0 | Execution |  | 0 | 48,512 | 17,024 | 1,516 | 532 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 1 | Memory |  | 0 | 97,024 | 34,048 | 3,032 | 1,064 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 2 | Program |  | 0 | 24,256 | 8,512 | 758 | 266 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 169,792 | 59,584 | 5,306 | 1,862 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | Execution |  | 0 | 64 |  | 2 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 1 | Memory |  | 0 | 1,728 |  | 54 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 2 | Program |  | 0 | 32 |  | 1 |  | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 3 | VariableRange |  | 0 | 14,304 |  | 447 |  | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 0 | Execution |  | 0 | 6,656 | 1,536 | 208 | 48 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 1 | Memory |  | 0 | 19,968 | 4,608 | 624 | 144 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 2 | Program |  | 0 | 3,328 | 768 | 104 | 24 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | 3 | VariableRange |  | 0 | 29,952 | 6,912 | 936 | 216 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 0 | Execution |  | 0 | 1,352,512 | 744,640 | 42,266 | 23,270 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 1 | Memory |  | 0 | 4,057,536 | 2,233,920 | 126,798 | 69,810 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 2 | Program |  | 0 | 676,256 | 372,320 | 21,133 | 11,635 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | 3 | VariableRange |  | 0 | 5,410,048 | 2,978,560 | 169,064 | 93,080 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | Execution |  | 0 | 1,118,784 | 978,368 | 34,962 | 30,574 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 1 | Memory |  | 0 | 3,356,352 | 2,935,104 | 104,886 | 91,722 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 2 | Program |  | 0 | 559,392 | 489,184 | 17,481 | 15,287 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 3 | VariableRange |  | 0 | 3,356,352 | 2,935,104 | 104,886 | 91,722 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | 6 | BitwiseLookup |  | 0 | 4,475,136 | 3,913,472 | 139,848 | 122,296 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 0 | Execution |  | 0 | 16,408,384 | 368,832 | 512,762 | 11,526 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 1 | Memory |  | 0 | 49,225,152 | 1,106,496 | 1,538,286 | 34,578 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 2 | Program |  | 0 | 8,204,192 | 184,416 | 256,381 | 5,763 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | 3 | VariableRange |  | 0 | 82,041,920 | 1,844,160 | 2,563,810 | 57,630 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | 6 | BitwiseLookup |  | 0 | 4,194,304 |  | 131,072 |  | 

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
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 65,536 | 37 | 2,424,832 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 32,768 | 44 | 1,441,792 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 38 | 1,245,184 | 
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
| app_proof | 0 | ProgramAir | prover |  | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 101 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 718 | 1,436 | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 526 | 1,052 | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 486 | 972 | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 358 | 716 | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 4 | 105 | 420 | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 277 | 17,728 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 213 | 13,632 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 4,096 | 38 | 155,648 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover |  | 0 | 8 | 145 | 1,160 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 369 | 738 | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 273 | 1,092 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 8,192 | 33 | 270,336 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 978 | 3,912 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | prover |  | 0 | 512 | 172 | 88,064 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover |  | 0 | 256 | 154 | 39,424 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover |  | 0 | 256 | 80 | 20,480 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover |  | 0 | 512 | 111 | 56,832 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | prover |  | 0 | 512 | 156 | 79,872 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 910 | 910 | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | prover |  | 0 | 512 | 107 | 54,784 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 256 | 40 | 10,240 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover |  | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover |  | 0 | 32,768 | 131 | 4,292,608 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover |  | 0 | 16,384 | 258 | 4,227,072 | 
| app_proof | 62 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover |  | 0 | 16,384 | 543 | 8,896,512 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 131,072 | 34 | 4,456,448 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 256 | 27 | 6,912 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 51 | 3,342,336 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 524,288 | 23 | 12,058,624 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> | prover |  | 0 | 32,768 | 16 | 524,288 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> | prover |  | 0 | 131,072 | 23 | 3,014,656 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> | prover |  | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 524,288 | 38 | 19,922,944 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 38 | 19,922,944 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 2,048 | 36 | 73,728 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 32,768 | 37 | 1,212,416 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 16,384 | 28 | 458,752 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 8,192 | 28 | 229,376 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 2,048 | 29 | 59,392 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 512 | 44 | 22,528 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1,024 | 22 | 22,528 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 128 | 33 | 4,224 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 32,768 | 28 | 917,504 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 42 | 1,376,256 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 262,144 | 29 | 7,602,176 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_cells_unpadded | metered_interaction_cells_padding | metered_constraint_eval_cells_unpadded | metered_constraint_eval_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 6,504 | 1,688 | 260,160 | 67,520 | 65,040 | 16,880 | 6,504 | 1,688 |  |  | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 10 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 495 | 165 | 
| app_proof | 100 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 4,456 | 3,736 | 5,347,200 | 4,483,200 | 1,336,800 | 1,120,800 | 4,456 | 3,736 | 147,048 | 123,288 | 
| app_proof | 101 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 11 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 196 |  | 
| app_proof | 12 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 5,744 |  | 1,436 |  | 1,102 |  | 776 |  | 
| app_proof | 13 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 4,208 |  | 1,052 |  | 718 |  | 114 |  | 
| app_proof | 14 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 3,888 |  | 972 |  | 750 |  | 476 |  | 
| app_proof | 15 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,864 |  | 716 |  | 494 |  | 80 |  | 
| app_proof | 16 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 4 |  | 1,680 |  | 420 |  | 212 |  | 288 |  | 
| app_proof | 17 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 69,804 | 1,108 | 17,451 | 277 | 13,986 | 222 | 5,355 | 85 | 
| app_proof | 18 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 53,676 | 852 | 13,419 | 213 | 9,954 | 158 | 2,898 | 46 | 
| app_proof | 19 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 2,059 | 2,037 | 312,968 | 309,624 | 78,242 | 77,406 | 16,472 | 16,296 | 8,236 | 8,148 | 
| app_proof | 20 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 258 |  | 
| app_proof | 21 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 22 | VmAirWrapper<IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 7 | 1 | 4,060 | 580 | 1,015 | 145 | 483 | 69 | 728 | 104 | 
| app_proof | 23 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,952 |  | 738 |  | 572 |  | 388 |  | 
| app_proof | 24 | VmAirWrapper<VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 3,276 | 1,092 | 819 | 273 | 570 | 190 | 171 | 57 | 
| app_proof | 25 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 26 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 250 |  | 
| app_proof | 27 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 28 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 29 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 216 |  | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 4,356 | 3,836 | 574,992 | 506,352 | 143,748 | 126,588 | 17,424 | 15,344 | 39,204 | 34,524 | 
| app_proof | 30 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 120 | 40 | 
| app_proof | 31 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 32 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 206 |  | 
| app_proof | 33 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 80 |  | 
| app_proof | 34 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 35 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 254 |  | 
| app_proof | 36 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 123 | 41 | 
| app_proof | 37 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 840 |  | 210 |  | 106 |  | 144 |  | 
| app_proof | 38 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 208 |  | 
| app_proof | 39 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,512 |  | 378 |  | 268 |  | 82 |  | 
| app_proof | 4 | VmAirWrapper<VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 11,736 | 3,912 | 2,934 | 978 | 2,649 | 883 | 855 | 285 | 
| app_proof | 40 | VmAirWrapper<IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 2,940 | 420 | 735 | 105 | 371 | 53 | 504 | 72 | 
| app_proof | 41 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,024 |  | 506 |  | 396 |  | 210 |  | 
| app_proof | 42 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,268 | 756 | 567 | 189 | 402 | 134 | 120 | 40 | 
| app_proof | 44 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 400 | 112 | 275,200 | 77,056 | 68,800 | 19,264 | 40,800 | 11,424 | 27,600 | 7,728 | 
| app_proof | 45 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 200 | 56 | 123,200 | 34,496 | 30,800 | 8,624 | 26,600 | 7,448 | 400 | 112 | 
| app_proof | 47 | VmAirWrapper<VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 200 | 56 | 64,000 | 17,920 | 16,000 | 4,480 | 9,400 | 2,632 | 3,800 | 1,064 | 
| app_proof | 48 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 295 | 217 | 130,980 | 96,348 | 32,745 | 24,087 | 21,240 | 15,624 | 2,360 | 1,736 | 
| app_proof | 49 | VmAirWrapper<VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 400 | 112 | 249,600 | 69,888 | 62,400 | 17,472 | 40,400 | 11,312 | 1,600 | 448 | 
| app_proof | 5 | VmAirWrapper<VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 1 |  | 3,640 |  | 910 |  | 744 |  | 332 |  | 
| app_proof | 50 | VmAirWrapper<VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 400 | 112 | 171,200 | 47,936 | 42,800 | 11,984 | 34,000 | 9,520 | 3,200 | 896 | 
| app_proof | 55 | VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 205 | 51 | 32,800 | 8,160 | 8,200 | 2,040 | 6,355 | 1,581 | 410 | 102 | 
| app_proof | 56 | RangeTupleCheckerAir<2> |  | 0 | 2,097,152 |  | 25,165,824 |  | 6,291,456 |  | 2,097,152 |  | 10,485,760 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> |  | 0 | 20,100 | 12,668 | 10,532,400 | 6,638,032 | 2,633,100 | 1,659,508 | 1,768,800 | 1,114,784 | 60,300 | 38,004 | 
| app_proof | 6 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 504 | 168 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 341,700 | 182,588 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 9,909,300 | 5,295,052 | 250,124,400 | 133,654,416 | 
| app_proof | 61 | KeccakfOpAir |  | 0 | 9,470 | 6,914 | 9,773,040 | 7,135,248 | 2,443,260 | 1,783,812 | 1,051,170 | 767,454 | 18,940 | 13,828 | 
| app_proof | 62 | KeccakfPermAir |  | 0 | 227,280 | 34,864 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 454,560 | 69,728 | 607,064,880 | 93,121,744 | 
| app_proof | 63 | XorinVmAir |  | 0 | 9,458 | 6,926 | 20,542,776 | 15,043,272 | 5,135,694 | 3,760,818 | 3,395,422 | 2,486,434 | 37,832 | 27,704 | 
| app_proof | 66 | VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 126,459 | 4,613 | 17,198,424 | 627,368 | 4,299,606 | 156,842 | 2,529,180 | 92,260 | 505,836 | 18,452 | 
| app_proof | 67 | VmAirWrapper<BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 233 | 23 | 25,164 | 2,484 | 6,291 | 621 | 3,262 | 322 | 1,864 | 184 | 
| app_proof | 69 | VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 55,501 | 10,035 | 11,322,204 | 2,047,140 | 2,830,551 | 511,785 | 1,054,519 | 190,665 | 1,221,022 | 220,770 | 
| app_proof | 7 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 199 |  | 
| app_proof | 70 | VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 319,639 | 204,649 | 29,406,788 | 18,827,708 | 7,351,697 | 4,706,927 | 5,114,224 | 3,274,384 | 1,598,195 | 1,023,245 | 
| app_proof | 71 | VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir> |  | 0 | 31,960 | 808 | 2,045,440 | 51,712 | 511,360 | 12,928 | 447,440 | 11,312 | 127,840 | 3,232 | 
| app_proof | 72 | VmAirWrapper<JalrAdapterAir, JalrCoreAir> |  | 0 | 84,015 | 47,057 | 7,729,380 | 4,329,244 | 1,932,345 | 1,082,311 | 1,260,225 | 705,855 | 420,075 | 235,285 | 
| app_proof | 73 | VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir> |  | 0 | 103,815 | 27,257 | 7,474,680 | 1,962,504 | 1,868,670 | 490,626 | 1,245,780 | 327,084 | 1,038,150 | 272,570 | 
| app_proof | 74 | VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 103,059 | 28,013 | 12,367,080 | 3,361,560 | 3,091,770 | 840,390 | 1,442,826 | 392,182 | 824,472 | 224,104 | 
| app_proof | 75 | VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 178,175 | 83,969 | 17,104,800 | 8,061,024 | 4,276,200 | 2,015,256 | 1,959,925 | 923,659 | 1,247,225 | 587,783 | 
| app_proof | 76 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 306,998 | 217,290 | 46,663,696 | 33,028,080 | 11,665,924 | 8,257,020 | 8,288,946 | 5,866,830 | 1,841,988 | 1,303,740 | 
| app_proof | 77 | VmAirWrapper<LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 290,210 | 234,078 | 44,111,920 | 35,579,856 | 11,027,980 | 8,894,964 | 7,545,460 | 6,086,028 | 1,741,260 | 1,404,468 | 
| app_proof | 78 | VmAirWrapper<StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 1,618 | 430 | 232,992 | 61,920 | 58,248 | 15,480 | 40,450 | 10,750 | 9,708 | 2,580 | 
| app_proof | 8 | VmAirWrapper<VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 6,348 | 2,116 | 1,587 | 529 | 1,398 | 466 | 558 | 186 | 
| app_proof | 80 | VmAirWrapper<LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 22,741 | 10,027 | 3,365,668 | 1,483,996 | 841,417 | 370,999 | 568,525 | 250,675 | 136,446 | 60,162 | 
| app_proof | 84 | VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 9,241 | 7,143 | 1,034,992 | 800,016 | 258,748 | 200,004 | 175,579 | 135,717 | 55,446 | 42,858 | 
| app_proof | 85 | VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 7,095 | 1,097 | 794,640 | 122,864 | 198,660 | 30,716 | 127,710 | 19,746 | 42,570 | 6,582 | 
| app_proof | 86 | VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 1,600 | 448 | 185,600 | 51,968 | 46,400 | 12,992 | 30,400 | 8,512 | 9,600 | 2,688 | 
| app_proof | 88 | VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 400 | 112 | 70,400 | 19,712 | 17,600 | 4,928 | 6,400 | 1,792 | 8,800 | 2,464 | 
| app_proof | 89 | VmAirWrapper<BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 758 | 266 | 66,704 | 23,408 | 16,676 | 5,852 | 10,612 | 3,724 | 3,790 | 1,330 | 
| app_proof | 9 | VmAirWrapper<VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,456 |  | 614 |  | 504 |  | 217 |  | 
| app_proof | 94 | VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 104 | 24 | 13,728 | 3,168 | 3,432 | 792 | 1,872 | 432 | 832 | 192 | 
| app_proof | 95 | VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 21,133 | 11,635 | 2,366,896 | 1,303,120 | 591,724 | 325,780 | 359,261 | 197,795 | 169,064 | 93,080 | 
| app_proof | 96 | VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 17,481 | 15,287 | 2,936,808 | 2,568,216 | 734,202 | 642,054 | 402,063 | 351,601 | 69,924 | 61,148 | 
| app_proof | 97 | VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 256,381 | 5,763 | 29,740,196 | 668,508 | 7,435,049 | 167,127 | 4,871,239 | 109,497 | 2,051,048 | 46,104 | 
| app_proof | 98 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 0 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 198 | 18 | 6 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 120 | 10 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 108 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 107 | 478 | 106 | 16 | 6 | 18 | 20 | 20 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,594,365 | 179 | 45 | 0 | 0 | 81 | 29 | 28 | 37 | 14 | 0 | 51 | 40 | 10 | 2 | 7 | 46 | 45 | 81 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,386,961 | 108 | 20 | 0 | 0 | 56 | 21 | 20 | 23 | 11 | 0 | 31 | 23 | 7 | 1 | 6 | 20 | 20 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,759,057 | 97 | 14 | 0 | 0 | 54 | 20 | 19 | 21 | 11 | 0 | 28 | 21 | 7 | 1 | 5 | 15 | 14 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 167,516,989 | 371 | 96 | 0 | 0 | 185 | 78 | 77 | 39 | 67 | 0 | 88 | 70 | 17 | 7 | 9 | 97 | 96 | 185 | 0 | 3 | 66 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,643 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 5 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,045,458,492 | 1,962 | 494 | 0 | 75 | 1,037 | 712 | 711 | 196 | 127 | 1 | 430 | 344 | 85 | 46 | 38 | 495 | 494 | 1,036 | 0 | 1 | 125 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 85,613,970 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 2,252 | 28 | 1,979,971 | 70.60 | 0 | 2,283 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_whir_memory_bytes | metered_secondary_peak_memory_bytes | metered_rs_code_matrix_memory_bytes | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_gkr_memory_bytes | metered_batch_constraint_memory_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 53 | 2 | 70 | 2,252 | 5 | 53 | 0 | 72 | 0 | 7 | 0 | 125,829,120 | 8,514,437,120 | 8,388,608,000 | 10,140,004,592 | 2,561,181,696 | 12,701,186,288 | 3,293,426,240 | 5,241,766,836 | 3 | 0 | 1 | 5 | 64 | 78 | 1,979,971 | 25.26 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 5 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/5bdc769320d126793f07740a98edbf95ce3ed8f6

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33816986882)
