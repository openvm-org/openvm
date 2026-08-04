| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  2.93 |  2.93 |  2.93 |
| app_proof |  2.05 |  2.05 |  2.05 |
| leaf |  0.46 |  0.46 |  0.46 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,030 |  2,030 |  2,030 |  2,030 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  22 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  88.92 | -          |  88.92 |  88.92 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `execute_preflight_insns` |  1,979,971 |  1,979,971 |  1,979,971 |  1,979,971 |
| `execute_preflight_time_ms` |  98 |  98 |  98 |  98 |
| `execute_preflight_insn_mi/s` |  20.20 | -          |  20.16 |  20.16 |
| `postflight_time_ms  ` |  26 |  26 |  26 |  26 |
| `postflight_memory_chronology_time_ms` |  2 |  2 |  2 |  2 |
| `postflight_program_index_time_ms` |  0 |  0 |  0 |  0 |
| `trace_gen_time_ms   ` |  125 |  125 |  125 |  125 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,765 |  1,765 |  1,765 |  1,765 |
| `prover.main_trace_commit_time_ms` |  511 |  511 |  511 |  511 |
| `prover.rap_constraints_time_ms` |  917 |  917 |  917 |  917 |
| `prover.openings_time_ms` |  336 |  336 |  336 |  336 |
| `prover.rap_constraints.logup_gkr_time_ms` |  110 |  110 |  110 |  110 |
| `prover.rap_constraints.round0_time_ms` |  617 |  617 |  617 |  617 |
| `prover.rap_constraints.mle_rounds_time_ms` |  188 |  188 |  188 |  188 |
| `prover.openings.stacked_reduction_time_ms` |  84 |  84 |  84 |  84 |
| `prover.openings.stacked_reduction.round0_time_ms` |  46 |  46 |  46 |  46 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.whir_time_ms` |  251 |  251 |  251 |  251 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  458 |  458 |  458 |  458 |
| `execute_preflight_time_ms` |  17 |  17 |  17 |  17 |
| `trace_gen_time_ms   ` |  109 |  109 |  109 |  109 |
| `generate_blob_total_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  349 |  349 |  349 |  349 |
| `prover.main_trace_commit_time_ms` |  98 |  98 |  98 |  98 |
| `prover.rap_constraints_time_ms` |  162 |  162 |  162 |  162 |
| `prover.openings_time_ms` |  88 |  88 |  88 |  88 |
| `prover.rap_constraints.logup_gkr_time_ms` |  57 |  57 |  57 |  57 |
| `prover.rap_constraints.round0_time_ms` |  66 |  66 |  66 |  66 |
| `prover.rap_constraints.mle_rounds_time_ms` |  39 |  39 |  39 |  39 |
| `prover.openings.stacked_reduction_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.stacked_reduction.round0_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  70 |  70 |  70 |  70 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  195 |  195 |  195 |  195 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  17 |  17 |  17 |  17 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  177 |  177 |  177 |  177 |
| `prover.main_trace_commit_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints_time_ms` |  81 |  81 |  81 |  81 |
| `prover.openings_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints.logup_gkr_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  36 |  36 |  36 |  36 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  38 |  38 |  38 |  38 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  118 |  118 |  118 |  118 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  107 |  107 |  107 |  107 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  55 |  55 |  55 |  55 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  23 |  23 |  23 |  23 |

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
| `prover.openings_time_ms` |  28 |  28 |  28 |  28 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9/kitchen_sink-e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 11.90 | app_proof.prover..0 |
| prover.rap_constraints | 8.89 | app_proof.prover..0 |
| prover.openings | 8.05 | app_proof.prover..0 |
| prover.merkle_tree | 8.05 | app_proof.prover..0 |
| prover.prove_whir_opening | 8.05 | app_proof.prover..0 |
| prover.rs_code_matrix | 8.05 | app_proof.prover..0 |
| prover.batch_constraints.round0 | 6.97 | app_proof.prover..0 |
| prover.batch_constraints.fold_ple_evals | 6.97 | app_proof.prover..0 |
| frac_sumcheck.gkr_rounds | 6.96 | app_proof.prover..0 |
| prover.batch_constraints.before_round0 | 6.96 | app_proof.prover..0 |
| frac_sumcheck.segment_tree | 6.92 | app_proof.prover..0 |
| prover.gkr_input_evals | 6.92 | app_proof.prover..0 |
| tracegen | 4.24 | app_proof..0 |
| prover.before_gkr_input_evals | 4.09 | app_proof.prover..0 |
| postflight | 1.23 | app_proof..0 |
| generate mem proving ctxs | 1.07 | app_proof..0 |
| set initial memory | 0.91 | app_proof..0 |
| tracegen.exp_bits_len | 0.82 | leaf.0 |
| tracegen.pow_checker | 0.82 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.82 | leaf.0 |
| tracegen.whir_folding | 0.75 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.75 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.75 | leaf.0 |
| tracegen.range_checker | 0.66 | leaf.0 |
| tracegen.public_values | 0.66 | leaf.0 |
| tracegen.proof_shape | 0.66 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- |
| 548 | 267,239 | 229,348 | 25 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 262 | 3 | 
| 100 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 230 | 2 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 548 | 247 | 3 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 356 | 151 | 3 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 372 | 167 | 3 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 244 | 103 | 3 | 
| 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 219 | 102 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 155 | 70 | 3 | 
| 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 8 | 11 | 2 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 67 | 155 | 3 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 283 | 150 | 3 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 187 | 102 | 3 | 
| 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 38 | 3 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 881 | 487 | 3 | 
| 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 107 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 102 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 70 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 100 | 307 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 99 | 582 | 3 | 
| 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 130 | 1 | 2 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 48 | 59 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 45 | 21 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 69 | 56 | 3 | 
| 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 98 | 4 | 2 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 741 | 342 | 2 | 
| 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 82 | 35 | 3 | 
| 51 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 62 | 3 | 
| 52 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 101 | 3 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 8 | 2 | 
| 54 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 2 | 2 | 
| 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 1 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 149 | 4 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 85 | 4 | 3 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 262 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 110 | 1 | 2 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 357 | 34 | 3 | 
| 64 | Rv64HintStoreAir | 1 | 17 | 12 | 3 | 
| 65 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 20 | 5 | 2 | 
| 66 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 14 | 20 | 3 | 
| 67 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<4, 16> |  | 20 | 43 | 3 | 
| 68 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 19 | 66 | 3 | 
| 69 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 16 | 6 | 3 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 230 | 2 | 
| 70 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 4 | 3 | 
| 71 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 8 | 3 | 
| 72 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 10 | 2 | 
| 73 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 23 | 3 | 
| 74 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 9 | 3 | 
| 75 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 29 | 9 | 3 | 
| 76 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 28 | 11 | 3 | 
| 77 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 27 | 9 | 3 | 
| 78 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 26 | 11 | 3 | 
| 79 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 27 | 12 | 3 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 262 | 3 | 
| 80 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<2, 1> |  | 26 | 9 | 3 | 
| 81 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 25 | 11 | 3 | 
| 82 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<2, 2> |  | 26 | 12 | 3 | 
| 83 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 19 | 8 | 3 | 
| 84 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 18 | 10 | 3 | 
| 85 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 19 | 11 | 3 | 
| 86 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<2, 16> |  | 17 | 28 | 3 | 
| 87 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 16 | 37 | 3 | 
| 88 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 14 | 5 | 3 | 
| 89 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 22 | 28 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 230 | 2 | 
| 90 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 21 | 37 | 3 | 
| 91 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 25 | 43 | 3 | 
| 92 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 24 | 66 | 3 | 
| 93 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 18 | 20 | 3 | 
| 94 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 17 | 8 | 3 | 
| 95 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 23 | 4 | 2 | 
| 96 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 19 | 11 | 3 | 
| 97 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 98 | PhantomAir |  | 3 | 1 | 2 | 
| 99 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 

| group | upload_preflight_program_time_ms | transport_pk_to_device_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prepare_preflight_time_ms | new_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen |  | 61 |  |  |  | 321 |  | 
| app_proof | 0 |  |  |  | 3 |  |  | 
| internal_for_leaf |  |  |  | 195 |  |  | 195 | 
| internal_recursive.0 |  |  |  | 118 |  |  | 118 | 
| internal_recursive.1 |  |  |  | 107 |  |  | 107 | 
| leaf |  |  | 458 |  |  |  | 458 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | program | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | BitwiseOperationLookupAir<8> |  | 0 | 0 | 
| app_proof | KeccakfOpAir |  | 0 | 1 | 
| app_proof | KeccakfPermAir |  | 0 | 56 | 
| app_proof | PhantomAir |  | 0 | 0 | 
| app_proof | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 1 | 
| app_proof | RangeTupleCheckerAir<2> |  | 0 | 0 | 
| app_proof | Rv64HintStoreAir |  | 0 | 0 | 
| app_proof | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 5 | 
| app_proof | Sha2BlockHasherVmAir<Sha512Config> |  | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> |  | 0 | 3 | 
| app_proof | Sha2MainAir<Sha512Config> |  | 0 | 0 | 
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
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<2, 2> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<4, 3> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 2 | 
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
| app_proof | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 8 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 0 | 0 | 
| app_proof | XorinVmAir |  | 0 | 0 | 

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

| group | air_id | air_name | phase | program | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover |  | 0 | 8,192 | 10 | 81,920 | 
| app_proof | 1 | VmConnectorAir | prover |  | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 100 | VariableRangeCheckerAir | prover |  | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 718 | 1,436 | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 526 | 1,052 | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 486 | 972 | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 358 | 716 | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 4 | 105 | 420 | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 277 | 17,728 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 64 | 213 | 13,632 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover |  | 0 | 4,096 | 38 | 155,648 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | prover |  | 0 | 8 | 145 | 1,160 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 369 | 738 | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 273 | 1,092 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover |  | 0 | 8,192 | 33 | 270,336 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 2 | 105 | 210 | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 189 | 378 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 978 | 3,912 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover |  | 0 | 8 | 105 | 840 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 2 | 253 | 506 | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 189 | 756 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | prover |  | 0 | 512 | 172 | 88,064 | 
| app_proof | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | prover |  | 0 | 256 | 154 | 39,424 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | prover |  | 0 | 256 | 80 | 20,480 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | prover |  | 0 | 512 | 111 | 56,832 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | prover |  | 0 | 512 | 156 | 79,872 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 910 | 910 | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> | prover |  | 0 | 512 | 107 | 54,784 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover |  | 0 | 256 | 40 | 10,240 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover |  | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover |  | 0 | 32,768 | 131 | 4,292,608 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover |  | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover |  | 0 | 16,384 | 258 | 4,227,072 | 
| app_proof | 62 | KeccakfPermAir | prover |  | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover |  | 0 | 16,384 | 543 | 8,896,512 | 
| app_proof | 65 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> | prover |  | 0 | 131,072 | 34 | 4,456,448 | 
| app_proof | 66 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> | prover |  | 0 | 256 | 27 | 6,912 | 
| app_proof | 68 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> | prover |  | 0 | 65,536 | 51 | 3,342,336 | 
| app_proof | 69 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> | prover |  | 0 | 524,288 | 23 | 12,058,624 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 70 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover |  | 0 | 32,768 | 16 | 524,288 | 
| app_proof | 71 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover |  | 0 | 131,072 | 22 | 2,883,584 | 
| app_proof | 72 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover |  | 0 | 131,072 | 17 | 2,228,224 | 
| app_proof | 73 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover |  | 0 | 131,072 | 30 | 3,932,160 | 
| app_proof | 74 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover |  | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 75 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> | prover |  | 0 | 524,288 | 40 | 20,971,520 | 
| app_proof | 76 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> | prover |  | 0 | 524,288 | 39 | 20,447,232 | 
| app_proof | 77 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> | prover |  | 0 | 2,048 | 38 | 77,824 | 
| app_proof | 79 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> | prover |  | 0 | 32,768 | 38 | 1,245,184 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 4 | 529 | 2,116 | 
| app_proof | 83 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> | prover |  | 0 | 16,384 | 29 | 475,136 | 
| app_proof | 84 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> | prover |  | 0 | 8,192 | 28 | 229,376 | 
| app_proof | 85 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> | prover |  | 0 | 2,048 | 29 | 59,392 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> | prover |  | 0 | 512 | 44 | 22,528 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> | prover |  | 0 | 1,024 | 22 | 22,528 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover |  | 0 | 1 | 614 | 614 | 
| app_proof | 93 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> | prover |  | 0 | 128 | 33 | 4,224 | 
| app_proof | 94 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> | prover |  | 0 | 32,768 | 28 | 917,504 | 
| app_proof | 95 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> | prover |  | 0 | 32,768 | 42 | 1,376,256 | 
| app_proof | 96 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> | prover |  | 0 | 262,144 | 29 | 7,602,176 | 
| app_proof | 97 | BitwiseOperationLookupAir<8> | prover |  | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 99 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover |  | 0 | 4,096 | 300 | 1,228,800 | 

| group | air_id | air_name | program | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir |  | 0 | 6,504 | 1,688 | 162,600 | 42,200 | 260,160 | 67,520 | 65,040 | 16,880 | 235,770 | 61,190 | 6,504 | 1,688 | 
| app_proof | 1 | VmConnectorAir |  | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 3,968 | 1,322 | 6,348 | 2,116 | 1,587 | 529 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 100 | VariableRangeCheckerAir |  | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,535 |  | 2,456 |  | 614 |  | 18,162 |  | 501 |  | 
| app_proof | 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 3,590 |  | 5,744 |  | 1,436 |  | 39,730 |  | 1,096 |  | 
| app_proof | 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,630 |  | 4,208 |  | 1,052 |  | 25,810 |  | 712 |  | 
| app_proof | 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 2,430 |  | 3,888 |  | 972 |  | 26,970 |  | 744 |  | 
| app_proof | 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,790 |  | 2,864 |  | 716 |  | 17,690 |  | 488 |  | 
| app_proof | 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 4 |  | 1,050 |  | 1,680 |  | 420 |  | 7,395 |  | 204 |  | 
| app_proof | 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 43,628 | 692 | 69,804 | 1,108 | 17,451 | 277 | 500,142 | 7,938 | 13,797 | 219 | 
| app_proof | 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 63 | 1 | 33,548 | 532 | 53,676 | 852 | 13,419 | 213 | 353,982 | 5,618 | 9,765 | 155 | 
| app_proof | 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 525 |  | 840 |  | 210 |  | 3,698 |  | 102 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> |  | 0 | 2,059 | 2,037 | 195,605 | 193,515 | 312,968 | 309,624 | 78,242 | 77,406 | 597,110 | 590,730 | 16,472 | 16,296 | 
| app_proof | 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 945 |  | 1,512 |  | 378 |  | 9,498 |  | 262 |  | 
| app_proof | 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 0 | 7 | 1 | 2,538 | 362 | 4,060 | 580 | 1,015 | 145 | 17,002 | 2,428 | 469 | 67 | 
| app_proof | 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,845 |  | 2,952 |  | 738 |  | 20,518 |  | 566 |  | 
| app_proof | 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 2,048 | 682 | 3,276 | 1,092 | 819 | 273 | 20,337 | 6,778 | 561 | 187 | 
| app_proof | 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 525 |  | 840 |  | 210 |  | 3,698 |  | 102 |  | 
| app_proof | 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 945 |  | 1,512 |  | 378 |  | 9,498 |  | 262 |  | 
| app_proof | 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 1,838 | 262 | 2,940 | 420 | 735 | 105 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 3 | MemoryMerkleAir<8> |  | 0 | 4,344 | 3,848 | 716,760 | 634,920 | 573,408 | 507,936 | 143,352 | 126,984 | 629,880 | 557,960 | 17,376 | 15,392 | 
| app_proof | 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,418 | 472 | 2,268 | 756 | 567 | 189 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 525 |  | 840 |  | 210 |  | 3,698 |  | 102 |  | 
| app_proof | 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 945 |  | 1,512 |  | 378 |  | 9,498 |  | 262 |  | 
| app_proof | 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 1,838 | 262 | 2,940 | 420 | 735 | 105 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,418 | 472 | 2,268 | 756 | 567 | 189 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 2 |  | 525 |  | 840 |  | 210 |  | 3,698 |  | 102 |  | 
| app_proof | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 945 |  | 1,512 |  | 378 |  | 9,498 |  | 262 |  | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 7,335 | 2,445 | 11,736 | 3,912 | 2,934 | 978 | 95,809 | 31,936 | 2,643 | 881 | 
| app_proof | 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 0 | 7 | 1 | 1,838 | 262 | 2,940 | 420 | 735 | 105 | 12,942 | 1,848 | 357 | 51 | 
| app_proof | 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 2 |  | 1,265 |  | 2,024 |  | 506 |  | 14,138 |  | 390 |  | 
| app_proof | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 1,418 | 472 | 2,268 | 756 | 567 | 189 | 14,247 | 4,748 | 393 | 131 | 
| app_proof | 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 0 | 400 | 112 | 172,000 | 48,160 | 275,200 | 77,056 | 68,800 | 19,264 | 1,435,500 | 401,940 | 39,600 | 11,088 | 
| app_proof | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 0 | 200 | 56 | 77,000 | 21,560 | 123,200 | 34,496 | 30,800 | 8,624 | 942,500 | 263,900 | 26,000 | 7,280 | 
| app_proof | 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 0 | 200 | 56 | 40,000 | 11,200 | 64,000 | 17,920 | 16,000 | 4,480 | 326,250 | 91,350 | 9,000 | 2,520 | 
| app_proof | 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 0 | 295 | 217 | 81,863 | 60,217 | 130,980 | 96,348 | 32,745 | 24,087 | 737,869 | 542,771 | 20,355 | 14,973 | 
| app_proof | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 0 | 400 | 112 | 156,000 | 43,680 | 249,600 | 69,888 | 62,400 | 17,472 | 1,421,000 | 397,880 | 39,200 | 10,976 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 0 | 1 |  | 2,275 |  | 3,640 |  | 910 |  | 26,862 |  | 741 |  | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16, true> |  | 0 | 400 | 112 | 107,000 | 29,960 | 171,200 | 47,936 | 42,800 | 11,984 | 1,189,000 | 332,920 | 32,800 | 9,184 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 0 | 205 | 51 | 20,500 | 5,100 | 32,800 | 8,160 | 8,200 | 2,040 | 230,369 | 57,311 | 6,355 | 1,581 | 
| app_proof | 56 | RangeTupleCheckerAir<2> |  | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> |  | 0 | 20,100 | 12,668 | 13,165,500 | 8,297,540 | 10,532,400 | 6,638,032 | 2,633,100 | 1,659,508 | 61,933,125 | 39,033,275 | 1,708,500 | 1,076,780 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 3,968 | 1,322 | 6,348 | 2,116 | 1,587 | 529 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> |  | 0 | 341,700 | 182,588 | 779,076,000 | 416,300,640 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 359,212,125 | 191,945,635 | 9,909,300 | 5,295,052 | 
| app_proof | 61 | KeccakfOpAir |  | 0 | 9,470 | 6,914 | 6,108,150 | 4,459,530 | 9,773,040 | 7,135,248 | 2,443,260 | 1,783,812 | 37,761,625 | 27,569,575 | 1,041,700 | 760,540 | 
| app_proof | 62 | KeccakfPermAir |  | 0 | 227,280 | 34,864 | 2,993,277,600 | 459,158,880 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 16,477,800 | 2,527,640 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir |  | 0 | 9,458 | 6,926 | 12,839,235 | 9,402,045 | 20,542,776 | 15,043,272 | 5,135,694 | 3,760,818 | 122,398,343 | 89,631,097 | 3,376,506 | 2,472,582 | 
| app_proof | 65 | VmAirWrapper<Rv64BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<8, 8> |  | 0 | 126,459 | 4,613 | 10,749,015 | 392,105 | 17,198,424 | 627,368 | 4,299,606 | 156,842 | 91,682,775 | 3,344,425 | 2,529,180 | 92,260 | 
| app_proof | 66 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<4, 16> |  | 0 | 233 | 23 | 15,728 | 1,552 | 25,164 | 2,484 | 6,291 | 621 | 118,248 | 11,672 | 3,262 | 322 | 
| app_proof | 68 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<4, 16> |  | 0 | 55,501 | 10,035 | 7,076,378 | 1,279,462 | 11,322,204 | 2,047,140 | 2,830,551 | 511,785 | 38,226,314 | 6,911,606 | 1,054,519 | 190,665 | 
| app_proof | 69 | VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<4, 16, true> |  | 0 | 319,639 | 204,649 | 18,379,243 | 11,767,317 | 29,406,788 | 18,827,708 | 7,351,697 | 4,706,927 | 185,390,620 | 118,696,420 | 5,114,224 | 3,274,384 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,535 |  | 2,456 |  | 614 |  | 18,162 |  | 501 |  | 
| app_proof | 70 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 0 | 31,960 | 808 | 1,278,400 | 32,320 | 2,045,440 | 51,712 | 511,360 | 12,928 | 16,219,700 | 410,060 | 447,440 | 11,312 | 
| app_proof | 71 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 0 | 84,015 | 47,057 | 4,620,825 | 2,588,135 | 7,393,320 | 4,141,016 | 1,848,330 | 1,035,254 | 45,683,157 | 25,587,243 | 1,260,225 | 705,855 | 
| app_proof | 72 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 0 | 103,815 | 27,257 | 4,412,138 | 1,158,422 | 7,059,420 | 1,853,476 | 1,764,855 | 463,369 | 45,159,525 | 11,856,795 | 1,245,780 | 327,084 | 
| app_proof | 73 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 0 | 103,059 | 28,013 | 7,729,425 | 2,100,975 | 12,367,080 | 3,361,560 | 3,091,770 | 840,390 | 52,302,443 | 14,216,597 | 1,442,826 | 392,182 | 
| app_proof | 74 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 0 | 178,175 | 83,969 | 10,690,500 | 5,038,140 | 17,104,800 | 8,061,024 | 4,276,200 | 2,015,256 | 71,047,282 | 33,482,638 | 1,959,925 | 923,659 | 
| app_proof | 75 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<8, 4> |  | 0 | 306,998 | 217,290 | 30,699,800 | 21,729,000 | 49,119,680 | 34,766,400 | 12,279,920 | 8,691,600 | 322,731,648 | 228,426,112 | 8,902,942 | 6,301,410 | 
| app_proof | 76 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadCoreAir<8, 5> |  | 0 | 290,210 | 234,078 | 28,295,475 | 22,822,605 | 45,272,760 | 36,516,168 | 11,318,190 | 9,129,042 | 294,563,150 | 237,589,170 | 8,125,880 | 6,554,184 | 
| app_proof | 77 | VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreCoreAir<4, 2> |  | 0 | 1,618 | 430 | 153,710 | 40,850 | 245,936 | 65,360 | 61,484 | 16,340 | 1,583,618 | 420,862 | 43,686 | 11,610 | 
| app_proof | 79 | VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendCoreAir<4, 3> |  | 0 | 22,741 | 10,027 | 2,160,395 | 952,565 | 3,456,632 | 1,524,104 | 864,158 | 381,026 | 22,257,754 | 9,813,926 | 614,007 | 270,729 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 0 | 3 | 1 | 3,968 | 1,322 | 6,348 | 2,116 | 1,587 | 529 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 83 | VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir> |  | 0 | 9,241 | 7,143 | 669,973 | 517,867 | 1,071,956 | 828,588 | 267,989 | 207,147 | 6,364,739 | 4,919,741 | 175,579 | 135,717 | 
| app_proof | 84 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir> |  | 0 | 7,095 | 1,097 | 496,650 | 76,790 | 794,640 | 122,864 | 198,660 | 30,716 | 4,629,488 | 715,792 | 127,710 | 19,746 | 
| app_proof | 85 | VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir> |  | 0 | 1,600 | 448 | 116,000 | 32,480 | 185,600 | 51,968 | 46,400 | 12,992 | 1,102,000 | 308,560 | 30,400 | 8,512 | 
| app_proof | 87 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<2, 16> |  | 0 | 400 | 112 | 44,000 | 12,320 | 70,400 | 19,712 | 17,600 | 4,928 | 232,000 | 64,960 | 6,400 | 1,792 | 
| app_proof | 88 | VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddICoreAir<2, 16, false> |  | 0 | 758 | 266 | 41,690 | 14,630 | 66,704 | 23,408 | 16,676 | 5,852 | 384,685 | 134,995 | 10,612 | 3,724 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 0 | 1 |  | 1,535 |  | 2,456 |  | 614 |  | 18,162 |  | 501 |  | 
| app_proof | 93 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<4, 16> |  | 0 | 104 | 24 | 8,580 | 1,980 | 13,728 | 3,168 | 3,432 | 792 | 67,860 | 15,660 | 1,872 | 432 | 
| app_proof | 94 | VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubCoreAir<2, 16, false> |  | 0 | 21,133 | 11,635 | 1,479,310 | 814,450 | 2,366,896 | 1,303,120 | 591,724 | 325,780 | 13,023,212 | 7,170,068 | 359,261 | 197,795 | 
| app_proof | 95 | VmAirWrapper<Rv64BaseAluRegAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 0 | 17,481 | 15,287 | 1,835,505 | 1,605,135 | 2,936,808 | 2,568,216 | 734,202 | 642,054 | 14,574,784 | 12,745,536 | 402,063 | 351,601 | 
| app_proof | 96 | VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<4, 16, true> |  | 0 | 256,381 | 5,763 | 18,587,623 | 417,817 | 29,740,196 | 668,508 | 7,435,049 | 167,127 | 176,582,414 | 3,969,266 | 4,871,239 | 109,497 | 
| app_proof | 97 | BitwiseOperationLookupAir<8> |  | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 99 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 0 | 4,444 | 3,748 | 3,333,000 | 2,811,000 | 5,332,800 | 4,497,600 | 1,333,200 | 1,124,400 | 161,095 | 135,865 | 4,444 | 3,748 | 

| group | backend | program | compile_metered_time_ms |
| --- | --- | --- | --- |
| app_proof | interpreter |  | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 17 | 195 | 17 | 5 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 118 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 107 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 109 | 458 | 109 | 18 | 6 | 17 | 23 | 23 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,981 | 177 | 46 | 0 | 0 | 81 | 29 | 28 | 36 | 15 | 0 | 49 | 38 | 10 | 2 | 7 | 46 | 46 | 81 | 0 | 1 | 12 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 107 | 20 | 0 | 0 | 55 | 21 | 20 | 23 | 10 | 0 | 30 | 23 | 7 | 1 | 6 | 20 | 20 | 55 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 96 | 14 | 0 | 0 | 53 | 20 | 19 | 21 | 11 | 0 | 28 | 20 | 7 | 1 | 5 | 15 | 14 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 167,484,221 | 349 | 98 | 0 | 0 | 162 | 66 | 64 | 39 | 57 | 0 | 88 | 70 | 17 | 7 | 9 | 98 | 98 | 162 | 0 | 3 | 56 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,643 | 2,013,265,921 | 

| group | phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | prover | 6 | 0 | 6 | 6 | 

| group | phase | program | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 1,046,822,460 | 1,765 | 511 | 0 | 0 | 917 | 617 | 616 | 188 | 110 | 1 | 336 | 251 | 84 | 46 | 38 | 511 | 511 | 917 | 0 | 1 | 108 | 0 | 0 | 

| group | phase | program | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover |  | 0 | 0 | 87,625,214 | 2,013,265,921 | 

| group | program | prove_segment_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 2,030 | 22 | 1,979,971 | 88.92 | 0 | 2,060 | 

| group | program | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | program_trace_gen_time_ms | postflight_time_ms | postflight_program_index_time_ms | postflight_memory_chronology_time_ms | poseidon2_prepare_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | merkle_update_time_ms | merkle_drop_time_ms | mem_merge_records_time_ms | generate_proving_ctxs_from_device_time_ms | executor_trace_gen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | connector_trace_gen_time_ms | boundary_trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof |  | 0 | 0 | 2 | 125 | 2,030 | 3 | 0 | 0 | 26 | 0 | 2 | 0 | 10,019,501,688 | 2,557,113,432 | 12,576,615,120 | 2,097,152 | 2 | 0 | 0 | 3 | 121 | 98 | 1,979,971 | 20.16 | 0 | 0 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/e298ad0738a0cac5ac3286ef9c6d52f9d043f2e9

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30937751913)
