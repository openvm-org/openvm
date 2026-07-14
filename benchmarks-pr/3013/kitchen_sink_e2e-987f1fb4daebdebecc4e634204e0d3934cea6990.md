| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  10.99 |  10.99 |  10.99 |
| app_proof |  1.95 |  1.95 |  1.95 |
| leaf |  0.37 |  0.37 |  0.37 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.10 |  0.10 |  0.10 |
| internal_recursive.1 |  0.09 |  0.09 |  0.09 |
| root |  0.89 |  0.89 |  0.89 |
| halo2_outer |  5.89 |  5.89 |  5.89 |
| halo2_wrapper |  1.53 |  1.53 |  1.53 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,932 |  1,932 |  1,932 |  1,932 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  22 | -          | -          | -          |
| `execute_metered_insns` |  1,979,971 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  87.97 | -          |  87.97 |  87.97 |
| `execute_preflight_insns` |  1,979,971 |  1,979,971 |  1,979,971 |  1,979,971 |
| `execute_preflight_time_ms` |  257 |  257 |  257 |  257 |
| `execute_preflight_insn_mi/s` |  35.43 | -          |  35.43 |  35.43 |
| `trace_gen_time_ms   ` |  83 |  83 |  83 |  83 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,591 |  1,591 |  1,591 |  1,591 |
| `prover.main_trace_commit_time_ms` |  578 |  578 |  578 |  578 |
| `prover.rap_constraints_time_ms` |  715 |  715 |  715 |  715 |
| `prover.openings_time_ms` |  297 |  297 |  297 |  297 |
| `prover.rap_constraints.logup_gkr_time_ms` |  100 |  100 |  100 |  100 |
| `prover.rap_constraints.round0_time_ms` |  488 |  488 |  488 |  488 |
| `prover.rap_constraints.mle_rounds_time_ms` |  125 |  125 |  125 |  125 |
| `prover.openings.stacked_reduction_time_ms` |  61 |  61 |  61 |  61 |
| `prover.openings.stacked_reduction.round0_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  32 |  32 |  32 |  32 |
| `prover.openings.whir_time_ms` |  235 |  235 |  235 |  235 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  370 |  370 |  370 |  370 |
| `execute_preflight_time_ms` |  18 |  18 |  18 |  18 |
| `trace_gen_time_ms   ` |  84 |  84 |  84 |  84 |
| `generate_blob_total_time_ms` |  9 |  9 |  9 |  9 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  285 |  285 |  285 |  285 |
| `prover.main_trace_commit_time_ms` |  81 |  81 |  81 |  81 |
| `prover.rap_constraints_time_ms` |  124 |  124 |  124 |  124 |
| `prover.openings_time_ms` |  79 |  79 |  79 |  79 |
| `prover.rap_constraints.logup_gkr_time_ms` |  52 |  52 |  52 |  52 |
| `prover.rap_constraints.round0_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.mle_rounds_time_ms` |  31 |  31 |  31 |  31 |
| `prover.openings.stacked_reduction_time_ms` |  12 |  12 |  12 |  12 |
| `prover.openings.stacked_reduction.round0_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  67 |  67 |  67 |  67 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  159 |  159 |  159 |  159 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  15 |  15 |  15 |  15 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  143 |  143 |  143 |  143 |
| `prover.main_trace_commit_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints_time_ms` |  62 |  62 |  62 |  62 |
| `prover.openings_time_ms` |  42 |  42 |  42 |  42 |
| `prover.rap_constraints.logup_gkr_time_ms` |  12 |  12 |  12 |  12 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  29 |  29 |  29 |  29 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  34 |  34 |  34 |  34 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  101 |  101 |  101 |  101 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  90 |  90 |  90 |  90 |
| `prover.main_trace_commit_time_ms` |  16 |  16 |  16 |  16 |
| `prover.rap_constraints_time_ms` |  48 |  48 |  48 |  48 |
| `prover.openings_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.logup_gkr_time_ms` |  9 |  9 |  9 |  9 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.stacked_reduction_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  19 |  19 |  19 |  19 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  91 |  91 |  91 |  91 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  81 |  81 |  81 |  81 |
| `prover.main_trace_commit_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints_time_ms` |  47 |  47 |  47 |  47 |
| `prover.openings_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.mle_rounds_time_ms` |  18 |  18 |  18 |  18 |
| `prover.openings.stacked_reduction_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.stacked_reduction.round0_time_ms` |  0 |  0 |  0 |  0 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.whir_time_ms` |  16 |  16 |  16 |  16 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  891 |  891 |  891 |  891 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  772 |  772 |  772 |  772 |
| `prover.main_trace_commit_time_ms` |  399 |  399 |  399 |  399 |
| `prover.rap_constraints_time_ms` |  69 |  69 |  69 |  69 |
| `prover.openings_time_ms` |  303 |  303 |  303 |  303 |
| `prover.rap_constraints.logup_gkr_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.mle_rounds_time_ms` |  30 |  30 |  30 |  30 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  296 |  296 |  296 |  296 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,892 |  5,892 |  5,892 |  5,892 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,533 |  1,533 |  1,533 |  1,533 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/987f1fb4daebdebecc4e634204e0d3934cea6990/kitchen_sink_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 13.11 | app_proof.prover.0 |
| prover.rap_constraints | 9.91 | app_proof.prover.0 |
| prover.openings | 9.12 | app_proof.prover.0 |
| prover.merkle_tree | 9.12 | app_proof.prover.0 |
| prover.prove_whir_opening | 9.12 | app_proof.prover.0 |
| prover.rs_code_matrix | 9.11 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 8.06 | app_proof.prover.0 |
| prover.batch_constraints.before_round0 | 8.06 | app_proof.prover.0 |
| prover.gkr_input_evals | 7.97 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 7.97 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 7.87 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 7.87 | app_proof.prover.0 |
| prover.before_gkr_input_evals | 4.99 | app_proof.prover.0 |
| set initial memory | 2.42 | app_proof.0 |
| generate mem proving ctxs | 2.42 | app_proof.0 |
| tracegen.pow_checker | 1.57 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.57 | leaf.0 |
| tracegen.exp_bits_len | 1.57 | leaf.0 |
| tracegen.whir_folding | 1.50 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.50 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.50 | leaf.0 |
| tracegen.public_values | 1.41 | leaf.0 |
| tracegen.range_checker | 1.41 | leaf.0 |
| tracegen.proof_shape | 1.41 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 602 | 32 | 25 | 0 | 0 | 0 | 2 | 4 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 0 | RootVerifierPvsAir |  | 109 | 37 | 4 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 12 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 548 | 286 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 13 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 356 | 190 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 14 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 372 | 194 | 3 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 15 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 244 | 130 | 3 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 23 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 32 | 3 | 
| 17 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 219 | 117 | 3 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 18 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 155 | 85 | 3 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 85 | 4 | 
| 19 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 19 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 20 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 20 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 21 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 21 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 22 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> |  | 67 | 170 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 23 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> |  | 19 | 16 | 3 | 
| 23 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 283 | 171 | 3 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 24 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 24 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> |  | 187 | 123 | 3 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 25 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> |  | 20 | 21 | 3 | 
| 25 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 26 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 26 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 27 | PhantomAir |  | 3 | 1 | 2 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 27 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 28 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 29 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 29 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 30 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 31 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 32 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 34 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 35 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 37 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> |  | 881 | 513 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 40 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 41 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 43 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftRightArithmeticCoreAir<16, 16> |  | 100 | 355 | 3 | 
| 44 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> |  | 99 | 660 | 3 | 
| 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> |  | 130 | 16 | 2 | 
| 46 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchLessThanCoreAir<16, 16> |  | 48 | 69 | 3 | 
| 47 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> |  | 45 | 31 | 3 | 
| 48 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> |  | 69 | 71 | 3 | 
| 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> |  | 98 | 19 | 2 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> |  | 741 | 381 | 2 | 
| 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> |  | 82 | 50 | 3 | 
| 51 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 52 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 53 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 54 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 56 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 57 | Sha2MainAir<Sha512Config> | 1 | 149 | 39 | 3 | 
| 58 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 59 | Sha2MainAir<Sha256Config> | 1 | 85 | 23 | 3 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 60 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 61 | KeccakfOpAir |  | 110 | 27 | 2 | 
| 62 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 63 | XorinVmAir |  | 357 | 92 | 3 | 
| 64 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 65 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 66 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 67 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 70 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 23 | 3 | 
| 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 32 | 3 | 
| 72 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 73 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 74 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 75 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 76 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 77 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> |  | 19 | 16 | 3 | 
| 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 79 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> |  | 20 | 21 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 80 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 81 | PhantomAir |  | 3 | 1 | 2 | 
| 82 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 83 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 70 |  |  |  |  |  |  |  | 280 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  | 1,932 |  |  |  |  |  |  |  | 22 | 1,979,971 | 87.97 | 0 |  |  | 1,961 |  | 
| halo2_keygen |  |  |  |  |  |  |  | 65 |  | 61,627 |  |  |  |  |  | 0 | 1 | 0 | 0 |  |  | 67 |  | 
| halo2_outer |  |  | 5,892 |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 1,533 |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  | 159 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 159 | 
| internal_recursive.0 |  |  |  |  |  | 101 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 101 | 
| internal_recursive.1 |  |  |  |  |  | 91 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 91 | 
| leaf |  |  |  |  | 370 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 370 | 
| root | 118 | 10 | 891 | 9 |  |  | 891 |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 | 0 |  | 891 | 
| root_keygen |  |  |  |  |  |  |  | 128 |  |  |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 130 |  | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 26 | KeccakfOpAir | 0 | 3,030,400 | 
| app_proof | Sha2MainAir<Sha256Config> | 28 | Sha2MainAir<Sha256Config> | 0 | 5,467,200 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 9,212,160 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 36,865,280 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 11 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 21,568 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 3,552,064 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 1,488,588 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 14 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 27,200 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,552,400 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 4,946,832 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 20 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 4,152,600 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 47 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 50 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 53 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 56 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 59 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 1,036 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 62 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 68 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 296 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 71 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 592 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 65 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 | 1,372 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 4,032,720 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 1,363,096 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 34,449,072 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 32 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 11,480 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 1,022,720 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 83 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 | 972 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 77 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 79 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 81 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 684 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 74 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 960 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 75 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 960 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 82 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 | 480 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 38 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 | 76,800 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 42 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 37,600 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 45 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 46 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 48 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 49 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 51 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 52 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 54 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 55 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 57 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 576 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 58 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 60 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 61 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 66 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 67 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 384 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 69 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 12,096 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 70 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 12,096 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 63 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 792 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 64 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 | 528 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 72 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 672 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 73 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 672 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 76 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 78 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 80 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 336 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 40 | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 | 27,200 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | 37 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | 0 | 76,800 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 39 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 | 56,640 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 43 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 | 76,800 | 
| app_proof | XorinVmAir | 24 | XorinVmAir | 0 | 6,204,448 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 |  | 8 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 |  | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 |  | 8 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 6, 24>, ModularIsEqualCoreAir<24, 4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 3 | 10 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 12, 12>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 12, 12>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, BitwiseLogicCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, 2, 4, 4, 8, 32, 32>, MultiplicationCoreAir<32, 8> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 6, 6>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapBranchU16AdapterAir<2, 4>, 2, 4, 4, 16>, BranchEqualCoreAir<16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, LessThanCoreAir<16, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, ShiftLogicalCoreAir<16, 16> | 0 |  | 1 | 
| app_proof | XorinVmAir | 0 |  | 8 | 

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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 93 | 4 | 
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
| agg_keygen | 32 | WhirRoundAir | 1 | 31 | 30 | 4 | 
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
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 49 | 6,272 | 
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
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 47 | 188 | 
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
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | ADD | 0 | 200 | 
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | SUB | 0 | 200 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 205 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 20,100 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 61 | KeccakfOpAir | KECCAKF | 0 | 9,470 | 
| app_proof | 63 | XorinVmAir | XORIN | 0 | 9,458 | 
| app_proof | 65 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 31,960 | 
| app_proof | 66 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 84,015 | 
| app_proof | 67 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 82,058 | 
| app_proof | 67 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 21,757 | 
| app_proof | 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGE | 0 | 18 | 
| app_proof | 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 30,573 | 
| app_proof | 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 72,468 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 123,385 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 54,790 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 
| app_proof | 70 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADB | 0 | 1,600 | 
| app_proof | 70 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 0 | 22,741 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADBU | 0 | 7,095 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 0 | 290,210 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREB | 0 | 9,241 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 0 | 306,998 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREW | 0 | 1,618 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SRLW | 0 | 400 | 
| app_proof | 75 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 0 | 28,161 | 
| app_proof | 75 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 0 | 27,340 | 
| app_proof | 76 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 337 | 
| app_proof | 77 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | ADDW | 0 | 758 | 
| app_proof | 77 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 0 | 21,133 | 
| app_proof | 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 126,677 | 
| app_proof | 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 16,863 | 
| app_proof | 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | XOR | 0 | 400 | 
| app_proof | 79 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 0 | 494,337 | 
| app_proof | 79 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 0 | 81,683 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 3 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 1 | 

| group | air_id | air_name | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| root | 0 | RootVerifierPvsAir | prover | 1 | 207 | 207 | 
| root | 1 | UserPvsCommitAir | prover | 8 | 30 | 240 | 
| root | 10 | EqSharpUniReceiverAir | prover | 4 | 17 | 68 | 
| root | 11 | EqUniAir | prover | 4 | 16 | 64 | 
| root | 12 | ExpressionClaimAir | prover | 128 | 32 | 4,096 | 
| root | 13 | InteractionsFoldingAir | prover | 8,192 | 37 | 303,104 | 
| root | 14 | ConstraintsFoldingAir | prover | 4,096 | 25 | 102,400 | 
| root | 15 | EqNegAir | prover | 8 | 40 | 320 | 
| root | 16 | TranscriptAir | prover | 4,096 | 44 | 180,224 | 
| root | 17 | Poseidon2Air<BabyBearParameters>, 1> | prover | 16,384 | 301 | 4,931,584 | 
| root | 18 | MerkleVerifyAir | prover | 8,192 | 37 | 303,104 | 
| root | 19 | ProofShapeAir<4, 8> | prover | 64 | 45 | 2,880 | 
| root | 2 | UserPvsInMemoryAir | prover | 32 | 20 | 640 | 
| root | 20 | PublicValuesAir | prover | 128 | 8 | 1,024 | 
| root | 21 | RangeCheckerAir<8> | prover | 256 | 2 | 512 | 
| root | 22 | GkrInputAir | prover | 1 | 26 | 26 | 
| root | 23 | GkrLayerAir | prover | 32 | 46 | 1,472 | 
| root | 24 | GkrLayerSumcheckAir | prover | 256 | 45 | 11,520 | 
| root | 25 | GkrXiSamplerAir | prover | 1 | 10 | 10 | 
| root | 26 | OpeningClaimsAir | prover | 2,048 | 63 | 129,024 | 
| root | 27 | UnivariateRoundAir | prover | 8 | 27 | 216 | 
| root | 28 | SumcheckRoundsAir | prover | 32 | 57 | 1,824 | 
| root | 29 | StackingClaimsAir | prover | 512 | 35 | 17,920 | 
| root | 3 | SymbolicExpressionAir<BabyBearParameters> | prover | 32,768 | 316 | 10,354,688 | 
| root | 30 | EqBaseAir | prover | 4 | 51 | 204 | 
| root | 31 | EqBitsAir | prover | 4,096 | 16 | 65,536 | 
| root | 32 | WhirRoundAir | prover | 4 | 46 | 184 | 
| root | 33 | SumcheckAir | prover | 16 | 38 | 608 | 
| root | 34 | WhirQueryAir | prover | 128 | 32 | 4,096 | 
| root | 35 | InitialOpenedValuesAir | prover | 8,192 | 89 | 729,088 | 
| root | 36 | NonInitialOpenedValuesAir | prover | 1,024 | 28 | 28,672 | 
| root | 37 | WhirFoldingAir | prover | 2,048 | 31 | 63,488 | 
| root | 38 | FinalPolyMleEvalAir | prover | 256 | 34 | 8,704 | 
| root | 39 | FinalPolyQueryEvalAir | prover | 16,384 | 45 | 737,280 | 
| root | 4 | FractionsFolderAir | prover | 64 | 29 | 1,856 | 
| root | 40 | PowerCheckerAir<2, 32> | prover | 32 | 4 | 128 | 
| root | 41 | ExpBitsLenAir | prover | 8,192 | 16 | 131,072 | 
| root | 5 | UnivariateSumcheckAir | prover | 16 | 24 | 384 | 
| root | 6 | MultilinearSumcheckAir | prover | 128 | 33 | 4,224 | 
| root | 7 | EqNsAir | prover | 32 | 41 | 1,312 | 
| root | 8 | Eq3bAir | prover | 16,384 | 25 | 409,600 | 
| root | 9 | EqSharpUniAir | prover | 4 | 17 | 68 | 

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
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | prover | 0 | 512 | 122 | 62,464 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 256 | 43 | 11,008 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | prover | 0 | 32,768 | 150 | 4,915,200 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 524,288 | 456 | 239,075,328 | 
| app_proof | 61 | KeccakfOpAir | prover | 0 | 16,384 | 284 | 4,653,056 | 
| app_proof | 62 | KeccakfPermAir | prover | 0 | 262,144 | 2,634 | 690,487,296 | 
| app_proof | 63 | XorinVmAir | prover | 0 | 16,384 | 669 | 10,960,896 | 
| app_proof | 65 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 32,768 | 17 | 557,056 | 
| app_proof | 66 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 131,072 | 24 | 3,145,728 | 
| app_proof | 67 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 131,072 | 18 | 2,359,296 | 
| app_proof | 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 131,072 | 32 | 4,194,304 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 6,815,744 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| app_proof | 70 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 32,768 | 46 | 1,507,328 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 1,048,576 | 54 | 56,623,104 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | prover | 0 | 512 | 59 | 30,208 | 
| app_proof | 75 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 0 | 65,536 | 66 | 4,325,376 | 
| app_proof | 76 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 512 | 38 | 19,456 | 
| app_proof | 77 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | prover | 0 | 32,768 | 33 | 1,081,344 | 
| app_proof | 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 262,144 | 46 | 12,058,624 | 
| app_proof | 79 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | prover | 0 | 1,048,576 | 34 | 35,651,584 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 4 | 547 | 2,188 | 
| app_proof | 80 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 82 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,228,800 | 
| app_proof | 83 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1 | 641 | 641 | 
| halo2_keygen | 0 | ProgramAir | prover | 0 | 1 | 10 | 10 | 
| halo2_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| halo2_keygen | 56 | RangeTupleCheckerAir<2> | prover | 0 | 2,097,152 | 3 | 6,291,456 | 
| halo2_keygen | 80 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| halo2_keygen | 82 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| halo2_keygen | 83 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| root_keygen | 0 | ProgramAir | prover | 0 | 1 | 10 | 10 | 
| root_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| root_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| root_keygen | 26 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| root_keygen | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| root_keygen | 29 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| root_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| root_keygen | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 

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
| app_proof | 50 | VmAirWrapper<Rv64VecHeapU16AdapterAir<2, 4, 4>, 2, 4, 4, 4, 16, 16>, AddSubCoreAir<16, 16> | 0 | 400 | 112 | 122,000 | 34,160 | 195,200 | 54,656 | 48,800 | 13,664 | 1,189,000 | 332,920 | 32,800 | 9,184 | 
| app_proof | 55 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 205 | 51 | 22,038 | 5,482 | 35,260 | 8,772 | 8,815 | 2,193 | 230,369 | 57,311 | 6,355 | 1,581 | 
| app_proof | 56 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| app_proof | 59 | Sha2MainAir<Sha256Config> | 0 | 20,100 | 12,668 | 15,075,000 | 9,501,000 | 12,060,000 | 7,600,800 | 3,015,000 | 1,900,200 | 61,933,125 | 39,033,275 | 1,708,500 | 1,076,780 | 
| app_proof | 6 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,103 | 1,367 | 6,564 | 2,188 | 1,641 | 547 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 60 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 341,700 | 182,588 | 779,076,000 | 416,300,640 | 623,260,800 | 333,040,512 | 155,815,200 | 83,260,128 | 359,212,125 | 191,945,635 | 9,909,300 | 5,295,052 | 
| app_proof | 61 | KeccakfOpAir | 0 | 9,470 | 6,914 | 6,723,700 | 4,908,940 | 10,757,920 | 7,854,304 | 2,689,480 | 1,963,576 | 37,761,625 | 27,569,575 | 1,041,700 | 760,540 | 
| app_proof | 62 | KeccakfPermAir | 0 | 227,280 | 34,864 | 2,993,277,600 | 459,158,880 | 2,394,622,080 | 367,327,104 | 598,655,520 | 91,831,776 | 16,477,800 | 2,527,640 | 454,560 | 69,728 | 
| app_proof | 63 | XorinVmAir | 0 | 9,458 | 6,926 | 15,818,505 | 11,583,735 | 25,309,608 | 18,533,976 | 6,327,402 | 4,633,494 | 122,398,343 | 89,631,097 | 3,376,506 | 2,472,582 | 
| app_proof | 65 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 31,960 | 808 | 1,358,300 | 34,340 | 2,173,280 | 54,944 | 543,320 | 13,736 | 16,219,700 | 410,060 | 447,440 | 11,312 | 
| app_proof | 66 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 84,015 | 47,057 | 5,040,900 | 2,823,420 | 8,065,440 | 4,517,472 | 2,016,360 | 1,129,368 | 45,683,157 | 25,587,243 | 1,260,225 | 705,855 | 
| app_proof | 67 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 103,815 | 27,257 | 4,671,675 | 1,226,565 | 7,474,680 | 1,962,504 | 1,868,670 | 490,626 | 45,159,525 | 11,856,795 | 1,245,780 | 327,084 | 
| app_proof | 68 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 103,059 | 28,013 | 8,244,720 | 2,241,040 | 13,191,552 | 3,585,664 | 3,297,888 | 896,416 | 52,302,443 | 14,216,597 | 1,442,826 | 392,182 | 
| app_proof | 69 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 178,175 | 83,969 | 11,581,375 | 5,457,985 | 18,530,200 | 8,732,776 | 4,632,550 | 2,183,194 | 71,047,282 | 33,482,638 | 1,959,925 | 923,659 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 1 |  | 1,603 |  | 2,564 |  | 641 |  | 18,162 |  | 501 |  | 
| app_proof | 70 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 24,341 | 8,427 | 2,799,215 | 969,105 | 4,478,744 | 1,550,568 | 1,119,686 | 387,642 | 19,411,948 | 6,720,532 | 535,502 | 185,394 | 
| app_proof | 71 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 615,162 | 433,414 | 83,046,870 | 58,510,890 | 132,874,992 | 93,617,424 | 33,218,748 | 23,404,356 | 557,490,563 | 392,781,437 | 15,379,050 | 10,835,350 | 
| app_proof | 73 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 400 | 112 | 59,000 | 16,520 | 94,400 | 26,432 | 23,600 | 6,608 | 319,000 | 89,320 | 8,800 | 2,464 | 
| app_proof | 75 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 55,501 | 10,035 | 9,157,665 | 1,655,775 | 14,652,264 | 2,649,240 | 3,663,066 | 662,310 | 50,297,782 | 9,094,218 | 1,387,525 | 250,875 | 
| app_proof | 76 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 337 | 175 | 32,015 | 16,625 | 51,224 | 26,600 | 12,806 | 6,650 | 232,109 | 120,531 | 6,403 | 3,325 | 
| app_proof | 77 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 21,891 | 10,877 | 1,806,008 | 897,352 | 2,889,612 | 1,435,764 | 722,403 | 358,941 | 15,077,427 | 7,491,533 | 415,929 | 206,663 | 
| app_proof | 78 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 143,940 | 118,204 | 16,553,100 | 13,593,460 | 26,484,960 | 21,749,536 | 6,621,240 | 5,437,384 | 125,227,800 | 102,837,480 | 3,454,560 | 2,836,896 | 
| app_proof | 79 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 576,020 | 472,556 | 48,961,700 | 40,167,260 | 78,338,720 | 64,267,616 | 19,584,680 | 16,066,904 | 417,614,500 | 342,603,100 | 11,520,400 | 9,451,120 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 3 | 1 | 4,103 | 1,367 | 6,564 | 2,188 | 1,641 | 547 | 50,460 | 16,820 | 1,392 | 464 | 
| app_proof | 80 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 82 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 8,462 | 7,922 | 6,346,500 | 5,941,500 | 10,154,400 | 9,506,400 | 2,538,600 | 2,376,600 | 306,748 | 287,172 | 8,462 | 7,922 | 
| app_proof | 83 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 9 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 1 |  | 1,603 |  | 2,564 |  | 641 |  | 18,162 |  | 501 |  | 
| halo2_keygen | 0 | ProgramAir | 0 | 1 |  | 25 |  | 40 |  | 10 |  | 37 |  | 1 |  | 
| halo2_keygen | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | 0 | 32 |  | 1,680 |  | 2,688 |  | 672 |  | 4,640 |  | 128 |  | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | 0 | 78 | 50 | 12,870 | 8,250 | 10,296 | 6,600 | 2,574 | 1,650 | 11,310 | 7,250 | 312 | 200 | 
| halo2_keygen | 56 | RangeTupleCheckerAir<2> | 0 | 2,097,152 |  | 31,457,280 |  | 25,165,824 |  | 6,291,456 |  | 76,021,760 |  | 2,097,152 |  | 
| halo2_keygen | 80 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| halo2_keygen | 82 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 110 | 18 | 82,500 | 13,500 | 132,000 | 21,600 | 33,000 | 5,400 | 3,988 | 652 | 110 | 18 | 
| halo2_keygen | 83 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| root_keygen | 0 | ProgramAir | 0 | 1 |  | 25 |  | 40 |  | 10 |  | 37 |  | 1 |  | 
| root_keygen | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| root_keygen | 2 | PersistentBoundaryAir<8> | 0 | 32 |  | 1,680 |  | 2,688 |  | 672 |  | 4,640 |  | 128 |  | 
| root_keygen | 26 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| root_keygen | 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 110 | 18 | 82,500 | 13,500 | 132,000 | 21,600 | 33,000 | 5,400 | 3,988 | 652 | 110 | 18 | 
| root_keygen | 29 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| root_keygen | 3 | MemoryMerkleAir<8> | 0 | 78 | 50 | 12,870 | 8,250 | 10,296 | 6,600 | 2,574 | 1,650 | 11,310 | 7,250 | 312 | 200 | 
| root_keygen | 9 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 3 | 
| halo2_keygen | interpreter | 0 | 
| root_keygen | interpreter | 0 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 15 | 159 | 15 | 5 | 1 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 10 | 101 | 10 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 91 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 84 | 370 | 84 | 18 | 9 | 18 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,577,981 | 143 | 38 | 0 | 0 | 62 | 20 | 19 | 29 | 12 | 0 | 42 | 34 | 7 | 1 | 6 | 38 | 38 | 62 | 0 | 1 | 11 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 90 | 16 | 0 | 0 | 48 | 17 | 17 | 20 | 9 | 0 | 26 | 19 | 6 | 1 | 5 | 16 | 16 | 48 | 0 | 1 | 9 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 81 | 11 | 0 | 0 | 47 | 17 | 16 | 18 | 10 | 0 | 22 | 16 | 5 | 0 | 4 | 11 | 11 | 47 | 0 | 1 | 9 | 0 | 0 | 
| leaf | 0 | prover | 167,484,093 | 285 | 81 | 0 | 0 | 124 | 40 | 39 | 31 | 52 | 0 | 79 | 67 | 12 | 4 | 7 | 81 | 81 | 124 | 0 | 2 | 51 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,723,587 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 36,896,643 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | halo2_section_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | prover |  |  | 3 |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  | 3 |  |  |  |  | 
| halo2_keygen | ifft_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 95 |  |  |  | 
| halo2_keygen | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_keygen | lagrange_to_coeff_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 95 |  |  |  | 
| halo2_keygen | multiexp_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 16 |  |  |  | 
| halo2_outer | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96 |  |  |  | 
| halo2_outer | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 49 |  |  |  | 
| halo2_outer | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5,174 |  |  |  | 
| halo2_outer | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 99 |  |  |  | 
| halo2_outer | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 47 |  |  |  | 
| halo2_outer | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 11 |  |  |  | 
| halo2_outer | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,599 |  |  |  | 
| halo2_outer | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_outer | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 176 |  |  |  | 
| halo2_outer | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 113 |  |  |  | 
| halo2_outer | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_outer | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 102 |  |  |  | 
| halo2_outer | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 136 |  |  |  | 
| halo2_outer | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 273 |  |  |  | 
| halo2_outer | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 72 |  |  |  | 
| halo2_outer | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 539 |  |  |  | 
| halo2_outer | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 67 |  |  |  | 
| halo2_outer | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 
| halo2_outer | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 666 |  |  |  | 
| halo2_outer | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  | 
| halo2_outer | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,530 |  |  |  | 
| halo2_outer | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 224 |  |  |  | 
| halo2_outer | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 802 |  |  |  | 
| halo2_outer | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 666 |  |  |  | 
| halo2_outer | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 136 |  |  |  | 
| halo2_outer | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,930 |  |  |  | 
| halo2_outer | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 264 |  |  |  | 
| halo2_outer | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 407 |  |  |  | 
| halo2_outer | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 54 |  |  |  | 
| halo2_outer | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 176 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 291 |  |  |  | 
| halo2_outer | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 72 |  |  |  | 
| halo2_outer | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96 |  |  |  | 
| halo2_outer | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 13 |  |  |  | 
| halo2_outer | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 264 |  |  |  | 
| halo2_outer | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 72 |  |  |  | 
| halo2_outer | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 789 |  |  |  | 
| halo2_wrapper | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_wrapper | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,350 |  |  |  | 
| halo2_wrapper | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 22 |  |  |  | 
| halo2_wrapper | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |  | 
| halo2_wrapper | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 360 |  |  |  | 
| halo2_wrapper | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_wrapper | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 89 |  |  |  | 
| halo2_wrapper | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 46 |  |  |  | 
| halo2_wrapper | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_wrapper | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 35 |  |  |  | 
| halo2_wrapper | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 89 |  |  |  | 
| halo2_wrapper | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 69 |  |  |  | 
| halo2_wrapper | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 80 |  |  |  | 
| halo2_wrapper | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 17 |  |  |  | 
| halo2_wrapper | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 128 |  |  |  | 
| halo2_wrapper | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_wrapper | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 266 |  |  |  | 
| halo2_wrapper | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 81 |  |  |  | 
| halo2_wrapper | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 164 |  |  |  | 
| halo2_wrapper | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 128 |  |  |  | 
| halo2_wrapper | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 35 |  |  |  | 
| halo2_wrapper | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 461 |  |  |  | 
| halo2_wrapper | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 158 |  |  |  | 
| halo2_wrapper | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 210 |  |  |  | 
| halo2_wrapper | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  | 
| halo2_wrapper | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 89 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 159 |  |  |  | 
| halo2_wrapper | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 69 |  |  |  | 
| halo2_wrapper | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  |  | 
| halo2_wrapper | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 158 |  |  |  | 
| halo2_wrapper | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 36 |  |  |  | 
| halo2_wrapper | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 110 |  |  |  | 
| root | prover | 18,533,671 | 772 | 398 | 0 | 0 | 69 | 19 | 17 | 30 | 20 | 0 | 303 | 296 | 7 | 1 | 5 | 399 | 398 | 69 | 0 | 76 |  | 11 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,089,189,244 | 1,591 | 578 | 0 | 0 | 715 | 488 | 487 | 125 | 100 | 0 | 297 | 235 | 61 | 28 | 32 | 578 | 578 | 715 | 0 | 1 | 99 | 0 | 0 | 
| halo2_keygen | prover | 0 | 8,531,435 | 40 | 4 | 0 | 0 | 25 | 6 | 6 | 5 | 13 | 0 | 11 | 8 | 2 | 0 | 2 | 4 | 4 | 25 | 0 | 1 | 12 | 0 | 0 | 
| root_keygen | prover | 0 | 5,385,707 | 101 | 26 | 0 | 0 | 20 | 5 | 5 | 4 | 10 | 0 | 53 | 51 | 2 | 0 | 2 | 26 | 26 | 20 | 0 | 3 | 9 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 94,208,254 | 2,013,265,921 | 
| halo2_keygen | prover | 0 | 0 | 2,490,671 | 2,013,265,921 | 
| root_keygen | prover | 0 | 0 | 1,442,095 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,535 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 1 | 83 | 1,932 | 83 | 0 | 10,266,403,236 | 2,848,104,492 | 13,114,507,728 | 2,097,152 | 0 | 2 | 257 | 1,979,971 | 35.43 | 
| halo2_keygen | 0 | 0 | 1 | 21 | 65 | 21 | 0 | 126,617,121 | 36,102 | 126,653,223 | 2,097,152 | 0 | 1 | 2 | 1 | 3.57 | 
| root_keygen | 0 | 0 | 3 | 25 | 128 | 25 | 0 | 76,023,329 | 36,102 | 76,059,431 | 2,097,152 | 0 | 3 | 1 | 1 | 1.85 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 3 | 0 | 3 | 3 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/987f1fb4daebdebecc4e634204e0d3934cea6990

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29313052571)
