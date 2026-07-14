| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  11.09 |  11.09 |  11.09 |
| app_proof |  0.23 |  0.23 |  0.23 |
| leaf |  0.17 |  0.17 |  0.17 |
| internal_for_leaf |  0.16 |  0.16 |  0.16 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |
| root |  1.44 |  1.44 |  1.44 |
| halo2_outer |  7.05 |  7.05 |  7.05 |
| halo2_wrapper |  1.81 |  1.81 |  1.81 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  220 |  220 |  220 |  220 |
| `compile_metered_time_ms` |  4 |  4 |  4 |  4 |
| `execute_metered_time_ms` |  5 | -          | -          | -          |
| `execute_metered_insns` |  112,210 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  19.09 | -          |  19.09 |  19.09 |
| `execute_preflight_insns` |  112,210 |  112,210 |  112,210 |  112,210 |
| `execute_preflight_time_ms` |  17 |  17 |  17 |  17 |
| `execute_preflight_insn_mi/s` |  13.11 | -          |  13.11 |  13.11 |
| `trace_gen_time_ms   ` |  52 |  52 |  52 |  52 |
| `set_initial_memory_time_ms` |  0 |  0 |  0 |  0 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  150 |  150 |  150 |  150 |
| `prover.main_trace_commit_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints_time_ms` |  107 |  107 |  107 |  107 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  73 |  73 |  73 |  73 |
| `prover.rap_constraints.mle_rounds_time_ms` |  15 |  15 |  15 |  15 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  173 |  173 |  173 |  173 |
| `execute_preflight_time_ms` |  6 |  6 |  6 |  6 |
| `trace_gen_time_ms   ` |  27 |  27 |  27 |  27 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  146 |  146 |  146 |  146 |
| `prover.main_trace_commit_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints_time_ms` |  78 |  78 |  78 |  78 |
| `prover.openings_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints.logup_gkr_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.round0_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  29 |  29 |  29 |  29 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  161 |  161 |  161 |  161 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  15 |  15 |  15 |  15 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  146 |  146 |  146 |  146 |
| `prover.main_trace_commit_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints_time_ms` |  73 |  73 |  73 |  73 |
| `prover.openings_time_ms` |  39 |  39 |  39 |  39 |
| `prover.rap_constraints.logup_gkr_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  34 |  34 |  34 |  34 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  29 |  29 |  29 |  29 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  118 |  118 |  118 |  118 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  107 |  107 |  107 |  107 |
| `prover.main_trace_commit_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints_time_ms` |  56 |  56 |  56 |  56 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  114 |  114 |  114 |  114 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  104 |  104 |  104 |  104 |
| `prover.main_trace_commit_time_ms` |  15 |  15 |  15 |  15 |
| `prover.rap_constraints_time_ms` |  53 |  53 |  53 |  53 |
| `prover.openings_time_ms` |  35 |  35 |  35 |  35 |
| `prover.rap_constraints.logup_gkr_time_ms` |  10 |  10 |  10 |  10 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  28 |  28 |  28 |  28 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,439 |  1,439 |  1,439 |  1,439 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,340 |  1,340 |  1,340 |  1,340 |
| `prover.main_trace_commit_time_ms` |  714 |  714 |  714 |  714 |
| `prover.rap_constraints_time_ms` |  103 |  103 |  103 |  103 |
| `prover.openings_time_ms` |  522 |  522 |  522 |  522 |
| `prover.rap_constraints.logup_gkr_time_ms` |  48 |  48 |  48 |  48 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  32 |  32 |  32 |  32 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  513 |  513 |  513 |  513 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,048 |  7,048 |  7,048 |  7,048 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,812 |  1,812 |  1,812 |  1,812 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94/ecrecover_e2e-6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 2.38 | app_proof.0 |
| set initial memory | 2.38 | app_proof.0 |
| prover.openings | 2.30 | root.prover |
| prover.merkle_tree | 2.30 | root.prover |
| prover.prove_whir_opening | 2.30 | root.prover |
| prover.rs_code_matrix | 2.30 | root.prover |
| prover.stacked_commit | 2.11 | root.prover |
| prover.rap_constraints | 1.54 | internal_for_leaf.0.prover |
| frac_sumcheck.gkr_rounds | 1.46 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 1.46 | leaf.0.prover |
| frac_sumcheck.segment_tree | 1.44 | leaf.0.prover |
| prover.gkr_input_evals | 1.44 | leaf.0.prover |
| prover.batch_constraints.round0 | 1.17 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.17 | leaf.0.prover |
| prover.before_gkr_input_evals | 1.04 | leaf.0.prover |
| tracegen.exp_bits_len | 1.03 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.03 | leaf.0 |
| tracegen.pow_checker | 1.03 | leaf.0 |
| tracegen.whir_folding | 0.97 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.97 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.97 | leaf.0 |
| tracegen.proof_shape | 0.96 | leaf.0 |
| tracegen.public_values | 0.96 | leaf.0 |
| tracegen.range_checker | 0.96 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 159 | 28 | 21 | 0 | 0 | 0 | 2 | 4 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 0 | RootVerifierPvsAir |  | 109 | 37 | 4 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 12 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 13 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 14 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 15 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 23 | 3 | 
| 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 32 | 3 | 
| 18 | KeccakfOpAir |  | 110 | 27 | 2 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 19 | KeccakfPermAir | 1 | 2 | 3,183 | 3 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 85 | 4 | 
| 19 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 20 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 20 | XorinVmAir |  | 357 | 92 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 21 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 21 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 23 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> |  | 19 | 16 | 3 | 
| 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 10 | 3 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 24 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 25 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> |  | 20 | 21 | 3 | 
| 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 26 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 27 | PhantomAir |  | 3 | 1 | 2 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 23 | 3 | 
| 28 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 32 | 3 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 29 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 29 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftRightArithmeticCoreAir<2, 16> |  | 23 | 69 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 30 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> |  | 22 | 108 | 3 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 31 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<4, 16> |  | 26 | 86 | 3 | 
| 32 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> |  | 25 | 139 | 3 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 33 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 34 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> |  | 19 | 16 | 3 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> |  | 24 | 16 | 3 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 36 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> |  | 20 | 21 | 3 | 
| 37 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 38 | PhantomAir |  | 3 | 1 | 2 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 39 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> |  | 464 | 280 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 40 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> |  | 501 | 257 | 2 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 195 | 117 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> |  | 131 | 85 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 9 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> |  | 51 | 118 | 3 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 61 |  |  |  |  |  |  |  | 262 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  | 220 |  |  |  |  |  |  |  | 5 | 112,210 | 19.09 | 0 |  |  | 235 |  | 
| halo2_keygen |  |  |  |  |  |  |  | 81 |  | 70,861 |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 84 |  | 
| halo2_outer |  |  | 7,048 |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  | 1,812 |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  | 161 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 162 | 
| internal_recursive.0 |  |  |  |  |  | 118 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 118 | 
| internal_recursive.1 |  |  |  |  |  | 114 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 114 | 
| leaf |  |  |  |  | 173 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 173 | 
| root | 98 | 10 | 1,439 | 9 |  |  | 1,439 |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 | 0 |  | 1,439 | 
| root_keygen |  |  |  |  |  |  |  | 186 |  |  |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 188 |  | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 26 | KeccakfOpAir | 0 | 1,600 | 
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 220 | 
| app_proof | Rv64HintStoreAir | 23 | Rv64HintStoreAir | 0 | 3,004 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 9 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 938,688 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 8 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 1,348,480 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 11 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 51,392 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 12 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 772,288 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 10 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 239,428 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 14 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 9,520 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 482,448 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 40,752 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 20 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 94,600 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 35 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 474,192 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 38 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 4,588 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 21 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 84,624 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 43,008 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 2,045,064 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 28 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 896 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 72,896 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 40 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 289,788 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 33 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2,688 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 34 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 4,032 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 36 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2,112 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 37 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 2,112 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 39 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 243,936 | 
| app_proof | XorinVmAir | 24 | XorinVmAir | 0 | 3,280 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 |  | 1 | 
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 |  | 5 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 |  | 10 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 |  | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 |  | 5 | 
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
| agg_keygen | 19 | ProofShapeAir<4, 8> | 1 | 78 | 89 | 4 | 
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
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 9,863,168 | 
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
| leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 4,096 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 606,208 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 204,800 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 640 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 9,863,168 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,212,416 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 45 | 2,880 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 256 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 26 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 1,472 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 11,520 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 10 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 516,096 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 864 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 1,824 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 71,680 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 7,864,320 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 408 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 262,144 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 47 | 188 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 608 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 16,384 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 729,088 | 
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
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 1,638,400 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 272 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 21 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 14 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 16 | 
| app_proof | 18 | KeccakfOpAir | KECCAKF | 0 | 5 | 
| app_proof | 20 | XorinVmAir | XORIN | 0 | 5 | 
| app_proof | 21 | Rv64HintStoreAir | HINT_BUFFER | 0 | 11 | 
| app_proof | 21 | Rv64HintStoreAir | HINT_STORED | 0 | 11 | 
| app_proof | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 2,278 | 
| app_proof | 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 1,763 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 2,078 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 287 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGE | 0 | 8 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 82 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLT | 0 | 70 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 689 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 6,542 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 3,509 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADB | 0 | 576 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 0 | 192 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADBU | 0 | 14,093 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 0 | 9,666 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREB | 0 | 2,417 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 0 | 10,328 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREH | 0 | 5 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREW | 0 | 10 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SLLW | 0 | 60 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | SRLW | 0 | 80 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SLL | 0 | 10,458 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | SRL | 0 | 1,609 | 
| app_proof | 33 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 803 | 
| app_proof | 34 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | ADDW | 0 | 3,254 | 
| app_proof | 34 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | SUBW | 0 | 267 | 
| app_proof | 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | AND | 0 | 3,773 | 
| app_proof | 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | OR | 0 | 9,891 | 
| app_proof | 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | XOR | 0 | 1,003 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | ADD | 0 | 20,309 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | SUB | 0 | 761 | 
| app_proof | 38 | PhantomAir | PHANTOM | 0 | 11 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | EcDouble | 0 | 1,271 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | EcAddNe | 0 | 726 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 30 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularMulDiv | 0 | 11 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | ModularAddSub | 0 | 11 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | IS_EQ | 0 | 3,203 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | SETUP_ISEQ | 0 | 1 | 

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
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 327,680 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 32 | 268 | 8,576 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 16 | 204 | 3,264 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 16 | 43 | 688 | 
| app_proof | 17 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 18 | KeccakfOpAir | prover | 0 | 8 | 284 | 2,272 | 
| app_proof | 19 | KeccakfPermAir | prover | 0 | 128 | 2,634 | 337,152 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 2,048 | 21 | 43,008 | 
| app_proof | 20 | XorinVmAir | prover | 0 | 8 | 669 | 5,352 | 
| app_proof | 21 | Rv64HintStoreAir | prover | 0 | 128 | 27 | 3,456 | 
| app_proof | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 4,096 | 17 | 69,632 | 
| app_proof | 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 2,048 | 24 | 49,152 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 4,096 | 18 | 73,728 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 16,384 | 26 | 425,984 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 1,024 | 46 | 47,104 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 65,536 | 54 | 3,538,944 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 33 | 135,168 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | prover | 0 | 256 | 59 | 15,104 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | prover | 0 | 16,384 | 66 | 1,081,344 | 
| app_proof | 33 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 1,024 | 38 | 38,912 | 
| app_proof | 34 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | prover | 0 | 4,096 | 33 | 135,168 | 
| app_proof | 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | prover | 0 | 16,384 | 46 | 753,664 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | prover | 0 | 32,768 | 34 | 1,114,112 | 
| app_proof | 37 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 38 | PhantomAir | prover | 0 | 16 | 6 | 96 | 
| app_proof | 39 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 2,048 | 300 | 614,400 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2,048 | 547 | 1,120,256 | 
| app_proof | 40 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 641 | 656,384 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 32 | 116 | 3,712 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 16 | 268 | 4,288 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | prover | 0 | 16 | 204 | 3,264 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | prover | 0 | 4,096 | 116 | 475,136 | 
| halo2_keygen | 0 | ProgramAir | prover | 0 | 1 | 10 | 10 | 
| halo2_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| halo2_keygen | 17 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| halo2_keygen | 37 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| halo2_keygen | 39 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| halo2_keygen | 40 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
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
| app_proof | 0 | ProgramAir | 0 | 17,293 | 15,475 | 432,325 | 386,875 | 691,720 | 619,000 | 172,930 | 154,750 | 626,872 | 560,968 | 17,293 | 15,475 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 21 | 11 | 14,070 | 7,370 | 22,512 | 11,792 | 5,628 | 2,948 | 148,444 | 77,756 | 4,095 | 2,145 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 14 | 2 | 7,140 | 1,020 | 11,424 | 1,632 | 2,856 | 408 | 66,483 | 9,497 | 1,834 | 262 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 16 |  | 1,720 |  | 2,752 |  | 688 |  | 17,980 |  | 496 |  | 
| app_proof | 17 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 18 | KeccakfOpAir | 0 | 5 | 3 | 3,550 | 2,130 | 5,680 | 3,408 | 1,420 | 852 | 19,938 | 11,962 | 550 | 330 | 
| app_proof | 19 | KeccakfPermAir | 0 | 120 | 8 | 1,580,400 | 105,360 | 1,264,320 | 84,288 | 316,080 | 21,072 | 8,700 | 580 | 240 | 16 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 1,826 | 222 | 95,865 | 11,655 | 153,384 | 18,648 | 38,346 | 4,662 | 264,770 | 32,190 | 7,304 | 888 | 
| app_proof | 20 | XorinVmAir | 0 | 5 | 3 | 8,363 | 5,017 | 13,380 | 8,028 | 3,345 | 2,007 | 64,707 | 38,823 | 1,785 | 1,071 | 
| app_proof | 21 | Rv64HintStoreAir | 0 | 115 | 13 | 15,525 | 1,755 | 12,420 | 1,404 | 3,105 | 351 | 70,869 | 8,011 | 1,955 | 221 | 
| app_proof | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 2,278 | 1,818 | 96,815 | 77,265 | 154,904 | 123,624 | 38,726 | 30,906 | 1,156,085 | 922,635 | 31,892 | 25,452 | 
| app_proof | 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 1,763 | 285 | 105,780 | 17,100 | 169,248 | 27,360 | 42,312 | 6,840 | 958,632 | 154,968 | 26,445 | 4,275 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 2,365 | 1,731 | 106,425 | 77,895 | 170,280 | 124,632 | 42,570 | 31,158 | 1,028,775 | 752,985 | 28,380 | 20,772 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 849 | 175 | 67,920 | 14,000 | 108,672 | 22,400 | 27,168 | 5,600 | 430,868 | 88,812 | 11,886 | 2,450 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 10,051 | 6,333 | 653,315 | 411,645 | 1,045,304 | 658,632 | 261,326 | 164,658 | 4,007,837 | 2,525,283 | 110,561 | 69,663 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 768 | 256 | 88,320 | 29,440 | 141,312 | 47,104 | 35,328 | 11,776 | 612,480 | 204,160 | 16,896 | 5,632 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 36,519 | 29,017 | 4,930,065 | 3,917,295 | 7,888,104 | 6,267,672 | 1,972,026 | 1,566,918 | 33,095,344 | 26,296,656 | 912,975 | 725,425 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 2,082 | 2,014 | 343,530 | 332,310 | 274,824 | 265,848 | 68,706 | 66,462 | 301,890 | 292,030 | 8,328 | 8,056 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftLogicalCoreAir<2, 16> | 0 | 140 | 116 | 20,650 | 17,110 | 33,040 | 27,376 | 8,260 | 6,844 | 111,650 | 92,510 | 3,080 | 2,552 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalCoreAir<4, 16> | 0 | 12,067 | 4,317 | 1,991,055 | 712,305 | 3,185,688 | 1,139,688 | 796,422 | 284,922 | 10,935,719 | 3,912,281 | 301,675 | 107,925 | 
| app_proof | 33 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 803 | 221 | 76,285 | 20,995 | 122,056 | 33,592 | 30,514 | 8,398 | 553,067 | 152,213 | 15,257 | 4,199 | 
| app_proof | 34 | VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubCoreAir<2, 16> | 0 | 3,521 | 575 | 290,483 | 47,437 | 464,772 | 75,900 | 116,193 | 18,975 | 2,425,089 | 396,031 | 66,899 | 10,925 | 
| app_proof | 35 | VmAirWrapper<Rv64BaseAluAdapterAir, BitwiseLogicCoreAir<8, 8> | 0 | 14,667 | 1,717 | 1,686,705 | 197,455 | 2,698,728 | 315,928 | 674,682 | 78,982 | 12,760,290 | 1,493,790 | 352,008 | 41,208 | 
| app_proof | 36 | VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<4, 16> | 0 | 21,070 | 11,698 | 1,790,950 | 994,330 | 2,865,520 | 1,590,928 | 716,380 | 397,732 | 15,275,750 | 8,481,050 | 421,400 | 233,960 | 
| app_proof | 37 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 38 | PhantomAir | 0 | 11 | 5 | 165 | 75 | 264 | 120 | 66 | 30 | 1,197 | 543 | 33 | 15 | 
| app_proof | 39 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 3,908 | 188 | 2,931,000 | 141,000 | 4,689,600 | 225,600 | 1,172,400 | 56,400 | 141,665 | 6,815 | 3,908 | 188 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8>, FieldExpressionCoreAir> | 0 | 1,271 | 777 | 1,738,093 | 1,062,547 | 2,780,948 | 1,700,076 | 695,237 | 425,019 | 21,378,220 | 13,069,140 | 589,744 | 360,528 | 
| app_proof | 40 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8>, FieldExpressionCoreAir> | 0 | 726 | 298 | 1,163,415 | 477,545 | 1,861,464 | 764,072 | 465,366 | 191,018 | 13,185,068 | 5,412,052 | 363,726 | 149,298 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 31 | 1 | 8,990 | 290 | 14,384 | 464 | 3,596 | 116 | 57,312 | 1,848 | 1,581 | 51 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 11 | 5 | 7,370 | 3,350 | 11,792 | 5,360 | 2,948 | 1,340 | 77,757 | 35,343 | 2,145 | 975 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4>, FieldExpressionCoreAir> | 0 | 11 | 5 | 5,610 | 2,550 | 8,976 | 4,080 | 2,244 | 1,020 | 52,237 | 23,743 | 1,441 | 655 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModU16AdapterAir<2, 4, 16>, ModularIsEqualCoreAir<16, 4, 16> | 0 | 3,204 | 892 | 929,160 | 258,680 | 1,486,656 | 413,888 | 371,664 | 103,472 | 5,923,395 | 1,649,085 | 163,404 | 45,492 | 
| halo2_keygen | 0 | ProgramAir | 0 | 1 |  | 25 |  | 40 |  | 10 |  | 37 |  | 1 |  | 
| halo2_keygen | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| halo2_keygen | 17 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | 0 | 32 |  | 1,680 |  | 2,688 |  | 672 |  | 4,640 |  | 128 |  | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | 0 | 78 | 50 | 12,870 | 8,250 | 10,296 | 6,600 | 2,574 | 1,650 | 11,310 | 7,250 | 312 | 200 | 
| halo2_keygen | 37 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| halo2_keygen | 39 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 110 | 18 | 82,500 | 13,500 | 132,000 | 21,600 | 33,000 | 5,400 | 3,988 | 652 | 110 | 18 | 
| halo2_keygen | 40 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
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
| app_proof | interpreter | 4 | 
| halo2_keygen | interpreter | 0 | 
| root_keygen | interpreter | 0 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 15 | 161 | 15 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 118 | 10 | 1 | 0 | 2 | 0 | 1 | 
| internal_recursive.1 | 1 | 10 | 114 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 27 | 173 | 26 | 6 | 0 | 6 | 0 | 0 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 26,560,317 | 146 | 33 | 0 | 0 | 73 | 25 | 24 | 34 | 14 | 0 | 39 | 29 | 9 | 2 | 7 | 33 | 33 | 73 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 107 | 20 | 0 | 0 | 56 | 21 | 20 | 23 | 11 | 0 | 30 | 22 | 8 | 1 | 6 | 20 | 20 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 104 | 15 | 0 | 0 | 53 | 20 | 19 | 21 | 10 | 0 | 35 | 28 | 7 | 1 | 6 | 15 | 15 | 53 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 36,202,813 | 146 | 29 | 0 | 0 | 78 | 32 | 31 | 23 | 23 | 0 | 38 | 29 | 8 | 2 | 6 | 29 | 29 | 78 | 0 | 3 | 22 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,803 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,678,659 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | halo2_section_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | prover |  |  | 6 |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  | 6 |  |  |  |  | 
| halo2_keygen | ifft_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 119 |  |  |  | 
| halo2_keygen | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_keygen | lagrange_to_coeff_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 119 |  |  |  | 
| halo2_keygen | multiexp_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_outer | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 44 |  |  |  | 
| halo2_outer | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 46 |  |  |  | 
| halo2_outer | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 6,182 |  |  |  | 
| halo2_outer | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 45 |  |  |  | 
| halo2_outer | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 56 |  |  |  | 
| halo2_outer | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 34 |  |  |  | 
| halo2_outer | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,659 |  |  |  | 
| halo2_outer | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 11 |  |  |  | 
| halo2_outer | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 271 |  |  |  | 
| halo2_outer | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 133 |  |  |  | 
| halo2_outer | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 165 |  |  |  | 
| halo2_outer | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 175 |  |  |  | 
| halo2_outer | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 315 |  |  |  | 
| halo2_outer | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94 |  |  |  | 
| halo2_outer | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 612 |  |  |  | 
| halo2_outer | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 68 |  |  |  | 
| halo2_outer | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |  | 
| halo2_outer | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 825 |  |  |  | 
| halo2_outer | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_outer | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,892 |  |  |  | 
| halo2_outer | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 345 |  |  |  | 
| halo2_outer | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,001 |  |  |  | 
| halo2_outer | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 825 |  |  |  | 
| halo2_outer | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 175 |  |  |  | 
| halo2_outer | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,096 |  |  |  | 
| halo2_outer | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 341 |  |  |  | 
| halo2_outer | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 483 |  |  |  | 
| halo2_outer | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 57 |  |  |  | 
| halo2_outer | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 59 |  |  |  | 
| halo2_outer | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 271 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 29 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 337 |  |  |  | 
| halo2_outer | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94 |  |  |  | 
| halo2_outer | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 44 |  |  |  | 
| halo2_outer | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 20 |  |  |  | 
| halo2_outer | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 341 |  |  |  | 
| halo2_outer | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97 |  |  |  | 
| halo2_outer | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 930 |  |  |  | 
| halo2_wrapper | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 16 |  |  |  | 
| halo2_wrapper | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,566 |  |  |  | 
| halo2_wrapper | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 27 |  |  |  | 
| halo2_wrapper | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_wrapper | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 414 |  |  |  | 
| halo2_wrapper | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 
| halo2_wrapper | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 104 |  |  |  | 
| halo2_wrapper | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 60 |  |  |  | 
| halo2_wrapper | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 40 |  |  |  | 
| halo2_wrapper | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 104 |  |  |  | 
| halo2_wrapper | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 93 |  |  |  | 
| halo2_wrapper | new_gpu_thread.instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  | 
| halo2_wrapper | permutation quotient poly part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 158 |  |  |  | 
| halo2_wrapper | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_wrapper | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 264 |  |  |  | 
| halo2_wrapper | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 91 |  |  |  | 
| halo2_wrapper | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 206 |  |  |  | 
| halo2_wrapper | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 158 |  |  |  | 
| halo2_wrapper | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 542 |  |  |  | 
| halo2_wrapper | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 213 |  |  |  | 
| halo2_wrapper | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 239 |  |  |  | 
| halo2_wrapper | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 20 |  |  |  | 
| halo2_wrapper | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 104 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 174 |  |  |  | 
| halo2_wrapper | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |  | 
| halo2_wrapper | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 213 |  |  |  | 
| halo2_wrapper | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 49 |  |  |  | 
| halo2_wrapper | witness.next_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 136 |  |  |  | 
| root | prover | 18,533,671 | 1,340 | 714 | 0 | 1 | 103 | 22 | 21 | 32 | 48 | 0 | 522 | 513 | 8 | 1 | 7 | 714 | 714 | 103 | 0 | 133 |  | 12 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 16,493,732 | 150 | 12 | 0 | 0 | 107 | 73 | 72 | 15 | 18 | 0 | 30 | 20 | 9 | 1 | 7 | 13 | 12 | 107 | 0 | 1 | 17 | 0 | 0 | 
| halo2_keygen | prover | 0 | 5,385,707 | 41 | 6 | 0 | 0 | 20 | 5 | 5 | 4 | 9 | 0 | 14 | 11 | 2 | 0 | 2 | 6 | 6 | 20 | 0 | 1 | 9 | 0 | 0 | 
| root_keygen | prover | 0 | 5,385,707 | 144 | 46 | 0 | 0 | 22 | 6 | 6 | 4 | 11 | 0 | 74 | 71 | 3 | 0 | 2 | 46 | 46 | 22 | 0 | 6 | 10 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 6,749,074 | 2,013,265,921 | 
| halo2_keygen | prover | 0 | 0 | 1,442,095 | 2,013,265,921 | 
| root_keygen | prover | 0 | 0 | 1,442,095 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,535 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 0 | 1 | 52 | 220 | 52 | 0 | 233,971,539 | 81,286,314 | 315,257,853 | 2,097,152 | 0 | 2 | 17 | 112,210 | 13.11 | 
| halo2_keygen | 0 | 0 | 1 | 39 | 81 | 39 | 0 | 76,023,329 | 36,102 | 76,059,431 | 2,097,152 | 0 | 1 | 1 | 1 | 1.07 | 
| root_keygen | 0 | 0 | 3 | 39 | 186 | 39 | 0 | 76,023,329 | 36,102 | 76,059,431 | 2,097,152 | 0 | 3 | 3 | 1 | 1.88 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| prover | 6 | 0 | 6 | 6 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/6e1f16212b2d87e749c6bcf9be02f5f8a8f8bc94

Instance Type: g7e.2xlarge+g7.4xlarge+g6e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29331464583)
