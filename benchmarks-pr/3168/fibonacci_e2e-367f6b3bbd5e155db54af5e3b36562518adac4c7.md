| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  9.67 |  8.51 |  8.51 |
| app_proof |  1.81 |  0.65 |  0.65 |
| leaf |  0.36 |  0.36 |  0.36 |
| internal_for_leaf |  0.20 |  0.20 |  0.20 |
| internal_recursive.0 |  0.12 |  0.12 |  0.12 |
| internal_recursive.1 |  0.11 |  0.11 |  0.11 |
| root |  1.42 |  1.42 |  1.42 |
| halo2_outer |  4.01 |  4.01 |  4.01 |
| halo2_wrapper |  1.64 |  1.64 |  1.64 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  443.25 |  1,773 |  613 |  308 |
| `execute_metered_time_ms` |  37 | -          | -          | -          |
| `execute_metered_insns` |  12,000,265 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  318.15 | -          |  318.15 |  318.15 |
| `execute_preflight_insns` |  3,000,066.25 |  12,000,265 |  3,495,000 |  1,515,265 |
| `execute_preflight_time_ms` |  94 |  376 |  115 |  50 |
| `execute_preflight_insn_mi/s` |  63.21 | -          |  66.49 |  58.44 |
| `trace_gen_time_ms   ` |  42.75 |  171 |  69 |  22 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  234 |  936 |  262 |  153 |
| `prover.main_trace_commit_time_ms` |  39.25 |  157 |  44 |  25 |
| `prover.rap_constraints_time_ms` |  149 |  596 |  169 |  96 |
| `prover.openings_time_ms` |  44.75 |  179 |  51 |  31 |
| `prover.rap_constraints.logup_gkr_time_ms` |  95.50 |  382 |  108 |  60 |
| `prover.rap_constraints.round0_time_ms` |  37.25 |  149 |  43 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  15 |  60 |  17 |  11 |
| `prover.openings.stacked_reduction_time_ms` |  10.25 |  41 |  12 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  4.75 |  19 |  6 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  20 |  6 |  4 |
| `prover.openings.whir_time_ms` |  33.75 |  135 |  40 |  23 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  362 |  362 |  362 |  362 |
| `execute_preflight_time_ms` |  1 |  1 |  1 |  1 |
| `trace_gen_time_ms   ` |  63 |  63 |  63 |  63 |
| `generate_blob_total_time_ms` |  4 |  4 |  4 |  4 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  299 |  299 |  299 |  299 |
| `prover.main_trace_commit_time_ms` |  88 |  88 |  88 |  88 |
| `prover.rap_constraints_time_ms` |  130 |  130 |  130 |  130 |
| `prover.openings_time_ms` |  80 |  80 |  80 |  80 |
| `prover.rap_constraints.logup_gkr_time_ms` |  26 |  26 |  26 |  26 |
| `prover.rap_constraints.round0_time_ms` |  65 |  65 |  65 |  65 |
| `prover.rap_constraints.mle_rounds_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.stacked_reduction_time_ms` |  16 |  16 |  16 |  16 |
| `prover.openings.stacked_reduction.round0_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  64 |  64 |  64 |  64 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  197 |  197 |  197 |  197 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  178 |  178 |  178 |  178 |
| `prover.main_trace_commit_time_ms` |  45 |  45 |  45 |  45 |
| `prover.rap_constraints_time_ms` |  80 |  80 |  80 |  80 |
| `prover.openings_time_ms` |  52 |  52 |  52 |  52 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  29 |  29 |  29 |  29 |
| `prover.rap_constraints.mle_rounds_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction_time_ms` |  10 |  10 |  10 |  10 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  41 |  41 |  41 |  41 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  119 |  119 |  119 |  119 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  12 |  12 |  12 |  12 |
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
| `total_proof_time_ms ` |  110 |  110 |  110 |  110 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  99 |  99 |  99 |  99 |
| `prover.main_trace_commit_time_ms` |  14 |  14 |  14 |  14 |
| `prover.rap_constraints_time_ms` |  54 |  54 |  54 |  54 |
| `prover.openings_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.logup_gkr_time_ms` |  11 |  11 |  11 |  11 |
| `prover.rap_constraints.round0_time_ms` |  20 |  20 |  20 |  20 |
| `prover.rap_constraints.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  22 |  22 |  22 |  22 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,420 |  1,420 |  1,420 |  1,420 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,321 |  1,321 |  1,321 |  1,321 |
| `prover.main_trace_commit_time_ms` |  713 |  713 |  713 |  713 |
| `prover.rap_constraints_time_ms` |  77 |  77 |  77 |  77 |
| `prover.openings_time_ms` |  530 |  530 |  530 |  530 |
| `prover.rap_constraints.logup_gkr_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  33 |  33 |  33 |  33 |
| `prover.openings.stacked_reduction_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.whir_time_ms` |  521 |  521 |  521 |  521 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,013 |  4,013 |  4,013 |  4,013 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,642 |  1,642 |  1,642 |  1,642 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/367f6b3bbd5e155db54af5e3b36562518adac4c7/fibonacci_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 5.36 | app_proof.3 |
| set initial memory | 5.36 | app_proof.1 |
| prover.batch_constraints.before_round0 | 3.73 | app_proof.prover.0 |
| frac_sumcheck.gkr_rounds | 3.73 | app_proof.prover.0 |
| prover.stacked_commit | 3.65 | leaf.0.prover |
| frac_sumcheck.segment_tree | 3.52 | app_proof.prover.0 |
| prover.gkr_input_evals | 3.52 | app_proof.prover.0 |
| prover.merkle_tree | 3.23 | leaf.0.prover |
| prover.openings | 3.23 | leaf.0.prover |
| prover.prove_whir_opening | 3.23 | leaf.0.prover |
| prover.rs_code_matrix | 3.23 | leaf.0.prover |
| prover.rap_constraints | 2.10 | leaf.0.prover |
| prover.batch_constraints.round0 | 1.94 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.94 | leaf.0.prover |
| tracegen.exp_bits_len | 1.51 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.51 | leaf.0 |
| tracegen.pow_checker | 1.51 | leaf.0 |
| prover.before_gkr_input_evals | 1.45 | leaf.0.prover |
| tracegen.whir_folding | 1.26 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 1.25 | leaf.0 |
| tracegen.whir_initial_opened_values | 1.25 | leaf.0 |
| tracegen.range_checker | 1.21 | leaf.0 |
| tracegen.proof_shape | 1.21 | leaf.0 |
| tracegen.public_values | 1.21 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 34 | 26 | 0 | 0 | 0 | 2 | 4 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 | 2 | 1 | 
| 0 | RootVerifierPvsAir |  | 109 | 37 | 4 | 
| 1 | UserPvsCommitAir | 1 | 5 | 41 | 4 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | EqSharpUniReceiverAir | 1 | 3 | 25 | 4 | 
| 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 16 | 9 | 3 | 
| 11 | EqUniAir | 1 | 3 | 31 | 4 | 
| 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 10 | 9 | 2 | 
| 12 | ExpressionClaimAir | 1 | 7 | 68 | 4 | 
| 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 13 | 25 | 3 | 
| 13 | InteractionsFoldingAir | 1 | 13 | 94 | 4 | 
| 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 14 | ConstraintsFoldingAir | 1 | 10 | 42 | 4 | 
| 14 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 18 | 18 | 3 | 
| 15 | EqNegAir | 1 | 8 | 83 | 4 | 
| 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 17 | 25 | 3 | 
| 16 | TranscriptAir | 1 | 17 | 84 | 4 | 
| 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 24 | 76 | 3 | 
| 17 | Poseidon2Air<BabyBearParameters>, 1> |  | 2 | 282 | 3 | 
| 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 18 | 28 | 3 | 
| 18 | MerkleVerifyAir |  | 6 | 22 | 3 | 
| 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 20 | 22 | 3 | 
| 19 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 19 | ProofShapeAir<4, 8> | 1 | 78 | 127 | 4 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 2 | UserPvsInMemoryAir | 1 | 3 | 13 | 4 | 
| 20 | PhantomAir |  | 3 | 1 | 2 | 
| 20 | PublicValuesAir | 1 | 4 | 18 | 4 | 
| 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 21 | RangeCheckerAir<8> | 1 | 1 | 3 | 2 | 
| 22 | GkrInputAir | 1 | 19 | 19 | 4 | 
| 22 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 23 | GkrLayerAir | 1 | 30 | 38 | 4 | 
| 24 | GkrLayerSumcheckAir | 1 | 21 | 59 | 4 | 
| 25 | GkrXiSamplerAir | 1 | 7 | 17 | 4 | 
| 26 | OpeningClaimsAir | 1 | 22 | 98 | 4 | 
| 27 | UnivariateRoundAir | 1 | 13 | 54 | 4 | 
| 28 | SumcheckRoundsAir | 1 | 21 | 69 | 4 | 
| 29 | StackingClaimsAir | 1 | 17 | 57 | 4 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 36 | 3 | 
| 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | 13 | 320 | 4 | 
| 30 | EqBaseAir | 1 | 8 | 89 | 4 | 
| 31 | EqBitsAir | 1 | 5 | 24 | 4 | 
| 32 | WhirRoundAir | 1 | 31 | 28 | 4 | 
| 33 | SumcheckAir | 1 | 19 | 47 | 4 | 
| 34 | WhirQueryAir | 1 | 5 | 51 | 4 | 
| 35 | InitialOpenedValuesAir | 1 | 13 | 145 | 4 | 
| 36 | NonInitialOpenedValuesAir | 1 | 4 | 42 | 4 | 
| 37 | WhirFoldingAir |  | 4 | 15 | 3 | 
| 38 | FinalPolyMleEvalAir |  | 13 | 19 | 4 | 
| 39 | FinalPolyQueryEvalAir | 1 | 5 | 120 | 4 | 
| 4 | FractionsFolderAir | 1 | 17 | 41 | 4 | 
| 4 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 25 | 64 | 3 | 
| 40 | PowerCheckerAir<2, 32> | 1 | 2 | 5 | 2 | 
| 41 | ExpBitsLenAir | 1 | 2 | 44 | 3 | 
| 5 | UnivariateSumcheckAir | 1 | 14 | 46 | 4 | 
| 5 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 24 | 11 | 2 | 
| 6 | MultilinearSumcheckAir | 1 | 14 | 60 | 4 | 
| 6 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 19 | 4 | 2 | 
| 7 | EqNsAir | 1 | 10 | 65 | 4 | 
| 7 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 
| 8 | Eq3bAir | 1 | 3 | 65 | 4 | 
| 8 | Rv32HintStoreAir | 1 | 18 | 17 | 3 | 
| 9 | EqSharpUniAir | 1 | 5 | 48 | 4 | 
| 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 12 | 5 | 3 | 

| group | transport_pk_to_device_time_ms | tracegen_attempt_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | root_time_ms | prove_segment_time_ms | populate_inputs_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | graph_witness_gen_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_cuda_time_ms | collect_pvs_time_ms | build_graph_program_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 62 |  |  |  |  |  |  |  |  | 261 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  | 308 |  |  |  |  |  |  |  |  |  | 37 | 12,000,265 | 318.15 | 0 |  |  |  |  | 1,846 |  | 
| halo2_keygen |  |  |  |  |  |  |  | 273 |  |  | 77,619 |  |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  | 4,129 |  | 274 |  | 
| halo2_outer |  |  | 4,013 |  |  |  |  |  | 3 |  |  |  | 23 | 184 |  |  |  |  |  |  |  |  | 0 |  |  |  |  | 
| halo2_wrapper |  |  | 1,642 |  |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  | 197 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 197 | 
| internal_recursive.0 |  |  |  |  |  | 119 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 119 | 
| internal_recursive.1 |  |  |  |  |  | 110 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 110 | 
| leaf |  |  |  |  | 362 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 363 | 
| root | 98 | 11 | 1,420 | 10 |  |  | 1,420 |  |  |  |  |  |  |  | 1 | 0 | 2 |  |  |  |  | 0 |  |  | 0 |  | 1,420 | 
| root_keygen |  |  |  |  |  |  |  | 397 |  |  |  |  |  |  |  |  |  | 0 | 1 | 0.01 | 0 |  |  |  |  | 398 |  | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | air_id | air_name | segment | trace_gen.record_arena_bytes |
| --- | --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 6 | PhantomAir | 0 | 20 | 
| app_proof | Rv32HintStoreAir | 18 | Rv32HintStoreAir | 0 | 132 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 109,042,128 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 36,346,700 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 208 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 18,639,640 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 240 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 7,456,192 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 396 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,980 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 224 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 109,044,000 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 36,348,000 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 18,640,000 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 7,456,000 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 109,044,000 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 36,348,000 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 18,640,000 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 7,456,000 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 47,274,552 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 15,757,404 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 8,081,080 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 80 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 3,232,512 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 440 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 3,120 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 112 | 

| group | air | segment | trace_gen.h2d_records_time_ms | single_trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | PhantomAir | 0 |  | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 2 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 11 | 11 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 10 | 10 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 10 | 10 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 3 | 3 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 4 | 4 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 0 | 0 | 

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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 71 | 284 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 37 | 151,552 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 38 | 4,980,736 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 43 | 5,504 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 2 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 1,024 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 512 | 
| leaf | 22 | GkrInputAir | 0 | prover | 4 | 26 | 104 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 128 | 46 | 5,888 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 2,048 | 45 | 92,160 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 4 | 10 | 40 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 4,096 | 63 | 258,048 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 128 | 27 | 3,456 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 128 | 57 | 7,296 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 8,192 | 35 | 286,720 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 8,192 | 60 | 491,520 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 47 | 736 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 131,072 | 89 | 11,665,408 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 6,144 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | opcode | segment | opcode_count |
| --- | --- | --- | --- | --- | --- |
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 0 | 9 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 0 | 232,992 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 0 | 14 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BGEU | 0 | 2 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLTU | 0 | 4 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 232,994 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 232,997 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 0 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 0 | 19 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | SLL | 0 | 4 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 0 | 698,975 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 2,096,954 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 0 | 2 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | OR | 0 | 3 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 5 | 
| app_proof | 20 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_BUFFER | 0 | 2 | 
| app_proof | 8 | Rv32HintStoreAir | HINT_STOREW | 0 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 0 | 8 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 1 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 233,000 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 1 | 699,000 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 1 | 2,097,000 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 2 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 2 | 233,000 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 2 | 233,000 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 2 | 699,000 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 2 | 2,097,000 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | JALR | 3 | 10 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | JAL | 3 | 101,010 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | LUI | 3 | 6 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | BLT | 3 | 2 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 3 | 101,015 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 3 | 101,012 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADBU | 3 | 7 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | LOADW | 3 | 14 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREB | 3 | 10 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | STOREW | 3 | 21 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | SLTU | 3 | 303,027 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | ADD | 3 | 909,125 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | AND | 3 | 1 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | AUIPC | 3 | 4 | 

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
| root | 18 | MerkleVerifyAir | prover | 8,192 | 38 | 311,296 | 
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
| app_proof | 0 | ProgramAir | prover | 0 | 8,192 | 12 | 98,304 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 262,144 | 18 | 4,718,592 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 8 | 32 | 256 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 524,288 | 26 | 13,631,488 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 64 | 41 | 2,624 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 4 | 53 | 212 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 2,097,152 | 36 | 75,497,472 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 64 | 21 | 1,344 | 
| app_proof | 20 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 256 | 33 | 8,448 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 1,572,864 | 
| app_proof | 8 | Rv32HintStoreAir | prover | 0 | 4 | 32 | 128 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 8 | 20 | 160 | 
| halo2_keygen | 0 | ProgramAir | prover | 0 | 1 | 12 | 12 | 
| halo2_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| halo2_keygen | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| halo2_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| halo2_keygen | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| halo2_keygen | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 1,572,864 | 
| root_keygen | 0 | ProgramAir | prover | 0 | 1 | 12 | 12 | 
| root_keygen | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| root_keygen | 19 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| root_keygen | 2 | PersistentBoundaryAir<8> | prover | 0 | 1 | 21 | 21 | 
| root_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 32 | 300 | 9,600 | 
| root_keygen | 22 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| root_keygen | 3 | MemoryMerkleAir<8> | prover | 0 | 64 | 33 | 2,112 | 
| root_keygen | 7 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 1 | 8,192 | 12 | 98,304 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 262,144 | 18 | 4,718,592 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 524,288 | 26 | 13,631,488 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 2,097,152 | 36 | 75,497,472 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 128 | 33 | 4,224 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 2 | 8,192 | 12 | 98,304 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 2 | 262,144 | 18 | 4,718,592 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 2 | 524,288 | 26 | 13,631,488 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 2 | 1,048,576 | 37 | 38,797,312 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 2 | 2,097,152 | 36 | 75,497,472 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 16 | 21 | 336 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 128 | 300 | 38,400 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 128 | 33 | 4,224 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 2 | 524,288 | 3 | 1,572,864 | 
| app_proof | 0 | ProgramAir | prover | 3 | 8,192 | 12 | 98,304 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 3 | 16 | 28 | 448 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 3 | 131,072 | 18 | 2,359,296 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 3 | 2 | 32 | 64 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 3 | 262,144 | 26 | 6,815,744 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 3 | 64 | 41 | 2,624 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 3 | 524,288 | 37 | 19,398,656 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 3 | 1,048,576 | 36 | 37,748,736 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 64 | 21 | 1,344 | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 76,800 | 
| app_proof | 22 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 256 | 33 | 8,448 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | prover | 3 | 524,288 | 3 | 1,572,864 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 3 | 4 | 20 | 80 | 

| group | air_id | air_name | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- | --- |
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 0 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 1 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | height | 2 | 1 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_cells_unpadded | metered_interaction_cells_padding | metered_constraint_eval_cells_unpadded | metered_constraint_eval_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 6,443 | 1,749 | 309,264 | 83,952 | 77,316 | 20,988 | 6,443 | 1,749 | 12,886 | 3,498 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 9 | 7 | 1,008 | 784 | 252 | 196 | 144 | 112 | 54 | 42 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 233,006 | 29,138 | 16,776,432 | 2,097,936 | 4,194,108 | 524,484 | 2,330,060 | 291,380 | 1,398,036 | 174,828 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 6 | 2 | 768 | 256 | 192 | 64 | 78 | 26 | 54 | 18 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 465,991 | 58,297 | 48,463,064 | 6,062,888 | 12,115,766 | 1,515,722 | 5,125,901 | 641,267 | 3,261,937 | 408,079 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 33 | 31 | 5,412 | 5,084 | 1,353 | 1,271 | 561 | 527 | 759 | 713 | 
| app_proof | 16 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 4 |  | 848 |  | 212 |  | 96 |  | 104 |  | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 698,975 | 349,601 | 103,448,300 | 51,740,948 | 25,862,075 | 12,935,237 | 12,581,550 | 6,292,818 | 9,086,675 | 4,544,813 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,096,964 | 188 | 301,962,816 | 27,072 | 75,490,704 | 6,768 | 41,939,280 | 3,760 | 27,260,532 | 2,444 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 512 |  | 43,008 |  | 10,752 |  | 2,048 |  | 1,024 |  | 
| app_proof | 20 | PhantomAir | 0 | 1 |  | 24 |  | 6 |  | 3 |  | 2 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 1,200 | 848 | 1,440,000 | 1,017,600 | 360,000 | 254,400 | 1,200 | 848 | 39,600 | 27,984 | 
| app_proof | 22 | VariableRangeCheckerAir | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 688 | 336 | 90,816 | 44,352 | 22,704 | 11,088 | 2,752 | 1,344 | 6,192 | 3,024 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 0 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| app_proof | 8 | Rv32HintStoreAir | 0 | 3 | 1 | 384 | 128 | 96 | 32 | 54 | 18 | 24 | 8 | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 640 |  | 160 |  | 96 |  | 48 |  | 
| halo2_keygen | 0 | ProgramAir | 0 | 1 |  | 48 |  | 12 |  | 1 |  | 2 |  | 
| halo2_keygen | 1 | VmConnectorAir | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| halo2_keygen | 19 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| halo2_keygen | 2 | PersistentBoundaryAir<8> | 0 | 128 |  | 10,752 |  | 2,688 |  | 512 |  | 256 |  | 
| halo2_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 300 | 212 | 360,000 | 254,400 | 90,000 | 63,600 | 300 | 212 | 9,900 | 6,996 | 
| halo2_keygen | 22 | VariableRangeCheckerAir | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| halo2_keygen | 3 | MemoryMerkleAir<8> | 0 | 172 | 84 | 22,704 | 11,088 | 5,676 | 2,772 | 688 | 336 | 1,548 | 756 | 
| halo2_keygen | 7 | RangeTupleCheckerAir<2> | 0 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| root_keygen | 0 | ProgramAir | 0 | 1 |  | 48 |  | 12 |  | 1 |  | 2 |  | 
| root_keygen | 1 | VmConnectorAir | 0 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| root_keygen | 19 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| root_keygen | 2 | PersistentBoundaryAir<8> | 0 | 128 |  | 10,752 |  | 2,688 |  | 512 |  | 256 |  | 
| root_keygen | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 300 | 212 | 360,000 | 254,400 | 90,000 | 63,600 | 300 | 212 | 9,900 | 6,996 | 
| root_keygen | 22 | VariableRangeCheckerAir | 0 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| root_keygen | 3 | MemoryMerkleAir<8> | 0 | 172 | 84 | 22,704 | 11,088 | 5,676 | 2,772 | 688 | 336 | 1,548 | 756 | 
| root_keygen | 7 | RangeTupleCheckerAir<2> | 0 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| app_proof | 0 | ProgramAir | 1 | 6,443 | 1,749 | 309,264 | 83,952 | 77,316 | 20,988 | 6,443 | 1,749 | 12,886 | 3,498 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 233,000 | 29,144 | 16,776,000 | 2,098,368 | 4,194,000 | 524,592 | 2,330,000 | 291,440 | 1,398,000 | 174,864 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 466,000 | 58,288 | 48,464,000 | 6,061,952 | 12,116,000 | 1,515,488 | 5,126,000 | 641,168 | 3,262,000 | 408,016 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 699,000 | 349,576 | 103,452,000 | 51,737,248 | 25,863,000 | 12,934,312 | 12,582,000 | 6,292,368 | 9,087,000 | 4,544,488 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 2,097,000 | 152 | 301,968,000 | 21,888 | 75,492,000 | 5,472 | 41,940,000 | 3,040 | 27,261,000 | 1,976 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 128 |  | 10,752 |  | 2,688 |  | 512 |  | 256 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 300 | 212 | 360,000 | 254,400 | 90,000 | 63,600 | 300 | 212 | 9,900 | 6,996 | 
| app_proof | 22 | VariableRangeCheckerAir | 1 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 172 | 84 | 22,704 | 11,088 | 5,676 | 2,772 | 688 | 336 | 1,548 | 756 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 1 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| app_proof | 0 | ProgramAir | 2 | 6,443 | 1,749 | 309,264 | 83,952 | 77,316 | 20,988 | 6,443 | 1,749 | 12,886 | 3,498 | 
| app_proof | 1 | VmConnectorAir | 2 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 233,000 | 29,144 | 16,776,000 | 2,098,368 | 4,194,000 | 524,592 | 2,330,000 | 291,440 | 1,398,000 | 174,864 | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 466,000 | 58,288 | 48,464,000 | 6,061,952 | 12,116,000 | 1,515,488 | 5,126,000 | 641,168 | 3,262,000 | 408,016 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 699,000 | 349,576 | 103,452,000 | 51,737,248 | 25,863,000 | 12,934,312 | 12,582,000 | 6,292,368 | 9,087,000 | 4,544,488 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 2,097,000 | 152 | 301,968,000 | 21,888 | 75,492,000 | 5,472 | 41,940,000 | 3,040 | 27,261,000 | 1,976 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 2 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 2 | 128 |  | 10,752 |  | 2,688 |  | 512 |  | 256 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 300 | 212 | 360,000 | 254,400 | 90,000 | 63,600 | 300 | 212 | 9,900 | 6,996 | 
| app_proof | 22 | VariableRangeCheckerAir | 2 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 2 | 172 | 84 | 22,704 | 11,088 | 5,676 | 2,772 | 688 | 336 | 1,548 | 756 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 2 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| app_proof | 0 | ProgramAir | 3 | 6,443 | 1,749 | 309,264 | 83,952 | 77,316 | 20,988 | 6,443 | 1,749 | 12,886 | 3,498 | 
| app_proof | 1 | VmConnectorAir | 3 | 2 |  | 48 |  | 12 |  | 10 |  | 6 |  | 
| app_proof | 10 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 10 | 6 | 1,120 | 672 | 280 | 168 | 160 | 96 | 60 | 36 | 
| app_proof | 11 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 101,016 | 30,056 | 7,273,152 | 2,164,032 | 1,818,288 | 541,008 | 1,010,160 | 300,560 | 606,096 | 180,336 | 
| app_proof | 12 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 2 |  | 256 |  | 64 |  | 26 |  | 18 |  | 
| app_proof | 13 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 202,027 | 60,117 | 21,010,808 | 6,252,168 | 5,252,702 | 1,563,042 | 2,222,297 | 661,287 | 1,414,189 | 420,819 | 
| app_proof | 15 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 52 | 12 | 8,528 | 1,968 | 2,132 | 492 | 884 | 204 | 1,196 | 276 | 
| app_proof | 17 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 303,027 | 221,261 | 44,847,996 | 32,746,628 | 11,211,999 | 8,186,657 | 5,454,486 | 3,982,698 | 3,939,351 | 2,876,393 | 
| app_proof | 18 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 909,126 | 139,450 | 130,914,144 | 20,080,800 | 32,728,536 | 5,020,200 | 18,182,520 | 2,789,000 | 11,818,638 | 1,812,850 | 
| app_proof | 19 | BitwiseOperationLookupAir<8> | 3 | 65,536 |  | 4,718,592 |  | 1,179,648 |  | 131,072 |  | 1,245,184 |  | 
| app_proof | 2 | PersistentBoundaryAir<8> | 3 | 512 |  | 43,008 |  | 10,752 |  | 2,048 |  | 1,024 |  | 
| app_proof | 21 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 1,200 | 848 | 1,440,000 | 1,017,600 | 360,000 | 254,400 | 1,200 | 848 | 39,600 | 27,984 | 
| app_proof | 22 | VariableRangeCheckerAir | 3 | 262,144 |  | 4,194,304 |  | 1,048,576 |  | 262,144 |  | 1,572,864 |  | 
| app_proof | 3 | MemoryMerkleAir<8> | 3 | 688 | 336 | 90,816 | 44,352 | 22,704 | 11,088 | 2,752 | 1,344 | 6,192 | 3,024 | 
| app_proof | 7 | RangeTupleCheckerAir<2> | 3 | 524,288 |  | 6,291,456 |  | 1,572,864 |  | 524,288 |  | 2,621,440 |  | 
| app_proof | 9 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 4 |  | 320 |  | 80 |  | 48 |  | 24 |  | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 197 | 17 | 5 | 0 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 12 | 119 | 12 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 110 | 10 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 63 | 362 | 62 | 28 | 4 | 1 | 1 | 1 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,627,133 | 178 | 45 | 0 | 0 | 80 | 29 | 28 | 37 | 13 | 0 | 52 | 41 | 10 | 2 | 7 | 45 | 45 | 80 | 0 | 1 | 13 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,386,961 | 107 | 20 | 0 | 0 | 56 | 21 | 20 | 23 | 11 | 0 | 30 | 22 | 8 | 1 | 6 | 20 | 20 | 56 | 0 | 1 | 10 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,759,057 | 99 | 14 | 0 | 0 | 54 | 20 | 19 | 22 | 11 | 0 | 30 | 22 | 7 | 1 | 6 | 14 | 14 | 54 | 0 | 1 | 10 | 0 | 0 | 
| leaf | 0 | prover | 148,065,326 | 299 | 88 | 0 | 0 | 130 | 65 | 64 | 38 | 26 | 0 | 80 | 64 | 16 | 6 | 9 | 88 | 88 | 130 | 0 | 3 | 25 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,806,473 | 2,013,265,921 | 

| group | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | halo2_section_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | prover |  |  | 6 |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 6 |  |  | 5 |  |  |  |  | 
| halo2_keygen | ifft_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 120 |  |  |  | 
| halo2_keygen | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_keygen | lagrange_to_coeff_many |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 120 |  |  |  | 
| halo2_keygen | multiexp_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 20 |  |  |  | 
| halo2_outer | add_blinding_factors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 82 |  |  |  | 
| halo2_outer | advice_msms |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 814 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 35 |  |  |  | 
| halo2_outer | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | batch_normalize_commitments |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | coeff_to_extended_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | commit_vanishing_h_x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 286 |  |  |  | 
| halo2_outer | commit_vanishing_random_poly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 125 |  |  |  | 
| halo2_outer | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 45 |  |  |  | 
| halo2_outer | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 35 |  |  |  | 
| halo2_outer | convert_raw_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,233 |  |  |  | 
| halo2_outer | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3,824 |  |  |  | 
| halo2_outer | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 44 |  |  |  | 
| halo2_outer | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 80 |  |  |  | 
| halo2_outer | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,039 |  |  |  | 
| halo2_outer | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_outer | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_pk_from_host |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 136 |  |  |  | 
| halo2_outer | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 164 |  |  |  | 
| halo2_outer | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | ifft_msm_instance_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,204 |  |  |  | 
| halo2_outer | instance_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 250 |  |  |  | 
| halo2_outer | instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 57 |  |  |  | 
| halo2_outer | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lagrange_to_coeff |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 18 |  |  |  | 
| halo2_outer | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 375 |  |  |  | 
| halo2_outer | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 116 |  |  |  | 
| halo2_outer | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 181 |  |  |  | 
| halo2_outer | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94 |  |  |  | 
| halo2_outer | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,204 |  |  |  | 
| halo2_outer | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_outer | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 416 |  |  |  | 
| halo2_outer | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_outer | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | permutation_quotient_poly_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_outer | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 53 |  |  |  | 
| halo2_outer | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,234 |  |  |  | 
| halo2_outer | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 375 |  |  |  | 
| halo2_outer | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 532 |  |  |  | 
| halo2_outer | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 416 |  |  |  | 
| halo2_outer | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 116 |  |  |  | 
| halo2_outer | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,039 |  |  |  | 
| halo2_outer | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 286 |  |  |  | 
| halo2_outer | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 356 |  |  |  | 
| halo2_outer | phase5_multiopen |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 300 |  |  |  | 
| halo2_outer | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 3 |  |  |  | 
| halo2_outer | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 136 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 26 |  |  |  | 
| halo2_outer | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 300 |  |  |  | 
| halo2_outer | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94 |  |  |  | 
| halo2_outer | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 45 |  |  |  | 
| halo2_outer | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | transcript_write_squeeze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_outer | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 286 |  |  |  | 
| halo2_outer | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | add_blinding_factors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | advice_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 25 |  |  |  | 
| halo2_wrapper | advice_msms |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 62 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8 |  |  |  | 
| halo2_wrapper | batch_eval_polynomial_device_out |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | batch_invert_witness |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 28 |  |  |  | 
| halo2_wrapper | batch_normalize_commitments |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | coeff_to_extended_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | column_pool.upload |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | commit_vanishing_h_x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 200 |  |  |  | 
| halo2_wrapper | commit_vanishing_random_poly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_in_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | compress_expressions_with_runtime_constants_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | construct_intermediate_sets |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9 |  |  |  | 
| halo2_wrapper | convert_raw_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 169 |  |  |  | 
| halo2_wrapper | cosetfft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | create_proof |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,384 |  |  |  | 
| halo2_wrapper | custom_gates |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | decode_assigned_into_denom_slice_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 28 |  |  |  | 
| halo2_wrapper | device_fold |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | distribute_powers_zeta_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.coeff_to_extended_part_many_device_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.divide_by_vanishing_poly_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.extended_to_coeff_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_device_input |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | domain.lagrange_to_coeff_many_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 25 |  |  |  | 
| halo2_wrapper | evaluate_h |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 300 |  |  |  | 
| halo2_wrapper | extended_from_lagrange_vec_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | fft_normal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4 |  |  |  | 
| halo2_wrapper | fft_normal_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_pk_from_host |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | gpu_quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | grand_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | grand_product_scan |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | h_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 72 |  |  |  | 
| halo2_wrapper | h_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_many_device_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | ifft_msm_instance_advice |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 153 |  |  |  | 
| halo2_wrapper | instance_ifft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 39 |  |  |  | 
| halo2_wrapper | instance_to_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 26 |  |  |  | 
| halo2_wrapper | kate_division_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_padded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kate_division_device_with_d_root |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | kzg.g_lagrange_device_first_touch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lagrange_to_coeff |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookup.evaluate.eval_at_block |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 10 |  |  |  | 
| halo2_wrapper | lookup_commit_permuted |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 64 |  |  |  | 
| halo2_wrapper | lookup_commit_product |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 
| halo2_wrapper | lookup_product_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 75 |  |  |  | 
| halo2_wrapper | multiexp_device_scalars_device_bases |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | new_gpu_thread |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 153 |  |  |  | 
| halo2_wrapper | permutation.evaluate.eval_at_loop |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97 |  |  |  | 
| halo2_wrapper | permutation_coset_fft |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_pk.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 5 |  |  |  | 
| halo2_wrapper | permutation_product_device_inputs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutation_quotient_poly_part |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permutations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | permute_expression_pair |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | phase1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 169 |  |  |  | 
| halo2_wrapper | phase2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 232 |  |  |  | 
| halo2_wrapper | phase3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 148 |  |  |  | 
| halo2_wrapper | phase3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97 |  |  |  | 
| halo2_wrapper | phase3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 50 |  |  |  | 
| halo2_wrapper | phase4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 300 |  |  |  | 
| halo2_wrapper | phase4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 200 |  |  |  | 
| halo2_wrapper | phase5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 153 |  |  |  | 
| halo2_wrapper | phase5_multiopen |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 133 |  |  |  | 
| halo2_wrapper | poly_multiply_add_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_scale_device_with_d_s_minus_one |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_scalar_at_zero_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | poly_sub_short_out_of_place_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 
| halo2_wrapper | quotient_lookups |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.add_permutation_constraints |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.calculate_constraints_full_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.new_with_device_selectors |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_lookups_gpu.take_values_device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | quotient_permutation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 133 |  |  |  | 
| halo2_wrapper | shplonk.final_l_x_kate_div |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.h_final_commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 48 |  |  |  | 
| halo2_wrapper | shplonk.l_x_device_reduce |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | shplonk.linearisation_contribution.rayon_worker |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | synthesize |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 150 |  |  |  | 
| halo2_wrapper | table_values |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | take_values_device_for_assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | transcript_write_squeeze |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.commit |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| halo2_wrapper | vanishing.construct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 200 |  |  |  | 
| halo2_wrapper | vanishing.evaluate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |  |  |  | 
| root | prover | 18,541,863 | 1,321 | 713 | 0 | 1 | 77 | 22 | 21 | 33 | 21 | 0 | 530 | 521 | 9 | 1 | 7 | 713 | 713 | 77 | 0 | 134 |  | 12 | 0 | 0 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prover.commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 136,634,694 | 262 | 44 | 0 | 0 | 169 | 43 | 43 | 17 | 107 | 0 | 48 | 35 | 12 | 6 | 6 | 44 | 44 | 169 | 0 | 1 | 107 | 0 | 0 | 
| halo2_keygen | prover | 0 | 3,812,845 | 39 | 6 | 0 | 0 | 18 | 5 | 5 | 4 | 8 | 0 | 14 | 11 | 2 | 0 | 2 | 6 | 6 | 18 | 0 | 1 | 8 | 0 | 0 | 
| root_keygen | prover | 0 | 3,812,845 | 143 | 45 | 0 | 0 | 22 | 6 | 6 | 4 | 10 | 0 | 75 | 71 | 3 | 0 | 2 | 46 | 45 | 22 | 0 | 6 | 10 | 0 | 0 | 
| app_proof | prover | 1 | 136,587,228 | 260 | 44 | 0 | 0 | 166 | 41 | 41 | 16 | 108 | 0 | 49 | 37 | 11 | 5 | 5 | 44 | 44 | 166 | 0 | 1 | 106 | 0 | 0 | 
| app_proof | prover | 2 | 136,587,228 | 261 | 44 | 0 | 0 | 165 | 41 | 40 | 16 | 107 | 0 | 51 | 40 | 11 | 5 | 5 | 44 | 44 | 165 | 0 | 1 | 107 | 0 | 0 | 
| app_proof | prover | 3 | 70,311,644 | 153 | 24 | 0 | 0 | 96 | 24 | 24 | 11 | 60 | 0 | 31 | 23 | 7 | 3 | 4 | 25 | 24 | 96 | 0 | 1 | 59 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 70,134,973 | 2,013,265,921 | 
| halo2_keygen | prover | 0 | 0 | 917,807 | 2,013,265,921 | 
| root_keygen | prover | 0 | 0 | 917,807 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 70,132,426 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 70,132,426 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 35,531,668 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,535 | 2,013,265,921 | 

| group | segment | vm.transport_init_memory_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_whir_memory_bytes | metered_secondary_peak_memory_bytes | metered_rs_code_matrix_memory_bytes | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_gkr_memory_bytes | metered_batch_constraint_memory_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 180 | 55 | 613 | 55 | 180 | 125,829,120 | 2,798,180,256 | 1,107,296,256 | 2,911,837,936 | 435,170,504 | 3,347,008,440 | 2,798,180,256 | 681,818,432 | 0 | 2 | 115 | 3,495,000 | 62.45 | 
| halo2_keygen | 0 | 122 | 109 | 273 | 109 | 122 | 125,829,120 | 432,210,272 | 33,554,432 | 447,790,640 | 283,024 | 448,073,664 | 432,210,272 | 312,107,072 | 0 | 1 | 1 | 1 | 9,223,372,036,854,775,807 | 
| root_keygen | 0 | 135 | 118 | 397 | 118 | 135 | 536,870,912 | 671,088,640 | 134,217,728 | 686,686,544 | 265,488 | 686,952,032 | 432,210,272 | 312,107,072 | 0 | 7 | 0 | 1 | 9,223,372,036,854,775,807 | 
| app_proof | 1 | 36 | 22 | 425 | 22 | 36 | 125,829,120 | 2,797,928,768 | 1,107,296,256 | 2,910,519,536 | 434,245,248 | 3,344,764,784 | 2,797,928,768 | 681,336,880 | 0 | 2 | 107 | 3,495,000 | 65.46 | 
| app_proof | 2 | 35 | 25 | 427 | 25 | 35 | 125,829,120 | 2,797,928,768 | 1,107,296,256 | 2,910,519,536 | 434,245,248 | 3,344,764,784 | 2,797,928,768 | 681,336,880 | 0 | 2 | 104 | 3,495,000 | 66.49 | 
| app_proof | 3 | 35 | 69 | 308 | 69 | 35 | 125,829,120 | 1,548,268,160 | 570,425,344 | 1,517,608,516 | 314,195,628 | 1,831,804,144 | 1,548,268,160 | 497,267,328 | 0 | 2 | 50 | 1,515,265 | 58.44 | 

| phase | stacked_commit_time_ms | rs_code_matrix_time_ms | prover.commit_time_ms | merkle_tree_time_ms | halo2_section_time_ms |
| --- | --- | --- | --- | --- | --- |
| kzg.g_device_first_touch |  |  |  |  | 27 | 
| kzg.g_lagrange_device_first_touch |  |  |  |  | 0 | 
| prover | 6 | 0 | 6 | 6 |  | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/367f6b3bbd5e155db54af5e3b36562518adac4c7

Instance Type: g7.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38793620937)
