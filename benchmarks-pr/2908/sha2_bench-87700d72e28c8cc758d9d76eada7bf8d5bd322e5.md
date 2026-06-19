| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  11.91 |  6.87 |  6.87 |
| app_proof |  10.21 |  5.17 |  5.17 |
| leaf |  1.02 |  1.02 |  1.02 |
| internal_for_leaf |  0.35 |  0.35 |  0.35 |
| internal_recursive.0 |  0.18 |  0.18 |  0.18 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,054.50 |  10,109 |  5,072 |  5,037 |
| `compile_metered_time_ms` |  3 |  3 |  3 |  3 |
| `execute_metered_time_ms` |  100 | -          | -          | -          |
| `execute_metered_insns` |  11,167,961 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  110.98 | -          |  110.98 |  110.98 |
| `execute_preflight_insns` |  5,583,980.50 |  11,167,961 |  5,703,000 |  5,464,961 |
| `execute_preflight_time_ms` |  295 |  590 |  396 |  194 |
| `execute_preflight_insn_mi/s` |  33.94 | -          |  35.50 |  32.37 |
| `trace_gen_time_ms   ` |  131 |  262 |  152 |  110 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  4,538 |  9,076 |  4,587 |  4,489 |
| `prover.main_trace_commit_time_ms` |  1,272.50 |  2,545 |  1,312 |  1,233 |
| `prover.rap_constraints_time_ms` |  2,391.50 |  4,783 |  2,398 |  2,385 |
| `prover.openings_time_ms` |  873 |  1,746 |  876 |  870 |
| `prover.rap_constraints.logup_gkr_time_ms` |  605 |  1,210 |  605 |  605 |
| `prover.rap_constraints.round0_time_ms` |  1,302.50 |  2,605 |  1,307 |  1,298 |
| `prover.rap_constraints.mle_rounds_time_ms` |  483 |  966 |  485 |  481 |
| `prover.openings.stacked_reduction_time_ms` |  206 |  412 |  206 |  206 |
| `prover.openings.stacked_reduction.round0_time_ms` |  129 |  258 |  129 |  129 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  76 |  152 |  76 |  76 |
| `prover.openings.whir_time_ms` |  666.50 |  1,333 |  669 |  664 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,018 |  1,018 |  1,018 |  1,018 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  122 |  122 |  122 |  122 |
| `generate_blob_total_time_ms` |  21 |  21 |  21 |  21 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  895 |  895 |  895 |  895 |
| `prover.main_trace_commit_time_ms` |  374 |  374 |  374 |  374 |
| `prover.rap_constraints_time_ms` |  230 |  230 |  230 |  230 |
| `prover.openings_time_ms` |  290 |  290 |  290 |  290 |
| `prover.rap_constraints.logup_gkr_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.round0_time_ms` |  111 |  111 |  111 |  111 |
| `prover.rap_constraints.mle_rounds_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings.stacked_reduction_time_ms` |  37 |  37 |  37 |  37 |
| `prover.openings.stacked_reduction.round0_time_ms` |  19 |  19 |  19 |  19 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  17 |  17 |  17 |  17 |
| `prover.openings.whir_time_ms` |  252 |  252 |  252 |  252 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  346 |  346 |  346 |  346 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  21 |  21 |  21 |  21 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  325 |  325 |  325 |  325 |
| `prover.main_trace_commit_time_ms` |  117 |  117 |  117 |  117 |
| `prover.rap_constraints_time_ms` |  108 |  108 |  108 |  108 |
| `prover.openings_time_ms` |  98 |  98 |  98 |  98 |
| `prover.rap_constraints.logup_gkr_time_ms` |  21 |  21 |  21 |  21 |
| `prover.rap_constraints.round0_time_ms` |  36 |  36 |  36 |  36 |
| `prover.rap_constraints.mle_rounds_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings.stacked_reduction_time_ms` |  13 |  13 |  13 |  13 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  9 |  9 |  9 |  9 |
| `prover.openings.whir_time_ms` |  85 |  85 |  85 |  85 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  183 |  183 |  183 |  183 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  171 |  171 |  171 |  171 |
| `prover.main_trace_commit_time_ms` |  47 |  47 |  47 |  47 |
| `prover.rap_constraints_time_ms` |  67 |  67 |  67 |  67 |
| `prover.openings_time_ms` |  55 |  55 |  55 |  55 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  8 |  8 |  8 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  6 |  6 |  6 |  6 |
| `prover.openings.whir_time_ms` |  47 |  47 |  47 |  47 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  150 |  150 |  150 |  150 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  139 |  139 |  139 |  139 |
| `prover.main_trace_commit_time_ms` |  32 |  32 |  32 |  32 |
| `prover.rap_constraints_time_ms` |  60 |  60 |  60 |  60 |
| `prover.openings_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints.logup_gkr_time_ms` |  13 |  13 |  13 |  13 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  25 |  25 |  25 |  25 |
| `prover.openings.stacked_reduction_time_ms` |  7 |  7 |  7 |  7 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  5 |  5 |  5 |  5 |
| `prover.openings.whir_time_ms` |  39 |  39 |  39 |  39 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/87700d72e28c8cc758d9d76eada7bf8d5bd322e5/sha2_bench-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.stacked_commit | 14.38 | app_proof.prover.1 |
| frac_sumcheck.gkr_rounds | 12.38 | app_proof.prover.1 |
| prover.batch_constraints.before_round0 | 12.38 | app_proof.prover.1 |
| prover.gkr_input_evals | 12.03 | app_proof.prover.0 |
| frac_sumcheck.segment_tree | 12.03 | app_proof.prover.0 |
| prover.rap_constraints | 10.63 | app_proof.prover.0 |
| prover.batch_constraints.round0 | 10.63 | app_proof.prover.0 |
| prover.batch_constraints.fold_ple_evals | 10.63 | app_proof.prover.0 |
| prover.openings | 9.70 | app_proof.prover.1 |
| prover.merkle_tree | 9.70 | app_proof.prover.1 |
| prover.prove_whir_opening | 9.70 | app_proof.prover.1 |
| prover.rs_code_matrix | 9.70 | app_proof.prover.1 |
| prover.before_gkr_input_evals | 4.89 | app_proof.prover.1 |
| tracegen.exp_bits_len | 1.11 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 1.11 | leaf.0 |
| tracegen.pow_checker | 1.11 | leaf.0 |
| tracegen.whir_folding | 0.98 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.98 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.98 | leaf.0 |
| generate mem proving ctxs | 0.87 | app_proof.1 |
| set initial memory | 0.87 | app_proof.1 |
| tracegen.proof_shape | 0.81 | leaf.0 |
| tracegen.public_values | 0.81 | leaf.0 |
| tracegen.range_checker | 0.81 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 130 | 9 | 0 | 267,239 | 228,668 | 9 | 64 | 

| air_id | air_name | need_rot | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| 0 | ProgramAir |  | 1 |  | 1 | 
| 1 | VmConnectorAir | 1 | 5 | 9 | 3 | 
| 10 | Sha2MainAir<Sha512Config> | 1 | 149 | 39 | 3 | 
| 11 | Sha2BlockHasherVmAir<Sha512Config> | 1 | 53 | 1,481 | 3 | 
| 12 | Sha2MainAir<Sha256Config> | 1 | 85 | 23 | 3 | 
| 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 29 | 754 | 3 | 
| 14 | Rv64HintStoreAir | 1 | 17 | 15 | 3 | 
| 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> |  | 14 | 5 | 3 | 
| 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> |  | 15 | 9 | 3 | 
| 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> |  | 12 | 11 | 2 | 
| 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> |  | 14 | 25 | 3 | 
| 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> |  | 11 | 11 | 3 | 
| 2 | PersistentBoundaryAir<8> |  | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> |  | 22 | 22 | 3 | 
| 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> |  | 25 | 31 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> |  | 29 | 77 | 3 | 
| 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> |  | 38 | 180 | 3 | 
| 24 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> |  | 19 | 30 | 3 | 
| 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> |  | 25 | 23 | 3 | 
| 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> |  | 32 | 34 | 3 | 
| 27 | BitwiseOperationLookupAir<8> | 1 | 2 | 19 | 2 | 
| 28 | PhantomAir |  | 3 | 1 | 2 | 
| 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 1 | 282 | 3 | 
| 3 | MemoryMerkleAir<8> | 1 | 4 | 35 | 3 | 
| 30 | VariableRangeCheckerAir | 1 | 1 | 10 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> |  | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> |  | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> |  | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> |  | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> |  | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 71 | 9 |  |  | 0 |  | 347 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 5,037 |  |  | 100 | 11,167,961 | 110.98 | 0 | 10,215 |  | 
| internal_for_leaf |  |  |  | 346 |  |  |  |  |  |  |  |  |  | 346 | 
| internal_recursive.0 |  |  |  | 183 |  |  |  |  |  |  |  |  |  | 183 | 
| internal_recursive.1 |  |  |  | 150 |  |  |  |  |  |  |  |  |  | 150 | 
| leaf |  |  | 1,018 |  |  |  |  |  |  |  |  |  |  | 1,018 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | Sha2MainAir<Sha256Config> | 0 | 4 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 26 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 2 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 28 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 72 | 
| app_proof | Sha2MainAir<Sha256Config> | 1 | 2 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 9 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 3 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 2 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 9 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 71 | 

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
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
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
| leaf | 31 | EqBitsAir | 0 | prover | 2,048 | 16 | 32,768 | 
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
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 0 | 83,663 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_BUFFER | 0 | 1 | 
| app_proof | 14 | Rv64HintStoreAir | HINT_STORED | 0 | 1 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 0 | 83,676 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 0 | 250,995 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 0 | 252,298 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 0 | 84,977 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 0 | 83,665 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 0 | 252,301 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 0 | 335,963 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 0 | 167,394 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 0 | 83,663 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 0 | 837,948 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 0 | 838,481 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | SRL | 0 | 1 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | SLL | 0 | 83,664 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | SRL | 0 | 83,666 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | SLTU | 0 | 2 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | ADD | 0 | 1 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | SUB | 0 | 83,663 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | ADD | 0 | 1,427,659 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | AND | 0 | 418,323 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | OR | 0 | 1 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | SUB | 0 | 250,992 | 
| app_proof | 28 | PhantomAir | PHANTOM | 0 | 1 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | MUL | 0 | 1 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | SHA256 | 1 | 80,178 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | AUIPC | 1 | 80,182 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | JALR | 1 | 240,541 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | JAL | 1 | 241,789 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | LUI | 1 | 81,432 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BGEU | 1 | 80,179 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLT | 1 | 1 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | BLTU | 1 | 241,790 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BEQ | 1 | 321,970 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | BNE | 1 | 160,371 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADB | 1 | 8 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | LOADW | 1 | 80,186 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | LOADD | 1 | 803,076 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STOREB | 1 | 40 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | STORED | 1 | 803,062 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | SLL | 1 | 1 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | SRL | 1 | 2 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | SLL | 1 | 80,192 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | SRL | 1 | 80,216 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | SUB | 1 | 80,179 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | ADD | 1 | 1,368,111 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | AND | 1 | 400,902 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | OR | 1 | 14 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | SUB | 1 | 240,538 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 0 | 131,072 | 150 | 19,660,800 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 0 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 14 | Rv64HintStoreAir | prover | 0 | 2 | 27 | 54 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 131,072 | 17 | 2,228,224 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 262,144 | 24 | 6,291,456 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 524,288 | 18 | 9,437,184 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 0 | 524,288 | 32 | 16,777,216 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 524,288 | 26 | 13,631,488 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 131,072 | 46 | 6,029,312 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 2,097,152 | 54 | 113,246,208 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 1 | 58 | 58 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 262,144 | 73 | 19,136,512 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | prover | 0 | 2 | 38 | 76 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 131,072 | 41 | 5,373,952 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 2,097,152 | 48 | 100,663,296 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 28 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 1 | 43 | 43 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 16,384 | 10 | 163,840 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | prover | 1 | 131,072 | 150 | 19,660,800 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | prover | 1 | 2,097,152 | 456 | 956,301,312 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 1 | 131,072 | 17 | 2,228,224 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 1 | 262,144 | 24 | 6,291,456 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 1 | 524,288 | 18 | 9,437,184 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | prover | 1 | 524,288 | 32 | 16,777,216 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 524,288 | 26 | 13,631,488 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 1,024 | 21 | 21,504 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 1 | 131,072 | 46 | 6,029,312 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 1 | 2,097,152 | 54 | 113,246,208 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 4 | 58 | 232 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 1 | 262,144 | 73 | 19,136,512 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 131,072 | 41 | 5,373,952 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 2,097,152 | 48 | 100,663,296 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 256 | 300 | 76,800 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 1,024 | 32 | 32,768 | 
| app_proof | 30 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 

| group | air_id | air_name | reason | segment | segmentation_trigger |
| --- | --- | --- | --- | --- | --- |
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | height | 0 | 1 | 

| group | air_id | air_name | segment | metered_rows_unpadded | metered_rows_padding | metered_main_secondary_memory_unpadded_bytes | metered_main_secondary_memory_padding_bytes | metered_main_memory_unpadded_bytes | metered_main_memory_padding_bytes | metered_main_cells_unpadded | metered_main_cells_padding | metered_interaction_memory_unpadded_bytes | metered_interaction_memory_padding_bytes | metered_interaction_cells_unpadded | metered_interaction_cells_padding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | 0 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 0 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 0 | 83,663 | 47,409 | 62,747,250 | 35,556,750 | 50,197,800 | 28,445,400 | 12,549,450 | 7,111,350 | 257,786,619 | 146,078,981 | 7,111,355 | 4,029,765 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 0 | 1,422,271 | 674,881 | 3,242,777,880 | 1,538,728,680 | 2,594,222,304 | 1,230,982,944 | 648,555,576 | 307,745,736 | 1,495,162,389 | 709,468,651 | 41,245,859 | 19,571,549 | 
| app_proof | 14 | Rv64HintStoreAir | 0 | 2 |  | 270 |  | 216 |  | 54 |  | 1,233 |  | 34 |  | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 83,676 | 47,396 | 3,556,230 | 2,014,330 | 5,689,968 | 3,222,928 | 1,422,492 | 805,732 | 42,465,570 | 24,053,470 | 1,171,464 | 663,544 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 250,995 | 11,149 | 15,059,700 | 668,940 | 24,095,520 | 1,070,304 | 6,023,880 | 267,576 | 136,478,532 | 6,062,268 | 3,764,925 | 167,235 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 337,275 | 187,013 | 15,177,375 | 8,415,585 | 24,283,800 | 13,464,936 | 6,070,950 | 3,366,234 | 146,714,625 | 81,350,655 | 4,047,300 | 2,244,156 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 0 | 335,966 | 188,322 | 26,877,280 | 15,065,760 | 43,003,648 | 24,105,216 | 10,750,912 | 6,026,304 | 170,502,745 | 95,573,415 | 4,703,524 | 2,636,508 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 503,357 | 20,931 | 32,718,205 | 1,360,515 | 52,349,128 | 2,176,824 | 13,087,282 | 544,206 | 200,713,604 | 8,346,236 | 5,536,927 | 230,241 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 0 | 1,152 | 896 | 60,480 | 47,040 | 96,768 | 75,264 | 24,192 | 18,816 | 167,040 | 129,920 | 4,608 | 3,584 | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 83,663 | 47,409 | 9,621,245 | 5,452,035 | 15,393,992 | 8,723,256 | 3,848,498 | 2,180,814 | 66,721,243 | 37,808,677 | 1,840,586 | 1,042,998 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 1,676,429 | 420,723 | 226,317,915 | 56,797,605 | 362,108,664 | 90,876,168 | 90,527,166 | 22,719,042 | 1,519,263,782 | 381,280,218 | 41,910,725 | 10,518,075 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 1 |  | 145 |  | 232 |  | 58 |  | 1,052 |  | 29 |  | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 167,330 | 94,814 | 30,537,725 | 17,303,555 | 48,860,360 | 27,685,688 | 12,215,090 | 6,921,422 | 230,497,075 | 130,606,285 | 6,358,540 | 3,602,932 | 
| app_proof | 24 | VmAirWrapper<Rv64BaseAluU16AdapterAir, LessThanCoreAir<4, 16> | 0 | 2 |  | 190 |  | 304 |  | 76 |  | 1,378 |  | 38 |  | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 83,664 | 47,408 | 8,575,560 | 4,859,320 | 13,720,896 | 7,774,912 | 3,430,224 | 1,943,728 | 75,820,500 | 42,963,500 | 2,091,600 | 1,185,200 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 2,096,975 | 177 | 251,637,000 | 21,240 | 402,619,200 | 33,984 | 100,654,800 | 8,496 | 2,432,491,000 | 205,320 | 67,103,200 | 5,664 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | 0 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 28 | PhantomAir | 0 | 1 |  | 15 |  | 24 |  | 6 |  | 109 |  | 3 |  | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 2,682 | 1,414 | 2,011,500 | 1,060,500 | 3,218,400 | 1,696,800 | 804,600 | 424,200 | 97,223 | 51,257 | 2,682 | 1,414 | 
| app_proof | 3 | MemoryMerkleAir<8> | 0 | 1,530 | 518 | 244,800 | 82,880 | 195,840 | 66,304 | 48,960 | 16,576 | 221,850 | 75,110 | 6,120 | 2,072 | 
| app_proof | 30 | VariableRangeCheckerAir | 0 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 |  | 108 |  | 172 |  | 43 |  | 1,124 |  | 31 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 0 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 
| app_proof | 0 | ProgramAir | 1 | 9,244 | 7,140 | 231,100 | 178,500 | 369,760 | 285,600 | 92,440 | 71,400 | 335,095 | 258,825 | 9,244 | 7,140 | 
| app_proof | 1 | VmConnectorAir | 1 | 2 |  | 60 |  | 48 |  | 12 |  | 363 |  | 10 |  | 
| app_proof | 12 | Sha2MainAir<Sha256Config> | 1 | 80,178 | 50,894 | 60,133,500 | 38,170,500 | 48,106,800 | 30,536,400 | 12,026,700 | 7,634,100 | 247,048,463 | 156,817,137 | 6,815,130 | 4,325,990 | 
| app_proof | 13 | Sha2BlockHasherVmAir<Sha256Config> | 1 | 1,363,026 | 734,126 | 3,107,699,280 | 1,673,807,280 | 2,486,159,424 | 1,339,045,824 | 621,539,856 | 334,761,456 | 1,432,881,083 | 771,749,957 | 39,527,754 | 21,289,654 | 
| app_proof | 15 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 1 | 80,182 | 50,890 | 3,407,735 | 2,162,825 | 5,452,376 | 3,460,520 | 1,363,094 | 865,130 | 40,692,365 | 25,826,675 | 1,122,548 | 712,460 | 
| app_proof | 16 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 1 | 240,541 | 21,603 | 14,432,460 | 1,296,180 | 23,091,936 | 2,073,888 | 5,772,984 | 518,472 | 130,794,169 | 11,746,631 | 3,608,115 | 324,045 | 
| app_proof | 17 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 1 | 323,221 | 201,067 | 14,544,945 | 9,048,015 | 23,271,912 | 14,476,824 | 5,817,978 | 3,619,206 | 140,601,135 | 87,464,145 | 3,878,652 | 2,412,804 | 
| app_proof | 18 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<4, 16> | 1 | 321,970 | 202,318 | 25,757,600 | 16,185,440 | 41,212,160 | 25,896,704 | 10,303,040 | 6,474,176 | 163,399,775 | 102,676,385 | 4,507,580 | 2,832,452 | 
| app_proof | 19 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 482,341 | 41,947 | 31,352,165 | 2,726,555 | 50,163,464 | 4,362,488 | 12,540,866 | 1,090,622 | 192,333,474 | 16,726,366 | 5,305,751 | 461,417 | 
| app_proof | 2 | PersistentBoundaryAir<8> | 1 | 1,024 |  | 53,760 |  | 86,016 |  | 21,504 |  | 148,480 |  | 4,096 |  | 
| app_proof | 20 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 1 | 80,194 | 50,878 | 9,222,310 | 5,850,970 | 14,755,696 | 9,361,552 | 3,688,924 | 2,340,388 | 63,954,715 | 40,575,205 | 1,764,268 | 1,119,316 | 
| app_proof | 21 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 1 | 1,606,178 | 490,974 | 216,834,030 | 66,281,490 | 346,934,448 | 106,050,384 | 86,733,612 | 26,512,596 | 1,455,598,813 | 444,945,187 | 40,154,450 | 12,274,350 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 1 | 3 | 1 | 435 | 145 | 696 | 232 | 174 | 58 | 3,154 | 1,051 | 87 | 29 | 
| app_proof | 23 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 1 | 160,408 | 101,736 | 29,274,460 | 18,566,820 | 46,839,136 | 29,706,912 | 11,709,784 | 7,426,728 | 220,962,020 | 140,141,340 | 6,095,504 | 3,865,968 | 
| app_proof | 25 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 1 | 80,179 | 50,893 | 8,218,348 | 5,216,532 | 13,149,356 | 8,346,452 | 3,287,339 | 2,086,613 | 72,662,219 | 46,121,781 | 2,004,475 | 1,272,325 | 
| app_proof | 26 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 2,009,565 | 87,587 | 241,147,800 | 10,510,440 | 385,836,480 | 16,816,704 | 96,459,120 | 4,204,176 | 2,331,095,400 | 101,600,920 | 64,306,080 | 2,802,784 | 
| app_proof | 27 | BitwiseOperationLookupAir<8> | 1 | 65,536 |  | 5,898,240 |  | 4,718,592 |  | 1,179,648 |  | 4,751,360 |  | 131,072 |  | 
| app_proof | 29 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,384 | 1,712 | 1,788,000 | 1,284,000 | 2,860,800 | 2,054,400 | 715,200 | 513,600 | 86,420 | 62,060 | 2,384 | 1,712 | 
| app_proof | 3 | MemoryMerkleAir<8> | 1 | 1,360 | 688 | 217,600 | 110,080 | 174,080 | 88,064 | 43,520 | 22,016 | 197,200 | 99,760 | 5,440 | 2,752 | 
| app_proof | 30 | VariableRangeCheckerAir | 1 | 262,144 |  | 5,242,880 |  | 4,194,304 |  | 1,048,576 |  | 9,502,720 |  | 262,144 |  | 
| app_proof | 9 | RangeTupleCheckerAir<2> | 1 | 1,048,576 |  | 15,728,640 |  | 12,582,912 |  | 3,145,728 |  | 38,010,880 |  | 1,048,576 |  | 

| group | backend | compile_metered_time_ms |
| --- | --- | --- |
| app_proof | interpreter | 3 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 21 | 346 | 21 | 6 | 1 | 2 | 2 | 2 | 
| internal_recursive.0 | 1 | 11 | 183 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 10 | 150 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 122 | 1,018 | 122 | 40 | 21 | 2 | 11 | 11 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,749 | 325 | 117 | 0 | 0 | 108 | 36 | 35 | 51 | 21 | 0 | 98 | 85 | 13 | 3 | 9 | 117 | 108 | 0 | 3 | 19 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,769 | 171 | 47 | 0 | 0 | 67 | 25 | 24 | 28 | 13 | 0 | 55 | 47 | 8 | 1 | 6 | 47 | 67 | 0 | 2 | 13 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,865 | 139 | 32 | 0 | 0 | 60 | 22 | 21 | 25 | 13 | 0 | 46 | 39 | 7 | 1 | 5 | 32 | 60 | 0 | 2 | 12 | 0 | 0 | 
| leaf | 0 | prover | 237,893,432 | 895 | 374 | 0 | 0 | 230 | 111 | 110 | 67 | 51 | 0 | 290 | 252 | 37 | 19 | 17 | 374 | 230 | 0 | 9 | 50 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,827 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,383 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,359 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 15,079,301 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 1,274,446,073 | 4,587 | 1,312 | 0 | 0 | 2,398 | 1,307 | 1,306 | 485 | 605 | 0 | 876 | 669 | 206 | 129 | 76 | 1,312 | 2,398 | 0 | 5 | 604 | 0 | 0 | 
| app_proof | prover | 1 | 1,274,446,068 | 4,489 | 1,233 | 0 | 0 | 2,385 | 1,298 | 1,297 | 481 | 605 | 0 | 870 | 664 | 206 | 129 | 76 | 1,233 | 2,385 | 0 | 5 | 603 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 234,250,641 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 234,250,622 | 2,013,265,921 | 

| group | segment | update_merkle_tree_time_ms | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | metered_memory_unpadded_bytes | metered_memory_padding_bytes | metered_memory_bytes | metered_interaction_memory_overhead_bytes | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 1 | 152 | 5,072 | 152 | 138 | 10,985,768,556 | 4,322,059,584 | 15,307,828,140 | 2,097,152 | 0 | 3 | 194 | 5,703,000 | 32.37 | 
| app_proof | 1 | 1 | 110 | 5,037 | 110 | 41 | 10,529,881,188 | 4,777,688,844 | 15,307,570,032 | 2,097,152 | 0 | 2 | 396 | 5,464,961 | 35.50 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/87700d72e28c8cc758d9d76eada7bf8d5bd322e5

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27821591264)
