| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  2.80 |  1.71 |  1.71 |
| app_proof |  1.64 |  0.55 |  0.55 |
| leaf |  0.54 |  0.54 |  0.54 |
| internal_for_leaf |  0.30 |  0.30 |  0.30 |
| internal_recursive.0 |  0.17 |  0.17 |  0.17 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  405.25 |  1,621 |  532 |  225 |
| `execute_metered_time_ms` |  19 | -          | -          | -          |
| `execute_metered_insns` |  4,000,051 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  200.61 | -          |  200.61 |  200.61 |
| `execute_preflight_insns` |  1,000,012.75 |  4,000,051 |  1,310,000 |  70,051 |
| `execute_preflight_time_ms` |  38.75 |  155 |  55 |  4 |
| `execute_preflight_insn_mi/s` |  35.12 | -          |  36.06 |  32.80 |
| `trace_gen_time_ms   ` |  72.25 |  289 |  101 |  57 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  228 |  912 |  280 |  77 |
| `prover.main_trace_commit_time_ms` |  48 |  192 |  59 |  16 |
| `prover.rap_constraints_time_ms` |  145.50 |  582 |  183 |  36 |
| `prover.openings_time_ms` |  33 |  132 |  37 |  24 |
| `prover.rap_constraints.logup_gkr_time_ms` |  100.25 |  401 |  129 |  18 |
| `prover.rap_constraints.round0_time_ms` |  28.75 |  115 |  36 |  9 |
| `prover.rap_constraints.mle_rounds_time_ms` |  15.25 |  61 |  18 |  7 |
| `prover.openings.stacked_reduction_time_ms` |  14.25 |  57 |  19 |  8 |
| `prover.openings.stacked_reduction.round0_time_ms` |  4.75 |  19 |  6 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  8.75 |  35 |  12 |  6 |
| `prover.openings.whir_time_ms` |  18.50 |  74 |  22 |  16 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  536 |  536 |  536 |  536 |
| `execute_preflight_time_ms` |  1 |  1 |  1 |  1 |
| `trace_gen_time_ms   ` |  59 |  59 |  59 |  59 |
| `generate_blob_total_time_ms` |  3 |  3 |  3 |  3 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  477 |  477 |  477 |  477 |
| `prover.main_trace_commit_time_ms` |  227 |  227 |  227 |  227 |
| `prover.rap_constraints_time_ms` |  182 |  182 |  182 |  182 |
| `prover.openings_time_ms` |  67 |  67 |  67 |  67 |
| `prover.rap_constraints.logup_gkr_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.round0_time_ms` |  79 |  79 |  79 |  79 |
| `prover.rap_constraints.mle_rounds_time_ms` |  51 |  51 |  51 |  51 |
| `prover.openings.stacked_reduction_time_ms` |  38 |  38 |  38 |  38 |
| `prover.openings.stacked_reduction.round0_time_ms` |  11 |  11 |  11 |  11 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.whir_time_ms` |  28 |  28 |  28 |  28 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  297 |  297 |  297 |  297 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  18 |  18 |  18 |  18 |
| `generate_blob_total_time_ms` |  1 |  1 |  1 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  279 |  279 |  279 |  279 |
| `prover.main_trace_commit_time_ms` |  118 |  118 |  118 |  118 |
| `prover.rap_constraints_time_ms` |  110 |  110 |  110 |  110 |
| `prover.openings_time_ms` |  50 |  50 |  50 |  50 |
| `prover.rap_constraints.logup_gkr_time_ms` |  25 |  25 |  25 |  25 |
| `prover.rap_constraints.round0_time_ms` |  33 |  33 |  33 |  33 |
| `prover.rap_constraints.mle_rounds_time_ms` |  50 |  50 |  50 |  50 |
| `prover.openings.stacked_reduction_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction.round0_time_ms` |  3 |  3 |  3 |  3 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.whir_time_ms` |  21 |  21 |  21 |  21 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  174 |  174 |  174 |  174 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  11 |  11 |  11 |  11 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  163 |  163 |  163 |  163 |
| `prover.main_trace_commit_time_ms` |  49 |  49 |  49 |  49 |
| `prover.rap_constraints_time_ms` |  72 |  72 |  72 |  72 |
| `prover.openings_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.logup_gkr_time_ms` |  19 |  19 |  19 |  19 |
| `prover.rap_constraints.round0_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  17 |  17 |  17 |  17 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  150 |  150 |  150 |  150 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  141 |  141 |  141 |  141 |
| `prover.main_trace_commit_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints_time_ms` |  65 |  65 |  65 |  65 |
| `prover.openings_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c/fibonacci-32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| prover.rap_constraints | 3.64 | leaf.0.prover |
| prover.batch_constraints.round0 | 3.52 | leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 3.52 | leaf.0.prover |
| prover.merkle_tree | 3.45 | leaf.0.prover |
| prover.openings | 3.45 | leaf.0.prover |
| prover.prove_whir_opening | 3.45 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 3.43 | leaf.0.prover |
| frac_sumcheck.gkr_rounds | 3.43 | leaf.0.prover |
| prover.gkr_input_evals | 3.37 | leaf.0.prover |
| frac_sumcheck.segment_tree | 3.37 | leaf.0.prover |
| prover.before_gkr_input_evals | 3.02 | leaf.0.prover |
| prover.stacked_commit | 3.02 | leaf.0.prover |
| prover.rs_code_matrix | 2.99 | leaf.0.prover |
| generate mem proving ctxs | 2.83 | app_proof.0 |
| set initial memory | 2.83 | app_proof.1 |
| tracegen.whir_final_poly_query_eval | 0.96 | leaf.0 |
| tracegen.pow_checker | 0.96 | leaf.0 |
| tracegen.exp_bits_len | 0.96 | leaf.0 |
| tracegen.whir_folding | 0.70 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.70 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.70 | leaf.0 |
| tracegen.proof_shape | 0.68 | leaf.0 |
| tracegen.public_values | 0.68 | leaf.0 |
| tracegen.range_checker | 0.68 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 9 | 0 | 267,111 | 227,339 | 9 | 62 | 

| air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| 0 | ProgramAir | 1 |  | 1 | 
| 1 | VmConnectorAir | 5 | 8 | 3 | 
| 10 | Rv64HintStoreAir | 20 | 15 | 3 | 
| 11 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| 12 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 18 | 9 | 3 | 
| 13 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| 14 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 20 | 37 | 3 | 
| 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 19 | 15 | 3 | 
| 16 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| 18 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 29 | 77 | 3 | 
| 19 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 38 | 180 | 3 | 
| 2 | PersistentBoundaryAir<8> | 4 | 3 | 3 | 
| 20 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 25 | 44 | 3 | 
| 21 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 25 | 23 | 3 | 
| 22 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 32 | 34 | 3 | 
| 23 | BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| 24 | PhantomAir | 3 |  | 1 | 
| 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| 26 | VariableRangeCheckerAir | 1 | 10 | 3 | 
| 3 | MemoryMerkleAir<8> | 4 | 33 | 3 | 
| 4 | VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 30 | 65 | 3 | 
| 5 | VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 41 | 104 | 3 | 
| 6 | VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 40 | 11 | 2 | 
| 7 | VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 5 | 2 | 
| 8 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 31 | 4 | 2 | 
| 9 | RangeTupleCheckerAir<2> | 1 | 8 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 69 | 9 |  |  | 0 |  | 346 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 225 |  |  | 19 | 4,000,051 | 200.61 | 0 | 1,647 |  | 
| internal_for_leaf |  |  |  | 297 |  |  |  |  |  |  |  |  |  | 297 | 
| internal_recursive.0 |  |  |  | 174 |  |  |  |  |  |  |  |  |  | 175 | 
| internal_recursive.1 |  |  |  | 150 |  |  |  |  |  |  |  |  |  | 150 | 
| leaf |  |  | 536 |  |  |  |  |  |  |  |  |  |  | 536 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 2 | 5 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 2 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 3 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 3 | 0 | 

| group | air_id | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | VerifierPvsAir | 69 | 213 | 4 | 
| agg_keygen | 1 | VmPvsAir | 30 | 54 | 4 | 
| agg_keygen | 10 | EqSharpUniReceiverAir | 3 | 25 | 4 | 
| agg_keygen | 11 | EqUniAir | 3 | 31 | 4 | 
| agg_keygen | 12 | ExpressionClaimAir | 7 | 68 | 4 | 
| agg_keygen | 13 | InteractionsFoldingAir | 13 | 94 | 4 | 
| agg_keygen | 14 | ConstraintsFoldingAir | 10 | 42 | 4 | 
| agg_keygen | 15 | EqNegAir | 8 | 83 | 4 | 
| agg_keygen | 16 | TranscriptAir | 17 | 84 | 4 | 
| agg_keygen | 17 | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| agg_keygen | 18 | MerkleVerifyAir | 6 | 22 | 3 | 
| agg_keygen | 19 | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| agg_keygen | 2 | UnsetPvsAir | 1 | 2 | 2 | 
| agg_keygen | 20 | PublicValuesAir | 4 | 18 | 4 | 
| agg_keygen | 21 | RangeCheckerAir<8> | 1 | 3 | 2 | 
| agg_keygen | 22 | GkrInputAir | 19 | 19 | 4 | 
| agg_keygen | 23 | GkrLayerAir | 30 | 38 | 4 | 
| agg_keygen | 24 | GkrLayerSumcheckAir | 21 | 59 | 4 | 
| agg_keygen | 25 | GkrXiSamplerAir | 7 | 17 | 4 | 
| agg_keygen | 26 | OpeningClaimsAir | 22 | 98 | 4 | 
| agg_keygen | 27 | UnivariateRoundAir | 13 | 54 | 4 | 
| agg_keygen | 28 | SumcheckRoundsAir | 21 | 69 | 4 | 
| agg_keygen | 29 | StackingClaimsAir | 17 | 57 | 4 | 
| agg_keygen | 3 | SymbolicExpressionAir<BabyBearParameters> | 52 | 32 | 4 | 
| agg_keygen | 30 | EqBaseAir | 8 | 89 | 4 | 
| agg_keygen | 31 | EqBitsAir | 5 | 24 | 4 | 
| agg_keygen | 32 | WhirRoundAir | 31 | 28 | 4 | 
| agg_keygen | 33 | SumcheckAir | 19 | 47 | 4 | 
| agg_keygen | 34 | WhirQueryAir | 5 | 51 | 4 | 
| agg_keygen | 35 | InitialOpenedValuesAir | 13 | 145 | 4 | 
| agg_keygen | 36 | NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| agg_keygen | 37 | WhirFoldingAir | 4 | 15 | 3 | 
| agg_keygen | 38 | FinalPolyMleEvalAir | 13 | 19 | 4 | 
| agg_keygen | 39 | FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| agg_keygen | 4 | FractionsFolderAir | 17 | 41 | 4 | 
| agg_keygen | 40 | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| agg_keygen | 41 | ExpBitsLenAir | 2 | 44 | 3 | 
| agg_keygen | 5 | UnivariateSumcheckAir | 14 | 46 | 4 | 
| agg_keygen | 6 | MultilinearSumcheckAir | 14 | 60 | 4 | 
| agg_keygen | 7 | EqNsAir | 10 | 65 | 4 | 
| agg_keygen | 8 | Eq3bAir | 3 | 65 | 4 | 
| agg_keygen | 9 | EqSharpUniAir | 5 | 48 | 4 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 69 | 
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
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 2,816 | 
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
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
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
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
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
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 69 | 
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
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 2,816 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 4 | 69 | 276 | 
| leaf | 1 | VmPvsAir | 0 | prover | 4 | 32 | 128 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 64 | 17 | 1,088 | 
| leaf | 11 | EqUniAir | 0 | prover | 32 | 16 | 512 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 8,192 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 4,096 | 37 | 151,552 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 2,048 | 25 | 51,200 | 
| leaf | 15 | EqNegAir | 0 | prover | 64 | 40 | 2,560 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 720,896 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 262,144 | 301 | 78,905,344 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 131,072 | 37 | 4,849,664 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 128 | 42 | 5,376 | 
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
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 16,384 | 60 | 983,040 | 
| leaf | 30 | EqBaseAir | 0 | prover | 32 | 51 | 1,632 | 
| leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 65,536 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 16 | 46 | 736 | 
| leaf | 33 | SumcheckAir | 0 | prover | 64 | 38 | 2,432 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 2,048 | 32 | 65,536 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 89 | 5,832,704 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 16,384 | 28 | 458,752 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 32,768 | 31 | 1,015,808 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 4,096 | 34 | 139,264 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 1,048,576 | 45 | 47,185,920 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 1,856 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 128 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 65,536 | 16 | 1,048,576 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 512 | 24 | 12,288 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 512 | 33 | 16,896 | 
| leaf | 7 | EqNsAir | 0 | prover | 128 | 41 | 5,248 | 
| leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 409,600 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 64 | 17 | 1,088 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 12 | 
| app_proof | 10 | Rv64HintStoreAir | prover | 0 | 2 | 38 | 76 | 
| app_proof | 11 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 8 | 25 | 200 | 
| app_proof | 12 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 4 | 32 | 128 | 
| app_proof | 13 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 1 | 23 | 23 | 
| app_proof | 14 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 2 | 44 | 88 | 
| app_proof | 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 262,144 | 38 | 9,961,472 | 
| app_proof | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 8 | 56 | 448 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 32 | 21 | 672 | 
| app_proof | 20 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | prover | 0 | 1 | 53 | 53 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,179,648 | 
| app_proof | 24 | PhantomAir | prover | 0 | 1 | 6 | 6 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 256 | 300 | 76,800 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 256 | 32 | 8,192 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 1 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 12 | 
| app_proof | 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 1 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 8 | 21 | 168 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 1 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,179,648 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 128 | 300 | 38,400 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 64 | 32 | 2,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 1 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 2 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 2 | 2 | 6 | 12 | 
| app_proof | 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 2 | 262,144 | 38 | 9,961,472 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 2 | 8 | 21 | 168 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 2 | 1,048,576 | 52 | 54,525,952 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 2 | 65,536 | 18 | 1,179,648 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 2 | 128 | 300 | 38,400 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 2 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 2 | 64 | 32 | 2,048 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 2 | 1,048,576 | 3 | 3,145,728 | 
| app_proof | 0 | ProgramAir | prover | 3 | 4,096 | 10 | 40,960 | 
| app_proof | 1 | VmConnectorAir | prover | 3 | 2 | 6 | 12 | 
| app_proof | 12 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 3 | 1 | 32 | 32 | 
| app_proof | 15 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 3 | 16,384 | 38 | 622,592 | 
| app_proof | 17 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 3 | 2 | 56 | 112 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 3 | 16 | 21 | 336 | 
| app_proof | 22 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 3 | 65,536 | 52 | 3,407,872 | 
| app_proof | 23 | BitwiseOperationLookupAir<8> | prover | 3 | 65,536 | 18 | 1,179,648 | 
| app_proof | 25 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 3 | 256 | 300 | 76,800 | 
| app_proof | 26 | VariableRangeCheckerAir | prover | 3 | 262,144 | 4 | 1,048,576 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 3 | 256 | 32 | 8,192 | 
| app_proof | 9 | RangeTupleCheckerAir<2> | prover | 3 | 1,048,576 | 3 | 3,145,728 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 18 | 297 | 17 | 5 | 1 | 2 | 1 | 1 | 
| internal_recursive.0 | 1 | 11 | 174 | 11 | 1 | 0 | 2 | 1 | 1 | 
| internal_recursive.1 | 1 | 9 | 150 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 59 | 536 | 59 | 24 | 3 | 1 | 0 | 0 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 38,610,683 | 279 | 118 | 0 | 7 | 110 | 33 | 33 | 50 | 25 | 0 | 50 | 21 | 28 | 3 | 24 | 118 | 110 | 0 | 2 | 25 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 15,378,703 | 163 | 48 | 0 | 4 | 72 | 24 | 24 | 28 | 19 | 0 | 41 | 17 | 23 | 1 | 21 | 49 | 72 | 0 | 2 | 18 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 9,750,799 | 141 | 33 | 0 | 4 | 65 | 23 | 22 | 24 | 17 | 0 | 41 | 18 | 22 | 1 | 20 | 34 | 65 | 0 | 2 | 16 | 0 | 0 | 
| leaf | 0 | prover | 142,599,078 | 477 | 226 | 0 | 14 | 182 | 79 | 78 | 51 | 51 | 0 | 67 | 28 | 38 | 11 | 26 | 227 | 182 | 0 | 5 | 48 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,733,762 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,383,941 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 69,989,034 | 280 | 58 | 0 | 1 | 183 | 35 | 35 | 18 | 129 | 0 | 37 | 18 | 19 | 6 | 12 | 58 | 183 | 0 | 3 | 127 | 0 | 0 | 
| app_proof | prover | 1 | 69,942,964 | 280 | 59 | 0 | 1 | 182 | 36 | 36 | 18 | 127 | 0 | 37 | 22 | 15 | 6 | 9 | 59 | 182 | 0 | 3 | 126 | 0 | 0 | 
| app_proof | prover | 2 | 69,942,964 | 275 | 59 | 0 | 1 | 181 | 35 | 35 | 18 | 127 | 0 | 34 | 18 | 15 | 6 | 8 | 59 | 181 | 0 | 3 | 126 | 0 | 0 | 
| app_proof | prover | 3 | 9,530,860 | 77 | 16 | 0 | 1 | 36 | 9 | 9 | 7 | 18 | 0 | 24 | 16 | 8 | 1 | 6 | 16 | 36 | 0 | 2 | 17 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 39,982,896 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 39,981,482 | 2,013,265,921 | 
| app_proof | prover | 2 | 0 | 39,981,482 | 2,013,265,921 | 
| app_proof | prover | 3 | 0 | 3,855,742 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 71 | 532 | 71 | 136 | 0 | 3 | 44 | 1,310,000 | 32.80 | 
| app_proof | 1 | 57 | 434 | 57 | 41 | 0 | 1 | 55 | 1,310,000 | 36.06 | 
| app_proof | 2 | 60 | 430 | 60 | 41 | 0 | 1 | 52 | 1,310,000 | 36.06 | 
| app_proof | 3 | 101 | 225 | 101 | 41 | 0 | 1 | 4 | 70,051 | 35.56 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/32909eaf2482cc2ef0ff0b4c3e8848d22c447e7c

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25830117427)
