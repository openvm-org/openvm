| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  1.50 |  1.50 |  1.50 |
| app_proof |  0.65 |  0.65 |  0.65 |
| leaf |  0.29 |  0.29 |  0.29 |
| internal_for_leaf |  0.24 |  0.24 |  0.24 |
| internal_recursive.0 |  0.17 |  0.17 |  0.17 |
| internal_recursive.1 |  0.15 |  0.15 |  0.15 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  645 |  645 |  645 |  645 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insns` |  112,210 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  18.20 | -          |  18.20 |  18.20 |
| `execute_preflight_insns` |  112,210 |  112,210 |  112,210 |  112,210 |
| `execute_preflight_time_ms` |  12 |  12 |  12 |  12 |
| `execute_preflight_insn_mi/s` |  12.55 | -          |  12.55 |  12.55 |
| `trace_gen_time_ms   ` |  248 |  248 |  248 |  248 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  214 |  214 |  214 |  214 |
| `prover.main_trace_commit_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints_time_ms` |  138 |  138 |  138 |  138 |
| `prover.openings_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints.logup_gkr_time_ms` |  39 |  39 |  39 |  39 |
| `prover.rap_constraints.round0_time_ms` |  78 |  78 |  78 |  78 |
| `prover.rap_constraints.mle_rounds_time_ms` |  19 |  19 |  19 |  19 |
| `prover.openings.stacked_reduction_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  19 |  19 |  19 |  19 |
| `prover.openings.whir_time_ms` |  16 |  16 |  16 |  16 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  292 |  292 |  292 |  292 |
| `execute_preflight_time_ms` |  7 |  7 |  7 |  7 |
| `trace_gen_time_ms   ` |  37 |  37 |  37 |  37 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  254 |  254 |  254 |  254 |
| `prover.main_trace_commit_time_ms` |  77 |  77 |  77 |  77 |
| `prover.rap_constraints_time_ms` |  122 |  122 |  122 |  122 |
| `prover.openings_time_ms` |  54 |  54 |  54 |  54 |
| `prover.rap_constraints.logup_gkr_time_ms` |  51 |  51 |  51 |  51 |
| `prover.rap_constraints.round0_time_ms` |  41 |  41 |  41 |  41 |
| `prover.rap_constraints.mle_rounds_time_ms` |  28 |  28 |  28 |  28 |
| `prover.openings.stacked_reduction_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.stacked_reduction.round0_time_ms` |  4 |  4 |  4 |  4 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.whir_time_ms` |  28 |  28 |  28 |  28 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  243 |  243 |  243 |  243 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  15 |  15 |  15 |  15 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  228 |  228 |  228 |  228 |
| `prover.main_trace_commit_time_ms` |  82 |  82 |  82 |  82 |
| `prover.rap_constraints_time_ms` |  99 |  99 |  99 |  99 |
| `prover.openings_time_ms` |  46 |  46 |  46 |  46 |
| `prover.rap_constraints.logup_gkr_time_ms` |  24 |  24 |  24 |  24 |
| `prover.rap_constraints.round0_time_ms` |  30 |  30 |  30 |  30 |
| `prover.rap_constraints.mle_rounds_time_ms` |  44 |  44 |  44 |  44 |
| `prover.openings.stacked_reduction_time_ms` |  26 |  26 |  26 |  26 |
| `prover.openings.stacked_reduction.round0_time_ms` |  2 |  2 |  2 |  2 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.whir_time_ms` |  20 |  20 |  20 |  20 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  167 |  167 |  167 |  167 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  10 |  10 |  10 |  10 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  157 |  157 |  157 |  157 |
| `prover.main_trace_commit_time_ms` |  47 |  47 |  47 |  47 |
| `prover.rap_constraints_time_ms` |  70 |  70 |  70 |  70 |
| `prover.openings_time_ms` |  38 |  38 |  38 |  38 |
| `prover.rap_constraints.logup_gkr_time_ms` |  18 |  18 |  18 |  18 |
| `prover.rap_constraints.round0_time_ms` |  23 |  23 |  23 |  23 |
| `prover.rap_constraints.mle_rounds_time_ms` |  27 |  27 |  27 |  27 |
| `prover.openings.stacked_reduction_time_ms` |  23 |  23 |  23 |  23 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  21 |  21 |  21 |  21 |
| `prover.openings.whir_time_ms` |  15 |  15 |  15 |  15 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  149 |  149 |  149 |  149 |
| `execute_preflight_time_ms` |  2 |  2 |  2 |  2 |
| `trace_gen_time_ms   ` |  9 |  9 |  9 |  9 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  139 |  139 |  139 |  139 |
| `prover.main_trace_commit_time_ms` |  34 |  34 |  34 |  34 |
| `prover.rap_constraints_time_ms` |  64 |  64 |  64 |  64 |
| `prover.openings_time_ms` |  40 |  40 |  40 |  40 |
| `prover.rap_constraints.logup_gkr_time_ms` |  17 |  17 |  17 |  17 |
| `prover.rap_constraints.round0_time_ms` |  22 |  22 |  22 |  22 |
| `prover.rap_constraints.mle_rounds_time_ms` |  24 |  24 |  24 |  24 |
| `prover.openings.stacked_reduction_time_ms` |  22 |  22 |  22 |  22 |
| `prover.openings.stacked_reduction.round0_time_ms` |  1 |  1 |  1 |  1 |
| `prover.openings.stacked_reduction.mle_rounds_time_ms` |  20 |  20 |  20 |  20 |
| `prover.openings.whir_time_ms` |  18 |  18 |  18 |  18 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



## GPU Memory Usage

![GPU Memory Usage](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/charts/516acf2b605bae3d10df65329981fd27dcceb188/ecrecover-516acf2b605bae3d10df65329981fd27dcceb188.memory.svg)

| Module | Max (GB) | Max At |
| --- | ---: | --- |
| generate mem proving ctxs | 4.84 | app_proof.0 |
| set initial memory | 4.84 | app_proof.0 |
| frac_sumcheck.gkr_rounds | 1.87 | leaf.0.prover |
| prover.batch_constraints.before_round0 | 1.87 | leaf.0.prover |
| prover.gkr_input_evals | 1.82 | leaf.0.prover |
| frac_sumcheck.segment_tree | 1.82 | leaf.0.prover |
| prover.rap_constraints | 1.80 | internal_for_leaf.0.prover |
| prover.batch_constraints.fold_ple_evals | 1.42 | internal_for_leaf.0.prover |
| prover.openings | 1.41 | internal_for_leaf.0.prover |
| prover.prove_whir_opening | 1.41 | internal_for_leaf.0.prover |
| prover.merkle_tree | 1.41 | internal_for_leaf.0.prover |
| prover.batch_constraints.round0 | 1.37 | internal_for_leaf.0.prover |
| prover.before_gkr_input_evals | 1.24 | internal_for_leaf.0.prover |
| prover.stacked_commit | 1.24 | internal_for_leaf.0.prover |
| prover.rs_code_matrix | 1.23 | internal_for_leaf.0.prover |
| tracegen.exp_bits_len | 0.54 | leaf.0 |
| tracegen.whir_final_poly_query_eval | 0.54 | leaf.0 |
| tracegen.pow_checker | 0.54 | leaf.0 |
| tracegen.whir_folding | 0.48 | leaf.0 |
| tracegen.whir_non_initial_opened_values | 0.47 | leaf.0 |
| tracegen.whir_initial_opened_values | 0.47 | leaf.0 |
| tracegen.proof_shape | 0.47 | leaf.0 |
| tracegen.public_values | 0.47 | leaf.0 |
| tracegen.range_checker | 0.47 | leaf.0 |

<details>
<summary>Detailed Metrics</summary>

| transport_pk_to_device_time_ms | stacked_commit_time_ms | rs_code_matrix_time_ms | proof_size_bytes.total | proof_size_bytes.compressed | merkle_tree_time_ms | memory_to_vec_partition_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 189 | 9 | 0 | 264,807 | 227,233 | 9 | 83 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| KeccakfOpAir | 210 | 27 | 2 | 
| KeccakfPermAir | 2 | 3,183 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| ProgramAir | 1 |  | 1 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| Rv64HintStoreAir | 20 | 15 | 3 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 24 | 34 | 3 | 
| VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 18 | 44 | 3 | 
| VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 30 | 180 | 3 | 
| VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 21 | 23 | 3 | 
| VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 25 | 77 | 3 | 
| VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 11 | 15 | 3 | 
| VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 13 | 37 | 3 | 
| VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 10 | 10 | 2 | 
| VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 49 | 214 | 3 | 
| VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 18 | 22 | 3 | 
| VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 17 | 31 | 3 | 
| VmAirWrapper<Rv64MultAdapterAir, DivRemCoreAir<8, 8> | 33 | 104 | 3 | 
| VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<8, 8> | 32 | 11 | 2 | 
| VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 23 | 4 | 2 | 
| VmAirWrapper<Rv64MultWAdapterAir, DivRemCoreAir<4, 8> | 26 | 65 | 3 | 
| VmAirWrapper<Rv64MultWAdapterAir, MultiplicationCoreAir<4, 8> | 20 | 5 | 2 | 
| VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 12 | 6 | 3 | 
| VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 463 | 280 | 3 | 
| VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 194 | 117 | 3 | 
| VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 500 | 257 | 2 | 
| VmConnectorAir | 5 | 8 | 3 | 
| XorinVmAir | 356 | 92 | 3 | 

| group | transport_pk_to_device_time_ms | stacked_commit_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_code_matrix_time_ms | prove_segment_time_ms | new_time_ms | merkle_tree_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 68 | 9 |  |  | 0 |  | 342 | 9 |  |  |  |  |  |  | 
| app_proof |  |  |  |  |  | 646 |  |  | 6 | 112,210 | 18.20 | 0 | 657 |  | 
| internal_for_leaf |  |  |  | 243 |  |  |  |  |  |  |  |  |  | 243 | 
| internal_recursive.0 |  |  |  | 167 |  |  |  |  |  |  |  |  |  | 167 | 
| internal_recursive.1 |  |  |  | 149 |  |  |  |  |  |  |  |  |  | 149 | 
| leaf |  |  | 292 |  |  |  |  |  |  |  |  |  |  | 292 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | KeccakfOpAir | 0 | 1 | 
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv64HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 30 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | 0 | 14 | 
| app_proof | XorinVmAir | 0 | 0 | 

| group | air_id | air_name | idx | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 345 | 
| internal_for_leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 152 | 
| internal_for_leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 464 | 
| internal_for_leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 224 | 
| internal_for_leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 7,680 | 
| internal_for_leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 729,088 | 
| internal_for_leaf | 14 | ConstraintsFoldingAir | 0 | prover | 4,096 | 25 | 266,240 | 
| internal_for_leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 1,152 | 
| internal_for_leaf | 16 | TranscriptAir | 0 | prover | 4,096 | 44 | 458,752 | 
| internal_for_leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 32,768 | 301 | 10,125,312 | 
| internal_for_leaf | 18 | MerkleVerifyAir | 0 | prover | 16,384 | 37 | 999,424 | 
| internal_for_leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 22,528 | 
| internal_for_leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| internal_for_leaf | 20 | PublicValuesAir | 0 | prover | 128 | 8 | 3,072 | 
| internal_for_leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| internal_for_leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 102 | 
| internal_for_leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 5,312 | 
| internal_for_leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 512 | 45 | 66,048 | 
| internal_for_leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 38 | 
| internal_for_leaf | 26 | OpeningClaimsAir | 0 | prover | 2,048 | 63 | 309,248 | 
| internal_for_leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 2,528 | 
| internal_for_leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 4,512 | 
| internal_for_leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 210,944 | 
| internal_for_leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 32,768 | 48 | 6,684,672 | 
| internal_for_leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 664 | 
| internal_for_leaf | 31 | EqBitsAir | 0 | prover | 4,096 | 16 | 147,456 | 
| internal_for_leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| internal_for_leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| internal_for_leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 8,192 | 89 | 1,155,072 | 
| internal_for_leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 180,224 | 
| internal_for_leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 385,024 | 
| internal_for_leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 88,064 | 
| internal_for_leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 17,039,360 | 
| internal_for_leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| internal_for_leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| internal_for_leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 393,216 | 
| internal_for_leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 10,240 | 
| internal_for_leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 11,392 | 
| internal_for_leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 2,592 | 
| internal_for_leaf | 8 | Eq3bAir | 0 | prover | 16,384 | 25 | 606,208 | 
| internal_for_leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 592 | 
| internal_recursive.0 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 345 | 
| internal_recursive.0 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 152 | 
| internal_recursive.0 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 116 | 
| internal_recursive.0 | 11 | EqUniAir | 1 | prover | 4 | 16 | 112 | 
| internal_recursive.0 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| internal_recursive.0 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 729,088 | 
| internal_recursive.0 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 266,240 | 
| internal_recursive.0 | 15 | EqNegAir | 1 | prover | 8 | 40 | 576 | 
| internal_recursive.0 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 458,752 | 
| internal_recursive.0 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 32,768 | 301 | 10,125,312 | 
| internal_recursive.0 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 499,712 | 
| internal_recursive.0 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 22,528 | 
| internal_recursive.0 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| internal_recursive.0 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 3,072 | 
| internal_recursive.0 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| internal_recursive.0 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 102 | 
| internal_recursive.0 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 5,312 | 
| internal_recursive.0 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 33,024 | 
| internal_recursive.0 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 38 | 
| internal_recursive.0 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 309,248 | 
| internal_recursive.0 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 632 | 
| internal_recursive.0 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 4,512 | 
| internal_recursive.0 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 52,736 | 
| internal_recursive.0 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 6,684,672 | 
| internal_recursive.0 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 332 | 
| internal_recursive.0 | 31 | EqBitsAir | 1 | prover | 2,048 | 16 | 73,728 | 
| internal_recursive.0 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 680 | 
| internal_recursive.0 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 1,824 | 
| internal_recursive.0 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 6,656 | 
| internal_recursive.0 | 35 | InitialOpenedValuesAir | 1 | prover | 16,384 | 89 | 2,310,144 | 
| internal_recursive.0 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 45,056 | 
| internal_recursive.0 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 96,256 | 
| internal_recursive.0 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 22,016 | 
| internal_recursive.0 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 1,064,960 | 
| internal_recursive.0 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| internal_recursive.0 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| internal_recursive.0 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 196,608 | 
| internal_recursive.0 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 1,280 | 
| internal_recursive.0 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 11,392 | 
| internal_recursive.0 | 7 | EqNsAir | 1 | prover | 32 | 41 | 2,592 | 
| internal_recursive.0 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 606,208 | 
| internal_recursive.0 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 148 | 
| internal_recursive.1 | 0 | VerifierPvsAir | 1 | prover | 1 | 69 | 345 | 
| internal_recursive.1 | 1 | VmPvsAir | 1 | prover | 1 | 32 | 152 | 
| internal_recursive.1 | 10 | EqSharpUniReceiverAir | 1 | prover | 4 | 17 | 116 | 
| internal_recursive.1 | 11 | EqUniAir | 1 | prover | 4 | 16 | 112 | 
| internal_recursive.1 | 12 | ExpressionClaimAir | 1 | prover | 128 | 32 | 7,680 | 
| internal_recursive.1 | 13 | InteractionsFoldingAir | 1 | prover | 8,192 | 37 | 729,088 | 
| internal_recursive.1 | 14 | ConstraintsFoldingAir | 1 | prover | 4,096 | 25 | 266,240 | 
| internal_recursive.1 | 15 | EqNegAir | 1 | prover | 8 | 40 | 576 | 
| internal_recursive.1 | 16 | TranscriptAir | 1 | prover | 4,096 | 44 | 458,752 | 
| internal_recursive.1 | 17 | Poseidon2Air<BabyBearParameters>, 1> | 1 | prover | 16,384 | 301 | 5,062,656 | 
| internal_recursive.1 | 18 | MerkleVerifyAir | 1 | prover | 8,192 | 37 | 499,712 | 
| internal_recursive.1 | 19 | ProofShapeAir<4, 8> | 1 | prover | 64 | 44 | 22,528 | 
| internal_recursive.1 | 2 | UnsetPvsAir | 1 | prover | 1 | 2 | 6 | 
| internal_recursive.1 | 20 | PublicValuesAir | 1 | prover | 128 | 8 | 3,072 | 
| internal_recursive.1 | 21 | RangeCheckerAir<8> | 1 | prover | 256 | 2 | 1,536 | 
| internal_recursive.1 | 22 | GkrInputAir | 1 | prover | 1 | 26 | 102 | 
| internal_recursive.1 | 23 | GkrLayerAir | 1 | prover | 32 | 46 | 5,312 | 
| internal_recursive.1 | 24 | GkrLayerSumcheckAir | 1 | prover | 256 | 45 | 33,024 | 
| internal_recursive.1 | 25 | GkrXiSamplerAir | 1 | prover | 1 | 10 | 38 | 
| internal_recursive.1 | 26 | OpeningClaimsAir | 1 | prover | 2,048 | 63 | 309,248 | 
| internal_recursive.1 | 27 | UnivariateRoundAir | 1 | prover | 8 | 27 | 632 | 
| internal_recursive.1 | 28 | SumcheckRoundsAir | 1 | prover | 32 | 57 | 4,512 | 
| internal_recursive.1 | 29 | StackingClaimsAir | 1 | prover | 512 | 35 | 52,736 | 
| internal_recursive.1 | 3 | SymbolicExpressionAir<BabyBearParameters> | 1 | prover | 32,768 | 48 | 6,684,672 | 
| internal_recursive.1 | 30 | EqBaseAir | 1 | prover | 4 | 51 | 332 | 
| internal_recursive.1 | 31 | EqBitsAir | 1 | prover | 4,096 | 16 | 147,456 | 
| internal_recursive.1 | 32 | WhirRoundAir | 1 | prover | 4 | 46 | 680 | 
| internal_recursive.1 | 33 | SumcheckAir | 1 | prover | 16 | 38 | 1,824 | 
| internal_recursive.1 | 34 | WhirQueryAir | 1 | prover | 128 | 32 | 6,656 | 
| internal_recursive.1 | 35 | InitialOpenedValuesAir | 1 | prover | 8,192 | 89 | 1,155,072 | 
| internal_recursive.1 | 36 | NonInitialOpenedValuesAir | 1 | prover | 1,024 | 28 | 45,056 | 
| internal_recursive.1 | 37 | WhirFoldingAir | 1 | prover | 2,048 | 31 | 96,256 | 
| internal_recursive.1 | 38 | FinalPolyMleEvalAir | 1 | prover | 256 | 34 | 22,016 | 
| internal_recursive.1 | 39 | FinalPolyQueryEvalAir | 1 | prover | 16,384 | 45 | 1,064,960 | 
| internal_recursive.1 | 4 | FractionsFolderAir | 1 | prover | 64 | 29 | 6,208 | 
| internal_recursive.1 | 40 | PowerCheckerAir<2, 32> | 1 | prover | 32 | 4 | 384 | 
| internal_recursive.1 | 41 | ExpBitsLenAir | 1 | prover | 8,192 | 16 | 196,608 | 
| internal_recursive.1 | 5 | UnivariateSumcheckAir | 1 | prover | 16 | 24 | 1,280 | 
| internal_recursive.1 | 6 | MultilinearSumcheckAir | 1 | prover | 128 | 33 | 11,392 | 
| internal_recursive.1 | 7 | EqNsAir | 1 | prover | 32 | 41 | 2,592 | 
| internal_recursive.1 | 8 | Eq3bAir | 1 | prover | 16,384 | 25 | 606,208 | 
| internal_recursive.1 | 9 | EqSharpUniAir | 1 | prover | 4 | 17 | 148 | 
| leaf | 0 | VerifierPvsAir | 0 | prover | 1 | 69 | 345 | 
| leaf | 1 | VmPvsAir | 0 | prover | 1 | 32 | 152 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 16 | 17 | 464 | 
| leaf | 11 | EqUniAir | 0 | prover | 8 | 16 | 224 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 128 | 32 | 7,680 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 16,384 | 37 | 1,458,176 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 532,480 | 
| leaf | 15 | EqNegAir | 0 | prover | 16 | 40 | 1,152 | 
| leaf | 16 | TranscriptAir | 0 | prover | 16,384 | 44 | 1,835,008 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 65,536 | 301 | 20,250,624 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 32,768 | 37 | 1,998,848 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 44 | 22,528 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 32 | 8 | 768 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 1 | 26 | 102 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 32 | 46 | 5,312 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 256 | 45 | 33,024 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 1 | 10 | 38 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 1,236,992 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 32 | 27 | 2,528 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 32 | 57 | 4,512 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 2,048 | 35 | 210,944 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 131,072 | 60 | 35,127,296 | 
| leaf | 30 | EqBaseAir | 0 | prover | 8 | 51 | 664 | 
| leaf | 31 | EqBitsAir | 0 | prover | 16,384 | 16 | 589,824 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 4 | 46 | 680 | 
| leaf | 33 | SumcheckAir | 0 | prover | 16 | 38 | 1,824 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 512 | 32 | 26,624 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 2,310,144 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 4,096 | 28 | 180,224 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 8,192 | 31 | 385,024 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 1,024 | 34 | 88,064 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 262,144 | 45 | 17,039,360 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 16,384 | 16 | 393,216 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 128 | 24 | 10,240 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 128 | 33 | 11,392 | 
| leaf | 7 | EqNsAir | 0 | prover | 32 | 41 | 2,592 | 
| leaf | 8 | Eq3bAir | 0 | prover | 65,536 | 25 | 2,424,832 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 16 | 17 | 592 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 32,768 | 10 | 458,752 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 32 | 290 | 34,112 | 
| app_proof | 11 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 226 | 11,936 | 
| app_proof | 16 | VmAirWrapper<Rv64MultAdapterAir, MultiplicationCoreAir<8, 8> | prover | 0 | 16 | 47 | 2,224 | 
| app_proof | 17 | RangeTupleCheckerAir<2> | prover | 0 | 1,048,576 | 3 | 7,340,032 | 
| app_proof | 18 | KeccakfOpAir | prover | 0 | 8 | 486 | 10,608 | 
| app_proof | 19 | KeccakfPermAir | prover | 0 | 128 | 2,634 | 338,176 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 4,096 | 20 | 131,072 | 
| app_proof | 20 | XorinVmAir | prover | 0 | 8 | 741 | 17,320 | 
| app_proof | 21 | Rv64HintStoreAir | prover | 0 | 128 | 38 | 15,104 | 
| app_proof | 22 | VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir> | prover | 0 | 4,096 | 25 | 299,008 | 
| app_proof | 23 | VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir> | prover | 0 | 2,048 | 32 | 196,608 | 
| app_proof | 24 | VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir> | prover | 0 | 4,096 | 23 | 258,048 | 
| app_proof | 25 | VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<8, 8> | prover | 0 | 1,024 | 44 | 98,304 | 
| app_proof | 26 | VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<8> | prover | 0 | 16,384 | 38 | 1,343,488 | 
| app_proof | 27 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendCoreAir<8, 8> | prover | 0 | 1,024 | 48 | 122,880 | 
| app_proof | 28 | VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<8> | prover | 0 | 65,536 | 56 | 8,126,464 | 
| app_proof | 29 | VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 256 | 66 | 42,496 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 4,096 | 32 | 196,608 | 
| app_proof | 30 | VmAirWrapper<Rv64BaseAluAdapterAir, ShiftCoreAir<8, 8> | prover | 0 | 16,384 | 77 | 3,227,648 | 
| app_proof | 31 | VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<8, 8> | prover | 0 | 1,024 | 53 | 128,000 | 
| app_proof | 32 | VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 4,096 | 49 | 544,768 | 
| app_proof | 33 | VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<8, 8> | prover | 0 | 65,536 | 52 | 9,699,328 | 
| app_proof | 34 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 35 | PhantomAir | prover | 0 | 16 | 6 | 288 | 
| app_proof | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 4,096 | 300 | 1,245,184 | 
| app_proof | 37 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 4 | VmAirWrapper<Rv64VecHeapAdapterAir<1, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 2,048 | 583 | 4,986,880 | 
| app_proof | 5 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 8, 8, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 1,024 | 679 | 2,743,296 | 
| app_proof | 6 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 32 | 188 | 12,288 | 
| app_proof | 7 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 290 | 17,056 | 
| app_proof | 8 | VmAirWrapper<Rv64VecHeapAdapterAir<2, 4, 4, 8, 8>, FieldExpressionCoreAir> | prover | 0 | 16 | 226 | 11,936 | 
| app_proof | 9 | VmAirWrapper<Rv64IsEqualModAdapterAir<2, 4, 8, 32>, ModularIsEqualCoreAir<32, 8, 8> | prover | 0 | 4,096 | 188 | 1,572,864 | 

| group | air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- | --- |
| agg_keygen | ConstraintsFoldingAir | 10 | 42 | 4 | 
| agg_keygen | Eq3bAir | 3 | 65 | 4 | 
| agg_keygen | EqBaseAir | 8 | 89 | 4 | 
| agg_keygen | EqBitsAir | 5 | 24 | 4 | 
| agg_keygen | EqNegAir | 8 | 83 | 4 | 
| agg_keygen | EqNsAir | 10 | 65 | 4 | 
| agg_keygen | EqSharpUniAir | 5 | 48 | 4 | 
| agg_keygen | EqSharpUniReceiverAir | 3 | 25 | 4 | 
| agg_keygen | EqUniAir | 3 | 31 | 4 | 
| agg_keygen | ExpBitsLenAir | 2 | 44 | 3 | 
| agg_keygen | ExpressionClaimAir | 7 | 68 | 4 | 
| agg_keygen | FinalPolyMleEvalAir | 13 | 19 | 4 | 
| agg_keygen | FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| agg_keygen | FractionsFolderAir | 17 | 41 | 4 | 
| agg_keygen | GkrInputAir | 19 | 19 | 4 | 
| agg_keygen | GkrLayerAir | 30 | 38 | 4 | 
| agg_keygen | GkrLayerSumcheckAir | 21 | 59 | 4 | 
| agg_keygen | GkrXiSamplerAir | 7 | 17 | 4 | 
| agg_keygen | InitialOpenedValuesAir | 13 | 145 | 4 | 
| agg_keygen | InteractionsFoldingAir | 13 | 94 | 4 | 
| agg_keygen | MerkleVerifyAir | 6 | 22 | 3 | 
| agg_keygen | MultilinearSumcheckAir | 14 | 60 | 4 | 
| agg_keygen | NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| agg_keygen | OpeningClaimsAir | 22 | 98 | 4 | 
| agg_keygen | Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| agg_keygen | PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| agg_keygen | ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| agg_keygen | PublicValuesAir | 4 | 18 | 4 | 
| agg_keygen | RangeCheckerAir<8> | 1 | 3 | 2 | 
| agg_keygen | StackingClaimsAir | 17 | 57 | 4 | 
| agg_keygen | SumcheckAir | 19 | 47 | 4 | 
| agg_keygen | SumcheckRoundsAir | 21 | 69 | 4 | 
| agg_keygen | SymbolicExpressionAir<BabyBearParameters> | 52 | 32 | 4 | 
| agg_keygen | TranscriptAir | 17 | 84 | 4 | 
| agg_keygen | UnivariateRoundAir | 13 | 54 | 4 | 
| agg_keygen | UnivariateSumcheckAir | 14 | 46 | 4 | 
| agg_keygen | UnsetPvsAir | 1 | 2 | 2 | 
| agg_keygen | VerifierPvsAir | 69 | 213 | 4 | 
| agg_keygen | VmPvsAir | 30 | 54 | 4 | 
| agg_keygen | WhirFoldingAir | 4 | 15 | 3 | 
| agg_keygen | WhirQueryAir | 5 | 51 | 4 | 
| agg_keygen | WhirRoundAir | 31 | 28 | 4 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_cuda_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 15 | 243 | 14 | 4 | 0 | 2 | 0 | 0 | 
| internal_recursive.0 | 1 | 10 | 167 | 9 | 1 | 0 | 2 | 0 | 0 | 
| internal_recursive.1 | 1 | 9 | 149 | 9 | 1 | 0 | 2 | 0 | 0 | 
| leaf | 0 | 37 | 292 | 37 | 5 | 0 | 7 | 8 | 8 | 

| group | idx | phase | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 39,955,203 | 228 | 82 | 0 | 5 | 99 | 30 | 29 | 44 | 24 | 0 | 46 | 20 | 26 | 2 | 23 | 82 | 99 | 0 | 2 | 23 | 0 | 0 | 
| internal_recursive.0 | 1 | prover | 23,651,975 | 157 | 47 | 0 | 3 | 70 | 23 | 23 | 27 | 18 | 0 | 38 | 15 | 23 | 1 | 21 | 47 | 70 | 0 | 2 | 17 | 0 | 0 | 
| internal_recursive.1 | 1 | prover | 17,507,975 | 139 | 33 | 0 | 3 | 64 | 22 | 22 | 24 | 17 | 0 | 40 | 18 | 22 | 1 | 20 | 34 | 64 | 0 | 2 | 16 | 0 | 0 | 
| leaf | 0 | prover | 86,202,627 | 254 | 77 | 0 | 5 | 122 | 41 | 41 | 28 | 51 | 0 | 54 | 28 | 26 | 4 | 22 | 77 | 122 | 0 | 5 | 48 | 0 | 0 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,348,738 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 9,851,522 | 2,013,265,921 | 

| group | phase | segment | total_cells | stark_prove_excluding_trace_time_ms | stacked_commit_time_ms | s'_0 -> s_0 cpu interpolations_time_ms | rs_code_matrix_time_ms | prover.rap_constraints_time_ms | prover.rap_constraints.round0_time_ms | prover.rap_constraints.ple_round0_time_ms | prover.rap_constraints.mle_rounds_time_ms | prover.rap_constraints.logup_gkr_time_ms | prover.rap_constraints.logup_gkr.input_evals_time_ms | prover.openings_time_ms | prover.openings.whir_time_ms | prover.openings.stacked_reduction_time_ms | prover.openings.stacked_reduction.round0_time_ms | prover.openings.stacked_reduction.mle_rounds_time_ms | prover.main_trace_commit_time_ms | prove_zerocheck_and_logup_gpu_time_ms | opened_rows_d2h_time_ms | merkle_tree_time_ms | fractional_sumcheck_gpu_time_ms | batch_open_rows_time_ms | LogupZerocheck::sumcheck_polys_batch_eval_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 47,033,916 | 214 | 37 | 0 | 17 | 138 | 78 | 77 | 19 | 39 | 0 | 38 | 16 | 21 | 2 | 19 | 38 | 138 | 0 | 2 | 38 | 0 | 0 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 6,815,258 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | system_trace_gen_time_ms | set_initial_memory_time_ms | memory_finalize_time_ms | generate_proving_ctxs_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 248 | 645 | 248 | 170 | 0 | 3 | 12 | 112,210 | 12.55 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/516acf2b605bae3d10df65329981fd27dcceb188

Max Segment Length: 1048576

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25065706258)
