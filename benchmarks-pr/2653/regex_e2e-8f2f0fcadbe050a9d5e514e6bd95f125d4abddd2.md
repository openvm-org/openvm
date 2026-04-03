| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  179.27 |  168.73 |  168.73 |
| app_proof |  22.06 |  11.51 |  11.51 |
| leaf |  4.63 |  4.63 |  4.63 |
| internal_for_leaf |  2.92 |  2.92 |  2.92 |
| internal_recursive.0 |  2.09 |  2.09 |  2.09 |
| internal_recursive.1 |  1.84 |  1.84 |  1.84 |
| root |  47.71 |  47.71 |  47.71 |
| halo2_outer |  74.73 |  74.73 |  74.73 |
| halo2_wrapper |  23.29 |  23.29 |  23.29 |


| app_proof |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,014 |  22,028 |  11,484 |  10,544 |
| `main_cells_used     ` |  81,960,359.50 |  163,920,719 |  90,181,874 |  73,738,845 |
| `total_cells_used    ` |  81,960,359.50 |  163,920,719 |  90,181,874 |  73,738,845 |
| `execute_metered_time_ms` |  29 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  139.38 | -          |  139.38 |  139.38 |
| `execute_preflight_insns` |  2,068,347 |  4,136,694 |  2,208,000 |  1,928,694 |
| `execute_preflight_time_ms` |  84 |  168 |  98 |  70 |
| `execute_preflight_insn_mi/s` |  37.33 | -          |  37.67 |  36.99 |
| `trace_gen_time_ms   ` |  360 |  720 |  450 |  270 |
| `memory_finalize_time_ms` |  4.50 |  9 |  8 |  1 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  10,177 |  20,354 |  10,476 |  9,878 |
| `prover.main_trace_commit_time_ms` |  487.50 |  975 |  542 |  433 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,633 |  4,633 |  4,633 |  4,633 |
| `execute_preflight_time_ms` |  28 |  28 |  28 |  28 |
| `trace_gen_time_ms   ` |  110 |  110 |  110 |  110 |
| `generate_blob_total_time_ms` |  2 |  2 |  2 |  2 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  4,523 |  4,523 |  4,523 |  4,523 |
| `prover.main_trace_commit_time_ms` |  537 |  537 |  537 |  537 |

| internal_for_leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,918 |  2,918 |  2,918 |  2,918 |
| `execute_preflight_time_ms` |  13 |  13 |  13 |  13 |
| `trace_gen_time_ms   ` |  36 |  36 |  36 |  36 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,882 |  2,882 |  2,882 |  2,882 |
| `prover.main_trace_commit_time_ms` |  304 |  304 |  304 |  304 |

| internal_recursive.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,094 |  2,094 |  2,094 |  2,094 |
| `execute_preflight_time_ms` |  9 |  9 |  9 |  9 |
| `trace_gen_time_ms   ` |  23 |  23 |  23 |  23 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  2,070 |  2,070 |  2,070 |  2,070 |
| `prover.main_trace_commit_time_ms` |  259 |  259 |  259 |  259 |

| internal_recursive.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,835 |  1,835 |  1,835 |  1,835 |
| `execute_preflight_time_ms` |  7 |  7 |  7 |  7 |
| `trace_gen_time_ms   ` |  19 |  19 |  19 |  19 |
| `generate_blob_total_time_ms` |  0 |  0 |  0 |  0 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  1,816 |  1,816 |  1,816 |  1,816 |
| `prover.main_trace_commit_time_ms` |  196 |  196 |  196 |  196 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  47,710 |  47,710 |  47,710 |  47,710 |
| `execute_preflight_time_ms` |  6 |  6 |  6 |  6 |
| __Prover__ |||||
| `stark_prove_excluding_trace_time_ms` |  47,710 |  47,710 |  47,710 |  47,710 |
| `prover.main_trace_commit_time_ms` |  23,269 |  23,269 |  23,269 |  23,269 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  74,732 |  74,732 |  74,732 |  74,732 |
| `halo2_verifier_k    ` |  23 |  23 |  23 |  23 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  23,292 |  23,292 |  23,292 |  23,292 |
| `halo2_wrapper_k     ` |  22 |  22 |  22 |  22 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|



<details>
<summary>Detailed Metrics</summary>

| transpose_to_rm_time_ms | trace_commit_cpu_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_matrix_time_ms | rs_encode_and_merkle_cpu_time_ms | row_hash_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | eval_to_coeff_phase_time_ms | digest_layers_time_ms | dft_batch_time_ms | compute_merkle_precomputation_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 104 | 23 | 1 | 101 | 20 | 16 | 0 | 0 | 0 | 6 | 14 | 30 | 34 | 4 | 4 | 

| air_name | interactions | constraints | constraint_deg |
| --- | --- | --- | --- |
| AccessAdapterAir<16> | 5 | 5 | 2 | 
| AccessAdapterAir<2> | 5 | 5 | 2 | 
| AccessAdapterAir<32> | 5 | 5 | 2 | 
| AccessAdapterAir<4> | 5 | 5 | 2 | 
| AccessAdapterAir<8> | 5 | 5 | 2 | 
| BitwiseOperationLookupAir<8> | 2 | 19 | 2 | 
| ConstraintsFoldingAir | 10 | 42 | 4 | 
| Eq3bAir | 3 | 65 | 4 | 
| EqBaseAir | 8 | 89 | 4 | 
| EqBitsAir | 5 | 24 | 4 | 
| EqNegAir | 8 | 83 | 4 | 
| EqNsAir | 10 | 65 | 4 | 
| EqSharpUniAir | 5 | 48 | 4 | 
| EqSharpUniReceiverAir | 3 | 25 | 4 | 
| EqUniAir | 3 | 31 | 4 | 
| ExpBitsLenAir | 2 | 44 | 3 | 
| ExpressionClaimAir | 7 | 68 | 4 | 
| FinalPolyMleEvalAir | 13 | 19 | 4 | 
| FinalPolyQueryEvalAir | 5 | 120 | 4 | 
| FractionsFolderAir | 17 | 41 | 4 | 
| GkrInputAir | 19 | 15 | 4 | 
| GkrLayerAir | 30 | 38 | 4 | 
| GkrLayerSumcheckAir | 21 | 59 | 4 | 
| GkrXiSamplerAir | 7 | 17 | 4 | 
| InitialOpenedValuesAir | 13 | 145 | 4 | 
| InteractionsFoldingAir | 13 | 94 | 4 | 
| KeccakVmAir | 321 | 4,247 | 3 | 
| MemoryMerkleAir<8> | 4 | 33 | 3 | 
| MerkleVerifyAir | 6 | 22 | 3 | 
| MultilinearSumcheckAir | 14 | 60 | 4 | 
| NonInitialOpenedValuesAir | 4 | 42 | 4 | 
| OpeningClaimsAir | 22 | 98 | 4 | 
| PersistentBoundaryAir<8> | 3 | 2 | 3 | 
| PhantomAir | 3 |  | 1 | 
| Poseidon2Air<BabyBearParameters>, 1> | 2 | 282 | 3 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 282 | 3 | 
| PowerCheckerAir<2, 32> | 2 | 5 | 2 | 
| ProgramAir | 1 |  | 1 | 
| ProofShapeAir<4, 8> | 77 | 82 | 4 | 
| PublicValuesAir | 4 | 18 | 4 | 
| RangeCheckerAir<8> | 1 | 3 | 2 | 
| RangeTupleCheckerAir<2> | 1 | 8 | 3 | 
| RootVerifierPvsAir | 108 | 36 | 2 | 
| Rv32HintStoreAir | 18 | 17 | 3 | 
| StackingClaimsAir | 17 | 57 | 4 | 
| SumcheckAir | 19 | 47 | 4 | 
| SumcheckRoundsAir | 21 | 69 | 4 | 
| SymbolicExpressionAir<BabyBearParameters> | 13 | 320 | 4 | 
| TranscriptAir | 17 | 84 | 4 | 
| UnivariateRoundAir | 13 | 54 | 4 | 
| UnivariateSumcheckAir | 14 | 46 | 4 | 
| UserPvsCommitAir | 5 | 41 | 4 | 
| UserPvsInMemoryAir | 3 | 13 | 4 | 
| VariableRangeCheckerAir | 1 | 10 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 22 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 28 | 3 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 76 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 11 | 3 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 25 | 3 | 
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 9 | 2 | 
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 9 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 18 | 18 | 3 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 25 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 25 | 64 | 3 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 24 | 11 | 2 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 4 | 2 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 5 | 3 | 
| VmConnectorAir | 5 | 8 | 3 | 
| WhirFoldingAir | 4 | 15 | 3 | 
| WhirQueryAir | 5 | 51 | 4 | 
| WhirRoundAir | 31 | 28 | 4 | 

| group | transpose_to_rm_time_ms | tracegen_attempt_time_ms | trace_commit_cpu_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | stacked_matrix_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | rs_encode_and_merkle_cpu_time_ms | row_hash_time_ms | root_time_ms | prove_segment_time_ms | new_time_ms | keygen_halo2_time_ms | halo2_wrapper_k | halo2_verifier_k | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | eval_to_coeff_phase_time_ms | digest_layers_time_ms | dft_batch_time_ms | compute_user_public_values_proof_time_ms | compute_merkle_precomputation_time_ms | apply_merkle_precomputation_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 5 |  | 110 |  |  | 0 |  |  | 108 | 23 |  |  | 574 |  |  |  |  |  |  |  |  |  |  |  | 8 | 38 | 32 |  |  |  |  |  | 
| app_proof |  |  |  |  |  |  |  |  |  |  |  | 10,544 |  |  |  |  |  |  |  |  |  | 29 | 4,136,694 | 139.38 |  |  |  | 0 |  |  | 22,088 |  | 
| halo2_keygen |  |  |  |  |  |  |  |  |  |  |  |  |  | 93,462 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_outer |  |  |  | 74,732 |  |  |  |  |  |  |  |  |  |  |  | 23 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  |  | 23,292 |  |  |  |  |  |  |  |  |  |  | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_for_leaf |  |  |  |  |  |  |  | 2,918 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,918 | 
| internal_recursive.0 |  |  |  |  |  |  |  | 2,094 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,094 | 
| internal_recursive.1 |  |  |  |  |  |  |  | 1,835 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,835 | 
| leaf |  |  |  |  |  |  | 4,633 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 4,633 | 
| root |  | 23 |  | 47,710 | 22 |  |  |  |  |  | 47,710 |  |  |  |  |  | 15 | 0 | 0 | 0 | 6 |  |  |  |  |  |  |  | 4 | 4 |  | 47,710 | 

| group | air | generate_cached_trace_time_ms |
| --- | --- | --- |
| agg_keygen | SymbolicExpressionAir | 0 | 
| root | MerkleVerify |  | 

| group | air | segment | single_trace_gen_time_ms |
| --- | --- | --- | --- |
| app_proof | PhantomAir | 0 | 0 | 
| app_proof | Rv32HintStoreAir | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 54 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 10 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 10 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 5 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 110 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 0 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 3 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2 | 
| app_proof | KeccakVmAir | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 46 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 9 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 8 | 
| app_proof | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 7 | 
| app_proof | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 1 | 
| app_proof | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 5 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 0 | 
| app_proof | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 98 | 
| app_proof | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 3 | 
| app_proof | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 1 | 

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
| internal_for_leaf | 35 | InitialOpenedValuesAir | 0 | prover | 16,384 | 89 | 2,310,144 | 
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
| leaf | 0 | VerifierPvsAir | 0 | prover | 2 | 69 | 690 | 
| leaf | 1 | VmPvsAir | 0 | prover | 2 | 32 | 304 | 
| leaf | 10 | EqSharpUniReceiverAir | 0 | prover | 32 | 17 | 928 | 
| leaf | 11 | EqUniAir | 0 | prover | 16 | 16 | 448 | 
| leaf | 12 | ExpressionClaimAir | 0 | prover | 256 | 32 | 15,360 | 
| leaf | 13 | InteractionsFoldingAir | 0 | prover | 8,192 | 37 | 729,088 | 
| leaf | 14 | ConstraintsFoldingAir | 0 | prover | 8,192 | 25 | 532,480 | 
| leaf | 15 | EqNegAir | 0 | prover | 32 | 40 | 2,304 | 
| leaf | 16 | TranscriptAir | 0 | prover | 8,192 | 44 | 917,504 | 
| leaf | 17 | Poseidon2Air<BabyBearParameters>, 1> | 0 | prover | 131,072 | 301 | 40,501,248 | 
| leaf | 18 | MerkleVerifyAir | 0 | prover | 65,536 | 37 | 3,997,696 | 
| leaf | 19 | ProofShapeAir<4, 8> | 0 | prover | 64 | 43 | 22,464 | 
| leaf | 2 | UnsetPvsAir | 0 | prover | 1 | 2 | 6 | 
| leaf | 20 | PublicValuesAir | 0 | prover | 64 | 8 | 1,536 | 
| leaf | 21 | RangeCheckerAir<8> | 0 | prover | 256 | 2 | 1,536 | 
| leaf | 22 | GkrInputAir | 0 | prover | 2 | 26 | 204 | 
| leaf | 23 | GkrLayerAir | 0 | prover | 64 | 46 | 10,624 | 
| leaf | 24 | GkrLayerSumcheckAir | 0 | prover | 1,024 | 45 | 132,096 | 
| leaf | 25 | GkrXiSamplerAir | 0 | prover | 2 | 10 | 76 | 
| leaf | 26 | OpeningClaimsAir | 0 | prover | 8,192 | 63 | 1,236,992 | 
| leaf | 27 | UnivariateRoundAir | 0 | prover | 64 | 27 | 5,056 | 
| leaf | 28 | SumcheckRoundsAir | 0 | prover | 64 | 57 | 9,024 | 
| leaf | 29 | StackingClaimsAir | 0 | prover | 4,096 | 35 | 421,888 | 
| leaf | 3 | SymbolicExpressionAir<BabyBearParameters> | 0 | prover | 65,536 | 60 | 17,563,648 | 
| leaf | 30 | EqBaseAir | 0 | prover | 16 | 51 | 1,328 | 
| leaf | 31 | EqBitsAir | 0 | prover | 8,192 | 16 | 294,912 | 
| leaf | 32 | WhirRoundAir | 0 | prover | 8 | 46 | 1,360 | 
| leaf | 33 | SumcheckAir | 0 | prover | 32 | 38 | 3,648 | 
| leaf | 34 | WhirQueryAir | 0 | prover | 1,024 | 32 | 53,248 | 
| leaf | 35 | InitialOpenedValuesAir | 0 | prover | 65,536 | 89 | 9,240,576 | 
| leaf | 36 | NonInitialOpenedValuesAir | 0 | prover | 8,192 | 28 | 360,448 | 
| leaf | 37 | WhirFoldingAir | 0 | prover | 16,384 | 31 | 770,048 | 
| leaf | 38 | FinalPolyMleEvalAir | 0 | prover | 2,048 | 34 | 176,128 | 
| leaf | 39 | FinalPolyQueryEvalAir | 0 | prover | 524,288 | 45 | 34,078,720 | 
| leaf | 4 | FractionsFolderAir | 0 | prover | 64 | 29 | 6,208 | 
| leaf | 40 | PowerCheckerAir<2, 32> | 0 | prover | 32 | 4 | 384 | 
| leaf | 41 | ExpBitsLenAir | 0 | prover | 32,768 | 16 | 786,432 | 
| leaf | 5 | UnivariateSumcheckAir | 0 | prover | 256 | 24 | 20,480 | 
| leaf | 6 | MultilinearSumcheckAir | 0 | prover | 256 | 33 | 22,784 | 
| leaf | 7 | EqNsAir | 0 | prover | 64 | 41 | 5,184 | 
| leaf | 8 | Eq3bAir | 0 | prover | 32,768 | 25 | 1,212,416 | 
| leaf | 9 | EqSharpUniAir | 0 | prover | 32 | 17 | 1,184 | 

| group | air_id | air_name | phase | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| root | 0 | RootVerifierPvsAir | prover | 1 | 206 | 638 | 
| root | 1 | UserPvsCommitAir | prover | 8 | 30 | 400 | 
| root | 10 | EqSharpUniReceiverAir | prover | 4 | 17 | 116 | 
| root | 11 | EqUniAir | prover | 4 | 16 | 112 | 
| root | 12 | ExpressionClaimAir | prover | 128 | 32 | 7,680 | 
| root | 13 | InteractionsFoldingAir | prover | 8,192 | 37 | 729,088 | 
| root | 14 | ConstraintsFoldingAir | prover | 4,096 | 25 | 266,240 | 
| root | 15 | EqNegAir | prover | 8 | 40 | 576 | 
| root | 16 | TranscriptAir | prover | 4,096 | 44 | 458,752 | 
| root | 17 | Poseidon2Air<BabyBearParameters>, 1> | prover | 16,384 | 301 | 5,062,656 | 
| root | 18 | MerkleVerifyAir | prover | 8,192 | 37 | 499,712 | 
| root | 19 | ProofShapeAir<4, 8> | prover | 64 | 44 | 22,528 | 
| root | 2 | UserPvsInMemoryAir | prover | 32 | 20 | 1,024 | 
| root | 20 | PublicValuesAir | prover | 128 | 8 | 3,072 | 
| root | 21 | RangeCheckerAir<8> | prover | 256 | 2 | 1,536 | 
| root | 22 | GkrInputAir | prover | 1 | 26 | 102 | 
| root | 23 | GkrLayerAir | prover | 32 | 46 | 5,312 | 
| root | 24 | GkrLayerSumcheckAir | prover | 256 | 45 | 33,024 | 
| root | 25 | GkrXiSamplerAir | prover | 1 | 10 | 38 | 
| root | 26 | OpeningClaimsAir | prover | 2,048 | 63 | 309,248 | 
| root | 27 | UnivariateRoundAir | prover | 8 | 27 | 632 | 
| root | 28 | SumcheckRoundsAir | prover | 32 | 57 | 4,512 | 
| root | 29 | StackingClaimsAir | prover | 512 | 35 | 52,736 | 
| root | 3 | SymbolicExpressionAir<BabyBearParameters> | prover | 32,768 | 316 | 12,058,624 | 
| root | 30 | EqBaseAir | prover | 4 | 51 | 332 | 
| root | 31 | EqBitsAir | prover | 4,096 | 16 | 147,456 | 
| root | 32 | WhirRoundAir | prover | 4 | 46 | 680 | 
| root | 33 | SumcheckAir | prover | 16 | 38 | 1,824 | 
| root | 34 | WhirQueryAir | prover | 128 | 32 | 6,656 | 
| root | 35 | InitialOpenedValuesAir | prover | 8,192 | 89 | 1,155,072 | 
| root | 36 | NonInitialOpenedValuesAir | prover | 1,024 | 28 | 45,056 | 
| root | 37 | WhirFoldingAir | prover | 2,048 | 31 | 96,256 | 
| root | 38 | FinalPolyMleEvalAir | prover | 256 | 34 | 22,016 | 
| root | 39 | FinalPolyQueryEvalAir | prover | 16,384 | 45 | 1,064,960 | 
| root | 4 | FractionsFolderAir | prover | 64 | 29 | 6,208 | 
| root | 40 | PowerCheckerAir<2, 32> | prover | 32 | 4 | 384 | 
| root | 41 | ExpBitsLenAir | prover | 8,192 | 16 | 196,608 | 
| root | 5 | UnivariateSumcheckAir | prover | 16 | 24 | 1,280 | 
| root | 6 | MultilinearSumcheckAir | prover | 128 | 33 | 11,392 | 
| root | 7 | EqNsAir | prover | 32 | 41 | 2,592 | 
| root | 8 | Eq3bAir | prover | 16,384 | 25 | 606,208 | 
| root | 9 | EqSharpUniAir | prover | 4 | 17 | 148 | 

| group | air_id | air_name | phase | segment | rows | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | ProgramAir | prover | 0 | 131,072 | 10 | 1,835,008 | 
| app_proof | 1 | VmConnectorAir | prover | 0 | 2 | 6 | 52 | 
| app_proof | 10 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | prover | 0 | 256 | 39 | 34,560 | 
| app_proof | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 0 | 32,768 | 31 | 3,506,176 | 
| app_proof | 12 | RangeTupleCheckerAir<2> | prover | 0 | 524,288 | 3 | 3,670,016 | 
| app_proof | 14 | Rv32HintStoreAir | prover | 0 | 16,384 | 32 | 1,703,936 | 
| app_proof | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 0 | 32,768 | 20 | 2,228,224 | 
| app_proof | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 0 | 131,072 | 28 | 12,058,624 | 
| app_proof | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 0 | 65,536 | 18 | 3,801,088 | 
| app_proof | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 0 | 131,072 | 32 | 11,010,048 | 
| app_proof | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 0 | 262,144 | 26 | 18,350,080 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 0 | 131,072 | 20 | 4,194,304 | 
| app_proof | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | prover | 0 | 1,024 | 36 | 110,592 | 
| app_proof | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 0 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 0 | 131,072 | 53 | 19,529,728 | 
| app_proof | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 0 | 16,384 | 37 | 1,785,856 | 
| app_proof | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 0 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 25 | BitwiseOperationLookupAir<8> | prover | 0 | 65,536 | 18 | 1,703,936 | 
| app_proof | 26 | PhantomAir | prover | 0 | 1 | 6 | 18 | 
| app_proof | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 0 | 16,384 | 300 | 4,980,736 | 
| app_proof | 28 | VariableRangeCheckerAir | prover | 0 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 0 | 131,072 | 32 | 6,291,456 | 
| app_proof | 6 | AccessAdapterAir<8> | prover | 0 | 131,072 | 17 | 4,849,664 | 
| app_proof | 9 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | prover | 0 | 256 | 59 | 40,704 | 
| app_proof | 0 | ProgramAir | prover | 1 | 131,072 | 10 | 1,835,008 | 
| app_proof | 1 | VmConnectorAir | prover | 1 | 2 | 6 | 52 | 
| app_proof | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | prover | 1 | 32,768 | 31 | 3,506,176 | 
| app_proof | 12 | RangeTupleCheckerAir<2> | prover | 1 | 524,288 | 3 | 3,670,016 | 
| app_proof | 13 | KeccakVmAir | prover | 1 | 32 | 3,163 | 142,304 | 
| app_proof | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | prover | 1 | 32,768 | 20 | 2,228,224 | 
| app_proof | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | prover | 1 | 65,536 | 28 | 6,029,312 | 
| app_proof | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | prover | 1 | 65,536 | 18 | 3,801,088 | 
| app_proof | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | prover | 1 | 131,072 | 32 | 11,010,048 | 
| app_proof | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | prover | 1 | 131,072 | 26 | 9,175,040 | 
| app_proof | 2 | PersistentBoundaryAir<8> | prover | 1 | 2,048 | 20 | 65,536 | 
| app_proof | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | prover | 1 | 32 | 36 | 3,456 | 
| app_proof | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | prover | 1 | 1,048,576 | 41 | 114,294,784 | 
| app_proof | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | prover | 1 | 131,072 | 53 | 19,529,728 | 
| app_proof | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | prover | 1 | 16,384 | 37 | 1,785,856 | 
| app_proof | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | prover | 1 | 1,048,576 | 36 | 121,634,816 | 
| app_proof | 25 | BitwiseOperationLookupAir<8> | prover | 1 | 65,536 | 18 | 1,703,936 | 
| app_proof | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | prover | 1 | 2,048 | 300 | 622,592 | 
| app_proof | 28 | VariableRangeCheckerAir | prover | 1 | 262,144 | 4 | 2,097,152 | 
| app_proof | 3 | MemoryMerkleAir<8> | prover | 1 | 2,048 | 32 | 98,304 | 
| app_proof | 6 | AccessAdapterAir<8> | prover | 1 | 2,048 | 17 | 75,776 | 

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
| agg_keygen | GkrInputAir | 19 | 15 | 4 | 
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

| group | idx | trace_gen_time_ms | total_proof_time_ms | subcircuit_generate_proving_ctxs_time_ms | generate_proving_ctxs_time_ms | generate_blob_time_ms | fill_valid_rows_time_ms | fill_padding_rows_time_ms | execute_preflight_time_ms | compute_merkle_precomputation_time_ms | apply_merkle_precomputation_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | 36 | 2,918 | 35 | 21 | 0 | 0 | 0 | 13 | 11 | 11 | 
| internal_recursive.0 | 1 | 23 | 2,094 | 23 | 13 | 0 | 0 | 0 | 9 | 7 | 7 | 
| internal_recursive.1 | 1 | 19 | 1,835 | 19 | 11 | 0 | 0 | 0 | 7 | 4 | 4 | 
| leaf | 0 | 110 | 4,633 | 109 | 80 | 2 | 0 | 0 | 28 | 23 | 23 | 

| group | idx | phase | whir_w_evals_accum_time_ms | whir_sumcheck_time_ms | whir_mle_conversion_time_ms | whir_f_evals_batch_time_ms | whir_dft_merkle_time_ms | transpose_to_rm_time_ms | trace_commit_cpu_time_ms | total_cells | stark_prove_excluding_trace_time_ms | stacked_round_eval_time_ms | stacked_round0_time_ms | stacked_matrix_time_ms | stacked_fold_ple_time_ms | stacked_fold_mle_time_ms | rs_encode_and_merkle_cpu_time_ms | row_hash_time_ms | prover.main_trace_commit_time_ms | prover.batch_constraints.mle_rounds_time_ms | prove_zerocheck_and_logup_time_ms | prove_whir_opening_cpu_time_ms | prove_stacked_opening_reduction_time_ms | fractional_sumcheck_time_ms | eval_to_coeff_phase_time_ms | digest_layers_time_ms | dft_batch_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 2 | 20 | 18 | 2 | 30 | 45 | 304 | 41,110,275 | 2,882 | 0 | 25 | 10 | 55 | 0 | 277 | 117 | 304 | 893 | 2,069 | 340 | 167 | 524 | 17 | 25 | 70 | 
| internal_recursive.0 | 1 | prover | 5 | 42 | 11 | 1 | 31 | 29 | 259 | 23,651,975 | 2,070 | 0 | 15 | 9 | 46 | 0 | 226 | 68 | 259 | 466 | 1,244 | 454 | 111 | 360 | 12 | 25 | 90 | 
| internal_recursive.1 | 1 | prover | 3 | 52 | 8 | 1 | 34 | 17 | 195 | 17,507,975 | 1,816 | 0 | 16 | 5 | 25 | 0 | 175 | 57 | 196 | 447 | 1,218 | 318 | 82 | 370 | 7 | 30 | 62 | 
| leaf | 0 | prover | 11 | 1 | 87 | 6 | 54 | 63 | 537 | 113,138,688 | 4,523 | 0 | 78 | 29 | 47 | 0 | 468 | 171 | 537 | 613 | 3,304 | 490 | 189 | 1,508 | 54 | 59 | 119 | 

| group | idx | phase | round | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 11 | 
| internal_for_leaf | 0 | prover | 1 | 7 | 
| internal_recursive.0 | 1 | prover | 0 | 11 | 
| internal_recursive.0 | 1 | prover | 1 | 5 | 
| internal_recursive.1 | 1 | prover | 0 | 10 | 
| internal_recursive.1 | 1 | prover | 1 | 6 | 
| leaf | 0 | prover | 0 | 21 | 
| leaf | 0 | prover | 1 | 10 | 

| group | idx | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| internal_for_leaf | 0 | prover | 0 | 3,455,234 | 2,013,265,921 | 
| internal_recursive.0 | 1 | prover | 0 | 2,068,318 | 2,013,265,921 | 
| internal_recursive.1 | 1 | prover | 0 | 1,939,294 | 2,013,265,921 | 
| leaf | 0 | prover | 0 | 8,492,611 | 2,013,265,921 | 

| group | phase | whir_w_evals_accum_time_ms | whir_sumcheck_time_ms | whir_mle_conversion_time_ms | whir_f_evals_batch_time_ms | whir_dft_merkle_time_ms | transpose_to_rm_time_ms | trace_commit_cpu_time_ms | total_cells | stark_prove_excluding_trace_time_ms | stacked_round_eval_time_ms | stacked_round0_time_ms | stacked_matrix_time_ms | stacked_fold_ple_time_ms | stacked_fold_mle_time_ms | rs_encode_and_merkle_cpu_time_ms | row_hash_time_ms | prover.main_trace_commit_time_ms | prover.batch_constraints.mle_rounds_time_ms | prove_zerocheck_and_logup_time_ms | prove_whir_opening_cpu_time_ms | prove_stacked_opening_reduction_time_ms | grind_pow_time_ms | fractional_sumcheck_time_ms | eval_to_coeff_phase_time_ms | digest_layers_time_ms | dft_batch_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| root | prover | 3 | 1,463 | 30 | 1 | 5,682 | 72 | 23,268 | 22,883,486 | 47,710 | 0 | 20 | 7 | 30 | 0 | 23,239 | 11,741 | 23,269 | 413 | 1,110 | 23,205 | 125 | 1,517 | 356 | 32 | 10,965 | 427 | 

| group | phase | round | merkle_tree_time_ms | grind_pow_time_ms |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 5,732 | 317 | 
| root | prover | 1 | 2,844 | 187 | 
| root | prover | 2 |  | 411 | 

| group | phase | round | segment | merkle_tree_time_ms |
| --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 13 | 
| app_proof | prover | 1 | 0 | 6 | 
| app_proof | prover | 0 | 1 | 10 | 
| app_proof | prover | 1 | 1 | 5 | 

| group | phase | segment | whir_w_evals_accum_time_ms | whir_sumcheck_time_ms | whir_mle_conversion_time_ms | whir_f_evals_batch_time_ms | whir_dft_merkle_time_ms | transpose_to_rm_time_ms | trace_commit_cpu_time_ms | total_cells | stark_prove_excluding_trace_time_ms | stacked_round_eval_time_ms | stacked_round0_time_ms | stacked_matrix_time_ms | stacked_fold_ple_time_ms | stacked_fold_mle_time_ms | rs_encode_and_merkle_cpu_time_ms | row_hash_time_ms | prover.main_trace_commit_time_ms | prover.batch_constraints.mle_rounds_time_ms | prove_zerocheck_and_logup_time_ms | prove_whir_opening_cpu_time_ms | prove_stacked_opening_reduction_time_ms | fractional_sumcheck_time_ms | eval_to_coeff_phase_time_ms | digest_layers_time_ms | dft_batch_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 11 | 1 | 111 | 12 | 36 | 52 | 542 | 339,711,558 | 10,476 | 0 | 128 | 71 | 68 | 0 | 403 | 149 | 542 | 882 | 9,169 | 492 | 271 | 7,069 | 72 | 33 | 95 | 
| app_proof | prover | 1 | 11 | 1 | 84 | 7 | 34 | 43 | 433 | 303,309,204 | 9,878 | 0 | 101 | 39 | 49 | 0 | 332 | 117 | 433 | 739 | 8,745 | 467 | 231 | 6,889 | 58 | 37 | 75 | 

| group | phase | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- | --- |
| app_proof | prover | 0 | 0 | 53,557,517 | 2,013,265,921 | 
| app_proof | prover | 1 | 0 | 49,189,482 | 2,013,265,921 | 

| group | phase | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| root | prover | 0 | 1,087,470 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | system_trace_gen_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_cells_used | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_proof | 0 | 450 | 11,484 | 90,181,874 | 450 | 7 | 8 | 90,181,874 | 98 | 2,208,000 | 36.99 | 
| app_proof | 1 | 270 | 10,544 | 73,738,845 | 270 | 8 | 1 | 73,738,845 | 70 | 1,928,694 | 37.67 | 

</details>



Commit: https://github.com/openvm-org/openvm/commit/8f2f0fcadbe050a9d5e514e6bd95f125d4abddd2

Max Segment Length: 1048576

Instance Type: m7a.8xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23952805545)
