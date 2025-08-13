| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  215.22 |  215.22 |
| kitchen_sink |  14.77 |  14.77 |
| leaf |  22.28 |  22.28 |
| internal.0 |  5.95 |  5.95 |
| root |  38.45 |  38.45 |
| halo2_outer |  89.59 |  89.59 |
| halo2_wrapper |  44.18 |  44.18 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  14,765 |  14,765 |  14,765 |  14,765 |
| `main_cells_used     ` |  904,738,232 |  904,738,232 |  904,738,232 |  904,738,232 |
| `total_cells_used    ` |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |
| `insns               ` |  153,644 |  307,288 |  153,644 |  153,644 |
| `execute_metered_time_ms` |  10 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  14.58 | -          |  14.58 |  14.58 |
| `execute_preflight_time_ms` |  103 |  103 |  103 |  103 |
| `execute_preflight_insn_mi/s` |  7.36 | -          |  7.36 |  7.36 |
| `trace_gen_time_ms   ` |  210 |  210 |  210 |  210 |
| `memory_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `stark_prove_excluding_trace_time_ms` |  14,204 |  14,204 |  14,204 |  14,204 |
| `main_trace_commit_time_ms` |  4,488 |  4,488 |  4,488 |  4,488 |
| `generate_perm_trace_time_ms` |  547 |  547 |  547 |  547 |
| `perm_trace_commit_time_ms` |  1,361 |  1,361 |  1,361 |  1,361 |
| `quotient_poly_compute_time_ms` |  6,282 |  6,282 |  6,282 |  6,282 |
| `quotient_poly_commit_time_ms` |  282 |  282 |  282 |  282 |
| `pcs_opening_time_ms ` |  1,200 |  1,200 |  1,200 |  1,200 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  22,276 |  22,276 |  22,276 |  22,276 |
| `main_cells_used     ` |  769,363,970 |  769,363,970 |  769,363,970 |  769,363,970 |
| `total_cells_used    ` |  2,050,270,596 |  2,050,270,596 |  2,050,270,596 |  2,050,270,596 |
| `insns               ` |  7,904,052 |  7,904,052 |  7,904,052 |  7,904,052 |
| `execute_preflight_time_ms` |  1,311 |  1,311 |  1,311 |  1,311 |
| `execute_preflight_insn_mi/s` |  6.96 | -          |  6.96 |  6.96 |
| `trace_gen_time_ms   ` |  1,638 |  1,638 |  1,638 |  1,638 |
| `memory_finalize_time_ms` |  17 |  17 |  17 |  17 |
| `stark_prove_excluding_trace_time_ms` |  18,247 |  18,247 |  18,247 |  18,247 |
| `main_trace_commit_time_ms` |  3,757 |  3,757 |  3,757 |  3,757 |
| `generate_perm_trace_time_ms` |  1,588 |  1,588 |  1,588 |  1,588 |
| `perm_trace_commit_time_ms` |  4,823 |  4,823 |  4,823 |  4,823 |
| `quotient_poly_compute_time_ms` |  2,150 |  2,150 |  2,150 |  2,150 |
| `quotient_poly_commit_time_ms` |  1,306 |  1,306 |  1,306 |  1,306 |
| `pcs_opening_time_ms ` |  4,619 |  4,619 |  4,619 |  4,619 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,950 |  5,950 |  5,950 |  5,950 |
| `main_cells_used     ` |  79,527,135 |  79,527,135 |  79,527,135 |  79,527,135 |
| `total_cells_used    ` |  143,464,085 |  143,464,085 |  143,464,085 |  143,464,085 |
| `insns               ` |  1,197,757 |  1,197,757 |  1,197,757 |  1,197,757 |
| `execute_preflight_time_ms` |  483 |  483 |  483 |  483 |
| `execute_preflight_insn_mi/s` |  3.65 | -          |  3.65 |  3.65 |
| `trace_gen_time_ms   ` |  181 |  181 |  181 |  181 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `stark_prove_excluding_trace_time_ms` |  4,267 |  4,267 |  4,267 |  4,267 |
| `main_trace_commit_time_ms` |  907 |  907 |  907 |  907 |
| `generate_perm_trace_time_ms` |  194 |  194 |  194 |  194 |
| `perm_trace_commit_time_ms` |  649 |  649 |  649 |  649 |
| `quotient_poly_compute_time_ms` |  606 |  606 |  606 |  606 |
| `quotient_poly_commit_time_ms` |  672 |  672 |  672 |  672 |
| `pcs_opening_time_ms ` |  1,234 |  1,234 |  1,234 |  1,234 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,448 |  38,448 |  38,448 |  38,448 |
| `main_cells_used     ` |  41,517,894 |  41,517,894 |  41,517,894 |  41,517,894 |
| `total_cells_used    ` |  64,200,316 |  64,200,316 |  64,200,316 |  64,200,316 |
| `insns               ` |  772,547 |  772,547 |  772,547 |  772,547 |
| `execute_preflight_time_ms` |  158 |  158 |  158 |  158 |
| `execute_preflight_insn_mi/s` |  5.37 | -          |  5.37 |  5.37 |
| `trace_gen_time_ms   ` |  115 |  115 |  115 |  115 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  38,175 |  38,175 |  38,175 |  38,175 |
| `main_trace_commit_time_ms` |  12,307 |  12,307 |  12,307 |  12,307 |
| `generate_perm_trace_time_ms` |  91 |  91 |  91 |  91 |
| `perm_trace_commit_time_ms` |  7,578 |  7,578 |  7,578 |  7,578 |
| `quotient_poly_compute_time_ms` |  723 |  723 |  723 |  723 |
| `quotient_poly_commit_time_ms` |  13,594 |  13,594 |  13,594 |  13,594 |
| `pcs_opening_time_ms ` |  3,871 |  3,871 |  3,871 |  3,871 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  89,589 |  89,589 |  89,589 |  89,589 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,181 |  44,181 |  44,181 |  44,181 |



<details>
<summary>Detailed Metrics</summary>

|  | vm.create_initial_state_time_ms | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_preflight_time_ms | execute_preflight_insn_mi/s | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 0 | 115 | 64,200,316 | 115 | 0 | 89,603 | 44,181 | 8 | 41,517,894 | 772,547 | 299 | 5.37 | 14,815 | 39,480 | 

| group | vm.reset_state_time_ms | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer |  | 89,589 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper |  | 44,181 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  |  | 5,951 |  | 3 |  |  |  | 2 |  |  |  | 
| kitchen_sink | 0 |  |  |  | 14,765 |  | 6 |  | 153,644 | 1 | 10 | 14.58 | 37 | 
| leaf |  |  | 22,283 |  |  | 1 |  |  |  | 1 |  |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| leaf | AccessAdapterAir<2> | 0 | 4,194,304 |  | 16 | 11 | 113,246,208 | 
| leaf | AccessAdapterAir<4> | 0 | 2,097,152 |  | 16 | 13 | 60,817,408 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | FriReducedOpeningAir | 0 | 8,388,608 |  | 84 | 27 | 931,135,488 | 
| leaf | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 1,048,576 |  | 312 | 398 | 744,488,960 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 36 | 29 | 272,629,760 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 40 | 21 | 127,926,272 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 40 | 27 | 35,127,296 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 1,048,576 |  | 36 | 38 | 77,594,624 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 8 | 13 | 2,752,512 | 
| root | AccessAdapterAir<8> | 0 | 4,096 |  | 8 | 17 | 102,400 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| root | JalRangeCheckAir | 0 | 32,768 |  | 12 | 12 | 786,432 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 84 | 398 | 15,794,176 | 
| root | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 524,288 |  | 12 | 29 | 21,495,808 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 12 | 23 | 4,587,520 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 12 | 22 | 2,176 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 16 | 21 | 9,699,328 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 16 | 27 | 2,818,048 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 12 | 38 | 3,276,800 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 5 | 26 | 
| root | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 12 | 2,621,440 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| kitchen_sink | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| kitchen_sink | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| kitchen_sink | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | KeccakVmAir | 0 | 262,144 |  | 1,056 | 3,163 | 1,105,985,536 | 
| kitchen_sink | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| kitchen_sink | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| kitchen_sink | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| kitchen_sink | RangeTupleCheckerAir<2> | 0 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | Sha256VmAir | 0 | 524,288 |  | 108 | 470 | 303,038,464 | 
| kitchen_sink | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 32,768 |  | 52 | 36 | 2,883,584 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,192 |  | 28 | 26 | 442,368 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| kitchen_sink | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,024 |  | 28 | 18 | 47,104 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 2,048 |  | 192 | 168 | 737,280 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 1,024 |  | 68 | 169 | 242,688 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 256 |  | 192 | 164 | 91,136 | 
| kitchen_sink | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 256 |  | 48 | 124 | 44,032 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 8 |  | 88 | 242 | 2,640 | 
| kitchen_sink | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,048 |  | 36 | 28 | 131,072 | 
| kitchen_sink | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 1,668 | 1,020 | 10,752 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 384 | 294 | 41,920 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 860 | 625 | 2,202 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 496 | 393 | 2,404 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 | 949 | 3,426 | 
| kitchen_sink | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | vm.reset_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 181 | 5,950 | 143,464,085 | 224,975,330 | 181 | 4,267 | 0 | 606 | 672 | 649 | 1,234 | 9 | 907 | 79,527,135 | 1,197,757 | 194 |  | 483 | 3.65 | 
| leaf | 0 | 0 | 1,638 | 22,276 | 2,050,270,596 | 2,500,267,498 | 1,638 | 18,247 | 0 | 2,150 | 1,306 | 4,823 | 4,619 | 17 | 3,757 | 769,363,970 | 7,904,052 | 1,588 |  | 1,311 | 6.96 | 
| root | 0 | 0 | 115 | 38,448 | 64,200,316 | 80,435,354 | 115 | 38,175 | 0 | 723 | 13,594 | 7,578 | 3,871 | 8 | 12,307 | 41,517,894 | 772,547 | 91 | 3 | 158 | 5.37 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 5,177,476 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 30,814,464 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 2,588,738 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 30,941,444 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 70,177,482 | 2,013,265,921 | 
| leaf | 0 | 0 | 39,125,124 | 2,013,265,921 | 
| leaf | 0 | 1 | 291,111,168 | 2,013,265,921 | 
| leaf | 0 | 2 | 19,562,562 | 2,013,265,921 | 
| leaf | 0 | 3 | 288,096,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 642,351,818 | 2,013,265,921 | 
| root | 0 | 0 | 2,572,420 | 2,013,265,921 | 
| root | 0 | 1 | 12,005,632 | 2,013,265,921 | 
| root | 0 | 2 | 1,286,210 | 2,013,265,921 | 
| root | 0 | 3 | 12,067,076 | 2,013,265,921 | 
| root | 0 | 4 | 65,536 | 2,013,265,921 | 
| root | 0 | 5 | 28,390,090 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 210 | 14,765 | 1,226,975,002 | 1,481,193,346 | 210 | 14,204 | 3 | 6,282 | 282 | 1,361 | 1,200 | 6 | 2 | 4,488 | 904,738,232 | 153,644 | 547 | 103 | 7.36 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,977,944 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 32,428,632 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 988,972 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 32,011,136 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 57,344 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 24,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 49,612,036 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 1,048,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,320 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 120,668,384 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/45182e505a44bdcbd6f9fb46180ecd108c3fac7d

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16946927424)
