| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  133.20 |  133.20 |
| halo2_outer |  88.83 |  88.83 |
| halo2_wrapper |  44.37 |  44.37 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `main_cells_used     ` |  899,579,940 |  899,579,940 |  899,579,940 |  899,579,940 |
| `total_cells_used    ` |  1,217,690,534 |  1,217,690,534 |  1,217,690,534 |  1,217,690,534 |
| `insns               ` |  154,596 |  309,192 |  154,596 |  154,596 |
| `execute_metered_time_ms` |  11 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  13.98 | -          |  13.98 |  13.98 |
| `execute_e3_time_ms  ` |  36 |  36 |  36 |  36 |
| `execute_e3_insn_mi/s` |  4.20 | -          |  4.20 |  4.20 |
| `memory_finalize_time_ms` |  3 |  3 |  3 |  3 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  79 |  79 |  79 |  79 |
| `stark_prove_excluding_trace_time_ms` |  14,208 |  14,208 |  14,208 |  14,208 |
| `main_trace_commit_time_ms` |  4,548 |  4,548 |  4,548 |  4,548 |
| `generate_perm_trace_time_ms` |  460 |  460 |  460 |  460 |
| `perm_trace_commit_time_ms` |  1,380 |  1,380 |  1,380 |  1,380 |
| `quotient_poly_compute_time_ms` |  6,260 |  6,260 |  6,260 |  6,260 |
| `quotient_poly_commit_time_ms` |  290 |  290 |  290 |  290 |
| `pcs_opening_time_ms ` |  1,223 |  1,223 |  1,223 |  1,223 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `main_cells_used     ` |  748,994,982 |  748,994,982 |  748,994,982 |  748,994,982 |
| `total_cells_used    ` |  2,003,008,544 |  2,003,008,544 |  2,003,008,544 |  2,003,008,544 |
| `insns               ` |  7,952,555 |  7,952,555 |  7,952,555 |  7,952,555 |
| `execute_e3_time_ms  ` |  2,436 |  2,436 |  2,436 |  2,436 |
| `execute_e3_insn_mi/s` |  3.26 | -          |  3.26 |  3.26 |
| `memory_finalize_time_ms` |  16 |  16 |  16 |  16 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  18,419 |  18,419 |  18,419 |  18,419 |
| `main_trace_commit_time_ms` |  3,812 |  3,812 |  3,812 |  3,812 |
| `generate_perm_trace_time_ms` |  1,650 |  1,650 |  1,650 |  1,650 |
| `perm_trace_commit_time_ms` |  4,885 |  4,885 |  4,885 |  4,885 |
| `quotient_poly_compute_time_ms` |  2,138 |  2,138 |  2,138 |  2,138 |
| `quotient_poly_commit_time_ms` |  1,329 |  1,329 |  1,329 |  1,329 |
| `pcs_opening_time_ms ` |  4,600 |  4,600 |  4,600 |  4,600 |

| internal_wrapper.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `main_cells_used     ` |  75,193,023 |  75,193,023 |  75,193,023 |  75,193,023 |
| `total_cells_used    ` |  134,680,325 |  134,680,325 |  134,680,325 |  134,680,325 |
| `insns               ` |  1,197,736 |  1,197,736 |  1,197,736 |  1,197,736 |
| `execute_e3_time_ms  ` |  495 |  495 |  495 |  495 |
| `execute_e3_insn_mi/s` |  2.42 | -          |  2.42 |  2.42 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  4,361 |  4,361 |  4,361 |  4,361 |
| `main_trace_commit_time_ms` |  919 |  919 |  919 |  919 |
| `generate_perm_trace_time_ms` |  196 |  196 |  196 |  196 |
| `perm_trace_commit_time_ms` |  667 |  667 |  667 |  667 |
| `quotient_poly_compute_time_ms` |  636 |  636 |  636 |  636 |
| `quotient_poly_commit_time_ms` |  690 |  690 |  690 |  690 |
| `pcs_opening_time_ms ` |  1,246 |  1,246 |  1,246 |  1,246 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `main_cells_used     ` |  40,016,494 |  40,016,494 |  40,016,494 |  40,016,494 |
| `total_cells_used    ` |  61,675,424 |  61,675,424 |  61,675,424 |  61,675,424 |
| `insns               ` |  772,536 |  772,536 |  772,536 |  772,536 |
| `execute_e3_time_ms  ` |  216 |  216 |  216 |  216 |
| `execute_e3_insn_mi/s` |  3.57 | -          |  3.57 |  3.57 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  38,405 |  38,405 |  38,405 |  38,405 |
| `main_trace_commit_time_ms` |  12,325 |  12,325 |  12,325 |  12,325 |
| `generate_perm_trace_time_ms` |  92 |  92 |  92 |  92 |
| `perm_trace_commit_time_ms` |  7,584 |  7,584 |  7,584 |  7,584 |
| `quotient_poly_compute_time_ms` |  710 |  710 |  710 |  710 |
| `quotient_poly_commit_time_ms` |  13,750 |  13,750 |  13,750 |  13,750 |
| `pcs_opening_time_ms ` |  3,929 |  3,929 |  3,929 |  3,929 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  88,831 |  88,831 |  88,831 |  88,831 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,371 |  44,371 |  44,371 |  44,371 |



<details>
<summary>Detailed Metrics</summary>

|  | wrapper_layer_time_ms | tracegen_time_ms | total_cells_used | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 6,246 | 116 | 112,193,444 | 88,844 | 44,371 | 7 | 75,218,490 | 1,198,208 | 215 | 3.59 | 0 | 15,992 | 39,794 | 

| group | tracegen_time_ms | total_proof_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | single_leaf_agg_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_children | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | compute_user_public_values_proof_time_ms | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer |  | 88,831 |  |  |  |  |  |  |  |  |  |  |  |  |  | 65,627,358 |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  | 44,371 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_wrapper.1 | 183 |  | 134,680,325 | 224,975,330 | 4,361 |  | 636 | 690 |  | 667 | 1,246 |  |  | 8 | 919 | 75,193,023 | 1,197,736 | 196 | 2 |  |  | 495 | 2.42 |  | 0 | 
| kitchen_sink |  |  |  |  |  |  |  |  | 15,912 |  |  |  | 24 |  |  |  | 154,596 |  | 1 | 11 | 13.98 |  |  | 55 |  | 
| leaf |  |  |  |  |  | 23,712 |  |  |  |  |  | 1 |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| internal_wrapper.1 | AccessAdapterAir<2> | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal_wrapper.1 | AccessAdapterAir<4> | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal_wrapper.1 | AccessAdapterAir<8> | 4,096 |  | 12 | 17 | 118,784 | 
| internal_wrapper.1 | FriReducedOpeningAir | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal_wrapper.1 | JalRangeCheckAir | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal_wrapper.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal_wrapper.1 | PhantomAir | 32,768 |  | 8 | 6 | 458,752 | 
| internal_wrapper.1 | ProgramAir | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal_wrapper.1 | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal_wrapper.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal_wrapper.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal_wrapper.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 64 |  | 16 | 23 | 2,496 | 
| internal_wrapper.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal_wrapper.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal_wrapper.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal_wrapper.1 | VmConnectorAir | 2 | 1 | 12 | 5 | 34 | 
| internal_wrapper.1 | VolatileBoundaryAir | 262,144 |  | 12 | 12 | 6,291,456 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
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
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 16 |  | 72 | 39 | 1,776 | 
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 1,668 | 1,020 | 10,752 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 384 | 294 | 41,920 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 860 | 625 | 2,202 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 496 | 393 | 2,404 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 | 949 | 3,426 | 
| kitchen_sink | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | tracegen_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,578 | 2,003,008,544 | 2,500,267,498 | 18,419 | 2,138 | 1,329 | 4,885 | 4,600 | 16 | 3,812 | 748,994,982 | 7,952,555 | 1,650 |  | 2,436 | 3.26 | 0 | 
| root | 0 | 119 | 61,675,424 | 80,435,354 | 38,405 | 710 | 13,750 | 7,584 | 3,929 | 7 | 12,325 | 40,016,494 | 772,536 | 92 | 3 | 216 | 3.57 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
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

| group | segment | tracegen_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 229 | 1,217,690,534 | 1,481,195,122 | 14,208 | 6,260 | 290 | 1,380 | 1,223 | 79 | 24 | 3 | 4,548 | 899,579,940 | 154,596 | 460 | 36 | 4.20 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,977,976 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 32,428,728 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 988,988 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 32,011,232 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 57,344 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 24,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 49,612,052 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 1,048,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,448 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 120,668,768 | 2,013,265,921 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| internal_wrapper.1 | 0 | 5,177,476 | 2,013,265,921 | 
| internal_wrapper.1 | 1 | 30,814,464 | 2,013,265,921 | 
| internal_wrapper.1 | 2 | 2,588,738 | 2,013,265,921 | 
| internal_wrapper.1 | 3 | 30,941,444 | 2,013,265,921 | 
| internal_wrapper.1 | 4 | 262,144 | 2,013,265,921 | 
| internal_wrapper.1 | 5 | 70,177,482 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/672c9efac76c533447d0b1c1abde30486ea31b59

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16476367982)
