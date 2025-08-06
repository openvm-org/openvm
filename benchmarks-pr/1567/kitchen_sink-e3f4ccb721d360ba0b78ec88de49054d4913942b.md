| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  216.84 |  216.84 |
| kitchen_sink |  14.40 |  14.40 |
| leaf |  21.73 |  21.73 |
| internal.0 |  4.87 |  4.87 |
| root |  38.53 |  38.53 |
| halo2_outer |  92.60 |  92.60 |
| halo2_wrapper |  44.70 |  44.70 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  14,404 |  14,404 |  14,404 |  14,404 |
| `main_cells_used     ` |  904,774,168 |  904,774,168 |  904,774,168 |  904,774,168 |
| `total_cells_used    ` |  1,227,056,610 |  1,227,056,610 |  1,227,056,610 |  1,227,056,610 |
| `insns               ` |  154,596 |  309,192 |  154,596 |  154,596 |
| `execute_metered_time_ms` |  10 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  14.70 | -          |  14.70 |  14.70 |
| `execute_e3_time_ms  ` |  33 |  33 |  33 |  33 |
| `execute_e3_insn_mi/s` |  4.58 | -          |  4.58 |  4.58 |
| `trace_gen_time_ms   ` |  211 |  211 |  211 |  211 |
| `memory_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  62 |  62 |  62 |  62 |
| `stark_prove_excluding_trace_time_ms` |  14,160 |  14,160 |  14,160 |  14,160 |
| `main_trace_commit_time_ms` |  4,486 |  4,486 |  4,486 |  4,486 |
| `generate_perm_trace_time_ms` |  461 |  461 |  461 |  461 |
| `perm_trace_commit_time_ms` |  1,359 |  1,359 |  1,359 |  1,359 |
| `quotient_poly_compute_time_ms` |  6,291 |  6,291 |  6,291 |  6,291 |
| `quotient_poly_commit_time_ms` |  289 |  289 |  289 |  289 |
| `pcs_opening_time_ms ` |  1,227 |  1,227 |  1,227 |  1,227 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  21,729 |  21,729 |  21,729 |  21,729 |
| `main_cells_used     ` |  772,081,628 |  772,081,628 |  772,081,628 |  772,081,628 |
| `total_cells_used    ` |  2,057,383,386 |  2,057,383,386 |  2,057,383,386 |  2,057,383,386 |
| `insns               ` |  7,952,490 |  7,952,490 |  7,952,490 |  7,952,490 |
| `execute_e3_time_ms  ` |  1,578 |  1,578 |  1,578 |  1,578 |
| `execute_e3_insn_mi/s` |  5.04 | -          |  5.04 |  5.04 |
| `trace_gen_time_ms   ` |  1,691 |  1,691 |  1,691 |  1,691 |
| `memory_finalize_time_ms` |  18 |  18 |  18 |  18 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  18,460 |  18,460 |  18,460 |  18,460 |
| `main_trace_commit_time_ms` |  3,757 |  3,757 |  3,757 |  3,757 |
| `generate_perm_trace_time_ms` |  1,712 |  1,712 |  1,712 |  1,712 |
| `perm_trace_commit_time_ms` |  4,818 |  4,818 |  4,818 |  4,818 |
| `quotient_poly_compute_time_ms` |  2,192 |  2,192 |  2,192 |  2,192 |
| `quotient_poly_commit_time_ms` |  1,313 |  1,313 |  1,313 |  1,313 |
| `pcs_opening_time_ms ` |  4,663 |  4,663 |  4,663 |  4,663 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,866 |  4,866 |  4,866 |  4,866 |
| `main_cells_used     ` |  79,527,363 |  79,527,363 |  79,527,363 |  79,527,363 |
| `total_cells_used    ` |  143,464,617 |  143,464,617 |  143,464,617 |  143,464,617 |
| `insns               ` |  1,197,776 |  1,197,776 |  1,197,776 |  1,197,776 |
| `execute_e3_time_ms  ` |  377 |  377 |  377 |  377 |
| `execute_e3_insn_mi/s` |  3.17 | -          |  3.17 |  3.17 |
| `trace_gen_time_ms   ` |  179 |  179 |  179 |  179 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  4,310 |  4,310 |  4,310 |  4,310 |
| `main_trace_commit_time_ms` |  924 |  924 |  924 |  924 |
| `generate_perm_trace_time_ms` |  196 |  196 |  196 |  196 |
| `perm_trace_commit_time_ms` |  655 |  655 |  655 |  655 |
| `quotient_poly_compute_time_ms` |  605 |  605 |  605 |  605 |
| `quotient_poly_commit_time_ms` |  683 |  683 |  683 |  683 |
| `pcs_opening_time_ms ` |  1,242 |  1,242 |  1,242 |  1,242 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,530 |  38,530 |  38,530 |  38,530 |
| `main_cells_used     ` |  41,516,646 |  41,516,646 |  41,516,646 |  41,516,646 |
| `total_cells_used    ` |  64,197,820 |  64,197,820 |  64,197,820 |  64,197,820 |
| `insns               ` |  772,443 |  772,443 |  772,443 |  772,443 |
| `execute_e3_time_ms  ` |  171 |  171 |  171 |  171 |
| `execute_e3_insn_mi/s` |  4.50 | -          |  4.50 |  4.50 |
| `trace_gen_time_ms   ` |  111 |  111 |  111 |  111 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  38,248 |  38,248 |  38,248 |  38,248 |
| `main_trace_commit_time_ms` |  12,311 |  12,311 |  12,311 |  12,311 |
| `generate_perm_trace_time_ms` |  88 |  88 |  88 |  88 |
| `perm_trace_commit_time_ms` |  7,579 |  7,579 |  7,579 |  7,579 |
| `quotient_poly_compute_time_ms` |  736 |  736 |  736 |  736 |
| `quotient_poly_commit_time_ms` |  13,634 |  13,634 |  13,634 |  13,634 |
| `pcs_opening_time_ms ` |  3,894 |  3,894 |  3,894 |  3,894 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  92,603 |  92,603 |  92,603 |  92,603 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,695 |  44,695 |  44,695 |  44,695 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 111 | 64,197,820 | 111 | 2 | 92,616 | 44,695 | 7 | 41,516,646 | 772,443 | 170 | 4.54 | 0 | 14,786 | 39,560 | 

| group | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer | 92,603 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper | 44,695 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  | 6,045 |  | 3 |  |  |  | 2 |  |  |  | 
| kitchen_sink |  |  |  | 14,733 |  | 6 |  | 154,596 | 1 | 10 | 14.70 | 37 | 
| leaf |  | 22,996 |  |  | 1 |  |  |  | 1 |  |  |  | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 179 | 4,866 | 143,464,617 | 224,975,330 | 179 | 4,310 | 2 | 605 | 683 | 655 | 1,242 | 8 | 924 | 79,527,363 | 1,197,776 | 196 |  | 377 | 3.17 | 0 | 
| leaf | 0 | 1,691 | 21,729 | 2,057,383,386 | 2,500,267,498 | 1,691 | 18,460 | 2 | 2,192 | 1,313 | 4,818 | 4,663 | 18 | 3,757 | 772,081,628 | 7,952,490 | 1,712 |  | 1,578 | 5.04 | 0 | 
| root | 0 | 111 | 38,530 | 64,197,820 | 80,435,354 | 111 | 38,248 | 2 | 736 | 13,634 | 7,579 | 3,894 | 7 | 12,311 | 41,516,646 | 772,443 | 88 | 3 | 171 | 4.50 | 0 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 211 | 14,404 | 1,227,056,610 | 1,481,195,122 | 211 | 14,160 | 2 | 6,291 | 289 | 1,359 | 1,227 | 62 | 6 | 2 | 4,486 | 904,774,168 | 154,596 | 461 | 33 | 4.58 | 0 | 

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

</details>


Commit: https://github.com/openvm-org/openvm/commit/e3f4ccb721d360ba0b78ec88de49054d4913942b

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16786723247)
