| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  223.01 |  223.01 |
| kitchen_sink |  15.42 |  15.42 |
| leaf |  26.15 |  26.15 |
| internal_wrapper.1 |  5.81 |  5.81 |
| root |  38.76 |  38.76 |
| halo2_outer |  92.50 |  92.50 |
| halo2_wrapper |  44.21 |  44.21 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,423 |  15,423 |  15,423 |  15,423 |
| `main_cells_used     ` |  898,132,732 |  898,132,732 |  898,132,732 |  898,132,732 |
| `total_cycles        ` |  154,595 |  154,595 |  154,595 |  154,595 |
| `execute_metered_time_ms` |  145 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  5.61 | -          |  5.61 |  5.61 |
| `execute_e3_time_ms  ` |  68 |  68 |  68 |  68 |
| `execute_e3_insn_mi/s` |  2.27 | -          |  2.27 |  2.27 |
| `trace_gen_time_ms   ` |  485 |  485 |  485 |  485 |
| `memory_finalize_time_ms` |  85 |  85 |  85 |  85 |
| `boundary_finalize_time_ms` |  1 |  1 |  1 |  1 |
| `merkle_finalize_time_ms` |  80 |  80 |  80 |  80 |
| `stark_prove_excluding_trace_time_ms` |  14,870 |  14,870 |  14,870 |  14,870 |
| `main_trace_commit_time_ms` |  5,044 |  5,044 |  5,044 |  5,044 |
| `perm_trace_commit_time_ms` |  1,506 |  1,506 |  1,506 |  1,506 |
| `quotient_poly_compute_time_ms` |  6,269 |  6,269 |  6,269 |  6,269 |
| `quotient_poly_commit_time_ms` |  306 |  306 |  306 |  306 |
| `pcs_opening_time_ms ` |  1,234 |  1,234 |  1,234 |  1,234 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  26,154 |  26,154 |  26,154 |  26,154 |
| `main_cells_used     ` |  732,639,889 |  732,639,889 |  732,639,889 |  732,639,889 |
| `total_cycles        ` |  7,991,094 |  7,991,094 |  7,991,094 |  7,991,094 |
| `execute_metered_time_ms` |  2,123 |  2,123 |  2,123 |  2,123 |
| `execute_metered_insn_mi/s` |  3.76 | -          |  3.76 |  3.76 |
| `execute_e3_time_ms  ` |  3,090 |  3,090 |  3,090 |  3,090 |
| `execute_e3_insn_mi/s` |  2.59 | -          |  2.59 |  2.59 |
| `trace_gen_time_ms   ` |  1,562 |  1,562 |  1,562 |  1,562 |
| `memory_finalize_time_ms` |  112 |  112 |  112 |  112 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  19,379 |  19,379 |  19,379 |  19,379 |
| `main_trace_commit_time_ms` |  4,169 |  4,169 |  4,169 |  4,169 |
| `perm_trace_commit_time_ms` |  5,411 |  5,411 |  5,411 |  5,411 |
| `quotient_poly_compute_time_ms` |  2,135 |  2,135 |  2,135 |  2,135 |
| `quotient_poly_commit_time_ms` |  1,432 |  1,432 |  1,432 |  1,432 |
| `pcs_opening_time_ms ` |  4,617 |  4,617 |  4,617 |  4,617 |

| internal_wrapper.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,811 |  5,811 |  5,811 |  5,811 |
| `main_cells_used     ` |  73,686,793 |  73,686,793 |  73,686,793 |  73,686,793 |
| `total_cycles        ` |  1,197,777 |  1,197,777 |  1,197,777 |  1,197,777 |
| `execute_metered_time_ms` |  483 |  483 |  483 |  483 |
| `execute_metered_insn_mi/s` |  2.48 | -          |  2.48 |  2.48 |
| `execute_e3_time_ms  ` |  575 |  575 |  575 |  575 |
| `execute_e3_insn_mi/s` |  2.08 | -          |  2.08 |  2.08 |
| `trace_gen_time_ms   ` |  192 |  192 |  192 |  192 |
| `memory_finalize_time_ms` |  22 |  22 |  22 |  22 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  4,561 |  4,561 |  4,561 |  4,561 |
| `main_trace_commit_time_ms` |  1,011 |  1,011 |  1,011 |  1,011 |
| `perm_trace_commit_time_ms` |  780 |  780 |  780 |  780 |
| `quotient_poly_compute_time_ms` |  601 |  601 |  601 |  601 |
| `quotient_poly_commit_time_ms` |  740 |  740 |  740 |  740 |
| `pcs_opening_time_ms ` |  1,228 |  1,228 |  1,228 |  1,228 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,763 |  38,763 |  38,763 |  38,763 |
| `main_cells_used     ` |  38,505,728 |  38,505,728 |  38,505,728 |  38,505,728 |
| `total_cycles        ` |  772,514 |  772,514 |  772,514 |  772,514 |
| `execute_e3_time_ms  ` |  262 |  262 |  262 |  262 |
| `execute_e3_insn_mi/s` |  2.95 | -          |  2.95 |  2.95 |
| `trace_gen_time_ms   ` |  119 |  119 |  119 |  119 |
| `memory_finalize_time_ms` |  17 |  17 |  17 |  17 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  38,382 |  38,382 |  38,382 |  38,382 |
| `main_trace_commit_time_ms` |  12,307 |  12,307 |  12,307 |  12,307 |
| `perm_trace_commit_time_ms` |  7,666 |  7,666 |  7,666 |  7,666 |
| `quotient_poly_compute_time_ms` |  686 |  686 |  686 |  686 |
| `quotient_poly_commit_time_ms` |  13,718 |  13,718 |  13,718 |  13,718 |
| `pcs_opening_time_ms ` |  3,897 |  3,897 |  3,897 |  3,897 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  92,498 |  92,498 |  92,498 |  92,498 |
| `main_cells_used     ` |  65,626,678 |  65,626,678 |  65,626,678 |  65,626,678 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,214 |  44,214 |  44,214 |  44,214 |



<details>
<summary>Detailed Metrics</summary>

|  | wrapper_layer_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | insns | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5,846 | 92,514 | 44,214 | 17 | 1,198,250 | 204 | 3.78 | 259 | 2.98 | 0 | 23,395 | 38,780 | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | single_leaf_agg_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_segments | num_children | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms_time_ms | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer |  | 92,498 |  |  |  |  |  |  |  |  |  |  |  |  |  | 65,626,678 |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  | 44,214 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_wrapper.1 | 192 | 5,811 | 1,197,777 | 224,975,330 | 4,561 |  | 601 | 740 | 780 | 1,228 |  |  |  | 22 | 1,011 | 73,686,793 | 1,197,778 | 196 | 2 |  | 483 | 2.48 | 575 | 2.08 | 0 | 
| kitchen_sink |  |  |  |  |  |  |  |  |  |  | 1 |  | 23 |  |  |  | 154,596 |  | 1 | 22,809 | 145 | 5.61 |  |  |  | 
| leaf |  |  |  |  |  | 26,526 |  |  |  |  |  | 1 |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 

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
| kitchen_sink | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,562 | 26,154 | 7,991,094 | 2,500,267,498 | 19,379 | 2,135 | 1,432 | 5,411 | 4,617 | 112 | 4,169 | 732,639,889 | 7,991,095 | 1,607 |  | 2,123 | 3.76 | 3,090 | 2.59 | 0 | 
| root | 0 | 119 | 38,763 | 772,514 | 80,435,354 | 38,382 | 686 | 13,718 | 7,666 | 3,897 | 17 | 12,307 | 38,505,728 | 772,515 | 89 | 3 |  |  | 262 | 2.95 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 39,125,124 | 2,013,265,921 | 
| leaf | 0 | 1 | 291,111,168 | 2,013,265,921 | 
| leaf | 0 | 2 | 19,562,562 | 2,013,265,921 | 
| leaf | 0 | 3 | 288,096,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 642,351,818 | 2,013,265,921 | 
| root | 0 | 0 | 2,252,928 | 2,013,265,921 | 
| root | 0 | 1 | 14,557,184 | 2,013,265,921 | 
| root | 0 | 2 | 1,126,464 | 2,013,265,921 | 
| root | 0 | 3 | 15,540,224 | 2,013,265,921 | 
| root | 0 | 4 | 262,144 | 2,013,265,921 | 
| root | 0 | 5 | 34,263,234 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 485 | 15,423 | 154,595 | 1,481,195,140 | 14,870 | 6,269 | 306 | 18,761 | 1,506 | 1,234 | 80 | 24 | 85 | 5,044 | 898,132,732 | 154,596 | 467 | 68 | 2.27 | 1 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,977,978 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 32,428,728 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 988,989 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 32,011,232 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 57,344 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 24,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 49,612,052 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 1,048,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,448 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 120,668,771 | 2,013,265,921 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| internal_wrapper.1 | 0 | 5,177,476 | 2,013,265,921 | 
| internal_wrapper.1 | 1 | 30,814,464 | 2,013,265,921 | 
| internal_wrapper.1 | 2 | 2,588,738 | 2,013,265,921 | 
| internal_wrapper.1 | 3 | 30,941,444 | 2,013,265,921 | 
| internal_wrapper.1 | 4 | 262,144 | 2,013,265,921 | 
| internal_wrapper.1 | 5 | 70,177,482 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/535355ae9a0cf061cffa6c09d34f373819ac7f50

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15915150996)
