| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  219.55 |  219.55 |
| kitchen_sink |  15.72 |  15.72 |
| leaf |  24.38 |  24.38 |
| internal_wrapper.1 |  5.37 |  5.37 |
| root |  38.79 |  38.79 |
| halo2_outer |  90.75 |  90.75 |
| halo2_wrapper |  44.54 |  44.54 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,718 |  15,718 |  15,718 |  15,718 |
| `main_cells_used     ` |  898,132,732 |  898,132,732 |  898,132,732 |  898,132,732 |
| `total_cycles        ` |  154,595 |  154,595 |  154,595 |  154,595 |
| `execute_metered_time_ms` |  46 |  46 |  46 |  46 |
| `execute_time_ms     ` |  84 |  84 |  84 |  84 |
| `trace_gen_time_ms   ` |  355 |  355 |  355 |  355 |
| `stark_prove_excluding_trace_time_ms` |  15,279 |  15,279 |  15,279 |  15,279 |
| `main_trace_commit_time_ms` |  5,173 |  5,173 |  5,173 |  5,173 |
| `generate_perm_trace_time_ms` |  507 |  507 |  507 |  507 |
| `perm_trace_commit_time_ms` |  1,538 |  1,538 |  1,538 |  1,538 |
| `quotient_poly_compute_time_ms` |  6,451 |  6,451 |  6,451 |  6,451 |
| `quotient_poly_commit_time_ms` |  310 |  310 |  310 |  310 |
| `pcs_opening_time_ms ` |  1,256 |  1,256 |  1,256 |  1,256 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  24,378 |  24,378 |  24,378 |  24,378 |
| `main_cells_used     ` |  732,641,233 |  732,641,233 |  732,641,233 |  732,641,233 |
| `total_cycles        ` |  7,991,206 |  7,991,206 |  7,991,206 |  7,991,206 |
| `execute_metered_time_ms` |  4,048 |  4,048 |  4,048 |  4,048 |
| `execute_time_ms     ` |  3,255 |  3,255 |  3,255 |  3,255 |
| `trace_gen_time_ms   ` |  1,584 |  1,584 |  1,584 |  1,584 |
| `stark_prove_excluding_trace_time_ms` |  19,539 |  19,539 |  19,539 |  19,539 |
| `main_trace_commit_time_ms` |  4,245 |  4,245 |  4,245 |  4,245 |
| `generate_perm_trace_time_ms` |  1,573 |  1,573 |  1,573 |  1,573 |
| `perm_trace_commit_time_ms` |  5,507 |  5,507 |  5,507 |  5,507 |
| `quotient_poly_compute_time_ms` |  2,115 |  2,115 |  2,115 |  2,115 |
| `quotient_poly_commit_time_ms` |  1,458 |  1,458 |  1,458 |  1,458 |
| `pcs_opening_time_ms ` |  4,635 |  4,635 |  4,635 |  4,635 |

| internal_wrapper.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,370 |  5,370 |  5,370 |  5,370 |
| `main_cells_used     ` |  73,686,661 |  73,686,661 |  73,686,661 |  73,686,661 |
| `total_cycles        ` |  1,197,766 |  1,197,766 |  1,197,766 |  1,197,766 |
| `execute_metered_time_ms` |  569 |  569 |  569 |  569 |
| `execute_time_ms     ` |  577 |  577 |  577 |  577 |
| `trace_gen_time_ms   ` |  203 |  203 |  203 |  203 |
| `stark_prove_excluding_trace_time_ms` |  4,590 |  4,590 |  4,590 |  4,590 |
| `main_trace_commit_time_ms` |  1,029 |  1,029 |  1,029 |  1,029 |
| `generate_perm_trace_time_ms` |  197 |  197 |  197 |  197 |
| `perm_trace_commit_time_ms` |  751 |  751 |  751 |  751 |
| `quotient_poly_compute_time_ms` |  598 |  598 |  598 |  598 |
| `quotient_poly_commit_time_ms` |  739 |  739 |  739 |  739 |
| `pcs_opening_time_ms ` |  1,270 |  1,270 |  1,270 |  1,270 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,792 |  38,792 |  38,792 |  38,792 |
| `main_cells_used     ` |  38,506,520 |  38,506,520 |  38,506,520 |  38,506,520 |
| `total_cycles        ` |  772,580 |  772,580 |  772,580 |  772,580 |
| `execute_time_ms     ` |  270 |  270 |  270 |  270 |
| `trace_gen_time_ms   ` |  128 |  128 |  128 |  128 |
| `stark_prove_excluding_trace_time_ms` |  38,394 |  38,394 |  38,394 |  38,394 |
| `main_trace_commit_time_ms` |  12,320 |  12,320 |  12,320 |  12,320 |
| `generate_perm_trace_time_ms` |  89 |  89 |  89 |  89 |
| `perm_trace_commit_time_ms` |  7,611 |  7,611 |  7,611 |  7,611 |
| `quotient_poly_compute_time_ms` |  712 |  712 |  712 |  712 |
| `quotient_poly_commit_time_ms` |  13,694 |  13,694 |  13,694 |  13,694 |
| `pcs_opening_time_ms ` |  3,877 |  3,877 |  3,877 |  3,877 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,754 |  90,754 |  90,754 |  90,754 |
| `main_cells_used     ` |  65,626,678 |  65,626,678 |  65,626,678 |  65,626,678 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,537 |  44,537 |  44,537 |  44,537 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms | execute_metered_time_ms |
| --- | --- |
|  | 267 | 269 | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_segments | num_children | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_time_ms | execute_metered_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer |  | 90,754 |  |  |  |  |  |  |  |  |  |  | 65,626,678 |  |  |  |  | 
| halo2_wrapper |  | 44,537 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal_wrapper.1 | 203 | 5,370 | 1,197,766 | 224,975,330 | 4,590 | 598 | 739 | 751 | 1,270 |  |  | 1,029 | 73,686,661 | 197 | 2 | 577 | 569 | 
| kitchen_sink |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  | 46 | 
| leaf |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  | 1 |  |  | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_time_ms | execute_metered_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,584 | 24,378 | 7,991,206 | 2,500,267,498 | 19,539 | 2,115 | 1,458 | 5,507 | 4,635 | 4,245 | 732,641,233 | 1,573 |  | 3,255 | 4,048 | 
| root | 0 | 128 | 38,792 | 772,580 | 80,435,354 | 38,394 | 712 | 13,694 | 7,611 | 3,877 | 12,320 | 38,506,520 | 89 | 3 | 270 |  | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 355 | 15,718 | 154,595 | 1,481,195,140 | 15,279 | 6,451 | 310 | 1,538 | 1,256 | 5,173 | 898,132,732 | 507 | 84 | 

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


Commit: https://github.com/openvm-org/openvm/commit/2fa8d0b8dbd95c596f9a74e833085ba5df35a8b2

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15759988908)
