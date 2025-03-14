| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  258.18 |  192.25 |
| fib_e2e |  18.86 |  2.78 |
| leaf |  23.27 |  3.84 |
| internal.0 |  34.60 |  9.75 |
| internal.1 |  11.15 |  5.58 |
| internal.2 |  5.51 |  5.51 |
| root |  38.56 |  38.56 |
| halo2_outer |  82.44 |  82.44 |
| halo2_wrapper |  43.78 |  43.78 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,694.86 |  18,864 |  2,783 |  2,585 |
| `main_cells_used     ` |  58,670,294.43 |  410,692,061 |  59,803,985 |  51,985,507 |
| `total_cycles        ` |  1,714,299.43 |  12,000,096 |  1,747,502 |  1,515,084 |
| `execute_time_ms     ` |  245.29 |  1,717 |  251 |  217 |
| `trace_gen_time_ms   ` |  587 |  4,109 |  610 |  513 |
| `stark_prove_excluding_trace_time_ms` |  1,862.57 |  13,038 |  1,942 |  1,842 |
| `main_trace_commit_time_ms` |  321.71 |  2,252 |  347 |  317 |
| `generate_perm_trace_time_ms` |  108.57 |  760 |  110 |  107 |
| `perm_trace_commit_time_ms` |  484.71 |  3,393 |  497 |  479 |
| `quotient_poly_compute_time_ms` |  248.43 |  1,739 |  250 |  245 |
| `quotient_poly_commit_time_ms` |  200.71 |  1,405 |  235 |  193 |
| `pcs_opening_time_ms ` |  493.14 |  3,452 |  500 |  488 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,324.86 |  23,274 |  3,839 |  3,205 |
| `main_cells_used     ` |  61,343,914.86 |  429,407,404 |  70,181,097 |  59,069,086 |
| `total_cycles        ` |  1,090,003.29 |  7,630,023 |  1,261,912 |  1,046,126 |
| `execute_time_ms     ` |  470.57 |  3,294 |  504 |  461 |
| `trace_gen_time_ms   ` |  579.71 |  4,058 |  669 |  547 |
| `stark_prove_excluding_trace_time_ms` |  2,274.57 |  15,922 |  2,666 |  2,196 |
| `main_trace_commit_time_ms` |  404 |  2,828 |  469 |  390 |
| `generate_perm_trace_time_ms` |  127.43 |  892 |  155 |  119 |
| `perm_trace_commit_time_ms` |  603.43 |  4,224 |  744 |  574 |
| `quotient_poly_compute_time_ms` |  313.86 |  2,197 |  375 |  300 |
| `quotient_poly_commit_time_ms` |  254.14 |  1,779 |  296 |  245 |
| `pcs_opening_time_ms ` |  566.57 |  3,966 |  622 |  553 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,650 |  34,600 |  9,753 |  5,433 |
| `main_cells_used     ` |  119,682,357 |  478,729,428 |  136,640,032 |  69,211,938 |
| `total_cycles        ` |  2,042,381 |  8,169,524 |  2,334,095 |  1,167,382 |
| `execute_time_ms     ` |  821.50 |  3,286 |  942 |  469 |
| `trace_gen_time_ms   ` |  1,107.75 |  4,431 |  1,289 |  668 |
| `stark_prove_excluding_trace_time_ms` |  6,720.75 |  26,883 |  7,538 |  4,296 |
| `main_trace_commit_time_ms` |  1,410.75 |  5,643 |  1,599 |  849 |
| `generate_perm_trace_time_ms` |  250.75 |  1,003 |  286 |  150 |
| `perm_trace_commit_time_ms` |  1,262.75 |  5,051 |  1,429 |  773 |
| `quotient_poly_compute_time_ms` |  1,151.50 |  4,606 |  1,297 |  726 |
| `quotient_poly_commit_time_ms` |  1,193.75 |  4,775 |  1,331 |  809 |
| `pcs_opening_time_ms ` |  1,445.25 |  5,781 |  1,600 |  985 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,573 |  11,146 |  5,580 |  5,566 |
| `main_cells_used     ` |  75,036,570 |  150,073,140 |  75,321,157 |  74,751,983 |
| `total_cycles        ` |  1,529,835.50 |  3,059,671 |  1,533,706 |  1,525,965 |
| `execute_time_ms     ` |  481 |  962 |  483 |  479 |
| `trace_gen_time_ms   ` |  734.50 |  1,469 |  752 |  717 |
| `stark_prove_excluding_trace_time_ms` |  4,357.50 |  8,715 |  4,384 |  4,331 |
| `main_trace_commit_time_ms` |  857.50 |  1,715 |  866 |  849 |
| `generate_perm_trace_time_ms` |  143 |  286 |  143 |  143 |
| `perm_trace_commit_time_ms` |  757.50 |  1,515 |  764 |  751 |
| `quotient_poly_compute_time_ms` |  709 |  1,418 |  714 |  704 |
| `quotient_poly_commit_time_ms` |  873 |  1,746 |  874 |  872 |
| `pcs_opening_time_ms ` |  1,012.50 |  2,025 |  1,018 |  1,007 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,507 |  5,507 |  5,507 |  5,507 |
| `main_cells_used     ` |  73,934,718 |  73,934,718 |  73,934,718 |  73,934,718 |
| `total_cycles        ` |  1,518,664 |  1,518,664 |  1,518,664 |  1,518,664 |
| `execute_time_ms     ` |  464 |  464 |  464 |  464 |
| `trace_gen_time_ms   ` |  744 |  744 |  744 |  744 |
| `stark_prove_excluding_trace_time_ms` |  4,299 |  4,299 |  4,299 |  4,299 |
| `main_trace_commit_time_ms` |  847 |  847 |  847 |  847 |
| `generate_perm_trace_time_ms` |  141 |  141 |  141 |  141 |
| `perm_trace_commit_time_ms` |  763 |  763 |  763 |  763 |
| `quotient_poly_compute_time_ms` |  709 |  709 |  709 |  709 |
| `quotient_poly_commit_time_ms` |  842 |  842 |  842 |  842 |
| `pcs_opening_time_ms ` |  993 |  993 |  993 |  993 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,561 |  38,561 |  38,561 |  38,561 |
| `main_cells_used     ` |  37,781,695 |  37,781,695 |  37,781,695 |  37,781,695 |
| `total_cycles        ` |  760,143 |  760,143 |  760,143 |  760,143 |
| `execute_time_ms     ` |  233 |  233 |  233 |  233 |
| `trace_gen_time_ms   ` |  379 |  379 |  379 |  379 |
| `stark_prove_excluding_trace_time_ms` |  37,949 |  37,949 |  37,949 |  37,949 |
| `main_trace_commit_time_ms` |  12,312 |  12,312 |  12,312 |  12,312 |
| `generate_perm_trace_time_ms` |  71 |  71 |  71 |  71 |
| `perm_trace_commit_time_ms` |  7,606 |  7,606 |  7,606 |  7,606 |
| `quotient_poly_compute_time_ms` |  837 |  837 |  837 |  837 |
| `quotient_poly_commit_time_ms` |  13,474 |  13,474 |  13,474 |  13,474 |
| `pcs_opening_time_ms ` |  3,643 |  3,643 |  3,643 |  3,643 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  82,441 |  82,441 |  82,441 |  82,441 |
| `main_cells_used     ` |  61,389,170 |  61,389,170 |  61,389,170 |  61,389,170 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  43,785 |  43,785 |  43,785 |  43,785 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 234 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 82,441 |  | 61,389,170 | 
| halo2_wrapper | 43,785 |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 3 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 3 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 1 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 2 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 3 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 1 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 2 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 3 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 1 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 2 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 3 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.0 | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 1 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 2 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 3 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.0 | VolatileBoundaryAir | 1 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.0 | VolatileBoundaryAir | 2 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.0 | VolatileBoundaryAir | 3 | 131,072 |  | 8 | 11 | 2,490,368 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 4 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | AccessAdapterAir<8> | 5 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.1 | JalRangeCheckAir | 4 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.1 | JalRangeCheckAir | 5 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.1 | PhantomAir | 4 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.1 | PhantomAir | 5 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.1 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VolatileBoundaryAir | 4 | 131,072 |  | 8 | 11 | 2,490,368 | 
| internal.1 | VolatileBoundaryAir | 5 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.2 | AccessAdapterAir<8> | 6 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.2 | JalRangeCheckAir | 6 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.2 | PhantomAir | 6 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.2 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 12 | 5 | 34 | 
| internal.2 | VolatileBoundaryAir | 6 | 131,072 |  | 8 | 11 | 2,490,368 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 1 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 2 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 3 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 4 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 5 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 1 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 2 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 3 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 4 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 5 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 2 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 4 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 5 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 6 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | FriReducedOpeningAir | 1 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 2 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 3 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 4 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 5 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 6 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 2 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 3 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 4 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 5 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 6 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 1 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | PhantomAir | 2 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | PhantomAir | 3 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | PhantomAir | 4 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | PhantomAir | 5 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | PhantomAir | 6 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 1 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 2 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 3 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 4 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 5 | 131,072 |  | 12 | 11 | 3,014,656 | 
| leaf | VolatileBoundaryAir | 6 | 131,072 |  | 12 | 11 | 3,014,656 | 
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
| root | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 11 | 2,490,368 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 32 |  | 16 | 17 | 1,056 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 32 |  | 16 | 17 | 1,056 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 32 |  | 12 | 20 | 1,024 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 32 |  | 12 | 20 | 1,024 | 
| fib_e2e | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 1 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 2 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 3 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 4 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 5 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 6 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8 |  | 36 | 28 | 512 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 1 |  | 36 | 28 | 64 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 52 | 41 | 1,488 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 4 |  | 52 | 41 | 372 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 1,262 | 9,735 | 2,334,084 | 345,418,210 | 7,538 | 1,290 | 1,331 | 1,429 | 1,600 | 1,599 | 136,640,032 | 283 | 935 | 
| internal.0 | 1 | 1,212 | 9,679 | 2,334,095 | 345,418,210 | 7,525 | 1,297 | 1,320 | 1,422 | 1,596 | 1,596 | 136,439,521 | 286 | 942 | 
| internal.0 | 2 | 1,289 | 9,753 | 2,333,963 | 345,418,210 | 7,524 | 1,293 | 1,315 | 1,427 | 1,600 | 1,599 | 136,437,937 | 284 | 940 | 
| internal.0 | 3 | 668 | 5,433 | 1,167,382 | 184,375,778 | 4,296 | 726 | 809 | 773 | 985 | 849 | 69,211,938 | 150 | 469 | 
| internal.1 | 4 | 752 | 5,566 | 1,533,706 | 182,790,626 | 4,331 | 704 | 872 | 751 | 1,007 | 849 | 75,321,157 | 143 | 483 | 
| internal.1 | 5 | 717 | 5,580 | 1,525,965 | 185,280,994 | 4,384 | 714 | 874 | 764 | 1,018 | 866 | 74,751,983 | 143 | 479 | 
| internal.2 | 6 | 744 | 5,507 | 1,518,664 | 182,790,626 | 4,299 | 709 | 842 | 763 | 993 | 847 | 73,934,718 | 141 | 464 | 
| leaf | 0 | 669 | 3,839 | 1,261,912 | 251,993,578 | 2,666 | 375 | 296 | 744 | 622 | 469 | 70,181,097 | 155 | 504 | 
| leaf | 1 | 563 | 3,226 | 1,046,126 | 201,692,650 | 2,198 | 303 | 246 | 574 | 553 | 390 | 59,069,086 | 127 | 465 | 
| leaf | 2 | 568 | 3,230 | 1,046,167 | 201,692,650 | 2,201 | 300 | 246 | 580 | 559 | 390 | 59,069,578 | 120 | 461 | 
| leaf | 3 | 554 | 3,225 | 1,046,132 | 201,692,650 | 2,209 | 305 | 245 | 578 | 555 | 392 | 59,069,158 | 129 | 462 | 
| leaf | 4 | 562 | 3,232 | 1,046,144 | 201,692,650 | 2,207 | 305 | 247 | 578 | 558 | 395 | 59,069,302 | 120 | 463 | 
| leaf | 5 | 547 | 3,205 | 1,046,126 | 201,692,650 | 2,196 | 303 | 245 | 580 | 554 | 390 | 59,069,086 | 119 | 462 | 
| leaf | 6 | 595 | 3,317 | 1,137,416 | 206,904,810 | 2,245 | 306 | 254 | 590 | 565 | 402 | 63,880,097 | 122 | 477 | 
| root | 0 | 379 | 38,561 | 760,143 | 80,304,282 | 37,949 | 837 | 13,474 | 7,606 | 3,643 | 12,312 | 37,781,695 | 71 | 233 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 9,764,996 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 4,882,498 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 49,824,004 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 115,483,338 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 9,764,996 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 4,882,498 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 49,824,004 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 115,483,338 | 2,013,265,921 | 
| internal.0 | 2 | 0 | 9,764,996 | 2,013,265,921 | 
| internal.0 | 2 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 2 | 2 | 4,882,498 | 2,013,265,921 | 
| internal.0 | 2 | 3 | 49,824,004 | 2,013,265,921 | 
| internal.0 | 2 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 2 | 5 | 115,483,338 | 2,013,265,921 | 
| internal.0 | 3 | 0 | 4,882,564 | 2,013,265,921 | 
| internal.0 | 3 | 1 | 26,358,016 | 2,013,265,921 | 
| internal.0 | 3 | 2 | 2,441,282 | 2,013,265,921 | 
| internal.0 | 3 | 3 | 25,698,564 | 2,013,265,921 | 
| internal.0 | 3 | 4 | 131,072 | 2,013,265,921 | 
| internal.0 | 3 | 5 | 59,904,714 | 2,013,265,921 | 
| internal.1 | 4 | 0 | 5,144,708 | 2,013,265,921 | 
| internal.1 | 4 | 1 | 23,748,864 | 2,013,265,921 | 
| internal.1 | 4 | 2 | 2,572,354 | 2,013,265,921 | 
| internal.1 | 4 | 3 | 23,085,316 | 2,013,265,921 | 
| internal.1 | 4 | 4 | 131,072 | 2,013,265,921 | 
| internal.1 | 4 | 5 | 55,075,530 | 2,013,265,921 | 
| internal.1 | 5 | 0 | 5,144,708 | 2,013,265,921 | 
| internal.1 | 5 | 1 | 24,011,008 | 2,013,265,921 | 
| internal.1 | 5 | 2 | 2,572,354 | 2,013,265,921 | 
| internal.1 | 5 | 3 | 23,347,460 | 2,013,265,921 | 
| internal.1 | 5 | 4 | 131,072 | 2,013,265,921 | 
| internal.1 | 5 | 5 | 55,599,818 | 2,013,265,921 | 
| internal.2 | 6 | 0 | 5,144,708 | 2,013,265,921 | 
| internal.2 | 6 | 1 | 23,748,864 | 2,013,265,921 | 
| internal.2 | 6 | 2 | 2,572,354 | 2,013,265,921 | 
| internal.2 | 6 | 3 | 23,085,316 | 2,013,265,921 | 
| internal.2 | 6 | 4 | 131,072 | 2,013,265,921 | 
| internal.2 | 6 | 5 | 55,075,530 | 2,013,265,921 | 
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,484,996 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 61,919,946 | 2,013,265,921 | 
| leaf | 1 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 1 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 1 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 1 | 3 | 19,796,228 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 46,721,738 | 2,013,265,921 | 
| leaf | 2 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 2 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 2 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 2 | 3 | 19,796,228 | 2,013,265,921 | 
| leaf | 2 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 2 | 5 | 46,721,738 | 2,013,265,921 | 
| leaf | 3 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 3 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 3 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 3 | 3 | 19,796,228 | 2,013,265,921 | 
| leaf | 3 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 3 | 5 | 46,721,738 | 2,013,265,921 | 
| leaf | 4 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 4 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 4 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 4 | 3 | 19,796,228 | 2,013,265,921 | 
| leaf | 4 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 4 | 5 | 46,721,738 | 2,013,265,921 | 
| leaf | 5 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 5 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 5 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 5 | 3 | 19,796,228 | 2,013,265,921 | 
| leaf | 5 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 5 | 5 | 46,721,738 | 2,013,265,921 | 
| leaf | 6 | 0 | 4,391,044 | 2,013,265,921 | 
| leaf | 6 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 6 | 2 | 2,195,522 | 2,013,265,921 | 
| leaf | 6 | 3 | 20,193,540 | 2,013,265,921 | 
| leaf | 6 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 6 | 5 | 47,764,170 | 2,013,265,921 | 
| root | 0 | 0 | 2,252,928 | 2,013,265,921 | 
| root | 0 | 1 | 14,557,184 | 2,013,265,921 | 
| root | 0 | 2 | 1,126,464 | 2,013,265,921 | 
| root | 0 | 3 | 14,753,792 | 2,013,265,921 | 
| root | 0 | 4 | 262,144 | 2,013,265,921 | 
| root | 0 | 5 | 33,476,802 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 591 | 2,783 | 1,747,502 | 160,733,404 | 1,942 | 250 | 235 | 497 | 500 | 347 | 59,803,985 | 109 | 250 | 
| fib_e2e | 1 | 610 | 2,702 | 1,747,502 | 160,683,596 | 1,842 | 250 | 194 | 479 | 488 | 317 | 59,780,519 | 107 | 250 | 
| fib_e2e | 2 | 604 | 2,695 | 1,747,502 | 160,683,596 | 1,843 | 245 | 193 | 483 | 491 | 318 | 59,780,510 | 108 | 248 | 
| fib_e2e | 3 | 600 | 2,701 | 1,747,502 | 160,683,596 | 1,850 | 249 | 194 | 482 | 493 | 318 | 59,780,501 | 109 | 251 | 
| fib_e2e | 4 | 585 | 2,696 | 1,747,502 | 160,683,596 | 1,860 | 249 | 198 | 484 | 498 | 317 | 59,780,520 | 110 | 251 | 
| fib_e2e | 5 | 606 | 2,702 | 1,747,502 | 160,683,596 | 1,846 | 249 | 195 | 481 | 489 | 318 | 59,780,519 | 109 | 250 | 
| fib_e2e | 6 | 513 | 2,585 | 1,515,084 | 160,730,640 | 1,855 | 247 | 196 | 487 | 493 | 317 | 51,985,507 | 108 | 217 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fib_e2e | 0 | 0 | 3,932,254 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 10,748,232 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 1,966,127 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 10,748,236 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 800 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 288 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 7,209,036 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 35,526,829 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 35,525,389 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 35,525,389 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 35,525,389 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 35,525,389 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 35,525,389 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 3,932,176 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,060 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,088 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,012 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 800 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 288 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,208,961 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,526,241 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/7749b49af03a239a8d79bfea64ed8f81ccb9ed0e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13864161202)
