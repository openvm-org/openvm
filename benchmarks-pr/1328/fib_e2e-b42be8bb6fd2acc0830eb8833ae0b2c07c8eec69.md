| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  351.16 |  246.58 |
| fib_e2e |  35.13 |  5.36 |
| leaf |  39.94 |  6.32 |
| internal.0 |  41.41 |  11.75 |
| internal.1 |  23.39 |  11.86 |
| internal.2 |  11.82 |  11.82 |
| root |  32.87 |  32.87 |
| halo2_outer |  118.88 |  118.88 |
| halo2_wrapper |  47.74 |  47.74 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,018.14 |  35,127 |  5,359 |  4,878 |
| `main_cells_used     ` |  58,670,271.71 |  410,691,902 |  59,803,937 |  51,985,456 |
| `total_cycles        ` |  1,515,083 |  1,515,083 |  1,515,083 |  1,515,083 |
| `execute_time_ms     ` |  333.14 |  2,332 |  345 |  296 |
| `trace_gen_time_ms   ` |  722.29 |  5,056 |  814 |  615 |
| `stark_prove_excluding_trace_time_ms` |  3,962.71 |  27,739 |  4,270 |  3,731 |
| `main_trace_commit_time_ms` |  694 |  4,858 |  842 |  611 |
| `generate_perm_trace_time_ms` |  131.14 |  918 |  153 |  120 |
| `perm_trace_commit_time_ms` |  815.71 |  5,710 |  877 |  714 |
| `quotient_poly_compute_time_ms` |  529.86 |  3,709 |  541 |  512 |
| `quotient_poly_commit_time_ms` |  704.29 |  4,930 |  795 |  682 |
| `pcs_opening_time_ms ` |  1,084.29 |  7,590 |  1,106 |  1,059 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,705.14 |  39,936 |  6,315 |  5,559 |
| `main_cells_used     ` |  43,769,435.14 |  306,386,046 |  50,685,677 |  41,980,387 |
| `total_cycles        ` |  1,053,418 |  7,373,926 |  1,243,098 |  1,004,233 |
| `execute_time_ms     ` |  302.71 |  2,119 |  333 |  293 |
| `trace_gen_time_ms   ` |  801.57 |  5,611 |  968 |  751 |
| `stark_prove_excluding_trace_time_ms` |  4,600.86 |  32,206 |  5,014 |  4,501 |
| `main_trace_commit_time_ms` |  909.43 |  6,366 |  1,009 |  881 |
| `generate_perm_trace_time_ms` |  109.14 |  764 |  117 |  107 |
| `perm_trace_commit_time_ms` |  855.14 |  5,986 |  922 |  824 |
| `quotient_poly_compute_time_ms` |  621.14 |  4,348 |  684 |  595 |
| `quotient_poly_commit_time_ms` |  930.86 |  6,516 |  1,015 |  898 |
| `pcs_opening_time_ms ` |  1,171.29 |  8,199 |  1,263 |  1,144 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,351.50 |  41,406 |  11,749 |  6,477 |
| `main_cells_used     ` |  85,667,284.75 |  342,669,139 |  97,631,919 |  49,884,982 |
| `total_cycles        ` |  2,082,576.25 |  8,330,305 |  2,380,153 |  1,190,374 |
| `execute_time_ms     ` |  561.25 |  2,245 |  648 |  323 |
| `trace_gen_time_ms   ` |  1,471 |  5,884 |  1,731 |  1,001 |
| `stark_prove_excluding_trace_time_ms` |  8,319.25 |  33,277 |  9,385 |  5,153 |
| `main_trace_commit_time_ms` |  1,735 |  6,940 |  1,954 |  1,146 |
| `generate_perm_trace_time_ms` |  201 |  804 |  230 |  118 |
| `perm_trace_commit_time_ms` |  1,624 |  6,496 |  1,864 |  925 |
| `quotient_poly_compute_time_ms` |  1,111.25 |  4,445 |  1,269 |  681 |
| `quotient_poly_commit_time_ms` |  1,613.25 |  6,453 |  1,825 |  1,009 |
| `pcs_opening_time_ms ` |  2,029.75 |  8,119 |  2,309 |  1,269 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,692.50 |  23,385 |  11,858 |  11,527 |
| `main_cells_used     ` |  99,360,362.50 |  198,720,725 |  99,566,651 |  99,154,074 |
| `total_cycles        ` |  2,405,309 |  4,810,618 |  2,413,149 |  2,397,469 |
| `execute_time_ms     ` |  680 |  1,360 |  683 |  677 |
| `trace_gen_time_ms   ` |  1,638.50 |  3,277 |  1,770 |  1,507 |
| `stark_prove_excluding_trace_time_ms` |  9,374 |  18,748 |  9,411 |  9,337 |
| `main_trace_commit_time_ms` |  1,923 |  3,846 |  1,927 |  1,919 |
| `generate_perm_trace_time_ms` |  230.50 |  461 |  233 |  228 |
| `perm_trace_commit_time_ms` |  1,848 |  3,696 |  1,853 |  1,843 |
| `quotient_poly_compute_time_ms` |  1,238 |  2,476 |  1,241 |  1,235 |
| `quotient_poly_commit_time_ms` |  1,834 |  3,668 |  1,838 |  1,830 |
| `pcs_opening_time_ms ` |  2,296.50 |  4,593 |  2,325 |  2,268 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,818 |  11,818 |  11,818 |  11,818 |
| `main_cells_used     ` |  99,566,543 |  99,566,543 |  99,566,543 |  99,566,543 |
| `total_cycles        ` |  2,413,137 |  2,413,137 |  2,413,137 |  2,413,137 |
| `execute_time_ms     ` |  682 |  682 |  682 |  682 |
| `trace_gen_time_ms   ` |  1,758 |  1,758 |  1,758 |  1,758 |
| `stark_prove_excluding_trace_time_ms` |  9,378 |  9,378 |  9,378 |  9,378 |
| `main_trace_commit_time_ms` |  1,934 |  1,934 |  1,934 |  1,934 |
| `generate_perm_trace_time_ms` |  230 |  230 |  230 |  230 |
| `perm_trace_commit_time_ms` |  1,844 |  1,844 |  1,844 |  1,844 |
| `quotient_poly_compute_time_ms` |  1,252 |  1,252 |  1,252 |  1,252 |
| `quotient_poly_commit_time_ms` |  1,825 |  1,825 |  1,825 |  1,825 |
| `pcs_opening_time_ms ` |  2,287 |  2,287 |  2,287 |  2,287 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  32,868 |  32,868 |  32,868 |  32,868 |
| `main_cells_used     ` |  50,896,820 |  50,896,820 |  50,896,820 |  50,896,820 |
| `total_cycles        ` |  1,207,226 |  1,207,226 |  1,207,226 |  1,207,226 |
| `execute_time_ms     ` |  395 |  395 |  395 |  395 |
| `trace_gen_time_ms   ` |  897 |  897 |  897 |  897 |
| `stark_prove_excluding_trace_time_ms` |  31,576 |  31,576 |  31,576 |  31,576 |
| `main_trace_commit_time_ms` |  10,175 |  10,175 |  10,175 |  10,175 |
| `generate_perm_trace_time_ms` |  117 |  117 |  117 |  117 |
| `perm_trace_commit_time_ms` |  9,635 |  9,635 |  9,635 |  9,635 |
| `quotient_poly_compute_time_ms` |  660 |  660 |  660 |  660 |
| `quotient_poly_commit_time_ms` |  7,234 |  7,234 |  7,234 |  7,234 |
| `pcs_opening_time_ms ` |  3,747 |  3,747 |  3,747 |  3,747 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  118,879 |  118,879 |  118,879 |  118,879 |
| `main_cells_used     ` |  74,770,560 |  74,770,560 |  74,770,560 |  74,770,560 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  47,737 |  47,737 |  47,737 |  47,737 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 345 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 118,879 |  | 74,770,560 | 
| halo2_wrapper | 47,737 |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 3 | 262,144 |  | 12 | 11 | 6,029,312 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 3 | 131,072 |  | 12 | 13 | 3,276,800 | 
| internal.0 | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 1 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 2 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 3 | 256 |  | 12 | 17 | 7,424 | 
| internal.0 | FriReducedOpeningAir | 0 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 1 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 2 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 25 | 7,995,392 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 160 | 399 | 18,317,312 | 
| internal.0 | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 1 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 2 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | PhantomAir | 3 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.0 | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 2 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 3 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 32,768 |  | 12 | 9 | 688,128 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 24 | 22 | 24,117,248 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 24 | 31 | 3,604,480 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 38 | 3,801,088 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 2 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 3 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 4 | 512 |  | 12 | 17 | 14,848 | 
| internal.1 | AccessAdapterAir<8> | 5 | 512 |  | 12 | 17 | 14,848 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.1 | PhantomAir | 4 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.1 | PhantomAir | 5 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.1 | ProgramAir | 4 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | ProgramAir | 5 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.1 | VolatileBoundaryAir | 5 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.2 | AccessAdapterAir<8> | 6 | 512 |  | 12 | 17 | 14,848 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.2 | PhantomAir | 6 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.2 | ProgramAir | 6 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 1 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 2 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 3 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 4 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 5 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 1 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 2 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 3 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 4 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 5 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<8> | 0 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 1 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 2 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 3 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 4 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 5 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 6 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | PhantomAir | 0 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 1 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 2 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 3 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 4 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 5 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 6 | 16,384 |  | 8 | 6 | 229,376 | 
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
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 1 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 2 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 3 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 4 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 5 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 6 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| root | AccessAdapterAir<8> | 0 | 256 |  | 12 | 17 | 7,424 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| root | PhantomAir | 0 | 16,384 |  | 8 | 6 | 229,376 | 
| root | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| root | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 32 |  | 12 | 17 | 928 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 12 | 32 | 11,264 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 32 |  | 8 | 20 | 896 | 
| fib_e2e | PhantomAir | 0 | 2 |  | 8 | 6 | 28 | 
| fib_e2e | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 2 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 3 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 4 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 5 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 6 | 1 |  | 8 | 6 | 14 | 
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
| fib_e2e | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 20 | 32 | 208 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8 |  | 20 | 28 | 384 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 1 |  | 20 | 28 | 48 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 40 | 1,088 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 4 |  | 28 | 40 | 272 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 21 | 296 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 1,430 | 11,435 | 2,380,137 | 276,382,680 | 9,373 | 1,244 | 1,815 | 1,859 | 2,267 | 1,954 | 97,631,919 | 229 | 632 | 
| internal.0 | 1 | 1,722 | 11,749 | 2,380,153 | 276,382,680 | 9,385 | 1,269 | 1,804 | 1,848 | 2,309 | 1,919 | 97,578,423 | 230 | 642 | 
| internal.0 | 2 | 1,731 | 11,745 | 2,379,641 | 276,382,680 | 9,366 | 1,251 | 1,825 | 1,864 | 2,274 | 1,921 | 97,573,815 | 227 | 648 | 
| internal.0 | 3 | 1,001 | 6,477 | 1,190,374 | 141,731,544 | 5,153 | 681 | 1,009 | 925 | 1,269 | 1,146 | 49,884,982 | 118 | 323 | 
| internal.1 | 4 | 1,507 | 11,527 | 2,413,149 | 276,382,680 | 9,337 | 1,235 | 1,830 | 1,853 | 2,268 | 1,919 | 99,566,651 | 228 | 683 | 
| internal.1 | 5 | 1,770 | 11,858 | 2,397,469 | 276,382,680 | 9,411 | 1,241 | 1,838 | 1,843 | 2,325 | 1,927 | 99,154,074 | 233 | 677 | 
| internal.2 | 6 | 1,758 | 11,818 | 2,413,137 | 276,382,680 | 9,378 | 1,252 | 1,825 | 1,844 | 2,287 | 1,934 | 99,566,543 | 230 | 682 | 
| leaf | 0 | 968 | 6,315 | 1,243,098 | 139,372,248 | 5,014 | 684 | 1,015 | 922 | 1,263 | 1,009 | 50,685,677 | 117 | 333 | 
| leaf | 1 | 792 | 5,645 | 1,004,781 | 127,198,936 | 4,559 | 636 | 898 | 832 | 1,169 | 913 | 41,985,319 | 107 | 294 | 
| leaf | 2 | 767 | 5,605 | 1,004,744 | 127,198,936 | 4,541 | 603 | 931 | 857 | 1,158 | 881 | 41,984,986 | 107 | 297 | 
| leaf | 3 | 751 | 5,572 | 1,004,837 | 127,198,936 | 4,528 | 603 | 913 | 847 | 1,160 | 892 | 41,985,823 | 109 | 293 | 
| leaf | 4 | 769 | 5,567 | 1,004,634 | 127,198,936 | 4,501 | 595 | 914 | 853 | 1,144 | 883 | 41,983,996 | 109 | 297 | 
| leaf | 5 | 763 | 5,559 | 1,004,233 | 127,198,936 | 4,502 | 618 | 913 | 824 | 1,148 | 888 | 41,980,387 | 107 | 294 | 
| leaf | 6 | 801 | 5,673 | 1,107,599 | 127,321,048 | 4,561 | 609 | 932 | 851 | 1,157 | 900 | 45,779,858 | 108 | 311 | 
| root | 0 | 897 | 32,868 | 1,207,226 | 141,731,544 | 31,576 | 660 | 7,234 | 9,635 | 3,747 | 10,175 | 50,896,820 | 117 | 395 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 762 | 5,359 |  | 122,458,092 | 4,270 | 541 | 795 | 877 | 1,059 | 842 | 59,803,937 | 153 | 327 | 
| fib_e2e | 1 | 814 | 4,886 |  | 122,409,910 | 3,731 | 531 | 689 | 714 | 1,062 | 611 | 59,780,497 | 121 | 341 | 
| fib_e2e | 2 | 798 | 5,137 |  | 122,409,910 | 3,999 | 532 | 683 | 852 | 1,083 | 708 | 59,780,490 | 137 | 340 | 
| fib_e2e | 3 | 693 | 4,932 |  | 122,409,910 | 3,897 | 512 | 682 | 804 | 1,106 | 662 | 59,780,508 | 127 | 342 | 
| fib_e2e | 4 | 694 | 4,945 |  | 122,409,910 | 3,910 | 533 | 696 | 744 | 1,104 | 703 | 59,780,507 | 127 | 341 | 
| fib_e2e | 5 | 680 | 4,990 |  | 122,409,910 | 3,965 | 530 | 686 | 860 | 1,099 | 667 | 59,780,507 | 120 | 345 | 
| fib_e2e | 6 | 615 | 4,878 | 1,515,083 | 122,456,198 | 3,967 | 530 | 699 | 859 | 1,077 | 665 | 51,985,456 | 133 | 296 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b42be8bb6fd2acc0830eb8833ae0b2c07c8eec69

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13097254898)
