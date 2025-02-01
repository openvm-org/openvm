| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  348.50 |  244.17 |
| fib_e2e |  35.01 |  5.29 |
| leaf |  39.82 |  6.20 |
| internal.0 |  40.97 |  11.61 |
| internal.1 |  23.32 |  11.68 |
| internal.2 |  11.68 |  11.68 |
| root |  32.79 |  32.79 |
| halo2_outer |  119.46 |  119.46 |
| halo2_wrapper |  45.47 |  45.47 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,001 |  35,007 |  5,293 |  4,900 |
| `main_cells_used     ` |  58,670,271.71 |  410,691,902 |  59,803,937 |  51,985,456 |
| `total_cycles        ` |  1,515,083 |  1,515,083 |  1,515,083 |  1,515,083 |
| `execute_time_ms     ` |  330.86 |  2,316 |  343 |  294 |
| `trace_gen_time_ms   ` |  701.43 |  4,910 |  841 |  610 |
| `stark_prove_excluding_trace_time_ms` |  3,968.71 |  27,781 |  4,242 |  3,841 |
| `main_trace_commit_time_ms` |  721 |  5,047 |  869 |  678 |
| `generate_perm_trace_time_ms` |  126.71 |  887 |  138 |  115 |
| `perm_trace_commit_time_ms` |  815.43 |  5,708 |  864 |  763 |
| `quotient_poly_compute_time_ms` |  536.71 |  3,757 |  543 |  532 |
| `quotient_poly_commit_time_ms` |  695 |  4,865 |  781 |  671 |
| `pcs_opening_time_ms ` |  1,070.86 |  7,496 |  1,112 |  1,050 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,689.14 |  39,824 |  6,198 |  5,583 |
| `main_cells_used     ` |  43,776,845.43 |  306,437,918 |  50,689,063 |  41,987,445 |
| `total_cycles        ` |  1,054,633.14 |  7,382,432 |  1,243,866 |  1,005,409 |
| `execute_time_ms     ` |  304.14 |  2,129 |  336 |  285 |
| `trace_gen_time_ms   ` |  794.71 |  5,563 |  882 |  765 |
| `stark_prove_excluding_trace_time_ms` |  4,590.29 |  32,132 |  4,980 |  4,500 |
| `main_trace_commit_time_ms` |  920.57 |  6,444 |  1,005 |  901 |
| `generate_perm_trace_time_ms` |  110.57 |  774 |  116 |  106 |
| `perm_trace_commit_time_ms` |  836.29 |  5,854 |  941 |  806 |
| `quotient_poly_compute_time_ms` |  615.14 |  4,306 |  688 |  575 |
| `quotient_poly_commit_time_ms` |  920.29 |  6,442 |  996 |  896 |
| `pcs_opening_time_ms ` |  1,183.57 |  8,285 |  1,229 |  1,164 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,241.25 |  40,965 |  11,609 |  6,440 |
| `main_cells_used     ` |  85,682,130 |  342,728,520 |  97,648,850 |  49,894,785 |
| `total_cycles        ` |  2,084,628.75 |  8,338,515 |  2,382,559 |  1,191,827 |
| `execute_time_ms     ` |  563 |  2,252 |  643 |  331 |
| `trace_gen_time_ms   ` |  1,434.25 |  5,737 |  1,612 |  1,147 |
| `stark_prove_excluding_trace_time_ms` |  8,244 |  32,976 |  9,379 |  4,962 |
| `main_trace_commit_time_ms` |  1,714 |  6,856 |  1,969 |  981 |
| `generate_perm_trace_time_ms` |  201.25 |  805 |  230 |  120 |
| `perm_trace_commit_time_ms` |  1,624.25 |  6,497 |  1,868 |  920 |
| `quotient_poly_compute_time_ms` |  1,109.75 |  4,439 |  1,263 |  673 |
| `quotient_poly_commit_time_ms` |  1,598 |  6,392 |  1,808 |  992 |
| `pcs_opening_time_ms ` |  1,992.50 |  7,970 |  2,259 |  1,271 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,659 |  23,318 |  11,684 |  11,634 |
| `main_cells_used     ` |  99,374,166 |  198,748,332 |  99,581,431 |  99,166,901 |
| `total_cycles        ` |  2,407,591.50 |  4,815,183 |  2,415,547 |  2,399,636 |
| `execute_time_ms     ` |  671.50 |  1,343 |  673 |  670 |
| `trace_gen_time_ms   ` |  1,572 |  3,144 |  1,622 |  1,522 |
| `stark_prove_excluding_trace_time_ms` |  9,415.50 |  18,831 |  9,439 |  9,392 |
| `main_trace_commit_time_ms` |  1,962 |  3,924 |  1,971 |  1,953 |
| `generate_perm_trace_time_ms` |  231.50 |  463 |  232 |  231 |
| `perm_trace_commit_time_ms` |  1,860 |  3,720 |  1,866 |  1,854 |
| `quotient_poly_compute_time_ms` |  1,252 |  2,504 |  1,265 |  1,239 |
| `quotient_poly_commit_time_ms` |  1,821.50 |  3,643 |  1,822 |  1,821 |
| `pcs_opening_time_ms ` |  2,284.50 |  4,569 |  2,308 |  2,261 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,676 |  11,676 |  11,676 |  11,676 |
| `main_cells_used     ` |  99,576,013 |  99,576,013 |  99,576,013 |  99,576,013 |
| `total_cycles        ` |  2,414,945 |  2,414,945 |  2,414,945 |  2,414,945 |
| `execute_time_ms     ` |  683 |  683 |  683 |  683 |
| `trace_gen_time_ms   ` |  1,631 |  1,631 |  1,631 |  1,631 |
| `stark_prove_excluding_trace_time_ms` |  9,362 |  9,362 |  9,362 |  9,362 |
| `main_trace_commit_time_ms` |  1,954 |  1,954 |  1,954 |  1,954 |
| `generate_perm_trace_time_ms` |  229 |  229 |  229 |  229 |
| `perm_trace_commit_time_ms` |  1,872 |  1,872 |  1,872 |  1,872 |
| `quotient_poly_compute_time_ms` |  1,256 |  1,256 |  1,256 |  1,256 |
| `quotient_poly_commit_time_ms` |  1,794 |  1,794 |  1,794 |  1,794 |
| `pcs_opening_time_ms ` |  2,251 |  2,251 |  2,251 |  2,251 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  32,789 |  32,789 |  32,789 |  32,789 |
| `main_cells_used     ` |  50,905,363 |  50,905,363 |  50,905,363 |  50,905,363 |
| `total_cycles        ` |  1,208,553 |  1,208,553 |  1,208,553 |  1,208,553 |
| `execute_time_ms     ` |  408 |  408 |  408 |  408 |
| `trace_gen_time_ms   ` |  806 |  806 |  806 |  806 |
| `stark_prove_excluding_trace_time_ms` |  31,575 |  31,575 |  31,575 |  31,575 |
| `main_trace_commit_time_ms` |  10,183 |  10,183 |  10,183 |  10,183 |
| `generate_perm_trace_time_ms` |  121 |  121 |  121 |  121 |
| `perm_trace_commit_time_ms` |  9,633 |  9,633 |  9,633 |  9,633 |
| `quotient_poly_compute_time_ms` |  666 |  666 |  666 |  666 |
| `quotient_poly_commit_time_ms` |  7,217 |  7,217 |  7,217 |  7,217 |
| `pcs_opening_time_ms ` |  3,751 |  3,751 |  3,751 |  3,751 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  119,456 |  119,456 |  119,456 |  119,456 |
| `main_cells_used     ` |  74,770,560 |  74,770,560 |  74,770,560 |  74,770,560 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  45,465 |  45,465 |  45,465 |  45,465 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 341 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 119,456 |  | 74,770,560 | 
| halo2_wrapper | 45,465 |  |  | 

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
| leaf | PhantomAir | 1 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 2 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 3 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 4 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 5 | 16,384 |  | 8 | 6 | 229,376 | 
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
| internal.0 | 0 | 1,400 | 11,421 | 2,382,559 | 276,382,680 | 9,379 | 1,263 | 1,808 | 1,849 | 2,259 | 1,969 | 97,648,850 | 227 | 642 | 
| internal.0 | 1 | 1,578 | 11,495 | 2,381,927 | 276,382,680 | 9,281 | 1,256 | 1,794 | 1,860 | 2,183 | 1,954 | 97,591,205 | 230 | 636 | 
| internal.0 | 2 | 1,612 | 11,609 | 2,382,202 | 276,382,680 | 9,354 | 1,247 | 1,798 | 1,868 | 2,257 | 1,952 | 97,593,680 | 228 | 643 | 
| internal.0 | 3 | 1,147 | 6,440 | 1,191,827 | 141,731,544 | 4,962 | 673 | 992 | 920 | 1,271 | 981 | 49,894,785 | 120 | 331 | 
| internal.1 | 4 | 1,522 | 11,634 | 2,415,547 | 276,382,680 | 9,439 | 1,239 | 1,821 | 1,866 | 2,308 | 1,971 | 99,581,431 | 231 | 673 | 
| internal.1 | 5 | 1,622 | 11,684 | 2,399,636 | 276,382,680 | 9,392 | 1,265 | 1,822 | 1,854 | 2,261 | 1,953 | 99,166,901 | 232 | 670 | 
| internal.2 | 6 | 1,631 | 11,676 | 2,414,945 | 276,382,680 | 9,362 | 1,256 | 1,794 | 1,872 | 2,251 | 1,954 | 99,576,013 | 229 | 683 | 
| leaf | 0 | 882 | 6,198 | 1,243,866 | 139,372,248 | 4,980 | 688 | 996 | 941 | 1,229 | 1,005 | 50,689,063 | 116 | 336 | 
| leaf | 1 | 782 | 5,583 | 1,005,957 | 127,313,624 | 4,516 | 599 | 896 | 830 | 1,170 | 904 | 41,992,377 | 113 | 285 | 
| leaf | 2 | 772 | 5,637 | 1,005,920 | 127,313,624 | 4,567 | 613 | 912 | 821 | 1,201 | 903 | 41,992,044 | 114 | 298 | 
| leaf | 3 | 765 | 5,589 | 1,006,014 | 127,313,624 | 4,523 | 614 | 908 | 815 | 1,167 | 909 | 41,992,890 | 106 | 301 | 
| leaf | 4 | 770 | 5,591 | 1,006,028 | 127,313,624 | 4,524 | 608 | 904 | 814 | 1,184 | 901 | 41,993,016 | 109 | 297 | 
| leaf | 5 | 773 | 5,597 | 1,005,409 | 127,313,624 | 4,522 | 609 | 903 | 827 | 1,164 | 908 | 41,987,445 | 108 | 302 | 
| leaf | 6 | 819 | 5,629 | 1,109,238 | 127,321,048 | 4,500 | 575 | 923 | 806 | 1,170 | 914 | 45,791,083 | 108 | 310 | 
| root | 0 | 806 | 32,789 | 1,208,553 | 141,731,544 | 31,575 | 666 | 7,217 | 9,633 | 3,751 | 10,183 | 50,905,363 | 121 | 408 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 728 | 5,293 |  | 122,458,092 | 4,242 | 543 | 781 | 858 | 1,050 | 869 | 59,803,937 | 138 | 323 | 
| fib_e2e | 1 | 841 | 5,024 |  | 122,409,910 | 3,841 | 534 | 671 | 763 | 1,053 | 688 | 59,780,497 | 128 | 342 | 
| fib_e2e | 2 | 684 | 4,900 |  | 122,409,910 | 3,878 | 532 | 683 | 785 | 1,058 | 695 | 59,780,490 | 124 | 338 | 
| fib_e2e | 3 | 677 | 4,918 |  | 122,409,910 | 3,902 | 540 | 686 | 774 | 1,092 | 678 | 59,780,508 | 128 | 339 | 
| fib_e2e | 4 | 691 | 5,017 |  | 122,409,910 | 3,989 | 538 | 687 | 840 | 1,070 | 724 | 59,780,507 | 127 | 337 | 
| fib_e2e | 5 | 679 | 4,948 |  | 122,409,910 | 3,926 | 533 | 684 | 824 | 1,061 | 706 | 59,780,507 | 115 | 343 | 
| fib_e2e | 6 | 610 | 4,907 | 1,515,083 | 122,456,198 | 4,003 | 537 | 673 | 864 | 1,112 | 687 | 51,985,456 | 127 | 294 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/aea5c78afafa70a6c6b754772ebd3cb8244e63e7

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13090658672)
