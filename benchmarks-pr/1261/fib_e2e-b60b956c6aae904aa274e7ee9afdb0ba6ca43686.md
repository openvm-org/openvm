| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  417.25 |  277.91 |
| fib_e2e |  41.25 |  6.22 |
| leaf |  54.33 |  8.14 |
| internal.0 |  58.48 |  16.75 |
| internal.1 |  33.21 |  16.83 |
| internal.2 |  16.86 |  16.86 |
| root |  39.71 |  39.71 |
| halo2_outer |  132.08 |  132.08 |
| halo2_wrapper |  41.32 |  41.32 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,893.29 |  41,253 |  6,217 |  5,507 |
| `main_cells_used     ` |  58,671,368.86 |  410,699,582 |  59,809,571 |  51,983,972 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  358.86 |  2,512 |  376 |  315 |
| `trace_gen_time_ms   ` |  856.29 |  5,994 |  968 |  745 |
| `stark_prove_excluding_trace_time_ms` |  4,678.14 |  32,747 |  4,963 |  4,296 |
| `main_trace_commit_time_ms` |  669.71 |  4,688 |  794 |  598 |
| `generate_perm_trace_time_ms` |  139.29 |  975 |  192 |  121 |
| `perm_trace_commit_time_ms` |  1,638.29 |  11,468 |  1,730 |  1,371 |
| `quotient_poly_compute_time_ms` |  768.57 |  5,380 |  798 |  740 |
| `quotient_poly_commit_time_ms` |  402 |  2,814 |  465 |  369 |
| `pcs_opening_time_ms ` |  1,057.86 |  7,405 |  1,074 |  1,027 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,761.57 |  54,331 |  8,138 |  7,628 |
| `main_cells_used     ` |  62,163,314.57 |  435,143,202 |  70,091,494 |  60,242,152 |
| `total_cycles        ` |  1,624,764.14 |  11,373,349 |  1,812,187 |  1,579,170 |
| `execute_time_ms     ` |  272.14 |  1,905 |  292 |  256 |
| `trace_gen_time_ms   ` |  1,196.57 |  8,376 |  1,232 |  1,173 |
| `stark_prove_excluding_trace_time_ms` |  6,292.86 |  44,050 |  6,614 |  6,185 |
| `main_trace_commit_time_ms` |  1,203.43 |  8,424 |  1,318 |  1,171 |
| `generate_perm_trace_time_ms` |  129.43 |  906 |  132 |  127 |
| `perm_trace_commit_time_ms` |  1,165.71 |  8,160 |  1,210 |  1,138 |
| `quotient_poly_compute_time_ms` |  1,488.29 |  10,418 |  1,563 |  1,405 |
| `quotient_poly_commit_time_ms` |  1,037.71 |  7,264 |  1,063 |  1,011 |
| `pcs_opening_time_ms ` |  1,265.43 |  8,858 |  1,330 |  1,230 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  14,620 |  58,480 |  16,746 |  8,978 |
| `main_cells_used     ` |  122,929,527 |  491,718,108 |  140,149,178 |  71,291,232 |
| `total_cycles        ` |  3,183,465.50 |  12,733,862 |  3,638,323 |  1,819,614 |
| `execute_time_ms     ` |  614.50 |  2,458 |  748 |  363 |
| `trace_gen_time_ms   ` |  2,342.75 |  9,371 |  2,657 |  1,565 |
| `stark_prove_excluding_trace_time_ms` |  11,662.75 |  46,651 |  13,403 |  7,050 |
| `main_trace_commit_time_ms` |  2,205.50 |  8,822 |  2,492 |  1,508 |
| `generate_perm_trace_time_ms` |  326.50 |  1,306 |  471 |  164 |
| `perm_trace_commit_time_ms` |  2,288.50 |  9,154 |  2,641 |  1,302 |
| `quotient_poly_compute_time_ms` |  2,749.50 |  10,998 |  3,141 |  1,608 |
| `quotient_poly_commit_time_ms` |  1,844.50 |  7,378 |  2,107 |  1,119 |
| `pcs_opening_time_ms ` |  2,246 |  8,984 |  2,576 |  1,347 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  16,603 |  33,206 |  16,827 |  16,379 |
| `main_cells_used     ` |  143,918,693.50 |  287,837,387 |  144,784,036 |  143,053,351 |
| `total_cycles        ` |  3,735,115.50 |  7,470,231 |  3,765,426 |  3,704,805 |
| `execute_time_ms     ` |  716 |  1,432 |  790 |  642 |
| `trace_gen_time_ms   ` |  2,696.50 |  5,393 |  2,710 |  2,683 |
| `stark_prove_excluding_trace_time_ms` |  13,190.50 |  26,381 |  13,354 |  13,027 |
| `main_trace_commit_time_ms` |  2,438.50 |  4,877 |  2,510 |  2,367 |
| `generate_perm_trace_time_ms` |  338.50 |  677 |  390 |  287 |
| `perm_trace_commit_time_ms` |  2,621 |  5,242 |  2,653 |  2,589 |
| `quotient_poly_compute_time_ms` |  3,131.50 |  6,263 |  3,136 |  3,127 |
| `quotient_poly_commit_time_ms` |  2,095.50 |  4,191 |  2,107 |  2,084 |
| `pcs_opening_time_ms ` |  2,563.50 |  5,127 |  2,579 |  2,548 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  16,858 |  16,858 |  16,858 |  16,858 |
| `main_cells_used     ` |  144,788,326 |  144,788,326 |  144,788,326 |  144,788,326 |
| `total_cycles        ` |  3,765,855 |  3,765,855 |  3,765,855 |  3,765,855 |
| `execute_time_ms     ` |  803 |  803 |  803 |  803 |
| `trace_gen_time_ms   ` |  2,710 |  2,710 |  2,710 |  2,710 |
| `stark_prove_excluding_trace_time_ms` |  13,345 |  13,345 |  13,345 |  13,345 |
| `main_trace_commit_time_ms` |  2,511 |  2,511 |  2,511 |  2,511 |
| `generate_perm_trace_time_ms` |  418 |  418 |  418 |  418 |
| `perm_trace_commit_time_ms` |  2,663 |  2,663 |  2,663 |  2,663 |
| `quotient_poly_compute_time_ms` |  3,103 |  3,103 |  3,103 |  3,103 |
| `quotient_poly_commit_time_ms` |  2,108 |  2,108 |  2,108 |  2,108 |
| `pcs_opening_time_ms ` |  2,539 |  2,539 |  2,539 |  2,539 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  39,712 |  39,712 |  39,712 |  39,712 |
| `main_cells_used     ` |  73,646,400 |  73,646,400 |  73,646,400 |  73,646,400 |
| `total_cycles        ` |  1,883,662 |  1,883,662 |  1,883,662 |  1,883,662 |
| `execute_time_ms     ` |  402 |  402 |  402 |  402 |
| `trace_gen_time_ms   ` |  1,702 |  1,702 |  1,702 |  1,702 |
| `stark_prove_excluding_trace_time_ms` |  37,608 |  37,608 |  37,608 |  37,608 |
| `main_trace_commit_time_ms` |  11,349 |  11,349 |  11,349 |  11,349 |
| `generate_perm_trace_time_ms` |  148 |  148 |  148 |  148 |
| `perm_trace_commit_time_ms` |  12,523 |  12,523 |  12,523 |  12,523 |
| `quotient_poly_compute_time_ms` |  1,576 |  1,576 |  1,576 |  1,576 |
| `quotient_poly_commit_time_ms` |  8,163 |  8,163 |  8,163 |  8,163 |
| `pcs_opening_time_ms ` |  3,845 |  3,845 |  3,845 |  3,845 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  132,084 |  132,084 |  132,084 |  132,084 |
| `main_cells_used     ` |  88,687,652 |  88,687,652 |  88,687,652 |  88,687,652 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  41,324 |  41,324 |  41,324 |  41,324 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 389 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 132,084 |  | 88,687,652 | 
| halo2_wrapper | 41,324 |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 3 | 262,144 |  | 16 | 11 | 7,077,888 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 3 | 131,072 |  | 16 | 13 | 3,801,088 | 
| internal.0 | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 1 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 2 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 3 | 256 |  | 16 | 17 | 8,448 | 
| internal.0 | FriReducedOpeningAir | 0 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 1 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 2 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 3 | 262,144 |  | 36 | 26 | 16,252,928 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 356 | 399 | 24,739,840 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 2 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 2 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 3 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 36 | 25 | 31,981,568 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 36 | 34 | 4,587,520 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 40 | 3,932,160 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 1 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 2 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 3 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.1 | AccessAdapterAir<8> | 4 | 512 |  | 16 | 17 | 16,896 | 
| internal.1 | AccessAdapterAir<8> | 5 | 512 |  | 16 | 17 | 16,896 | 
| internal.1 | FriReducedOpeningAir | 4 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.1 | FriReducedOpeningAir | 5 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.1 | PhantomAir | 4 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | PhantomAir | 5 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | ProgramAir | 4 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | ProgramAir | 5 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.1 | VolatileBoundaryAir | 5 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.2 | AccessAdapterAir<8> | 6 | 512 |  | 16 | 17 | 16,896 | 
| internal.2 | FriReducedOpeningAir | 6 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.2 | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.2 | ProgramAir | 6 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 1 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 2 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 3 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 4 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 5 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 1 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 2 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 3 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 4 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 5 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 1 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 2 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 3 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 4 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 5 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 6 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 1 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 2 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 4 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 5 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 6 | 32,768 |  | 8 | 6 | 458,752 | 
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
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 2 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 3 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 4 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 5 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 6 | 524,288 |  | 8 | 11 | 9,961,472 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| root | AccessAdapterAir<8> | 0 | 256 |  | 16 | 17 | 8,448 | 
| root | FriReducedOpeningAir | 0 | 262,144 |  | 36 | 26 | 16,252,928 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| root | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| root | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| root | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 32 |  | 24 | 17 | 1,312 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 20 | 32 | 13,312 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 20 | 32 | 13,312 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 32 |  | 12 | 20 | 1,024 | 
| fib_e2e | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
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
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 1 |  | 36 | 28 | 64 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 8 |  | 72 | 40 | 896 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 2,657 | 16,227 | 3,638,323 | 373,902,296 | 12,966 | 3,129 | 2,088 | 2,577 | 2,546 | 2,348 | 140,149,178 | 276 | 604 | 
| internal.0 | 1 | 2,549 | 16,529 | 3,637,902 | 373,902,296 | 13,232 | 3,141 | 2,064 | 2,641 | 2,515 | 2,474 | 140,138,244 | 395 | 748 | 
| internal.0 | 2 | 2,600 | 16,746 | 3,638,023 | 373,902,296 | 13,403 | 3,120 | 2,107 | 2,634 | 2,576 | 2,492 | 140,139,454 | 471 | 743 | 
| internal.0 | 3 | 1,565 | 8,978 | 1,819,614 | 190,491,352 | 7,050 | 1,608 | 1,119 | 1,302 | 1,347 | 1,508 | 71,291,232 | 164 | 363 | 
| internal.1 | 4 | 2,710 | 16,379 | 3,765,426 | 373,902,296 | 13,027 | 3,127 | 2,107 | 2,589 | 2,548 | 2,367 | 144,784,036 | 287 | 642 | 
| internal.1 | 5 | 2,683 | 16,827 | 3,704,805 | 373,902,296 | 13,354 | 3,136 | 2,084 | 2,653 | 2,579 | 2,510 | 143,053,351 | 390 | 790 | 
| internal.2 | 6 | 2,710 | 16,858 | 3,765,855 | 373,902,296 | 13,345 | 3,103 | 2,108 | 2,663 | 2,539 | 2,511 | 144,788,326 | 418 | 803 | 
| leaf | 0 | 1,232 | 8,138 | 1,812,187 | 180,005,592 | 6,614 | 1,563 | 1,058 | 1,210 | 1,330 | 1,318 | 70,091,494 | 132 | 292 | 
| leaf | 1 | 1,199 | 7,723 | 1,579,226 | 170,306,264 | 6,260 | 1,509 | 1,040 | 1,152 | 1,230 | 1,199 | 60,242,712 | 128 | 264 | 
| leaf | 2 | 1,197 | 7,699 | 1,579,270 | 170,306,264 | 6,246 | 1,473 | 1,036 | 1,184 | 1,249 | 1,172 | 60,243,152 | 129 | 256 | 
| leaf | 3 | 1,182 | 7,649 | 1,579,170 | 170,306,264 | 6,203 | 1,483 | 1,022 | 1,151 | 1,240 | 1,177 | 60,242,152 | 127 | 264 | 
| leaf | 4 | 1,176 | 7,690 | 1,579,501 | 170,306,264 | 6,240 | 1,514 | 1,011 | 1,146 | 1,249 | 1,185 | 60,245,462 | 132 | 274 | 
| leaf | 5 | 1,173 | 7,628 | 1,579,570 | 170,306,264 | 6,185 | 1,471 | 1,034 | 1,138 | 1,238 | 1,171 | 60,246,152 | 130 | 270 | 
| leaf | 6 | 1,217 | 7,804 | 1,664,425 | 177,720,280 | 6,302 | 1,405 | 1,063 | 1,179 | 1,322 | 1,202 | 63,832,078 | 128 | 285 | 
| root | 0 | 1,702 | 39,712 | 1,883,662 | 190,491,352 | 37,608 | 1,576 | 8,163 | 12,523 | 3,845 | 11,349 | 73,646,400 | 148 | 402 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 901 | 6,217 | 1,747,603 | 197,440,542 | 4,963 | 762 | 461 | 1,682 | 1,069 | 794 | 59,809,571 | 192 | 353 | 
| fib_e2e | 1 | 968 | 5,865 | 1,747,502 | 197,384,386 | 4,528 | 740 | 374 | 1,559 | 1,053 | 677 | 59,780,906 | 123 | 369 | 
| fib_e2e | 2 | 846 | 5,854 | 1,747,502 | 197,384,386 | 4,632 | 753 | 369 | 1,730 | 1,027 | 618 | 59,780,897 | 133 | 376 | 
| fib_e2e | 3 | 845 | 5,507 | 1,747,502 | 197,384,386 | 4,296 | 759 | 375 | 1,371 | 1,069 | 598 | 59,781,216 | 123 | 366 | 
| fib_e2e | 4 | 852 | 5,968 | 1,747,502 | 197,384,386 | 4,751 | 793 | 465 | 1,721 | 1,044 | 603 | 59,781,515 | 121 | 365 | 
| fib_e2e | 5 | 837 | 5,994 | 1,747,502 | 197,384,386 | 4,789 | 798 | 386 | 1,689 | 1,074 | 690 | 59,781,505 | 149 | 368 | 
| fib_e2e | 6 | 745 | 5,848 | 1,515,024 | 197,432,594 | 4,788 | 775 | 384 | 1,716 | 1,069 | 708 | 51,983,972 | 134 | 315 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b60b956c6aae904aa274e7ee9afdb0ba6ca43686

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12937293626)
