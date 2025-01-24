| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  415.39 |  276.95 |
| fib_e2e |  41.06 |  6.29 |
| leaf |  54.21 |  8.09 |
| internal.0 |  57.81 |  16.42 |
| internal.1 |  32.67 |  16.50 |
| internal.2 |  16.74 |  16.74 |
| root |  39.62 |  39.62 |
| halo2_outer |  132.48 |  132.48 |
| halo2_wrapper |  40.80 |  40.80 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,865.71 |  41,060 |  6,291 |  5,436 |
| `main_cells_used     ` |  58,671,368.86 |  410,699,582 |  59,809,571 |  51,983,972 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  362.86 |  2,540 |  377 |  318 |
| `trace_gen_time_ms   ` |  848.14 |  5,937 |  923 |  741 |
| `stark_prove_excluding_trace_time_ms` |  4,654.71 |  32,583 |  5,030 |  4,210 |
| `main_trace_commit_time_ms` |  645.14 |  4,516 |  793 |  571 |
| `generate_perm_trace_time_ms` |  148.71 |  1,041 |  185 |  123 |
| `perm_trace_commit_time_ms` |  1,626.86 |  11,388 |  1,766 |  1,279 |
| `quotient_poly_compute_time_ms` |  797.14 |  5,580 |  865 |  716 |
| `quotient_poly_commit_time_ms` |  413.29 |  2,893 |  542 |  367 |
| `pcs_opening_time_ms ` |  1,020.86 |  7,146 |  1,058 |  990 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,744.57 |  54,212 |  8,090 |  7,638 |
| `main_cells_used     ` |  62,163,836 |  435,146,852 |  70,090,544 |  60,241,592 |
| `total_cycles        ` |  1,624,816.29 |  11,373,714 |  1,812,092 |  1,579,114 |
| `execute_time_ms     ` |  266.43 |  1,865 |  287 |  251 |
| `trace_gen_time_ms   ` |  1,219 |  8,533 |  1,279 |  1,191 |
| `stark_prove_excluding_trace_time_ms` |  6,259.14 |  43,814 |  6,524 |  6,170 |
| `main_trace_commit_time_ms` |  1,202.43 |  8,417 |  1,275 |  1,179 |
| `generate_perm_trace_time_ms` |  130.71 |  915 |  146 |  125 |
| `perm_trace_commit_time_ms` |  1,153.71 |  8,076 |  1,204 |  1,131 |
| `quotient_poly_compute_time_ms` |  1,481.43 |  10,370 |  1,562 |  1,341 |
| `quotient_poly_commit_time_ms` |  1,040.14 |  7,281 |  1,108 |  1,009 |
| `pcs_opening_time_ms ` |  1,248 |  8,736 |  1,303 |  1,213 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  14,452.75 |  57,811 |  16,417 |  8,953 |
| `main_cells_used     ` |  122,929,707 |  491,718,828 |  140,146,368 |  71,293,162 |
| `total_cycles        ` |  3,183,483.50 |  12,733,934 |  3,638,276 |  1,819,807 |
| `execute_time_ms     ` |  624.75 |  2,499 |  767 |  366 |
| `trace_gen_time_ms   ` |  2,333 |  9,332 |  2,620 |  1,557 |
| `stark_prove_excluding_trace_time_ms` |  11,495 |  45,980 |  13,043 |  7,030 |
| `main_trace_commit_time_ms` |  2,311.50 |  9,246 |  2,695 |  1,506 |
| `generate_perm_trace_time_ms` |  246.25 |  985 |  283 |  147 |
| `perm_trace_commit_time_ms` |  2,255.25 |  9,021 |  2,576 |  1,322 |
| `quotient_poly_compute_time_ms` |  2,654 |  10,616 |  3,011 |  1,611 |
| `quotient_poly_commit_time_ms` |  1,839.25 |  7,357 |  2,090 |  1,114 |
| `pcs_opening_time_ms ` |  2,185.75 |  8,743 |  2,560 |  1,326 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  16,333.50 |  32,667 |  16,505 |  16,162 |
| `main_cells_used     ` |  143,917,153.50 |  287,834,307 |  144,783,016 |  143,051,291 |
| `total_cycles        ` |  3,734,961.50 |  7,469,923 |  3,765,324 |  3,704,599 |
| `execute_time_ms     ` |  714.50 |  1,429 |  781 |  648 |
| `trace_gen_time_ms   ` |  2,705.50 |  5,411 |  2,737 |  2,674 |
| `stark_prove_excluding_trace_time_ms` |  12,913.50 |  25,827 |  13,050 |  12,777 |
| `main_trace_commit_time_ms` |  2,548 |  5,096 |  2,721 |  2,375 |
| `generate_perm_trace_time_ms` |  283 |  566 |  288 |  278 |
| `perm_trace_commit_time_ms` |  2,561.50 |  5,123 |  2,562 |  2,561 |
| `quotient_poly_compute_time_ms` |  3,007.50 |  6,015 |  3,033 |  2,982 |
| `quotient_poly_commit_time_ms` |  2,087 |  4,174 |  2,094 |  2,080 |
| `pcs_opening_time_ms ` |  2,423 |  4,846 |  2,424 |  2,422 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  16,744 |  16,744 |  16,744 |  16,744 |
| `main_cells_used     ` |  144,784,836 |  144,784,836 |  144,784,836 |  144,784,836 |
| `total_cycles        ` |  3,765,506 |  3,765,506 |  3,765,506 |  3,765,506 |
| `execute_time_ms     ` |  797 |  797 |  797 |  797 |
| `trace_gen_time_ms   ` |  2,687 |  2,687 |  2,687 |  2,687 |
| `stark_prove_excluding_trace_time_ms` |  13,260 |  13,260 |  13,260 |  13,260 |
| `main_trace_commit_time_ms` |  2,704 |  2,704 |  2,704 |  2,704 |
| `generate_perm_trace_time_ms` |  288 |  288 |  288 |  288 |
| `perm_trace_commit_time_ms` |  2,567 |  2,567 |  2,567 |  2,567 |
| `quotient_poly_compute_time_ms` |  3,026 |  3,026 |  3,026 |  3,026 |
| `quotient_poly_commit_time_ms` |  2,095 |  2,095 |  2,095 |  2,095 |
| `pcs_opening_time_ms ` |  2,577 |  2,577 |  2,577 |  2,577 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  39,618 |  39,618 |  39,618 |  39,618 |
| `main_cells_used     ` |  73,644,060 |  73,644,060 |  73,644,060 |  73,644,060 |
| `total_cycles        ` |  1,883,428 |  1,883,428 |  1,883,428 |  1,883,428 |
| `execute_time_ms     ` |  399 |  399 |  399 |  399 |
| `trace_gen_time_ms   ` |  1,684 |  1,684 |  1,684 |  1,684 |
| `stark_prove_excluding_trace_time_ms` |  37,535 |  37,535 |  37,535 |  37,535 |
| `main_trace_commit_time_ms` |  11,349 |  11,349 |  11,349 |  11,349 |
| `generate_perm_trace_time_ms` |  152 |  152 |  152 |  152 |
| `perm_trace_commit_time_ms` |  12,536 |  12,536 |  12,536 |  12,536 |
| `quotient_poly_compute_time_ms` |  1,552 |  1,552 |  1,552 |  1,552 |
| `quotient_poly_commit_time_ms` |  8,134 |  8,134 |  8,134 |  8,134 |
| `pcs_opening_time_ms ` |  3,809 |  3,809 |  3,809 |  3,809 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  132,484 |  132,484 |  132,484 |  132,484 |
| `main_cells_used     ` |  88,687,652 |  88,687,652 |  88,687,652 |  88,687,652 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  40,798 |  40,798 |  40,798 |  40,798 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 386 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 132,484 |  | 88,687,652 | 
| halo2_wrapper | 40,798 |  |  | 

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
| internal.0 | 0 | 2,599 | 16,093 | 3,638,042 | 373,902,296 | 12,877 | 3,011 | 2,090 | 2,576 | 2,560 | 2,357 | 140,146,368 | 280 | 617 | 
| internal.0 | 1 | 2,556 | 16,348 | 3,638,276 | 373,902,296 | 13,043 | 3,003 | 2,082 | 2,564 | 2,421 | 2,695 | 140,141,984 | 275 | 749 | 
| internal.0 | 2 | 2,620 | 16,417 | 3,637,809 | 373,902,296 | 13,030 | 2,991 | 2,071 | 2,559 | 2,436 | 2,688 | 140,137,314 | 283 | 767 | 
| internal.0 | 3 | 1,557 | 8,953 | 1,819,807 | 190,491,352 | 7,030 | 1,611 | 1,114 | 1,322 | 1,326 | 1,506 | 71,293,162 | 147 | 366 | 
| internal.1 | 4 | 2,737 | 16,162 | 3,765,324 | 373,902,296 | 12,777 | 3,033 | 2,094 | 2,561 | 2,422 | 2,375 | 144,783,016 | 288 | 648 | 
| internal.1 | 5 | 2,674 | 16,505 | 3,704,599 | 373,902,296 | 13,050 | 2,982 | 2,080 | 2,562 | 2,424 | 2,721 | 143,051,291 | 278 | 781 | 
| internal.2 | 6 | 2,687 | 16,744 | 3,765,506 | 373,902,296 | 13,260 | 3,026 | 2,095 | 2,567 | 2,577 | 2,704 | 144,784,836 | 288 | 797 | 
| leaf | 0 | 1,279 | 8,090 | 1,812,092 | 180,005,592 | 6,524 | 1,562 | 1,049 | 1,204 | 1,298 | 1,275 | 70,090,544 | 133 | 287 | 
| leaf | 1 | 1,194 | 7,648 | 1,579,194 | 170,306,264 | 6,203 | 1,499 | 1,010 | 1,132 | 1,232 | 1,199 | 60,242,392 | 129 | 251 | 
| leaf | 2 | 1,204 | 7,656 | 1,579,114 | 170,306,264 | 6,189 | 1,497 | 1,009 | 1,149 | 1,223 | 1,180 | 60,241,592 | 129 | 263 | 
| leaf | 3 | 1,207 | 7,706 | 1,579,158 | 170,306,264 | 6,244 | 1,513 | 1,034 | 1,149 | 1,240 | 1,179 | 60,242,032 | 125 | 255 | 
| leaf | 4 | 1,191 | 7,647 | 1,579,539 | 170,306,264 | 6,191 | 1,478 | 1,035 | 1,137 | 1,227 | 1,185 | 60,245,842 | 126 | 265 | 
| leaf | 5 | 1,207 | 7,638 | 1,579,570 | 170,306,264 | 6,170 | 1,480 | 1,036 | 1,131 | 1,213 | 1,182 | 60,246,152 | 127 | 261 | 
| leaf | 6 | 1,251 | 7,827 | 1,665,047 | 177,720,280 | 6,293 | 1,341 | 1,108 | 1,174 | 1,303 | 1,217 | 63,838,298 | 146 | 283 | 
| root | 0 | 1,684 | 39,618 | 1,883,428 | 190,491,352 | 37,535 | 1,552 | 8,134 | 12,536 | 3,809 | 11,349 | 73,644,060 | 152 | 399 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 888 | 6,291 | 1,747,603 | 197,440,542 | 5,030 | 857 | 542 | 1,656 | 995 | 793 | 59,809,571 | 185 | 373 | 
| fib_e2e | 1 | 923 | 5,766 | 1,747,502 | 197,384,386 | 4,474 | 807 | 385 | 1,564 | 1,020 | 571 | 59,780,906 | 123 | 369 | 
| fib_e2e | 2 | 846 | 6,048 | 1,747,502 | 197,384,386 | 4,825 | 865 | 380 | 1,766 | 1,031 | 649 | 59,780,897 | 132 | 377 | 
| fib_e2e | 3 | 859 | 5,436 | 1,747,502 | 197,384,386 | 4,210 | 719 | 367 | 1,279 | 1,048 | 619 | 59,781,216 | 176 | 367 | 
| fib_e2e | 4 | 844 | 5,791 | 1,747,502 | 197,384,386 | 4,581 | 716 | 385 | 1,652 | 1,058 | 632 | 59,781,515 | 133 | 366 | 
| fib_e2e | 5 | 836 | 5,997 | 1,747,502 | 197,384,386 | 4,791 | 824 | 445 | 1,743 | 990 | 663 | 59,781,505 | 124 | 370 | 
| fib_e2e | 6 | 741 | 5,731 | 1,515,024 | 197,432,594 | 4,672 | 792 | 389 | 1,728 | 1,004 | 589 | 51,983,972 | 168 | 318 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/77fbcdeba97af790aef062cfd2cd125d13c125ae

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12943210977)
