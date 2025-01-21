| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  531.43 |  372.31 |
| fib_e2e |  42.63 |  6.26 |
| leaf |  54.42 |  8.32 |
| internal.0 |  76.93 |  22.13 |
| internal.1 |  45.82 |  23.97 |
| internal.2 |  24.32 |  24.32 |
| internal.3 |  12.86 |  12.86 |
| root |  54.54 |  54.54 |
| halo2_outer |  177.75 |  177.75 |
| halo2_wrapper |  42.17 |  42.17 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,089.43 |  42,626 |  6,262 |  5,941 |
| `main_cells_used     ` |  58,688,632.86 |  410,820,430 |  59,826,835 |  52,001,236 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  345.43 |  2,418 |  361 |  313 |
| `trace_gen_time_ms   ` |  886.14 |  6,203 |  997 |  780 |
| `stark_prove_excluding_trace_time_ms` |  4,857.86 |  34,005 |  4,982 |  4,781 |
| `main_trace_commit_time_ms` |  647 |  4,529 |  786 |  598 |
| `generate_perm_trace_time_ms` |  143.43 |  1,004 |  199 |  117 |
| `perm_trace_commit_time_ms` |  1,807.43 |  12,652 |  1,901 |  1,640 |
| `quotient_poly_compute_time_ms` |  796.86 |  5,578 |  850 |  732 |
| `quotient_poly_commit_time_ms` |  450 |  3,150 |  531 |  381 |
| `pcs_opening_time_ms ` |  1,010.43 |  7,073 |  1,055 |  992 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,774.43 |  54,421 |  8,319 |  7,566 |
| `main_cells_used     ` |  68,592,328.71 |  480,146,301 |  78,853,998 |  66,097,823 |
| `total_cycles        ` |  1,763,616.86 |  12,345,318 |  1,996,096 |  1,706,267 |
| `execute_time_ms     ` |  324.57 |  2,272 |  356 |  308 |
| `trace_gen_time_ms   ` |  1,128.57 |  7,900 |  1,392 |  1,056 |
| `stark_prove_excluding_trace_time_ms` |  6,321.29 |  44,249 |  6,571 |  6,198 |
| `main_trace_commit_time_ms` |  1,269.57 |  8,887 |  1,283 |  1,247 |
| `generate_perm_trace_time_ms` |  134.29 |  940 |  144 |  129 |
| `perm_trace_commit_time_ms` |  1,185 |  8,295 |  1,207 |  1,163 |
| `quotient_poly_compute_time_ms` |  1,533.57 |  10,735 |  1,656 |  1,477 |
| `quotient_poly_commit_time_ms` |  973.71 |  6,816 |  1,054 |  928 |
| `pcs_opening_time_ms ` |  1,222.43 |  8,557 |  1,276 |  1,197 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  19,232.25 |  76,929 |  22,128 |  11,893 |
| `main_cells_used     ` |  153,553,125.75 |  614,212,503 |  175,219,831 |  88,891,592 |
| `total_cycles        ` |  4,023,055.75 |  16,092,223 |  4,600,918 |  2,304,351 |
| `execute_time_ms     ` |  845.50 |  3,382 |  1,039 |  484 |
| `trace_gen_time_ms   ` |  2,565.25 |  10,261 |  2,930 |  1,557 |
| `stark_prove_excluding_trace_time_ms` |  15,821.50 |  63,286 |  18,159 |  9,852 |
| `main_trace_commit_time_ms` |  3,080.75 |  12,323 |  3,739 |  1,902 |
| `generate_perm_trace_time_ms` |  416.25 |  1,665 |  577 |  360 |
| `perm_trace_commit_time_ms` |  2,821.50 |  11,286 |  3,226 |  1,686 |
| `quotient_poly_compute_time_ms` |  3,526.25 |  14,105 |  3,970 |  2,238 |
| `quotient_poly_commit_time_ms` |  2,381.25 |  9,525 |  2,722 |  1,420 |
| `pcs_opening_time_ms ` |  3,592.25 |  14,369 |  4,155 |  2,025 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  22,911 |  45,822 |  23,968 |  21,854 |
| `main_cells_used     ` |  184,687,487.50 |  369,374,975 |  185,922,336 |  183,452,639 |
| `total_cycles        ` |  4,866,522.50 |  9,733,045 |  4,908,500 |  4,824,545 |
| `execute_time_ms     ` |  954.50 |  1,909 |  960 |  949 |
| `trace_gen_time_ms   ` |  2,984.50 |  5,969 |  3,084 |  2,885 |
| `stark_prove_excluding_trace_time_ms` |  18,972 |  37,944 |  20,123 |  17,821 |
| `main_trace_commit_time_ms` |  3,593.50 |  7,187 |  3,610 |  3,577 |
| `generate_perm_trace_time_ms` |  384 |  768 |  412 |  356 |
| `perm_trace_commit_time_ms` |  3,485.50 |  6,971 |  3,799 |  3,172 |
| `quotient_poly_compute_time_ms` |  4,310 |  8,620 |  4,655 |  3,965 |
| `quotient_poly_commit_time_ms` |  2,909.50 |  5,819 |  3,174 |  2,645 |
| `pcs_opening_time_ms ` |  4,288 |  8,576 |  4,472 |  4,104 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  24,316 |  24,316 |  24,316 |  24,316 |
| `main_cells_used     ` |  186,076,569 |  186,076,569 |  186,076,569 |  186,076,569 |
| `total_cycles        ` |  4,911,823 |  4,911,823 |  4,911,823 |  4,911,823 |
| `execute_time_ms     ` |  961 |  961 |  961 |  961 |
| `trace_gen_time_ms   ` |  3,110 |  3,110 |  3,110 |  3,110 |
| `stark_prove_excluding_trace_time_ms` |  20,245 |  20,245 |  20,245 |  20,245 |
| `main_trace_commit_time_ms` |  3,963 |  3,963 |  3,963 |  3,963 |
| `generate_perm_trace_time_ms` |  405 |  405 |  405 |  405 |
| `perm_trace_commit_time_ms` |  3,805 |  3,805 |  3,805 |  3,805 |
| `quotient_poly_compute_time_ms` |  4,525 |  4,525 |  4,525 |  4,525 |
| `quotient_poly_commit_time_ms` |  3,096 |  3,096 |  3,096 |  3,096 |
| `pcs_opening_time_ms ` |  4,448 |  4,448 |  4,448 |  4,448 |

| internal.3 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  12,858 |  12,858 |  12,858 |  12,858 |
| `main_cells_used     ` |  94,286,204 |  94,286,204 |  94,286,204 |  94,286,204 |
| `total_cycles        ` |  2,456,347 |  2,456,347 |  2,456,347 |  2,456,347 |
| `execute_time_ms     ` |  542 |  542 |  542 |  542 |
| `trace_gen_time_ms   ` |  2,360 |  2,360 |  2,360 |  2,360 |
| `stark_prove_excluding_trace_time_ms` |  9,956 |  9,956 |  9,956 |  9,956 |
| `main_trace_commit_time_ms` |  1,772 |  1,772 |  1,772 |  1,772 |
| `generate_perm_trace_time_ms` |  211 |  211 |  211 |  211 |
| `perm_trace_commit_time_ms` |  1,899 |  1,899 |  1,899 |  1,899 |
| `quotient_poly_compute_time_ms` |  2,327 |  2,327 |  2,327 |  2,327 |
| `quotient_poly_commit_time_ms` |  1,582 |  1,582 |  1,582 |  1,582 |
| `pcs_opening_time_ms ` |  2,163 |  2,163 |  2,163 |  2,163 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  54,537 |  54,537 |  54,537 |  54,537 |
| `main_cells_used     ` |  91,381,444 |  91,381,444 |  91,381,444 |  91,381,444 |
| `total_cycles        ` |  2,370,583 |  2,370,583 |  2,370,583 |  2,370,583 |
| `execute_time_ms     ` |  521 |  521 |  521 |  521 |
| `trace_gen_time_ms   ` |  1,626 |  1,626 |  1,626 |  1,626 |
| `stark_prove_excluding_trace_time_ms` |  52,390 |  52,390 |  52,390 |  52,390 |
| `main_trace_commit_time_ms` |  15,678 |  15,678 |  15,678 |  15,678 |
| `generate_perm_trace_time_ms` |  189 |  189 |  189 |  189 |
| `perm_trace_commit_time_ms` |  16,484 |  16,484 |  16,484 |  16,484 |
| `quotient_poly_compute_time_ms` |  2,181 |  2,181 |  2,181 |  2,181 |
| `quotient_poly_commit_time_ms` |  10,970 |  10,970 |  10,970 |  10,970 |
| `pcs_opening_time_ms ` |  6,884 |  6,884 |  6,884 |  6,884 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  177,749 |  177,749 |  177,749 |  177,749 |
| `main_cells_used     ` |  121,401,931 |  121,401,931 |  121,401,931 |  121,401,931 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  42,173 |  42,173 |  42,173 |  42,173 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 474 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 177,749 |  | 121,401,931 | 
| halo2_wrapper | 42,173 |  |  | 

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
| internal.0 | FriReducedOpeningAir | 0 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 1 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 2 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
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
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
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
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 76 | 64 | 36,700,160 | 
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
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
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
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.2 | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.2 | ProgramAir | 6 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.3 | AccessAdapterAir<2> | 7 | 262,144 |  | 16 | 11 | 7,077,888 | 
| internal.3 | AccessAdapterAir<4> | 7 | 131,072 |  | 16 | 13 | 3,801,088 | 
| internal.3 | AccessAdapterAir<8> | 7 | 256 |  | 16 | 17 | 8,448 | 
| internal.3 | FriReducedOpeningAir | 7 | 131,072 |  | 76 | 64 | 18,350,080 | 
| internal.3 | NativePoseidon2Air<BabyBearParameters>, 1> | 7 | 32,768 |  | 356 | 399 | 24,739,840 | 
| internal.3 | PhantomAir | 7 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.3 | ProgramAir | 7 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.3 | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.3 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 7 | 524,288 |  | 28 | 23 | 26,738,688 | 
| internal.3 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 7 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.3 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 7 | 64 |  | 16 | 23 | 2,496 | 
| internal.3 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 7 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.3 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 7 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.3 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 7 | 65,536 |  | 36 | 34 | 4,587,520 | 
| internal.3 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 7 | 65,536 |  | 20 | 40 | 3,932,160 | 
| internal.3 | VmConnectorAir | 7 | 2 | 1 | 8 | 4 | 24 | 
| internal.3 | VolatileBoundaryAir | 7 | 524,288 |  | 8 | 11 | 9,961,472 | 
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
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 76 | 64 | 18,350,080 | 
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
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 32,768 |  | 20 | 40 | 1,966,080 | 
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
| root | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| root | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| root | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
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
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 256 |  | 8 | 300 | 78,848 | 
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
| internal.0 | 0 | 2,846 | 20,876 | 4,600,918 | 485,837,784 | 17,170 | 3,938 | 2,689 | 3,164 | 4,049 | 2,962 | 175,219,831 | 365 | 860 | 
| internal.0 | 1 | 2,928 | 22,032 | 4,593,091 | 485,837,784 | 18,105 | 3,959 | 2,694 | 3,210 | 4,140 | 3,739 | 175,046,680 | 360 | 999 | 
| internal.0 | 2 | 2,930 | 22,128 | 4,593,863 | 485,837,784 | 18,159 | 3,970 | 2,722 | 3,226 | 4,155 | 3,720 | 175,054,400 | 363 | 1,039 | 
| internal.0 | 3 | 1,557 | 11,893 | 2,304,351 | 246,459,096 | 9,852 | 2,238 | 1,420 | 1,686 | 2,025 | 1,902 | 88,891,592 | 577 | 484 | 
| internal.1 | 4 | 2,885 | 23,968 | 4,908,500 | 549,800,920 | 20,123 | 4,655 | 3,174 | 3,799 | 4,472 | 3,610 | 185,922,336 | 412 | 960 | 
| internal.1 | 5 | 3,084 | 21,854 | 4,824,545 | 485,837,784 | 17,821 | 3,965 | 2,645 | 3,172 | 4,104 | 3,577 | 183,452,639 | 356 | 949 | 
| internal.2 | 6 | 3,110 | 24,316 | 4,911,823 | 549,800,920 | 20,245 | 4,525 | 3,096 | 3,805 | 4,448 | 3,963 | 186,076,569 | 405 | 961 | 
| internal.3 | 7 | 2,360 | 12,858 | 2,456,347 | 278,440,664 | 9,956 | 2,327 | 1,582 | 1,899 | 2,163 | 1,772 | 94,286,204 | 211 | 542 | 
| leaf | 0 | 1,392 | 8,319 | 1,996,096 | 188,263,128 | 6,571 | 1,656 | 1,054 | 1,200 | 1,276 | 1,249 | 78,853,998 | 132 | 356 | 
| leaf | 1 | 1,063 | 7,612 | 1,706,815 | 180,529,880 | 6,241 | 1,489 | 966 | 1,207 | 1,197 | 1,247 | 66,103,303 | 133 | 308 | 
| leaf | 2 | 1,056 | 7,566 | 1,707,384 | 180,529,880 | 6,198 | 1,480 | 928 | 1,163 | 1,206 | 1,274 | 66,108,993 | 144 | 312 | 
| leaf | 3 | 1,073 | 7,649 | 1,707,207 | 180,529,880 | 6,261 | 1,505 | 948 | 1,177 | 1,210 | 1,283 | 66,107,223 | 136 | 315 | 
| leaf | 4 | 1,067 | 7,641 | 1,706,267 | 180,529,880 | 6,256 | 1,487 | 952 | 1,176 | 1,227 | 1,276 | 66,097,823 | 136 | 318 | 
| leaf | 5 | 1,101 | 7,641 | 1,707,448 | 180,529,880 | 6,213 | 1,477 | 949 | 1,174 | 1,204 | 1,277 | 66,109,633 | 129 | 327 | 
| leaf | 6 | 1,148 | 7,993 | 1,814,101 | 185,977,816 | 6,509 | 1,641 | 1,019 | 1,198 | 1,237 | 1,281 | 70,765,328 | 130 | 336 | 
| root | 0 | 1,626 | 54,537 | 2,370,583 | 246,459,096 | 52,390 | 2,181 | 10,970 | 16,484 | 6,884 | 15,678 | 91,381,444 | 189 | 521 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 929 | 6,262 | 1,747,603 | 197,440,542 | 4,982 | 817 | 531 | 1,640 | 1,006 | 786 | 59,826,835 | 199 | 351 | 
| fib_e2e | 1 | 906 | 6,056 | 1,747,502 | 197,423,810 | 4,834 | 815 | 442 | 1,814 | 1,004 | 623 | 59,798,170 | 133 | 316 | 
| fib_e2e | 2 | 997 | 6,161 | 1,747,502 | 197,423,810 | 4,803 | 802 | 438 | 1,803 | 992 | 628 | 59,798,161 | 137 | 361 | 
| fib_e2e | 3 | 859 | 6,135 | 1,747,502 | 197,423,810 | 4,915 | 806 | 458 | 1,901 | 1,011 | 616 | 59,798,480 | 120 | 361 | 
| fib_e2e | 4 | 861 | 5,999 | 1,747,502 | 197,423,810 | 4,781 | 732 | 381 | 1,884 | 1,055 | 609 | 59,798,779 | 117 | 357 | 
| fib_e2e | 5 | 871 | 6,072 | 1,747,502 | 197,423,810 | 4,842 | 850 | 496 | 1,786 | 993 | 598 | 59,798,769 | 117 | 359 | 
| fib_e2e | 6 | 780 | 5,941 | 1,515,024 | 197,432,594 | 4,848 | 756 | 404 | 1,824 | 1,012 | 669 | 52,001,236 | 181 | 313 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/3c5c4a34ee5a0f5f91fb9ff5b57fb6cbd8ce773e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12881616966)
