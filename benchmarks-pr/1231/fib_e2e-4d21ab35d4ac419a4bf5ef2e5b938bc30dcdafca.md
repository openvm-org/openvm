| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  799.50 |  589.27 |
| fib_e2e |  43.26 |  6.41 |
| leaf |  91.33 |  13.94 |
| internal.0 |  96.44 |  27.58 |
| internal.1 |  54.93 |  27.80 |
| internal.2 |  27.78 |  27.78 |
| root |  67.51 |  67.51 |
| halo2_outer |  341.20 |  341.20 |
| halo2_wrapper |  77.05 |  77.05 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,180.43 |  43,263 |  6,407 |  5,981 |
| `main_cells_used     ` |  58,688,632.86 |  410,820,430 |  59,826,835 |  52,001,236 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  342.29 |  2,396 |  357 |  307 |
| `trace_gen_time_ms   ` |  904.43 |  6,331 |  1,049 |  760 |
| `stark_prove_excluding_trace_time_ms` |  4,933.71 |  34,536 |  5,047 |  4,865 |
| `main_trace_commit_time_ms` |  730.71 |  5,115 |  811 |  631 |
| `generate_perm_trace_time_ms` |  136.29 |  954 |  202 |  120 |
| `perm_trace_commit_time_ms` |  1,780.14 |  12,461 |  1,864 |  1,671 |
| `quotient_poly_compute_time_ms` |  837.71 |  5,864 |  861 |  784 |
| `quotient_poly_commit_time_ms` |  430.86 |  3,016 |  470 |  388 |
| `pcs_opening_time_ms ` |  1,015.43 |  7,108 |  1,061 |  958 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  13,047 |  91,329 |  13,941 |  12,756 |
| `main_cells_used     ` |  94,454,107 |  661,178,749 |  109,820,759 |  90,793,265 |
| `total_cycles        ` |  2,640,018 |  18,480,126 |  3,056,781 |  2,540,783 |
| `execute_time_ms     ` |  453.43 |  3,174 |  592 |  419 |
| `trace_gen_time_ms   ` |  1,896.71 |  13,277 |  2,168 |  1,818 |
| `stark_prove_excluding_trace_time_ms` |  10,696.86 |  74,878 |  11,181 |  10,497 |
| `main_trace_commit_time_ms` |  1,965.57 |  13,759 |  2,053 |  1,926 |
| `generate_perm_trace_time_ms` |  223.71 |  1,566 |  238 |  219 |
| `perm_trace_commit_time_ms` |  1,999.29 |  13,995 |  2,073 |  1,966 |
| `quotient_poly_compute_time_ms` |  2,428.57 |  17,000 |  2,553 |  2,391 |
| `quotient_poly_commit_time_ms` |  1,812.86 |  12,690 |  1,882 |  1,768 |
| `pcs_opening_time_ms ` |  2,264 |  15,848 |  2,393 |  2,205 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  24,108.75 |  96,435 |  27,580 |  14,466 |
| `main_cells_used     ` |  197,079,031.25 |  788,316,125 |  224,963,607 |  113,543,044 |
| `total_cycles        ` |  5,728,204.25 |  22,912,817 |  6,549,955 |  3,275,347 |
| `execute_time_ms     ` |  1,200.75 |  4,803 |  1,480 |  534 |
| `trace_gen_time_ms   ` |  3,818.75 |  15,275 |  4,366 |  2,715 |
| `stark_prove_excluding_trace_time_ms` |  19,089.25 |  76,357 |  21,796 |  11,217 |
| `main_trace_commit_time_ms` |  3,563.75 |  14,255 |  4,056 |  2,181 |
| `generate_perm_trace_time_ms` |  392.25 |  1,569 |  451 |  239 |
| `perm_trace_commit_time_ms` |  3,650.75 |  14,603 |  4,225 |  2,103 |
| `quotient_poly_compute_time_ms` |  4,125.25 |  16,501 |  4,709 |  2,476 |
| `quotient_poly_commit_time_ms` |  3,190.50 |  12,762 |  3,639 |  1,870 |
| `pcs_opening_time_ms ` |  4,164.25 |  16,657 |  4,777 |  2,345 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  27,463.50 |  54,927 |  27,798 |  27,129 |
| `main_cells_used     ` |  231,080,938.50 |  462,161,877 |  232,325,284 |  229,836,593 |
| `total_cycles        ` |  6,716,515 |  13,433,030 |  6,770,999 |  6,662,031 |
| `execute_time_ms     ` |  1,428 |  2,856 |  1,503 |  1,353 |
| `trace_gen_time_ms   ` |  4,193 |  8,386 |  4,452 |  3,934 |
| `stark_prove_excluding_trace_time_ms` |  21,842.50 |  43,685 |  21,843 |  21,842 |
| `main_trace_commit_time_ms` |  4,062 |  8,124 |  4,067 |  4,057 |
| `generate_perm_trace_time_ms` |  447.50 |  895 |  450 |  445 |
| `perm_trace_commit_time_ms` |  4,148.50 |  8,297 |  4,208 |  4,089 |
| `quotient_poly_compute_time_ms` |  4,761.50 |  9,523 |  4,798 |  4,725 |
| `quotient_poly_commit_time_ms` |  3,633 |  7,266 |  3,636 |  3,630 |
| `pcs_opening_time_ms ` |  4,788 |  9,576 |  4,805 |  4,771 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  27,779 |  27,779 |  27,779 |  27,779 |
| `main_cells_used     ` |  232,328,024 |  232,328,024 |  232,328,024 |  232,328,024 |
| `total_cycles        ` |  6,771,273 |  6,771,273 |  6,771,273 |  6,771,273 |
| `execute_time_ms     ` |  1,532 |  1,532 |  1,532 |  1,532 |
| `trace_gen_time_ms   ` |  4,480 |  4,480 |  4,480 |  4,480 |
| `stark_prove_excluding_trace_time_ms` |  21,767 |  21,767 |  21,767 |  21,767 |
| `main_trace_commit_time_ms` |  4,066 |  4,066 |  4,066 |  4,066 |
| `generate_perm_trace_time_ms` |  443 |  443 |  443 |  443 |
| `perm_trace_commit_time_ms` |  4,136 |  4,136 |  4,136 |  4,136 |
| `quotient_poly_compute_time_ms` |  4,673 |  4,673 |  4,673 |  4,673 |
| `quotient_poly_commit_time_ms` |  3,641 |  3,641 |  3,641 |  3,641 |
| `pcs_opening_time_ms ` |  4,806 |  4,806 |  4,806 |  4,806 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  67,510 |  67,510 |  67,510 |  67,510 |
| `main_cells_used     ` |  117,281,351 |  117,281,351 |  117,281,351 |  117,281,351 |
| `total_cycles        ` |  3,387,164 |  3,387,164 |  3,387,164 |  3,387,164 |
| `execute_time_ms     ` |  727 |  727 |  727 |  727 |
| `trace_gen_time_ms   ` |  2,374 |  2,374 |  2,374 |  2,374 |
| `stark_prove_excluding_trace_time_ms` |  64,409 |  64,409 |  64,409 |  64,409 |
| `main_trace_commit_time_ms` |  19,604 |  19,604 |  19,604 |  19,604 |
| `generate_perm_trace_time_ms` |  227 |  227 |  227 |  227 |
| `perm_trace_commit_time_ms` |  20,552 |  20,552 |  20,552 |  20,552 |
| `quotient_poly_compute_time_ms` |  2,207 |  2,207 |  2,207 |  2,207 |
| `quotient_poly_commit_time_ms` |  14,467 |  14,467 |  14,467 |  14,467 |
| `pcs_opening_time_ms ` |  7,350 |  7,350 |  7,350 |  7,350 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  341,201 |  341,201 |  341,201 |  341,201 |
| `main_cells_used     ` |  299,445,783 |  299,445,783 |  299,445,783 |  299,445,783 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  77,051 |  77,051 |  77,051 |  77,051 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 572 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 341,201 |  | 299,445,783 | 
| halo2_wrapper | 77,051 |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 2 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 3 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 2 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 3 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 1 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 2 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 3 | 65,536 |  | 16 | 17 | 2,162,688 | 
| internal.0 | FriReducedOpeningAir | 0 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 1 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 2 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 36 | 348 | 12,582,912 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 2 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
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
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| internal.1 | AccessAdapterAir<2> | 4 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.1 | AccessAdapterAir<2> | 5 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.1 | AccessAdapterAir<4> | 4 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.1 | AccessAdapterAir<4> | 5 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.1 | AccessAdapterAir<8> | 4 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.1 | AccessAdapterAir<8> | 5 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.1 | PhantomAir | 4 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | PhantomAir | 5 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.1 | VolatileBoundaryAir | 5 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.2 | AccessAdapterAir<2> | 6 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.2 | AccessAdapterAir<4> | 6 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.2 | AccessAdapterAir<8> | 6 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.2 | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.2 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 3 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 4 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 5 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 6 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 3 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 4 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 5 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 6 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 1 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 2 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 3 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 4 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 5 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 6 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 36 | 348 | 12,582,912 | 
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
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| root | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| root | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| root | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| root | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| internal.0 | 0 | 3,879 | 26,841 | 6,549,955 | 602,671,576 | 21,642 | 4,644 | 3,618 | 4,134 | 4,777 | 4,026 | 224,963,607 | 441 | 1,320 | 
| internal.0 | 1 | 4,315 | 27,580 | 6,544,548 | 602,671,576 | 21,796 | 4,672 | 3,635 | 4,225 | 4,767 | 4,056 | 224,912,642 | 438 | 1,469 | 
| internal.0 | 2 | 4,366 | 27,548 | 6,542,967 | 602,671,576 | 21,702 | 4,709 | 3,639 | 4,141 | 4,768 | 3,992 | 224,896,832 | 451 | 1,480 | 
| internal.0 | 3 | 2,715 | 14,466 | 3,275,347 | 303,696,344 | 11,217 | 2,476 | 1,870 | 2,103 | 2,345 | 2,181 | 113,543,044 | 239 | 534 | 
| internal.1 | 4 | 3,934 | 27,129 | 6,770,999 | 602,671,576 | 21,842 | 4,798 | 3,636 | 4,089 | 4,805 | 4,067 | 232,325,284 | 445 | 1,353 | 
| internal.1 | 5 | 4,452 | 27,798 | 6,662,031 | 602,671,576 | 21,843 | 4,725 | 3,630 | 4,208 | 4,771 | 4,057 | 229,836,593 | 450 | 1,503 | 
| internal.2 | 6 | 4,480 | 27,779 | 6,771,273 | 602,671,576 | 21,767 | 4,673 | 3,641 | 4,136 | 4,806 | 4,066 | 232,328,024 | 443 | 1,532 | 
| leaf | 0 | 2,168 | 13,941 | 3,056,781 | 301,730,264 | 11,181 | 2,553 | 1,882 | 2,073 | 2,393 | 2,039 | 109,820,759 | 238 | 592 | 
| leaf | 1 | 1,869 | 12,865 | 2,540,864 | 271,256,024 | 10,561 | 2,406 | 1,810 | 1,975 | 2,222 | 1,926 | 90,794,075 | 219 | 435 | 
| leaf | 2 | 1,848 | 12,794 | 2,540,783 | 271,256,024 | 10,527 | 2,391 | 1,783 | 1,982 | 2,216 | 1,931 | 90,793,265 | 221 | 419 | 
| leaf | 3 | 1,818 | 12,778 | 2,540,872 | 271,256,024 | 10,541 | 2,404 | 1,788 | 1,966 | 2,217 | 1,943 | 90,794,155 | 220 | 419 | 
| leaf | 4 | 1,830 | 12,756 | 2,541,791 | 271,256,024 | 10,497 | 2,394 | 1,768 | 1,970 | 2,205 | 1,936 | 90,803,345 | 221 | 429 | 
| leaf | 5 | 1,831 | 12,825 | 2,541,259 | 271,256,024 | 10,566 | 2,410 | 1,794 | 1,981 | 2,229 | 1,931 | 90,798,025 | 219 | 428 | 
| leaf | 6 | 1,913 | 13,370 | 2,717,776 | 299,436,504 | 11,005 | 2,442 | 1,865 | 2,048 | 2,366 | 2,053 | 97,375,125 | 228 | 452 | 
| root | 0 | 2,374 | 67,510 | 3,387,164 | 303,696,344 | 64,409 | 2,207 | 14,467 | 20,552 | 7,350 | 19,604 | 117,281,351 | 227 | 727 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 1,049 | 6,407 | 1,747,603 | 197,440,542 | 5,001 | 784 | 470 | 1,671 | 1,061 | 811 | 59,826,835 | 202 | 357 | 
| fib_e2e | 1 | 952 | 6,308 | 1,747,502 | 197,423,810 | 5,047 | 846 | 429 | 1,809 | 1,035 | 792 | 59,798,170 | 133 | 309 | 
| fib_e2e | 2 | 961 | 6,241 | 1,747,502 | 197,423,810 | 4,924 | 850 | 424 | 1,786 | 1,030 | 710 | 59,798,161 | 122 | 356 | 
| fib_e2e | 3 | 858 | 6,114 | 1,747,502 | 197,423,810 | 4,901 | 831 | 437 | 1,803 | 958 | 746 | 59,798,480 | 123 | 355 | 
| fib_e2e | 4 | 869 | 6,090 | 1,747,502 | 197,423,810 | 4,865 | 861 | 425 | 1,753 | 1,010 | 691 | 59,798,779 | 122 | 356 | 
| fib_e2e | 5 | 882 | 6,122 | 1,747,502 | 197,423,810 | 4,884 | 840 | 443 | 1,864 | 983 | 631 | 59,798,769 | 120 | 356 | 
| fib_e2e | 6 | 760 | 5,981 | 1,515,024 | 197,432,594 | 4,914 | 852 | 388 | 1,775 | 1,031 | 734 | 52,001,236 | 132 | 307 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4d21ab35d4ac419a4bf5ef2e5b938bc30dcdafca

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12824473669)
