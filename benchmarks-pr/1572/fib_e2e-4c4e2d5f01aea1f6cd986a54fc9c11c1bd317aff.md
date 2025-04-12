| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  250.90 |  204.03 |
| fib_e2e |  18.30 |  2.72 |
| leaf |  22.85 |  3.72 |
| internal.0 |  30.05 |  17.88 |
| internal.1 |  5.39 |  5.39 |
| root |  38.95 |  38.95 |
| halo2_outer |  91.28 |  91.28 |
| halo2_wrapper |  44.09 |  44.09 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,614.43 |  18,301 |  2,720 |  2,535 |
| `main_cells_used     ` |  57,756,792.14 |  404,297,545 |  58,898,160 |  51,100,924 |
| `total_cycles        ` |  1,714,325.29 |  12,000,277 |  1,747,603 |  1,515,164 |
| `execute_time_ms     ` |  230.71 |  1,615 |  239 |  203 |
| `trace_gen_time_ms   ` |  596.43 |  4,175 |  620 |  544 |
| `stark_prove_excluding_trace_time_ms` |  1,787.29 |  12,511 |  1,861 |  1,767 |
| `main_trace_commit_time_ms` |  288.71 |  2,021 |  316 |  283 |
| `generate_perm_trace_time_ms` |  115.43 |  808 |  120 |  112 |
| `perm_trace_commit_time_ms` |  442.71 |  3,099 |  444 |  441 |
| `quotient_poly_compute_time_ms` |  249.71 |  1,748 |  255 |  245 |
| `quotient_poly_commit_time_ms` |  188.57 |  1,320 |  218 |  181 |
| `pcs_opening_time_ms ` |  497.71 |  3,484 |  510 |  492 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,263.86 |  22,847 |  3,723 |  3,136 |
| `main_cells_used     ` |  61,732,738.29 |  432,129,168 |  70,148,670 |  58,838,845 |
| `total_cycles        ` |  1,100,551.29 |  7,703,859 |  1,260,305 |  1,044,785 |
| `execute_time_ms     ` |  480.57 |  3,364 |  509 |  468 |
| `trace_gen_time_ms   ` |  587.14 |  4,110 |  664 |  540 |
| `stark_prove_excluding_trace_time_ms` |  2,196.14 |  15,373 |  2,554 |  2,124 |
| `main_trace_commit_time_ms` |  367.29 |  2,571 |  421 |  353 |
| `generate_perm_trace_time_ms` |  135.29 |  947 |  163 |  126 |
| `perm_trace_commit_time_ms` |  561.43 |  3,930 |  701 |  534 |
| `quotient_poly_compute_time_ms` |  313 |  2,191 |  372 |  300 |
| `quotient_poly_commit_time_ms` |  242.71 |  1,699 |  279 |  233 |
| `pcs_opening_time_ms ` |  571.14 |  3,998 |  613 |  553 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,022.50 |  30,045 |  17,882 |  12,163 |
| `main_cells_used     ` |  237,686,621.50 |  475,373,243 |  271,464,582 |  203,908,661 |
| `total_cycles        ` |  4,081,979.50 |  8,163,959 |  4,665,081 |  3,498,878 |
| `execute_time_ms     ` |  1,673 |  3,346 |  1,918 |  1,428 |
| `trace_gen_time_ms   ` |  2,162.50 |  4,325 |  2,442 |  1,883 |
| `stark_prove_excluding_trace_time_ms` |  11,187 |  22,374 |  13,522 |  8,852 |
| `main_trace_commit_time_ms` |  2,343.50 |  4,687 |  2,790 |  1,897 |
| `generate_perm_trace_time_ms` |  487 |  974 |  592 |  382 |
| `perm_trace_commit_time_ms` |  2,055 |  4,110 |  2,515 |  1,595 |
| `quotient_poly_compute_time_ms` |  2,073 |  4,146 |  2,467 |  1,679 |
| `quotient_poly_commit_time_ms` |  1,847.50 |  3,695 |  2,239 |  1,456 |
| `pcs_opening_time_ms ` |  2,376.50 |  4,753 |  2,914 |  1,839 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,386 |  5,386 |  5,386 |  5,386 |
| `main_cells_used     ` |  75,976,403 |  75,976,403 |  75,976,403 |  75,976,403 |
| `total_cycles        ` |  1,539,950 |  1,539,950 |  1,539,950 |  1,539,950 |
| `execute_time_ms     ` |  501 |  501 |  501 |  501 |
| `trace_gen_time_ms   ` |  724 |  724 |  724 |  724 |
| `stark_prove_excluding_trace_time_ms` |  4,161 |  4,161 |  4,161 |  4,161 |
| `main_trace_commit_time_ms` |  774 |  774 |  774 |  774 |
| `generate_perm_trace_time_ms` |  153 |  153 |  153 |  153 |
| `perm_trace_commit_time_ms` |  693 |  693 |  693 |  693 |
| `quotient_poly_compute_time_ms` |  722 |  722 |  722 |  722 |
| `quotient_poly_commit_time_ms` |  807 |  807 |  807 |  807 |
| `pcs_opening_time_ms ` |  1,006 |  1,006 |  1,006 |  1,006 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,951 |  38,951 |  38,951 |  38,951 |
| `main_cells_used     ` |  37,521,939 |  37,521,939 |  37,521,939 |  37,521,939 |
| `total_cycles        ` |  759,194 |  759,194 |  759,194 |  759,194 |
| `execute_time_ms     ` |  238 |  238 |  238 |  238 |
| `trace_gen_time_ms   ` |  396 |  396 |  396 |  396 |
| `stark_prove_excluding_trace_time_ms` |  38,317 |  38,317 |  38,317 |  38,317 |
| `main_trace_commit_time_ms` |  12,325 |  12,325 |  12,325 |  12,325 |
| `generate_perm_trace_time_ms` |  77 |  77 |  77 |  77 |
| `perm_trace_commit_time_ms` |  7,613 |  7,613 |  7,613 |  7,613 |
| `quotient_poly_compute_time_ms` |  853 |  853 |  853 |  853 |
| `quotient_poly_commit_time_ms` |  13,735 |  13,735 |  13,735 |  13,735 |
| `pcs_opening_time_ms ` |  3,690 |  3,690 |  3,690 |  3,690 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  91,276 |  91,276 |  91,276 |  91,276 |
| `main_cells_used     ` |  62,297,054 |  62,297,054 |  62,297,054 |  62,297,054 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,095 |  44,095 |  44,095 |  44,095 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 247 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 91,276 |  | 62,297,054 | 
| halo2_wrapper | 44,095 |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<4> | 1 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<8> | 0 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | AccessAdapterAir<8> | 1 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | FriReducedOpeningAir | 0 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | FriReducedOpeningAir | 1 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | JalRangeCheckAir | 0 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | JalRangeCheckAir | 1 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 1 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.1 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 2 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | FriReducedOpeningAir | 2 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.1 | JalRangeCheckAir | 2 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.1 | PhantomAir | 2 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.1 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VolatileBoundaryAir | 2 | 262,144 |  | 12 | 12 | 6,291,456 | 
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
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 1 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 2 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 3 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 4 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 5 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 6 | 131,072 |  | 20 | 12 | 4,194,304 | 
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
| fib_e2e | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 64 |  | 16 | 17 | 2,112 | 
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
| fib_e2e | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 64 |  | 12 | 20 | 2,048 | 
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
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 512 |  | 8 | 300 | 157,696 | 
| fib_e2e | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 2 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 3 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 4 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 5 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 6 | 8,192 |  | 8 | 10 | 147,456 | 
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
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 2 |  | 32 | 32 | 128 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 64 |  | 52 | 41 | 5,952 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 2,442 | 17,882 | 4,665,081 | 682,445,282 | 13,522 | 2,467 | 2,239 | 2,515 | 2,914 | 2,790 | 271,464,582 | 592 | 1,918 | 
| internal.0 | 1 | 1,883 | 12,163 | 3,498,878 | 472,992,226 | 8,852 | 1,679 | 1,456 | 1,595 | 1,839 | 1,897 | 203,908,661 | 382 | 1,428 | 
| internal.1 | 2 | 724 | 5,386 | 1,539,950 | 186,591,714 | 4,161 | 722 | 807 | 693 | 1,006 | 774 | 75,976,403 | 153 | 501 | 
| leaf | 0 | 660 | 3,723 | 1,260,305 | 253,173,226 | 2,554 | 372 | 279 | 701 | 613 | 421 | 70,148,670 | 163 | 509 | 
| leaf | 1 | 569 | 3,164 | 1,044,801 | 202,872,298 | 2,124 | 300 | 233 | 534 | 568 | 356 | 58,839,037 | 127 | 471 | 
| leaf | 2 | 555 | 3,166 | 1,044,785 | 202,872,298 | 2,143 | 301 | 235 | 537 | 577 | 354 | 58,838,845 | 134 | 468 | 
| leaf | 3 | 540 | 3,136 | 1,044,846 | 202,872,298 | 2,125 | 303 | 237 | 538 | 553 | 358 | 58,839,577 | 131 | 471 | 
| leaf | 4 | 564 | 3,166 | 1,044,865 | 202,872,298 | 2,131 | 304 | 237 | 541 | 565 | 353 | 58,839,805 | 126 | 471 | 
| leaf | 5 | 558 | 3,156 | 1,044,844 | 202,872,298 | 2,128 | 301 | 234 | 538 | 555 | 357 | 58,839,553 | 137 | 470 | 
| leaf | 6 | 664 | 3,336 | 1,219,413 | 208,084,458 | 2,168 | 310 | 244 | 541 | 567 | 372 | 67,783,681 | 129 | 504 | 
| root | 0 | 396 | 38,951 | 759,194 | 80,435,354 | 38,317 | 853 | 13,735 | 7,613 | 3,690 | 12,325 | 37,521,939 | 77 | 238 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 19,529,860 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 100,188,416 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 9,764,930 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 99,909,892 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 230,310,602 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 65,323,264 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 64,782,596 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 148,914,890 | 2,013,265,921 | 
| internal.1 | 2 | 0 | 5,144,708 | 2,013,265,921 | 
| internal.1 | 2 | 1 | 24,011,008 | 2,013,265,921 | 
| internal.1 | 2 | 2 | 2,572,354 | 2,013,265,921 | 
| internal.1 | 2 | 3 | 24,133,892 | 2,013,265,921 | 
| internal.1 | 2 | 4 | 131,072 | 2,013,265,921 | 
| internal.1 | 2 | 5 | 56,386,250 | 2,013,265,921 | 
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 
| leaf | 1 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 1 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 1 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 1 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 47,114,954 | 2,013,265,921 | 
| leaf | 2 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 2 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 2 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 2 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 2 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 2 | 5 | 47,114,954 | 2,013,265,921 | 
| leaf | 3 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 3 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 3 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 3 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 3 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 3 | 5 | 47,114,954 | 2,013,265,921 | 
| leaf | 4 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 4 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 4 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 4 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 4 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 4 | 5 | 47,114,954 | 2,013,265,921 | 
| leaf | 5 | 0 | 4,227,204 | 2,013,265,921 | 
| leaf | 5 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 5 | 2 | 2,113,602 | 2,013,265,921 | 
| leaf | 5 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 5 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 5 | 5 | 47,114,954 | 2,013,265,921 | 
| leaf | 6 | 0 | 4,391,044 | 2,013,265,921 | 
| leaf | 6 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 6 | 2 | 2,195,522 | 2,013,265,921 | 
| leaf | 6 | 3 | 20,586,756 | 2,013,265,921 | 
| leaf | 6 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 6 | 5 | 48,157,386 | 2,013,265,921 | 
| root | 0 | 0 | 2,252,928 | 2,013,265,921 | 
| root | 0 | 1 | 14,557,184 | 2,013,265,921 | 
| root | 0 | 2 | 1,126,464 | 2,013,265,921 | 
| root | 0 | 3 | 15,540,224 | 2,013,265,921 | 
| root | 0 | 4 | 262,144 | 2,013,265,921 | 
| root | 0 | 5 | 34,263,234 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 620 | 2,720 | 1,747,603 | 160,811,212 | 1,861 | 251 | 218 | 441 | 510 | 316 | 58,898,160 | 120 | 239 | 
| fib_e2e | 1 | 594 | 2,594 | 1,747,502 | 160,757,324 | 1,767 | 245 | 181 | 443 | 497 | 284 | 58,859,694 | 114 | 233 | 
| fib_e2e | 2 | 597 | 2,610 | 1,747,502 | 160,757,324 | 1,779 | 255 | 186 | 441 | 495 | 283 | 58,859,694 | 114 | 234 | 
| fib_e2e | 3 | 604 | 2,615 | 1,747,502 | 160,757,324 | 1,776 | 247 | 183 | 444 | 500 | 284 | 58,859,684 | 115 | 235 | 
| fib_e2e | 4 | 607 | 2,610 | 1,747,502 | 160,757,324 | 1,769 | 248 | 184 | 444 | 494 | 283 | 58,859,695 | 112 | 234 | 
| fib_e2e | 5 | 609 | 2,617 | 1,747,502 | 160,757,324 | 1,771 | 249 | 182 | 443 | 492 | 284 | 58,859,694 | 116 | 237 | 
| fib_e2e | 6 | 544 | 2,535 | 1,515,164 | 160,892,348 | 1,788 | 253 | 186 | 443 | 496 | 287 | 51,100,924 | 117 | 203 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fib_e2e | 0 | 0 | 3,932,302 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 10,748,488 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 1,966,151 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 10,748,492 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 7,209,044 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 35,531,581 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 3,932,346 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,632 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,173 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,700 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,209,020 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,532,231 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4c4e2d5f01aea1f6cd986a54fc9c11c1bd317aff

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14414890695)
