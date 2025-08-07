| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  242.68 |  197.66 |
| fib_e2e |  19.71 |  3.04 |
| leaf |  16.46 |  3.05 |
| internal.0 |  25.46 |  10.53 |
| internal.1 |  7.01 |  7.01 |
| root |  38.62 |  38.62 |
| halo2_outer |  90.87 |  90.87 |
| halo2_wrapper |  44.49 |  44.49 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,815.14 |  19,706 |  3,035 |  2,125 |
| `main_cells_used     ` |  58,704,834.14 |  410,933,839 |  59,842,060 |  51,906,075 |
| `total_cells_used    ` |  144,147,948.14 |  1,009,035,637 |  146,796,066 |  128,298,001 |
| `insns               ` |  3,000,052.50 |  24,000,420 |  12,000,210 |  1,512,210 |
| `execute_metered_time_ms` |  62 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  192.96 | -          |  192.96 |  192.96 |
| `execute_e3_time_ms  ` |  76.86 |  538 |  80 |  68 |
| `execute_e3_insn_mi/s` |  22.12 | -          |  22.21 |  21.74 |
| `trace_gen_time_ms   ` |  230.57 |  1,614 |  235 |  211 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  39.71 |  278 |  41 |  39 |
| `stark_prove_excluding_trace_time_ms` |  2,507.71 |  17,554 |  2,722 |  1,846 |
| `main_trace_commit_time_ms` |  473.14 |  3,312 |  531 |  344 |
| `generate_perm_trace_time_ms` |  191.57 |  1,341 |  217 |  138 |
| `perm_trace_commit_time_ms` |  567.57 |  3,973 |  614 |  383 |
| `quotient_poly_compute_time_ms` |  239.57 |  1,677 |  255 |  182 |
| `quotient_poly_commit_time_ms` |  216.14 |  1,513 |  265 |  174 |
| `pcs_opening_time_ms ` |  813.43 |  5,694 |  861 |  617 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,352 |  16,464 |  3,046 |  2,141 |
| `main_cells_used     ` |  63,012,434.71 |  441,087,043 |  72,249,645 |  60,332,841 |
| `total_cells_used    ` |  147,165,599.57 |  1,030,159,197 |  170,631,171 |  140,164,583 |
| `insns               ` |  1,066,060 |  7,462,420 |  1,260,152 |  1,007,133 |
| `execute_e3_time_ms  ` |  377.57 |  2,643 |  397 |  371 |
| `execute_e3_insn_mi/s` |  2.82 | -          |  3.17 |  2.67 |
| `trace_gen_time_ms   ` |  154.14 |  1,079 |  181 |  145 |
| `memory_finalize_time_ms` |  8 |  56 |  9 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  1,820.29 |  12,742 |  2,468 |  1,620 |
| `main_trace_commit_time_ms` |  347.43 |  2,432 |  475 |  300 |
| `generate_perm_trace_time_ms` |  128.71 |  901 |  174 |  115 |
| `perm_trace_commit_time_ms` |  415.71 |  2,910 |  576 |  369 |
| `quotient_poly_compute_time_ms` |  203.29 |  1,423 |  265 |  181 |
| `quotient_poly_commit_time_ms` |  166.29 |  1,164 |  240 |  149 |
| `pcs_opening_time_ms ` |  552.71 |  3,869 |  728 |  497 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,485.67 |  25,457 |  10,525 |  4,604 |
| `main_cells_used     ` |  162,337,257.67 |  487,011,773 |  207,105,147 |  74,651,697 |
| `total_cells_used    ` |  292,156,125 |  876,468,375 |  371,613,013 |  136,142,499 |
| `insns               ` |  2,660,572.33 |  7,981,717 |  3,422,696 |  1,151,496 |
| `execute_e3_time_ms  ` |  798.67 |  2,396 |  1,025 |  356 |
| `execute_e3_insn_mi/s` |  3.31 | -          |  3.36 |  3.23 |
| `trace_gen_time_ms   ` |  405.67 |  1,217 |  516 |  186 |
| `memory_finalize_time_ms` |  9 |  27 |  10 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  7,281.33 |  21,844 |  8,984 |  4,062 |
| `main_trace_commit_time_ms` |  1,653.33 |  4,960 |  2,115 |  828 |
| `generate_perm_trace_time_ms` |  322.67 |  968 |  403 |  171 |
| `perm_trace_commit_time_ms` |  1,273 |  3,819 |  1,572 |  682 |
| `quotient_poly_compute_time_ms` |  1,014 |  3,042 |  1,260 |  547 |
| `quotient_poly_commit_time_ms` |  1,084.33 |  3,253 |  1,353 |  626 |
| `pcs_opening_time_ms ` |  1,929 |  5,787 |  2,302 |  1,203 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,007 |  7,007 |  7,007 |  7,007 |
| `main_cells_used     ` |  116,589,053 |  116,589,053 |  116,589,053 |  116,589,053 |
| `total_cells_used    ` |  207,811,259 |  207,811,259 |  207,811,259 |  207,811,259 |
| `insns               ` |  2,330,533 |  2,330,533 |  2,330,533 |  2,330,533 |
| `execute_e3_time_ms  ` |  549 |  549 |  549 |  549 |
| `execute_e3_insn_mi/s` |  4.24 | -          |  4.24 |  4.24 |
| `trace_gen_time_ms   ` |  321 |  321 |  321 |  321 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  6,137 |  6,137 |  6,137 |  6,137 |
| `main_trace_commit_time_ms` |  1,331 |  1,331 |  1,331 |  1,331 |
| `generate_perm_trace_time_ms` |  264 |  264 |  264 |  264 |
| `perm_trace_commit_time_ms` |  1,088 |  1,088 |  1,088 |  1,088 |
| `quotient_poly_compute_time_ms` |  798 |  798 |  798 |  798 |
| `quotient_poly_commit_time_ms` |  912 |  912 |  912 |  912 |
| `pcs_opening_time_ms ` |  1,738 |  1,738 |  1,738 |  1,738 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,615 |  38,615 |  38,615 |  38,615 |
| `main_cells_used     ` |  42,135,743 |  42,135,743 |  42,135,743 |  42,135,743 |
| `total_cells_used    ` |  65,006,501 |  65,006,501 |  65,006,501 |  65,006,501 |
| `insns               ` |  779,780 |  779,780 |  779,780 |  779,780 |
| `execute_e3_time_ms  ` |  189 |  189 |  189 |  189 |
| `execute_e3_insn_mi/s` |  4.11 | -          |  4.11 |  4.11 |
| `trace_gen_time_ms   ` |  117 |  117 |  117 |  117 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  38,309 |  38,309 |  38,309 |  38,309 |
| `main_trace_commit_time_ms` |  12,311 |  12,311 |  12,311 |  12,311 |
| `generate_perm_trace_time_ms` |  90 |  90 |  90 |  90 |
| `perm_trace_commit_time_ms` |  7,577 |  7,577 |  7,577 |  7,577 |
| `quotient_poly_compute_time_ms` |  716 |  716 |  716 |  716 |
| `quotient_poly_commit_time_ms` |  13,704 |  13,704 |  13,704 |  13,704 |
| `pcs_opening_time_ms ` |  3,905 |  3,905 |  3,905 |  3,905 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,873 |  90,873 |  90,873 |  90,873 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,494 |  44,494 |  44,494 |  44,494 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 117 | 65,006,501 | 117 | 2 | 90,886 | 44,494 | 7 | 42,135,743 | 779,780 | 189 | 4.10 | 0 | 21,448 | 39,725 | 

| group | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e |  |  |  | 2,342 |  | 6 |  | 12,000,210 | 1 | 62 | 192.96 | 37 | 
| halo2_outer | 90,873 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper | 44,494 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  | 5,845 |  | 3 |  |  |  | 2 |  |  |  | 
| internal.1 |  |  | 8,246 |  | 3 |  |  |  | 2 |  |  |  | 
| leaf |  | 3,782 |  |  | 1 |  |  |  | 1 |  |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<4> | 1 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | AccessAdapterAir<8> | 1 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | AccessAdapterAir<8> | 2 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 1 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 2 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 1 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 2 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 2 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 1 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 2 | 131,072 |  | 12 | 12 | 3,145,728 | 
| internal.1 | AccessAdapterAir<2> | 3 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 3 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 3 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | FriReducedOpeningAir | 3 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.1 | JalRangeCheckAir | 3 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.1 | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.1 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.1 | VmConnectorAir | 3 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VolatileBoundaryAir | 3 | 262,144 |  | 12 | 12 | 6,291,456 | 
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
| leaf | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 2 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 3 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 4 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 5 | 32,768 |  | 12 | 6 | 589,824 | 
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
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 524,288 |  | 36 | 29 | 34,078,720 | 
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
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 256 |  | 8 | 300 | 78,848 | 
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
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
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
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 2 |  | 32 | 32 | 128 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 64 |  | 52 | 41 | 5,952 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 4 |  | 28 | 20 | 192 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 516 | 10,525 | 371,613,013 | 472,992,226 | 516 | 8,984 | 2 | 1,235 | 1,353 | 1,572 | 2,302 | 10 | 2,115 | 207,105,147 | 3,422,696 | 403 |  | 1,025 | 3.34 | 0 | 
| internal.0 | 1 | 515 | 10,328 | 368,712,863 | 472,992,226 | 515 | 8,798 | 2 | 1,260 | 1,274 | 1,565 | 2,282 | 9 | 2,017 | 205,254,929 | 3,407,525 | 394 |  | 1,015 | 3.36 | 0 | 
| internal.0 | 2 | 186 | 4,604 | 136,142,499 | 185,031,138 | 186 | 4,062 | 2 | 547 | 626 | 682 | 1,203 | 8 | 828 | 74,651,697 | 1,151,496 | 171 |  | 356 | 3.23 | 0 | 
| internal.1 | 3 | 321 | 7,007 | 207,811,259 | 302,819,810 | 321 | 6,137 | 2 | 798 | 912 | 1,088 | 1,738 | 9 | 1,331 | 116,589,053 | 2,330,533 | 264 |  | 549 | 4.24 | 0 | 
| leaf | 0 | 181 | 3,046 | 170,631,171 | 253,173,226 | 181 | 2,468 | 2 | 265 | 240 | 576 | 728 | 8 | 475 | 72,249,645 | 1,260,152 | 174 |  | 397 | 3.17 | 0 | 
| leaf | 1 | 145 | 2,160 | 140,164,583 | 169,088,490 | 145 | 1,639 | 2 | 188 | 150 | 371 | 504 | 9 | 301 | 60,332,841 | 1,007,133 | 117 |  | 376 | 2.67 | 0 | 
| leaf | 2 | 147 | 2,153 | 140,168,343 | 169,088,490 | 147 | 1,634 | 2 | 189 | 149 | 372 | 504 | 8 | 301 | 60,333,969 | 1,007,227 | 115 |  | 372 | 2.70 | 0 | 
| leaf | 3 | 146 | 2,175 | 140,165,303 | 169,088,490 | 146 | 1,658 | 2 | 193 | 149 | 372 | 497 | 8 | 324 | 60,333,057 | 1,007,151 | 118 |  | 371 | 2.71 | 0 | 
| leaf | 4 | 146 | 2,209 | 140,166,863 | 169,088,490 | 146 | 1,687 | 2 | 189 | 149 | 391 | 503 | 7 | 333 | 60,333,525 | 1,007,190 | 117 |  | 376 | 2.67 | 0 | 
| leaf | 5 | 145 | 2,141 | 140,165,623 | 169,088,490 | 145 | 1,620 | 2 | 181 | 149 | 369 | 501 | 8 | 300 | 60,333,153 | 1,007,159 | 116 |  | 376 | 2.68 | 0 | 
| leaf | 6 | 169 | 2,580 | 158,697,311 | 208,084,458 | 169 | 2,036 | 2 | 218 | 178 | 459 | 632 | 8 | 398 | 67,170,853 | 1,166,408 | 144 |  | 375 | 3.11 | 0 | 
| root | 0 | 117 | 38,615 | 65,006,501 | 80,435,354 | 117 | 38,309 | 2 | 716 | 13,704 | 7,577 | 3,905 | 8 | 12,311 | 42,135,743 | 779,780 | 90 | 3 | 189 | 4.11 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 65,323,264 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 64,782,596 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 148,914,890 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 65,323,264 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 64,782,596 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 148,914,890 | 2,013,265,921 | 
| internal.0 | 2 | 0 | 4,882,564 | 2,013,265,921 | 
| internal.0 | 2 | 1 | 26,358,016 | 2,013,265,921 | 
| internal.0 | 2 | 2 | 2,441,282 | 2,013,265,921 | 
| internal.0 | 2 | 3 | 26,091,780 | 2,013,265,921 | 
| internal.0 | 2 | 4 | 131,072 | 2,013,265,921 | 
| internal.0 | 2 | 5 | 60,297,930 | 2,013,265,921 | 
| internal.1 | 3 | 0 | 8,454,276 | 2,013,265,921 | 
| internal.1 | 3 | 1 | 40,132,864 | 2,013,265,921 | 
| internal.1 | 3 | 2 | 4,227,138 | 2,013,265,921 | 
| internal.1 | 3 | 3 | 40,386,820 | 2,013,265,921 | 
| internal.1 | 3 | 4 | 262,144 | 2,013,265,921 | 
| internal.1 | 3 | 5 | 93,856,458 | 2,013,265,921 | 
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 
| leaf | 1 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 1 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 1 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 1 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 2 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 2 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 2 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 2 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 2 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 2 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 3 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 3 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 3 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 3 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 3 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 3 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 4 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 4 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 4 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 4 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 4 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 4 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 5 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 5 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 5 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 5 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 5 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 5 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 6 | 0 | 4,391,044 | 2,013,265,921 | 
| leaf | 6 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 6 | 2 | 2,195,522 | 2,013,265,921 | 
| leaf | 6 | 3 | 20,586,756 | 2,013,265,921 | 
| leaf | 6 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 6 | 5 | 48,157,386 | 2,013,265,921 | 
| root | 0 | 0 | 2,572,420 | 2,013,265,921 | 
| root | 0 | 1 | 12,005,632 | 2,013,265,921 | 
| root | 0 | 2 | 1,286,210 | 2,013,265,921 | 
| root | 0 | 3 | 12,067,076 | 2,013,265,921 | 
| root | 0 | 4 | 65,536 | 2,013,265,921 | 
| root | 0 | 5 | 28,390,090 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 235 | 3,035 | 146,796,066 | 253,084,876 | 235 | 2,722 | 2 | 252 | 265 | 595 | 861 | 40 | 6 | 0 | 531 | 59,842,060 | 1,748,000 | 211 | 78 | 22.19 | 0 | 
| fib_e2e | 1 | 234 | 2,917 | 146,788,271 | 253,031,994 | 234 | 2,603 | 2 | 255 | 213 | 591 | 841 | 40 | 6 | 0 | 479 | 59,837,129 | 1,748,000 | 217 | 80 | 21.74 | 0 | 
| fib_e2e | 2 | 233 | 2,902 | 146,788,335 | 253,031,994 | 233 | 2,591 | 2 | 247 | 213 | 608 | 841 | 39 | 6 | 0 | 482 | 59,837,145 | 1,748,000 | 197 | 78 | 22.20 | 0 | 
| fib_e2e | 3 | 234 | 2,912 | 146,788,358 | 253,031,994 | 234 | 2,600 | 2 | 250 | 219 | 592 | 848 | 40 | 6 | 0 | 491 | 59,837,156 | 1,748,000 | 191 | 78 | 22.16 | 0 | 
| fib_e2e | 4 | 234 | 2,922 | 146,788,271 | 253,031,994 | 234 | 2,610 | 2 | 250 | 213 | 614 | 842 | 39 | 6 | 0 | 489 | 59,837,129 | 1,748,000 | 197 | 78 | 22.20 | 0 | 
| fib_e2e | 5 | 233 | 2,893 | 146,788,335 | 253,031,994 | 233 | 2,582 | 2 | 241 | 216 | 590 | 844 | 39 | 6 | 0 | 496 | 59,837,145 | 1,748,000 | 190 | 78 | 22.21 | 0 | 
| fib_e2e | 6 | 211 | 2,125 | 128,298,001 | 160,813,290 | 211 | 1,846 | 2 | 182 | 174 | 383 | 617 | 41 | 6 | 0 | 344 | 51,906,075 | 1,512,210 | 138 | 68 | 22.18 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fib_e2e | 0 | 0 | 6,029,422 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 17,039,880 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 3,014,711 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 17,039,836 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 12,451,904 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 56,502,857 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 3,932,336 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,624 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,168 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,692 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,209,000 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,531,924 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/8a71af1b2eb97d8fc15a88d5847cc07b2eb1d3bf

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16803775964)
