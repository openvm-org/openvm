| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  253.73 |  201.15 |
| fib_e2e |  19.74 |  3.05 |
| leaf |  23.63 |  3.99 |
| internal.0 |  27.11 |  10.87 |
| internal.1 |  7.66 |  7.66 |
| root |  38.56 |  38.56 |
| halo2_outer |  92.52 |  92.52 |
| halo2_wrapper |  44.44 |  44.44 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,820.43 |  19,743 |  3,051 |  2,148 |
| `main_cells_used     ` |  58,704,834.14 |  410,933,839 |  59,842,060 |  51,906,075 |
| `total_cells_used    ` |  144,147,948.14 |  1,009,035,637 |  146,796,066 |  128,298,001 |
| `insns               ` |  3,000,052.50 |  24,000,420 |  12,000,210 |  1,512,210 |
| `execute_metered_time_ms` |  59 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  200.21 | -          |  200.21 |  200.21 |
| `execute_preflight_time_ms` |  64.43 |  451 |  67 |  52 |
| `execute_preflight_insn_mi/s` |  37.62 | -          |  37.69 |  37.52 |
| `trace_gen_time_ms   ` |  222.29 |  1,556 |  226 |  204 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  2,316 |  16,212 |  2,514 |  1,689 |
| `main_trace_commit_time_ms` |  411.43 |  2,880 |  459 |  306 |
| `generate_perm_trace_time_ms` |  185.29 |  1,297 |  211 |  122 |
| `perm_trace_commit_time_ms` |  491 |  3,437 |  521 |  334 |
| `quotient_poly_compute_time_ms` |  230.14 |  1,611 |  244 |  165 |
| `quotient_poly_commit_time_ms` |  198.29 |  1,388 |  242 |  155 |
| `pcs_opening_time_ms ` |  793.86 |  5,557 |  839 |  600 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,375.57 |  23,629 |  3,991 |  3,168 |
| `main_cells_used     ` |  63,012,643.86 |  441,088,507 |  72,247,425 |  60,332,853 |
| `total_cells_used    ` |  147,166,296.71 |  1,030,164,077 |  170,623,771 |  140,164,623 |
| `insns               ` |  1,066,077.43 |  7,462,542 |  1,259,967 |  1,007,134 |
| `execute_preflight_time_ms` |  471 |  3,297 |  491 |  465 |
| `execute_preflight_insn_mi/s` |  3.39 | -          |  3.83 |  3.22 |
| `trace_gen_time_ms   ` |  153.71 |  1,076 |  181 |  144 |
| `memory_finalize_time_ms` |  9.43 |  66 |  11 |  8 |
| `stark_prove_excluding_trace_time_ms` |  1,709.14 |  11,964 |  2,300 |  1,520 |
| `main_trace_commit_time_ms` |  308.86 |  2,162 |  418 |  272 |
| `generate_perm_trace_time_ms` |  127.71 |  894 |  176 |  112 |
| `perm_trace_commit_time_ms` |  370.71 |  2,595 |  503 |  322 |
| `quotient_poly_compute_time_ms` |  203.29 |  1,423 |  264 |  180 |
| `quotient_poly_commit_time_ms` |  151 |  1,057 |  221 |  135 |
| `pcs_opening_time_ms ` |  541.57 |  3,791 |  713 |  487 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,037 |  27,111 |  10,865 |  5,466 |
| `main_cells_used     ` |  162,336,269.67 |  487,008,809 |  207,104,151 |  74,651,709 |
| `total_cells_used    ` |  292,153,819.67 |  876,461,459 |  371,610,689 |  136,142,527 |
| `insns               ` |  2,660,490 |  7,981,470 |  3,422,613 |  1,151,497 |
| `execute_preflight_time_ms` |  812.33 |  2,437 |  1,002 |  449 |
| `execute_preflight_insn_mi/s` |  4.02 | -          |  4.10 |  3.90 |
| `trace_gen_time_ms   ` |  391.33 |  1,174 |  497 |  181 |
| `memory_finalize_time_ms` |  9.67 |  29 |  11 |  8 |
| `stark_prove_excluding_trace_time_ms` |  6,740.33 |  20,221 |  8,294 |  3,746 |
| `main_trace_commit_time_ms` |  1,454.67 |  4,364 |  1,855 |  696 |
| `generate_perm_trace_time_ms` |  319 |  957 |  396 |  168 |
| `perm_trace_commit_time_ms` |  1,087.67 |  3,263 |  1,349 |  586 |
| `quotient_poly_compute_time_ms` |  1,007.67 |  3,023 |  1,254 |  541 |
| `quotient_poly_commit_time_ms` |  961.33 |  2,884 |  1,217 |  560 |
| `pcs_opening_time_ms ` |  1,904.67 |  5,714 |  2,264 |  1,191 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,663 |  7,663 |  7,663 |  7,663 |
| `main_cells_used     ` |  116,590,805 |  116,590,805 |  116,590,805 |  116,590,805 |
| `total_cells_used    ` |  207,815,347 |  207,815,347 |  207,815,347 |  207,815,347 |
| `insns               ` |  2,330,679 |  2,330,679 |  2,330,679 |  2,330,679 |
| `execute_preflight_time_ms` |  599 |  599 |  599 |  599 |
| `execute_preflight_insn_mi/s` |  5.33 | -          |  5.33 |  5.33 |
| `trace_gen_time_ms   ` |  310 |  310 |  310 |  310 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `stark_prove_excluding_trace_time_ms` |  5,662 |  5,662 |  5,662 |  5,662 |
| `main_trace_commit_time_ms` |  1,184 |  1,184 |  1,184 |  1,184 |
| `generate_perm_trace_time_ms` |  260 |  260 |  260 |  260 |
| `perm_trace_commit_time_ms` |  890 |  890 |  890 |  890 |
| `quotient_poly_compute_time_ms` |  829 |  829 |  829 |  829 |
| `quotient_poly_commit_time_ms` |  795 |  795 |  795 |  795 |
| `pcs_opening_time_ms ` |  1,700 |  1,700 |  1,700 |  1,700 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,561 |  38,561 |  38,561 |  38,561 |
| `main_cells_used     ` |  42,136,547 |  42,136,547 |  42,136,547 |  42,136,547 |
| `total_cells_used    ` |  65,008,109 |  65,008,109 |  65,008,109 |  65,008,109 |
| `insns               ` |  779,847 |  779,847 |  779,847 |  779,847 |
| `execute_preflight_time_ms` |  167 |  167 |  167 |  167 |
| `execute_preflight_insn_mi/s` |  5.10 | -          |  5.10 |  5.10 |
| `trace_gen_time_ms   ` |  118 |  118 |  118 |  118 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `stark_prove_excluding_trace_time_ms` |  38,276 |  38,276 |  38,276 |  38,276 |
| `main_trace_commit_time_ms` |  12,311 |  12,311 |  12,311 |  12,311 |
| `generate_perm_trace_time_ms` |  90 |  90 |  90 |  90 |
| `perm_trace_commit_time_ms` |  7,581 |  7,581 |  7,581 |  7,581 |
| `quotient_poly_compute_time_ms` |  686 |  686 |  686 |  686 |
| `quotient_poly_commit_time_ms` |  13,685 |  13,685 |  13,685 |  13,685 |
| `pcs_opening_time_ms ` |  3,905 |  3,905 |  3,905 |  3,905 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  92,522 |  92,522 |  92,522 |  92,522 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,440 |  44,440 |  44,440 |  44,440 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_preflight_time_ms | execute_preflight_insn_mi/s | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 118 | 65,008,109 | 118 | 0 | 92,536 | 44,440 | 8 | 42,136,547 | 779,847 | 307 | 5.22 | 19,848 | 39,661 | 

| group | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e |  |  |  | 2,148 |  | 6 |  | 12,000,210 | 1 | 59 | 200.21 | 36 | 
| halo2_outer | 92,522 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper | 44,440 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  | 5,468 |  | 3 |  |  |  | 2 |  |  |  | 
| internal.1 |  |  | 7,665 |  | 3 |  |  |  | 2 |  |  |  | 
| leaf |  | 3,577 |  |  | 1 |  |  |  | 1 |  |  |  | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 497 | 10,865 | 371,610,689 | 472,992,226 | 497 | 8,294 | 0 | 1,228 | 1,217 | 1,328 | 2,264 | 11 | 1,855 | 207,104,151 | 3,422,613 | 396 |  | 1,002 | 4.06 | 
| internal.0 | 1 | 496 | 10,780 | 368,708,243 | 472,992,226 | 496 | 8,181 | 0 | 1,254 | 1,107 | 1,349 | 2,259 | 10 | 1,813 | 205,252,949 | 3,407,360 | 393 |  | 986 | 4.10 | 
| internal.0 | 2 | 181 | 5,466 | 136,142,527 | 185,031,138 | 181 | 3,746 | 0 | 541 | 560 | 586 | 1,191 | 8 | 696 | 74,651,709 | 1,151,497 | 168 |  | 449 | 3.90 | 
| internal.1 | 3 | 310 | 7,663 | 207,815,347 | 302,819,810 | 310 | 5,662 | 0 | 829 | 795 | 890 | 1,700 | 9 | 1,184 | 116,590,805 | 2,330,679 | 260 |  | 599 | 5.33 | 
| leaf | 0 | 181 | 3,991 | 170,623,771 | 253,173,226 | 181 | 2,300 | 0 | 264 | 221 | 503 | 713 | 11 | 418 | 72,247,425 | 1,259,967 | 176 |  | 491 | 3.83 | 
| leaf | 1 | 145 | 3,168 | 140,169,343 | 169,088,490 | 145 | 1,530 | 0 | 192 | 135 | 324 | 489 | 9 | 274 | 60,334,269 | 1,007,252 | 112 |  | 467 | 3.23 | 
| leaf | 2 | 145 | 3,200 | 140,164,623 | 169,088,490 | 145 | 1,552 | 0 | 199 | 135 | 322 | 489 | 8 | 286 | 60,332,853 | 1,007,134 | 113 |  | 467 | 3.22 | 
| leaf | 3 | 148 | 3,268 | 140,165,903 | 169,088,490 | 148 | 1,605 | 1 | 193 | 135 | 368 | 504 | 9 | 286 | 60,333,237 | 1,007,166 | 114 |  | 467 | 3.23 | 
| leaf | 4 | 144 | 3,183 | 140,169,103 | 169,088,490 | 144 | 1,520 | 0 | 180 | 135 | 324 | 487 | 8 | 272 | 60,334,197 | 1,007,246 | 113 |  | 469 | 3.22 | 
| leaf | 5 | 145 | 3,243 | 140,167,463 | 169,088,490 | 145 | 1,574 | 0 | 186 | 135 | 351 | 492 | 11 | 277 | 60,333,705 | 1,007,205 | 128 |  | 471 | 3.22 | 
| leaf | 6 | 168 | 3,576 | 158,703,871 | 208,084,458 | 168 | 1,883 | 0 | 209 | 161 | 403 | 617 | 10 | 349 | 67,172,821 | 1,166,572 | 138 |  | 465 | 3.75 | 
| root | 0 | 118 | 38,561 | 65,008,109 | 80,435,354 | 118 | 38,276 | 0 | 686 | 13,685 | 7,581 | 3,905 | 7 | 12,311 | 42,136,547 | 779,847 | 90 | 3 | 167 | 5.10 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 226 | 3,051 | 146,796,066 | 253,084,876 | 226 | 2,514 | 0 | 239 | 242 | 520 | 839 | 6 | 0 | 459 | 59,842,060 | 1,748,000 | 211 | 67 | 37.58 | 
| fib_e2e | 1 | 225 | 2,932 | 146,788,271 | 253,031,994 | 225 | 2,409 | 110 | 238 | 212 | 515 | 825 | 6 | 0 | 426 | 59,837,129 | 1,748,000 | 189 | 67 | 37.52 | 
| fib_e2e | 2 | 225 | 2,925 | 146,788,335 | 253,031,994 | 225 | 2,415 | 110 | 238 | 195 | 516 | 824 | 6 | 0 | 426 | 59,837,145 | 1,748,000 | 211 | 66 | 37.56 | 
| fib_e2e | 3 | 225 | 2,890 | 146,788,358 | 253,031,994 | 225 | 2,385 | 110 | 244 | 187 | 521 | 822 | 6 | 0 | 414 | 59,837,156 | 1,748,000 | 189 | 67 | 37.62 | 
| fib_e2e | 4 | 226 | 2,921 | 146,788,271 | 253,031,994 | 226 | 2,423 | 110 | 243 | 208 | 517 | 829 | 6 | 0 | 431 | 59,837,129 | 1,748,000 | 187 | 66 | 37.69 | 
| fib_e2e | 5 | 225 | 2,876 | 146,788,335 | 253,031,994 | 225 | 2,377 | 110 | 244 | 189 | 514 | 818 | 6 | 0 | 418 | 59,837,145 | 1,748,000 | 188 | 66 | 37.69 | 
| fib_e2e | 6 | 204 | 2,148 | 128,298,001 | 160,813,290 | 204 | 1,689 | 95 | 165 | 155 | 334 | 600 | 6 | 0 | 306 | 51,906,075 | 1,512,210 | 122 | 52 | 37.67 | 

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


Commit: https://github.com/openvm-org/openvm/commit/9736d5e3447b4eafa571e7bafc4de60b0cea06c5

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16896065795)
