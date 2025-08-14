| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  255.08 |  200.35 |
| fib_e2e |  20.83 |  3.21 |
| leaf |  24.16 |  4.12 |
| internal.0 |  28.58 |  11.50 |
| internal.1 |  8.04 |  8.04 |
| root |  38.55 |  38.55 |
| halo2_outer |  90.31 |  90.31 |
| halo2_wrapper |  44.55 |  44.55 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,976.14 |  20,833 |  3,214 |  2,264 |
| `main_cells_used     ` |  58,704,834.14 |  410,933,839 |  59,842,060 |  51,906,075 |
| `total_cells_used    ` |  144,147,948.14 |  1,009,035,637 |  146,796,066 |  128,298,001 |
| `execute_metered_time_ms` |  60 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  197.61 | -          |  197.61 |  197.61 |
| `execute_preflight_insns` |  1,714,315.71 |  12,000,210 |  1,748,000 |  1,512,210 |
| `execute_preflight_time_ms` |  65.14 |  456 |  68 |  54 |
| `execute_preflight_insn_mi/s` |  37.10 | -          |  37.24 |  36.69 |
| `trace_gen_time_ms   ` |  222.43 |  1,557 |  227 |  203 |
| `memory_finalize_time_ms` |  0.14 |  1 |  1 |  0 |
| `stark_prove_excluding_trace_time_ms` |  2,469.71 |  17,288 |  2,673 |  1,796 |
| `main_trace_commit_time_ms` |  467.14 |  3,270 |  523 |  342 |
| `generate_perm_trace_time_ms` |  185.71 |  1,300 |  215 |  128 |
| `perm_trace_commit_time_ms` |  561 |  3,927 |  597 |  377 |
| `quotient_poly_compute_time_ms` |  235.14 |  1,646 |  252 |  174 |
| `quotient_poly_commit_time_ms` |  214 |  1,498 |  273 |  171 |
| `pcs_opening_time_ms ` |  801.29 |  5,609 |  841 |  599 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,451.14 |  24,158 |  4,123 |  3,227 |
| `main_cells_used     ` |  63,012,388.43 |  441,086,719 |  72,248,817 |  60,332,613 |
| `total_cells_used    ` |  147,165,445.29 |  1,030,158,117 |  170,628,411 |  140,163,823 |
| `execute_preflight_insns` |  1,066,056.14 |  7,462,393 |  1,260,083 |  1,007,114 |
| `execute_preflight_time_ms` |  459.86 |  3,219 |  476 |  454 |
| `execute_preflight_insn_mi/s` |  3.49 | -          |  3.96 |  3.30 |
| `trace_gen_time_ms   ` |  153.57 |  1,075 |  180 |  145 |
| `memory_finalize_time_ms` |  8.43 |  59 |  10 |  8 |
| `stark_prove_excluding_trace_time_ms` |  1,788.14 |  12,517 |  2,433 |  1,593 |
| `main_trace_commit_time_ms` |  340.71 |  2,385 |  471 |  297 |
| `generate_perm_trace_time_ms` |  128.71 |  901 |  178 |  114 |
| `perm_trace_commit_time_ms` |  409.57 |  2,867 |  572 |  367 |
| `quotient_poly_compute_time_ms` |  197.43 |  1,382 |  255 |  168 |
| `quotient_poly_commit_time_ms` |  163.71 |  1,146 |  239 |  145 |
| `pcs_opening_time_ms ` |  542 |  3,794 |  715 |  488 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,527.67 |  28,583 |  11,503 |  5,812 |
| `main_cells_used     ` |  162,337,209.67 |  487,011,629 |  207,106,479 |  74,652,513 |
| `total_cells_used    ` |  292,156,013 |  876,468,039 |  371,616,121 |  136,144,403 |
| `execute_preflight_insns` |  2,660,568.33 |  7,981,705 |  3,422,807 |  1,151,564 |
| `execute_preflight_time_ms` |  802.67 |  2,408 |  991 |  441 |
| `execute_preflight_insn_mi/s` |  4.11 | -          |  4.17 |  4.02 |
| `trace_gen_time_ms   ` |  409.67 |  1,229 |  521 |  188 |
| `memory_finalize_time_ms` |  11.33 |  34 |  13 |  9 |
| `stark_prove_excluding_trace_time_ms` |  7,234.67 |  21,704 |  8,911 |  4,099 |
| `main_trace_commit_time_ms` |  1,648.33 |  4,945 |  2,114 |  818 |
| `generate_perm_trace_time_ms` |  324 |  972 |  401 |  170 |
| `perm_trace_commit_time_ms` |  1,244.33 |  3,733 |  1,535 |  673 |
| `quotient_poly_compute_time_ms` |  1,017.67 |  3,053 |  1,264 |  560 |
| `quotient_poly_commit_time_ms` |  1,054.33 |  3,163 |  1,320 |  675 |
| `pcs_opening_time_ms ` |  1,940 |  5,820 |  2,317 |  1,199 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,044 |  8,044 |  8,044 |  8,044 |
| `main_cells_used     ` |  116,589,677 |  116,589,677 |  116,589,677 |  116,589,677 |
| `total_cells_used    ` |  207,812,715 |  207,812,715 |  207,812,715 |  207,812,715 |
| `execute_preflight_insns` |  2,330,585 |  2,330,585 |  2,330,585 |  2,330,585 |
| `execute_preflight_time_ms` |  582 |  582 |  582 |  582 |
| `execute_preflight_insn_mi/s` |  5.43 | -          |  5.43 |  5.43 |
| `trace_gen_time_ms   ` |  324 |  324 |  324 |  324 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `stark_prove_excluding_trace_time_ms` |  6,051 |  6,051 |  6,051 |  6,051 |
| `main_trace_commit_time_ms` |  1,342 |  1,342 |  1,342 |  1,342 |
| `generate_perm_trace_time_ms` |  266 |  266 |  266 |  266 |
| `perm_trace_commit_time_ms` |  1,020 |  1,020 |  1,020 |  1,020 |
| `quotient_poly_compute_time_ms` |  827 |  827 |  827 |  827 |
| `quotient_poly_commit_time_ms` |  868 |  868 |  868 |  868 |
| `pcs_opening_time_ms ` |  1,721 |  1,721 |  1,721 |  1,721 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,552 |  38,552 |  38,552 |  38,552 |
| `main_cells_used     ` |  42,136,007 |  42,136,007 |  42,136,007 |  42,136,007 |
| `total_cells_used    ` |  65,007,029 |  65,007,029 |  65,007,029 |  65,007,029 |
| `execute_preflight_insns` |  779,802 |  779,802 |  779,802 |  779,802 |
| `execute_preflight_time_ms` |  162 |  162 |  162 |  162 |
| `execute_preflight_insn_mi/s` |  5.27 | -          |  5.27 |  5.27 |
| `trace_gen_time_ms   ` |  114 |  114 |  114 |  114 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  38,276 |  38,276 |  38,276 |  38,276 |
| `main_trace_commit_time_ms` |  12,304 |  12,304 |  12,304 |  12,304 |
| `generate_perm_trace_time_ms` |  87 |  87 |  87 |  87 |
| `perm_trace_commit_time_ms` |  7,574 |  7,574 |  7,574 |  7,574 |
| `quotient_poly_compute_time_ms` |  704 |  704 |  704 |  704 |
| `quotient_poly_commit_time_ms` |  13,724 |  13,724 |  13,724 |  13,724 |
| `pcs_opening_time_ms ` |  3,868 |  3,868 |  3,868 |  3,868 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,305 |  90,305 |  90,305 |  90,305 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,548 |  44,548 |  44,548 |  44,548 |



<details>
<summary>Detailed Metrics</summary>

|  | vm.create_initial_state_time_ms | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 0 | 114 | 65,007,029 | 114 | 0 | 90,319 | 44,548 | 7 | 42,136,007 | 297 | 779,802 | 5.39 | 20,934 | 39,625 | 

| group | vm.reset_state_time_ms | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 |  |  |  | 2,264 |  | 6 |  | 1 | 60 | 12,000,210 | 197.61 | 35 | 
| halo2_outer |  | 90,305 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper |  | 44,548 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  |  | 5,814 |  | 3 |  |  | 2 |  |  |  |  | 
| internal.1 |  |  |  | 8,046 |  | 3 |  |  | 2 |  |  |  |  | 
| leaf |  |  | 3,698 |  |  | 1 |  |  | 1 |  |  |  |  | 

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

| group | idx | vm.reset_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 521 | 11,503 | 371,616,121 | 472,992,226 | 521 | 8,911 | 0 | 1,229 | 1,320 | 1,525 | 2,317 | 13 | 2,114 | 207,106,479 | 401 |  | 991 | 3,422,807 | 4.14 | 
| internal.0 | 1 | 4 | 520 | 11,268 | 368,707,515 | 472,992,226 | 520 | 8,694 | 0 | 1,264 | 1,168 | 1,535 | 2,304 | 12 | 2,013 | 205,252,637 | 401 |  | 976 | 3,407,334 | 4.17 | 
| internal.0 | 2 | 5 | 188 | 5,812 | 136,144,403 | 185,031,138 | 188 | 4,099 | 0 | 560 | 675 | 673 | 1,199 | 9 | 818 | 74,652,513 | 170 |  | 441 | 1,151,564 | 4.02 | 
| internal.1 | 3 | 4 | 324 | 8,044 | 207,812,715 | 302,819,810 | 324 | 6,051 | 0 | 827 | 868 | 1,020 | 1,721 | 9 | 1,342 | 116,589,677 | 266 |  | 582 | 2,330,585 | 5.43 | 
| leaf | 0 | 0 | 180 | 4,123 | 170,628,411 | 253,173,226 | 180 | 2,433 | 0 | 255 | 239 | 572 | 715 | 9 | 471 | 72,248,817 | 178 |  | 476 | 1,260,083 | 3.96 | 
| leaf | 1 | 4 | 145 | 3,227 | 140,165,903 | 169,088,490 | 145 | 1,603 | 0 | 176 | 146 | 367 | 488 | 8 | 297 | 60,333,237 | 117 |  | 455 | 1,007,166 | 3.33 | 
| leaf | 2 | 4 | 145 | 3,301 | 140,167,543 | 169,088,490 | 145 | 1,630 | 0 | 189 | 147 | 367 | 489 | 8 | 320 | 60,333,729 | 114 |  | 461 | 1,007,207 | 3.30 | 
| leaf | 3 | 4 | 145 | 3,262 | 140,163,823 | 169,088,490 | 145 | 1,613 | 0 | 181 | 146 | 368 | 492 | 8 | 303 | 60,332,613 | 117 |  | 456 | 1,007,114 | 3.32 | 
| leaf | 4 | 4 | 145 | 3,255 | 140,166,423 | 169,088,490 | 145 | 1,593 | 0 | 168 | 145 | 370 | 492 | 10 | 297 | 60,333,393 | 115 |  | 459 | 1,007,179 | 3.33 | 
| leaf | 5 | 4 | 146 | 3,294 | 140,164,223 | 169,088,490 | 146 | 1,635 | 0 | 186 | 146 | 368 | 495 | 8 | 316 | 60,332,733 | 119 |  | 458 | 1,007,124 | 3.32 | 
| leaf | 6 | 4 | 169 | 3,696 | 158,701,791 | 208,084,458 | 169 | 2,010 | 0 | 227 | 177 | 455 | 623 | 8 | 381 | 67,172,197 | 141 |  | 454 | 1,166,520 | 3.87 | 
| root | 0 | 0 | 114 | 38,552 | 65,007,029 | 80,435,354 | 114 | 38,276 | 0 | 704 | 13,724 | 7,574 | 3,868 | 8 | 12,304 | 42,136,007 | 87 | 3 | 162 | 779,802 | 5.27 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 226 | 3,214 | 146,796,066 | 253,084,876 | 226 | 2,673 | 0 | 250 | 273 | 592 | 839 | 6 | 1 | 523 | 59,842,060 | 190 | 67 | 1,748,000 | 37.15 | 
| fib_e2e | 1 | 225 | 3,066 | 146,788,271 | 253,031,994 | 225 | 2,551 | 112 | 239 | 216 | 597 | 829 | 6 | 0 | 480 | 59,837,129 | 186 | 66 | 1,748,000 | 37.20 | 
| fib_e2e | 2 | 225 | 3,092 | 146,788,335 | 253,031,994 | 225 | 2,583 | 112 | 245 | 209 | 588 | 836 | 6 | 0 | 489 | 59,837,145 | 210 | 67 | 1,748,000 | 37.19 | 
| fib_e2e | 3 | 225 | 3,045 | 146,788,358 | 253,031,994 | 225 | 2,544 | 112 | 240 | 211 | 590 | 841 | 6 | 0 | 471 | 59,837,156 | 184 | 67 | 1,748,000 | 37.24 | 
| fib_e2e | 4 | 226 | 3,063 | 146,788,271 | 253,031,994 | 226 | 2,556 | 113 | 246 | 209 | 589 | 830 | 6 | 0 | 490 | 59,837,129 | 187 | 68 | 1,748,000 | 36.69 | 
| fib_e2e | 5 | 227 | 3,089 | 146,788,335 | 253,031,994 | 227 | 2,585 | 113 | 252 | 209 | 594 | 835 | 6 | 0 | 475 | 59,837,145 | 215 | 67 | 1,748,000 | 37.08 | 
| fib_e2e | 6 | 203 | 2,264 | 128,298,001 | 160,813,290 | 203 | 1,796 | 97 | 174 | 171 | 377 | 599 | 6 | 0 | 342 | 51,906,075 | 128 | 54 | 1,512,210 | 37.15 | 

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


Commit: https://github.com/openvm-org/openvm/commit/ed00aedaae96866ba69f76c17df20c507bba3ccb

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16972639651)
