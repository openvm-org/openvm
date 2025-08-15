| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  218.86 |  218.86 |
| kitchen_sink |  15.45 |  15.45 |
| leaf |  23.61 |  23.61 |
| internal.0 |  6.34 |  6.34 |
| root |  38.56 |  38.56 |
| halo2_outer |  90.68 |  90.68 |
| halo2_wrapper |  44.20 |  44.20 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,452 |  15,452 |  15,452 |  15,452 |
| `main_cells_used     ` |  904,738,232 |  904,738,232 |  904,738,232 |  904,738,232 |
| `total_cells_used    ` |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |
| `execute_metered_time_ms` |  10 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  14.56 | -          |  14.56 |  14.56 |
| `execute_preflight_insns` |  153,644 |  153,644 |  153,644 |  153,644 |
| `execute_preflight_time_ms` |  103 |  103 |  103 |  103 |
| `execute_preflight_insn_mi/s` |  7.35 | -          |  7.35 |  7.35 |
| `trace_gen_time_ms   ` |  209 |  209 |  209 |  209 |
| `memory_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `stark_prove_excluding_trace_time_ms` |  14,892 |  14,892 |  14,892 |  14,892 |
| `main_trace_commit_time_ms` |  5,080 |  5,080 |  5,080 |  5,080 |
| `generate_perm_trace_time_ms` |  454 |  454 |  454 |  454 |
| `perm_trace_commit_time_ms` |  1,527 |  1,527 |  1,527 |  1,527 |
| `quotient_poly_compute_time_ms` |  6,253 |  6,253 |  6,253 |  6,253 |
| `quotient_poly_commit_time_ms` |  309 |  309 |  309 |  309 |
| `pcs_opening_time_ms ` |  1,221 |  1,221 |  1,221 |  1,221 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  23,615 |  23,615 |  23,615 |  23,615 |
| `main_cells_used     ` |  769,361,834 |  769,361,834 |  769,361,834 |  769,361,834 |
| `total_cells_used    ` |  2,050,263,476 |  2,050,263,476 |  2,050,263,476 |  2,050,263,476 |
| `execute_preflight_insns` |  7,903,874 |  7,903,874 |  7,903,874 |  7,903,874 |
| `execute_preflight_time_ms` |  1,307 |  1,307 |  1,307 |  1,307 |
| `execute_preflight_insn_mi/s` |  6.95 | -          |  6.95 |  6.95 |
| `trace_gen_time_ms   ` |  1,693 |  1,693 |  1,693 |  1,693 |
| `memory_finalize_time_ms` |  18 |  18 |  18 |  18 |
| `stark_prove_excluding_trace_time_ms` |  19,586 |  19,586 |  19,586 |  19,586 |
| `main_trace_commit_time_ms` |  4,217 |  4,217 |  4,217 |  4,217 |
| `generate_perm_trace_time_ms` |  1,682 |  1,682 |  1,682 |  1,682 |
| `perm_trace_commit_time_ms` |  5,484 |  5,484 |  5,484 |  5,484 |
| `quotient_poly_compute_time_ms` |  2,148 |  2,148 |  2,148 |  2,148 |
| `quotient_poly_commit_time_ms` |  1,454 |  1,454 |  1,454 |  1,454 |
| `pcs_opening_time_ms ` |  4,594 |  4,594 |  4,594 |  4,594 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,338 |  6,338 |  6,338 |  6,338 |
| `main_cells_used     ` |  79,527,399 |  79,527,399 |  79,527,399 |  79,527,399 |
| `total_cells_used    ` |  143,464,701 |  143,464,701 |  143,464,701 |  143,464,701 |
| `execute_preflight_insns` |  1,197,779 |  1,197,779 |  1,197,779 |  1,197,779 |
| `execute_preflight_time_ms` |  495 |  495 |  495 |  495 |
| `execute_preflight_insn_mi/s` |  3.59 | -          |  3.59 |  3.59 |
| `trace_gen_time_ms   ` |  184 |  184 |  184 |  184 |
| `memory_finalize_time_ms` |  13 |  13 |  13 |  13 |
| `stark_prove_excluding_trace_time_ms` |  4,628 |  4,628 |  4,628 |  4,628 |
| `main_trace_commit_time_ms` |  1,018 |  1,018 |  1,018 |  1,018 |
| `generate_perm_trace_time_ms` |  190 |  190 |  190 |  190 |
| `perm_trace_commit_time_ms` |  748 |  748 |  748 |  748 |
| `quotient_poly_compute_time_ms` |  617 |  617 |  617 |  617 |
| `quotient_poly_commit_time_ms` |  740 |  740 |  740 |  740 |
| `pcs_opening_time_ms ` |  1,310 |  1,310 |  1,310 |  1,310 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,558 |  38,558 |  38,558 |  38,558 |
| `main_cells_used     ` |  41,517,678 |  41,517,678 |  41,517,678 |  41,517,678 |
| `total_cells_used    ` |  64,199,884 |  64,199,884 |  64,199,884 |  64,199,884 |
| `execute_preflight_insns` |  772,529 |  772,529 |  772,529 |  772,529 |
| `execute_preflight_time_ms` |  162 |  162 |  162 |  162 |
| `execute_preflight_insn_mi/s` |  5.24 | -          |  5.24 |  5.24 |
| `trace_gen_time_ms   ` |  113 |  113 |  113 |  113 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  38,283 |  38,283 |  38,283 |  38,283 |
| `main_trace_commit_time_ms` |  12,323 |  12,323 |  12,323 |  12,323 |
| `generate_perm_trace_time_ms` |  86 |  86 |  86 |  86 |
| `perm_trace_commit_time_ms` |  7,580 |  7,580 |  7,580 |  7,580 |
| `quotient_poly_compute_time_ms` |  728 |  728 |  728 |  728 |
| `quotient_poly_commit_time_ms` |  13,630 |  13,630 |  13,630 |  13,630 |
| `pcs_opening_time_ms ` |  3,902 |  3,902 |  3,902 |  3,902 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,681 |  90,681 |  90,681 |  90,681 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,204 |  44,204 |  44,204 |  44,204 |

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,538 |  38,538 |  38,538 |  38,538 |
| `main_cells_used     ` |  41,528,500 |  41,528,500 |  41,528,500 |  41,528,500 |
| `total_cells_used    ` |  64,211,846 |  64,211,846 |  64,211,846 |  64,211,846 |
| `execute_preflight_insns` |  772,316 |  772,316 |  772,316 |  772,316 |
| `execute_preflight_time_ms` |  160 |  160 |  160 |  160 |
| `execute_preflight_insn_mi/s` |  5.29 | -          |  5.29 |  5.29 |
| `trace_gen_time_ms   ` |  114 |  114 |  114 |  114 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `stark_prove_excluding_trace_time_ms` |  38,264 |  38,264 |  38,264 |  38,264 |
| `main_trace_commit_time_ms` |  12,320 |  12,320 |  12,320 |  12,320 |
| `generate_perm_trace_time_ms` |  89 |  89 |  89 |  89 |
| `perm_trace_commit_time_ms` |  7,592 |  7,592 |  7,592 |  7,592 |
| `quotient_poly_compute_time_ms` |  689 |  689 |  689 |  689 |
| `quotient_poly_commit_time_ms` |  13,657 |  13,657 |  13,657 |  13,657 |
| `pcs_opening_time_ms ` |  3,898 |  3,898 |  3,898 |  3,898 |



<details>
<summary>Detailed Metrics</summary>

|  | vm.create_initial_state_time_ms | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_cells_used | keygen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 0 | 114 | 64,199,884 | 114 | 0 | 90,695 | 44,204 | 23 | 8 | 41,517,678 | 172,792 | 324 | 772,529 | 5.31 | 15,502 | 39,593 | 

| group | vm.reset_state_time_ms | vm.create_initial_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_children | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | halo2_total_cells | halo2_keygen_time_ms | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_keygen | 0 | 0 | 114 | 38,538 | 64,211,846 | 80,435,354 | 114 | 38,264 | 0 |  |  | 689 | 13,657 |  | 7,592 | 3,898 |  |  | 7 | 12,320 | 41,528,500 | 5,447,564 | 19,019 | 89 |  | 160 | 772,316 | 5.29 |  |  |  |  | 
| halo2_outer |  |  |  | 90,681 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 65,627,358 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  |  |  | 44,204 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  |  |  |  |  |  |  |  |  | 6,339 |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| kitchen_sink | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 15,452 |  |  |  | 7 |  |  |  |  |  |  | 1 |  |  |  | 10 | 153,644 | 14.56 | 37 | 
| leaf |  |  |  |  |  |  |  |  |  | 23,622 |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| halo2_keygen | AccessAdapterAir<2> | 262,144 |  | 8 | 11 | 4,980,736 | 
| halo2_keygen | AccessAdapterAir<4> | 131,072 |  | 8 | 13 | 2,752,512 | 
| halo2_keygen | AccessAdapterAir<8> | 4,096 |  | 8 | 17 | 102,400 | 
| halo2_keygen | FriReducedOpeningAir | 131,072 |  | 24 | 27 | 6,684,672 | 
| halo2_keygen | JalRangeCheckAir | 32,768 |  | 12 | 12 | 786,432 | 
| halo2_keygen | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 84 | 398 | 15,794,176 | 
| halo2_keygen | PhantomAir | 8,192 |  | 8 | 6 | 114,688 | 
| halo2_keygen | ProgramAir | 131,072 |  | 8 | 10 | 2,359,296 | 
| halo2_keygen | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| halo2_keygen | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 524,288 |  | 12 | 29 | 21,495,808 | 
| halo2_keygen | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 12 | 23 | 4,587,520 | 
| halo2_keygen | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 64 |  | 12 | 22 | 2,176 | 
| halo2_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 16 | 21 | 9,699,328 | 
| halo2_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 65,536 |  | 16 | 27 | 2,818,048 | 
| halo2_keygen | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 65,536 |  | 12 | 38 | 3,276,800 | 
| halo2_keygen | VmConnectorAir | 2 | 1 | 8 | 5 | 26 | 
| halo2_keygen | VolatileBoundaryAir | 131,072 |  | 8 | 12 | 2,621,440 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| leaf | AccessAdapterAir<2> | 0 | 4,194,304 |  | 16 | 11 | 113,246,208 | 
| leaf | AccessAdapterAir<4> | 0 | 2,097,152 |  | 16 | 13 | 60,817,408 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | FriReducedOpeningAir | 0 | 8,388,608 |  | 84 | 27 | 931,135,488 | 
| leaf | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 1,048,576 |  | 312 | 398 | 744,488,960 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 36 | 29 | 272,629,760 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 40 | 21 | 127,926,272 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 40 | 27 | 35,127,296 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 1,048,576 |  | 36 | 38 | 77,594,624 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
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
| kitchen_sink | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| kitchen_sink | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| kitchen_sink | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| kitchen_sink | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | KeccakVmAir | 0 | 262,144 |  | 1,056 | 3,163 | 1,105,985,536 | 
| kitchen_sink | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| kitchen_sink | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| kitchen_sink | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| kitchen_sink | RangeTupleCheckerAir<2> | 0 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | Sha256VmAir | 0 | 524,288 |  | 108 | 470 | 303,038,464 | 
| kitchen_sink | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 32,768 |  | 52 | 36 | 2,883,584 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,192 |  | 28 | 26 | 442,368 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| kitchen_sink | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,024 |  | 28 | 18 | 47,104 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 2,048 |  | 192 | 168 | 737,280 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 1,024 |  | 68 | 169 | 242,688 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 256 |  | 192 | 164 | 91,136 | 
| kitchen_sink | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 256 |  | 48 | 124 | 44,032 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 8 |  | 88 | 242 | 2,640 | 
| kitchen_sink | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,048 |  | 36 | 28 | 131,072 | 
| kitchen_sink | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 1,668 | 1,020 | 10,752 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 384 | 294 | 41,920 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 860 | 625 | 2,202 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 496 | 393 | 2,404 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 | 949 | 3,426 | 
| kitchen_sink | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | cell_tracker_span | simple_advice_cells | lookup_advice_cells | fixed_cells |
| --- | --- | --- | --- | --- |
| halo2_keygen | VerifierProgram | 509,456 | 164,237 | 166,961 | 
| halo2_keygen | VerifierProgram;CheckTraceHeightConstraints | 5,316 | 1,125 | 1,942 | 
| halo2_keygen | VerifierProgram;PoseidonCell | 29,400 |  | 8,700 | 
| halo2_keygen | VerifierProgram;stage-c-build-rounds | 18,401 | 2,528 | 6,510 | 
| halo2_keygen | VerifierProgram;stage-c-build-rounds;PoseidonCell | 46,550 |  | 13,775 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs | 1,280,292 | 197,458 | 466,987 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;PoseidonCell | 3,839,150 |  | 1,136,075 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 40,526 | 4,276 | 18,076 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 56,350 |  | 16,675 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;cache-generator-powers | 70,410 | 12,000 | 21,630 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 8,549,550 | 353,940 | 1,581,960 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;pre-compute-rounds-context | 76,224 | 11,116 | 22,232 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 53,280 |  | 6,660 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 9,926,550 |  | 2,940,300 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 8,854,140 | 253,980 | 2,764,710 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 1,088,820 | 184,470 | 307,410 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 109,440 |  | 13,680 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 16,764,840 |  | 4,965,840 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 1,671,570 | 62,940 | 513,270 | 
| halo2_keygen | VerifierProgram;stage-e-verify-constraints | 9,499,973 | 1,889,049 | 2,918,862 | 

| group | idx | vm.reset_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 184 | 6,338 | 143,464,701 | 224,975,330 | 184 | 4,628 | 0 | 617 | 740 | 748 | 1,310 | 13 | 1,018 | 79,527,399 | 190 |  | 495 | 1,197,779 | 3.59 | 
| leaf | 0 | 0 | 1,693 | 23,615 | 2,050,263,476 | 2,500,267,498 | 1,693 | 19,586 | 0 | 2,148 | 1,454 | 5,484 | 4,594 | 18 | 4,217 | 769,361,834 | 1,682 |  | 1,307 | 7,903,874 | 6.95 | 
| root | 0 | 0 | 113 | 38,558 | 64,199,884 | 80,435,354 | 113 | 38,283 | 0 | 728 | 13,630 | 7,580 | 3,902 | 8 | 12,323 | 41,517,678 | 86 | 3 | 162 | 772,529 | 5.24 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 5,177,476 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 30,814,464 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 2,588,738 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 30,941,444 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 70,177,482 | 2,013,265,921 | 
| leaf | 0 | 0 | 39,125,124 | 2,013,265,921 | 
| leaf | 0 | 1 | 291,111,168 | 2,013,265,921 | 
| leaf | 0 | 2 | 19,562,562 | 2,013,265,921 | 
| leaf | 0 | 3 | 288,096,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 642,351,818 | 2,013,265,921 | 
| root | 0 | 0 | 2,572,420 | 2,013,265,921 | 
| root | 0 | 1 | 12,005,632 | 2,013,265,921 | 
| root | 0 | 2 | 1,286,210 | 2,013,265,921 | 
| root | 0 | 3 | 12,067,076 | 2,013,265,921 | 
| root | 0 | 4 | 65,536 | 2,013,265,921 | 
| root | 0 | 5 | 28,390,090 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 209 | 15,452 | 1,226,975,002 | 1,481,193,346 | 209 | 14,892 | 3 | 6,253 | 309 | 1,527 | 1,221 | 6 | 2 | 5,080 | 904,738,232 | 454 | 103 | 153,644 | 7.35 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,977,944 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 32,428,632 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 988,972 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 32,011,136 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 57,344 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 24,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 49,612,036 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 1,048,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,320 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 120,668,384 | 2,013,265,921 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| halo2_keygen | 0 | 2,572,420 | 2,013,265,921 | 
| halo2_keygen | 1 | 12,005,632 | 2,013,265,921 | 
| halo2_keygen | 2 | 1,286,210 | 2,013,265,921 | 
| halo2_keygen | 3 | 12,067,076 | 2,013,265,921 | 
| halo2_keygen | 4 | 65,536 | 2,013,265,921 | 
| halo2_keygen | 5 | 28,390,090 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4bc7f056dbb8e710a89a34fdd78fa01a78a13dd7

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16993062652)
