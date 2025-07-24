| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  241.68 |  198.65 |
| fib_e2e |  18.64 |  2.87 |
| leaf |  16.12 |  2.96 |
| internal.0 |  24.17 |  10.07 |
| internal.1 |  6.56 |  6.56 |
| root |  38.66 |  38.66 |
| halo2_outer |  93.28 |  93.28 |
| halo2_wrapper |  44.18 |  44.18 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,663.57 |  18,645 |  2,872 |  1,989 |
| `main_cells_used     ` |  58,706,920.43 |  410,948,443 |  59,844,647 |  51,910,452 |
| `total_cells_used    ` |  144,151,794.43 |  1,009,062,561 |  146,800,305 |  128,306,942 |
| `insns               ` |  3,000,069.50 |  24,000,556 |  12,000,278 |  1,512,278 |
| `execute_metered_time_ms` |  70 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  170.48 | -          |  170.48 |  170.48 |
| `execute_e3_time_ms  ` |  77.86 |  545 |  81 |  68 |
| `execute_e3_insn_mi/s` |  21.92 | -          |  22.06 |  21.53 |
| `trace_gen_time_ms   ` |  248.43 |  1,739 |  253 |  229 |
| `memory_finalize_time_ms` |  0.43 |  3 |  1 |  0 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  56.71 |  397 |  58 |  55 |
| `stark_prove_excluding_trace_time_ms` |  2,337.29 |  16,361 |  2,538 |  1,692 |
| `main_trace_commit_time_ms` |  417.71 |  2,924 |  471 |  308 |
| `generate_perm_trace_time_ms` |  190 |  1,330 |  216 |  124 |
| `perm_trace_commit_time_ms` |  500.71 |  3,505 |  548 |  331 |
| `quotient_poly_compute_time_ms` |  231.71 |  1,622 |  253 |  162 |
| `quotient_poly_commit_time_ms` |  197.29 |  1,381 |  259 |  158 |
| `pcs_opening_time_ms ` |  794.43 |  5,561 |  830 |  604 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,302.86 |  16,120 |  2,958 |  2,114 |
| `main_cells_used     ` |  62,253,667.43 |  435,775,672 |  72,208,359 |  59,356,877 |
| `total_cells_used    ` |  145,351,448.86 |  1,017,460,142 |  170,453,693 |  137,856,387 |
| `insns               ` |  1,066,087 |  7,462,609 |  1,260,142 |  1,007,167 |
| `execute_e3_time_ms  ` |  443.14 |  3,102 |  472 |  435 |
| `execute_e3_insn_mi/s` |  2.40 | -          |  2.67 |  2.28 |
| `trace_gen_time_ms   ` |  158.57 |  1,110 |  187 |  149 |
| `memory_finalize_time_ms` |  7.86 |  55 |  10 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  1,701.14 |  11,908 |  2,299 |  1,524 |
| `main_trace_commit_time_ms` |  310.71 |  2,175 |  408 |  275 |
| `generate_perm_trace_time_ms` |  127.86 |  895 |  181 |  113 |
| `perm_trace_commit_time_ms` |  364.86 |  2,554 |  504 |  324 |
| `quotient_poly_compute_time_ms` |  195.71 |  1,370 |  256 |  174 |
| `quotient_poly_commit_time_ms` |  156.57 |  1,096 |  231 |  135 |
| `pcs_opening_time_ms ` |  540.29 |  3,782 |  714 |  488 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,056.33 |  24,169 |  10,068 |  4,389 |
| `main_cells_used     ` |  157,055,274.33 |  471,165,823 |  201,499,009 |  70,097,913 |
| `total_cells_used    ` |  281,474,976.33 |  844,424,929 |  360,287,507 |  126,913,051 |
| `insns               ` |  2,660,631 |  7,981,893 |  3,422,867 |  1,151,551 |
| `execute_e3_time_ms  ` |  953.33 |  2,860 |  1,225 |  425 |
| `execute_e3_insn_mi/s` |  2.77 | -          |  2.81 |  2.70 |
| `trace_gen_time_ms   ` |  395.33 |  1,186 |  502 |  182 |
| `memory_finalize_time_ms` |  9 |  27 |  11 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  6,707.67 |  20,123 |  8,341 |  3,782 |
| `main_trace_commit_time_ms` |  1,443.33 |  4,330 |  1,853 |  722 |
| `generate_perm_trace_time_ms` |  326.67 |  980 |  412 |  171 |
| `perm_trace_commit_time_ms` |  1,102 |  3,306 |  1,362 |  590 |
| `quotient_poly_compute_time_ms` |  992.33 |  2,977 |  1,238 |  540 |
| `quotient_poly_commit_time_ms` |  937.67 |  2,813 |  1,198 |  565 |
| `pcs_opening_time_ms ` |  1,899.67 |  5,699 |  2,274 |  1,186 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,556 |  6,556 |  6,556 |  6,556 |
| `main_cells_used     ` |  115,991,897 |  115,991,897 |  115,991,897 |  115,991,897 |
| `total_cells_used    ` |  206,606,327 |  206,606,327 |  206,606,327 |  206,606,327 |
| `insns               ` |  2,330,638 |  2,330,638 |  2,330,638 |  2,330,638 |
| `execute_e3_time_ms  ` |  647 |  647 |  647 |  647 |
| `execute_e3_insn_mi/s` |  3.60 | -          |  3.60 |  3.60 |
| `trace_gen_time_ms   ` |  313 |  313 |  313 |  313 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  5,596 |  5,596 |  5,596 |  5,596 |
| `main_trace_commit_time_ms` |  1,160 |  1,160 |  1,160 |  1,160 |
| `generate_perm_trace_time_ms` |  262 |  262 |  262 |  262 |
| `perm_trace_commit_time_ms` |  890 |  890 |  890 |  890 |
| `quotient_poly_compute_time_ms` |  797 |  797 |  797 |  797 |
| `quotient_poly_commit_time_ms` |  793 |  793 |  793 |  793 |
| `pcs_opening_time_ms ` |  1,688 |  1,688 |  1,688 |  1,688 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,660 |  38,660 |  38,660 |  38,660 |
| `main_cells_used     ` |  40,667,595 |  40,667,595 |  40,667,595 |  40,667,595 |
| `total_cells_used    ` |  62,537,737 |  62,537,737 |  62,537,737 |  62,537,737 |
| `insns               ` |  779,782 |  779,782 |  779,782 |  779,782 |
| `execute_e3_time_ms  ` |  221 |  221 |  221 |  221 |
| `execute_e3_insn_mi/s` |  3.53 | -          |  3.53 |  3.53 |
| `trace_gen_time_ms   ` |  111 |  111 |  111 |  111 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  38,328 |  38,328 |  38,328 |  38,328 |
| `main_trace_commit_time_ms` |  12,306 |  12,306 |  12,306 |  12,306 |
| `generate_perm_trace_time_ms` |  88 |  88 |  88 |  88 |
| `perm_trace_commit_time_ms` |  7,571 |  7,571 |  7,571 |  7,571 |
| `quotient_poly_compute_time_ms` |  703 |  703 |  703 |  703 |
| `quotient_poly_commit_time_ms` |  13,722 |  13,722 |  13,722 |  13,722 |
| `pcs_opening_time_ms ` |  3,927 |  3,927 |  3,927 |  3,927 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  93,277 |  93,277 |  93,277 |  93,277 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,184 |  44,184 |  44,184 |  44,184 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 111 | 62,537,737 | 93,289 | 44,184 | 7 | 40,667,595 | 779,782 | 225 | 3.46 | 0 | 27,820 | 39,755 | 

| group | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e |  |  |  | 3,276 |  | 24 |  | 12,000,278 | 1 | 70 | 170.48 | 56 | 
| halo2_outer | 93,277 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper | 44,184 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  | 5,618 |  | 3 |  |  |  | 2 |  |  |  | 
| internal.1 |  |  | 7,789 |  | 3 |  |  |  | 2 |  |  |  | 
| leaf |  | 3,710 |  |  | 1 |  |  |  | 1 |  |  |  | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 502 | 10,068 | 360,287,507 | 472,992,226 | 8,341 | 1,238 | 1,198 | 1,362 | 2,274 | 11 | 1,853 | 201,499,009 | 3,422,867 | 412 |  | 1,225 | 2.79 | 0 | 
| internal.0 | 1 | 502 | 9,712 | 357,224,371 | 472,992,226 | 8,000 | 1,199 | 1,050 | 1,354 | 2,239 | 9 | 1,755 | 199,568,901 | 3,407,475 | 397 |  | 1,210 | 2.81 | 0 | 
| internal.0 | 2 | 182 | 4,389 | 126,913,051 | 185,031,138 | 3,782 | 540 | 565 | 590 | 1,186 | 7 | 722 | 70,097,913 | 1,151,551 | 171 |  | 425 | 2.70 | 0 | 
| internal.1 | 3 | 313 | 6,556 | 206,606,327 | 302,819,810 | 5,596 | 797 | 793 | 890 | 1,688 | 8 | 1,160 | 115,991,897 | 2,330,638 | 262 |  | 647 | 3.60 | 0 | 
| leaf | 0 | 187 | 2,958 | 170,453,693 | 253,173,226 | 2,299 | 256 | 231 | 504 | 714 | 7 | 408 | 72,208,359 | 1,260,142 | 181 |  | 472 | 2.67 | 0 | 
| leaf | 1 | 150 | 2,114 | 137,857,307 | 169,088,490 | 1,524 | 174 | 138 | 325 | 493 | 7 | 277 | 59,357,153 | 1,007,190 | 114 |  | 440 | 2.28 | 0 | 
| leaf | 2 | 150 | 2,137 | 137,856,507 | 169,088,490 | 1,548 | 177 | 137 | 349 | 488 | 8 | 279 | 59,356,913 | 1,007,170 | 113 |  | 439 | 2.29 | 0 | 
| leaf | 3 | 149 | 2,130 | 137,856,387 | 169,088,490 | 1,545 | 177 | 136 | 325 | 491 | 8 | 292 | 59,356,877 | 1,007,167 | 118 |  | 436 | 2.31 | 0 | 
| leaf | 4 | 150 | 2,131 | 137,859,387 | 169,088,490 | 1,546 | 185 | 138 | 324 | 490 | 7 | 289 | 59,357,777 | 1,007,242 | 115 |  | 435 | 2.31 | 0 | 
| leaf | 5 | 150 | 2,116 | 137,858,547 | 169,088,490 | 1,528 | 180 | 135 | 326 | 491 | 10 | 275 | 59,357,525 | 1,007,221 | 115 |  | 438 | 2.30 | 0 | 
| leaf | 6 | 174 | 2,534 | 157,718,314 | 208,084,458 | 1,918 | 221 | 181 | 401 | 615 | 8 | 355 | 66,781,068 | 1,166,477 | 139 |  | 442 | 2.63 | 0 | 
| root | 0 | 111 | 38,660 | 62,537,737 | 80,435,354 | 38,328 | 703 | 13,722 | 7,571 | 3,927 | 7 | 12,306 | 40,667,595 | 779,782 | 88 | 3 | 221 | 3.53 | 0 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 253 | 2,872 | 146,800,305 | 253,085,900 | 2,538 | 238 | 259 | 548 | 830 | 58 | 24 | 1 | 471 | 59,844,647 | 1,748,000 | 186 | 81 | 21.53 | 0 | 
| fib_e2e | 1 | 252 | 2,733 | 146,791,066 | 253,031,994 | 2,402 | 236 | 190 | 538 | 827 | 57 | 24 | 0 | 417 | 59,838,672 | 1,748,000 | 188 | 79 | 22.06 | 0 | 
| fib_e2e | 2 | 252 | 2,741 | 146,791,058 | 253,031,994 | 2,410 | 250 | 189 | 516 | 827 | 57 | 24 | 1 | 434 | 59,838,664 | 1,748,000 | 187 | 79 | 21.87 | 0 | 
| fib_e2e | 3 | 251 | 2,743 | 146,791,066 | 253,031,994 | 2,413 | 243 | 193 | 517 | 821 | 56 | 24 | 0 | 421 | 59,838,672 | 1,748,000 | 215 | 79 | 22.04 | 0 | 
| fib_e2e | 4 | 251 | 2,770 | 146,791,066 | 253,031,994 | 2,440 | 253 | 194 | 515 | 827 | 56 | 24 | 0 | 433 | 59,838,672 | 1,748,000 | 214 | 79 | 22.06 | 0 | 
| fib_e2e | 5 | 251 | 2,797 | 146,791,058 | 253,031,994 | 2,466 | 240 | 198 | 540 | 825 | 55 | 24 | 0 | 440 | 59,838,664 | 1,748,000 | 216 | 80 | 21.81 | 0 | 
| fib_e2e | 6 | 229 | 1,989 | 128,306,942 | 160,892,330 | 1,692 | 162 | 158 | 331 | 604 | 58 | 24 | 1 | 308 | 51,910,452 | 1,512,278 | 124 | 68 | 22.04 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fib_e2e | 0 | 0 | 6,029,454 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 17,039,944 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 3,014,727 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 17,039,948 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 12,451,924 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 56,503,101 | 2,013,265,921 | 
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
| fib_e2e | 6 | 0 | 3,932,344 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,632 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,172 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,700 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,209,020 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,532,228 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b269dbbacf2ed2cfca8d41ab61e4af8d5dc4e6ff

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16488673058)
