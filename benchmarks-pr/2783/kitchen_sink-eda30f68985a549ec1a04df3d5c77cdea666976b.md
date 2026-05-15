| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  17.12 |  12.71 | 12.71 |
| kitchen_sink |  3.50 |  2.07 |  2.07 |
| leaf |  10.77 |  7.80 |  7.80 |
| internal.0 |  2.84 |  2.84 |  2.84 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,748 |  3,496 |  2,060 |  1,436 |
| `main_cells_used     ` |  9,168,104 |  18,336,208 |  10,084,970 |  8,251,238 |
| `total_cells_used    ` |  38,025,394 |  76,050,788 |  39,702,092 |  36,348,696 |
| `execute_metered_time_ms` |  7 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  20.18 | -          |  20.18 |  20.18 |
| `execute_preflight_insns` |  77,381.50 |  154,763 |  121,000 |  33,763 |
| `execute_preflight_time_ms` |  117 |  234 |  137 |  97 |
| `execute_preflight_insn_mi/s` |  8.55 | -          |  11.18 |  5.91 |
| `trace_gen_time_ms   ` |  135 |  270 |  139 |  131 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  1,405 |  2,810 |  1,645 |  1,165 |
| `main_trace_commit_time_ms` |  192 |  384 |  204 |  180 |
| `generate_perm_trace_time_ms` |  75.50 |  151 |  109 |  42 |
| `perm_trace_commit_time_ms` |  80.35 |  160.70 |  96.38 |  64.32 |
| `quotient_poly_compute_time_ms` |  372.56 |  745.11 |  451.90 |  293.22 |
| `quotient_poly_commit_time_ms` |  9.37 |  18.73 |  10.25 |  8.48 |
| `pcs_opening_time_ms ` |  657.50 |  1,315 |  746 |  569 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,386 |  10,772 |  7,799 |  2,973 |
| `main_cells_used     ` |  62,635,775 |  125,271,550 |  83,660,740 |  41,610,810 |
| `total_cells_used    ` |  143,447,889 |  286,895,778 |  193,635,846 |  93,259,932 |
| `execute_preflight_insns` |  6,292,807.50 |  12,585,615 |  9,313,806 |  3,271,809 |
| `execute_preflight_time_ms` |  1,958 |  3,916 |  2,503 |  1,413 |
| `execute_preflight_insn_mi/s` |  5.92 | -          |  6.07 |  5.77 |
| `trace_gen_time_ms   ` |  399 |  798 |  585 |  213 |
| `memory_finalize_time_ms` |  31.50 |  63 |  39 |  24 |
| `stark_prove_excluding_trace_time_ms` |  3,027 |  6,054 |  4,709 |  1,345 |
| `main_trace_commit_time_ms` |  453 |  906 |  751 |  155 |
| `generate_perm_trace_time_ms` |  218 |  436 |  391 |  45 |
| `perm_trace_commit_time_ms` |  936.100 |  1,873.99 |  1,622.85 |  251.14 |
| `quotient_poly_compute_time_ms` |  340.24 |  680.49 |  512.77 |  167.72 |
| `quotient_poly_commit_time_ms` |  100.99 |  201.98 |  171.57 |  30.41 |
| `pcs_opening_time_ms ` |  962.50 |  1,925 |  1,233 |  692 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,842 |  2,842 |  2,842 |  2,842 |
| `main_cells_used     ` |  20,605,812 |  20,605,812 |  20,605,812 |  20,605,812 |
| `total_cells_used    ` |  43,733,898 |  43,733,898 |  43,733,898 |  43,733,898 |
| `execute_preflight_insns` |  4,044,755 |  4,044,755 |  4,044,755 |  4,044,755 |
| `execute_preflight_time_ms` |  1,504 |  1,504 |  1,504 |  1,504 |
| `execute_preflight_insn_mi/s` |  6.36 | -          |  6.36 |  6.36 |
| `trace_gen_time_ms   ` |  157 |  157 |  157 |  157 |
| `memory_finalize_time_ms` |  15 |  15 |  15 |  15 |
| `stark_prove_excluding_trace_time_ms` |  1,179 |  1,179 |  1,179 |  1,179 |
| `main_trace_commit_time_ms` |  259 |  259 |  259 |  259 |
| `generate_perm_trace_time_ms` |  40 |  40 |  40 |  40 |
| `perm_trace_commit_time_ms` |  189.11 |  189.11 |  189.11 |  189.11 |
| `quotient_poly_compute_time_ms` |  276.55 |  276.55 |  276.55 |  276.55 |
| `quotient_poly_commit_time_ms` |  74.55 |  74.55 |  74.55 |  74.55 |
| `pcs_opening_time_ms ` |  336 |  336 |  336 |  336 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | app_prove_time_ms | agg_layer_time_ms | AppExecutionCommit::compute_time_ms |
| --- | --- | --- | --- |
|  | 60 | 3,522 | 2,851 | 162 | 

| group | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 |  | 2,849 |  | 3 | 2 |  |  |  |  | 
| kitchen_sink |  |  | 1,436 |  | 1 | 7 | 154,763 | 20.18 | 0 | 
| leaf | 2,978 |  |  | 1 | 1 |  |  |  |  | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 1 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | 11 | JalRangeCheckAir | 0 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | 15 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 3 | VolatileBoundaryAir | 0 | 524,288 |  | 12 | 12 | 12,582,912 | 
| internal.0 | 4 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | 5 | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | 6 | AccessAdapterAir<8> | 0 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 524,288 |  | 160 | 398 | 292,552,704 | 
| internal.0 | 8 | FriReducedOpeningAir | 0 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 524,288 |  | 20 | 38 | 30,408,704 | 
| leaf | 0 | ProgramAir | 0 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | 0 | ProgramAir | 1 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 8,388,608 |  | 36 | 29 | 545,259,520 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 1 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 40 | 27 | 35,127,296 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 40 | 21 | 127,926,272 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | 15 | PhantomAir | 0 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 4,194,304 |  | 16 | 11 | 113,246,208 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 2,097,152 |  | 16 | 13 | 60,817,408 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 1,048,576 |  | 312 | 398 | 744,488,960 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 524,288 |  | 312 | 398 | 372,244,480 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 16,777,216 |  | 84 | 27 | 1,862,270,976 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 4,194,304 |  | 84 | 27 | 465,567,744 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 1,048,576 |  | 36 | 38 | 77,594,624 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| kitchen_sink | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| kitchen_sink | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 1 |  | 1,340 | 949 | 2,289 | 
| kitchen_sink | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 860 | 625 | 1,485 | 
| kitchen_sink | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | 14 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 860 | 625 | 1,485 | 
| kitchen_sink | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | 16 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 860 | 625 | 1,485 | 
| kitchen_sink | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 956 | 757 | 3,426 | 
| kitchen_sink | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 572 | 565 | 2,274 | 
| kitchen_sink | 19 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 604 | 497 | 2,202 | 
| kitchen_sink | 2 | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| kitchen_sink | 20 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 348 | 369 | 1,434 | 
| kitchen_sink | 21 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4 |  | 56 | 166 | 888 | 
| kitchen_sink | 22 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 368 | 287 | 41,920 | 
| kitchen_sink | 23 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 240 | 223 | 29,632 | 
| kitchen_sink | 24 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 56 | 166 | 444 | 
| kitchen_sink | 25 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 26 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 192 | 199 | 782 | 
| kitchen_sink | 27 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 8 |  | 88 | 242 | 2,640 | 
| kitchen_sink | 28 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 496 | 393 | 1,778 | 
| kitchen_sink | 29 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 304 | 297 | 2,404 | 
| kitchen_sink | 3 | MemoryMerkleAir<8> | 0 | 8,192 |  | 16 | 32 | 393,216 | 
| kitchen_sink | 30 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 56 | 166 | 444 | 
| kitchen_sink | 31 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 32 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 192 | 199 | 782 | 
| kitchen_sink | 33 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | 34 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 35 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 192 | 199 | 1,564 | 
| kitchen_sink | 36 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 56 | 166 | 444 | 
| kitchen_sink | 37 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 38 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 192 | 199 | 782 | 
| kitchen_sink | 39 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | 40 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 41 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 192 | 199 | 1,564 | 
| kitchen_sink | 42 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 2 |  | 56 | 166 | 444 | 
| kitchen_sink | 43 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 44 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 192 | 199 | 782 | 
| kitchen_sink | 45 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | 46 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 320 | 263 | 1,166 | 
| kitchen_sink | 47 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 192 | 199 | 1,564 | 
| kitchen_sink | 49 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 256 |  | 192 | 164 | 91,136 | 
| kitchen_sink | 51 | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 256 |  | 48 | 124 | 44,032 | 
| kitchen_sink | 52 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 512 |  | 68 | 169 | 121,344 | 
| kitchen_sink | 53 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 1,024 |  | 192 | 168 | 368,640 | 
| kitchen_sink | 56 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 16 |  | 52 | 31 | 1,328 | 
| kitchen_sink | 57 | RangeTupleCheckerAir<2> | 0 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | 58 | Sha256VmAir | 0 | 262,144 |  | 108 | 471 | 151,781,376 | 
| kitchen_sink | 59 | KeccakVmAir | 0 | 131,072 |  | 1,056 | 3,163 | 552,992,768 | 
| kitchen_sink | 6 | AccessAdapterAir<8> | 0 | 262,144 |  | 16 | 17 | 8,650,752 | 
| kitchen_sink | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,048 |  | 36 | 28 | 131,072 | 
| kitchen_sink | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,024 |  | 28 | 18 | 47,104 | 
| kitchen_sink | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| kitchen_sink | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 4,096 |  | 28 | 26 | 221,184 | 
| kitchen_sink | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| kitchen_sink | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 8,192 |  | 52 | 53 | 860,160 | 
| kitchen_sink | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1,024 |  | 40 | 37 | 78,848 | 
| kitchen_sink | 7 | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| kitchen_sink | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 32,768 |  | 52 | 36 | 2,883,584 | 
| kitchen_sink | 71 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | 74 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | 8 | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| kitchen_sink | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 1,668 | 1,020 | 10,752 | 
| kitchen_sink | 0 | ProgramAir | 1 | 16,384 |  | 8 | 10 | 294,912 | 
| kitchen_sink | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| kitchen_sink | 2 | PersistentBoundaryAir<8> | 1 | 4,096 |  | 12 | 20 | 131,072 | 
| kitchen_sink | 3 | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| kitchen_sink | 49 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 64 |  | 192 | 164 | 22,784 | 
| kitchen_sink | 51 | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 1 | 64 |  | 48 | 124 | 11,008 | 
| kitchen_sink | 52 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 1 | 256 |  | 68 | 169 | 60,672 | 
| kitchen_sink | 53 | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 512 |  | 192 | 168 | 184,320 | 
| kitchen_sink | 57 | RangeTupleCheckerAir<2> | 1 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | 58 | Sha256VmAir | 1 | 262,144 |  | 108 | 471 | 151,781,376 | 
| kitchen_sink | 59 | KeccakVmAir | 1 | 131,072 |  | 1,056 | 3,163 | 552,992,768 | 
| kitchen_sink | 6 | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| kitchen_sink | 61 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 256 |  | 28 | 20 | 12,288 | 
| kitchen_sink | 62 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 512 |  | 36 | 28 | 32,768 | 
| kitchen_sink | 63 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 256 |  | 28 | 18 | 11,776 | 
| kitchen_sink | 64 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 512 |  | 32 | 32 | 32,768 | 
| kitchen_sink | 65 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 2,048 |  | 28 | 26 | 110,592 | 
| kitchen_sink | 67 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 32,768 |  | 52 | 41 | 3,047,424 | 
| kitchen_sink | 68 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 4,096 |  | 52 | 53 | 430,080 | 
| kitchen_sink | 69 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 512 |  | 40 | 37 | 39,424 | 
| kitchen_sink | 7 | AccessAdapterAir<16> | 1 | 131,072 |  | 16 | 25 | 5,373,952 | 
| kitchen_sink | 70 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 8,192 |  | 52 | 36 | 720,896 | 
| kitchen_sink | 71 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | 73 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | 74 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | 8 | AccessAdapterAir<32> | 1 | 2,048 |  | 16 | 41 | 116,736 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 157 | 2,842 | 43,733,898 | 732,015,074 | 157 | 1,179 | 0 | 276.55 | 74.55 | 6 | 189.11 | 336 | 230 | 336 | 15 | 259 | 20,605,812 | 40 | 1,504 | 4,044,755 | 6.36 | 44 | 353 | 0 | 335 | 
| leaf | 0 | 585 | 7,799 | 193,635,846 | 3,758,099,946 | 585 | 4,709 | 0 | 512.77 | 171.57 | 6 | 1,622.85 | 1,233 | 2,015 | 1,233 | 39 | 751 | 83,660,740 | 391 | 2,503 | 9,313,806 | 5.77 | 206 | 708 | 38 | 1,233 | 
| leaf | 1 | 213 | 2,973 | 93,259,932 | 1,225,346,538 | 213 | 1,345 | 0 | 167.72 | 30.41 | 6 | 251.14 | 692 | 297 | 692 | 24 | 155 | 41,610,810 | 45 | 1,413 | 3,271,809 | 6.07 | 43 | 199 | 0 | 692 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 16,384,132 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 104,882,432 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 8,192,066 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 105,398,532 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 1,048,576 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 236,298,954 | 2,013,265,921 | 
| leaf | 0 | 0 | 66,453,636 | 2,013,265,921 | 
| leaf | 0 | 1 | 471,466,240 | 2,013,265,921 | 
| leaf | 0 | 2 | 33,226,818 | 2,013,265,921 | 
| leaf | 0 | 3 | 468,451,588 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 1,044,054,730 | 2,013,265,921 | 
| leaf | 1 | 0 | 19,202,180 | 2,013,265,921 | 
| leaf | 1 | 1 | 140,296,448 | 2,013,265,921 | 
| leaf | 1 | 2 | 9,601,090 | 2,013,265,921 | 
| leaf | 1 | 3 | 142,115,076 | 2,013,265,921 | 
| leaf | 1 | 4 | 1,048,576 | 2,013,265,921 | 
| leaf | 1 | 5 | 314,622,666 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 139 | 2,060 | 39,702,092 | 760,861,898 | 139 | 1,645 | 0 | 451.90 | 10.25 | 17 | 96.38 | 746 | 231 | 745 | 0 | 204 | 10,084,970 | 109 | 137 | 121,000 | 11.18 | 62 | 462 | 3 | 745 | 
| kitchen_sink | 1 | 131 | 1,436 | 36,348,696 | 747,405,610 | 131 | 1,165 | 0 | 293.22 | 8.48 | 10 | 64.32 | 569 | 112 | 569 | 0 | 180 | 8,251,238 | 42 | 97 | 33,763 | 5.91 | 29 | 302 | 1 | 569 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,161,732 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 16,740,560 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 580,866 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 16,673,036 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 32,768 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 16,384 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 25,159,888 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 524,288 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,256 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 63,408,626 | 2,013,265,921 | 
| kitchen_sink | 1 | 0 | 886,532 | 2,013,265,921 | 
| kitchen_sink | 1 | 1 | 15,905,024 | 2,013,265,921 | 
| kitchen_sink | 1 | 2 | 443,266 | 2,013,265,921 | 
| kitchen_sink | 1 | 3 | 15,593,732 | 2,013,265,921 | 
| kitchen_sink | 1 | 4 | 16,384 | 2,013,265,921 | 
| kitchen_sink | 1 | 5 | 8,192 | 2,013,265,921 | 
| kitchen_sink | 1 | 6 | 24,984,768 | 2,013,265,921 | 
| kitchen_sink | 1 | 7 | 524,288 | 2,013,265,921 | 
| kitchen_sink | 1 | 8 | 2,048 | 2,013,265,921 | 
| kitchen_sink | 1 | 9 | 60,875,082 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/eda30f68985a549ec1a04df3d5c77cdea666976b

Max Segment Length: 4194204

Instance Type: g6e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25932210868)
