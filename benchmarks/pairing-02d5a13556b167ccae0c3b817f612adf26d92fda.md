| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  4.50 |  2.65 |
| pairing |  1.47 |  0.85 |
| leaf |  2.97 |  1.74 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  735.50 |  1,471 |  850 |  621 |
| `main_cells_used     ` |  12,704,151 |  25,408,302 |  15,909,408 |  9,498,894 |
| `total_cells_used    ` |  27,472,173 |  54,944,346 |  32,480,114 |  22,464,232 |
| `execute_metered_time_ms` |  66 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  26.24 | -          |  26.24 |  26.24 |
| `execute_preflight_insns` |  872,871 |  1,745,742 |  1,149,000 |  596,742 |
| `execute_preflight_time_ms` |  82.50 |  165 |  105 |  60 |
| `execute_preflight_insn_mi/s` |  14.82 | -          |  18.40 |  11.25 |
| `trace_gen_time_ms   ` |  188.50 |  377 |  193 |  184 |
| `memory_finalize_time_ms` |  0.50 |  1 |  1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` |  359 |  718 |  391 |  327 |
| `main_trace_commit_time_ms` |  63.50 |  127 |  74 |  53 |
| `generate_perm_trace_time_ms` |  33.50 |  67 |  38 |  29 |
| `perm_trace_commit_time_ms` |  77.47 |  154.94 |  89.63 |  65.31 |
| `quotient_poly_compute_time_ms` |  83.71 |  167.41 |  93.65 |  73.76 |
| `quotient_poly_commit_time_ms` |  17.53 |  35.06 |  20.01 |  15.05 |
| `pcs_opening_time_ms ` |  77.50 |  155 |  85 |  70 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,484 |  2,968 |  1,738 |  1,230 |
| `main_cells_used     ` |  16,200,472 |  32,400,944 |  16,332,746 |  16,068,198 |
| `total_cells_used    ` |  39,332,920 |  78,665,840 |  39,650,868 |  39,014,972 |
| `execute_preflight_insns` |  1,952,444.50 |  3,904,889 |  1,991,942 |  1,912,947 |
| `execute_preflight_time_ms` |  598.50 |  1,197 |  898 |  299 |
| `execute_preflight_insn_mi/s` |  7.51 | -          |  7.58 |  7.45 |
| `trace_gen_time_ms   ` |  119.50 |  239 |  128 |  111 |
| `memory_finalize_time_ms` |  12 |  24 |  13 |  11 |
| `stark_prove_excluding_trace_time_ms` |  765 |  1,530 |  802 |  728 |
| `main_trace_commit_time_ms` |  146 |  292 |  150 |  142 |
| `generate_perm_trace_time_ms` |  55 |  110 |  59 |  51 |
| `perm_trace_commit_time_ms` |  226.75 |  453.51 |  231.22 |  222.29 |
| `quotient_poly_compute_time_ms` |  173.08 |  346.17 |  175.76 |  170.41 |
| `quotient_poly_commit_time_ms` |  33.31 |  66.63 |  34.32 |  32.31 |
| `pcs_opening_time_ms ` |  126.50 |  253 |  147 |  106 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 73 | 1,544 | 2,976 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 1,740 |  | 1 | 1 |  |  |  |  | 
| pairing |  | 621 |  | 1 | 66 | 1,745,742 | 26.24 | 0 | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | 0 | ProgramAir | 1 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 11 | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | 11 | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 15 | PhantomAir | 0 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 0 | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 604 | 497 | 4,509,696 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 348 | 369 | 5,873,664 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 348 | 369 | 2,936,832 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 16 |  | 56 | 166 | 3,552 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 256 |  | 320 | 263 | 149,248 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 192 | 199 | 25,024 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 4 |  | 192 | 199 | 1,564 | 
| pairing | 2 | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 2 | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 64 |  | 72 | 39 | 7,104 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 256 |  | 52 | 31 | 21,248 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 23 | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 8,192 |  | 28 | 20 | 393,216 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16,384 |  | 36 | 28 | 1,048,576 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| pairing | 3 | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 3 | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 512 |  | 52 | 53 | 53,760 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 262,144 |  | 52 | 36 | 23,068,672 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 35 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 8,192 |  | 8 | 300 | 2,523,136 | 
| pairing | 37 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 37 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 6 | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | 6 | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | 7 | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 7 | AccessAdapterAir<16> | 1 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 8 | AccessAdapterAir<32> | 0 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | 8 | AccessAdapterAir<32> | 1 | 65,536 |  | 16 | 41 | 3,735,552 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 572 | 
| leaf | PhantomAir | 2 | 3 | 5 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 11 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| pairing | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| pairing | PhantomAir | 2 | 3 | 5 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| pairing | ProgramAir | 1 | 1 | 4 | 
| pairing | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| pairing | Rv32HintStoreAir | 2 | 18 | 28 | 
| pairing | VariableRangeCheckerAir | 1 | 1 | 4 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| pairing | VmConnectorAir | 2 | 5 | 11 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 128 | 1,230 | 39,650,868 | 410,701,290 | 128 | 802 | 0 | 175.76 | 34.32 | 4 | 231.22 | 147 | 292 | 147 | 13 | 150 | 16,332,746 | 59 | 299 | 1,991,942 | 7.58 | 39 | 211 | 0 | 147 | 
| leaf | 1 | 111 | 1,738 | 39,014,972 | 396,840,426 | 111 | 728 | 0 | 170.41 | 32.31 | 4 | 222.29 | 106 | 274 | 106 | 11 | 142 | 16,068,198 | 51 | 898 | 1,912,947 | 7.45 | 38 | 204 | 0 | 106 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 7,241,860 | 2,013,265,921 | 
| leaf | 0 | 1 | 44,744,960 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,620,930 | 2,013,265,921 | 
| leaf | 0 | 3 | 44,335,364 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 100,991,690 | 2,013,265,921 | 
| leaf | 1 | 0 | 7,274,628 | 2,013,265,921 | 
| leaf | 1 | 1 | 43,172,096 | 2,013,265,921 | 
| leaf | 1 | 2 | 3,637,314 | 2,013,265,921 | 
| leaf | 1 | 3 | 43,286,788 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 98,419,402 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 184 | 850 | 32,480,114 | 172,558,444 | 184 | 391 | 0 | 93.65 | 20.01 | 6 | 89.63 | 70 | 130 | 70 | 1 | 74 | 15,909,408 | 38 | 105 | 1,149,000 | 11.25 | 27 | 114 | 1 | 70 | 
| pairing | 1 | 193 | 621 | 22,464,232 | 119,958,502 | 193 | 327 | 1 | 73.76 | 15.05 | 6 | 65.31 | 85 | 98 | 85 | 0 | 53 | 9,498,894 | 29 | 60 | 596,742 | 18.40 | 20 | 89 | 1 | 85 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 2,833,302 | 2,013,265,921 | 
| pairing | 0 | 1 | 10,207,312 | 2,013,265,921 | 
| pairing | 0 | 2 | 1,416,651 | 2,013,265,921 | 
| pairing | 0 | 3 | 13,908,628 | 2,013,265,921 | 
| pairing | 0 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 0 | 6 | 3,151,888 | 2,013,265,921 | 
| pairing | 0 | 7 | 3,072 | 2,013,265,921 | 
| pairing | 0 | 8 | 32,585,813 | 2,013,265,921 | 
| pairing | 1 | 0 | 1,939,628 | 2,013,265,921 | 
| pairing | 1 | 1 | 6,975,568 | 2,013,265,921 | 
| pairing | 1 | 2 | 969,814 | 2,013,265,921 | 
| pairing | 1 | 3 | 9,239,640 | 2,013,265,921 | 
| pairing | 1 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 1 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 1 | 6 | 1,573,480 | 2,013,265,921 | 
| pairing | 1 | 7 | 1,536 | 2,013,265,921 | 
| pairing | 1 | 8 | 21,756,434 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/02d5a13556b167ccae0c3b817f612adf26d92fda

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/19998250068)
