| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: red'>(+1 [+7.3%])</span> 9.74 | <span style='color: red'>(+0 [+7.5%])</span> 5.41 | 5.41 |
| pairing | <span style='color: red'>(+0 [+9.3%])</span> 2.81 | <span style='color: red'>(+0 [+11.1%])</span> 1.48 |  1.48 |
| leaf | <span style='color: red'>(+0 [+6.6%])</span> 6.94 | <span style='color: red'>(+0 [+6.3%])</span> 3.93 |  3.93 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+119 [+9.5%])</span> 1,371.50 | <span style='color: red'>(+238 [+9.5%])</span> 2,743 | <span style='color: red'>(+148 [+11.6%])</span> 1,420 | <span style='color: red'>(+90 [+7.3%])</span> 1,323 |
| `main_cells_used     ` |  12,754,256 |  25,508,512 |  15,874,642 |  9,633,870 |
| `total_cells_used    ` |  27,530,950 |  55,061,900 |  32,402,324 |  22,659,576 |
| `execute_metered_time_ms` |  63 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-0.6%])</span> 27.82 | -          | <span style='color: green'>(-0 [-0.6%])</span> 27.82 | <span style='color: green'>(-0 [-0.6%])</span> 27.82 |
| `execute_preflight_insns` |  886,012.50 |  1,772,025 |  1,158,000 |  614,025 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+0.6%])</span> 83.50 | <span style='color: red'>(+1 [+0.6%])</span> 167 |  118 | <span style='color: red'>(+1 [+2.1%])</span> 49 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-1.1%])</span> 14.94 | -          | <span style='color: green'>(-0 [-1.9%])</span> 18.29 |  11.60 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.5%])</span> 185 | <span style='color: green'>(-2 [-0.5%])</span> 370 |  204 | <span style='color: green'>(-2 [-1.2%])</span> 166 |
| `memory_finalize_time_ms` |  0.50 |  1 |  1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+122 [+13.8%])</span> 1,003 | <span style='color: red'>(+243 [+13.8%])</span> 2,006 | <span style='color: red'>(+90 [+9.6%])</span> 1,029 | <span style='color: red'>(+153 [+18.6%])</span> 977 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+0 [+0.8%])</span> 64.50 | <span style='color: red'>(+1 [+0.8%])</span> 129 | <span style='color: red'>(+3 [+4.2%])</span> 75 | <span style='color: green'>(-2 [-3.6%])</span> 54 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+17.7%])</span> 36.50 | <span style='color: red'>(+11 [+17.7%])</span> 73 | <span style='color: red'>(+12 [+35.3%])</span> 46 | <span style='color: green'>(-1 [-3.6%])</span> 27 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+1 [+1.1%])</span> 77.44 | <span style='color: red'>(+2 [+1.1%])</span> 154.88 | <span style='color: green'>(-1 [-0.8%])</span> 89.43 | <span style='color: red'>(+2 [+3.8%])</span> 65.45 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+3 [+3.5%])</span> 82.42 | <span style='color: red'>(+6 [+3.5%])</span> 164.84 | <span style='color: red'>(+2 [+2.0%])</span> 91.40 | <span style='color: red'>(+4 [+5.5%])</span> 73.44 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+0 [+2.8%])</span> 17.86 | <span style='color: red'>(+1 [+2.8%])</span> 35.73 | <span style='color: red'>(+0 [+1.4%])</span> 19.90 | <span style='color: red'>(+1 [+4.5%])</span> 15.82 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+110 [+18.1%])</span> 718.50 | <span style='color: red'>(+220 [+18.1%])</span> 1,437 | <span style='color: red'>(+84 [+11.9%])</span> 787 | <span style='color: red'>(+136 [+26.5%])</span> 650 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+214 [+6.6%])</span> 3,468 | <span style='color: red'>(+427 [+6.6%])</span> 6,936 | <span style='color: red'>(+232 [+6.3%])</span> 3,931 | <span style='color: red'>(+195 [+6.9%])</span> 3,005 |
| `main_cells_used     ` |  20,422,671 |  40,845,342 |  20,633,100 |  20,212,242 |
| `total_cells_used    ` |  49,523,299 |  99,046,598 |  50,026,586 |  49,020,012 |
| `execute_preflight_insns` |  2,991,159.50 |  5,982,319 |  3,053,456 |  2,928,863 |
| `execute_preflight_time_ms` | <span style='color: red'>(+17 [+1.7%])</span> 1,038.50 | <span style='color: red'>(+34 [+1.7%])</span> 2,077 | <span style='color: red'>(+52 [+3.4%])</span> 1,580 | <span style='color: green'>(-18 [-3.5%])</span> 497 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+1.9%])</span> 6.76 | -          | <span style='color: red'>(+0 [+1.2%])</span> 6.79 | <span style='color: red'>(+0 [+2.7%])</span> 6.73 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.6%])</span> 165.50 | <span style='color: green'>(-2 [-0.6%])</span> 331 | <span style='color: green'>(-4 [-2.3%])</span> 168 | <span style='color: red'>(+2 [+1.2%])</span> 163 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-6.1%])</span> 15.50 | <span style='color: green'>(-2 [-6.1%])</span> 31 | <span style='color: green'>(-2 [-11.1%])</span> 16 |  15 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+198 [+9.6%])</span> 2,263 | <span style='color: red'>(+397 [+9.6%])</span> 4,526 | <span style='color: red'>(+218 [+10.3%])</span> 2,339 | <span style='color: red'>(+179 [+8.9%])</span> 2,187 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.2%])</span> 296.50 | <span style='color: green'>(-1 [-0.2%])</span> 593 | <span style='color: red'>(+1 [+0.3%])</span> 298 | <span style='color: green'>(-2 [-0.7%])</span> 295 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-0 [-0.5%])</span> 98.50 | <span style='color: green'>(-1 [-0.5%])</span> 197 | <span style='color: green'>(-5 [-4.4%])</span> 109 | <span style='color: red'>(+4 [+4.8%])</span> 88 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-3 [-0.6%])</span> 454.90 | <span style='color: green'>(-5 [-0.6%])</span> 909.81 | <span style='color: green'>(-3 [-0.6%])</span> 455.06 | <span style='color: green'>(-2 [-0.5%])</span> 454.75 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+4 [+1.2%])</span> 335.96 | <span style='color: red'>(+8 [+1.2%])</span> 671.91 | <span style='color: red'>(+2 [+0.6%])</span> 336.89 | <span style='color: red'>(+6 [+1.8%])</span> 335.03 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-0.9%])</span> 66.71 | <span style='color: green'>(-1 [-0.9%])</span> 133.43 | <span style='color: green'>(-1 [-0.8%])</span> 67.07 | <span style='color: green'>(-1 [-1.1%])</span> 66.36 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+199 [+24.7%])</span> 1,003.50 | <span style='color: red'>(+398 [+24.7%])</span> 2,007 | <span style='color: red'>(+223 [+26.4%])</span> 1,069 | <span style='color: red'>(+175 [+22.9%])</span> 938 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 70 | 2,813 | 6,949 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 3,934 |  | 1 | 1 |  |  |  |  | 
| pairing |  | 1,323 |  | 1 | 63 | 1,772,025 | 27.82 | 0 | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | 0 | ProgramAir | 1 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 1 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | 15 | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 524,288 |  | 20 | 12 | 16,777,216 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 524,288 |  | 20 | 12 | 16,777,216 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 348 | 369 | 5,873,664 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 16 |  | 56 | 166 | 3,552 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 192 | 199 | 25,024 | 
| pairing | 2 | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 23 | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | 3 | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 35 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | 37 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 6 | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | 7 | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 8 | AccessAdapterAir<32> | 0 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | 0 | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 604 | 497 | 4,509,696 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 348 | 369 | 2,936,832 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 16 |  | 56 | 166 | 3,552 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 256 |  | 320 | 263 | 149,248 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 4 |  | 192 | 199 | 1,564 | 
| pairing | 2 | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 64 |  | 72 | 39 | 7,104 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 256 |  | 52 | 31 | 21,248 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 8,192 |  | 28 | 20 | 393,216 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16,384 |  | 36 | 28 | 1,048,576 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| pairing | 3 | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 512 |  | 52 | 53 | 53,760 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 262,144 |  | 52 | 36 | 23,068,672 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | 37 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 6 | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | 7 | AccessAdapterAir<16> | 1 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 8 | AccessAdapterAir<32> | 1 | 65,536 |  | 16 | 41 | 3,735,552 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 574 | 
| leaf | PhantomAir | 2 | 3 | 6 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 12 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | MemoryMerkleAir<8> | 2 | 4 | 41 | 
| pairing | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| pairing | PhantomAir | 2 | 3 | 6 | 
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
| pairing | VmConnectorAir | 2 | 5 | 12 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 168 | 3,005 | 50,026,586 | 761,351,658 | 168 | 2,339 | 0 | 335.03 | 66.36 | 6 | 455.06 | 1,069 | 565 | 1,069 | 16 | 295 | 20,633,100 | 109 | 497 | 3,053,456 | 6.79 | 83 | 408 | 5 | 1,068 | 
| leaf | 1 | 163 | 3,931 | 49,020,012 | 761,351,658 | 163 | 2,187 | 0 | 336.89 | 67.07 | 7 | 454.75 | 938 | 543 | 938 | 15 | 298 | 20,212,242 | 88 | 1,580 | 2,928,863 | 6.73 | 79 | 406 | 0 | 937 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 13,959,300 | 2,013,265,921 | 
| leaf | 0 | 1 | 84,721,920 | 2,013,265,921 | 
| leaf | 0 | 2 | 6,979,650 | 2,013,265,921 | 
| leaf | 0 | 3 | 84,967,684 | 2,013,265,921 | 
| leaf | 0 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 0 | 5 | 191,939,274 | 2,013,265,921 | 
| leaf | 1 | 0 | 13,959,300 | 2,013,265,921 | 
| leaf | 1 | 1 | 84,721,920 | 2,013,265,921 | 
| leaf | 1 | 2 | 6,979,650 | 2,013,265,921 | 
| leaf | 1 | 3 | 84,967,684 | 2,013,265,921 | 
| leaf | 1 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 1 | 5 | 191,939,274 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 166 | 1,420 | 32,402,324 | 172,560,220 | 166 | 977 | 0 | 91.40 | 19.90 | 8 | 89.43 | 650 | 139 | 649 | 1 | 75 | 15,874,642 | 46 | 118 | 1,158,000 | 11.60 | 27 | 112 | 1 | 649 | 
| pairing | 1 | 204 | 1,323 | 22,659,576 | 122,481,638 | 204 | 1,029 | 1 | 73.44 | 15.82 | 8 | 65.45 | 787 | 96 | 787 | 0 | 54 | 9,633,870 | 27 | 49 | 614,025 | 18.29 | 21 | 89 | 1 | 787 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 2,833,318 | 2,013,265,921 | 
| pairing | 0 | 1 | 10,207,392 | 2,013,265,921 | 
| pairing | 0 | 2 | 1,416,659 | 2,013,265,921 | 
| pairing | 0 | 3 | 13,908,708 | 2,013,265,921 | 
| pairing | 0 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 0 | 6 | 3,151,904 | 2,013,265,921 | 
| pairing | 0 | 7 | 3,072 | 2,013,265,921 | 
| pairing | 0 | 8 | 32,586,013 | 2,013,265,921 | 
| pairing | 1 | 0 | 1,939,628 | 2,013,265,921 | 
| pairing | 1 | 1 | 6,975,568 | 2,013,265,921 | 
| pairing | 1 | 2 | 969,814 | 2,013,265,921 | 
| pairing | 1 | 3 | 9,239,640 | 2,013,265,921 | 
| pairing | 1 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 1 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 1 | 6 | 1,573,480 | 2,013,265,921 | 
| pairing | 1 | 7 | 1,536 | 2,013,265,921 | 
| pairing | 1 | 8 | 21,764,626 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/eda30f68985a549ec1a04df3d5c77cdea666976b

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25932210868)
