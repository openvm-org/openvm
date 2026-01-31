| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: green'>(-1 [-3.6%])</span> 21.76 | <span style='color: green'>(-1 [-11.5%])</span> 11.23 | 11.23 |
| pairing | <span style='color: red'>(+0 [+1.7%])</span> 8.42 | <span style='color: green'>(-1 [-14.0%])</span> 4.29 |  4.29 |
| leaf | <span style='color: green'>(-1 [-6.6%])</span> 13.34 | <span style='color: green'>(-1 [-9.8%])</span> 6.94 |  6.94 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+70 [+1.7%])</span> 4,177.50 | <span style='color: red'>(+139 [+1.7%])</span> 8,355 | <span style='color: green'>(-701 [-14.2%])</span> 4,228 | <span style='color: red'>(+840 [+25.6%])</span> 4,127 |
| `main_cells_used     ` |  12,704,151 |  25,408,302 |  15,909,408 |  9,498,894 |
| `total_cells_used    ` |  27,472,173 |  54,944,346 |  32,480,114 |  22,464,232 |
| `execute_metered_time_ms` |  63 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  27.56 | -          |  27.56 |  27.56 |
| `execute_preflight_insns` |  872,871 |  1,745,742 |  1,149,000 |  596,742 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+0.6%])</span> 83.50 | <span style='color: red'>(+1 [+0.6%])</span> 167 |  119 | <span style='color: red'>(+1 [+2.1%])</span> 48 |
| `execute_preflight_insn_mi/s` |  14.76 | -          | <span style='color: green'>(-0 [-0.2%])</span> 18.14 | <span style='color: red'>(+0 [+0.4%])</span> 11.39 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-4 [-2.1%])</span> 184.50 | <span style='color: green'>(-8 [-2.1%])</span> 369 | <span style='color: green'>(-4 [-1.9%])</span> 204 | <span style='color: green'>(-4 [-2.4%])</span> 165 |
| `memory_finalize_time_ms` |  0.50 |  1 |  1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+74 [+2.0%])</span> 3,804 | <span style='color: red'>(+149 [+2.0%])</span> 7,608 | <span style='color: green'>(-698 [-15.1%])</span> 3,934 | <span style='color: red'>(+847 [+30.0%])</span> 3,674 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.5%])</span> 65 | <span style='color: green'>(-2 [-1.5%])</span> 130 | <span style='color: green'>(-1 [-1.3%])</span> 76 | <span style='color: green'>(-1 [-1.8%])</span> 54 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+4 [+7.8%])</span> 48.50 | <span style='color: red'>(+7 [+7.8%])</span> 97 | <span style='color: red'>(+2 [+3.8%])</span> 55 | <span style='color: red'>(+5 [+13.5%])</span> 42 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.4%])</span> 77.63 | <span style='color: green'>(-2 [-1.4%])</span> 155.25 | <span style='color: green'>(-1 [-1.5%])</span> 89.66 | <span style='color: green'>(-1 [-1.3%])</span> 65.59 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-0.7%])</span> 84.55 | <span style='color: green'>(-1 [-0.7%])</span> 169.11 | <span style='color: red'>(+1 [+0.7%])</span> 94.53 | <span style='color: green'>(-2 [-2.4%])</span> 74.57 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+0 [+0.3%])</span> 17.72 | <span style='color: red'>(+0 [+0.3%])</span> 35.44 | <span style='color: red'>(+1 [+4.7%])</span> 20.18 | <span style='color: green'>(-1 [-5.0%])</span> 15.26 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+74 [+2.1%])</span> 3,505.50 | <span style='color: red'>(+147 [+2.1%])</span> 7,011 | <span style='color: green'>(-683 [-15.7%])</span> 3,677 | <span style='color: red'>(+830 [+33.1%])</span> 3,334 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-472 [-6.6%])</span> 6,670.50 | <span style='color: green'>(-943 [-6.6%])</span> 13,341 | <span style='color: green'>(-753 [-9.8%])</span> 6,943 | <span style='color: green'>(-190 [-2.9%])</span> 6,398 |
| `main_cells_used     ` |  20,420,666 |  40,841,332 |  20,631,060 |  20,210,272 |
| `total_cells_used    ` |  49,519,454 |  99,038,908 |  50,022,658 |  49,016,250 |
| `execute_preflight_insns` |  2,989,065 |  5,978,130 |  3,051,239 |  2,926,891 |
| `execute_preflight_time_ms` | <span style='color: green'>(-10 [-1.0%])</span> 1,031 | <span style='color: green'>(-20 [-1.0%])</span> 2,062 | <span style='color: green'>(-31 [-2.0%])</span> 1,531 | <span style='color: red'>(+11 [+2.1%])</span> 531 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-1.5%])</span> 6.36 | -          | <span style='color: green'>(-0 [-1.4%])</span> 6.37 | <span style='color: green'>(-0 [-1.5%])</span> 6.34 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-4 [-2.1%])</span> 162.50 | <span style='color: green'>(-7 [-2.1%])</span> 325 | <span style='color: green'>(-5 [-2.9%])</span> 167 | <span style='color: green'>(-2 [-1.2%])</span> 158 |
| `memory_finalize_time_ms` |  15.50 |  31 | <span style='color: red'>(+1 [+6.2%])</span> 17 | <span style='color: green'>(-1 [-6.7%])</span> 14 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-458 [-7.7%])</span> 5,476 | <span style='color: green'>(-915 [-7.7%])</span> 10,952 | <span style='color: green'>(-274 [-4.6%])</span> 5,699 | <span style='color: green'>(-641 [-10.9%])</span> 5,253 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-3 [-1.0%])</span> 298 | <span style='color: green'>(-6 [-1.0%])</span> 596 | <span style='color: green'>(-4 [-1.3%])</span> 298 | <span style='color: green'>(-2 [-0.7%])</span> 298 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-64 [-29.5%])</span> 154 | <span style='color: green'>(-129 [-29.5%])</span> 308 | <span style='color: green'>(-170 [-52.5%])</span> 154 | <span style='color: red'>(+41 [+36.3%])</span> 154 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-7 [-1.6%])</span> 455.77 | <span style='color: green'>(-14 [-1.6%])</span> 911.54 | <span style='color: green'>(-6 [-1.4%])</span> 456.83 | <span style='color: green'>(-8 [-1.8%])</span> 454.70 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-12 [-3.5%])</span> 336.93 | <span style='color: green'>(-24 [-3.5%])</span> 673.86 | <span style='color: green'>(-13 [-3.8%])</span> 337.77 | <span style='color: green'>(-11 [-3.2%])</span> 336.09 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+1.2%])</span> 69.52 | <span style='color: red'>(+2 [+1.2%])</span> 139.03 | <span style='color: red'>(+1 [+1.1%])</span> 70.14 | <span style='color: red'>(+1 [+1.3%])</span> 68.89 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-372 [-8.2%])</span> 4,155 | <span style='color: green'>(-744 [-8.2%])</span> 8,310 | <span style='color: green'>(-296 [-6.3%])</span> 4,375 | <span style='color: green'>(-448 [-10.2%])</span> 3,935 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 74 | 8,425 | 13,354 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 6,947 |  | 1 | 1 |  |  |  |  | 
| pairing |  | 4,228 |  | 1 | 63 | 1,745,742 | 27.56 | 0 | 

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
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
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
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 8,192 |  | 8 | 300 | 2,523,136 | 
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
| leaf | 0 | 167 | 6,398 | 50,022,658 | 761,351,658 | 167 | 5,699 | 0 | 337.77 | 68.89 | 6 | 456.83 | 4,375 | 612 | 4,375 | 17 | 298 | 20,631,060 | 154 | 531 | 3,051,239 | 6.34 | 83 | 412 | 4 | 4,375 | 
| leaf | 1 | 158 | 6,943 | 49,016,250 | 761,351,658 | 158 | 5,253 | 0 | 336.09 | 70.14 | 6 | 454.70 | 3,935 | 609 | 3,935 | 14 | 298 | 20,210,272 | 154 | 1,531 | 2,926,891 | 6.37 | 80 | 408 | 0 | 3,935 | 

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
| pairing | 0 | 165 | 4,127 | 32,480,114 | 172,558,444 | 165 | 3,674 | 0 | 94.53 | 20.18 | 8 | 89.66 | 3,334 | 148 | 3,334 | 1 | 76 | 15,909,408 | 55 | 119 | 1,149,000 | 11.39 | 27 | 115 | 1 | 3,333 | 
| pairing | 1 | 204 | 4,228 | 22,464,232 | 119,958,502 | 204 | 3,934 | 1 | 74.57 | 15.26 | 8 | 65.59 | 3,677 | 111 | 3,677 | 0 | 54 | 9,498,894 | 42 | 48 | 596,742 | 18.14 | 20 | 90 | 1 | 3,677 | 

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


Commit: https://github.com/openvm-org/openvm/commit/740c351bcd9f37e14e86eb6e65c4dafb76d0b042

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/21538578670)
