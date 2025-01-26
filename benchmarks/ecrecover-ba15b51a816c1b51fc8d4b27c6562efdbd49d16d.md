| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.9%])</span> 21.09 | <span style='color: red'>(+0 [+0.9%])</span> 21.09 |
| ecrecover_program | <span style='color: red'>(+0 [+0.4%])</span> 2.57 | <span style='color: red'>(+0 [+0.4%])</span> 2.57 |
| leaf | <span style='color: red'>(+0 [+1.0%])</span> 18.52 | <span style='color: red'>(+0 [+1.0%])</span> 18.52 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+11 [+0.4%])</span> 2,569 | <span style='color: red'>(+11 [+0.4%])</span> 2,569 | <span style='color: red'>(+11 [+0.4%])</span> 2,569 | <span style='color: red'>(+11 [+0.4%])</span> 2,569 |
| `main_cells_used     ` |  15,055,723 |  15,055,723 |  15,055,723 |  15,055,723 |
| `total_cycles        ` |  284,567 |  284,567 |  284,567 |  284,567 |
| `execute_time_ms     ` | <span style='color: red'>(+4 [+2.7%])</span> 153 | <span style='color: red'>(+4 [+2.7%])</span> 153 | <span style='color: red'>(+4 [+2.7%])</span> 153 | <span style='color: red'>(+4 [+2.7%])</span> 153 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-4 [-1.6%])</span> 253 | <span style='color: green'>(-4 [-1.6%])</span> 253 | <span style='color: green'>(-4 [-1.6%])</span> 253 | <span style='color: green'>(-4 [-1.6%])</span> 253 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+11 [+0.5%])</span> 2,163 | <span style='color: red'>(+11 [+0.5%])</span> 2,163 | <span style='color: red'>(+11 [+0.5%])</span> 2,163 | <span style='color: red'>(+11 [+0.5%])</span> 2,163 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+1.1%])</span> 354 | <span style='color: red'>(+4 [+1.1%])</span> 354 | <span style='color: red'>(+4 [+1.1%])</span> 354 | <span style='color: red'>(+4 [+1.1%])</span> 354 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-9 [-3.9%])</span> 224 | <span style='color: green'>(-9 [-3.9%])</span> 224 | <span style='color: green'>(-9 [-3.9%])</span> 224 | <span style='color: green'>(-9 [-3.9%])</span> 224 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+9 [+1.9%])</span> 477 | <span style='color: red'>(+9 [+1.9%])</span> 477 | <span style='color: red'>(+9 [+1.9%])</span> 477 | <span style='color: red'>(+9 [+1.9%])</span> 477 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+10 [+1.5%])</span> 669 | <span style='color: red'>(+10 [+1.5%])</span> 669 | <span style='color: red'>(+10 [+1.5%])</span> 669 | <span style='color: red'>(+10 [+1.5%])</span> 669 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+176 [+1.0%])</span> 18,519 | <span style='color: red'>(+176 [+1.0%])</span> 18,519 | <span style='color: red'>(+176 [+1.0%])</span> 18,519 | <span style='color: red'>(+176 [+1.0%])</span> 18,519 |
| `main_cells_used     ` |  193,313,917 |  193,313,917 |  193,313,917 |  193,313,917 |
| `total_cycles        ` |  4,157,783 |  4,157,783 |  4,157,783 |  4,157,783 |
| `execute_time_ms     ` | <span style='color: green'>(-3 [-0.3%])</span> 929 | <span style='color: green'>(-3 [-0.3%])</span> 929 | <span style='color: green'>(-3 [-0.3%])</span> 929 | <span style='color: green'>(-3 [-0.3%])</span> 929 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+53 [+1.5%])</span> 3,543 | <span style='color: red'>(+53 [+1.5%])</span> 3,543 | <span style='color: red'>(+53 [+1.5%])</span> 3,543 | <span style='color: red'>(+53 [+1.5%])</span> 3,543 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+126 [+0.9%])</span> 14,047 | <span style='color: red'>(+126 [+0.9%])</span> 14,047 | <span style='color: red'>(+126 [+0.9%])</span> 14,047 | <span style='color: red'>(+126 [+0.9%])</span> 14,047 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-15 [-0.5%])</span> 2,903 | <span style='color: green'>(-15 [-0.5%])</span> 2,903 | <span style='color: green'>(-15 [-0.5%])</span> 2,903 | <span style='color: green'>(-15 [-0.5%])</span> 2,903 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+2.6%])</span> 360 | <span style='color: red'>(+9 [+2.6%])</span> 360 | <span style='color: red'>(+9 [+2.6%])</span> 360 | <span style='color: red'>(+9 [+2.6%])</span> 360 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-42 [-1.4%])</span> 2,879 | <span style='color: green'>(-42 [-1.4%])</span> 2,879 | <span style='color: green'>(-42 [-1.4%])</span> 2,879 | <span style='color: green'>(-42 [-1.4%])</span> 2,879 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-7 [-0.3%])</span> 2,081 | <span style='color: green'>(-7 [-0.3%])</span> 2,081 | <span style='color: green'>(-7 [-0.3%])</span> 2,081 | <span style='color: green'>(-7 [-0.3%])</span> 2,081 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+38 [+1.4%])</span> 2,687 | <span style='color: red'>(+38 [+1.4%])</span> 2,687 | <span style='color: red'>(+38 [+1.4%])</span> 2,687 | <span style='color: red'>(+38 [+1.4%])</span> 2,687 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+143 [+4.8%])</span> 3,134 | <span style='color: red'>(+143 [+4.8%])</span> 3,134 | <span style='color: red'>(+143 [+4.8%])</span> 3,134 | <span style='color: red'>(+143 [+4.8%])</span> 3,134 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,160 | 11 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 4 | 321 | 4,380 | 
| ecrecover_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| ecrecover_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| ecrecover_program | PhantomAir | 4 | 3 | 4 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 4 | 25 | 217 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 411 | 378 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 4 | 156 | 150 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 422 | 351 | 
| ecrecover_program | VmConnectorAir | 4 | 3 | 8 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 11 | 
| leaf | FriReducedOpeningAir | 4 | 31 | 53 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 176 | 555 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 11 | 20 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 17 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 17 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 36 | 26 | 65,011,712 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 216 | 399 | 80,609,280 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 1,048,576 |  | 8 | 10 | 18,874,368 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 16 | 23 | 40,894,464 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 25 | 51,380,224 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 34 | 7,602,176 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 40 | 15,728,640 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 2,097,152 |  | 8 | 11 | 39,845,888 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 12 | 25 | 606,208 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 12 | 11 | 5,888 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 12 | 41 | 434,176 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 12 | 13 | 3,200 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 12 | 17 | 950,272 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 532 | 3,163 | 472,960 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 12 | 32 | 180,224 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 8 | 20 | 114,688 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 8 | 6 | 896 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 24 | 32 | 14,336 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 28 | 36 | 8,388,608 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 24 | 37 | 124,928 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 28 | 53 | 1,327,104 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 16 | 26 | 688,128 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 20 | 32 | 1,703,936 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 16 | 18 | 278,528 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 32 | 166 | 811,008 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 20 | 28 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 28 | 35 | 258,048 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 28 | 40 | 8,912,896 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 40 | 39 | 632 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 28 | 31 | 241,664 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 16 | 21 | 151,552 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 416 | 543 | 1,964,032 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 160 | 261 | 13,472 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 428 | 619 | 1,072,128 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 3,543 | 18,519 | 4,157,783 | 466,306,008 | 14,047 | 2,081 | 2,687 | 2,879 | 3,134 | 2,903 | 193,313,917 | 360 | 929 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 253 | 2,569 | 284,567 | 38,417,467 | 2,163 | 224 | 477 | 354 | 669 | 394 | 15,055,723 | 33 | 153 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ba15b51a816c1b51fc8d4b27c6562efdbd49d16d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12978466023)