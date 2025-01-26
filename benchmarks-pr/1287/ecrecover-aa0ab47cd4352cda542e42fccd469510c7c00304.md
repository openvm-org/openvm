| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-4.5%])</span> 21.04 | <span style='color: green'>(-1 [-4.5%])</span> 21.04 |
| ecrecover_program | <span style='color: red'>(+0 [+1.2%])</span> 2.57 | <span style='color: red'>(+0 [+1.2%])</span> 2.57 |
| leaf | <span style='color: green'>(-1 [-5.2%])</span> 18.47 | <span style='color: green'>(-1 [-5.2%])</span> 18.47 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+30 [+1.2%])</span> 2,569 | <span style='color: red'>(+30 [+1.2%])</span> 2,569 | <span style='color: red'>(+30 [+1.2%])</span> 2,569 | <span style='color: red'>(+30 [+1.2%])</span> 2,569 |
| `main_cells_used     ` |  15,075,033 |  15,075,033 |  15,075,033 |  15,075,033 |
| `total_cycles        ` |  285,401 |  285,401 |  285,401 |  285,401 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-1.3%])</span> 148 | <span style='color: green'>(-2 [-1.3%])</span> 148 | <span style='color: green'>(-2 [-1.3%])</span> 148 | <span style='color: green'>(-2 [-1.3%])</span> 148 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-13 [-4.8%])</span> 256 | <span style='color: green'>(-13 [-4.8%])</span> 256 | <span style='color: green'>(-13 [-4.8%])</span> 256 | <span style='color: green'>(-13 [-4.8%])</span> 256 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+45 [+2.1%])</span> 2,165 | <span style='color: red'>(+45 [+2.1%])</span> 2,165 | <span style='color: red'>(+45 [+2.1%])</span> 2,165 | <span style='color: red'>(+45 [+2.1%])</span> 2,165 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+1 [+0.2%])</span> 401 | <span style='color: red'>(+1 [+0.2%])</span> 401 | <span style='color: red'>(+1 [+0.2%])</span> 401 | <span style='color: red'>(+1 [+0.2%])</span> 401 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-12 [-27.3%])</span> 32 | <span style='color: green'>(-12 [-27.3%])</span> 32 | <span style='color: green'>(-12 [-27.3%])</span> 32 | <span style='color: green'>(-12 [-27.3%])</span> 32 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-191 [-34.7%])</span> 360 | <span style='color: green'>(-191 [-34.7%])</span> 360 | <span style='color: green'>(-191 [-34.7%])</span> 360 | <span style='color: green'>(-191 [-34.7%])</span> 360 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+62 [+37.6%])</span> 227 | <span style='color: red'>(+62 [+37.6%])</span> 227 | <span style='color: red'>(+62 [+37.6%])</span> 227 | <span style='color: red'>(+62 [+37.6%])</span> 227 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+164 [+51.1%])</span> 485 | <span style='color: red'>(+164 [+51.1%])</span> 485 | <span style='color: red'>(+164 [+51.1%])</span> 485 | <span style='color: red'>(+164 [+51.1%])</span> 485 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+21 [+3.3%])</span> 648 | <span style='color: red'>(+21 [+3.3%])</span> 648 | <span style='color: red'>(+21 [+3.3%])</span> 648 | <span style='color: red'>(+21 [+3.3%])</span> 648 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1020 [-5.2%])</span> 18,474 | <span style='color: green'>(-1020 [-5.2%])</span> 18,474 | <span style='color: green'>(-1020 [-5.2%])</span> 18,474 | <span style='color: green'>(-1020 [-5.2%])</span> 18,474 |
| `main_cells_used     ` | <span style='color: green'>(-10344011 [-5.1%])</span> 193,207,306 | <span style='color: green'>(-10344011 [-5.1%])</span> 193,207,306 | <span style='color: green'>(-10344011 [-5.1%])</span> 193,207,306 | <span style='color: green'>(-10344011 [-5.1%])</span> 193,207,306 |
| `total_cycles        ` | <span style='color: green'>(-8900 [-0.2%])</span> 4,156,012 | <span style='color: green'>(-8900 [-0.2%])</span> 4,156,012 | <span style='color: green'>(-8900 [-0.2%])</span> 4,156,012 | <span style='color: green'>(-8900 [-0.2%])</span> 4,156,012 |
| `execute_time_ms     ` | <span style='color: green'>(-51 [-5.0%])</span> 978 | <span style='color: green'>(-51 [-5.0%])</span> 978 | <span style='color: green'>(-51 [-5.0%])</span> 978 | <span style='color: green'>(-51 [-5.0%])</span> 978 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-103 [-2.8%])</span> 3,589 | <span style='color: green'>(-103 [-2.8%])</span> 3,589 | <span style='color: green'>(-103 [-2.8%])</span> 3,589 | <span style='color: green'>(-103 [-2.8%])</span> 3,589 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-866 [-5.9%])</span> 13,907 | <span style='color: green'>(-866 [-5.9%])</span> 13,907 | <span style='color: green'>(-866 [-5.9%])</span> 13,907 | <span style='color: green'>(-866 [-5.9%])</span> 13,907 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-25 [-0.8%])</span> 2,923 | <span style='color: green'>(-25 [-0.8%])</span> 2,923 | <span style='color: green'>(-25 [-0.8%])</span> 2,923 | <span style='color: green'>(-25 [-0.8%])</span> 2,923 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-49 [-11.7%])</span> 370 | <span style='color: green'>(-49 [-11.7%])</span> 370 | <span style='color: green'>(-49 [-11.7%])</span> 370 | <span style='color: green'>(-49 [-11.7%])</span> 370 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-654 [-18.2%])</span> 2,933 | <span style='color: green'>(-654 [-18.2%])</span> 2,933 | <span style='color: green'>(-654 [-18.2%])</span> 2,933 | <span style='color: green'>(-654 [-18.2%])</span> 2,933 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-22 [-1.0%])</span> 2,088 | <span style='color: green'>(-22 [-1.0%])</span> 2,088 | <span style='color: green'>(-22 [-1.0%])</span> 2,088 | <span style='color: green'>(-22 [-1.0%])</span> 2,088 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+34 [+1.3%])</span> 2,622 | <span style='color: red'>(+34 [+1.3%])</span> 2,622 | <span style='color: red'>(+34 [+1.3%])</span> 2,622 | <span style='color: red'>(+34 [+1.3%])</span> 2,622 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-150 [-4.8%])</span> 2,966 | <span style='color: green'>(-150 [-4.8%])</span> 2,966 | <span style='color: green'>(-150 [-4.8%])</span> 2,966 | <span style='color: green'>(-150 [-4.8%])</span> 2,966 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,168 | 12 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 4 | 321 | 4,382 | 
| ecrecover_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| ecrecover_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| ecrecover_program | PhantomAir | 4 | 3 | 4 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 4 | 15 | 13 | 
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
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 532 | 3,164 | 473,088 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 12 | 32 | 180,224 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 8 | 20 | 114,688 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 8 | 6 | 896 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 28 | 36 | 8,388,608 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 24 | 37 | 124,928 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 28 | 53 | 1,327,104 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 16 | 26 | 688,128 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 20 | 32 | 1,703,936 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 16 | 18 | 278,528 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 256 |  | 20 | 26 | 11,776 | 
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
| leaf | 0 | 3,589 | 18,474 | 4,156,012 | 466,306,008 | 13,907 | 2,088 | 2,622 | 2,933 | 2,966 | 2,923 | 193,207,306 | 370 | 978 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 256 | 2,569 | 285,401 | 38,415,035 | 2,165 | 227 | 485 | 360 | 648 | 401 | 15,075,033 | 32 | 148 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/aa0ab47cd4352cda542e42fccd469510c7c00304

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12970781882)
