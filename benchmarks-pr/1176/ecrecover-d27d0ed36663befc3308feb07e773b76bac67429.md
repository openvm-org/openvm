| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+11 [+23.4%])</span> 55.88 | <span style='color: red'>(+11 [+23.4%])</span> 55.88 |
| ecrecover_program | <span style='color: green'>(-0 [-2.8%])</span> 2.59 | <span style='color: green'>(-0 [-2.8%])</span> 2.59 |
| leaf | <span style='color: red'>(+11 [+25.1%])</span> 53.30 | <span style='color: red'>(+11 [+25.1%])</span> 53.30 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-75 [-2.8%])</span> 2,585 | <span style='color: green'>(-75 [-2.8%])</span> 2,585 | <span style='color: green'>(-75 [-2.8%])</span> 2,585 | <span style='color: green'>(-75 [-2.8%])</span> 2,585 |
| `main_cells_used     ` |  15,074,875 |  15,074,875 |  15,074,875 |  15,074,875 |
| `total_cycles        ` |  285,169 |  285,169 |  285,169 |  285,169 |
| `execute_time_ms     ` | <span style='color: red'>(+7 [+4.2%])</span> 173 | <span style='color: red'>(+7 [+4.2%])</span> 173 | <span style='color: red'>(+7 [+4.2%])</span> 173 | <span style='color: red'>(+7 [+4.2%])</span> 173 |
| `trace_gen_time_ms   ` |  274 |  274 |  274 |  274 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-82 [-3.7%])</span> 2,138 | <span style='color: green'>(-82 [-3.7%])</span> 2,138 | <span style='color: green'>(-82 [-3.7%])</span> 2,138 | <span style='color: green'>(-82 [-3.7%])</span> 2,138 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-9 [-2.3%])</span> 388 | <span style='color: green'>(-9 [-2.3%])</span> 388 | <span style='color: green'>(-9 [-2.3%])</span> 388 | <span style='color: green'>(-9 [-2.3%])</span> 388 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.3%])</span> 42 | <span style='color: green'>(-1 [-2.3%])</span> 42 | <span style='color: green'>(-1 [-2.3%])</span> 42 | <span style='color: green'>(-1 [-2.3%])</span> 42 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.8%])</span> 520 | <span style='color: green'>(-4 [-0.8%])</span> 520 | <span style='color: green'>(-4 [-0.8%])</span> 520 | <span style='color: green'>(-4 [-0.8%])</span> 520 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-46 [-15.1%])</span> 259 | <span style='color: green'>(-46 [-15.1%])</span> 259 | <span style='color: green'>(-46 [-15.1%])</span> 259 | <span style='color: green'>(-46 [-15.1%])</span> 259 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-4 [-1.3%])</span> 302 | <span style='color: green'>(-4 [-1.3%])</span> 302 | <span style='color: green'>(-4 [-1.3%])</span> 302 | <span style='color: green'>(-4 [-1.3%])</span> 302 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-19 [-3.0%])</span> 623 | <span style='color: green'>(-19 [-3.0%])</span> 623 | <span style='color: green'>(-19 [-3.0%])</span> 623 | <span style='color: green'>(-19 [-3.0%])</span> 623 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+10677 [+25.1%])</span> 53,295 | <span style='color: red'>(+10677 [+25.1%])</span> 53,295 | <span style='color: red'>(+10677 [+25.1%])</span> 53,295 | <span style='color: red'>(+10677 [+25.1%])</span> 53,295 |
| `main_cells_used     ` | <span style='color: green'>(-20535973 [-4.7%])</span> 419,456,718 | <span style='color: green'>(-20535973 [-4.7%])</span> 419,456,718 | <span style='color: green'>(-20535973 [-4.7%])</span> 419,456,718 | <span style='color: green'>(-20535973 [-4.7%])</span> 419,456,718 |
| `total_cycles        ` | <span style='color: red'>(+295345 [+3.1%])</span> 9,947,143 | <span style='color: red'>(+295345 [+3.1%])</span> 9,947,143 | <span style='color: red'>(+295345 [+3.1%])</span> 9,947,143 | <span style='color: red'>(+295345 [+3.1%])</span> 9,947,143 |
| `execute_time_ms     ` | <span style='color: red'>(+318 [+10.3%])</span> 3,391 | <span style='color: red'>(+318 [+10.3%])</span> 3,391 | <span style='color: red'>(+318 [+10.3%])</span> 3,391 | <span style='color: red'>(+318 [+10.3%])</span> 3,391 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-225 [-2.7%])</span> 7,969 | <span style='color: green'>(-225 [-2.7%])</span> 7,969 | <span style='color: green'>(-225 [-2.7%])</span> 7,969 | <span style='color: green'>(-225 [-2.7%])</span> 7,969 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+10584 [+33.8%])</span> 41,935 | <span style='color: red'>(+10584 [+33.8%])</span> 41,935 | <span style='color: red'>(+10584 [+33.8%])</span> 41,935 | <span style='color: red'>(+10584 [+33.8%])</span> 41,935 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+1119 [+17.7%])</span> 7,448 | <span style='color: red'>(+1119 [+17.7%])</span> 7,448 | <span style='color: red'>(+1119 [+17.7%])</span> 7,448 | <span style='color: red'>(+1119 [+17.7%])</span> 7,448 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+106 [+13.6%])</span> 883 | <span style='color: red'>(+106 [+13.6%])</span> 883 | <span style='color: red'>(+106 [+13.6%])</span> 883 | <span style='color: red'>(+106 [+13.6%])</span> 883 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+1284 [+20.4%])</span> 7,587 | <span style='color: red'>(+1284 [+20.4%])</span> 7,587 | <span style='color: red'>(+1284 [+20.4%])</span> 7,587 | <span style='color: red'>(+1284 [+20.4%])</span> 7,587 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2832 [+42.8%])</span> 9,454 | <span style='color: red'>(+2832 [+42.8%])</span> 9,454 | <span style='color: red'>(+2832 [+42.8%])</span> 9,454 | <span style='color: red'>(+2832 [+42.8%])</span> 9,454 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1763 [+34.7%])</span> 6,845 | <span style='color: red'>(+1763 [+34.7%])</span> 6,845 | <span style='color: red'>(+1763 [+34.7%])</span> 6,845 | <span style='color: red'>(+1763 [+34.7%])</span> 6,845 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3480 [+55.8%])</span> 9,715 | <span style='color: red'>(+3480 [+55.8%])</span> 9,715 | <span style='color: red'>(+3480 [+55.8%])</span> 9,715 | <span style='color: red'>(+3480 [+55.8%])</span> 9,715 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,156 | 13 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,571 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 223 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 411 | 449 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, ModularAddSubCoreAir> | 2 | 94 | 126 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, ModularMulDivCoreAir> | 2 | 156 | 188 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 456 | 
| ecrecover_program | VmConnectorAir | 2 | 3 | 9 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 31 | 302 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 23 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 24 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 2,097,152 |  | 16 | 11 | 56,623,104 | 
| leaf | AccessAdapterAir<4> | 0 | 1,048,576 |  | 16 | 13 | 30,408,704 | 
| leaf | AccessAdapterAir<8> | 0 | 262,144 |  | 16 | 17 | 8,650,752 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 36 | 348 | 50,331,648 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 1,048,576 |  | 8 | 10 | 18,874,368 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 4,194,304 |  | 28 | 23 | 213,909,504 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 8,388,608 |  | 20 | 30 | 419,430,400 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 4,194,304 |  | 20 | 31 | 213,909,504 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 40 | 15,728,640 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 2,097,152 |  | 8 | 11 | 39,845,888 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 24 | 25 | 802,816 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 24 | 11 | 8,960 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 24 | 41 | 532,480 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 24 | 13 | 4,736 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 24 | 17 | 1,343,488 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,288 | 3,164 | 569,856 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 20 | 32 | 212,992 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 12 | 6 | 1,152 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 80 | 36 | 15,204,352 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 48 | 26 | 1,212,416 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 56 | 32 | 2,883,584 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 44 | 18 | 507,904 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 256 |  | 36 | 26 | 15,872 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 76 | 35 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 72 | 40 | 14,680,064 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 100 | 39 | 1,112 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 80 | 31 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 21 | 200,704 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, ModularAddSubCoreAir> | 0 | 8 |  | 192 | 199 | 3,128 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, ModularMulDivCoreAir> | 0 | 16 |  | 316 | 261 | 9,232 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 7,969 | 53,295 | 9,947,143 | 1,220,217,304 | 41,935 | 9,454 | 6,845 | 7,587 | 9,715 | 7,448 | 419,456,718 | 883 | 3,391 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 274 | 2,585 | 285,169 | 56,159,799 | 2,138 | 259 | 302 | 520 | 623 | 388 | 15,074,875 | 42 | 173 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/d27d0ed36663befc3308feb07e773b76bac67429

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12667058444)