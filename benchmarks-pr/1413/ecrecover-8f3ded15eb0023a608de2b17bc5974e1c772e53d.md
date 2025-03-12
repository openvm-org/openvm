| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-8.7%])</span> 14.50 | <span style='color: green'>(-1 [-8.7%])</span> 14.50 |
| ecrecover_program | <span style='color: green'>(-1 [-43.1%])</span> 1.45 | <span style='color: green'>(-1 [-43.1%])</span> 1.45 |
| leaf | <span style='color: green'>(-0 [-2.1%])</span> 13.05 | <span style='color: green'>(-0 [-2.1%])</span> 13.05 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1097 [-43.1%])</span> 1,448 | <span style='color: green'>(-1097 [-43.1%])</span> 1,448 | <span style='color: green'>(-1097 [-43.1%])</span> 1,448 | <span style='color: green'>(-1097 [-43.1%])</span> 1,448 |
| `main_cells_used     ` |  15,586,346 |  15,586,346 |  15,586,346 |  15,586,346 |
| `total_cycles        ` |  295,181 |  295,181 |  295,181 |  295,181 |
| `execute_time_ms     ` | <span style='color: green'>(-1 [-0.7%])</span> 149 | <span style='color: green'>(-1 [-0.7%])</span> 149 | <span style='color: green'>(-1 [-0.7%])</span> 149 | <span style='color: green'>(-1 [-0.7%])</span> 149 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-10 [-5.3%])</span> 180 | <span style='color: green'>(-10 [-5.3%])</span> 180 | <span style='color: green'>(-10 [-5.3%])</span> 180 | <span style='color: green'>(-10 [-5.3%])</span> 180 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1086 [-49.3%])</span> 1,119 | <span style='color: green'>(-1086 [-49.3%])</span> 1,119 | <span style='color: green'>(-1086 [-49.3%])</span> 1,119 | <span style='color: green'>(-1086 [-49.3%])</span> 1,119 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-197 [-51.8%])</span> 183 | <span style='color: green'>(-197 [-51.8%])</span> 183 | <span style='color: green'>(-197 [-51.8%])</span> 183 | <span style='color: green'>(-197 [-51.8%])</span> 183 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 | <span style='color: red'>(+1 [+3.1%])</span> 33 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-155 [-43.8%])</span> 199 | <span style='color: green'>(-155 [-43.8%])</span> 199 | <span style='color: green'>(-155 [-43.8%])</span> 199 | <span style='color: green'>(-155 [-43.8%])</span> 199 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-97 [-43.9%])</span> 124 | <span style='color: green'>(-97 [-43.9%])</span> 124 | <span style='color: green'>(-97 [-43.9%])</span> 124 | <span style='color: green'>(-97 [-43.9%])</span> 124 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-356 [-69.0%])</span> 160 | <span style='color: green'>(-356 [-69.0%])</span> 160 | <span style='color: green'>(-356 [-69.0%])</span> 160 | <span style='color: green'>(-356 [-69.0%])</span> 160 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-283 [-41.1%])</span> 405 | <span style='color: green'>(-283 [-41.1%])</span> 405 | <span style='color: green'>(-283 [-41.1%])</span> 405 | <span style='color: green'>(-283 [-41.1%])</span> 405 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-282 [-2.1%])</span> 13,055 | <span style='color: green'>(-282 [-2.1%])</span> 13,055 | <span style='color: green'>(-282 [-2.1%])</span> 13,055 | <span style='color: green'>(-282 [-2.1%])</span> 13,055 |
| `main_cells_used     ` | <span style='color: red'>(+110514437 [+82.6%])</span> 244,300,402 | <span style='color: red'>(+110514437 [+82.6%])</span> 244,300,402 | <span style='color: red'>(+110514437 [+82.6%])</span> 244,300,402 | <span style='color: red'>(+110514437 [+82.6%])</span> 244,300,402 |
| `total_cycles        ` | <span style='color: red'>(+735882 [+32.7%])</span> 2,989,206 | <span style='color: red'>(+735882 [+32.7%])</span> 2,989,206 | <span style='color: red'>(+735882 [+32.7%])</span> 2,989,206 | <span style='color: red'>(+735882 [+32.7%])</span> 2,989,206 |
| `execute_time_ms     ` | <span style='color: red'>(+345 [+45.5%])</span> 1,103 | <span style='color: red'>(+345 [+45.5%])</span> 1,103 | <span style='color: red'>(+345 [+45.5%])</span> 1,103 | <span style='color: red'>(+345 [+45.5%])</span> 1,103 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+622 [+35.9%])</span> 2,353 | <span style='color: red'>(+622 [+35.9%])</span> 2,353 | <span style='color: red'>(+622 [+35.9%])</span> 2,353 | <span style='color: red'>(+622 [+35.9%])</span> 2,353 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1249 [-11.5%])</span> 9,599 | <span style='color: green'>(-1249 [-11.5%])</span> 9,599 | <span style='color: green'>(-1249 [-11.5%])</span> 9,599 | <span style='color: green'>(-1249 [-11.5%])</span> 9,599 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-550 [-24.1%])</span> 1,728 | <span style='color: green'>(-550 [-24.1%])</span> 1,728 | <span style='color: green'>(-550 [-24.1%])</span> 1,728 | <span style='color: green'>(-550 [-24.1%])</span> 1,728 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+367 [+115.4%])</span> 685 | <span style='color: red'>(+367 [+115.4%])</span> 685 | <span style='color: red'>(+367 [+115.4%])</span> 685 | <span style='color: red'>(+367 [+115.4%])</span> 685 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+856 [+37.0%])</span> 3,169 | <span style='color: red'>(+856 [+37.0%])</span> 3,169 | <span style='color: red'>(+856 [+37.0%])</span> 3,169 | <span style='color: red'>(+856 [+37.0%])</span> 3,169 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-334 [-20.0%])</span> 1,332 | <span style='color: green'>(-334 [-20.0%])</span> 1,332 | <span style='color: green'>(-334 [-20.0%])</span> 1,332 | <span style='color: green'>(-334 [-20.0%])</span> 1,332 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1064 [-54.7%])</span> 881 | <span style='color: green'>(-1064 [-54.7%])</span> 881 | <span style='color: green'>(-1064 [-54.7%])</span> 881 | <span style='color: green'>(-1064 [-54.7%])</span> 881 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-528 [-22.7%])</span> 1,794 | <span style='color: green'>(-528 [-22.7%])</span> 1,794 | <span style='color: green'>(-528 [-22.7%])</span> 1,794 | <span style='color: green'>(-528 [-22.7%])</span> 1,794 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 701 | 7 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<64> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,511 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 223 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 411 | 476 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 156 | 188 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 451 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 10 | 
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 70 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 571 | 
| leaf | PhantomAir | 2 | 3 | 5 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 10 | 
| leaf | VolatileBoundaryAir | 2 | 4 | 17 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| leaf | FriReducedOpeningAir | 0 | 4,194,304 |  | 84 | 27 | 465,567,744 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 312 | 399 | 186,384,384 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 16 | 25 | 671,744 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 64 |  | 16 | 13 | 1,856 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 52 | 36 | 11,534,336 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 52 | 36 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 72 | 59 | 1,048 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 52 | 31 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 20 | 196,608 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 316 | 261 | 18,464 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 2,353 | 13,055 | 2,989,206 | 999,656,938 | 9,599 | 1,332 | 881 | 3,169 | 1,794 | 1,728 | 244,300,402 | 685 | 1,103 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 180 | 1,448 | 295,181 | 48,197,161 | 1,119 | 124 | 160 | 199 | 405 | 183 | 15,586,346 | 33 | 149 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/8f3ded15eb0023a608de2b17bc5974e1c772e53d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13803688608)
