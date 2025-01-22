| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.8%])</span> 25 | <span style='color: green'>(-0 [-0.8%])</span> 25 |
| ecrecover_program | <span style='color: red'>(+0 [+0.3%])</span> 2.60 | <span style='color: red'>(+0 [+0.3%])</span> 2.60 |
| leaf | <span style='color: green'>(-0 [-0.9%])</span> 22.40 | <span style='color: green'>(-0 [-0.9%])</span> 22.40 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+9 [+0.3%])</span> 2,604 | <span style='color: red'>(+9 [+0.3%])</span> 2,604 | <span style='color: red'>(+9 [+0.3%])</span> 2,604 | <span style='color: red'>(+9 [+0.3%])</span> 2,604 |
| `main_cells_used     ` |  15,075,033 |  15,075,033 |  15,075,033 |  15,075,033 |
| `total_cycles        ` |  285,401 |  285,401 |  285,401 |  285,401 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.7%])</span> 146 | <span style='color: red'>(+1 [+0.7%])</span> 146 | <span style='color: red'>(+1 [+0.7%])</span> 146 | <span style='color: red'>(+1 [+0.7%])</span> 146 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-3 [-1.1%])</span> 258 | <span style='color: green'>(-3 [-1.1%])</span> 258 | <span style='color: green'>(-3 [-1.1%])</span> 258 | <span style='color: green'>(-3 [-1.1%])</span> 258 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+11 [+0.5%])</span> 2,200 | <span style='color: red'>(+11 [+0.5%])</span> 2,200 | <span style='color: red'>(+11 [+0.5%])</span> 2,200 | <span style='color: red'>(+11 [+0.5%])</span> 2,200 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-1.0%])</span> 393 | <span style='color: green'>(-4 [-1.0%])</span> 393 | <span style='color: green'>(-4 [-1.0%])</span> 393 | <span style='color: green'>(-4 [-1.0%])</span> 393 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+9 [+1.7%])</span> 534 | <span style='color: red'>(+9 [+1.7%])</span> 534 | <span style='color: red'>(+9 [+1.7%])</span> 534 | <span style='color: red'>(+9 [+1.7%])</span> 534 |
| `quotient_poly_compute_time_ms` |  294 |  294 |  294 |  294 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+9 [+3.0%])</span> 311 | <span style='color: red'>(+9 [+3.0%])</span> 311 | <span style='color: red'>(+9 [+3.0%])</span> 311 | <span style='color: red'>(+9 [+3.0%])</span> 311 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-0.3%])</span> 624 | <span style='color: green'>(-2 [-0.3%])</span> 624 | <span style='color: green'>(-2 [-0.3%])</span> 624 | <span style='color: green'>(-2 [-0.3%])</span> 624 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-205 [-0.9%])</span> 22,397 | <span style='color: green'>(-205 [-0.9%])</span> 22,397 | <span style='color: green'>(-205 [-0.9%])</span> 22,397 | <span style='color: green'>(-205 [-0.9%])</span> 22,397 |
| `main_cells_used     ` |  241,409,329 |  241,409,329 |  241,409,329 |  241,409,329 |
| `total_cycles        ` |  4,166,623 |  4,166,623 |  4,166,623 |  4,166,623 |
| `execute_time_ms     ` | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-71 [-1.8%])</span> 3,856 | <span style='color: green'>(-71 [-1.8%])</span> 3,856 | <span style='color: green'>(-71 [-1.8%])</span> 3,856 | <span style='color: green'>(-71 [-1.8%])</span> 3,856 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-150 [-0.8%])</span> 17,615 | <span style='color: green'>(-150 [-0.8%])</span> 17,615 | <span style='color: green'>(-150 [-0.8%])</span> 17,615 | <span style='color: green'>(-150 [-0.8%])</span> 17,615 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+17 [+0.5%])</span> 3,289 | <span style='color: red'>(+17 [+0.5%])</span> 3,289 | <span style='color: red'>(+17 [+0.5%])</span> 3,289 | <span style='color: red'>(+17 [+0.5%])</span> 3,289 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.3%])</span> 391 | <span style='color: green'>(-1 [-0.3%])</span> 391 | <span style='color: green'>(-1 [-0.3%])</span> 391 | <span style='color: green'>(-1 [-0.3%])</span> 391 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-22 [-0.6%])</span> 3,949 | <span style='color: green'>(-22 [-0.6%])</span> 3,949 | <span style='color: green'>(-22 [-0.6%])</span> 3,949 | <span style='color: green'>(-22 [-0.6%])</span> 3,949 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+32 [+0.7%])</span> 4,501 | <span style='color: red'>(+32 [+0.7%])</span> 4,501 | <span style='color: red'>(+32 [+0.7%])</span> 4,501 | <span style='color: red'>(+32 [+0.7%])</span> 4,501 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+7 [+0.3%])</span> 2,581 | <span style='color: red'>(+7 [+0.3%])</span> 2,581 | <span style='color: red'>(+7 [+0.3%])</span> 2,581 | <span style='color: red'>(+7 [+0.3%])</span> 2,581 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-184 [-6.0%])</span> 2,900 | <span style='color: green'>(-184 [-6.0%])</span> 2,900 | <span style='color: green'>(-184 [-6.0%])</span> 2,900 | <span style='color: green'>(-184 [-6.0%])</span> 2,900 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,010 | 11 | 

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
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 2 | 411 | 513 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 156 | 189 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 456 | 
| ecrecover_program | VmConnectorAir | 2 | 3 | 9 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 176 | 590 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 23 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 356 | 399 | 98,959,360 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 36 | 34 | 9,175,040 | 
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
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 316 | 261 | 18,464 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 3,856 | 22,397 | 4,166,623 | 590,040,024 | 17,615 | 4,501 | 2,581 | 3,949 | 2,900 | 3,289 | 241,409,329 | 391 | 926 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 258 | 2,604 | 285,401 | 56,172,159 | 2,200 | 294 | 311 | 534 | 624 | 393 | 15,075,033 | 40 | 146 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12915051802)
