| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-4.5%])</span> 21.09 | <span style='color: green'>(-1 [-4.5%])</span> 21.09 |
| ecrecover_program | <span style='color: green'>(-0 [-3.7%])</span> 2.40 | <span style='color: green'>(-0 [-3.7%])</span> 2.40 |
| leaf | <span style='color: green'>(-1 [-4.6%])</span> 18.68 | <span style='color: green'>(-1 [-4.6%])</span> 18.68 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-92 [-3.7%])</span> 2,404 | <span style='color: green'>(-92 [-3.7%])</span> 2,404 | <span style='color: green'>(-92 [-3.7%])</span> 2,404 | <span style='color: green'>(-92 [-3.7%])</span> 2,404 |
| `main_cells_used     ` |  15,075,033 |  15,075,033 |  15,075,033 |  15,075,033 |
| `total_cycles        ` |  285,401 |  285,401 |  285,401 |  285,401 |
| `execute_time_ms     ` |  147 |  147 |  147 |  147 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-14 [-5.3%])</span> 251 | <span style='color: green'>(-14 [-5.3%])</span> 251 | <span style='color: green'>(-14 [-5.3%])</span> 251 | <span style='color: green'>(-14 [-5.3%])</span> 251 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-78 [-3.7%])</span> 2,006 | <span style='color: green'>(-78 [-3.7%])</span> 2,006 | <span style='color: green'>(-78 [-3.7%])</span> 2,006 | <span style='color: green'>(-78 [-3.7%])</span> 2,006 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.3%])</span> 392 | <span style='color: green'>(-1 [-0.3%])</span> 392 | <span style='color: green'>(-1 [-0.3%])</span> 392 | <span style='color: green'>(-1 [-0.3%])</span> 392 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-8 [-19.0%])</span> 34 | <span style='color: green'>(-8 [-19.0%])</span> 34 | <span style='color: green'>(-8 [-19.0%])</span> 34 | <span style='color: green'>(-8 [-19.0%])</span> 34 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-78 [-14.3%])</span> 468 | <span style='color: green'>(-78 [-14.3%])</span> 468 | <span style='color: green'>(-78 [-14.3%])</span> 468 | <span style='color: green'>(-78 [-14.3%])</span> 468 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-0.6%])</span> 156 | <span style='color: green'>(-1 [-0.6%])</span> 156 | <span style='color: green'>(-1 [-0.6%])</span> 156 | <span style='color: green'>(-1 [-0.6%])</span> 156 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+8 [+2.5%])</span> 322 | <span style='color: red'>(+8 [+2.5%])</span> 322 | <span style='color: red'>(+8 [+2.5%])</span> 322 | <span style='color: red'>(+8 [+2.5%])</span> 322 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+2 [+0.3%])</span> 621 | <span style='color: red'>(+2 [+0.3%])</span> 621 | <span style='color: red'>(+2 [+0.3%])</span> 621 | <span style='color: red'>(+2 [+0.3%])</span> 621 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-894 [-4.6%])</span> 18,681 | <span style='color: green'>(-894 [-4.6%])</span> 18,681 | <span style='color: green'>(-894 [-4.6%])</span> 18,681 | <span style='color: green'>(-894 [-4.6%])</span> 18,681 |
| `main_cells_used     ` | <span style='color: green'>(-3899710 [-1.9%])</span> 199,651,537 | <span style='color: green'>(-3899710 [-1.9%])</span> 199,651,537 | <span style='color: green'>(-3899710 [-1.9%])</span> 199,651,537 | <span style='color: green'>(-3899710 [-1.9%])</span> 199,651,537 |
| `total_cycles        ` | <span style='color: green'>(-67002 [-1.6%])</span> 4,097,903 | <span style='color: green'>(-67002 [-1.6%])</span> 4,097,903 | <span style='color: green'>(-67002 [-1.6%])</span> 4,097,903 | <span style='color: green'>(-67002 [-1.6%])</span> 4,097,903 |
| `execute_time_ms     ` | <span style='color: green'>(-20 [-2.0%])</span> 992 | <span style='color: green'>(-20 [-2.0%])</span> 992 | <span style='color: green'>(-20 [-2.0%])</span> 992 | <span style='color: green'>(-20 [-2.0%])</span> 992 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+4 [+0.1%])</span> 3,615 | <span style='color: red'>(+4 [+0.1%])</span> 3,615 | <span style='color: red'>(+4 [+0.1%])</span> 3,615 | <span style='color: red'>(+4 [+0.1%])</span> 3,615 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-878 [-5.9%])</span> 14,074 | <span style='color: green'>(-878 [-5.9%])</span> 14,074 | <span style='color: green'>(-878 [-5.9%])</span> 14,074 | <span style='color: green'>(-878 [-5.9%])</span> 14,074 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-12 [-0.4%])</span> 2,933 | <span style='color: green'>(-12 [-0.4%])</span> 2,933 | <span style='color: green'>(-12 [-0.4%])</span> 2,933 | <span style='color: green'>(-12 [-0.4%])</span> 2,933 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-53 [-12.6%])</span> 367 | <span style='color: green'>(-53 [-12.6%])</span> 367 | <span style='color: green'>(-53 [-12.6%])</span> 367 | <span style='color: green'>(-53 [-12.6%])</span> 367 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-703 [-19.3%])</span> 2,945 | <span style='color: green'>(-703 [-19.3%])</span> 2,945 | <span style='color: green'>(-703 [-19.3%])</span> 2,945 | <span style='color: green'>(-703 [-19.3%])</span> 2,945 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-45 [-2.1%])</span> 2,083 | <span style='color: green'>(-45 [-2.1%])</span> 2,083 | <span style='color: green'>(-45 [-2.1%])</span> 2,083 | <span style='color: green'>(-45 [-2.1%])</span> 2,083 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+25 [+0.9%])</span> 2,657 | <span style='color: red'>(+25 [+0.9%])</span> 2,657 | <span style='color: red'>(+25 [+0.9%])</span> 2,657 | <span style='color: red'>(+25 [+0.9%])</span> 2,657 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-91 [-2.9%])</span> 3,085 | <span style='color: green'>(-91 [-2.9%])</span> 3,085 | <span style='color: green'>(-91 [-2.9%])</span> 3,085 | <span style='color: green'>(-91 [-2.9%])</span> 3,085 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,180 | 11 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<64> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,513 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 36 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 223 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 80 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 411 | 481 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 156 | 189 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 456 | 
| ecrecover_program | VmConnectorAir | 2 | 3 | 9 | 
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
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 16 | 25 | 671,744 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,164 | 540,160 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 12 | 6 | 1,152 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 52 | 36 | 11,534,336 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 256 |  | 36 | 26 | 15,872 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 48 | 35 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 40 | 12,058,624 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 52 | 31 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 21 | 200,704 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 316 | 261 | 18,464 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 3,615 | 18,681 | 4,097,903 | 466,306,008 | 14,074 | 2,083 | 2,657 | 2,945 | 3,085 | 2,933 | 199,651,537 | 367 | 992 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 251 | 2,404 | 285,401 | 47,898,015 | 2,006 | 156 | 322 | 468 | 621 | 392 | 15,075,033 | 34 | 147 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/cadd0d1f57c845f80378b02baed0424e3c2f394d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12968961704)
