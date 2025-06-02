| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+8.6%])</span> 1.51 | <span style='color: red'>(+0 [+8.6%])</span> 1.51 |
| ecrecover_program | <span style='color: red'>(+0 [+8.6%])</span> 1.51 | <span style='color: red'>(+0 [+8.6%])</span> 1.51 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+119 [+8.6%])</span> 1,508 | <span style='color: red'>(+119 [+8.6%])</span> 1,508 | <span style='color: red'>(+119 [+8.6%])</span> 1,508 | <span style='color: red'>(+119 [+8.6%])</span> 1,508 |
| `main_cells_used     ` | <span style='color: red'>(+2112345 [+14.6%])</span> 16,582,531 | <span style='color: red'>(+2112345 [+14.6%])</span> 16,582,531 | <span style='color: red'>(+2112345 [+14.6%])</span> 16,582,531 | <span style='color: red'>(+2112345 [+14.6%])</span> 16,582,531 |
| `total_cycles        ` | <span style='color: red'>(+62024 [+21.4%])</span> 351,471 | <span style='color: red'>(+62024 [+21.4%])</span> 351,471 | <span style='color: red'>(+62024 [+21.4%])</span> 351,471 | <span style='color: red'>(+62024 [+21.4%])</span> 351,471 |
| `execute_time_ms     ` | <span style='color: red'>(+12 [+8.2%])</span> 158 | <span style='color: red'>(+12 [+8.2%])</span> 158 | <span style='color: red'>(+12 [+8.2%])</span> 158 | <span style='color: red'>(+12 [+8.2%])</span> 158 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+24 [+13.3%])</span> 205 | <span style='color: red'>(+24 [+13.3%])</span> 205 | <span style='color: red'>(+24 [+13.3%])</span> 205 | <span style='color: red'>(+24 [+13.3%])</span> 205 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+83 [+7.8%])</span> 1,145 | <span style='color: red'>(+83 [+7.8%])</span> 1,145 | <span style='color: red'>(+83 [+7.8%])</span> 1,145 | <span style='color: red'>(+83 [+7.8%])</span> 1,145 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+5 [+2.8%])</span> 183 | <span style='color: red'>(+5 [+2.8%])</span> 183 | <span style='color: red'>(+5 [+2.8%])</span> 183 | <span style='color: red'>(+5 [+2.8%])</span> 183 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+12 [+35.3%])</span> 46 | <span style='color: red'>(+12 [+35.3%])</span> 46 | <span style='color: red'>(+12 [+35.3%])</span> 46 | <span style='color: red'>(+12 [+35.3%])</span> 46 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+16 [+9.9%])</span> 177 | <span style='color: red'>(+16 [+9.9%])</span> 177 | <span style='color: red'>(+16 [+9.9%])</span> 177 | <span style='color: red'>(+16 [+9.9%])</span> 177 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+20 [+18.5%])</span> 128 | <span style='color: red'>(+20 [+18.5%])</span> 128 | <span style='color: red'>(+20 [+18.5%])</span> 128 | <span style='color: red'>(+20 [+18.5%])</span> 128 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-6 [-3.9%])</span> 149 | <span style='color: green'>(-6 [-3.9%])</span> 149 | <span style='color: green'>(-6 [-3.9%])</span> 149 | <span style='color: green'>(-6 [-3.9%])</span> 149 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+37 [+9.0%])</span> 448 | <span style='color: red'>(+37 [+9.0%])</span> 448 | <span style='color: red'>(+37 [+9.0%])</span> 448 | <span style='color: red'>(+37 [+9.0%])</span> 448 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 909 | 8 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,513 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
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
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 11 | 

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
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 16,384 |  | 28 | 18 | 753,664 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 52 | 36 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 262,144 |  | 52 | 41 | 24,379,392 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 52 | 31 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 205 | 1,508 | 351,471 | 62,418,746 | 1,145 | 128 | 149 | 177 | 448 | 183 | 16,582,531 | 46 | 158 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 1,072,132 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 3,182,736 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 536,066 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 5,001,884 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 936,152 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 16,448 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 11,707,978 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/44fe2a6cf0f543bc15f335be16dcedd3161a49aa

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15386042427)
