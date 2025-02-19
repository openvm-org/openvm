| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+2.7%])</span> 2.51 | <span style='color: red'>(+0 [+2.7%])</span> 2.51 |
| ecrecover_program | <span style='color: red'>(+0 [+2.7%])</span> 2.51 | <span style='color: red'>(+0 [+2.7%])</span> 2.51 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+66 [+2.7%])</span> 2,506 | <span style='color: red'>(+66 [+2.7%])</span> 2,506 | <span style='color: red'>(+66 [+2.7%])</span> 2,506 | <span style='color: red'>(+66 [+2.7%])</span> 2,506 |
| `main_cells_used     ` |  15,055,723 |  15,055,723 |  15,055,723 |  15,055,723 |
| `total_cycles        ` |  284,567 |  284,567 |  284,567 |  284,567 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-1.3%])</span> 147 | <span style='color: green'>(-2 [-1.3%])</span> 147 | <span style='color: green'>(-2 [-1.3%])</span> 147 | <span style='color: green'>(-2 [-1.3%])</span> 147 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+8 [+4.7%])</span> 180 | <span style='color: red'>(+8 [+4.7%])</span> 180 | <span style='color: red'>(+8 [+4.7%])</span> 180 | <span style='color: red'>(+8 [+4.7%])</span> 180 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+60 [+2.8%])</span> 2,179 | <span style='color: red'>(+60 [+2.8%])</span> 2,179 | <span style='color: red'>(+60 [+2.8%])</span> 2,179 | <span style='color: red'>(+60 [+2.8%])</span> 2,179 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+6 [+1.6%])</span> 388 | <span style='color: red'>(+6 [+1.6%])</span> 388 | <span style='color: red'>(+6 [+1.6%])</span> 388 | <span style='color: red'>(+6 [+1.6%])</span> 388 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+5 [+16.1%])</span> 36 | <span style='color: red'>(+5 [+16.1%])</span> 36 | <span style='color: red'>(+5 [+16.1%])</span> 36 | <span style='color: red'>(+5 [+16.1%])</span> 36 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+7 [+2.0%])</span> 359 | <span style='color: red'>(+7 [+2.0%])</span> 359 | <span style='color: red'>(+7 [+2.0%])</span> 359 | <span style='color: red'>(+7 [+2.0%])</span> 359 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-0.4%])</span> 225 | <span style='color: green'>(-1 [-0.4%])</span> 225 | <span style='color: green'>(-1 [-0.4%])</span> 225 | <span style='color: green'>(-1 [-0.4%])</span> 225 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+38 [+8.3%])</span> 498 | <span style='color: red'>(+38 [+8.3%])</span> 498 | <span style='color: red'>(+38 [+8.3%])</span> 498 | <span style='color: red'>(+38 [+8.3%])</span> 498 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+4 [+0.6%])</span> 659 | <span style='color: red'>(+4 [+0.6%])</span> 659 | <span style='color: red'>(+4 [+0.6%])</span> 659 | <span style='color: red'>(+4 [+0.6%])</span> 659 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,161 | 11 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 180 | 2,506 | 284,567 | 38,417,467 | 2,179 | 225 | 498 | 359 | 659 | 388 | 15,055,723 | 36 | 147 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/2d9a747f9ce0ad6c89b46b19ad51fa0f916e3558

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13417585380)
