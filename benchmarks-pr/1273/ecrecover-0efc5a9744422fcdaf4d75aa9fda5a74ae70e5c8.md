| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.9%])</span> 2.54 | <span style='color: green'>(-0 [-1.9%])</span> 2.54 |
| ecrecover_program | <span style='color: green'>(-0 [-1.9%])</span> 2.54 | <span style='color: green'>(-0 [-1.9%])</span> 2.54 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-49 [-1.9%])</span> 2,537 | <span style='color: green'>(-49 [-1.9%])</span> 2,537 | <span style='color: green'>(-49 [-1.9%])</span> 2,537 | <span style='color: green'>(-49 [-1.9%])</span> 2,537 |
| `main_cells_used     ` |  15,075,033 |  15,075,033 |  15,075,033 |  15,075,033 |
| `total_cycles        ` |  285,401 |  285,401 |  285,401 |  285,401 |
| `execute_time_ms     ` | <span style='color: green'>(-1 [-0.6%])</span> 153 | <span style='color: green'>(-1 [-0.6%])</span> 153 | <span style='color: green'>(-1 [-0.6%])</span> 153 | <span style='color: green'>(-1 [-0.6%])</span> 153 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.8%])</span> 254 | <span style='color: red'>(+2 [+0.8%])</span> 254 | <span style='color: red'>(+2 [+0.8%])</span> 254 | <span style='color: red'>(+2 [+0.8%])</span> 254 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-50 [-2.3%])</span> 2,130 | <span style='color: green'>(-50 [-2.3%])</span> 2,130 | <span style='color: green'>(-50 [-2.3%])</span> 2,130 | <span style='color: green'>(-50 [-2.3%])</span> 2,130 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-1.0%])</span> 388 | <span style='color: green'>(-4 [-1.0%])</span> 388 | <span style='color: green'>(-4 [-1.0%])</span> 388 | <span style='color: green'>(-4 [-1.0%])</span> 388 |
| `generate_perm_trace_time_ms` |  32 |  32 |  32 |  32 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-26 [-7.0%])</span> 345 | <span style='color: green'>(-26 [-7.0%])</span> 345 | <span style='color: green'>(-26 [-7.0%])</span> 345 | <span style='color: green'>(-26 [-7.0%])</span> 345 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-4 [-1.7%])</span> 227 | <span style='color: green'>(-4 [-1.7%])</span> 227 | <span style='color: green'>(-4 [-1.7%])</span> 227 | <span style='color: green'>(-4 [-1.7%])</span> 227 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-8 [-1.7%])</span> 472 | <span style='color: green'>(-8 [-1.7%])</span> 472 | <span style='color: green'>(-8 [-1.7%])</span> 472 | <span style='color: green'>(-8 [-1.7%])</span> 472 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-1.2%])</span> 653 | <span style='color: green'>(-8 [-1.2%])</span> 653 | <span style='color: green'>(-8 [-1.2%])</span> 653 | <span style='color: green'>(-8 [-1.2%])</span> 653 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,160 | 12 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 254 | 2,537 | 285,401 | 38,415,035 | 2,130 | 227 | 472 | 345 | 653 | 388 | 15,075,033 | 32 | 153 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/0efc5a9744422fcdaf4d75aa9fda5a74ae70e5c8

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12972192013)