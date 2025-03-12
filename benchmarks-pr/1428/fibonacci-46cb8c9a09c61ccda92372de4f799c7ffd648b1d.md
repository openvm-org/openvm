| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.8%])</span> 5.01 | <span style='color: red'>(+0 [+0.8%])</span> 5.01 |
| fibonacci_program | <span style='color: red'>(+0 [+0.8%])</span> 5.01 | <span style='color: red'>(+0 [+0.8%])</span> 5.01 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+40 [+0.8%])</span> 5,008 | <span style='color: red'>(+40 [+0.8%])</span> 5,008 | <span style='color: red'>(+40 [+0.8%])</span> 5,008 | <span style='color: red'>(+40 [+0.8%])</span> 5,008 |
| `main_cells_used     ` |  51,485,167 |  51,485,167 |  51,485,167 |  51,485,167 |
| `total_cycles        ` |  1,500,096 |  1,500,096 |  1,500,096 |  1,500,096 |
| `execute_time_ms     ` |  289 |  289 |  289 |  289 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+7 [+1.1%])</span> 623 | <span style='color: red'>(+7 [+1.1%])</span> 623 | <span style='color: red'>(+7 [+1.1%])</span> 623 | <span style='color: red'>(+7 [+1.1%])</span> 623 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+33 [+0.8%])</span> 4,096 | <span style='color: red'>(+33 [+0.8%])</span> 4,096 | <span style='color: red'>(+33 [+0.8%])</span> 4,096 | <span style='color: red'>(+33 [+0.8%])</span> 4,096 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+11 [+1.3%])</span> 862 | <span style='color: red'>(+11 [+1.3%])</span> 862 | <span style='color: red'>(+11 [+1.3%])</span> 862 | <span style='color: red'>(+11 [+1.3%])</span> 862 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+12 [+8.4%])</span> 155 | <span style='color: red'>(+12 [+8.4%])</span> 155 | <span style='color: red'>(+12 [+8.4%])</span> 155 | <span style='color: red'>(+12 [+8.4%])</span> 155 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+0.5%])</span> 801 | <span style='color: red'>(+4 [+0.5%])</span> 801 | <span style='color: red'>(+4 [+0.5%])</span> 801 | <span style='color: red'>(+4 [+0.5%])</span> 801 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-7 [-1.3%])</span> 522 | <span style='color: green'>(-7 [-1.3%])</span> 522 | <span style='color: green'>(-7 [-1.3%])</span> 522 | <span style='color: green'>(-7 [-1.3%])</span> 522 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+0.1%])</span> 755 | <span style='color: red'>(+1 [+0.1%])</span> 755 | <span style='color: red'>(+1 [+0.1%])</span> 755 | <span style='color: red'>(+1 [+0.1%])</span> 755 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+10 [+1.0%])</span> 995 | <span style='color: red'>(+10 [+1.0%])</span> 995 | <span style='color: red'>(+10 [+1.0%])</span> 995 | <span style='color: red'>(+10 [+1.0%])</span> 995 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 405 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| fibonacci_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| fibonacci_program | PhantomAir | 4 | 3 | 4 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 4 | 18 | 23 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 20 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 18 | 36 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 24 | 85 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 27 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 34 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 76 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 12 | 11 | 
| fibonacci_program | VmConnectorAir | 4 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 8 | 6 | 14 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 20 | 32 | 416 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 20 | 28 | 768 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 41 | 1,104 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 20 | 288 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 623 | 5,008 | 1,500,096 | 122,458,688 | 4,096 | 522 | 755 | 801 | 995 | 862 | 51,485,167 | 155 | 289 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/46cb8c9a09c61ccda92372de4f799c7ffd648b1d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13817759214)
