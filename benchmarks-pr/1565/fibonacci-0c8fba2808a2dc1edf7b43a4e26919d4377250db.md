| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+2 [+86.8%])</span> 5 | <span style='color: red'>(+2 [+86.8%])</span> 5 |
| fibonacci_program | <span style='color: red'>(+2 [+86.8%])</span> 5 | <span style='color: red'>(+2 [+86.8%])</span> 5 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2325 [+86.8%])</span> 5,005 | <span style='color: red'>(+2325 [+86.8%])</span> 5,005 | <span style='color: red'>(+2325 [+86.8%])</span> 5,005 | <span style='color: red'>(+2325 [+86.8%])</span> 5,005 |
| `main_cells_used     ` |  50,589,503 |  50,589,503 |  50,589,503 |  50,589,503 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_time_ms     ` | <span style='color: green'>(-1 [-0.4%])</span> 230 | <span style='color: green'>(-1 [-0.4%])</span> 230 | <span style='color: green'>(-1 [-0.4%])</span> 230 | <span style='color: green'>(-1 [-0.4%])</span> 230 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+1 [+0.2%])</span> 552 | <span style='color: red'>(+1 [+0.2%])</span> 552 | <span style='color: red'>(+1 [+0.2%])</span> 552 | <span style='color: red'>(+1 [+0.2%])</span> 552 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+2325 [+122.5%])</span> 4,223 | <span style='color: red'>(+2325 [+122.5%])</span> 4,223 | <span style='color: red'>(+2325 [+122.5%])</span> 4,223 | <span style='color: red'>(+2325 [+122.5%])</span> 4,223 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+25 [+7.5%])</span> 360 | <span style='color: red'>(+25 [+7.5%])</span> 360 | <span style='color: red'>(+25 [+7.5%])</span> 360 | <span style='color: red'>(+25 [+7.5%])</span> 360 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-247 [-52.2%])</span> 226 | <span style='color: green'>(-247 [-52.2%])</span> 226 | <span style='color: green'>(-247 [-52.2%])</span> 226 | <span style='color: green'>(-247 [-52.2%])</span> 226 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+574 [+226.0%])</span> 828 | <span style='color: red'>(+574 [+226.0%])</span> 828 | <span style='color: red'>(+574 [+226.0%])</span> 828 | <span style='color: red'>(+574 [+226.0%])</span> 828 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+27 [+12.4%])</span> 245 | <span style='color: red'>(+27 [+12.4%])</span> 245 | <span style='color: red'>(+27 [+12.4%])</span> 245 | <span style='color: red'>(+27 [+12.4%])</span> 245 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+806 [+161.8%])</span> 1,304 | <span style='color: red'>(+806 [+161.8%])</span> 1,304 | <span style='color: red'>(+806 [+161.8%])</span> 1,304 | <span style='color: red'>(+806 [+161.8%])</span> 1,304 |
| `sumcheck_prove_batch_ms` |  653 |  653 |  653 |  653 |
| `gkr_prove_batch_ms  ` |  830 |  830 |  830 |  830 |
| `gkr_gen_layers_ms   ` |  110 |  110 |  110 |  110 |



<details>
<summary>Detailed Metrics</summary>

|  | generate_perm_trace_time_ms |
| --- |
|  | 288 | 

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 248 | 4 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 10 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 10 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 10 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 10 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 10 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 37 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 4 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 2 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 2 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 19 | 
| fibonacci_program | VariableRangeCheckerAir | 2 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 32 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 80 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 29 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 29 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 68 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 8 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 9 | 
| fibonacci_program | VmConnectorAir | 2 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 128 |  | 12 | 17 | 3,712 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 12 | 32 | 22,528 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 128 |  | 12 | 20 | 4,096 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 12 | 300 | 79,872 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 12 | 10 | 180,224 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 12 | 1 | 6,815,744 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 12 | 32 | 176 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 12 | 36 | 50,331,648 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 12 | 37 | 25,690,112 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 12 | 26 | 9,961,472 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 12 | 32 | 352 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 12 | 18 | 3,932,160 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32 |  | 12 | 28 | 1,280 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 12 | 41 | 6,784 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_gen_layers_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 552 | 5,005 | 1,500,277 | 101,356,100 | 653 | 4,223 | 828 | 245 | 226 | 1,304 | 360 | 50,589,503 | 830 | 110 | 230 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,542 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,749,400 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,271 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,749,532 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 1,664 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 640 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,100 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,535,101 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/0c8fba2808a2dc1edf7b43a4e26919d4377250db

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14386950043)
