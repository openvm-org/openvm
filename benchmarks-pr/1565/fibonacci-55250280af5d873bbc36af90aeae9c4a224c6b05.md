| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+2 [+89.0%])</span> 5.09 | <span style='color: red'>(+2 [+89.0%])</span> 5.09 |
| fibonacci_program | <span style='color: red'>(+2 [+89.0%])</span> 5.09 | <span style='color: red'>(+2 [+89.0%])</span> 5.09 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2396 [+89.0%])</span> 5,087 | <span style='color: red'>(+2396 [+89.0%])</span> 5,087 | <span style='color: red'>(+2396 [+89.0%])</span> 5,087 | <span style='color: red'>(+2396 [+89.0%])</span> 5,087 |
| `main_cells_used     ` |  50,589,503 |  50,589,503 |  50,589,503 |  50,589,503 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.2%])</span> 558 | <span style='color: green'>(-1 [-0.2%])</span> 558 | <span style='color: green'>(-1 [-0.2%])</span> 558 | <span style='color: green'>(-1 [-0.2%])</span> 558 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+2396 [+126.1%])</span> 4,296 | <span style='color: red'>(+2396 [+126.1%])</span> 4,296 | <span style='color: red'>(+2396 [+126.1%])</span> 4,296 | <span style='color: red'>(+2396 [+126.1%])</span> 4,296 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+33 [+10.0%])</span> 363 | <span style='color: red'>(+33 [+10.0%])</span> 363 | <span style='color: red'>(+33 [+10.0%])</span> 363 | <span style='color: red'>(+33 [+10.0%])</span> 363 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+15 [+13.3%])</span> 128 | <span style='color: red'>(+15 [+13.3%])</span> 128 | <span style='color: red'>(+15 [+13.3%])</span> 128 | <span style='color: red'>(+15 [+13.3%])</span> 128 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-249 [-52.1%])</span> 229 | <span style='color: green'>(-249 [-52.1%])</span> 229 | <span style='color: green'>(-249 [-52.1%])</span> 229 | <span style='color: green'>(-249 [-52.1%])</span> 229 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+578 [+229.4%])</span> 830 | <span style='color: red'>(+578 [+229.4%])</span> 830 | <span style='color: red'>(+578 [+229.4%])</span> 830 | <span style='color: red'>(+578 [+229.4%])</span> 830 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+25 [+11.3%])</span> 247 | <span style='color: red'>(+25 [+11.3%])</span> 247 | <span style='color: red'>(+25 [+11.3%])</span> 247 | <span style='color: red'>(+25 [+11.3%])</span> 247 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+817 [+163.1%])</span> 1,318 | <span style='color: red'>(+817 [+163.1%])</span> 1,318 | <span style='color: red'>(+817 [+163.1%])</span> 1,318 | <span style='color: red'>(+817 [+163.1%])</span> 1,318 |
| `sumcheck_prove_batch_ms` |  604 |  604 |  604 |  604 |
| `gkr_prove_batch_ms  ` |  770 |  770 |  770 |  770 |
| `gkr_gen_layers_ms   ` |  104 |  104 |  104 |  104 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 253 | 5 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_gen_layers_ms | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 558 | 5,087 | 1,500,277 | 101,356,100 | 604 | 4,296 | 830 | 247 | 229 | 1,318 | 363 | 50,589,503 | 770 | 104 | 128 | 233 | 

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


Commit: https://github.com/openvm-org/openvm/commit/55250280af5d873bbc36af90aeae9c4a224c6b05

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14389565219)
