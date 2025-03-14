| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-2.2%])</span> 2.63 | <span style='color: green'>(-0 [-2.2%])</span> 2.63 |
| fibonacci_program | <span style='color: green'>(-0 [-2.2%])</span> 2.63 | <span style='color: green'>(-0 [-2.2%])</span> 2.63 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-58 [-2.2%])</span> 2,633 | <span style='color: green'>(-58 [-2.2%])</span> 2,633 | <span style='color: green'>(-58 [-2.2%])</span> 2,633 | <span style='color: green'>(-58 [-2.2%])</span> 2,633 |
| `main_cells_used     ` |  51,485,167 |  51,485,167 |  51,485,167 |  51,485,167 |
| `total_cycles        ` |  1,500,096 |  1,500,096 |  1,500,096 |  1,500,096 |
| `execute_time_ms     ` | <span style='color: green'>(-7 [-3.0%])</span> 227 | <span style='color: green'>(-7 [-3.0%])</span> 227 | <span style='color: green'>(-7 [-3.0%])</span> 227 | <span style='color: green'>(-7 [-3.0%])</span> 227 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-40 [-7.1%])</span> 521 | <span style='color: green'>(-40 [-7.1%])</span> 521 | <span style='color: green'>(-40 [-7.1%])</span> 521 | <span style='color: green'>(-40 [-7.1%])</span> 521 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-11 [-0.6%])</span> 1,885 | <span style='color: green'>(-11 [-0.6%])</span> 1,885 | <span style='color: green'>(-11 [-0.6%])</span> 1,885 | <span style='color: green'>(-11 [-0.6%])</span> 1,885 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.6%])</span> 325 | <span style='color: green'>(-2 [-0.6%])</span> 325 | <span style='color: green'>(-2 [-0.6%])</span> 325 | <span style='color: green'>(-2 [-0.6%])</span> 325 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-2.7%])</span> 109 | <span style='color: green'>(-3 [-2.7%])</span> 109 | <span style='color: green'>(-3 [-2.7%])</span> 109 | <span style='color: green'>(-3 [-2.7%])</span> 109 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+1 [+0.2%])</span> 473 | <span style='color: red'>(+1 [+0.2%])</span> 473 | <span style='color: red'>(+1 [+0.2%])</span> 473 | <span style='color: red'>(+1 [+0.2%])</span> 473 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+4 [+1.6%])</span> 253 | <span style='color: red'>(+4 [+1.6%])</span> 253 | <span style='color: red'>(+4 [+1.6%])</span> 253 | <span style='color: red'>(+4 [+1.6%])</span> 253 |
| `quotient_poly_commit_time_ms` |  221 |  221 |  221 |  221 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 255 | 4 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| fibonacci_program | VmConnectorAir | 2 | 5 | 10 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 16 | 17 | 1,056 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 12 | 20 | 1,024 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 52 | 41 | 1,488 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 521 | 2,633 | 1,500,096 | 160,733,916 | 1,885 | 253 | 221 | 473 | 500 | 325 | 51,485,167 | 109 | 227 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,270 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,748,264 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,135 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,748,300 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 800 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 288 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,044 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,526,957 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f2ab00cbbbb9fd67ea68b43c767561a4cb2c3095

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13866584267)
