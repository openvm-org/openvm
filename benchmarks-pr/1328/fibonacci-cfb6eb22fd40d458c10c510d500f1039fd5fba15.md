| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.7%])</span> 5.01 | <span style='color: red'>(+0 [+0.7%])</span> 5.01 |
| fibonacci_program | <span style='color: red'>(+0 [+0.7%])</span> 5.01 | <span style='color: red'>(+0 [+0.7%])</span> 5.01 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+35 [+0.7%])</span> 5,012 | <span style='color: red'>(+35 [+0.7%])</span> 5,012 | <span style='color: red'>(+35 [+0.7%])</span> 5,012 | <span style='color: red'>(+35 [+0.7%])</span> 5,012 |
| `main_cells_used     ` |  51,485,080 |  51,485,080 |  51,485,080 |  51,485,080 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+0.6%])</span> 313 | <span style='color: red'>(+2 [+0.6%])</span> 313 | <span style='color: red'>(+2 [+0.6%])</span> 313 | <span style='color: red'>(+2 [+0.6%])</span> 313 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+4 [+0.6%])</span> 676 | <span style='color: red'>(+4 [+0.6%])</span> 676 | <span style='color: red'>(+4 [+0.6%])</span> 676 | <span style='color: red'>(+4 [+0.6%])</span> 676 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+29 [+0.7%])</span> 4,023 | <span style='color: red'>(+29 [+0.7%])</span> 4,023 | <span style='color: red'>(+29 [+0.7%])</span> 4,023 | <span style='color: red'>(+29 [+0.7%])</span> 4,023 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+6 [+0.8%])</span> 800 | <span style='color: red'>(+6 [+0.8%])</span> 800 | <span style='color: red'>(+6 [+0.8%])</span> 800 | <span style='color: red'>(+6 [+0.8%])</span> 800 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+6.4%])</span> 149 | <span style='color: red'>(+9 [+6.4%])</span> 149 | <span style='color: red'>(+9 [+6.4%])</span> 149 | <span style='color: red'>(+9 [+6.4%])</span> 149 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+5 [+0.7%])</span> 745 | <span style='color: red'>(+5 [+0.7%])</span> 745 | <span style='color: red'>(+5 [+0.7%])</span> 745 | <span style='color: red'>(+5 [+0.7%])</span> 745 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-0.2%])</span> 515 | <span style='color: green'>(-1 [-0.2%])</span> 515 | <span style='color: green'>(-1 [-0.2%])</span> 515 | <span style='color: green'>(-1 [-0.2%])</span> 515 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+5 [+0.7%])</span> 729 | <span style='color: red'>(+5 [+0.7%])</span> 729 | <span style='color: red'>(+5 [+0.7%])</span> 729 | <span style='color: red'>(+5 [+0.7%])</span> 729 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+4 [+0.4%])</span> 1,081 | <span style='color: red'>(+4 [+0.4%])</span> 1,081 | <span style='color: red'>(+4 [+0.4%])</span> 1,081 | <span style='color: red'>(+4 [+0.4%])</span> 1,081 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 394 | 5 | 

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
| fibonacci_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| fibonacci_program | VmConnectorAir | 4 | 3 | 8 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 8 | 6 | 28 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 20 | 32 | 208 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 20 | 28 | 768 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 40 | 1,088 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 21 | 296 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 676 | 5,012 | 1,500,095 | 122,458,476 | 4,023 | 515 | 729 | 745 | 1,081 | 800 | 51,485,080 | 149 | 313 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/cfb6eb22fd40d458c10c510d500f1039fd5fba15

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13086962730)
