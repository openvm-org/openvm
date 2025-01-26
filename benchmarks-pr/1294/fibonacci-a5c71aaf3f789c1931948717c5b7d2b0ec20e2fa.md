| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.3%])</span> 5.26 | <span style='color: red'>(+0 [+3.3%])</span> 5.26 |
| fibonacci_program | <span style='color: red'>(+0 [+3.3%])</span> 5.26 | <span style='color: red'>(+0 [+3.3%])</span> 5.26 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+170 [+3.3%])</span> 5,265 | <span style='color: red'>(+170 [+3.3%])</span> 5,265 | <span style='color: red'>(+170 [+3.3%])</span> 5,265 | <span style='color: red'>(+170 [+3.3%])</span> 5,265 |
| `main_cells_used     ` |  51,485,080 |  51,485,080 |  51,485,080 |  51,485,080 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-0.6%])</span> 311 | <span style='color: green'>(-2 [-0.6%])</span> 311 | <span style='color: green'>(-2 [-0.6%])</span> 311 | <span style='color: green'>(-2 [-0.6%])</span> 311 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+18 [+2.3%])</span> 807 | <span style='color: red'>(+18 [+2.3%])</span> 807 | <span style='color: red'>(+18 [+2.3%])</span> 807 | <span style='color: red'>(+18 [+2.3%])</span> 807 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+154 [+3.9%])</span> 4,147 | <span style='color: red'>(+154 [+3.9%])</span> 4,147 | <span style='color: red'>(+154 [+3.9%])</span> 4,147 | <span style='color: red'>(+154 [+3.9%])</span> 4,147 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+68 [+8.5%])</span> 864 | <span style='color: red'>(+68 [+8.5%])</span> 864 | <span style='color: red'>(+68 [+8.5%])</span> 864 | <span style='color: red'>(+68 [+8.5%])</span> 864 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+5 [+3.8%])</span> 138 | <span style='color: red'>(+5 [+3.8%])</span> 138 | <span style='color: red'>(+5 [+3.8%])</span> 138 | <span style='color: red'>(+5 [+3.8%])</span> 138 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+44 [+5.9%])</span> 795 | <span style='color: red'>(+44 [+5.9%])</span> 795 | <span style='color: red'>(+44 [+5.9%])</span> 795 | <span style='color: red'>(+44 [+5.9%])</span> 795 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-12 [-2.3%])</span> 507 | <span style='color: green'>(-12 [-2.3%])</span> 507 | <span style='color: green'>(-12 [-2.3%])</span> 507 | <span style='color: green'>(-12 [-2.3%])</span> 507 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+25 [+3.4%])</span> 763 | <span style='color: red'>(+25 [+3.4%])</span> 763 | <span style='color: red'>(+25 [+3.4%])</span> 763 | <span style='color: red'>(+25 [+3.4%])</span> 763 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+23 [+2.2%])</span> 1,076 | <span style='color: red'>(+23 [+2.2%])</span> 1,076 | <span style='color: red'>(+23 [+2.2%])</span> 1,076 | <span style='color: red'>(+23 [+2.2%])</span> 1,076 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 403 | 5 | 

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
| fibonacci_program | 0 | 807 | 5,265 | 1,500,095 | 122,458,476 | 4,147 | 507 | 763 | 795 | 1,076 | 864 | 51,485,080 | 138 | 311 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/a5c71aaf3f789c1931948717c5b7d2b0ec20e2fa

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12977285375)