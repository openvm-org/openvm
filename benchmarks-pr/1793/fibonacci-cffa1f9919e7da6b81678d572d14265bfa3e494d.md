| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-4.1%])</span> 2.61 | <span style='color: green'>(-0 [-4.1%])</span> 2.61 |
| fibonacci_program | <span style='color: green'>(-0 [-4.2%])</span> 2.50 | <span style='color: green'>(-0 [-4.2%])</span> 2.50 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-109 [-4.2%])</span> 2,497 | <span style='color: green'>(-109 [-4.2%])</span> 2,497 | <span style='color: green'>(-109 [-4.2%])</span> 2,497 | <span style='color: green'>(-109 [-4.2%])</span> 2,497 |
| `main_cells_used     ` |  50,589,503 |  50,589,503 |  50,589,503 |  50,589,503 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_metered_time_ms` | <span style='color: green'>(-2 [-1.7%])</span> 115 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+1 [+3.2%])</span> 17.48 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: red'>(+27 [+23.3%])</span> 143 | <span style='color: red'>(+27 [+23.3%])</span> 143 | <span style='color: red'>(+27 [+23.3%])</span> 143 | <span style='color: red'>(+27 [+23.3%])</span> 143 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-2 [-18.8%])</span> 10.47 | -          | <span style='color: green'>(-2 [-18.8%])</span> 10.47 | <span style='color: green'>(-2 [-18.8%])</span> 10.47 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-245 [-34.7%])</span> 462 | <span style='color: green'>(-245 [-34.7%])</span> 462 | <span style='color: green'>(-245 [-34.7%])</span> 462 | <span style='color: green'>(-245 [-34.7%])</span> 462 |
| `memory_finalize_time_ms` | <span style='color: green'>(-247 [-79.4%])</span> 64 | <span style='color: green'>(-247 [-79.4%])</span> 64 | <span style='color: green'>(-247 [-79.4%])</span> 64 | <span style='color: green'>(-247 [-79.4%])</span> 64 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-1 [-1.6%])</span> 62 | <span style='color: green'>(-1 [-1.6%])</span> 62 | <span style='color: green'>(-1 [-1.6%])</span> 62 | <span style='color: green'>(-1 [-1.6%])</span> 62 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+109 [+6.1%])</span> 1,892 | <span style='color: red'>(+109 [+6.1%])</span> 1,892 | <span style='color: red'>(+109 [+6.1%])</span> 1,892 | <span style='color: red'>(+109 [+6.1%])</span> 1,892 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+34 [+10.6%])</span> 356 | <span style='color: red'>(+34 [+10.6%])</span> 356 | <span style='color: red'>(+34 [+10.6%])</span> 356 | <span style='color: red'>(+34 [+10.6%])</span> 356 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+44 [+12.9%])</span> 386 | <span style='color: red'>(+44 [+12.9%])</span> 386 | <span style='color: red'>(+44 [+12.9%])</span> 386 | <span style='color: red'>(+44 [+12.9%])</span> 386 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-1.2%])</span> 170 | <span style='color: green'>(-2 [-1.2%])</span> 170 | <span style='color: green'>(-2 [-1.2%])</span> 170 | <span style='color: green'>(-2 [-1.2%])</span> 170 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+24 [+13.2%])</span> 206 | <span style='color: red'>(+24 [+13.2%])</span> 206 | <span style='color: red'>(+24 [+13.2%])</span> 206 | <span style='color: red'>(+24 [+13.2%])</span> 206 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+8 [+1.3%])</span> 624 | <span style='color: red'>(+8 [+1.3%])</span> 624 | <span style='color: red'>(+8 [+1.3%])</span> 624 | <span style='color: red'>(+8 [+1.3%])</span> 624 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 249 | 6 | 6,620 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 1 | 22 | 1,500,278 | 1 | 6,055 | 115 | 17.48 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
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
| fibonacci_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 128 |  | 16 | 17 | 4,224 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 16 | 32 | 24,576 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 128 |  | 12 | 20 | 4,096 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32 |  | 36 | 28 | 2,048 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 52 | 41 | 11,904 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 20 | 768 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 462 | 2,497 | 1,500,277 | 160,837,996 | 1,892 | 170 | 206 | 2,066 | 386 | 624 | 62 | 23 | 64 | 356 | 50,589,503 | 1,500,278 | 141 | 143 | 10.47 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/cffa1f9919e7da6b81678d572d14265bfa3e494d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15888792362)
