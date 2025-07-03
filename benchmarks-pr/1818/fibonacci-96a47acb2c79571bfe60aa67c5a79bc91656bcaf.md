| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.8%])</span> 2.47 | <span style='color: red'>(+0 [+0.8%])</span> 2.47 |
| fibonacci_program | <span style='color: red'>(+0 [+0.9%])</span> 2.36 | <span style='color: red'>(+0 [+0.9%])</span> 2.36 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+22 [+0.9%])</span> 2,356 | <span style='color: red'>(+22 [+0.9%])</span> 2,356 | <span style='color: red'>(+22 [+0.9%])</span> 2,356 | <span style='color: red'>(+22 [+0.9%])</span> 2,356 |
| `main_cells_used     ` |  50,589,231 |  50,589,231 |  50,589,231 |  50,589,231 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_metered_time_ms` | <span style='color: green'>(-2 [-1.7%])</span> 114 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+2.7%])</span> 17.45 | -          | <span style='color: red'>(+0 [+2.7%])</span> 17.45 | <span style='color: red'>(+0 [+2.7%])</span> 17.45 |
| `execute_e3_time_ms  ` | <span style='color: red'>(+1 [+1.0%])</span> 97 | <span style='color: red'>(+1 [+1.0%])</span> 97 | <span style='color: red'>(+1 [+1.0%])</span> 97 | <span style='color: red'>(+1 [+1.0%])</span> 97 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-0.4%])</span> 15.41 | -          | <span style='color: green'>(-0 [-0.4%])</span> 15.41 | <span style='color: green'>(-0 [-0.4%])</span> 15.41 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+9 [+2.0%])</span> 461 | <span style='color: red'>(+9 [+2.0%])</span> 461 | <span style='color: red'>(+9 [+2.0%])</span> 461 | <span style='color: red'>(+9 [+2.0%])</span> 461 |
| `memory_finalize_time_ms` |  65 |  65 |  65 |  65 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` |  63 |  63 |  63 |  63 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+12 [+0.7%])</span> 1,798 | <span style='color: red'>(+12 [+0.7%])</span> 1,798 | <span style='color: red'>(+12 [+0.7%])</span> 1,798 | <span style='color: red'>(+12 [+0.7%])</span> 1,798 |
| `main_trace_commit_time_ms` |  324 |  324 |  324 |  324 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.7%])</span> 138 | <span style='color: green'>(-1 [-0.7%])</span> 138 | <span style='color: green'>(-1 [-0.7%])</span> 138 | <span style='color: green'>(-1 [-0.7%])</span> 138 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.9%])</span> 351 | <span style='color: red'>(+3 [+0.9%])</span> 351 | <span style='color: red'>(+3 [+0.9%])</span> 351 | <span style='color: red'>(+3 [+0.9%])</span> 351 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.6%])</span> 175 | <span style='color: red'>(+1 [+0.6%])</span> 175 | <span style='color: red'>(+1 [+0.6%])</span> 175 | <span style='color: red'>(+1 [+0.6%])</span> 175 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+6 [+3.3%])</span> 188 | <span style='color: red'>(+6 [+3.3%])</span> 188 | <span style='color: red'>(+6 [+3.3%])</span> 188 | <span style='color: red'>(+6 [+3.3%])</span> 188 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3 [+0.5%])</span> 610 | <span style='color: red'>(+3 [+0.5%])</span> 610 | <span style='color: red'>(+3 [+0.5%])</span> 610 | <span style='color: red'>(+3 [+0.5%])</span> 610 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 243 | 5 | 6,481 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 1 | 22 | 1,500,278 | 1 | 5,931 | 114 | 17.45 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 461 | 2,356 | 1,500,277 | 160,837,996 | 1,798 | 175 | 188 | 1,971 | 351 | 610 | 63 | 24 | 65 | 324 | 50,589,231 | 1,500,278 | 138 | 97 | 15.41 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/96a47acb2c79571bfe60aa67c5a79bc91656bcaf

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16059335861)
