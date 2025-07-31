| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.9%])</span> 2.13 | <span style='color: green'>(-0 [-0.9%])</span> 2.13 |
| fibonacci_program | <span style='color: green'>(-0 [-0.9%])</span> 2.12 | <span style='color: green'>(-0 [-0.9%])</span> 2.12 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-19 [-0.9%])</span> 2,118 | <span style='color: green'>(-19 [-0.9%])</span> 2,118 | <span style='color: green'>(-19 [-0.9%])</span> 2,118 | <span style='color: green'>(-19 [-0.9%])</span> 2,118 |
| `main_cells_used     ` |  51,503,521 |  51,503,521 |  51,503,521 |  51,503,521 |
| `total_cells_used    ` |  127,358,595 |  127,358,595 |  127,358,595 |  127,358,595 |
| `insns               ` |  1,500,210 |  3,000,420 |  1,500,210 |  1,500,210 |
| `execute_metered_time_ms` |  7 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-2 [-1.2%])</span> 207.07 | -          | <span style='color: green'>(-2 [-1.2%])</span> 207.07 | <span style='color: green'>(-2 [-1.2%])</span> 207.07 |
| `execute_e3_time_ms  ` | <span style='color: red'>(+1 [+1.4%])</span> 73 | <span style='color: red'>(+1 [+1.4%])</span> 73 | <span style='color: red'>(+1 [+1.4%])</span> 73 | <span style='color: red'>(+1 [+1.4%])</span> 73 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-0.9%])</span> 20.46 | -          | <span style='color: green'>(-0 [-0.9%])</span> 20.46 | <span style='color: green'>(-0 [-0.9%])</span> 20.46 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-6 [-2.7%])</span> 213 | <span style='color: green'>(-6 [-2.7%])</span> 213 | <span style='color: green'>(-6 [-2.7%])</span> 213 | <span style='color: green'>(-6 [-2.7%])</span> 213 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` |  45 |  45 |  45 |  45 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-14 [-0.8%])</span> 1,832 | <span style='color: green'>(-14 [-0.8%])</span> 1,832 | <span style='color: green'>(-14 [-0.8%])</span> 1,832 | <span style='color: green'>(-14 [-0.8%])</span> 1,832 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-8 [-2.4%])</span> 331 | <span style='color: green'>(-8 [-2.4%])</span> 331 | <span style='color: green'>(-8 [-2.4%])</span> 331 | <span style='color: green'>(-8 [-2.4%])</span> 331 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-12 [-7.8%])</span> 142 | <span style='color: green'>(-12 [-7.8%])</span> 142 | <span style='color: green'>(-12 [-7.8%])</span> 142 | <span style='color: green'>(-12 [-7.8%])</span> 142 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.8%])</span> 359 | <span style='color: red'>(+3 [+0.8%])</span> 359 | <span style='color: red'>(+3 [+0.8%])</span> 359 | <span style='color: red'>(+3 [+0.8%])</span> 359 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-1.7%])</span> 170 | <span style='color: green'>(-3 [-1.7%])</span> 170 | <span style='color: green'>(-3 [-1.7%])</span> 170 | <span style='color: green'>(-3 [-1.7%])</span> 170 |
| `quotient_poly_commit_time_ms` |  192 |  192 |  192 |  192 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+5 [+0.8%])</span> 633 | <span style='color: red'>(+5 [+0.8%])</span> 633 | <span style='color: red'>(+5 [+0.8%])</span> 633 | <span style='color: red'>(+5 [+0.8%])</span> 633 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 210 | 5 | 2,406 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 2,358 | 6 | 1,500,210 | 1 | 7 | 207.07 | 37 | 

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
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 52 | 41 | 11,904 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 20 | 768 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 213 | 2,118 | 127,358,595 | 160,836,972 | 213 | 1,832 | 2 | 170 | 192 | 359 | 633 | 45 | 7 | 0 | 331 | 51,503,521 | 1,500,210 | 142 | 73 | 20.46 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,510 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,749,336 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,255 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,749,404 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 1,664 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 640 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,084 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,534,845 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/df57d45bef41be9b4d2625822c08ad169e5d49a2

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16654642033)
