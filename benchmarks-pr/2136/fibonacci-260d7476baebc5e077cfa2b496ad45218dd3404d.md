| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+45.2%])</span> 1.36 | <span style='color: green'>(-0 [-11.6%])</span> 0.83 |
| fibonacci_program | <span style='color: red'>(+0 [+45.5%])</span> 1.35 | <span style='color: green'>(-0 [-11.8%])</span> 0.82 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-253 [-27.2%])</span> 676 | <span style='color: red'>(+423 [+45.5%])</span> 1,352 | <span style='color: green'>(-110 [-11.8%])</span> 819 | <span style='color: green'>(-396 [-42.6%])</span> 533 |
| `main_cells_used     ` | <span style='color: green'>(-6251 [-0.6%])</span> 1,053,981 | <span style='color: red'>(+1047730 [+98.8%])</span> 2,107,962 | <span style='color: red'>(+7300 [+0.7%])</span> 1,067,532 | <span style='color: green'>(-19802 [-1.9%])</span> 1,040,430 |
| `total_cells_used    ` |  9,699,511 | <span style='color: red'>(+9691388 [+99.8%])</span> 19,399,022 |  9,714,030 | <span style='color: green'>(-22642 [-0.2%])</span> 9,684,992 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+11.1%])</span> 10 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-6 [-4.0%])</span> 147.69 | -          | <span style='color: green'>(-6 [-4.0%])</span> 147.69 | <span style='color: green'>(-6 [-4.0%])</span> 147.69 |
| `execute_preflight_insns` | <span style='color: green'>(-750105 [-50.0%])</span> 750,105 |  1,500,210 | <span style='color: green'>(-626210 [-41.7%])</span> 874,000 | <span style='color: green'>(-874000 [-58.3%])</span> 626,210 |
| `execute_preflight_time_ms` | <span style='color: green'>(-13 [-31.0%])</span> 29 | <span style='color: red'>(+16 [+38.1%])</span> 58 | <span style='color: green'>(-10 [-23.8%])</span> 32 | <span style='color: green'>(-16 [-38.1%])</span> 26 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+0.2%])</span> 36.29 | -          | <span style='color: red'>(+2 [+5.9%])</span> 38.36 | <span style='color: green'>(-2 [-5.6%])</span> 34.21 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+8 [+4.0%])</span> 206 | <span style='color: red'>(+214 [+108.1%])</span> 412 | <span style='color: red'>(+14 [+7.1%])</span> 212 | <span style='color: red'>(+2 [+1.0%])</span> 200 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-195 [-36.0%])</span> 347 | <span style='color: red'>(+152 [+28.0%])</span> 694 | <span style='color: green'>(-107 [-19.7%])</span> 435 | <span style='color: green'>(-283 [-52.2%])</span> 259 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-28 [-38.4%])</span> 45 | <span style='color: red'>(+17 [+23.3%])</span> 90 | <span style='color: green'>(-14 [-19.2%])</span> 59 | <span style='color: green'>(-42 [-57.5%])</span> 31 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-7 [-25.9%])</span> 20 | <span style='color: red'>(+13 [+48.1%])</span> 40 |  27 | <span style='color: green'>(-14 [-51.9%])</span> 13 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-37 [-37.9%])</span> 60.94 | <span style='color: red'>(+24 [+24.2%])</span> 121.89 | <span style='color: green'>(-18 [-18.5%])</span> 80.04 | <span style='color: green'>(-56 [-57.4%])</span> 41.85 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-22 [-35.6%])</span> 39.80 | <span style='color: red'>(+18 [+28.8%])</span> 79.60 | <span style='color: green'>(-11 [-17.8%])</span> 50.77 | <span style='color: green'>(-33 [-53.3%])</span> 28.83 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-6 [-27.6%])</span> 16.72 | <span style='color: red'>(+10 [+44.9%])</span> 33.45 | <span style='color: green'>(-2 [-10.3%])</span> 20.71 | <span style='color: green'>(-10 [-44.8%])</span> 12.74 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-94 [-36.8%])</span> 162.50 | <span style='color: red'>(+68 [+26.5%])</span> 325 | <span style='color: green'>(-62 [-24.1%])</span> 195 | <span style='color: green'>(-127 [-49.4%])</span> 130 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 73 | 329 | 1,528 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 533 | 59 | 1 | 10 | 1,500,210 | 147.69 | 160 | 

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
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | AccessAdapterAir<8> | 1 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | MemoryMerkleAir<8> | 1 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PersistentBoundaryAir<8> | 1 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 131,072 |  | 40 | 37 | 10,092,544 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 4 |  | 28 | 20 | 192 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 212 | 819 | 9,684,992 | 130,532,556 | 212 | 435 | 0 | 50.77 | 20.71 | 4 | 80.04 | 195 | 108 | 195 | 0 | 59 | 1,040,430 | 27 | 26 | 874,000 | 34.21 | 18 | 72 | 1 | 195 | 
| fibonacci_program | 1 | 200 | 533 | 9,714,030 | 74,305,770 | 200 | 259 | 1 | 28.83 | 12.74 | 4 | 41.85 | 130 | 55 | 130 | 0 | 31 | 1,067,532 | 13 | 32 | 626,210 | 38.36 | 10 | 41 | 1 | 130 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,014,766 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 8,520,200 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,507,383 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 8,520,156 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 6,225,984 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 28,715,593 | 2,013,265,921 | 
| fibonacci_program | 1 | 0 | 1,704,112 | 2,013,265,921 | 
| fibonacci_program | 1 | 1 | 4,588,240 | 2,013,265,921 | 
| fibonacci_program | 1 | 2 | 852,056 | 2,013,265,921 | 
| fibonacci_program | 1 | 3 | 4,588,308 | 2,013,265,921 | 
| fibonacci_program | 1 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 1 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 1 | 6 | 3,211,304 | 2,013,265,921 | 
| fibonacci_program | 1 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 1 | 8 | 15,871,124 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/260d7476baebc5e077cfa2b496ad45218dd3404d

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17746720247)
