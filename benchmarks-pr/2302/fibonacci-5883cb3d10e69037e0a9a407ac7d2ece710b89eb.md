| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.7%])</span> 1.04 | <span style='color: green'>(-0 [-4.2%])</span> 0.60 |
| fibonacci_program | <span style='color: green'>(-0 [-0.7%])</span> 1.03 | <span style='color: green'>(-0 [-4.3%])</span> 0.58 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-4 [-0.7%])</span> 516 | <span style='color: green'>(-7 [-0.7%])</span> 1,032 | <span style='color: green'>(-26 [-4.3%])</span> 582 | <span style='color: red'>(+19 [+4.4%])</span> 450 |
| `main_cells_used     ` | <span style='color: red'>(+1174 [+0.1%])</span> 1,051,375 | <span style='color: red'>(+2348 [+0.1%])</span> 2,102,750 |  1,064,414 | <span style='color: red'>(+2350 [+0.2%])</span> 1,038,336 |
| `total_cells_used    ` |  9,694,601 |  19,389,202 |  9,708,456 |  9,680,746 |
| `execute_metered_time_ms` |  13 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-0.3%])</span> 114.03 | -          | <span style='color: green'>(-0 [-0.3%])</span> 114.03 | <span style='color: green'>(-0 [-0.3%])</span> 114.03 |
| `execute_preflight_insns` |  750,134.50 |  1,500,269 |  873,000 |  627,269 |
| `execute_preflight_time_ms` |  29.50 |  59 | <span style='color: green'>(-1 [-2.9%])</span> 33 | <span style='color: red'>(+1 [+4.0%])</span> 26 |
| `execute_preflight_insn_mi/s` |  36.45 | -          | <span style='color: red'>(+0 [+0.6%])</span> 38.30 | <span style='color: green'>(-0 [-0.8%])</span> 34.59 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.5%])</span> 188 | <span style='color: green'>(-2 [-0.5%])</span> 376 | <span style='color: green'>(-3 [-1.5%])</span> 193 | <span style='color: red'>(+1 [+0.5%])</span> 183 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1 [-0.5%])</span> 194 | <span style='color: green'>(-2 [-0.5%])</span> 388 | <span style='color: green'>(-22 [-10.1%])</span> 195 | <span style='color: red'>(+20 [+11.6%])</span> 193 |
| `main_trace_commit_time_ms` |  30.50 |  61 |  33 |  28 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-10 [-38.5%])</span> 16 | <span style='color: green'>(-20 [-38.5%])</span> 32 | <span style='color: green'>(-11 [-35.5%])</span> 20 | <span style='color: green'>(-9 [-42.9%])</span> 12 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+0 [+0.4%])</span> 40.95 | <span style='color: red'>(+0 [+0.4%])</span> 81.90 | <span style='color: red'>(+0 [+0.5%])</span> 43.61 | <span style='color: red'>(+0 [+0.2%])</span> 38.29 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-3.0%])</span> 29.100 | <span style='color: green'>(-2 [-3.0%])</span> 59.99 | <span style='color: green'>(-4 [-11.6%])</span> 30.19 | <span style='color: red'>(+2 [+7.7%])</span> 29.81 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-8.1%])</span> 11.46 | <span style='color: green'>(-2 [-8.1%])</span> 22.92 | <span style='color: green'>(-1 [-9.1%])</span> 11.56 | <span style='color: green'>(-1 [-7.2%])</span> 11.36 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+10 [+19.0%])</span> 62.50 | <span style='color: red'>(+20 [+19.0%])</span> 125 | <span style='color: red'>(+10 [+16.4%])</span> 71 | <span style='color: red'>(+10 [+22.7%])</span> 54 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 342 | 1,051 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 450 | 1 | 13 | 1,500,269 | 114.03 | 0 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | 0 | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 13 | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 4 |  | 28 | 20 | 192 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 1 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 4 |  | 52 | 53 | 420 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 131,072 |  | 40 | 37 | 10,092,544 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 25 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 1 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 1 | 64 |  | 16 | 17 | 2,112 | 

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
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 29 | 
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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 193 | 582 | 9,680,746 | 84,399,632 | 193 | 195 | 0 | 29.81 | 11.36 | 4 | 43.61 | 54 | 65 | 54 | 0 | 33 | 1,038,336 | 20 | 26 | 873,000 | 34.59 | 14 | 41 | 0 | 54 | 
| fibonacci_program | 1 | 183 | 450 | 9,708,456 | 74,305,770 | 183 | 193 | 1 | 30.19 | 11.56 | 4 | 38.29 | 71 | 51 | 70 | 0 | 28 | 1,064,414 | 12 | 33 | 627,269 | 38.30 | 11 | 42 | 0 | 70 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 1,966,294 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 5,374,752 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 983,147 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 5,374,840 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 3,604,580 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 18,230,717 | 2,013,265,921 | 
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


Commit: https://github.com/openvm-org/openvm/commit/5883cb3d10e69037e0a9a407ac7d2ece710b89eb

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20012120450)
