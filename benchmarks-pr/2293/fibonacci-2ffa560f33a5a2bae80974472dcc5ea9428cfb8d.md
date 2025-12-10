| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+4.1%])</span> 1.06 | <span style='color: red'>(+0 [+2.7%])</span> 0.61 |
| fibonacci_program | <span style='color: red'>(+0 [+3.9%])</span> 1.05 | <span style='color: red'>(+0 [+2.2%])</span> 0.59 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+20 [+3.9%])</span> 525.50 | <span style='color: red'>(+39 [+3.9%])</span> 1,051 | <span style='color: red'>(+13 [+2.2%])</span> 593 | <span style='color: red'>(+26 [+6.0%])</span> 458 |
| `main_cells_used     ` |  1,050,201 |  2,100,402 |  1,064,416 |  1,035,986 |
| `total_cells_used    ` |  9,692,939 |  19,385,878 |  9,708,170 |  9,677,708 |
| `execute_metered_time_ms` | <span style='color: red'>(+3 [+30.0%])</span> 13 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-36 [-24.1%])</span> 112.79 | -          | <span style='color: green'>(-36 [-24.1%])</span> 112.79 | <span style='color: green'>(-36 [-24.1%])</span> 112.79 |
| `execute_preflight_insns` |  750,104.50 |  1,500,209 |  873,000 |  627,209 |
| `execute_preflight_time_ms` |  29 |  58 |  33 |  25 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+1 [+2.9%])</span> 37.86 | -          | <span style='color: red'>(+1 [+3.4%])</span> 40.03 | <span style='color: red'>(+1 [+2.3%])</span> 35.68 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+4 [+2.4%])</span> 192 | <span style='color: red'>(+9 [+2.4%])</span> 384 | <span style='color: red'>(+6 [+3.1%])</span> 199 | <span style='color: red'>(+3 [+1.6%])</span> 185 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+16 [+8.7%])</span> 200 | <span style='color: red'>(+32 [+8.7%])</span> 400 | <span style='color: red'>(+9 [+4.7%])</span> 201 | <span style='color: red'>(+23 [+13.1%])</span> 199 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+0 [+1.7%])</span> 30.50 | <span style='color: red'>(+1 [+1.7%])</span> 61 | <span style='color: red'>(+1 [+3.1%])</span> 33 |  28 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-22.0%])</span> 19.50 | <span style='color: green'>(-11 [-22.0%])</span> 39 | <span style='color: green'>(-8 [-28.6%])</span> 20 | <span style='color: green'>(-3 [-13.6%])</span> 19 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+1 [+3.1%])</span> 41.100 | <span style='color: red'>(+3 [+3.1%])</span> 83.100 | <span style='color: red'>(+1 [+3.2%])</span> 45.10 | <span style='color: red'>(+1 [+3.0%])</span> 38.90 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+2.0%])</span> 32.39 | <span style='color: red'>(+1 [+2.0%])</span> 64.79 | <span style='color: red'>(+1 [+3.2%])</span> 34.21 | <span style='color: red'>(+0 [+0.6%])</span> 30.58 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+6.5%])</span> 13.08 | <span style='color: red'>(+2 [+6.5%])</span> 26.15 | <span style='color: red'>(+1 [+7.8%])</span> 13.50 | <span style='color: red'>(+1 [+5.2%])</span> 12.65 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+18 [+41.2%])</span> 60 | <span style='color: red'>(+35 [+41.2%])</span> 120 | <span style='color: red'>(+24 [+54.5%])</span> 68 | <span style='color: red'>(+11 [+26.8%])</span> 52 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 311 | 1,070 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 458 | 1 | 13 | 1,500,209 | 112.79 | 0 | 

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
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 1 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 64 |  | 52 | 41 | 5,952 | 
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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 199 | 593 | 9,677,708 | 84,395,212 | 199 | 201 | 0 | 34.21 | 13.50 | 4 | 45.10 | 52 | 66 | 52 | 0 | 33 | 1,035,986 | 20 | 25 | 873,000 | 35.68 | 14 | 48 | 0 | 52 | 
| fibonacci_program | 1 | 185 | 458 | 9,708,170 | 74,305,770 | 185 | 199 | 1 | 30.58 | 12.65 | 4 | 38.90 | 68 | 58 | 68 | 0 | 28 | 1,064,416 | 19 | 33 | 627,209 | 40.03 | 11 | 43 | 0 | 68 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 1,966,190 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 5,374,472 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 983,095 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 5,374,428 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 3,604,544 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 18,229,833 | 2,013,265,921 | 
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


Commit: https://github.com/openvm-org/openvm/commit/2ffa560f33a5a2bae80974472dcc5ea9428cfb8d

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20114346204)
