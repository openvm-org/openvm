| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.4%])</span> 1.02 | <span style='color: green'>(-0 [-0.3%])</span> 0.59 |
| fibonacci_program | <span style='color: red'>(+0 [+0.4%])</span> 1.01 | <span style='color: green'>(-0 [-0.3%])</span> 0.58 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2 [+0.4%])</span> 506 | <span style='color: red'>(+4 [+0.4%])</span> 1,012 | <span style='color: green'>(-2 [-0.3%])</span> 578 | <span style='color: red'>(+6 [+1.4%])</span> 434 |
| `main_cells_used     ` | <span style='color: red'>(+1174 [+0.1%])</span> 1,051,375 | <span style='color: red'>(+2348 [+0.1%])</span> 2,102,750 |  1,064,414 | <span style='color: red'>(+2350 [+0.2%])</span> 1,038,336 |
| `total_cells_used    ` |  9,694,601 |  19,389,202 |  9,708,456 |  9,680,746 |
| `execute_metered_time_ms` |  13 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+1 [+0.6%])</span> 114.06 | -          | <span style='color: red'>(+1 [+0.6%])</span> 114.06 | <span style='color: red'>(+1 [+0.6%])</span> 114.06 |
| `execute_preflight_insns` |  750,134.50 |  1,500,269 |  873,000 |  627,269 |
| `execute_preflight_time_ms` |  29 |  58 |  33 |  25 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+1 [+3.6%])</span> 37.86 | -          | <span style='color: red'>(+1 [+3.3%])</span> 39.54 | <span style='color: red'>(+1 [+4.0%])</span> 36.17 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+1 [+0.5%])</span> 188 | <span style='color: red'>(+2 [+0.5%])</span> 376 | <span style='color: red'>(+2 [+1.0%])</span> 193 |  183 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+2 [+1.0%])</span> 196.50 | <span style='color: red'>(+4 [+1.0%])</span> 393 | <span style='color: green'>(-2 [-0.9%])</span> 217 | <span style='color: red'>(+6 [+3.5%])</span> 176 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+0 [+1.7%])</span> 30.50 | <span style='color: red'>(+1 [+1.7%])</span> 61 | <span style='color: red'>(+1 [+3.1%])</span> 33 |  28 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-0 [-2.7%])</span> 18 | <span style='color: green'>(-1 [-2.7%])</span> 36 | <span style='color: green'>(-1 [-5.0%])</span> 19 |  17 |
| `perm_trace_commit_time_ms` |  40.95 |  81.90 |  43.74 |  38.16 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+0 [+1.0%])</span> 31.06 | <span style='color: red'>(+1 [+1.0%])</span> 62.12 | <span style='color: green'>(-2 [-5.4%])</span> 32.32 | <span style='color: red'>(+2 [+9.0%])</span> 29.80 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+5.3%])</span> 12.14 | <span style='color: red'>(+1 [+5.3%])</span> 24.29 | <span style='color: red'>(+1 [+5.6%])</span> 12.65 | <span style='color: red'>(+1 [+5.0%])</span> 11.64 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1 [+1.7%])</span> 61.50 | <span style='color: red'>(+2 [+1.7%])</span> 123 | <span style='color: green'>(-2 [-2.6%])</span> 76 | <span style='color: red'>(+4 [+9.3%])</span> 47 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 342 | 1,030 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 434 | 1 | 13 | 1,500,269 | 114.06 | 0 | 

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
| fibonacci_program | Rv32HintStoreAir | 2 | 19 | 28 | 
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
| fibonacci_program | 0 | 193 | 578 | 9,680,746 | 84,399,632 | 193 | 217 | 0 | 32.32 | 12.65 | 5 | 43.74 | 76 | 61 | 76 | 0 | 33 | 1,038,336 | 17 | 25 | 873,000 | 36.17 | 14 | 45 | 1 | 76 | 
| fibonacci_program | 1 | 183 | 434 | 9,708,456 | 74,305,770 | 183 | 176 | 1 | 29.80 | 11.64 | 4 | 38.16 | 47 | 58 | 47 | 0 | 28 | 1,064,414 | 19 | 33 | 627,269 | 39.54 | 11 | 41 | 0 | 47 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 1,966,294 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 5,374,752 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 983,147 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 5,374,840 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 3,604,584 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 18,230,721 | 2,013,265,921 | 
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


Commit: https://github.com/openvm-org/openvm/commit/98ec64bde8cd779fd524003151aa4d0a44af2ee1

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/19874764087)
