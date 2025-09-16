| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+27.6%])</span> 1.22 | <span style='color: green'>(-0 [-28.8%])</span> 0.68 |
| fibonacci_program | <span style='color: red'>(+0 [+27.7%])</span> 1.21 | <span style='color: green'>(-0 [-29.2%])</span> 0.67 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-343 [-36.1%])</span> 606 | <span style='color: red'>(+263 [+27.7%])</span> 1,212 | <span style='color: green'>(-277 [-29.2%])</span> 672 | <span style='color: green'>(-409 [-43.1%])</span> 540 |
| `main_cells_used     ` | <span style='color: green'>(-6251 [-0.6%])</span> 1,053,981 | <span style='color: red'>(+1047730 [+98.8%])</span> 2,107,962 | <span style='color: red'>(+7300 [+0.7%])</span> 1,067,532 | <span style='color: green'>(-19802 [-1.9%])</span> 1,040,430 |
| `total_cells_used    ` |  9,699,511 | <span style='color: red'>(+9691388 [+99.8%])</span> 19,399,022 |  9,714,030 | <span style='color: green'>(-22642 [-0.2%])</span> 9,684,992 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+12.5%])</span> 9 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-20 [-11.9%])</span> 151.26 | -          | <span style='color: green'>(-20 [-11.9%])</span> 151.26 | <span style='color: green'>(-20 [-11.9%])</span> 151.26 |
| `execute_preflight_insns` | <span style='color: green'>(-750105 [-50.0%])</span> 750,105 |  1,500,210 | <span style='color: green'>(-627210 [-41.8%])</span> 873,000 | <span style='color: green'>(-873000 [-58.2%])</span> 627,210 |
| `execute_preflight_time_ms` | <span style='color: green'>(-13 [-31.0%])</span> 29 | <span style='color: red'>(+16 [+38.1%])</span> 58 | <span style='color: green'>(-9 [-21.4%])</span> 33 | <span style='color: green'>(-17 [-40.5%])</span> 25 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+1 [+1.7%])</span> 36.89 | -          | <span style='color: red'>(+2 [+6.9%])</span> 38.77 | <span style='color: green'>(-1 [-3.5%])</span> 35.01 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+6 [+2.7%])</span> 210.50 | <span style='color: red'>(+216 [+105.4%])</span> 421 | <span style='color: red'>(+10 [+4.9%])</span> 215 | <span style='color: red'>(+1 [+0.5%])</span> 206 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-284 [-51.1%])</span> 272 | <span style='color: green'>(-12 [-2.2%])</span> 544 | <span style='color: green'>(-272 [-48.9%])</span> 284 | <span style='color: green'>(-296 [-53.2%])</span> 260 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-41 [-55.4%])</span> 33 | <span style='color: green'>(-8 [-10.8%])</span> 66 | <span style='color: green'>(-39 [-52.7%])</span> 35 | <span style='color: green'>(-43 [-58.1%])</span> 31 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+10.3%])</span> 32 | <span style='color: red'>(+35 [+120.7%])</span> 64 | <span style='color: red'>(+7 [+24.1%])</span> 36 | <span style='color: green'>(-1 [-3.4%])</span> 28 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-56 [-55.8%])</span> 44.42 | <span style='color: green'>(-12 [-11.6%])</span> 88.84 | <span style='color: green'>(-53 [-52.6%])</span> 47.64 | <span style='color: green'>(-59 [-59.0%])</span> 41.20 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-34 [-53.6%])</span> 29.53 | <span style='color: green'>(-5 [-7.2%])</span> 59.05 | <span style='color: green'>(-31 [-48.9%])</span> 32.51 | <span style='color: green'>(-37 [-58.3%])</span> 26.54 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-10 [-40.7%])</span> 14.16 | <span style='color: red'>(+4 [+18.6%])</span> 28.31 | <span style='color: green'>(-9 [-38.1%])</span> 14.78 | <span style='color: green'>(-10 [-43.3%])</span> 13.53 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-144 [-55.2%])</span> 117.50 | <span style='color: green'>(-27 [-10.3%])</span> 235 | <span style='color: green'>(-137 [-52.3%])</span> 125 | <span style='color: green'>(-152 [-58.0%])</span> 110 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 75 | 325 | 1,395 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 540 | 67 | 1 | 9 | 1,500,210 | 151.26 | 168 | 

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
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
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
| fibonacci_program | 0 | 215 | 672 | 9,684,992 | 84,395,212 | 215 | 284 | 0 | 32.51 | 14.78 | 4 | 47.64 | 125 | 76 | 125 | 0 | 35 | 1,040,430 | 28 | 25 | 873,000 | 35.01 | 13 | 47 | 1 | 125 | 
| fibonacci_program | 1 | 206 | 540 | 9,714,030 | 74,305,770 | 206 | 260 | 1 | 26.54 | 13.53 | 4 | 41.20 | 110 | 77 | 110 | 0 | 31 | 1,067,532 | 36 | 33 | 627,210 | 38.77 | 10 | 40 | 1 | 110 | 

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


Commit: https://github.com/openvm-org/openvm/commit/7fe4ed27bb0715ad43597bd4d93a35ba3ee0fa81

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17774136682)
