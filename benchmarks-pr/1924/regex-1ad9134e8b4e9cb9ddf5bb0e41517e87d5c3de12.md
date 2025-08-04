| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  6.95 | <span style='color: red'>(+0 [+0.2%])</span> 4.51 |
| regex_program |  6.92 | <span style='color: red'>(+0 [+0.2%])</span> 4.48 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,457.50 |  6,915 | <span style='color: red'>(+8 [+0.2%])</span> 4,477 | <span style='color: green'>(-5 [-0.2%])</span> 2,438 |
| `main_cells_used     ` |  81,834,937 |  163,669,874 |  90,228,333 |  73,441,541 |
| `total_cells_used    ` |  192,008,621 |  384,017,242 |  209,713,811 |  174,303,431 |
| `insns               ` |  2,739,064.67 |  8,217,194 |  4,108,597 |  1,897,597 |
| `execute_metered_time_ms` | <span style='color: green'>(-1 [-3.0%])</span> 32 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+1 [+1.0%])</span> 125.47 | -          | <span style='color: red'>(+1 [+1.0%])</span> 125.47 | <span style='color: red'>(+1 [+1.0%])</span> 125.47 |
| `execute_e3_time_ms  ` |  118 |  236 | <span style='color: red'>(+1 [+0.8%])</span> 134 | <span style='color: green'>(-1 [-1.0%])</span> 102 |
| `execute_e3_insn_mi/s` |  17.45 | -          | <span style='color: red'>(+0 [+0.2%])</span> 18.44 | <span style='color: green'>(-0 [-0.2%])</span> 16.47 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.5%])</span> 389.50 | <span style='color: red'>(+4 [+0.5%])</span> 779 | <span style='color: red'>(+1 [+0.2%])</span> 503 | <span style='color: red'>(+3 [+1.1%])</span> 276 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-13.3%])</span> 6.50 | <span style='color: green'>(-2 [-13.3%])</span> 13 | <span style='color: green'>(-2 [-15.4%])</span> 11 |  2 |
| `boundary_finalize_time_ms` |  2 |  4 |  4 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-1 [-0.7%])</span> 139 | <span style='color: green'>(-2 [-0.7%])</span> 278 | <span style='color: green'>(-2 [-0.9%])</span> 226 |  52 |
| `stark_prove_excluding_trace_time_ms` |  2,950 |  5,900 | <span style='color: red'>(+6 [+0.2%])</span> 3,840 | <span style='color: green'>(-7 [-0.3%])</span> 2,060 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-7 [-1.2%])</span> 567 | <span style='color: green'>(-14 [-1.2%])</span> 1,134 | <span style='color: green'>(-1 [-0.1%])</span> 765 | <span style='color: green'>(-13 [-3.4%])</span> 369 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+2.7%])</span> 230 | <span style='color: red'>(+12 [+2.7%])</span> 460 | <span style='color: red'>(+7 [+2.3%])</span> 306 | <span style='color: red'>(+5 [+3.4%])</span> 154 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-10 [-1.6%])</span> 580.50 | <span style='color: green'>(-19 [-1.6%])</span> 1,161 | <span style='color: green'>(-14 [-1.8%])</span> 759 | <span style='color: green'>(-5 [-1.2%])</span> 402 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.3%])</span> 319.50 | <span style='color: red'>(+2 [+0.3%])</span> 639 | <span style='color: green'>(-2 [-0.5%])</span> 424 | <span style='color: red'>(+4 [+1.9%])</span> 215 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.7%])</span> 285 | <span style='color: red'>(+4 [+0.7%])</span> 570 | <span style='color: red'>(+4 [+1.1%])</span> 377 |  193 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+5 [+0.5%])</span> 959 | <span style='color: red'>(+10 [+0.5%])</span> 1,918 | <span style='color: red'>(+9 [+0.8%])</span> 1,201 | <span style='color: red'>(+1 [+0.1%])</span> 717 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 506 | 19 | 7,511 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 2,675 | 6 | 4,108,597 | 1 | 32 | 125.47 | 45 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,513 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| regex_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 52 | 41 | 195,035,136 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 256 |  | 72 | 59 | 33,536 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 503 | 4,477 | 209,713,811 | 373,183,036 | 502 | 3,840 | 2 | 424 | 377 | 759 | 1,201 | 226 | 7 | 11 | 765 | 90,228,333 | 2,211,000 | 306 | 134 | 16.47 | 4 | 
| regex_program | 1 | 276 | 2,438 | 174,303,431 | 196,838,026 | 276 | 2,060 | 2 | 215 | 193 | 402 | 717 | 52 | 6 | 2 | 369 | 73,441,541 | 1,897,597 | 154 | 102 | 18.44 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 7,965,702 | 2,013,265,921 | 
| regex_program | 0 | 1 | 22,979,584 | 2,013,265,921 | 
| regex_program | 0 | 2 | 3,982,851 | 2,013,265,921 | 
| regex_program | 0 | 3 | 28,094,468 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,669,056 | 2,013,265,921 | 
| regex_program | 0 | 7 | 135,168 | 2,013,265,921 | 
| regex_program | 0 | 8 | 71,678,221 | 2,013,265,921 | 
| regex_program | 1 | 0 | 4,358,276 | 2,013,265,921 | 
| regex_program | 1 | 1 | 12,037,120 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,179,138 | 2,013,265,921 | 
| regex_program | 1 | 3 | 15,047,780 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,887,424 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 38,711,914 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/1ad9134e8b4e9cb9ddf5bb0e41517e87d5c3de12

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16728081517)
