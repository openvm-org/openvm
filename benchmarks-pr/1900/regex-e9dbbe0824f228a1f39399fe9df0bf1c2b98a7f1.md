| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.7%])</span> 7.57 | <span style='color: red'>(+0 [+1.5%])</span> 4.78 |
| regex_program | <span style='color: red'>(+0 [+0.7%])</span> 7.54 | <span style='color: red'>(+0 [+1.5%])</span> 4.75 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+25 [+0.7%])</span> 3,768.50 | <span style='color: red'>(+50 [+0.7%])</span> 7,537 | <span style='color: red'>(+71 [+1.5%])</span> 4,748 | <span style='color: green'>(-21 [-0.7%])</span> 2,789 |
| `main_cells_used     ` |  82,715,253 |  165,430,506 |  91,153,694 |  74,276,812 |
| `total_cells_used    ` |  194,047,811 |  388,095,622 |  211,864,680 |  176,230,942 |
| `insns               ` |  2,776,955.33 |  8,330,866 |  4,165,433 |  1,921,433 |
| `execute_metered_time_ms` |  33 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-1 [-0.5%])</span> 124.06 | -          | <span style='color: green'>(-1 [-0.5%])</span> 124.06 | <span style='color: green'>(-1 [-0.5%])</span> 124.06 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-2 [-2.0%])</span> 119.50 | <span style='color: green'>(-5 [-2.0%])</span> 239 | <span style='color: green'>(-3 [-2.2%])</span> 135 | <span style='color: green'>(-2 [-1.9%])</span> 104 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+0 [+2.2%])</span> 17.45 | -          | <span style='color: red'>(+0 [+2.2%])</span> 18.40 | <span style='color: red'>(+0 [+2.2%])</span> 16.50 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+13 [+3.2%])</span> 413.50 | <span style='color: red'>(+26 [+3.2%])</span> 827 | <span style='color: red'>(+13 [+2.5%])</span> 526 | <span style='color: red'>(+13 [+4.5%])</span> 301 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-13.3%])</span> 6.50 | <span style='color: green'>(-2 [-13.3%])</span> 13 |  11 | <span style='color: green'>(-2 [-50.0%])</span> 2 |
| `boundary_finalize_time_ms` |  2 |  4 |  4 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` |  156 |  312 | <span style='color: green'>(-2 [-0.8%])</span> 242 | <span style='color: red'>(+2 [+2.9%])</span> 70 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+14 [+0.5%])</span> 3,235.50 | <span style='color: red'>(+29 [+0.5%])</span> 6,471 | <span style='color: red'>(+61 [+1.5%])</span> 4,087 | <span style='color: green'>(-32 [-1.3%])</span> 2,384 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-12 [-1.8%])</span> 644 | <span style='color: green'>(-24 [-1.8%])</span> 1,288 | <span style='color: red'>(+2 [+0.2%])</span> 843 | <span style='color: green'>(-26 [-5.5%])</span> 445 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+4 [+1.6%])</span> 257.50 | <span style='color: red'>(+8 [+1.6%])</span> 515 | <span style='color: red'>(+10 [+3.3%])</span> 312 | <span style='color: green'>(-2 [-1.0%])</span> 203 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+0.7%])</span> 687 | <span style='color: red'>(+9 [+0.7%])</span> 1,374 | <span style='color: red'>(+4 [+0.5%])</span> 852 | <span style='color: red'>(+5 [+1.0%])</span> 522 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-0.4%])</span> 336.50 | <span style='color: green'>(-3 [-0.4%])</span> 673 | <span style='color: green'>(-2 [-0.5%])</span> 435 | <span style='color: green'>(-1 [-0.4%])</span> 238 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-3 [-1.0%])</span> 300 | <span style='color: green'>(-6 [-1.0%])</span> 600 | <span style='color: green'>(-3 [-0.8%])</span> 393 | <span style='color: green'>(-3 [-1.4%])</span> 207 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+22 [+2.3%])</span> 1,003.50 | <span style='color: red'>(+45 [+2.3%])</span> 2,007 | <span style='color: red'>(+49 [+4.1%])</span> 1,247 | <span style='color: green'>(-4 [-0.5%])</span> 760 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 501 | 21 | 10,219 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 3,993 | 24 | 4,165,433 | 1 | 33 | 124.06 | 62 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 237 | 4,450 | 
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
| regex_program | KeccakVmAir | 1 | 32 |  | 820 | 3,104 | 125,568 | 
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
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
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
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 526 | 4,748 | 211,864,680 | 373,166,268 | 4,087 | 435 | 393 | 852 | 1,247 | 242 | 25 | 11 | 843 | 91,153,694 | 2,244,000 | 312 | 135 | 16.50 | 4 | 
| regex_program | 1 | 301 | 2,789 | 176,230,942 | 242,965,930 | 2,384 | 238 | 207 | 522 | 760 | 70 | 24 | 2 | 445 | 74,276,812 | 1,921,433 | 203 | 104 | 18.40 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 7,965,446 | 2,013,265,921 | 
| regex_program | 0 | 1 | 22,978,816 | 2,013,265,921 | 
| regex_program | 0 | 2 | 3,982,723 | 2,013,265,921 | 
| regex_program | 0 | 3 | 28,093,700 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,800 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 71,675,021 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,406,852 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,181,504 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,703,426 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,192,164 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,508,864 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,194,986 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/e9dbbe0824f228a1f39399fe9df0bf1c2b98a7f1

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16513546536)
