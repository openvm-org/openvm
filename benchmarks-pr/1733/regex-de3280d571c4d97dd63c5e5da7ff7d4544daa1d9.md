| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+18.0%])</span> 8.88 | <span style='color: red'>(+1 [+33.2%])</span> 5.62 |
| regex_program | <span style='color: red'>(+1 [+18.7%])</span> 8.56 | <span style='color: red'>(+1 [+35.7%])</span> 5.29 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+674 [+18.7%])</span> 4,279.50 | <span style='color: red'>(+1349 [+18.7%])</span> 8,559 | <span style='color: red'>(+1393 [+35.7%])</span> 5,292 | <span style='color: green'>(-44 [-1.3%])</span> 3,267 |
| `main_cells_used     ` |  83,224,793 |  166,449,586 |  93,439,220 |  73,010,366 |
| `total_cycles        ` |  2,082,716 |  4,165,432 |  2,243,700 |  1,921,732 |
| `execute_metered_time_ms` | <span style='color: red'>(+7 [+2.2%])</span> 324 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-2.1%])</span> 14.47 | -          | <span style='color: green'>(-0 [-2.1%])</span> 14.47 | <span style='color: green'>(-0 [-2.1%])</span> 14.47 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-62 [-27.0%])</span> 167.50 | <span style='color: green'>(-124 [-27.0%])</span> 335 | <span style='color: green'>(-68 [-27.3%])</span> 181 | <span style='color: green'>(-56 [-26.7%])</span> 154 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+3 [+37.0%])</span> 12.40 | -          | <span style='color: red'>(+3 [+36.6%])</span> 12.45 | <span style='color: red'>(+3 [+37.4%])</span> 12.35 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+568 [+76.1%])</span> 1,313.50 | <span style='color: red'>(+1135 [+76.1%])</span> 2,627 | <span style='color: red'>(+1126 [+145.5%])</span> 1,900 | <span style='color: red'>(+9 [+1.3%])</span> 727 |
| `memory_finalize_time_ms` | <span style='color: red'>(+578 [+339.3%])</span> 749 | <span style='color: red'>(+1157 [+339.3%])</span> 1,498 | <span style='color: red'>(+1133 [+425.9%])</span> 1,399 | <span style='color: red'>(+24 [+32.0%])</span> 99 |
| `boundary_finalize_time_ms` | <span style='color: red'>(+0 [+20.0%])</span> 3 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: red'>(+4 [+2.3%])</span> 156.50 | <span style='color: red'>(+7 [+2.3%])</span> 313 | <span style='color: red'>(+5 [+2.1%])</span> 241 | <span style='color: red'>(+2 [+2.9%])</span> 72 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+169 [+6.4%])</span> 2,798.50 | <span style='color: red'>(+338 [+6.4%])</span> 5,597 | <span style='color: red'>(+335 [+11.6%])</span> 3,211 | <span style='color: red'>(+3 [+0.1%])</span> 2,386 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+30 [+5.9%])</span> 541.50 | <span style='color: red'>(+60 [+5.9%])</span> 1,083 | <span style='color: red'>(+62 [+10.7%])</span> 640 | <span style='color: green'>(-2 [-0.4%])</span> 443 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+22 [+3.7%])</span> 595.50 | <span style='color: red'>(+43 [+3.7%])</span> 1,191 | <span style='color: red'>(+47 [+7.5%])</span> 674 | <span style='color: green'>(-4 [-0.8%])</span> 517 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+49 [+18.3%])</span> 317 | <span style='color: red'>(+98 [+18.3%])</span> 634 | <span style='color: red'>(+96 [+32.3%])</span> 393 | <span style='color: red'>(+2 [+0.8%])</span> 241 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+44 [+18.3%])</span> 287.50 | <span style='color: red'>(+89 [+18.3%])</span> 575 | <span style='color: red'>(+87 [+31.1%])</span> 367 | <span style='color: red'>(+2 [+1.0%])</span> 208 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+28 [+3.6%])</span> 825.50 | <span style='color: red'>(+57 [+3.6%])</span> 1,651 | <span style='color: red'>(+47 [+5.6%])</span> 879 | <span style='color: red'>(+10 [+1.3%])</span> 772 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 548 | 20 | 16,689 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 2 | 25 | 4,165,433 | 1 | 7,046 | 324 | 14.47 | 

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
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
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
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,900 | 5,292 | 2,243,700 | 275,648,700 | 3,211 | 393 | 367 | 3,436 | 674 | 879 | 241 | 26 | 1,399 | 640 | 93,439,220 | 2,243,700 | 247 | 181 | 12.35 | 6 | 
| regex_program | 1 | 727 | 3,267 | 1,921,732 | 242,975,388 | 2,386 | 241 | 208 | 2,981 | 517 | 772 | 72 | 25 | 99 | 443 | 73,010,366 | 1,921,733 | 196 | 154 | 12.45 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 5,868,294 | 2,013,265,921 | 
| regex_program | 0 | 1 | 16,687,360 | 2,013,265,921 | 
| regex_program | 0 | 2 | 2,934,147 | 2,013,265,921 | 
| regex_program | 0 | 3 | 19,705,092 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,800 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 53,849,229 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,406,854 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,182,848 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,703,427 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,193,508 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,508,864 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,197,677 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/de3280d571c4d97dd63c5e5da7ff7d4544daa1d9

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16011847940)
