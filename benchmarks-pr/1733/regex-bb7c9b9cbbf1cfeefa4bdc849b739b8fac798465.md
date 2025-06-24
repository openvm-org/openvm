| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.5%])</span> 7.43 | <span style='color: red'>(+0 [+3.7%])</span> 4.22 |
| regex_program | <span style='color: red'>(+0 [+3.7%])</span> 7.08 | <span style='color: red'>(+0 [+4.0%])</span> 3.87 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+125 [+3.7%])</span> 3,540 | <span style='color: red'>(+250 [+3.7%])</span> 7,080 | <span style='color: red'>(+149 [+4.0%])</span> 3,872 | <span style='color: red'>(+101 [+3.3%])</span> 3,208 |
| `main_cells_used     ` | <span style='color: green'>(-247350 [-0.3%])</span> 83,012,378 | <span style='color: green'>(-494700 [-0.3%])</span> 166,024,756 | <span style='color: green'>(-484738 [-0.5%])</span> 93,017,586 |  73,007,170 |
| `total_cycles        ` |  2,082,716 |  4,165,432 |  2,243,700 |  1,921,732 |
| `execute_metered_time_ms` |  347 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  11.100 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: red'>(+202 [+67.8%])</span> 498.50 | <span style='color: red'>(+403 [+67.8%])</span> 997 | <span style='color: red'>(+215 [+66.6%])</span> 538 | <span style='color: red'>(+188 [+69.4%])</span> 459 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-3 [-40.5%])</span> 4.17 | -          | <span style='color: green'>(-3 [-41.0%])</span> 4.18 | <span style='color: green'>(-3 [-40.0%])</span> 4.16 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+61 [+12.1%])</span> 565.50 | <span style='color: red'>(+122 [+12.1%])</span> 1,131 | <span style='color: red'>(+64 [+10.9%])</span> 651 | <span style='color: red'>(+58 [+13.7%])</span> 480 |
| `memory_finalize_time_ms` | <span style='color: red'>(+45 [+19.7%])</span> 274 | <span style='color: red'>(+90 [+19.7%])</span> 548 | <span style='color: red'>(+43 [+14.9%])</span> 332 | <span style='color: red'>(+47 [+27.8%])</span> 216 |
| `boundary_finalize_time_ms` | <span style='color: red'>(+1 [+40.0%])</span> 3.50 | <span style='color: red'>(+2 [+40.0%])</span> 7 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+inf%])</span> 1 |
| `merkle_finalize_time_ms` | <span style='color: red'>(+8 [+3.5%])</span> 220.50 | <span style='color: red'>(+15 [+3.5%])</span> 441 | <span style='color: red'>(+8 [+3.0%])</span> 272 | <span style='color: red'>(+7 [+4.3%])</span> 169 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-138 [-5.3%])</span> 2,476 | <span style='color: green'>(-275 [-5.3%])</span> 4,952 | <span style='color: green'>(-130 [-4.6%])</span> 2,683 | <span style='color: green'>(-145 [-6.0%])</span> 2,269 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-52 [-10.2%])</span> 456 | <span style='color: green'>(-104 [-10.2%])</span> 912 | <span style='color: green'>(-56 [-9.8%])</span> 515 | <span style='color: green'>(-48 [-10.8%])</span> 397 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+4.2%])</span> 221.50 | <span style='color: red'>(+18 [+4.2%])</span> 443 | <span style='color: red'>(+17 [+7.5%])</span> 243 | <span style='color: red'>(+1 [+0.5%])</span> 200 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-65 [-11.6%])</span> 497.50 | <span style='color: green'>(-130 [-11.6%])</span> 995 | <span style='color: green'>(-75 [-12.3%])</span> 533 | <span style='color: green'>(-55 [-10.6%])</span> 462 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+10 [+3.8%])</span> 270 | <span style='color: red'>(+20 [+3.8%])</span> 540 | <span style='color: red'>(+12 [+4.2%])</span> 298 | <span style='color: red'>(+8 [+3.4%])</span> 242 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-36 [-13.8%])</span> 224.50 | <span style='color: green'>(-72 [-13.8%])</span> 449 | <span style='color: green'>(-22 [-7.9%])</span> 255 | <span style='color: green'>(-50 [-20.5%])</span> 194 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-3 [-0.4%])</span> 799 | <span style='color: green'>(-6 [-0.4%])</span> 1,598 | <span style='color: green'>(-5 [-0.6%])</span> 835 | <span style='color: green'>(-1 [-0.1%])</span> 763 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | commit_exe_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 2 | 591 | 4,165,433 | 1 | 347 | 11.100 | 18 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 651 | 3,872 | 2,243,700 | 275,648,700 | 2,683 | 298 | 255 | 533 | 835 | 272 | 332 | 515 | 93,017,586 | 2,243,700 | 243 | 538 | 4.16 | 6 | 
| regex_program | 1 | 480 | 3,208 | 1,921,732 | 242,975,388 | 2,269 | 242 | 194 | 462 | 763 | 169 | 216 | 397 | 73,007,170 | 1,921,733 | 200 | 459 | 4.18 | 1 | 

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


Commit: https://github.com/openvm-org/openvm/commit/bb7c9b9cbbf1cfeefa4bdc849b739b8fac798465

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15853266552)
