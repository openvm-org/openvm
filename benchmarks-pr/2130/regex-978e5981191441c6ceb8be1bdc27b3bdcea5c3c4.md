| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+2.1%])</span> 3.06 | <span style='color: red'>(+0 [+1.7%])</span> 1.78 |
| regex_program | <span style='color: red'>(+0 [+2.1%])</span> 3.02 | <span style='color: red'>(+0 [+1.7%])</span> 1.75 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+31 [+2.1%])</span> 1,511 | <span style='color: red'>(+62 [+2.1%])</span> 3,022 | <span style='color: red'>(+29 [+1.7%])</span> 1,748 | <span style='color: red'>(+33 [+2.7%])</span> 1,274 |
| `main_cells_used     ` |  6,700,302 |  13,400,604 |  10,857,266 | <span style='color: red'>(+6168 [+0.2%])</span> 2,543,338 |
| `total_cells_used    ` |  17,689,684 |  35,379,368 |  23,389,324 |  11,990,044 |
| `execute_metered_time_ms` |  35 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+1 [+0.6%])</span> 115.83 | -          | <span style='color: red'>(+1 [+0.6%])</span> 115.83 | <span style='color: red'>(+1 [+0.6%])</span> 115.83 |
| `execute_preflight_insns` |  2,054,279.50 |  4,108,559 |  2,211,000 |  1,897,559 |
| `execute_preflight_time_ms` | <span style='color: red'>(+2 [+1.5%])</span> 100.50 | <span style='color: red'>(+3 [+1.5%])</span> 201 | <span style='color: red'>(+6 [+5.4%])</span> 117 | <span style='color: green'>(-3 [-3.4%])</span> 84 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+0.6%])</span> 29.55 | -          | <span style='color: green'>(-1 [-3.6%])</span> 30.03 | <span style='color: red'>(+1 [+5.4%])</span> 29.08 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+4 [+1.4%])</span> 315.50 | <span style='color: red'>(+9 [+1.4%])</span> 631 |  334 | <span style='color: red'>(+9 [+3.1%])</span> 297 |
| `memory_finalize_time_ms` |  2 |  4 |  4 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+23 [+2.3%])</span> 1,006.50 | <span style='color: red'>(+46 [+2.3%])</span> 2,013 | <span style='color: red'>(+20 [+1.6%])</span> 1,238 | <span style='color: red'>(+26 [+3.5%])</span> 775 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+2 [+1.8%])</span> 144.50 | <span style='color: red'>(+5 [+1.8%])</span> 289 | <span style='color: red'>(+5 [+2.7%])</span> 189 |  100 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-2.5%])</span> 58 | <span style='color: green'>(-3 [-2.5%])</span> 116 | <span style='color: red'>(+2 [+3.1%])</span> 67 | <span style='color: green'>(-5 [-9.3%])</span> 49 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+2 [+1.3%])</span> 176.63 | <span style='color: red'>(+5 [+1.3%])</span> 353.26 | <span style='color: red'>(+3 [+1.2%])</span> 229.65 | <span style='color: red'>(+2 [+1.5%])</span> 123.61 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+7 [+4.2%])</span> 170.40 | <span style='color: red'>(+14 [+4.2%])</span> 340.80 | <span style='color: red'>(+11 [+6.0%])</span> 188.18 | <span style='color: red'>(+3 [+2.0%])</span> 152.61 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+3.1%])</span> 37.68 | <span style='color: red'>(+2 [+3.1%])</span> 75.37 | <span style='color: red'>(+1 [+2.2%])</span> 50.12 | <span style='color: red'>(+1 [+5.0%])</span> 25.25 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+12 [+2.9%])</span> 412 | <span style='color: red'>(+23 [+2.9%])</span> 824 | <span style='color: red'>(+6 [+1.1%])</span> 545 | <span style='color: red'>(+17 [+6.5%])</span> 279 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 57 | 608 | 3,251 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 1,274 | 41 | 1 | 35 | 4,108,559 | 115.83 | 168 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 297 | 1,748 | 23,389,324 | 373,183,036 | 297 | 1,238 | 0 | 152.61 | 50.12 | 6 | 229.65 | 545 | 297 | 545 | 4 | 189 | 10,857,266 | 67 | 84 | 2,211,000 | 29.08 | 46 | 204 | 2 | 544 | 
| regex_program | 1 | 334 | 1,274 | 11,990,044 | 196,838,026 | 334 | 775 | 2 | 188.18 | 25.25 | 6 | 123.61 | 279 | 176 | 279 | 0 | 100 | 2,543,338 | 49 | 117 | 1,897,559 | 30.03 | 26 | 218 | 2 | 279 | 

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


Commit: https://github.com/openvm-org/openvm/commit/978e5981191441c6ceb8be1bdc27b3bdcea5c3c4

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17648664437)
