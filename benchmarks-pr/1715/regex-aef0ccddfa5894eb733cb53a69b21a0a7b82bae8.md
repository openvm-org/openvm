| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-2 [-26.3%])</span> 5.77 | <span style='color: red'>(+1 [+33.4%])</span> 5.77 |
| regex_program | <span style='color: green'>(-2 [-26.3%])</span> 5.77 | <span style='color: red'>(+1 [+33.4%])</span> 5.77 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1856 [+47.5%])</span> 5,766 | <span style='color: green'>(-2053 [-26.3%])</span> 5,766 | <span style='color: red'>(+1443 [+33.4%])</span> 5,766 | <span style='color: red'>(+2270 [+64.9%])</span> 5,766 |
| `main_cells_used     ` | <span style='color: red'>(+82639120 [+99.3%])</span> 165,894,696 | <span style='color: green'>(-616456 [-0.4%])</span> 165,894,696 | <span style='color: red'>(+72393897 [+77.4%])</span> 165,894,696 | <span style='color: red'>(+92884343 [+127.2%])</span> 165,894,696 |
| `total_cycles        ` | <span style='color: red'>(+2082819 [+100.0%])</span> 4,165,432 |  4,165,432 | <span style='color: red'>(+1921717 [+85.6%])</span> 4,165,432 | <span style='color: red'>(+2243921 [+116.8%])</span> 4,165,432 |
| `execute_metered_time_ms` |  624 |  624 |  624 |  624 |
| `execute_time_ms     ` | <span style='color: red'>(+260 [+73.2%])</span> 615 | <span style='color: green'>(-95 [-13.4%])</span> 615 | <span style='color: red'>(+220 [+55.7%])</span> 615 | <span style='color: red'>(+300 [+95.2%])</span> 615 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-186 [-18.8%])</span> 801 | <span style='color: green'>(-1173 [-59.4%])</span> 801 | <span style='color: green'>(-366 [-31.4%])</span> 801 | <span style='color: green'>(-6 [-0.7%])</span> 801 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+1782 [+69.4%])</span> 4,350 | <span style='color: green'>(-785 [-15.3%])</span> 4,350 | <span style='color: red'>(+1589 [+57.6%])</span> 4,350 | <span style='color: red'>(+1976 [+83.2%])</span> 4,350 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+380 [+76.5%])</span> 877 | <span style='color: green'>(-117 [-11.8%])</span> 877 | <span style='color: red'>(+323 [+58.3%])</span> 877 | <span style='color: red'>(+437 [+99.3%])</span> 877 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+173 [+80.5%])</span> 388 | <span style='color: green'>(-42 [-9.8%])</span> 388 | <span style='color: red'>(+155 [+66.5%])</span> 388 | <span style='color: red'>(+191 [+97.0%])</span> 388 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+390 [+71.8%])</span> 934 | <span style='color: green'>(-153 [-14.1%])</span> 934 | <span style='color: red'>(+363 [+63.6%])</span> 934 | <span style='color: red'>(+418 [+81.0%])</span> 934 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+384 [+143.3%])</span> 652 | <span style='color: red'>(+116 [+21.6%])</span> 652 | <span style='color: red'>(+372 [+132.9%])</span> 652 | <span style='color: red'>(+396 [+154.7%])</span> 652 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+133 [+55.0%])</span> 375 | <span style='color: green'>(-109 [-22.5%])</span> 375 | <span style='color: red'>(+95 [+33.9%])</span> 375 | <span style='color: red'>(+171 [+83.8%])</span> 375 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+321 [+40.5%])</span> 1,114 | <span style='color: green'>(-472 [-29.8%])</span> 1,114 | <span style='color: red'>(+280 [+33.6%])</span> 1,114 | <span style='color: red'>(+362 [+48.1%])</span> 1,114 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | fri.log_blowup | execute_metered_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- | --- | --- |
| regex_program | 1 | 575 | 1 | 624 | 18 | 

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
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 28 | 26 | 28,311,552 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 32 | 32 | 16,777,216 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 52 | 41 | 195,035,136 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 52 | 31 | 5,439,488 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 28 | 20 | 3,145,728 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 801 | 5,766 | 4,165,432 | 511,713,308 | 4,350 | 652 | 375 | 934 | 1,114 | 877 | 165,894,696 | 388 | 615 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 11,438,918 | 2,013,265,921 | 
| regex_program | 0 | 1 | 32,222,272 | 2,013,265,921 | 
| regex_program | 0 | 2 | 5,719,459 | 2,013,265,921 | 
| regex_program | 0 | 3 | 37,992,516 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 13,161,280 | 2,013,265,921 | 
| regex_program | 0 | 7 | 265,216 | 2,013,265,921 | 
| regex_program | 0 | 8 | 102,651,053 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/aef0ccddfa5894eb733cb53a69b21a0a7b82bae8

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15537919575)
