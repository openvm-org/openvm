| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+6 [+72.4%])</span> 13.81 | <span style='color: red'>(+3 [+69.1%])</span> 7.81 |
| regex_program | <span style='color: red'>(+6 [+72.4%])</span> 13.81 | <span style='color: red'>(+3 [+69.1%])</span> 7.81 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2900 [+72.4%])</span> 6,906.50 | <span style='color: red'>(+5800 [+72.4%])</span> 13,813 | <span style='color: red'>(+3190 [+69.1%])</span> 7,808 | <span style='color: red'>(+2610 [+76.9%])</span> 6,005 |
| `main_cells_used     ` | <span style='color: green'>(-967038 [-1.2%])</span> 82,727,686.50 | <span style='color: green'>(-1934077 [-1.2%])</span> 165,455,373 | <span style='color: green'>(-1013642 [-1.1%])</span> 92,686,348 | <span style='color: green'>(-920435 [-1.2%])</span> 72,769,025 |
| `total_cycles        ` |  2,069,920 |  4,139,840 |  2,225,737 |  1,914,103 |
| `execute_time_ms     ` | <span style='color: red'>(+52 [+15.3%])</span> 391 | <span style='color: red'>(+104 [+15.3%])</span> 782 | <span style='color: red'>(+94 [+25.1%])</span> 469 | <span style='color: red'>(+10 [+3.3%])</span> 313 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+222 [+22.5%])</span> 1,205.50 | <span style='color: red'>(+443 [+22.5%])</span> 2,411 | <span style='color: red'>(+96 [+8.1%])</span> 1,275 | <span style='color: red'>(+347 [+44.0%])</span> 1,136 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+2626 [+97.9%])</span> 5,310 | <span style='color: red'>(+5253 [+97.9%])</span> 10,620 | <span style='color: red'>(+3000 [+97.9%])</span> 6,064 | <span style='color: red'>(+2253 [+97.8%])</span> 4,556 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+630 [+128.2%])</span> 1,120.50 | <span style='color: red'>(+1259 [+128.2%])</span> 2,241 | <span style='color: red'>(+749 [+130.5%])</span> 1,323 | <span style='color: red'>(+510 [+125.0%])</span> 918 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+1.2%])</span> 173 | <span style='color: red'>(+4 [+1.2%])</span> 346 | <span style='color: green'>(-3 [-1.5%])</span> 200 | <span style='color: red'>(+7 [+5.0%])</span> 146 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+332 [+47.4%])</span> 1,031.50 | <span style='color: red'>(+663 [+47.4%])</span> 2,063 | <span style='color: red'>(+368 [+45.8%])</span> 1,172 | <span style='color: red'>(+295 [+49.5%])</span> 891 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+386 [+106.9%])</span> 748 | <span style='color: red'>(+773 [+106.9%])</span> 1,496 | <span style='color: red'>(+458 [+109.8%])</span> 875 | <span style='color: red'>(+315 [+102.9%])</span> 621 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+653 [+226.7%])</span> 941 | <span style='color: red'>(+1306 [+226.7%])</span> 1,882 | <span style='color: red'>(+758 [+227.6%])</span> 1,091 | <span style='color: red'>(+548 [+225.5%])</span> 791 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+626 [+94.6%])</span> 1,286.50 | <span style='color: red'>(+1251 [+94.6%])</span> 2,573 | <span style='color: red'>(+669 [+92.4%])</span> 1,393 | <span style='color: red'>(+582 [+97.3%])</span> 1,180 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 735 | 46 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 4 | 321 | 4,380 | 
| regex_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| regex_program | PhantomAir | 4 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| regex_program | VmConnectorAir | 4 | 3 | 8 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 12 | 17 | 59,392 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 532 | 3,163 | 3,695 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 532 | 3,163 | 118,240 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 12 | 32 | 180,224 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 8 | 20 | 3,670,016 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 8 | 20 | 57,344 | 
| regex_program | PhantomAir | 0 | 512 |  | 8 | 6 | 7,168 | 
| regex_program | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 24 | 32 | 917,504 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 28 | 36 | 33,554,432 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 16 | 26 | 5,505,024 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 20 | 28 | 6,291,456 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 20 | 28 | 3,145,728 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 28 | 35 | 64,512 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 28 | 35 | 126 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 28 | 40 | 71,303,168 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 28 | 40 | 71,303,168 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 40 | 57 | 12,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 40 | 39 | 20,224 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 16 | 21 | 1,212,416 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 16 | 21 | 1,212,416 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,275 | 7,808 | 2,225,737 | 209,921,543 | 6,064 | 875 | 1,091 | 1,172 | 1,393 | 1,323 | 92,686,348 | 200 | 469 | 
| regex_program | 1 | 1,136 | 6,005 | 1,914,103 | 149,454,692 | 4,556 | 621 | 791 | 891 | 1,180 | 918 | 72,769,025 | 146 | 313 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ede56ac7d0fead88374d56442f568eff3e5ba57e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13909079269)
