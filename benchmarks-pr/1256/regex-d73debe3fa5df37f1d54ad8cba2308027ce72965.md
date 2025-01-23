| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+2.8%])</span> 18.47 | <span style='color: red'>(+1 [+2.8%])</span> 18.47 |
| regex_program | <span style='color: red'>(+1 [+2.8%])</span> 18.47 | <span style='color: red'>(+1 [+2.8%])</span> 18.47 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+503 [+2.8%])</span> 18,466 | <span style='color: red'>(+503 [+2.8%])</span> 18,466 | <span style='color: red'>(+503 [+2.8%])</span> 18,466 | <span style='color: red'>(+503 [+2.8%])</span> 18,466 |
| `main_cells_used     ` | <span style='color: green'>(-1177482 [-0.7%])</span> 163,833,427 | <span style='color: green'>(-1177482 [-0.7%])</span> 163,833,427 | <span style='color: green'>(-1177482 [-0.7%])</span> 163,833,427 | <span style='color: green'>(-1177482 [-0.7%])</span> 163,833,427 |
| `total_cycles        ` | <span style='color: green'>(-51068 [-1.2%])</span> 4,139,836 | <span style='color: green'>(-51068 [-1.2%])</span> 4,139,836 | <span style='color: green'>(-51068 [-1.2%])</span> 4,139,836 | <span style='color: green'>(-51068 [-1.2%])</span> 4,139,836 |
| `execute_time_ms     ` | <span style='color: red'>(+3 [+0.3%])</span> 1,013 | <span style='color: red'>(+3 [+0.3%])</span> 1,013 | <span style='color: red'>(+3 [+0.3%])</span> 1,013 | <span style='color: red'>(+3 [+0.3%])</span> 1,013 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-39 [-1.3%])</span> 3,054 | <span style='color: green'>(-39 [-1.3%])</span> 3,054 | <span style='color: green'>(-39 [-1.3%])</span> 3,054 | <span style='color: green'>(-39 [-1.3%])</span> 3,054 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+539 [+3.9%])</span> 14,399 | <span style='color: red'>(+539 [+3.9%])</span> 14,399 | <span style='color: red'>(+539 [+3.9%])</span> 14,399 | <span style='color: red'>(+539 [+3.9%])</span> 14,399 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-16 [-0.7%])</span> 2,379 | <span style='color: green'>(-16 [-0.7%])</span> 2,379 | <span style='color: green'>(-16 [-0.7%])</span> 2,379 | <span style='color: green'>(-16 [-0.7%])</span> 2,379 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+13 [+2.7%])</span> 500 | <span style='color: red'>(+13 [+2.7%])</span> 500 | <span style='color: red'>(+13 [+2.7%])</span> 500 | <span style='color: red'>(+13 [+2.7%])</span> 500 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-49 [-0.9%])</span> 5,185 | <span style='color: green'>(-49 [-0.9%])</span> 5,185 | <span style='color: green'>(-49 [-0.9%])</span> 5,185 | <span style='color: green'>(-49 [-0.9%])</span> 5,185 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+697 [+38.6%])</span> 2,505 | <span style='color: red'>(+697 [+38.6%])</span> 2,505 | <span style='color: red'>(+697 [+38.6%])</span> 2,505 | <span style='color: red'>(+697 [+38.6%])</span> 2,505 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-31 [-2.6%])</span> 1,171 | <span style='color: green'>(-31 [-2.6%])</span> 1,171 | <span style='color: green'>(-31 [-2.6%])</span> 1,171 | <span style='color: green'>(-31 [-2.6%])</span> 1,171 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-74 [-2.7%])</span> 2,657 | <span style='color: green'>(-74 [-2.7%])</span> 2,657 | <span style='color: green'>(-74 [-2.7%])</span> 2,657 | <span style='color: green'>(-74 [-2.7%])</span> 2,657 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 640 | 43 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,571 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 19 | 35 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| regex_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 0 | 64 |  | 24 | 11 | 2,240 | 
| regex_program | AccessAdapterAir<4> | 0 | 32 |  | 24 | 13 | 1,184 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 24 | 17 | 5,373,952 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 1,288 | 3,164 | 142,464 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PhantomAir | 0 | 512 |  | 12 | 6 | 9,216 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 80 | 32 | 1,835,008 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 80 | 36 | 243,269,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 48 | 26 | 38,797,312 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 56 | 32 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 76 | 35 | 113,664 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 72 | 40 | 234,881,024 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 104 | 57 | 20,608 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 100 | 39 | 35,584 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 80 | 31 | 7,274,496 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 28 | 21 | 3,211,264 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,054 | 18,466 | 4,139,836 | 633,271,680 | 14,399 | 2,505 | 1,171 | 5,185 | 2,657 | 2,379 | 163,833,427 | 500 | 1,013 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/d73debe3fa5df37f1d54ad8cba2308027ce72965

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12938365014)
