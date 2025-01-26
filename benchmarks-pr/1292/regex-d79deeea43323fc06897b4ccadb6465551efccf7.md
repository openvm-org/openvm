| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.5%])</span> 14.56 | <span style='color: green'>(-0 [-0.5%])</span> 14.56 |
| regex_program | <span style='color: green'>(-0 [-0.5%])</span> 14.56 | <span style='color: green'>(-0 [-0.5%])</span> 14.56 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-66 [-0.5%])</span> 14,562 | <span style='color: green'>(-66 [-0.5%])</span> 14,562 | <span style='color: green'>(-66 [-0.5%])</span> 14,562 | <span style='color: green'>(-66 [-0.5%])</span> 14,562 |
| `main_cells_used     ` |  165,010,909 |  165,010,909 |  165,010,909 |  165,010,909 |
| `total_cycles        ` |  4,190,904 |  4,190,904 |  4,190,904 |  4,190,904 |
| `execute_time_ms     ` | <span style='color: green'>(-5 [-0.5%])</span> 997 | <span style='color: green'>(-5 [-0.5%])</span> 997 | <span style='color: green'>(-5 [-0.5%])</span> 997 | <span style='color: green'>(-5 [-0.5%])</span> 997 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+11 [+0.4%])</span> 2,885 | <span style='color: red'>(+11 [+0.4%])</span> 2,885 | <span style='color: red'>(+11 [+0.4%])</span> 2,885 | <span style='color: red'>(+11 [+0.4%])</span> 2,885 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-72 [-0.7%])</span> 10,680 | <span style='color: green'>(-72 [-0.7%])</span> 10,680 | <span style='color: green'>(-72 [-0.7%])</span> 10,680 | <span style='color: green'>(-72 [-0.7%])</span> 10,680 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-23 [-1.0%])</span> 2,387 | <span style='color: green'>(-23 [-1.0%])</span> 2,387 | <span style='color: green'>(-23 [-1.0%])</span> 2,387 | <span style='color: green'>(-23 [-1.0%])</span> 2,387 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+0.5%])</span> 390 | <span style='color: red'>(+2 [+0.5%])</span> 390 | <span style='color: red'>(+2 [+0.5%])</span> 390 | <span style='color: red'>(+2 [+0.5%])</span> 390 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-66 [-3.1%])</span> 2,095 | <span style='color: green'>(-66 [-3.1%])</span> 2,095 | <span style='color: green'>(-66 [-3.1%])</span> 2,095 | <span style='color: green'>(-66 [-3.1%])</span> 2,095 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-0.1%])</span> 1,528 | <span style='color: green'>(-2 [-0.1%])</span> 1,528 | <span style='color: green'>(-2 [-0.1%])</span> 1,528 | <span style='color: green'>(-2 [-0.1%])</span> 1,528 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+18 [+1.0%])</span> 1,885 | <span style='color: red'>(+18 [+1.0%])</span> 1,885 | <span style='color: red'>(+18 [+1.0%])</span> 1,885 | <span style='color: red'>(+18 [+1.0%])</span> 1,885 |
| `pcs_opening_time_ms ` |  2,385 |  2,385 |  2,385 |  2,385 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 746 | 42 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 4 | 321 | 4,382 | 
| regex_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| regex_program | PhantomAir | 4 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 4 | 15 | 13 | 
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
| regex_program | AccessAdapterAir<2> | 0 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 0 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 532 | 3,164 | 118,272 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 8 | 20 | 3,670,016 | 
| regex_program | PhantomAir | 0 | 512 |  | 8 | 6 | 7,168 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 28 | 36 | 134,217,728 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 24 | 37 | 3,997,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 28 | 53 | 21,233,664 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 16 | 26 | 22,020,096 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 20 | 32 | 13,631,488 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 16,384 |  | 20 | 26 | 753,664 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 20 | 28 | 6,291,456 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 28 | 35 | 64,512 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 28 | 40 | 142,606,336 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 40 | 57 | 12,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 40 | 39 | 20,224 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 28 | 31 | 3,866,624 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 16 | 21 | 2,424,832 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 2,885 | 14,562 | 4,190,904 | 384,102,008 | 10,680 | 1,528 | 1,885 | 2,095 | 2,385 | 2,387 | 165,010,909 | 390 | 997 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/d79deeea43323fc06897b4ccadb6465551efccf7

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12972660549)