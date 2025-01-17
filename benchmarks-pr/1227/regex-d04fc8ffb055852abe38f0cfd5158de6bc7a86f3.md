| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.2%])</span> 18.95 | <span style='color: red'>(+0 [+0.2%])</span> 18.95 |
| regex_program | <span style='color: red'>(+0 [+0.2%])</span> 18.95 | <span style='color: red'>(+0 [+0.2%])</span> 18.95 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+31 [+0.2%])</span> 18,953 | <span style='color: red'>(+31 [+0.2%])</span> 18,953 | <span style='color: red'>(+31 [+0.2%])</span> 18,953 | <span style='color: red'>(+31 [+0.2%])</span> 18,953 |
| `main_cells_used     ` |  165,028,173 |  165,028,173 |  165,028,173 |  165,028,173 |
| `total_cycles        ` |  4,190,904 |  4,190,904 |  4,190,904 |  4,190,904 |
| `execute_time_ms     ` | <span style='color: green'>(-10 [-0.9%])</span> 1,121 | <span style='color: green'>(-10 [-0.9%])</span> 1,121 | <span style='color: green'>(-10 [-0.9%])</span> 1,121 | <span style='color: green'>(-10 [-0.9%])</span> 1,121 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-34 [-1.0%])</span> 3,320 | <span style='color: green'>(-34 [-1.0%])</span> 3,320 | <span style='color: green'>(-34 [-1.0%])</span> 3,320 | <span style='color: green'>(-34 [-1.0%])</span> 3,320 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+75 [+0.5%])</span> 14,512 | <span style='color: red'>(+75 [+0.5%])</span> 14,512 | <span style='color: red'>(+75 [+0.5%])</span> 14,512 | <span style='color: red'>(+75 [+0.5%])</span> 14,512 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.2%])</span> 2,383 | <span style='color: green'>(-4 [-0.2%])</span> 2,383 | <span style='color: green'>(-4 [-0.2%])</span> 2,383 | <span style='color: green'>(-4 [-0.2%])</span> 2,383 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-1.2%])</span> 485 | <span style='color: green'>(-6 [-1.2%])</span> 485 | <span style='color: green'>(-6 [-1.2%])</span> 485 | <span style='color: green'>(-6 [-1.2%])</span> 485 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-34 [-0.7%])</span> 5,095 | <span style='color: green'>(-34 [-0.7%])</span> 5,095 | <span style='color: green'>(-34 [-0.7%])</span> 5,095 | <span style='color: green'>(-34 [-0.7%])</span> 5,095 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+124 [+5.0%])</span> 2,618 | <span style='color: red'>(+124 [+5.0%])</span> 2,618 | <span style='color: red'>(+124 [+5.0%])</span> 2,618 | <span style='color: red'>(+124 [+5.0%])</span> 2,618 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-33 [-2.7%])</span> 1,173 | <span style='color: green'>(-33 [-2.7%])</span> 1,173 | <span style='color: green'>(-33 [-2.7%])</span> 1,173 | <span style='color: green'>(-33 [-2.7%])</span> 1,173 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+27 [+1.0%])</span> 2,755 | <span style='color: red'>(+27 [+1.0%])</span> 2,755 | <span style='color: red'>(+27 [+1.0%])</span> 2,755 | <span style='color: red'>(+27 [+1.0%])</span> 2,755 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 617 | 42 | 

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
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
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
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 80 | 36 | 243,269,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 48 | 26 | 38,797,312 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 56 | 32 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 16,384 |  | 36 | 26 | 1,015,808 | 
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
| regex_program | 0 | 3,320 | 18,953 | 4,190,904 | 632,452,480 | 14,512 | 2,618 | 1,173 | 5,095 | 2,755 | 2,383 | 165,028,173 | 485 | 1,121 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/d04fc8ffb055852abe38f0cfd5158de6bc7a86f3

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12822284862)
