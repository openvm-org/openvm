| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-3.4%])</span> 6.24 | <span style='color: green'>(-0 [-3.4%])</span> 6.24 |
| fibonacci_program | <span style='color: green'>(-0 [-3.4%])</span> 6.24 | <span style='color: green'>(-0 [-3.4%])</span> 6.24 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-219 [-3.4%])</span> 6,235 | <span style='color: green'>(-219 [-3.4%])</span> 6,235 | <span style='color: green'>(-219 [-3.4%])</span> 6,235 | <span style='color: green'>(-219 [-3.4%])</span> 6,235 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: green'>(-18 [-4.1%])</span> 420 | <span style='color: green'>(-18 [-4.1%])</span> 420 | <span style='color: green'>(-18 [-4.1%])</span> 420 | <span style='color: green'>(-18 [-4.1%])</span> 420 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-12 [-1.3%])</span> 902 | <span style='color: green'>(-12 [-1.3%])</span> 902 | <span style='color: green'>(-12 [-1.3%])</span> 902 | <span style='color: green'>(-12 [-1.3%])</span> 902 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-189 [-3.7%])</span> 4,913 | <span style='color: green'>(-189 [-3.7%])</span> 4,913 | <span style='color: green'>(-189 [-3.7%])</span> 4,913 | <span style='color: green'>(-189 [-3.7%])</span> 4,913 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-50 [-6.2%])</span> 760 | <span style='color: green'>(-50 [-6.2%])</span> 760 | <span style='color: green'>(-50 [-6.2%])</span> 760 | <span style='color: green'>(-50 [-6.2%])</span> 760 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.6%])</span> 175 | <span style='color: green'>(-1 [-0.6%])</span> 175 | <span style='color: green'>(-1 [-0.6%])</span> 175 | <span style='color: green'>(-1 [-0.6%])</span> 175 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-101 [-5.7%])</span> 1,657 | <span style='color: green'>(-101 [-5.7%])</span> 1,657 | <span style='color: green'>(-101 [-5.7%])</span> 1,657 | <span style='color: green'>(-101 [-5.7%])</span> 1,657 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-18 [-2.2%])</span> 790 | <span style='color: green'>(-18 [-2.2%])</span> 790 | <span style='color: green'>(-18 [-2.2%])</span> 790 | <span style='color: green'>(-18 [-2.2%])</span> 790 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-4 [-0.8%])</span> 495 | <span style='color: green'>(-4 [-0.8%])</span> 495 | <span style='color: green'>(-4 [-0.8%])</span> 495 | <span style='color: green'>(-4 [-0.8%])</span> 495 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-16 [-1.5%])</span> 1,033 | <span style='color: green'>(-16 [-1.5%])</span> 1,033 | <span style='color: green'>(-16 [-1.5%])</span> 1,033 | <span style='color: green'>(-16 [-1.5%])</span> 1,033 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 341 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 902 | 6,235 | 1,500,137 | 197,453,854 | 4,913 | 790 | 495 | 1,657 | 1,033 | 760 | 51,505,102 | 175 | 420 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/1551844ec5ef6307590eb7279b2b83eef3dddd9b

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12656634146)
