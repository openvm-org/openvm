| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+5 [+75.3%])</span> 10.64 | <span style='color: red'>(+5 [+75.3%])</span> 10.64 |
| fibonacci_program | <span style='color: red'>(+5 [+75.3%])</span> 10.64 | <span style='color: red'>(+5 [+75.3%])</span> 10.64 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4571 [+75.3%])</span> 10,642 | <span style='color: red'>(+4571 [+75.3%])</span> 10,642 | <span style='color: red'>(+4571 [+75.3%])</span> 10,642 | <span style='color: red'>(+4571 [+75.3%])</span> 10,642 |
| `main_cells_used     ` |  51,486,676 |  51,486,676 |  51,486,676 |  51,486,676 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+4545 [+1510.0%])</span> 4,846 | <span style='color: red'>(+4545 [+1510.0%])</span> 4,846 | <span style='color: red'>(+4545 [+1510.0%])</span> 4,846 | <span style='color: red'>(+4545 [+1510.0%])</span> 4,846 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-45 [-5.2%])</span> 816 | <span style='color: green'>(-45 [-5.2%])</span> 816 | <span style='color: green'>(-45 [-5.2%])</span> 816 | <span style='color: green'>(-45 [-5.2%])</span> 816 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+71 [+1.4%])</span> 4,980 | <span style='color: red'>(+71 [+1.4%])</span> 4,980 | <span style='color: red'>(+71 [+1.4%])</span> 4,980 | <span style='color: red'>(+71 [+1.4%])</span> 4,980 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+8 [+1.0%])</span> 802 | <span style='color: red'>(+8 [+1.0%])</span> 802 | <span style='color: red'>(+8 [+1.0%])</span> 802 | <span style='color: red'>(+8 [+1.0%])</span> 802 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.2%])</span> 1,616 | <span style='color: green'>(-4 [-0.2%])</span> 1,616 | <span style='color: green'>(-4 [-0.2%])</span> 1,616 | <span style='color: green'>(-4 [-0.2%])</span> 1,616 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+22 [+2.9%])</span> 789 | <span style='color: red'>(+22 [+2.9%])</span> 789 | <span style='color: red'>(+22 [+2.9%])</span> 789 | <span style='color: red'>(+22 [+2.9%])</span> 789 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-20 [-3.8%])</span> 503 | <span style='color: green'>(-20 [-3.8%])</span> 503 | <span style='color: green'>(-20 [-3.8%])</span> 503 | <span style='color: green'>(-20 [-3.8%])</span> 503 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+66 [+6.4%])</span> 1,090 | <span style='color: red'>(+66 [+6.4%])</span> 1,090 | <span style='color: red'>(+66 [+6.4%])</span> 1,090 | <span style='color: red'>(+66 [+6.4%])</span> 1,090 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 373 | 5 | 

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

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 32,401,620 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 144 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 11,100,074 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 106 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 2,600,104 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,600,130 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 96 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 64 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,800,018 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 162 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 78 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 364 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 520 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 600 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 168 | 
| fibonacci_program | PhantomAir |  | PHANTOM | 0 | 12 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 20 | 32 | 13,312 | 
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

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 900,054 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 300,002 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 2 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 200,009 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 5 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 100,010 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 3 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 13 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 28 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 9 | 
| fibonacci_program | AccessAdapter<8> | 0 | 36 | 
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | 176 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fibonacci_program | Boundary | 0 | 36 | 
| fibonacci_program | Merkle | 0 | 228 | 
| fibonacci_program | PhantomAir | 0 | 2 | 
| fibonacci_program | ProgramChip | 0 | 3,275 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program |  | ADD | 0 | 900,045 | 
| fibonacci_program |  | AND | 0 | 2 | 
| fibonacci_program |  | AUIPC | 0 | 9 | 
| fibonacci_program |  | BEQ | 0 | 100,004 | 
| fibonacci_program |  | BGEU | 0 | 3 | 
| fibonacci_program |  | BLTU | 0 | 2 | 
| fibonacci_program |  | BNE | 0 | 100,005 | 
| fibonacci_program |  | HINT_STOREW | 0 | 3 | 
| fibonacci_program |  | JAL | 0 | 100,001 | 
| fibonacci_program |  | JALR | 0 | 13 | 
| fibonacci_program |  | LOADW | 0 | 13 | 
| fibonacci_program |  | LUI | 0 | 9 | 
| fibonacci_program |  | OR | 0 | 1 | 
| fibonacci_program |  | PHANTOM | 0 | 2 | 
| fibonacci_program |  | SLL | 0 | 2 | 
| fibonacci_program |  | SLTU | 0 | 300,002 | 
| fibonacci_program |  | STOREW | 0 | 15 | 
| fibonacci_program |  | SUB | 0 | 4 | 
| fibonacci_program |  | XOR | 0 | 2 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 816 | 10,642 | 1,500,137 | 197,440,542 | 4,980 | 789 | 503 | 1,616 | 1,090 | 802 | 51,486,676 | 177 | 4,846 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/fibonacci-abde525bf4e10ed9cf514c242f9d423e9e8eff91-fibonacci_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/abde525bf4e10ed9cf514c242f9d423e9e8eff91

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12933141073)
