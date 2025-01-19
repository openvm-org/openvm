| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+59 [+314.1%])</span> 78.32 | <span style='color: red'>(+59 [+314.1%])</span> 78.32 |
| regex_program | <span style='color: red'>(+59 [+314.1%])</span> 78.32 | <span style='color: red'>(+59 [+314.1%])</span> 78.32 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+59408 [+314.1%])</span> 78,323 | <span style='color: red'>(+59408 [+314.1%])</span> 78,323 | <span style='color: red'>(+59408 [+314.1%])</span> 78,323 | <span style='color: red'>(+59408 [+314.1%])</span> 78,323 |
| `main_cells_used     ` | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 |
| `total_cycles        ` | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 |
| `execute_time_ms     ` | <span style='color: red'>(+59249 [+5220.2%])</span> 60,384 | <span style='color: red'>(+59249 [+5220.2%])</span> 60,384 | <span style='color: red'>(+59249 [+5220.2%])</span> 60,384 | <span style='color: red'>(+59249 [+5220.2%])</span> 60,384 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+120 [+3.6%])</span> 3,428 | <span style='color: red'>(+120 [+3.6%])</span> 3,428 | <span style='color: red'>(+120 [+3.6%])</span> 3,428 | <span style='color: red'>(+120 [+3.6%])</span> 3,428 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+39 [+0.3%])</span> 14,511 | <span style='color: red'>(+39 [+0.3%])</span> 14,511 | <span style='color: red'>(+39 [+0.3%])</span> 14,511 | <span style='color: red'>(+39 [+0.3%])</span> 14,511 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+5 [+0.2%])</span> 2,390 | <span style='color: red'>(+5 [+0.2%])</span> 2,390 | <span style='color: red'>(+5 [+0.2%])</span> 2,390 | <span style='color: red'>(+5 [+0.2%])</span> 2,390 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+5 [+1.0%])</span> 498 | <span style='color: red'>(+5 [+1.0%])</span> 498 | <span style='color: red'>(+5 [+1.0%])</span> 498 | <span style='color: red'>(+5 [+1.0%])</span> 498 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+11 [+0.2%])</span> 5,140 | <span style='color: red'>(+11 [+0.2%])</span> 5,140 | <span style='color: red'>(+11 [+0.2%])</span> 5,140 | <span style='color: red'>(+11 [+0.2%])</span> 5,140 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-36 [-1.4%])</span> 2,550 | <span style='color: green'>(-36 [-1.4%])</span> 2,550 | <span style='color: green'>(-36 [-1.4%])</span> 2,550 | <span style='color: green'>(-36 [-1.4%])</span> 2,550 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-2 [-0.2%])</span> 1,205 | <span style='color: green'>(-2 [-0.2%])</span> 1,205 | <span style='color: green'>(-2 [-0.2%])</span> 1,205 | <span style='color: green'>(-2 [-0.2%])</span> 1,205 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+58 [+2.2%])</span> 2,726 | <span style='color: red'>(+58 [+2.2%])</span> 2,726 | <span style='color: red'>(+58 [+2.2%])</span> 2,726 | <span style='color: red'>(+58 [+2.2%])</span> 2,726 |



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

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 36,618,768 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 1,912,104 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 847,584 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 1,532,952 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 344,232 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLT | 0 | 185 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 1,237,798 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 11,318,044 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRA | 0 | 53 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 269,770 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 4,880,538 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,691,832 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGE | 0 | 9,408 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 3,890,944 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 164,512 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 2,273,600 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,190,322 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 800,964 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 331,942 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 3,652,404 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 24,255 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADH | 0 | 280 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 1,093,200 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADHU | 0 | 3,800 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 45,715,640 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 509,480 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREH | 0 | 402,960 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 30,916,880 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 6,498 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 9,516 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 1,614,697 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 830,676 | 
| regex_program | KeccakVmAir |  | KECCAK256 | 0 | 75,936 | 
| regex_program | PhantomAir |  | PHANTOM | 0 | 1,734 | 

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

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,145,990 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 33,459 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 218,639 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 291,245 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 198,077 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 110,627 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 12,767 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 130,443 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 701 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 1,966,049 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 114 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 244 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 52,087 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 39,557 | 
| regex_program | AccessAdapter<2> | 0 | 42 | 
| regex_program | AccessAdapter<4> | 0 | 22 | 
| regex_program | AccessAdapter<8> | 0 | 69,206 | 
| regex_program | Arc<BabyBearParameters>, 1> | 0 | 14,005 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| regex_program | Boundary | 0 | 69,206 | 
| regex_program | KeccakVmAir | 0 | 24 | 
| regex_program | Merkle | 0 | 70,444 | 
| regex_program | PhantomAir | 0 | 289 | 
| regex_program | ProgramChip | 0 | 89,891 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 
| regex_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| regex_program |  | ADD | 0 | 1,017,188 | 
| regex_program |  | AND | 0 | 53,114 | 
| regex_program |  | AUIPC | 0 | 39,557 | 
| regex_program |  | BEQ | 0 | 187,713 | 
| regex_program |  | BGE | 0 | 294 | 
| regex_program |  | BGEU | 0 | 121,592 | 
| regex_program |  | BLT | 0 | 5,141 | 
| regex_program |  | BLTU | 0 | 71,050 | 
| regex_program |  | BNE | 0 | 103,532 | 
| regex_program |  | DIVU | 0 | 114 | 
| regex_program |  | HINT_STOREW | 0 | 12,767 | 
| regex_program |  | JAL | 0 | 66,129 | 
| regex_program |  | JALR | 0 | 130,443 | 
| regex_program |  | KECCAK256 | 0 | 1 | 
| regex_program |  | LOADB | 0 | 693 | 
| regex_program |  | LOADBU | 0 | 27,330 | 
| regex_program |  | LOADH | 0 | 8 | 
| regex_program |  | LOADHU | 0 | 95 | 
| regex_program |  | LOADW | 0 | 1,142,891 | 
| regex_program |  | LUI | 0 | 44,498 | 
| regex_program |  | MUL | 0 | 52,087 | 
| regex_program |  | MULHU | 0 | 244 | 
| regex_program |  | OR | 0 | 23,544 | 
| regex_program |  | PHANTOM | 0 | 289 | 
| regex_program |  | SLL | 0 | 213,548 | 
| regex_program |  | SLT | 0 | 5 | 
| regex_program |  | SLTU | 0 | 33,454 | 
| regex_program |  | SRA | 0 | 1 | 
| regex_program |  | SRL | 0 | 5,090 | 
| regex_program |  | STOREB | 0 | 12,737 | 
| regex_program |  | STOREH | 0 | 10,074 | 
| regex_program |  | STOREW | 0 | 772,922 | 
| regex_program |  | SUB | 0 | 42,582 | 
| regex_program |  | XOR | 0 | 9,562 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,428 | 78,323 | 4,200,289 | 632,452,480 | 14,511 | 2,550 | 1,205 | 5,140 | 2,726 | 2,390 | 165,198,010 | 498 | 60,384 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/regex-c710bdc3f383aceed4ae546933cba5cbdb898e19-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/c710bdc3f383aceed4ae546933cba5cbdb898e19

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12851930243)
