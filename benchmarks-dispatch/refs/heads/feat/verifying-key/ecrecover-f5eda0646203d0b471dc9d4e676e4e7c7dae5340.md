| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+83.1%])</span> 2.54 | <span style='color: red'>(+1 [+83.1%])</span> 2.54 |
| ecrecover_program | <span style='color: red'>(+1 [+83.1%])</span> 2.54 | <span style='color: red'>(+1 [+83.1%])</span> 2.54 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1154 [+83.1%])</span> 2,542 | <span style='color: red'>(+1154 [+83.1%])</span> 2,542 | <span style='color: red'>(+1154 [+83.1%])</span> 2,542 | <span style='color: red'>(+1154 [+83.1%])</span> 2,542 |
| `main_cells_used     ` | <span style='color: red'>(+1634970 [+11.3%])</span> 16,105,156 | <span style='color: red'>(+1634970 [+11.3%])</span> 16,105,156 | <span style='color: red'>(+1634970 [+11.3%])</span> 16,105,156 | <span style='color: red'>(+1634970 [+11.3%])</span> 16,105,156 |
| `total_cycles        ` | <span style='color: red'>(+45054 [+15.6%])</span> 334,501 | <span style='color: red'>(+45054 [+15.6%])</span> 334,501 | <span style='color: red'>(+45054 [+15.6%])</span> 334,501 | <span style='color: red'>(+45054 [+15.6%])</span> 334,501 |
| `execute_time_ms     ` | <span style='color: red'>(+1135 [+777.4%])</span> 1,281 | <span style='color: red'>(+1135 [+777.4%])</span> 1,281 | <span style='color: red'>(+1135 [+777.4%])</span> 1,281 | <span style='color: red'>(+1135 [+777.4%])</span> 1,281 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+20 [+11.0%])</span> 202 | <span style='color: red'>(+20 [+11.0%])</span> 202 | <span style='color: red'>(+20 [+11.0%])</span> 202 | <span style='color: red'>(+20 [+11.0%])</span> 202 |
| `stark_prove_excluding_trace_time_ms` |  1,059 |  1,059 |  1,059 |  1,059 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-10 [-5.5%])</span> 171 | <span style='color: green'>(-10 [-5.5%])</span> 171 | <span style='color: green'>(-10 [-5.5%])</span> 171 | <span style='color: green'>(-10 [-5.5%])</span> 171 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+6.1%])</span> 35 | <span style='color: red'>(+2 [+6.1%])</span> 35 | <span style='color: red'>(+2 [+6.1%])</span> 35 | <span style='color: red'>(+2 [+6.1%])</span> 35 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-5 [-3.0%])</span> 160 | <span style='color: green'>(-5 [-3.0%])</span> 160 | <span style='color: green'>(-5 [-3.0%])</span> 160 | <span style='color: green'>(-5 [-3.0%])</span> 160 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-2 [-1.4%])</span> 145 | <span style='color: green'>(-2 [-1.4%])</span> 145 | <span style='color: green'>(-2 [-1.4%])</span> 145 | <span style='color: green'>(-2 [-1.4%])</span> 145 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+13 [+3.2%])</span> 424 | <span style='color: red'>(+13 [+3.2%])</span> 424 | <span style='color: red'>(+13 [+3.2%])</span> 424 | <span style='color: red'>(+13 [+3.2%])</span> 424 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 916 | 9 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,513 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 2,851,236 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 630,972 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 272,664 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 307,548 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 29,880 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 91,390 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 271,625 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 262,403 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 430,534 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 151,788 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 14,016 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 640 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 686,528 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 24,084 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 120,438 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 536,678 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 237,384 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 169,344 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 158,916 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 1,049,682 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 1,135,085 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 3,010,343 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 295 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 80,104 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 84,640 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcDouble | 0 | 695,237 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularAddSub | 0 | 5,970 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 13,676 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcAddNe | 0 | 453,750 | 
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,560 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 66 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 6,656 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 352 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 16 | 25 | 671,744 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 1,024 |  | 16 | 11 | 27,648 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 512 |  | 16 | 13 | 14,848 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 52 | 36 | 11,534,336 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 8,192 |  | 52 | 36 | 720,896 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 72 | 59 | 1,048 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 52 | 31 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 320 | 263 | 37,312 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 113,675 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,470 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 10,076 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 22,397 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 21,912 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 8,029 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 3,214 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 8,478 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 4,704 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 130,586 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 2,584 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 4,233 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 41 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 13,688 | 
| ecrecover_program | AccessAdapter<2> | 0 | 650 | 
| ecrecover_program | AccessAdapter<32> | 0 | 6,844 | 
| ecrecover_program | AccessAdapter<4> | 0 | 364 | 
| ecrecover_program | AccessAdapter<8> | 0 | 28,150 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,173 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 3,206 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,502 | 
| ecrecover_program | PhantomAir | 0 | 11 | 
| ecrecover_program | ProgramChip | 0 | 17,139 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 219 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 79,201 | 
| ecrecover_program |  | AND | 0 | 17,527 | 
| ecrecover_program |  | AUIPC | 0 | 4,233 | 
| ecrecover_program |  | BEQ | 0 | 16,559 | 
| ecrecover_program |  | BGEU | 0 | 438 | 
| ecrecover_program |  | BLT | 0 | 20 | 
| ecrecover_program |  | BLTU | 0 | 21,454 | 
| ecrecover_program |  | BNE | 0 | 5,838 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_BUFFER | 0 | 11 | 
| ecrecover_program |  | HINT_STOREW | 0 | 11 | 
| ecrecover_program |  | IS_EQ | 0 | 3,233 | 
| ecrecover_program |  | JAL | 0 | 1,338 | 
| ecrecover_program |  | JALR | 0 | 8,478 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 4,704 | 
| ecrecover_program |  | LOADBU | 0 | 3,876 | 
| ecrecover_program |  | LOADW | 0 | 25,602 | 
| ecrecover_program |  | LUI | 0 | 6,691 | 
| ecrecover_program |  | MUL | 0 | 2,584 | 
| ecrecover_program |  | ModularAddSub | 0 | 30 | 
| ecrecover_program |  | ModularMulDiv | 0 | 52 | 
| ecrecover_program |  | OR | 0 | 7,574 | 
| ecrecover_program |  | PHANTOM | 0 | 11 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 5,125 | 
| ecrecover_program |  | SLTU | 0 | 2,470 | 
| ecrecover_program |  | SRL | 0 | 4,951 | 
| ecrecover_program |  | STOREB | 0 | 27,685 | 
| ecrecover_program |  | STOREW | 0 | 73,423 | 
| ecrecover_program |  | SUB | 0 | 8,543 | 
| ecrecover_program |  | XOR | 0 | 830 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 202 | 2,542 | 334,501 | 50,573,290 | 1,059 | 111 | 145 | 160 | 424 | 171 | 16,105,156 | 35 | 1,281 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 801,892 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 2,409,488 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 400,946 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 3,982,476 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 911,680 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 16,448 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 9,501,874 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-f5eda0646203d0b471dc9d4e676e4e7c7dae5340/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/f5eda0646203d0b471dc9d4e676e4e7c7dae5340

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15231328270)
