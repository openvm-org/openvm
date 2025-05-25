| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+80.6%])</span> 2.51 | <span style='color: red'>(+1 [+80.6%])</span> 2.51 |
| ecrecover_program | <span style='color: red'>(+1 [+80.6%])</span> 2.51 | <span style='color: red'>(+1 [+80.6%])</span> 2.51 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1119 [+80.6%])</span> 2,507 | <span style='color: red'>(+1119 [+80.6%])</span> 2,507 | <span style='color: red'>(+1119 [+80.6%])</span> 2,507 | <span style='color: red'>(+1119 [+80.6%])</span> 2,507 |
| `main_cells_used     ` | <span style='color: red'>(+1216843 [+8.4%])</span> 15,687,029 | <span style='color: red'>(+1216843 [+8.4%])</span> 15,687,029 | <span style='color: red'>(+1216843 [+8.4%])</span> 15,687,029 | <span style='color: red'>(+1216843 [+8.4%])</span> 15,687,029 |
| `total_cycles        ` | <span style='color: red'>(+34760 [+12.0%])</span> 324,207 | <span style='color: red'>(+34760 [+12.0%])</span> 324,207 | <span style='color: red'>(+34760 [+12.0%])</span> 324,207 | <span style='color: red'>(+34760 [+12.0%])</span> 324,207 |
| `execute_time_ms     ` | <span style='color: red'>(+1091 [+747.3%])</span> 1,237 | <span style='color: red'>(+1091 [+747.3%])</span> 1,237 | <span style='color: red'>(+1091 [+747.3%])</span> 1,237 | <span style='color: red'>(+1091 [+747.3%])</span> 1,237 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+16 [+8.8%])</span> 198 | <span style='color: red'>(+16 [+8.8%])</span> 198 | <span style='color: red'>(+16 [+8.8%])</span> 198 | <span style='color: red'>(+16 [+8.8%])</span> 198 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+12 [+1.1%])</span> 1,072 | <span style='color: red'>(+12 [+1.1%])</span> 1,072 | <span style='color: red'>(+12 [+1.1%])</span> 1,072 | <span style='color: red'>(+12 [+1.1%])</span> 1,072 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-13 [-7.2%])</span> 168 | <span style='color: green'>(-13 [-7.2%])</span> 168 | <span style='color: green'>(-13 [-7.2%])</span> 168 | <span style='color: green'>(-13 [-7.2%])</span> 168 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+3.0%])</span> 34 | <span style='color: red'>(+1 [+3.0%])</span> 34 | <span style='color: red'>(+1 [+3.0%])</span> 34 | <span style='color: red'>(+1 [+3.0%])</span> 34 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+22 [+13.3%])</span> 187 | <span style='color: red'>(+22 [+13.3%])</span> 187 | <span style='color: red'>(+22 [+13.3%])</span> 187 | <span style='color: red'>(+22 [+13.3%])</span> 187 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.9%])</span> 110 | <span style='color: red'>(+1 [+0.9%])</span> 110 | <span style='color: red'>(+1 [+0.9%])</span> 110 | <span style='color: red'>(+1 [+0.9%])</span> 110 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-5 [-3.4%])</span> 142 | <span style='color: green'>(-5 [-3.4%])</span> 142 | <span style='color: green'>(-5 [-3.4%])</span> 142 | <span style='color: green'>(-5 [-3.4%])</span> 142 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+4 [+1.0%])</span> 415 | <span style='color: red'>(+4 [+1.0%])</span> 415 | <span style='color: red'>(+4 [+1.0%])</span> 415 | <span style='color: red'>(+4 [+1.0%])</span> 415 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 908 | 7 | 

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
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 2,757,708 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 604,332 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 257,904 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 306,180 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 6,480 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 86,210 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 252,386 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 241,468 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 427,206 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 136,656 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 13,696 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 640 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 679,168 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 23,490 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 120,078 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 535,848 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 234,024 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 142,524 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 131,651 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 1,032,585 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 1,083,220 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 2,982,668 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 295 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 195 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 79,794 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 83,440 | 
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
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 64 |  | 16 | 13 | 1,856 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
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
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 52 | 36 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 72 | 59 | 1,048 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 52 | 31 | 339,968 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 320 | 263 | 37,312 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 109,239 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,330 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 9,318 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 21,687 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 21,672 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 7,976 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 3,209 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 8,358 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 3,959 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 127,564 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 2,574 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 4,173 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 41 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 13,648 | 
| ecrecover_program | AccessAdapter<32> | 0 | 6,824 | 
| ecrecover_program | AccessAdapter<4> | 0 | 34 | 
| ecrecover_program | AccessAdapter<8> | 0 | 28,054 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,123 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 3,216 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,500 | 
| ecrecover_program | PhantomAir | 0 | 11 | 
| ecrecover_program | ProgramChip | 0 | 14,505 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 219 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 76,603 | 
| ecrecover_program |  | AND | 0 | 16,787 | 
| ecrecover_program |  | AUIPC | 0 | 4,173 | 
| ecrecover_program |  | BEQ | 0 | 16,431 | 
| ecrecover_program |  | BGEU | 0 | 428 | 
| ecrecover_program |  | BLT | 0 | 20 | 
| ecrecover_program |  | BLTU | 0 | 21,224 | 
| ecrecover_program |  | BNE | 0 | 5,256 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_BUFFER | 0 | 11 | 
| ecrecover_program |  | HINT_STOREW | 0 | 11 | 
| ecrecover_program |  | IS_EQ | 0 | 3,228 | 
| ecrecover_program |  | JAL | 0 | 1,305 | 
| ecrecover_program |  | JALR | 0 | 8,358 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 3,959 | 
| ecrecover_program |  | LOADBU | 0 | 3,211 | 
| ecrecover_program |  | LOADW | 0 | 25,185 | 
| ecrecover_program |  | LUI | 0 | 6,671 | 
| ecrecover_program |  | MUL | 0 | 2,574 | 
| ecrecover_program |  | MULHU | 0 | 5 | 
| ecrecover_program |  | ModularAddSub | 0 | 30 | 
| ecrecover_program |  | ModularMulDiv | 0 | 52 | 
| ecrecover_program |  | OR | 0 | 7,164 | 
| ecrecover_program |  | PHANTOM | 0 | 11 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 4,762 | 
| ecrecover_program |  | SLTU | 0 | 2,330 | 
| ecrecover_program |  | SRL | 0 | 4,556 | 
| ecrecover_program |  | STOREB | 0 | 26,420 | 
| ecrecover_program |  | STOREW | 0 | 72,748 | 
| ecrecover_program |  | SUB | 0 | 8,505 | 
| ecrecover_program |  | XOR | 0 | 180 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 198 | 2,507 | 324,207 | 49,878,178 | 1,072 | 110 | 142 | 187 | 415 | 168 | 15,687,029 | 34 | 1,237 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 793,716 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 2,380,544 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 396,858 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 3,942,716 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 911,688 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 16,512 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 9,404,594 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-7de60b443537e1fb2968a3afaa236dfaf622b871/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/7de60b443537e1fb2968a3afaa236dfaf622b871

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15240460956)
