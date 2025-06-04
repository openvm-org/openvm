| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+16.0%])</span> 1.62 | <span style='color: red'>(+0 [+16.0%])</span> 1.62 |
| ecrecover_program | <span style='color: red'>(+0 [+16.0%])</span> 1.62 | <span style='color: red'>(+0 [+16.0%])</span> 1.62 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+223 [+16.0%])</span> 1,619 | <span style='color: red'>(+223 [+16.0%])</span> 1,619 | <span style='color: red'>(+223 [+16.0%])</span> 1,619 | <span style='color: red'>(+223 [+16.0%])</span> 1,619 |
| `main_cells_used     ` | <span style='color: green'>(-6101761 [-42.2%])</span> 8,368,425 | <span style='color: green'>(-6101761 [-42.2%])</span> 8,368,425 | <span style='color: green'>(-6101761 [-42.2%])</span> 8,368,425 | <span style='color: green'>(-6101761 [-42.2%])</span> 8,368,425 |
| `total_cycles        ` | <span style='color: green'>(-148606 [-51.3%])</span> 140,841 | <span style='color: green'>(-148606 [-51.3%])</span> 140,841 | <span style='color: green'>(-148606 [-51.3%])</span> 140,841 | <span style='color: green'>(-148606 [-51.3%])</span> 140,841 |
| `execute_time_ms     ` | <span style='color: red'>(+449 [+305.4%])</span> 596 | <span style='color: red'>(+449 [+305.4%])</span> 596 | <span style='color: red'>(+449 [+305.4%])</span> 596 | <span style='color: red'>(+449 [+305.4%])</span> 596 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-68 [-37.6%])</span> 113 | <span style='color: green'>(-68 [-37.6%])</span> 113 | <span style='color: green'>(-68 [-37.6%])</span> 113 | <span style='color: green'>(-68 [-37.6%])</span> 113 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-158 [-14.8%])</span> 910 | <span style='color: green'>(-158 [-14.8%])</span> 910 | <span style='color: green'>(-158 [-14.8%])</span> 910 | <span style='color: green'>(-158 [-14.8%])</span> 910 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-29 [-17.7%])</span> 135 | <span style='color: green'>(-29 [-17.7%])</span> 135 | <span style='color: green'>(-29 [-17.7%])</span> 135 | <span style='color: green'>(-29 [-17.7%])</span> 135 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-5 [-15.2%])</span> 28 | <span style='color: green'>(-5 [-15.2%])</span> 28 | <span style='color: green'>(-5 [-15.2%])</span> 28 | <span style='color: green'>(-5 [-15.2%])</span> 28 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-31 [-20.4%])</span> 121 | <span style='color: green'>(-31 [-20.4%])</span> 121 | <span style='color: green'>(-31 [-20.4%])</span> 121 | <span style='color: green'>(-31 [-20.4%])</span> 121 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-30 [-27.5%])</span> 79 | <span style='color: green'>(-30 [-27.5%])</span> 79 | <span style='color: green'>(-30 [-27.5%])</span> 79 | <span style='color: green'>(-30 [-27.5%])</span> 79 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-12 [-8.8%])</span> 125 | <span style='color: green'>(-12 [-8.8%])</span> 125 | <span style='color: green'>(-12 [-8.8%])</span> 125 | <span style='color: green'>(-12 [-8.8%])</span> 125 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-51 [-11.1%])</span> 407 | <span style='color: green'>(-51 [-11.1%])</span> 407 | <span style='color: green'>(-51 [-11.1%])</span> 407 | <span style='color: green'>(-51 [-11.1%])</span> 407 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 918 | 8 | 

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
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 1,348,200 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 232,200 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 165,456 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 32,076 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 6,480 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 84,360 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 252,386 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 241,468 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 190,346 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 157,716 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 13,696 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 640 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 105,696 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 34,020 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 51,282 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 946,366 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 159,936 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 143,244 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 131,241 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 620,658 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 251,166 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 531,237 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 295 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 195 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 1,116 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 56,980 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcDouble | 0 | 695,237 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularAddSub | 0 | 3,980 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 8,416 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcAddNe | 0 | 453,750 | 
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,560 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 66 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 6,656 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 352 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 4,096 |  | 16 | 25 | 167,936 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 64 |  | 16 | 13 | 1,856 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 8,192 |  | 16 | 17 | 270,336 | 
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
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8,192 |  | 56 | 166 | 1,818,624 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 52 | 36 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 72 | 59 | 1,048 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 64 |  | 52 | 31 | 5,312 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 20 | 196,608 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 49,567 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,280 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 9,318 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 13,387 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 3,751 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 4,739 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 5,682 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 5,712 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 3,979 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 37,422 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 36 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 2,850 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 21 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 3,032 | 
| ecrecover_program | AccessAdapter<32> | 0 | 1,516 | 
| ecrecover_program | AccessAdapter<4> | 0 | 34 | 
| ecrecover_program | AccessAdapter<8> | 0 | 6,778 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,126 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 3,104 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,378 | 
| ecrecover_program | PhantomAir | 0 | 11 | 
| ecrecover_program | ProgramChip | 0 | 13,908 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 219 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 37,450 | 
| ecrecover_program |  | AND | 0 | 6,450 | 
| ecrecover_program |  | AUIPC | 0 | 2,850 | 
| ecrecover_program |  | BEQ | 0 | 7,321 | 
| ecrecover_program |  | BGEU | 0 | 428 | 
| ecrecover_program |  | BLT | 0 | 20 | 
| ecrecover_program |  | BLTU | 0 | 3,303 | 
| ecrecover_program |  | BNE | 0 | 6,066 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_BUFFER | 0 | 11 | 
| ecrecover_program |  | HINT_STOREW | 0 | 11 | 
| ecrecover_program |  | IS_EQ | 0 | 5,701 | 
| ecrecover_program |  | JAL | 0 | 1,890 | 
| ecrecover_program |  | JALR | 0 | 5,712 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 3,979 | 
| ecrecover_program |  | LOADBU | 0 | 3,201 | 
| ecrecover_program |  | LOADW | 0 | 15,138 | 
| ecrecover_program |  | LUI | 0 | 2,849 | 
| ecrecover_program |  | MUL | 0 | 36 | 
| ecrecover_program |  | MULHU | 0 | 5 | 
| ecrecover_program |  | ModularAddSub | 0 | 20 | 
| ecrecover_program |  | ModularMulDiv | 0 | 32 | 
| ecrecover_program |  | OR | 0 | 4,596 | 
| ecrecover_program |  | PHANTOM | 0 | 11 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 4,762 | 
| ecrecover_program |  | SLTU | 0 | 2,280 | 
| ecrecover_program |  | SRL | 0 | 4,556 | 
| ecrecover_program |  | STOREB | 0 | 6,126 | 
| ecrecover_program |  | STOREW | 0 | 12,957 | 
| ecrecover_program |  | SUB | 0 | 891 | 
| ecrecover_program |  | XOR | 0 | 180 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 113 | 1,619 | 140,841 | 33,460,242 | 910 | 79 | 125 | 121 | 407 | 135 | 8,368,425 | 28 | 596 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 416,916 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,260,096 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 208,458 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,696,268 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 506,088 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 384 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 6,050,770 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/20dc1ecb42fd53cb87e7252cf0b9d4284cea7ed8

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15432494046)
