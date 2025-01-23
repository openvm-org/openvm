| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+2 [+72.0%])</span> 4.49 | <span style='color: red'>(+2 [+72.0%])</span> 4.49 |
| ecrecover_program | <span style='color: red'>(+2 [+72.0%])</span> 4.49 | <span style='color: red'>(+2 [+72.0%])</span> 4.49 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1881 [+72.0%])</span> 4,492 | <span style='color: red'>(+1881 [+72.0%])</span> 4,492 | <span style='color: red'>(+1881 [+72.0%])</span> 4,492 | <span style='color: red'>(+1881 [+72.0%])</span> 4,492 |
| `main_cells_used     ` | <span style='color: red'>(+155632 [+1.0%])</span> 15,230,665 | <span style='color: red'>(+155632 [+1.0%])</span> 15,230,665 | <span style='color: red'>(+155632 [+1.0%])</span> 15,230,665 | <span style='color: red'>(+155632 [+1.0%])</span> 15,230,665 |
| `total_cycles        ` | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 |
| `execute_time_ms     ` | <span style='color: red'>(+1922 [+1316.4%])</span> 2,068 | <span style='color: red'>(+1922 [+1316.4%])</span> 2,068 | <span style='color: red'>(+1922 [+1316.4%])</span> 2,068 | <span style='color: red'>(+1922 [+1316.4%])</span> 2,068 |
| `trace_gen_time_ms   ` |  259 |  259 |  259 |  259 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-41 [-1.9%])</span> 2,165 | <span style='color: green'>(-41 [-1.9%])</span> 2,165 | <span style='color: green'>(-41 [-1.9%])</span> 2,165 | <span style='color: green'>(-41 [-1.9%])</span> 2,165 |
| `main_trace_commit_time_ms` |  390 |  390 |  390 |  390 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-5 [-11.6%])</span> 38 | <span style='color: green'>(-5 [-11.6%])</span> 38 | <span style='color: green'>(-5 [-11.6%])</span> 38 | <span style='color: green'>(-5 [-11.6%])</span> 38 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-8 [-1.5%])</span> 527 | <span style='color: green'>(-8 [-1.5%])</span> 527 | <span style='color: green'>(-8 [-1.5%])</span> 527 | <span style='color: green'>(-8 [-1.5%])</span> 527 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-31 [-9.9%])</span> 283 | <span style='color: green'>(-31 [-9.9%])</span> 283 | <span style='color: green'>(-31 [-9.9%])</span> 283 | <span style='color: green'>(-31 [-9.9%])</span> 283 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+11 [+3.7%])</span> 306 | <span style='color: red'>(+11 [+3.7%])</span> 306 | <span style='color: red'>(+11 [+3.7%])</span> 306 | <span style='color: red'>(+11 [+3.7%])</span> 306 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-1.3%])</span> 618 | <span style='color: green'>(-8 [-1.3%])</span> 618 | <span style='color: green'>(-8 [-1.3%])</span> 618 | <span style='color: green'>(-8 [-1.3%])</span> 618 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,032 | 12 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,571 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 223 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 2 | 411 | 513 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 156 | 189 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 456 | 
| ecrecover_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 2,645,532 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 559,512 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 250,740 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 318,600 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 900 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 74,407 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 228,536 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 238,023 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 275,912 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 124,202 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 29,600 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 384 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 719,648 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 22,734 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 50,292 | 
| ecrecover_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 5,564 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 531,698 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 186,060 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 132,300 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 98,000 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 553,840 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 1,037,520 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 2,702,880 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 285 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 195 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 79,329 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 71,022 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,EcDoubleCoreAir> |  | EcDouble | 0 | 690,153 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularAddSub | 0 | 2,388 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 8,352 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcAddNe | 0 | 449,394 | 
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,680 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 270 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 24 | 25 | 802,816 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 24 | 11 | 8,960 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 24 | 41 | 532,480 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 24 | 13 | 4,736 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 24 | 17 | 1,343,488 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,288 | 3,164 | 569,856 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 20 | 32 | 212,992 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 12 | 6 | 1,152 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 2,048 |  | 8 | 300 | 630,784 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 80 | 36 | 15,204,352 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 48 | 26 | 1,212,416 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 56 | 32 | 2,883,584 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 44 | 18 | 253,952 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 256 |  | 36 | 26 | 15,872 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 76 | 35 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 72 | 40 | 14,680,064 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 104 | 57 | 1,288 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 100 | 39 | 1,112 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 80 | 31 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 21 | 200,704 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 316 | 261 | 18,464 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 104,869 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,011 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 8,803 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 15,389 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 23,426 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 4,057 | 
| ecrecover_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 214 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 3,194 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 6,645 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 3,780 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 109,806 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 2,559 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 3,383 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,EcDoubleCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 21 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 13,306 | 
| ecrecover_program | AccessAdapter<2> | 0 | 132 | 
| ecrecover_program | AccessAdapter<32> | 0 | 6,654 | 
| ecrecover_program | AccessAdapter<4> | 0 | 68 | 
| ecrecover_program | AccessAdapter<8> | 0 | 27,216 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,008 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 2,990 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,236 | 
| ecrecover_program | PhantomAir | 0 | 45 | 
| ecrecover_program | ProgramChip | 0 | 8,624 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 73,487 | 
| ecrecover_program |  | AND | 0 | 15,542 | 
| ecrecover_program |  | AUIPC | 0 | 3,383 | 
| ecrecover_program |  | BEQ | 0 | 10,612 | 
| ecrecover_program |  | BGEU | 0 | 925 | 
| ecrecover_program |  | BLT | 0 | 12 | 
| ecrecover_program |  | BLTU | 0 | 22,489 | 
| ecrecover_program |  | BNE | 0 | 4,777 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_STOREW | 0 | 214 | 
| ecrecover_program |  | IS_EQ | 0 | 3,203 | 
| ecrecover_program |  | JAL | 0 | 1,263 | 
| ecrecover_program |  | JALR | 0 | 6,645 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 3,780 | 
| ecrecover_program |  | LOADBU | 0 | 2,450 | 
| ecrecover_program |  | LOADW | 0 | 13,846 | 
| ecrecover_program |  | LUI | 0 | 2,794 | 
| ecrecover_program |  | MUL | 0 | 2,559 | 
| ecrecover_program |  | MULHU | 0 | 5 | 
| ecrecover_program |  | ModularAddSub | 0 | 12 | 
| ecrecover_program |  | ModularMulDiv | 0 | 32 | 
| ecrecover_program |  | OR | 0 | 6,965 | 
| ecrecover_program |  | PHANTOM | 0 | 45 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 4,312 | 
| ecrecover_program |  | SLTU | 0 | 2,011 | 
| ecrecover_program |  | SRL | 0 | 4,491 | 
| ecrecover_program |  | STOREB | 0 | 25,938 | 
| ecrecover_program |  | STOREW | 0 | 67,572 | 
| ecrecover_program |  | SUB | 0 | 8,850 | 
| ecrecover_program |  | XOR | 0 | 25 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 259 | 4,492 | 290,248 | 55,288,711 | 2,165 | 283 | 306 | 527 | 618 | 390 | 15,230,665 | 38 | 2,068 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/abde525bf4e10ed9cf514c242f9d423e9e8eff91/ecrecover-abde525bf4e10ed9cf514c242f9d423e9e8eff91-ecrecover_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/abde525bf4e10ed9cf514c242f9d423e9e8eff91

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12933141073)
