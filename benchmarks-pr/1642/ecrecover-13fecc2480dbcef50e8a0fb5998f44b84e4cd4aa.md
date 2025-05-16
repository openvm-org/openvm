| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+72 [+5163.9%])</span> 72.96 | <span style='color: red'>(+3 [+208.4%])</span> 4.27 |
| ecrecover_program | <span style='color: red'>(+72 [+5163.9%])</span> 72.96 | <span style='color: red'>(+3 [+208.4%])</span> 4.27 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2667 [+192.4%])</span> 4,053.17 | <span style='color: red'>(+71571 [+5163.9%])</span> 72,957 | <span style='color: red'>(+2888 [+208.4%])</span> 4,274 | <span style='color: red'>(+2457 [+177.3%])</span> 3,843 |
| `main_cells_used     ` | <span style='color: red'>(+84819634 [+586.2%])</span> 99,289,820.17 | <span style='color: red'>(+1772746577 [+12251.0%])</span> 1,787,216,763 | <span style='color: red'>(+86928943 [+600.7%])</span> 101,399,129 | <span style='color: red'>(+73020455 [+504.6%])</span> 87,490,641 |
| `total_cycles        ` | <span style='color: red'>(+2275642 [+786.2%])</span> 2,565,088.94 | <span style='color: red'>(+45882154 [+15851.7%])</span> 46,171,601 | <span style='color: red'>(+2320292 [+801.6%])</span> 2,609,739 | <span style='color: red'>(+1965738 [+679.1%])</span> 2,255,185 |
| `execute_time_ms     ` | <span style='color: red'>(+301 [+206.2%])</span> 447.06 | <span style='color: red'>(+7901 [+5411.6%])</span> 8,047 | <span style='color: red'>(+330 [+226.0%])</span> 476 | <span style='color: red'>(+249 [+170.5%])</span> 395 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+892 [+492.6%])</span> 1,072.61 | <span style='color: red'>(+19126 [+10566.9%])</span> 19,307 | <span style='color: red'>(+942 [+520.4%])</span> 1,123 | <span style='color: red'>(+796 [+439.8%])</span> 977 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+1474 [+139.2%])</span> 2,533.50 | <span style='color: red'>(+44544 [+4206.2%])</span> 45,603 | <span style='color: red'>(+1671 [+157.8%])</span> 2,730 | <span style='color: red'>(+1321 [+124.7%])</span> 2,380 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+294 [+166.1%])</span> 471 | <span style='color: red'>(+8301 [+4689.8%])</span> 8,478 | <span style='color: red'>(+360 [+203.4%])</span> 537 | <span style='color: red'>(+257 [+145.2%])</span> 434 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+186 [+548.5%])</span> 220.50 | <span style='color: red'>(+3935 [+11573.5%])</span> 3,969 | <span style='color: red'>(+210 [+617.6%])</span> 244 | <span style='color: red'>(+161 [+473.5%])</span> 195 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+398 [+240.0%])</span> 564.39 | <span style='color: red'>(+9993 [+6019.9%])</span> 10,159 | <span style='color: red'>(+432 [+260.2%])</span> 598 | <span style='color: red'>(+365 [+219.9%])</span> 531 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+267 [+252.1%])</span> 373.28 | <span style='color: red'>(+6613 [+6238.7%])</span> 6,719 | <span style='color: red'>(+295 [+278.3%])</span> 401 | <span style='color: red'>(+241 [+227.4%])</span> 347 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+60 [+39.1%])</span> 212.83 | <span style='color: red'>(+3678 [+2403.9%])</span> 3,831 | <span style='color: red'>(+120 [+78.4%])</span> 273 | <span style='color: red'>(+45 [+29.4%])</span> 198 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+272 [+66.6%])</span> 681.44 | <span style='color: red'>(+11857 [+2899.0%])</span> 12,266 | <span style='color: red'>(+321 [+78.5%])</span> 730 | <span style='color: red'>(+248 [+60.6%])</span> 657 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 18 | 917 | 9 | 

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

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<2> | 10 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<2> | 14 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<2> | 17 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<2> | 3 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<2> | 7 | 256 |  | 16 | 11 | 6,912 | 
| ecrecover_program | AccessAdapterAir<4> | 10 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<4> | 14 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<4> | 17 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<4> | 3 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<4> | 7 | 128 |  | 16 | 13 | 3,712 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 1 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 10 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 11 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 12 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 13 | 2,048 |  | 16 | 17 | 67,584 | 
| ecrecover_program | AccessAdapterAir<8> | 14 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 15 | 2,048 |  | 16 | 17 | 67,584 | 
| ecrecover_program | AccessAdapterAir<8> | 16 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 17 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 3 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 4 | 2,048 |  | 16 | 17 | 67,584 | 
| ecrecover_program | AccessAdapterAir<8> | 5 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 6 | 2,048 |  | 16 | 17 | 67,584 | 
| ecrecover_program | AccessAdapterAir<8> | 7 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 8 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | AccessAdapterAir<8> | 9 | 4,096 |  | 16 | 17 | 135,168 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 13 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 14 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 15 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 16 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 17 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 1 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 10 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | KeccakVmAir | 11 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 12 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 13 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 14 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | KeccakVmAir | 15 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 16 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 17 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | KeccakVmAir | 2 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 3 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | KeccakVmAir | 4 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 5 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 6 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 7 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | KeccakVmAir | 8 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | KeccakVmAir | 9 | 1 |  | 1,056 | 3,163 | 4,219 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 10 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 11 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 12 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 13 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 14 | 8,192 |  | 16 | 32 | 393,216 | 
| ecrecover_program | MemoryMerkleAir<8> | 15 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 16 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 17 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 3 | 8,192 |  | 16 | 32 | 393,216 | 
| ecrecover_program | MemoryMerkleAir<8> | 4 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 5 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 6 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 7 | 8,192 |  | 16 | 32 | 393,216 | 
| ecrecover_program | MemoryMerkleAir<8> | 8 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 9 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 1 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 10 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 11 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 12 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 13 | 2,048 |  | 12 | 20 | 65,536 | 
| ecrecover_program | PersistentBoundaryAir<8> | 14 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 15 | 2,048 |  | 12 | 20 | 65,536 | 
| ecrecover_program | PersistentBoundaryAir<8> | 16 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 17 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 3 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 4 | 2,048 |  | 12 | 20 | 65,536 | 
| ecrecover_program | PersistentBoundaryAir<8> | 5 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 6 | 2,048 |  | 12 | 20 | 65,536 | 
| ecrecover_program | PersistentBoundaryAir<8> | 7 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 8 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 9 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| ecrecover_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 10 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 11 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 12 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 13 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 14 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 15 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 16 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 17 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 7 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 8 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | PhantomAir | 9 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 13 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 14 | 8,192 |  | 8 | 300 | 2,523,136 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 15 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 16 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 17 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 8,192 |  | 8 | 300 | 2,523,136 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 8,192 |  | 8 | 300 | 2,523,136 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 10 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 11 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 12 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 13 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 14 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 15 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 16 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 17 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 2 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 3 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 4 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 5 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 6 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 7 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 8 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 9 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 10 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 11 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 12 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 13 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 14 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 15 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 16 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 17 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 7 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 8 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 9 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 64 |  | 44 | 32 | 4,864 | 
| ecrecover_program | Rv32HintStoreAir | 10 | 64 |  | 44 | 32 | 4,864 | 
| ecrecover_program | Rv32HintStoreAir | 14 | 64 |  | 44 | 32 | 4,864 | 
| ecrecover_program | Rv32HintStoreAir | 3 | 64 |  | 44 | 32 | 4,864 | 
| ecrecover_program | Rv32HintStoreAir | 7 | 64 |  | 44 | 32 | 4,864 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 13 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 14 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 15 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 16 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 17 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 13 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 14 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 15 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 16 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 17 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 524,288 |  | 40 | 37 | 40,370,176 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 13 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 14 | 524,288 |  | 40 | 37 | 40,370,176 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 15 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 16 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 17 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 524,288 |  | 40 | 37 | 40,370,176 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 262,144 |  | 40 | 37 | 20,185,088 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 11 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 12 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 13 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 14 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 15 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 16 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 17 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 5 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 6 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 7 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 8 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 9 | 262,144 |  | 52 | 53 | 27,525,120 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 14 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 15 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 16 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 17 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 32,768 |  | 28 | 26 | 1,769,472 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 16,384 |  | 32 | 32 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 10 | 16,384 |  | 32 | 32 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 11 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 12 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | 16,384 |  | 32 | 32 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 15 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 16 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 17 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 16,384 |  | 32 | 32 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 5 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 7 | 16,384 |  | 32 | 32 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 8 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 9 | 32,768 |  | 32 | 32 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 13 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 14 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 16 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 17 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 8,192 |  | 28 | 18 | 376,832 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 10 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 11 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 12 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 13 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 14 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 15 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 17 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 5 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 7 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 8 | 16,384 |  | 36 | 28 | 1,048,576 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 9 | 32,768 |  | 36 | 28 | 2,097,152 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 10 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 11 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 12 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 13 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 14 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 15 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 16 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 17 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 5 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 6 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 7 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 8 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 9 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 10 | 524,288 |  | 52 | 41 | 48,758,784 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 12 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 13 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 14 | 524,288 |  | 52 | 41 | 48,758,784 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 15 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 16 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 524,288 |  | 52 | 41 | 48,758,784 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 5 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 7 | 524,288 |  | 52 | 41 | 48,758,784 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 8 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 9 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 10 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 11 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 12 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 13 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 14 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 15 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 16 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 17 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 3 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 5 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 6 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 7 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 8 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 9 | 262,144 |  | 72 | 39 | 29,097,984 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 10 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 11 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 12 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 13 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 14 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 15 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 16 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 17 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 5 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 6 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 7 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 8 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 9 | 262,144 |  | 52 | 31 | 21,757,952 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 10 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 11 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 13 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 14 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 15 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 16 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 5 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 7 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 8 | 8,192 |  | 28 | 20 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 9 | 16,384 |  | 28 | 20 | 786,432 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 14 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 15 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 16 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 17 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 1,068 | 4,274 | 2,541,665 | 322,532,297 | 2,730 | 401 | 247 | 598 | 694 | 537 | 97,761,407 | 244 | 476 | 
| ecrecover_program | 1 | 1,077 | 4,092 | 2,602,669 | 305,717,431 | 2,565 | 383 | 213 | 575 | 681 | 476 | 101,027,624 | 226 | 450 | 
| ecrecover_program | 10 | 1,024 | 3,879 | 2,532,979 | 275,356,700 | 2,411 | 351 | 226 | 533 | 658 | 434 | 97,912,625 | 200 | 444 | 
| ecrecover_program | 11 | 1,086 | 4,106 | 2,604,689 | 304,275,639 | 2,568 | 377 | 205 | 598 | 680 | 476 | 100,654,649 | 223 | 452 | 
| ecrecover_program | 12 | 1,110 | 4,115 | 2,599,740 | 305,717,431 | 2,560 | 379 | 210 | 569 | 689 | 476 | 101,107,887 | 228 | 445 | 
| ecrecover_program | 13 | 1,123 | 4,111 | 2,607,921 | 304,142,519 | 2,532 | 374 | 204 | 568 | 679 | 474 | 100,534,569 | 223 | 456 | 
| ecrecover_program | 14 | 1,073 | 3,894 | 2,529,848 | 275,373,084 | 2,380 | 347 | 198 | 532 | 657 | 435 | 98,139,259 | 198 | 441 | 
| ecrecover_program | 15 | 1,077 | 4,122 | 2,607,517 | 304,142,519 | 2,588 | 375 | 207 | 568 | 718 | 484 | 100,687,227 | 227 | 457 | 
| ecrecover_program | 16 | 1,077 | 4,080 | 2,600,346 | 305,717,431 | 2,555 | 379 | 208 | 572 | 686 | 478 | 101,116,961 | 222 | 448 | 
| ecrecover_program | 17 | 977 | 4,042 | 2,255,185 | 303,532,316 | 2,670 | 378 | 273 | 580 | 730 | 474 | 87,490,641 | 224 | 395 | 
| ecrecover_program | 2 | 1,087 | 4,086 | 2,604,689 | 304,275,639 | 2,550 | 378 | 203 | 567 | 678 | 474 | 100,486,028 | 241 | 449 | 
| ecrecover_program | 3 | 1,012 | 3,843 | 2,530,959 | 274,111,516 | 2,394 | 348 | 203 | 531 | 663 | 440 | 98,026,885 | 199 | 437 | 
| ecrecover_program | 4 | 1,083 | 4,062 | 2,606,911 | 304,142,519 | 2,527 | 377 | 205 | 565 | 677 | 472 | 100,572,959 | 222 | 452 | 
| ecrecover_program | 5 | 1,091 | 4,108 | 2,601,053 | 305,717,431 | 2,563 | 382 | 223 | 569 | 681 | 475 | 101,150,849 | 223 | 454 | 
| ecrecover_program | 6 | 1,100 | 4,080 | 2,607,012 | 304,142,519 | 2,531 | 377 | 200 | 566 | 673 | 476 | 100,529,599 | 228 | 449 | 
| ecrecover_program | 7 | 1,090 | 3,915 | 2,530,757 | 275,373,084 | 2,384 | 348 | 203 | 531 | 657 | 441 | 98,159,605 | 195 | 441 | 
| ecrecover_program | 8 | 1,104 | 4,090 | 2,597,922 | 304,275,639 | 2,537 | 381 | 201 | 566 | 679 | 474 | 100,458,860 | 223 | 449 | 
| ecrecover_program | 9 | 1,048 | 4,058 | 2,609,739 | 306,978,999 | 2,558 | 384 | 202 | 571 | 686 | 482 | 101,399,129 | 223 | 452 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 6,951,050 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 20,705,754 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 3,475,525 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 24,177,118 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 8,241,482 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 67,675,601 | 2,013,265,921 | 
| ecrecover_program | 1 | 0 | 6,541,320 | 2,013,265,921 | 
| ecrecover_program | 1 | 1 | 19,345,498 | 2,013,265,921 | 
| ecrecover_program | 1 | 2 | 3,270,660 | 2,013,265,921 | 
| ecrecover_program | 1 | 3 | 22,882,398 | 2,013,265,921 | 
| ecrecover_program | 1 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 1 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 1 | 6 | 7,544,970 | 2,013,265,921 | 
| ecrecover_program | 1 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 1 | 8 | 63,709,518 | 2,013,265,921 | 
| ecrecover_program | 10 | 0 | 5,951,686 | 2,013,265,921 | 
| ecrecover_program | 10 | 1 | 17,645,888 | 2,013,265,921 | 
| ecrecover_program | 10 | 2 | 2,975,843 | 2,013,265,921 | 
| ecrecover_program | 10 | 3 | 20,133,828 | 2,013,265,921 | 
| ecrecover_program | 10 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 10 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 10 | 6 | 8,303,104 | 2,013,265,921 | 
| ecrecover_program | 10 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 10 | 8 | 59,135,021 | 2,013,265,921 | 
| ecrecover_program | 11 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 11 | 1 | 19,263,578 | 2,013,265,921 | 
| ecrecover_program | 11 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 11 | 3 | 22,734,942 | 2,013,265,921 | 
| ecrecover_program | 11 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 11 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 11 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 11 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 11 | 8 | 63,349,070 | 2,013,265,921 | 
| ecrecover_program | 12 | 0 | 6,541,320 | 2,013,265,921 | 
| ecrecover_program | 12 | 1 | 19,345,498 | 2,013,265,921 | 
| ecrecover_program | 12 | 2 | 3,270,660 | 2,013,265,921 | 
| ecrecover_program | 12 | 3 | 22,882,398 | 2,013,265,921 | 
| ecrecover_program | 12 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 12 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 12 | 6 | 7,544,970 | 2,013,265,921 | 
| ecrecover_program | 12 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 12 | 8 | 63,709,518 | 2,013,265,921 | 
| ecrecover_program | 13 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 13 | 1 | 19,255,386 | 2,013,265,921 | 
| ecrecover_program | 13 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 13 | 3 | 22,730,846 | 2,013,265,921 | 
| ecrecover_program | 13 | 4 | 14,336 | 2,013,265,921 | 
| ecrecover_program | 13 | 5 | 6,144 | 2,013,265,921 | 
| ecrecover_program | 13 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 13 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 13 | 8 | 63,332,686 | 2,013,265,921 | 
| ecrecover_program | 14 | 0 | 5,902,534 | 2,013,265,921 | 
| ecrecover_program | 14 | 1 | 17,563,968 | 2,013,265,921 | 
| ecrecover_program | 14 | 2 | 2,951,267 | 2,013,265,921 | 
| ecrecover_program | 14 | 3 | 19,986,372 | 2,013,265,921 | 
| ecrecover_program | 14 | 4 | 28,672 | 2,013,265,921 | 
| ecrecover_program | 14 | 5 | 12,288 | 2,013,265,921 | 
| ecrecover_program | 14 | 6 | 8,245,760 | 2,013,265,921 | 
| ecrecover_program | 14 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 14 | 8 | 58,795,053 | 2,013,265,921 | 
| ecrecover_program | 15 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 15 | 1 | 19,255,386 | 2,013,265,921 | 
| ecrecover_program | 15 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 15 | 3 | 22,730,846 | 2,013,265,921 | 
| ecrecover_program | 15 | 4 | 14,336 | 2,013,265,921 | 
| ecrecover_program | 15 | 5 | 6,144 | 2,013,265,921 | 
| ecrecover_program | 15 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 15 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 15 | 8 | 63,332,686 | 2,013,265,921 | 
| ecrecover_program | 16 | 0 | 6,541,320 | 2,013,265,921 | 
| ecrecover_program | 16 | 1 | 19,345,498 | 2,013,265,921 | 
| ecrecover_program | 16 | 2 | 3,270,660 | 2,013,265,921 | 
| ecrecover_program | 16 | 3 | 22,882,398 | 2,013,265,921 | 
| ecrecover_program | 16 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 16 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 16 | 6 | 7,544,970 | 2,013,265,921 | 
| ecrecover_program | 16 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 16 | 8 | 63,709,518 | 2,013,265,921 | 
| ecrecover_program | 17 | 0 | 6,459,462 | 2,013,265,921 | 
| ecrecover_program | 17 | 1 | 19,201,984 | 2,013,265,921 | 
| ecrecover_program | 17 | 2 | 3,229,731 | 2,013,265,921 | 
| ecrecover_program | 17 | 3 | 22,672,964 | 2,013,265,921 | 
| ecrecover_program | 17 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 17 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 17 | 6 | 7,491,904 | 2,013,265,921 | 
| ecrecover_program | 17 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 17 | 8 | 63,180,717 | 2,013,265,921 | 
| ecrecover_program | 2 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 2 | 1 | 19,263,578 | 2,013,265,921 | 
| ecrecover_program | 2 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 2 | 3 | 22,734,942 | 2,013,265,921 | 
| ecrecover_program | 2 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 2 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 2 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 2 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 2 | 8 | 63,349,070 | 2,013,265,921 | 
| ecrecover_program | 3 | 0 | 5,902,534 | 2,013,265,921 | 
| ecrecover_program | 3 | 1 | 17,563,968 | 2,013,265,921 | 
| ecrecover_program | 3 | 2 | 2,951,267 | 2,013,265,921 | 
| ecrecover_program | 3 | 3 | 19,986,372 | 2,013,265,921 | 
| ecrecover_program | 3 | 4 | 28,672 | 2,013,265,921 | 
| ecrecover_program | 3 | 5 | 12,288 | 2,013,265,921 | 
| ecrecover_program | 3 | 6 | 8,245,760 | 2,013,265,921 | 
| ecrecover_program | 3 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 3 | 8 | 58,790,957 | 2,013,265,921 | 
| ecrecover_program | 4 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 4 | 1 | 19,255,386 | 2,013,265,921 | 
| ecrecover_program | 4 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 4 | 3 | 22,730,846 | 2,013,265,921 | 
| ecrecover_program | 4 | 4 | 14,336 | 2,013,265,921 | 
| ecrecover_program | 4 | 5 | 6,144 | 2,013,265,921 | 
| ecrecover_program | 4 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 4 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 4 | 8 | 63,332,686 | 2,013,265,921 | 
| ecrecover_program | 5 | 0 | 6,541,320 | 2,013,265,921 | 
| ecrecover_program | 5 | 1 | 19,345,498 | 2,013,265,921 | 
| ecrecover_program | 5 | 2 | 3,270,660 | 2,013,265,921 | 
| ecrecover_program | 5 | 3 | 22,882,398 | 2,013,265,921 | 
| ecrecover_program | 5 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 5 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 5 | 6 | 7,544,970 | 2,013,265,921 | 
| ecrecover_program | 5 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 5 | 8 | 63,709,518 | 2,013,265,921 | 
| ecrecover_program | 6 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 6 | 1 | 19,255,386 | 2,013,265,921 | 
| ecrecover_program | 6 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 6 | 3 | 22,730,846 | 2,013,265,921 | 
| ecrecover_program | 6 | 4 | 14,336 | 2,013,265,921 | 
| ecrecover_program | 6 | 5 | 6,144 | 2,013,265,921 | 
| ecrecover_program | 6 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 6 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 6 | 8 | 63,332,686 | 2,013,265,921 | 
| ecrecover_program | 7 | 0 | 5,902,534 | 2,013,265,921 | 
| ecrecover_program | 7 | 1 | 17,563,968 | 2,013,265,921 | 
| ecrecover_program | 7 | 2 | 2,951,267 | 2,013,265,921 | 
| ecrecover_program | 7 | 3 | 19,986,372 | 2,013,265,921 | 
| ecrecover_program | 7 | 4 | 28,672 | 2,013,265,921 | 
| ecrecover_program | 7 | 5 | 12,288 | 2,013,265,921 | 
| ecrecover_program | 7 | 6 | 8,245,760 | 2,013,265,921 | 
| ecrecover_program | 7 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 7 | 8 | 58,795,053 | 2,013,265,921 | 
| ecrecover_program | 8 | 0 | 6,492,168 | 2,013,265,921 | 
| ecrecover_program | 8 | 1 | 19,263,578 | 2,013,265,921 | 
| ecrecover_program | 8 | 2 | 3,246,084 | 2,013,265,921 | 
| ecrecover_program | 8 | 3 | 22,734,942 | 2,013,265,921 | 
| ecrecover_program | 8 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 8 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 8 | 6 | 7,487,626 | 2,013,265,921 | 
| ecrecover_program | 8 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 8 | 8 | 63,349,070 | 2,013,265,921 | 
| ecrecover_program | 9 | 0 | 6,541,320 | 2,013,265,921 | 
| ecrecover_program | 9 | 1 | 19,345,498 | 2,013,265,921 | 
| ecrecover_program | 9 | 2 | 3,270,660 | 2,013,265,921 | 
| ecrecover_program | 9 | 3 | 22,882,398 | 2,013,265,921 | 
| ecrecover_program | 9 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 9 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 9 | 6 | 7,544,970 | 2,013,265,921 | 
| ecrecover_program | 9 | 7 | 3,145,728 | 2,013,265,921 | 
| ecrecover_program | 9 | 8 | 63,713,614 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/13fecc2480dbcef50e8a0fb5998f44b84e4cd4aa

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15074103953)
