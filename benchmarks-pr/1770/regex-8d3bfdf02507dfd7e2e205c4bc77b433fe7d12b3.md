| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+8.8%])</span> 8.57 | <span style='color: green'>(-1 [-21.9%])</span> 3.67 |
| regex_program | <span style='color: red'>(+1 [+8.8%])</span> 8.57 | <span style='color: green'>(-1 [-22.4%])</span> 3.41 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1795 [-45.6%])</span> 2,143.75 | <span style='color: red'>(+697 [+8.8%])</span> 8,575 | <span style='color: green'>(-983 [-22.4%])</span> 3,408 | <span style='color: green'>(-2316 [-66.4%])</span> 1,171 |
| `main_cells_used     ` | <span style='color: green'>(-40740796 [-48.9%])</span> 42,514,780 | <span style='color: red'>(+3547968 [+2.1%])</span> 170,059,120 | <span style='color: green'>(-6244620 [-6.7%])</span> 87,256,179 | <span style='color: green'>(-65142187 [-89.2%])</span> 7,868,166 |
| `total_cycles        ` | <span style='color: green'>(-1041255 [-50.0%])</span> 1,041,358 |  4,165,432 |  2,241,800 | <span style='color: green'>(-1804211 [-93.9%])</span> 117,300 |
| `execute_metered_time_ms` |  337 |  337 |  337 |  337 |
| `execute_time_ms     ` | <span style='color: green'>(-212 [-59.9%])</span> 142.50 | <span style='color: green'>(-140 [-19.7%])</span> 570 | <span style='color: green'>(-91 [-23.0%])</span> 304 | <span style='color: green'>(-296 [-94.0%])</span> 19 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-604 [-62.6%])</span> 360.25 | <span style='color: green'>(-487 [-25.3%])</span> 1,441 | <span style='color: green'>(-662 [-57.0%])</span> 499 | <span style='color: green'>(-504 [-65.7%])</span> 263 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-979 [-37.4%])</span> 1,641 | <span style='color: red'>(+1324 [+25.3%])</span> 6,564 | <span style='color: green'>(-230 [-8.1%])</span> 2,605 | <span style='color: green'>(-1537 [-63.9%])</span> 868 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-216 [-42.9%])</span> 287 | <span style='color: red'>(+143 [+14.2%])</span> 1,148 | <span style='color: green'>(-54 [-9.7%])</span> 503 | <span style='color: green'>(-335 [-74.8%])</span> 113 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-110 [-49.2%])</span> 113.50 | <span style='color: red'>(+7 [+1.6%])</span> 454 | <span style='color: red'>(+4 [+1.7%])</span> 236 | <span style='color: green'>(-190 [-88.4%])</span> 25 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-239 [-41.8%])</span> 331.75 | <span style='color: red'>(+186 [+16.3%])</span> 1,327 | <span style='color: green'>(-63 [-10.1%])</span> 558 | <span style='color: green'>(-397 [-76.3%])</span> 123 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-112 [-43.1%])</span> 148.25 | <span style='color: red'>(+72 [+13.8%])</span> 593 | <span style='color: green'>(-23 [-8.1%])</span> 261 | <span style='color: green'>(-173 [-73.0%])</span> 64 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-74 [-30.0%])</span> 171.50 | <span style='color: red'>(+196 [+40.0%])</span> 686 | <span style='color: green'>(-37 [-13.1%])</span> 245 | <span style='color: green'>(-97 [-46.6%])</span> 111 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-228 [-28.1%])</span> 582 | <span style='color: red'>(+709 [+43.8%])</span> 2,328 | <span style='color: green'>(-54 [-6.4%])</span> 796 | <span style='color: green'>(-372 [-48.4%])</span> 397 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | fri.log_blowup | execute_metered_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- | --- | --- |
| regex_program | 4 | 588 | 1 | 337 | 19 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,513 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| regex_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| regex_program | AccessAdapterAir<8> | 1 | 65,536 |  | 16 | 17 | 2,162,688 | 
| regex_program | AccessAdapterAir<8> | 2 | 32,768 |  | 16 | 17 | 1,081,344 | 
| regex_program | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 3 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 65,536 |  | 16 | 32 | 3,145,728 | 
| regex_program | MemoryMerkleAir<8> | 1 | 65,536 |  | 16 | 32 | 3,145,728 | 
| regex_program | MemoryMerkleAir<8> | 2 | 65,536 |  | 16 | 32 | 3,145,728 | 
| regex_program | MemoryMerkleAir<8> | 3 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 65,536 |  | 12 | 20 | 2,097,152 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 65,536 |  | 12 | 20 | 2,097,152 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 32,768 |  | 12 | 20 | 1,048,576 | 
| regex_program | PersistentBoundaryAir<8> | 3 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 4,096 |  | 8 | 300 | 1,261,568 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 32,768 |  | 52 | 36 | 2,883,584 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1,024 |  | 40 | 37 | 78,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 256 |  | 40 | 37 | 19,712 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 32,768 |  | 40 | 37 | 2,523,136 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 16,384 |  | 40 | 37 | 1,261,568 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 16,384 |  | 52 | 53 | 1,720,320 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 16,384 |  | 28 | 26 | 884,736 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 28 | 26 | 14,155,776 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 16,384 |  | 32 | 32 | 1,048,576 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2,048 |  | 32 | 32 | 131,072 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 2,048 |  | 36 | 28 | 131,072 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 4 |  | 52 | 36 | 352 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 1 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 64 |  | 72 | 39 | 7,104 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 16 |  | 72 | 39 | 1,776 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 128 |  | 52 | 31 | 10,624 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 1,024 |  | 52 | 31 | 84,992 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 1,024 |  | 28 | 20 | 49,152 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 263 | 1,192 | 150,300 | 38,535,548 | 906 | 69 | 127 | 123 | 406 | 146 | 11,939,181 | 28 | 23 | 
| regex_program | 1 | 284 | 1,171 | 117,300 | 30,898,972 | 868 | 64 | 111 | 153 | 397 | 113 | 7,868,166 | 25 | 19 | 
| regex_program | 2 | 499 | 3,408 | 2,241,800 | 260,949,804 | 2,605 | 261 | 245 | 558 | 796 | 503 | 87,256,179 | 236 | 304 | 
| regex_program | 3 | 395 | 2,804 | 1,656,032 | 195,953,308 | 2,185 | 199 | 203 | 493 | 729 | 386 | 62,995,594 | 165 | 224 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 389,510 | 2,013,265,921 | 
| regex_program | 0 | 1 | 1,332,352 | 2,013,265,921 | 
| regex_program | 0 | 2 | 194,755 | 2,013,265,921 | 
| regex_program | 0 | 3 | 1,362,052 | 2,013,265,921 | 
| regex_program | 0 | 4 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 5 | 131,072 | 2,013,265,921 | 
| regex_program | 0 | 6 | 447,552 | 2,013,265,921 | 
| regex_program | 0 | 7 | 1,024 | 2,013,265,921 | 
| regex_program | 0 | 8 | 5,185,421 | 2,013,265,921 | 
| regex_program | 1 | 0 | 279,822 | 2,013,265,921 | 
| regex_program | 1 | 1 | 1,048,344 | 2,013,265,921 | 
| regex_program | 1 | 2 | 139,911 | 2,013,265,921 | 
| regex_program | 1 | 3 | 1,138,472 | 2,013,265,921 | 
| regex_program | 1 | 4 | 262,144 | 2,013,265,921 | 
| regex_program | 1 | 5 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 6 | 248,064 | 2,013,265,921 | 
| regex_program | 1 | 7 | 7,168 | 2,013,265,921 | 
| regex_program | 1 | 8 | 4,307,669 | 2,013,265,921 | 
| regex_program | 2 | 0 | 5,832,742 | 2,013,265,921 | 
| regex_program | 2 | 1 | 16,187,488 | 2,013,265,921 | 
| regex_program | 2 | 2 | 2,916,371 | 2,013,265,921 | 
| regex_program | 2 | 3 | 19,398,756 | 2,013,265,921 | 
| regex_program | 2 | 4 | 229,376 | 2,013,265,921 | 
| regex_program | 2 | 5 | 98,304 | 2,013,265,921 | 
| regex_program | 2 | 6 | 6,619,152 | 2,013,265,921 | 
| regex_program | 2 | 7 | 131,200 | 2,013,265,921 | 
| regex_program | 2 | 8 | 52,466,061 | 2,013,265,921 | 
| regex_program | 3 | 0 | 4,325,510 | 2,013,265,921 | 
| regex_program | 3 | 1 | 12,004,352 | 2,013,265,921 | 
| regex_program | 3 | 2 | 2,162,755 | 2,013,265,921 | 
| regex_program | 3 | 3 | 15,015,012 | 2,013,265,921 | 
| regex_program | 3 | 4 | 8,192 | 2,013,265,921 | 
| regex_program | 3 | 5 | 4,096 | 2,013,265,921 | 
| regex_program | 3 | 6 | 3,805,504 | 2,013,265,921 | 
| regex_program | 3 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 3 | 8 | 38,507,117 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/8d3bfdf02507dfd7e2e205c4bc77b433fe7d12b3

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15780331818)
