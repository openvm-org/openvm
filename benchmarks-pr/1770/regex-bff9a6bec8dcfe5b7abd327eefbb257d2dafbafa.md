| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  8.90 |  3.81 |
| regex_program |  8.53 |  3.45 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,133.25 |  8,533 |  3,447 |  1,133 |
| `main_cells_used     ` |  42,514,780 |  170,059,120 |  87,256,179 |  7,868,166 |
| `total_cycles        ` |  1,041,358 |  4,165,432 |  2,241,800 |  117,300 |
| `execute_metered_time_ms` |  364 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  11.43 | -          | -          | -          |
| `execute_e3_time_ms  ` |  143.75 |  575 |  307 |  20 |
| `execute_e3_insn_mi/s` |  6.66 | -          |  7.38 |  5.76 |
| `trace_gen_time_ms   ` |  363.25 |  1,453 |  498 |  263 |
| `memory_finalize_time_ms` |  204.25 |  817 |  229 |  169 |
| `boundary_finalize_time_ms` |  1.75 |  7 |  3 |  0 |
| `merkle_finalize_time_ms` |  190.75 |  763 |  212 |  161 |
| `stark_prove_excluding_trace_time_ms` |  1,626.25 |  6,505 |  2,642 |  826 |
| `main_trace_commit_time_ms` |  292 |  1,168 |  519 |  112 |
| `generate_perm_trace_time_ms` |  110.50 |  442 |  230 |  28 |
| `perm_trace_commit_time_ms` |  309.75 |  1,239 |  559 |  113 |
| `quotient_poly_compute_time_ms` |  156.75 |  627 |  277 |  62 |
| `quotient_poly_commit_time_ms` |  169.75 |  679 |  246 |  107 |
| `pcs_opening_time_ms ` |  581.50 |  2,326 |  807 |  397 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | commit_exe_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 4 | 584 | 4,165,433 | 1 | 364 | 11.43 | 19 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 263 | 1,199 | 150,300 | 38,535,548 | 912 | 73 | 124 | 122 | 411 | 185 | 201 | 148 | 11,939,181 | 150,300 | 30 | 24 | 6.20 | 2 | 
| regex_program | 1 | 287 | 1,133 | 117,300 | 30,898,972 | 826 | 62 | 107 | 113 | 397 | 212 | 229 | 112 | 7,868,166 | 117,300 | 28 | 20 | 5.76 | 3 | 
| regex_program | 2 | 498 | 3,447 | 2,241,800 | 260,949,804 | 2,642 | 277 | 246 | 559 | 807 | 205 | 218 | 519 | 87,256,179 | 2,241,800 | 230 | 307 | 7.28 | 2 | 
| regex_program | 3 | 405 | 2,754 | 1,656,032 | 195,953,308 | 2,125 | 215 | 202 | 445 | 711 | 161 | 169 | 389 | 62,995,594 | 1,656,033 | 154 | 224 | 7.38 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/bff9a6bec8dcfe5b7abd327eefbb257d2dafbafa

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15781954587)
