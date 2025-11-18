| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+4 [+175.4%])</span> 6.59 | <span style='color: red'>(+1 [+142.5%])</span> 1.89 |
| regex_program | <span style='color: red'>(+4 [+178.1%])</span> 6.56 | <span style='color: red'>(+1 [+149.7%])</span> 1.86 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1050 [+178.1%])</span> 1,640.25 | <span style='color: red'>(+4202 [+178.1%])</span> 6,561 | <span style='color: red'>(+1115 [+149.7%])</span> 1,860 | <span style='color: red'>(+747 [+139.1%])</span> 1,284 |
| `main_cells_used     ` | <span style='color: red'>(+6528 [+0.1%])</span> 4,430,331.50 | <span style='color: red'>(+26110 [+0.1%])</span> 17,721,326 | <span style='color: red'>(+16126 [+0.1%])</span> 10,915,296 | <span style='color: red'>(+13832 [+0.6%])</span> 2,191,166 |
| `total_cells_used    ` |  14,629,907.50 |  58,519,630 |  23,469,282 | <span style='color: red'>(+17872 [+0.2%])</span> 11,596,936 |
| `execute_metered_time_ms` | <span style='color: green'>(-2 [-5.6%])</span> 34 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+6 [+5.1%])</span> 119.80 | -          | <span style='color: red'>(+6 [+5.1%])</span> 119.80 | <span style='color: red'>(+6 [+5.1%])</span> 119.80 |
| `execute_preflight_insns` | <span style='color: green'>(-13819 [-1.3%])</span> 1,020,556.75 | <span style='color: green'>(-55275 [-1.3%])</span> 4,082,227 | <span style='color: red'>(+9000 [+0.8%])</span> 1,113,000 | <span style='color: green'>(-80275 [-9.7%])</span> 746,227 |
| `execute_preflight_time_ms` | <span style='color: red'>(+1218 [+2266.0%])</span> 1,271.75 | <span style='color: red'>(+4872 [+2266.0%])</span> 5,087 | <span style='color: red'>(+1326 [+2040.0%])</span> 1,391 | <span style='color: red'>(+885 [+1966.7%])</span> 930 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-30 [-97.3%])</span> 0.82 | -          | <span style='color: green'>(-31 [-97.4%])</span> 0.82 | <span style='color: green'>(-27 [-97.1%])</span> 0.81 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-158 [-89.2%])</span> 19.25 | <span style='color: green'>(-633 [-89.2%])</span> 77 | <span style='color: green'>(-167 [-79.1%])</span> 44 | <span style='color: green'>(-144 [-94.1%])</span> 9 |
| `memory_finalize_time_ms` |  1 |  4 |  4 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-9 [-3.1%])</span> 282 | <span style='color: green'>(-36 [-3.1%])</span> 1,128 | <span style='color: green'>(-45 [-13.2%])</span> 297 | <span style='color: green'>(-19 [-7.0%])</span> 251 |
| `main_trace_commit_time_ms` |  51.75 |  207 | <span style='color: green'>(-1 [-1.8%])</span> 56 |  50 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-18.8%])</span> 27 | <span style='color: green'>(-25 [-18.8%])</span> 108 | <span style='color: green'>(-18 [-33.3%])</span> 36 | <span style='color: green'>(-1 [-5.0%])</span> 19 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.7%])</span> 63.98 | <span style='color: green'>(-2 [-0.7%])</span> 255.93 | <span style='color: green'>(-2 [-2.8%])</span> 68.91 | <span style='color: green'>(-0 [-0.7%])</span> 55.41 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-0 [-0.5%])</span> 58.13 | <span style='color: green'>(-1 [-0.5%])</span> 232.50 | <span style='color: green'>(-1 [-1.7%])</span> 68.52 | <span style='color: green'>(-1 [-1.2%])</span> 52.31 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+4.4%])</span> 15.30 | <span style='color: red'>(+3 [+4.4%])</span> 61.22 | <span style='color: red'>(+1 [+5.7%])</span> 17.67 | <span style='color: red'>(+0 [+2.0%])</span> 13.12 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-3.1%])</span> 63 | <span style='color: green'>(-8 [-3.1%])</span> 252 | <span style='color: red'>(+4 [+4.9%])</span> 86 | <span style='color: green'>(-9 [-16.1%])</span> 47 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 612 | 6,779 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 1,284 | 40 | 1 | 34 | 4,082,227 | 119.80 | 167 | 

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

| group | air_name | segment | rows_used | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 69,458 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 582 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 2 | 584 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 3 | 966 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 3 |  | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 70,578 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | MemoryMerkleAir<8> | 1 | 912 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 2 | 922 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 3 | 1,628 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 69,458 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 582 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 584 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 3 | 966 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PhantomAir | 0 |  | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 13,969 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 832 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 813 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 1,448 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 97,864 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 97,864 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 2 | 97,864 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 3 | 97,864 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 2 | 524,288 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 3 | 524,288 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 |  | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 2 | 262,144 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 3 | 262,144 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 |  | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 |  | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 |  | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 |  | 262,144 |  | 52 | 36 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 |  | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 |  | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 |  | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 |  | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 |  | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 |  | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 |  | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 |  | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 |  | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 |  | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 |  | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 |  | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 |  | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 |  | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 |  | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 |  | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 |  | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 |  | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 |  | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 |  | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 |  | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 |  | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 |  | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 |  | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 |  | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 |  | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 |  | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 |  | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 |  | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 |  | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 |  | 256 |  | 72 | 59 | 33,536 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 |  | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 |  | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 |  | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 |  | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 |  | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 |  | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 |  | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 |  | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 |  | 8,192 |  | 28 | 20 | 393,216 | 
| regex_program | VmConnectorAir | 0 | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 2 | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 3 | 2 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 44 | 1,860 | 23,469,282 | 150,778,428 | 44 | 294 | 0 | 58.82 | 17.67 | 5 | 68.91 | 54 | 105 | 54 | 4 | 56 | 10,915,296 | 36 | 1,378 | 1,110,000 | 0.81 | 21 | 77 | 1 | 54 | 
| regex_program | 1 | 9 | 1,727 | 11,602,178 | 132,052,010 | 9 | 286 | 1 | 52.31 | 15.64 | 4 | 66.01 | 65 | 101 | 64 | 0 | 51 | 2,196,472 | 34 | 1,391 | 1,113,000 | 0.82 | 17 | 68 | 0 | 64 | 
| regex_program | 2 | 9 | 1,690 | 11,596,936 | 132,052,010 | 9 | 251 | 1 | 52.86 | 14.79 | 4 | 65.60 | 47 | 85 | 47 | 0 | 50 | 2,191,166 | 19 | 1,388 | 1,113,000 | 0.82 | 16 | 68 | 0 | 47 | 
| regex_program | 3 | 15 | 1,284 | 11,851,234 | 103,456,394 | 15 | 297 | 0 | 68.52 | 13.12 | 5 | 55.41 | 86 | 77 | 86 | 0 | 50 | 2,418,392 | 19 | 930 | 746,227 | 0.82 | 18 | 82 | 1 | 86 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 2,870,278 | 2,013,265,921 | 
| regex_program | 0 | 1 | 8,479,744 | 2,013,265,921 | 
| regex_program | 0 | 2 | 1,435,139 | 2,013,265,921 | 
| regex_program | 0 | 3 | 9,728,004 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 3,302,144 | 2,013,265,921 | 
| regex_program | 0 | 7 | 69,632 | 2,013,265,921 | 
| regex_program | 0 | 8 | 27,736,333 | 2,013,265,921 | 
| regex_program | 1 | 0 | 2,899,972 | 2,013,265,921 | 
| regex_program | 1 | 1 | 7,983,104 | 2,013,265,921 | 
| regex_program | 1 | 2 | 1,449,986 | 2,013,265,921 | 
| regex_program | 1 | 3 | 9,619,460 | 2,013,265,921 | 
| regex_program | 1 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 1 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 1 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 1 | 8 | 26,358,794 | 2,013,265,921 | 
| regex_program | 2 | 0 | 2,899,972 | 2,013,265,921 | 
| regex_program | 2 | 1 | 7,983,104 | 2,013,265,921 | 
| regex_program | 2 | 2 | 1,449,986 | 2,013,265,921 | 
| regex_program | 2 | 3 | 9,619,460 | 2,013,265,921 | 
| regex_program | 2 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 2 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 2 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 2 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 2 | 8 | 26,358,794 | 2,013,265,921 | 
| regex_program | 3 | 0 | 2,162,820 | 2,013,265,921 | 
| regex_program | 3 | 1 | 6,003,712 | 2,013,265,921 | 
| regex_program | 3 | 2 | 1,081,410 | 2,013,265,921 | 
| regex_program | 3 | 3 | 7,509,092 | 2,013,265,921 | 
| regex_program | 3 | 4 | 7,168 | 2,013,265,921 | 
| regex_program | 3 | 5 | 3,072 | 2,013,265,921 | 
| regex_program | 3 | 6 | 1,904,960 | 2,013,265,921 | 
| regex_program | 3 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 3 | 8 | 19,788,394 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c/regex-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/24aba0b7fef8ee4b02c8f8b6668e414c76f3c16c

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/19482381865)
