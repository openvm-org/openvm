| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-18.9%])</span> 2.42 | <span style='color: green'>(-0 [-18.7%])</span> 0.74 |
| regex_program | <span style='color: green'>(-1 [-19.1%])</span> 2.38 | <span style='color: green'>(-0 [-19.4%])</span> 0.71 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-141 [-19.1%])</span> 596 | <span style='color: green'>(-564 [-19.1%])</span> 2,384 | <span style='color: green'>(-170 [-19.4%])</span> 707 | <span style='color: green'>(-132 [-19.5%])</span> 545 |
| `main_cells_used     ` | <span style='color: red'>(+14079 [+0.3%])</span> 4,429,800.50 | <span style='color: red'>(+56316 [+0.3%])</span> 17,719,202 | <span style='color: red'>(+41674 [+0.4%])</span> 10,906,308 | <span style='color: red'>(+3046 [+0.1%])</span> 2,188,964 |
| `total_cells_used    ` | <span style='color: red'>(+18631 [+0.1%])</span> 14,622,488.50 | <span style='color: red'>(+74524 [+0.1%])</span> 58,489,954 | <span style='color: red'>(+53426 [+0.2%])</span> 23,448,686 |  11,588,294 |
| `execute_metered_time_ms` |  33 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-3 [-2.5%])</span> 121.05 | -          | <span style='color: green'>(-3 [-2.5%])</span> 121.05 | <span style='color: green'>(-3 [-2.5%])</span> 121.05 |
| `execute_preflight_insns` |  1,027,337.50 |  4,109,350 |  1,108,000 | <span style='color: red'>(+867 [+0.1%])</span> 791,350 |
| `execute_preflight_time_ms` | <span style='color: green'>(-2 [-2.8%])</span> 52.25 | <span style='color: green'>(-6 [-2.8%])</span> 209 | <span style='color: green'>(-1 [-1.5%])</span> 65 | <span style='color: green'>(-2 [-4.4%])</span> 43 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+2 [+5.7%])</span> 32.23 | -          | <span style='color: red'>(+1 [+4.4%])</span> 33.42 | <span style='color: red'>(+1 [+2.8%])</span> 29.08 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-20 [-9.9%])</span> 182.25 | <span style='color: green'>(-80 [-9.9%])</span> 729 | <span style='color: green'>(-22 [-9.2%])</span> 216 | <span style='color: green'>(-20 [-11.2%])</span> 159 |
| `memory_finalize_time_ms` | <span style='color: green'>(-0 [-25.0%])</span> 0.75 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-119 [-28.9%])</span> 292.75 | <span style='color: green'>(-477 [-28.9%])</span> 1,171 | <span style='color: green'>(-131 [-29.6%])</span> 312 | <span style='color: green'>(-103 [-26.8%])</span> 281 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-7.9%])</span> 52.25 | <span style='color: green'>(-18 [-7.9%])</span> 209 | <span style='color: green'>(-5 [-8.1%])</span> 57 | <span style='color: green'>(-5 [-9.3%])</span> 49 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+8.5%])</span> 31.75 | <span style='color: red'>(+10 [+8.5%])</span> 127 | <span style='color: red'>(+16 [+44.4%])</span> 52 | <span style='color: red'>(+2 [+9.5%])</span> 23 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-6 [-7.9%])</span> 65.42 | <span style='color: green'>(-23 [-7.9%])</span> 261.68 | <span style='color: green'>(-7 [-8.5%])</span> 70.99 | <span style='color: green'>(-3 [-5.9%])</span> 55.73 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+3 [+5.6%])</span> 60.09 | <span style='color: red'>(+13 [+5.6%])</span> 240.34 | <span style='color: red'>(+2 [+2.9%])</span> 69.75 | <span style='color: red'>(+4 [+7.7%])</span> 54.12 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-3 [-15.6%])</span> 15.25 | <span style='color: green'>(-11 [-15.6%])</span> 61.01 | <span style='color: green'>(-3 [-14.8%])</span> 17.96 | <span style='color: green'>(-2 [-16.3%])</span> 12.76 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-112 [-63.4%])</span> 64.50 | <span style='color: green'>(-446 [-63.4%])</span> 258 | <span style='color: green'>(-120 [-64.2%])</span> 67 | <span style='color: green'>(-97 [-61.8%])</span> 60 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 58 | 605 | 2,604 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 585 | 41 | 1 | 33 | 4,109,350 | 121.05 | 169 | 

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
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 2 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 3 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | MemoryMerkleAir<8> | 1 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 2 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 3 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 3 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1,024 |  | 8 | 300 | 315,392 | 
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
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 262,144 |  | 52 | 36 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 16,384 |  | 40 | 37 | 1,261,568 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 256 |  | 72 | 59 | 33,536 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 8,192 |  | 28 | 20 | 393,216 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 216 | 707 | 23,448,686 | 150,778,428 | 216 | 296 | 0 | 61.40 | 17.96 | 5 | 70.99 | 60 | 98 | 60 | 3 | 57 | 10,906,308 | 26 | 44 | 1,102,000 | 29.08 | 21 | 80 | 1 | 60 | 
| regex_program | 1 | 159 | 547 | 11,597,248 | 132,682,794 | 159 | 282 | 1 | 55.08 | 15.31 | 4 | 68.83 | 66 | 92 | 66 | 0 | 51 | 2,197,502 | 23 | 65 | 1,108,000 | 33.02 | 17 | 70 | 0 | 66 | 
| regex_program | 2 | 166 | 545 | 11,588,294 | 132,682,794 | 166 | 281 | 1 | 54.12 | 14.98 | 4 | 66.13 | 65 | 92 | 65 | 0 | 52 | 2,188,964 | 26 | 57 | 1,108,000 | 33.42 | 17 | 69 | 0 | 65 | 
| regex_program | 3 | 188 | 585 | 11,855,726 | 103,522,954 | 188 | 312 | 0 | 69.75 | 12.76 | 5 | 55.73 | 67 | 111 | 67 | 0 | 49 | 2,426,428 | 52 | 43 | 791,350 | 33.40 | 18 | 83 | 1 | 67 | 

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
| regex_program | 1 | 0 | 2,916,356 | 2,013,265,921 | 
| regex_program | 1 | 1 | 8,032,256 | 2,013,265,921 | 
| regex_program | 1 | 2 | 1,458,178 | 2,013,265,921 | 
| regex_program | 1 | 3 | 9,668,612 | 2,013,265,921 | 
| regex_program | 1 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 1 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,309,568 | 2,013,265,921 | 
| regex_program | 1 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 1 | 8 | 26,506,250 | 2,013,265,921 | 
| regex_program | 2 | 0 | 2,916,356 | 2,013,265,921 | 
| regex_program | 2 | 1 | 8,032,256 | 2,013,265,921 | 
| regex_program | 2 | 2 | 1,458,178 | 2,013,265,921 | 
| regex_program | 2 | 3 | 9,668,612 | 2,013,265,921 | 
| regex_program | 2 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 2 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 2 | 6 | 3,309,568 | 2,013,265,921 | 
| regex_program | 2 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 2 | 8 | 26,506,250 | 2,013,265,921 | 
| regex_program | 3 | 0 | 2,162,820 | 2,013,265,921 | 
| regex_program | 3 | 1 | 6,007,808 | 2,013,265,921 | 
| regex_program | 3 | 2 | 1,081,410 | 2,013,265,921 | 
| regex_program | 3 | 3 | 7,511,140 | 2,013,265,921 | 
| regex_program | 3 | 4 | 8,192 | 2,013,265,921 | 
| regex_program | 3 | 5 | 4,096 | 2,013,265,921 | 
| regex_program | 3 | 6 | 1,904,960 | 2,013,265,921 | 
| regex_program | 3 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 3 | 8 | 19,796,586 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/6dfbfe08160a97d5ed045bd2943d97accaadc38f

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/18026041009)
