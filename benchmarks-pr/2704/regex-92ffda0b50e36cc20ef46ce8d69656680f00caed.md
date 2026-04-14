| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: green'>(-1 [-5.9%])</span> 16.54 | <span style='color: red'>(+0 [+3.1%])</span> 5.02 | 5.02 |
| regex_program | <span style='color: green'>(-1 [-5.9%])</span> 16.54 | <span style='color: red'>(+0 [+3.1%])</span> 5.02 |  5.02 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-260 [-5.9%])</span> 4,127 | <span style='color: green'>(-1041 [-5.9%])</span> 16,508 | <span style='color: red'>(+149 [+3.1%])</span> 4,986 | <span style='color: green'>(-255 [-7.1%])</span> 3,318 |
| `main_cells_used     ` |  4,425,166.50 |  17,700,666 |  10,894,092 |  2,182,338 |
| `total_cells_used    ` |  14,620,978.50 |  58,483,914 |  23,441,006 |  11,584,340 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+2.9%])</span> 36 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-1 [-1.0%])</span> 113.69 | -          | <span style='color: green'>(-1 [-1.0%])</span> 113.69 | <span style='color: green'>(-1 [-1.0%])</span> 113.69 |
| `execute_preflight_insns` |  1,034,284.50 |  4,137,138 |  1,104,000 |  826,138 |
| `execute_preflight_time_ms` | <span style='color: green'>(-1 [-1.6%])</span> 45 | <span style='color: green'>(-3 [-1.6%])</span> 180 | <span style='color: green'>(-1 [-1.8%])</span> 55 | <span style='color: green'>(-1 [-2.9%])</span> 34 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+1 [+2.0%])</span> 31.93 | -          | <span style='color: red'>(+1 [+2.6%])</span> 32.63 | <span style='color: red'>(+1 [+3.3%])</span> 31.04 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.7%])</span> 177.25 | <span style='color: green'>(-5 [-0.7%])</span> 709 | <span style='color: green'>(-1 [-0.5%])</span> 192 | <span style='color: red'>(+1 [+0.6%])</span> 171 |
| `memory_finalize_time_ms` |  1 |  4 |  4 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-259 [-6.3%])</span> 3,830.25 | <span style='color: green'>(-1037 [-6.3%])</span> 15,321 | <span style='color: red'>(+151 [+3.3%])</span> 4,725 | <span style='color: green'>(-253 [-7.6%])</span> 3,057 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.4%])</span> 51 | <span style='color: green'>(-3 [-1.4%])</span> 204 | <span style='color: green'>(-3 [-5.2%])</span> 55 |  49 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-37 [-46.8%])</span> 41.75 | <span style='color: green'>(-147 [-46.8%])</span> 167 | <span style='color: green'>(-43 [-41.3%])</span> 61 | <span style='color: green'>(-9 [-25.7%])</span> 26 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.3%])</span> 63.50 | <span style='color: green'>(-1 [-0.3%])</span> 254.01 | <span style='color: red'>(+1 [+1.6%])</span> 70.39 | <span style='color: red'>(+1 [+0.9%])</span> 55.23 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-1.5%])</span> 58.01 | <span style='color: green'>(-4 [-1.5%])</span> 232.03 | <span style='color: green'>(-0 [-0.6%])</span> 68.75 | <span style='color: red'>(+1 [+1.0%])</span> 52.38 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-3.9%])</span> 15.18 | <span style='color: green'>(-2 [-3.9%])</span> 60.72 | <span style='color: green'>(-1 [-4.6%])</span> 17.60 | <span style='color: green'>(-0 [-0.7%])</span> 13.19 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-220 [-5.8%])</span> 3,597 | <span style='color: green'>(-881 [-5.8%])</span> 14,388 | <span style='color: red'>(+210 [+4.9%])</span> 4,504 | <span style='color: green'>(-243 [-7.9%])</span> 2,846 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 535 | 16,558 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| regex_program | 4,665 | 1 | 36 | 4,137,138 | 113.69 | 0 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 10 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 14 | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 26 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | 28 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 3 | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | 6 | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | 9 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 256 |  | 72 | 59 | 33,536 | 
| regex_program | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 1 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | 28 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 3 | MemoryMerkleAir<8> | 1 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | 6 | AccessAdapterAir<8> | 1 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | 0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 1 | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 2 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | 28 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 3 | MemoryMerkleAir<8> | 2 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | 6 | AccessAdapterAir<8> | 2 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | 0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 1 | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 13 | KeccakVmAir | 3 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 8,192 |  | 28 | 20 | 393,216 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 3 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 262,144 |  | 52 | 36 | 23,068,672 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | 28 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 3 | MemoryMerkleAir<8> | 3 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | 6 | AccessAdapterAir<8> | 3 | 1,024 |  | 16 | 17 | 33,792 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 171 | 3,539 | 23,441,006 | 150,778,428 | 171 | 3,142 | 0 | 58.14 | 17.60 | 7 | 70.39 | 2,876 | 133 | 2,875 | 4 | 55 | 10,894,092 | 61 | 55 | 1,103,000 | 31.04 | 21 | 76 | 0 | 2,875 | 
| regex_program | 1 | 173 | 4,986 | 11,593,696 | 128,513,066 | 173 | 4,725 | 1 | 52.38 | 14.82 | 6 | 63.80 | 4,504 | 103 | 4,504 | 0 | 49 | 2,191,510 | 39 | 45 | 1,104,000 | 32.63 | 16 | 67 | 0 | 4,504 | 
| regex_program | 2 | 173 | 3,318 | 11,584,340 | 128,513,066 | 173 | 3,057 | 1 | 52.77 | 15.11 | 6 | 64.59 | 2,846 | 91 | 2,845 | 0 | 50 | 2,182,338 | 26 | 46 | 1,104,000 | 31.69 | 16 | 68 | 0 | 2,845 | 
| regex_program | 3 | 192 | 4,665 | 11,864,872 | 103,456,394 | 192 | 4,397 | 0 | 68.75 | 13.19 | 7 | 55.23 | 4,162 | 101 | 4,161 | 0 | 50 | 2,432,726 | 41 | 34 | 826,138 | 32.36 | 19 | 83 | 0 | 4,161 | 

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
| regex_program | 1 | 0 | 2,768,900 | 2,013,265,921 | 
| regex_program | 1 | 1 | 7,720,960 | 2,013,265,921 | 
| regex_program | 1 | 2 | 1,384,450 | 2,013,265,921 | 
| regex_program | 1 | 3 | 9,357,316 | 2,013,265,921 | 
| regex_program | 1 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 1 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 1 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 1 | 8 | 25,637,898 | 2,013,265,921 | 
| regex_program | 2 | 0 | 2,768,900 | 2,013,265,921 | 
| regex_program | 2 | 1 | 7,720,960 | 2,013,265,921 | 
| regex_program | 2 | 2 | 1,384,450 | 2,013,265,921 | 
| regex_program | 2 | 3 | 9,357,316 | 2,013,265,921 | 
| regex_program | 2 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 2 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 2 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 2 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 2 | 8 | 25,637,898 | 2,013,265,921 | 
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


Commit: https://github.com/openvm-org/openvm/commit/92ffda0b50e36cc20ef46ce8d69656680f00caed

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24377793752)
