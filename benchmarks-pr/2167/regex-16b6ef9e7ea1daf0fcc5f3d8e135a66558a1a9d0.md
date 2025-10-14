| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+1.5%])</span> 2.40 | <span style='color: red'>(+0 [+3.3%])</span> 0.78 |
| regex_program | <span style='color: red'>(+0 [+1.5%])</span> 2.36 | <span style='color: red'>(+0 [+3.3%])</span> 0.74 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+9 [+1.5%])</span> 590.50 | <span style='color: red'>(+35 [+1.5%])</span> 2,362 | <span style='color: red'>(+24 [+3.3%])</span> 742 | <span style='color: red'>(+12 [+2.3%])</span> 534 |
| `main_cells_used     ` |  4,423,804 |  17,695,216 |  10,899,170 |  2,177,334 |
| `total_cells_used    ` |  14,619,340 |  58,477,360 |  23,446,148 |  11,579,064 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+3.0%])</span> 34 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-3 [-2.7%])</span> 118.76 | -          | <span style='color: green'>(-3 [-2.7%])</span> 118.76 | <span style='color: green'>(-3 [-2.7%])</span> 118.76 |
| `execute_preflight_insns` |  1,034,375.50 |  4,137,502 |  1,104,000 |  826,502 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+0.5%])</span> 54.50 | <span style='color: red'>(+1 [+0.5%])</span> 218 | <span style='color: green'>(-2 [-2.9%])</span> 67 | <span style='color: red'>(+2 [+4.5%])</span> 46 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-1 [-4.3%])</span> 30.59 | -          | <span style='color: green'>(-2 [-5.0%])</span> 31.67 | <span style='color: green'>(-1 [-1.8%])</span> 27.92 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-0.8%])</span> 178.50 | <span style='color: green'>(-6 [-0.8%])</span> 714 | <span style='color: green'>(-1 [-0.5%])</span> 214 | <span style='color: red'>(+1 [+0.6%])</span> 155 |
| `memory_finalize_time_ms` | <span style='color: green'>(-0 [-25.0%])</span> 0.75 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+10 [+3.7%])</span> 288.75 | <span style='color: red'>(+41 [+3.7%])</span> 1,155 | <span style='color: red'>(+26 [+8.5%])</span> 333 | <span style='color: red'>(+6 [+2.3%])</span> 263 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-0 [-1.0%])</span> 51 | <span style='color: green'>(-2 [-1.0%])</span> 204 | <span style='color: green'>(-1 [-1.8%])</span> 56 |  49 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+10 [+41.8%])</span> 34.75 | <span style='color: red'>(+41 [+41.8%])</span> 139 | <span style='color: red'>(+26 [+89.7%])</span> 55 | <span style='color: red'>(+1 [+5.6%])</span> 19 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-2 [-2.3%])</span> 63.49 | <span style='color: green'>(-6 [-2.3%])</span> 253.97 | <span style='color: green'>(-4 [-5.5%])</span> 67.55 | <span style='color: green'>(-1 [-1.0%])</span> 54.89 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-4.1%])</span> 56.86 | <span style='color: green'>(-10 [-4.1%])</span> 227.43 | <span style='color: red'>(+0 [+0.5%])</span> 70.49 | <span style='color: green'>(-2 [-3.4%])</span> 51.58 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-0 [-1.2%])</span> 14.83 | <span style='color: green'>(-1 [-1.2%])</span> 59.33 | <span style='color: red'>(+0 [+0.1%])</span> 16.67 | <span style='color: green'>(-0 [-3.5%])</span> 12.70 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+4 [+7.5%])</span> 64.50 | <span style='color: red'>(+18 [+7.5%])</span> 258 | <span style='color: red'>(+11 [+15.1%])</span> 84 | <span style='color: green'>(-1 [-1.9%])</span> 51 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 611 | 2,607 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 534 | 67 | 1 | 34 | 4,137,502 | 118.76 | 194 | 

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
| regex_program | AccessAdapterAir<8> | 3 | 1,024 |  | 16 | 17 | 33,792 | 
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
| regex_program | PersistentBoundaryAir<8> | 3 | 1,024 |  | 12 | 20 | 32,768 | 
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
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 65,536 |  | 28 | 26 | 3,538,944 | 
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
| regex_program | 0 | 214 | 742 | 23,446,148 | 150,778,428 | 214 | 333 | 0 | 51.58 | 16.67 | 5 | 67.55 | 84 | 123 | 84 | 3 | 56 | 10,899,170 | 55 | 46 | 1,103,000 | 27.92 | 21 | 68 | 1 | 84 | 
| regex_program | 1 | 155 | 548 | 11,598,276 | 128,513,066 | 154 | 284 | 1 | 52.59 | 14.83 | 4 | 65.39 | 62 | 103 | 62 | 0 | 50 | 2,196,106 | 37 | 67 | 1,104,000 | 31.67 | 16 | 67 | 0 | 62 | 
| regex_program | 2 | 162 | 538 | 11,579,064 | 128,513,066 | 162 | 275 | 1 | 52.77 | 15.13 | 4 | 66.13 | 61 | 95 | 61 | 0 | 49 | 2,177,334 | 28 | 59 | 1,104,000 | 31.44 | 16 | 68 | 0 | 61 | 
| regex_program | 3 | 183 | 534 | 11,853,872 | 103,456,394 | 182 | 263 | 0 | 70.49 | 12.70 | 5 | 54.89 | 51 | 77 | 51 | 0 | 49 | 2,422,606 | 19 | 46 | 826,502 | 31.34 | 18 | 84 | 1 | 51 | 

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


Commit: https://github.com/openvm-org/openvm/commit/16b6ef9e7ea1daf0fcc5f3d8e135a66558a1a9d0

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/18510136279)
