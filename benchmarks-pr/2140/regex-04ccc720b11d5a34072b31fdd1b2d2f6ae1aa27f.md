| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-2.2%])</span> 3.06 | <span style='color: green'>(-0 [-0.9%])</span> 0.90 |
| regex_program | <span style='color: green'>(-0 [-2.1%])</span> 3.02 | <span style='color: green'>(-0 [-0.5%])</span> 0.87 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-16 [-2.1%])</span> 756 | <span style='color: green'>(-64 [-2.1%])</span> 3,024 | <span style='color: green'>(-4 [-0.5%])</span> 866 | <span style='color: green'>(-33 [-4.6%])</span> 679 |
| `main_cells_used     ` |  4,415,721.50 |  17,662,886 |  10,864,634 |  2,185,918 |
| `total_cells_used    ` |  14,603,857.50 |  58,415,430 |  23,395,260 |  11,583,256 |
| `execute_metered_time_ms` | <span style='color: green'>(-4 [-11.1%])</span> 32 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+13 [+12.1%])</span> 124.64 | -          | <span style='color: red'>(+13 [+12.1%])</span> 124.64 | <span style='color: red'>(+13 [+12.1%])</span> 124.64 |
| `execute_preflight_insns` |  1,027,120.75 |  4,108,483 |  1,108,000 |  790,483 |
| `execute_preflight_time_ms` | <span style='color: red'>(+1 [+1.4%])</span> 54.75 | <span style='color: red'>(+3 [+1.4%])</span> 219 |  67 | <span style='color: red'>(+1 [+2.3%])</span> 45 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-1 [-2.7%])</span> 30.07 | -          | <span style='color: green'>(-1 [-2.3%])</span> 31.39 | <span style='color: green'>(-1 [-4.8%])</span> 26.96 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-3 [-1.6%])</span> 194 | <span style='color: green'>(-13 [-1.6%])</span> 776 | <span style='color: green'>(-2 [-0.9%])</span> 229 | <span style='color: green'>(-3 [-1.7%])</span> 171 |
| `memory_finalize_time_ms` | <span style='color: green'>(-0 [-25.0%])</span> 0.75 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(-1 [-25.0%])</span> 3 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-14 [-3.0%])</span> 439.25 | <span style='color: green'>(-54 [-3.0%])</span> 1,757 | <span style='color: green'>(-16 [-3.4%])</span> 461 | <span style='color: green'>(-29 [-6.8%])</span> 400 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-2.6%])</span> 55.50 | <span style='color: green'>(-6 [-2.6%])</span> 222 | <span style='color: green'>(-2 [-3.2%])</span> 60 | <span style='color: green'>(-1 [-1.9%])</span> 53 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-8.9%])</span> 43.25 | <span style='color: green'>(-17 [-8.9%])</span> 173 | <span style='color: red'>(+6 [+8.3%])</span> 78 | <span style='color: green'>(-12 [-35.3%])</span> 22 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.3%])</span> 69.35 | <span style='color: green'>(-4 [-1.3%])</span> 277.42 | <span style='color: red'>(+1 [+1.0%])</span> 77.11 | <span style='color: green'>(-1 [-0.9%])</span> 59.20 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+0 [+0.6%])</span> 74.09 | <span style='color: red'>(+2 [+0.6%])</span> 296.35 | <span style='color: red'>(+4 [+2.9%])</span> 136.83 | <span style='color: red'>(+0 [+0.5%])</span> 51.27 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+0 [+2.0%])</span> 17.72 | <span style='color: red'>(+1 [+2.0%])</span> 70.90 | <span style='color: green'>(-0 [-1.8%])</span> 19.79 | <span style='color: green'>(-1 [-4.7%])</span> 14.31 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-4.4%])</span> 175.75 | <span style='color: green'>(-32 [-4.4%])</span> 703 | <span style='color: green'>(-19 [-9.2%])</span> 187 | <span style='color: green'>(-1 [-0.6%])</span> 157 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 61 | 612 | 3,242 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 746 | 40 | 1 | 32 | 4,108,483 | 124.64 | 167 | 

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
| regex_program | 0 | 229 | 866 | 23,395,260 | 150,778,428 | 229 | 442 | 0 | 56.51 | 19.79 | 5 | 77.11 | 187 | 116 | 187 | 3 | 60 | 10,864,634 | 38 | 48 | 1,102,000 | 26.96 | 20 | 77 | 2 | 187 | 
| regex_program | 1 | 171 | 733 | 11,589,074 | 132,682,794 | 171 | 454 | 1 | 51.74 | 18.40 | 5 | 70.07 | 179 | 149 | 179 | 0 | 54 | 2,191,608 | 78 | 67 | 1,108,000 | 31.12 | 16 | 70 | 1 | 179 | 
| regex_program | 2 | 178 | 679 | 11,583,256 | 132,682,794 | 178 | 400 | 1 | 51.27 | 18.40 | 5 | 71.04 | 180 | 93 | 180 | 0 | 55 | 2,185,918 | 22 | 59 | 1,108,000 | 31.39 | 16 | 70 | 1 | 180 | 
| regex_program | 3 | 198 | 746 | 11,847,840 | 103,522,954 | 198 | 461 | 0 | 136.83 | 14.31 | 5 | 59.20 | 157 | 97 | 157 | 0 | 53 | 2,420,726 | 35 | 45 | 790,483 | 30.79 | 17 | 152 | 2 | 157 | 

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


Commit: https://github.com/openvm-org/openvm/commit/04ccc720b11d5a34072b31fdd1b2d2f6ae1aa27f

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17782225184)
