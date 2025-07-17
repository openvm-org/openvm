| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+71.8%])</span> 2.57 | <span style='color: green'>(-0 [-3.7%])</span> 1.44 |
| ecrecover_program | <span style='color: red'>(+1 [+75.5%])</span> 2.49 | <span style='color: green'>(-0 [-3.9%])</span> 1.36 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-174 [-12.3%])</span> 1,246 | <span style='color: red'>(+1072 [+75.5%])</span> 2,492 | <span style='color: green'>(-56 [-3.9%])</span> 1,364 | <span style='color: green'>(-292 [-20.6%])</span> 1,128 |
| `main_cells_used     ` | <span style='color: green'>(-3873678 [-47.4%])</span> 4,305,870.50 | <span style='color: red'>(+432192 [+5.3%])</span> 8,611,741 | <span style='color: green'>(-672428 [-8.2%])</span> 7,507,121 | <span style='color: green'>(-7074929 [-86.5%])</span> 1,104,620 |
| `total_cycles        ` | <span style='color: green'>(-68642 [-50.0%])</span> 68,641.50 |  137,283 | <span style='color: green'>(-9283 [-6.8%])</span> 128,000 | <span style='color: green'>(-128000 [-93.2%])</span> 9,283 |
| `execute_metered_time_ms` |  74 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+0.7%])</span> 1.85 | -          | <span style='color: red'>(+0 [+0.7%])</span> 1.85 | <span style='color: red'>(+0 [+0.7%])</span> 1.85 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-42 [-50.0%])</span> 42 |  84 | <span style='color: green'>(-13 [-15.5%])</span> 71 | <span style='color: green'>(-71 [-84.5%])</span> 13 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-23.1%])</span> 1.24 | -          | <span style='color: red'>(+0 [+11.3%])</span> 1.80 | <span style='color: green'>(-1 [-57.4%])</span> 0.69 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+80 [+22.8%])</span> 428.50 | <span style='color: red'>(+508 [+145.6%])</span> 857 | <span style='color: red'>(+157 [+45.0%])</span> 506 | <span style='color: red'>(+2 [+0.6%])</span> 351 |
| `memory_finalize_time_ms` | <span style='color: green'>(-4 [-5.8%])</span> 73.50 | <span style='color: red'>(+69 [+88.5%])</span> 147 | <span style='color: red'>(+3 [+3.8%])</span> 81 | <span style='color: green'>(-12 [-15.4%])</span> 66 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-5 [-6.9%])</span> 67 | <span style='color: red'>(+62 [+86.1%])</span> 134 | <span style='color: red'>(+2 [+2.8%])</span> 74 | <span style='color: green'>(-12 [-16.7%])</span> 60 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-212 [-21.4%])</span> 775.50 | <span style='color: red'>(+564 [+57.1%])</span> 1,551 | <span style='color: green'>(-45 [-4.6%])</span> 942 | <span style='color: green'>(-378 [-38.3%])</span> 609 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-40 [-28.3%])</span> 102.50 | <span style='color: red'>(+62 [+43.4%])</span> 205 | <span style='color: green'>(-5 [-3.5%])</span> 138 | <span style='color: green'>(-76 [-53.1%])</span> 67 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-21.4%])</span> 22 | <span style='color: red'>(+16 [+57.1%])</span> 44 | <span style='color: red'>(+2 [+7.1%])</span> 30 | <span style='color: green'>(-14 [-50.0%])</span> 14 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-68 [-41.2%])</span> 97 | <span style='color: red'>(+29 [+17.6%])</span> 194 | <span style='color: green'>(-40 [-24.2%])</span> 125 | <span style='color: green'>(-96 [-58.2%])</span> 69 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-17 [-21.0%])</span> 64 | <span style='color: red'>(+47 [+58.0%])</span> 128 | <span style='color: red'>(+3 [+3.7%])</span> 84 | <span style='color: green'>(-37 [-45.7%])</span> 44 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-32 [-23.9%])</span> 102 | <span style='color: red'>(+70 [+52.2%])</span> 204 | <span style='color: green'>(-3 [-2.2%])</span> 131 | <span style='color: green'>(-61 [-45.5%])</span> 73 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-49 [-11.6%])</span> 372 | <span style='color: red'>(+323 [+76.7%])</span> 744 | <span style='color: green'>(-3 [-0.7%])</span> 418 | <span style='color: green'>(-95 [-22.6%])</span> 326 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 907 | 9 | 11,795 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 2 | 24 | 137,284 | 1 | 5,231 | 74 | 1.85 | 

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
| ecrecover_program | AccessAdapterAir<16> | 0 | 4,096 |  | 16 | 25 | 167,936 | 
| ecrecover_program | AccessAdapterAir<16> | 1 | 512 |  | 16 | 25 | 20,992 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | AccessAdapterAir<32> | 1 | 256 |  | 16 | 41 | 14,592 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| ecrecover_program | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | KeccakVmAir | 1 | 32 |  | 1,056 | 3,163 | 135,008 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | MemoryMerkleAir<8> | 1 | 2,048 |  | 16 | 32 | 98,304 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PersistentBoundaryAir<8> | 1 | 1,024 |  | 12 | 20 | 32,768 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 4,096 |  | 52 | 36 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 256 |  | 40 | 37 | 19,712 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 512 |  | 52 | 53 | 53,760 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 2,048 |  | 28 | 26 | 110,592 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 256 |  | 32 | 32 | 16,384 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 512 |  | 28 | 18 | 23,552 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 1,024 |  | 56 | 166 | 227,328 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 256 |  | 36 | 28 | 16,384 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 8,192 |  | 52 | 36 | 720,896 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 256 |  | 52 | 36 | 22,528 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 4,096 |  | 52 | 41 | 380,928 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 4 |  | 72 | 39 | 444 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 1 |  | 72 | 39 | 111 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 4 |  | 52 | 31 | 332 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 128 |  | 28 | 20 | 6,144 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 256 |  | 836 | 547 | 354,048 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 1 |  | 192 | 199 | 391 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 128 |  | 860 | 625 | 190,080 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 351 | 1,364 | 128,000 | 32,924,886 | 942 | 84 | 131 | 2,068 | 125 | 418 | 74 | 25 | 81 | 138 | 7,507,121 | 128,000 | 30 | 71 | 1.80 | 0 | 
| ecrecover_program | 1 | 506 | 1,128 | 9,283 | 11,106,330 | 609 | 44 | 73 | 1,577 | 69 | 326 | 60 | 23 | 66 | 67 | 1,104,620 | 9,284 | 14 | 13 | 0.69 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 396,364 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,239,256 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 198,182 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,663,724 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 471,268 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 160 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,947,898 | 2,013,265,921 | 
| ecrecover_program | 1 | 0 | 27,734 | 2,013,265,921 | 
| ecrecover_program | 1 | 1 | 94,846 | 2,013,265,921 | 
| ecrecover_program | 1 | 2 | 13,867 | 2,013,265,921 | 
| ecrecover_program | 1 | 3 | 252,227 | 2,013,265,921 | 
| ecrecover_program | 1 | 4 | 7,168 | 2,013,265,921 | 
| ecrecover_program | 1 | 5 | 3,072 | 2,013,265,921 | 
| ecrecover_program | 1 | 6 | 33,223 | 2,013,265,921 | 
| ecrecover_program | 1 | 7 | 24 | 2,013,265,921 | 
| ecrecover_program | 1 | 8 | 1,384,481 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/32dfc88eb7b1c76bea954ed6fba19fe25503bebe

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16333776877)
