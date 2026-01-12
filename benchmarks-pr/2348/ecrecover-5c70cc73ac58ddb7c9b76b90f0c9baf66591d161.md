| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: red'>(+0 [+1.4%])</span> 0.75 | <span style='color: red'>(+0 [+1.4%])</span> 0.75 | 0.74 |
| ecrecover_program | <span style='color: red'>(+0 [+1.4%])</span> 0.75 | <span style='color: red'>(+0 [+1.4%])</span> 0.75 |  0.75 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+10 [+1.4%])</span> 739 | <span style='color: red'>(+10 [+1.4%])</span> 739 | <span style='color: red'>(+10 [+1.4%])</span> 739 | <span style='color: red'>(+10 [+1.4%])</span> 739 |
| `main_cells_used     ` |  2,265,100 |  2,265,100 |  2,265,100 |  2,265,100 |
| `total_cells_used    ` |  11,346,486 |  11,346,486 |  11,346,486 |  11,346,486 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-1.8%])</span> 19.51 | -          | <span style='color: green'>(-0 [-1.8%])</span> 19.51 | <span style='color: green'>(-0 [-1.8%])</span> 19.51 |
| `execute_preflight_insns` |  122,859 |  122,859 |  122,859 |  122,859 |
| `execute_preflight_time_ms` | <span style='color: green'>(-1 [-1.6%])</span> 61 | <span style='color: green'>(-1 [-1.6%])</span> 61 | <span style='color: green'>(-1 [-1.6%])</span> 61 | <span style='color: green'>(-1 [-1.6%])</span> 61 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+2.2%])</span> 2.26 | -          | <span style='color: red'>(+0 [+2.2%])</span> 2.26 | <span style='color: red'>(+0 [+2.2%])</span> 2.26 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-0.9%])</span> 210 | <span style='color: green'>(-2 [-0.9%])</span> 210 | <span style='color: green'>(-2 [-0.9%])</span> 210 | <span style='color: green'>(-2 [-0.9%])</span> 210 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+12 [+4.2%])</span> 299 | <span style='color: red'>(+12 [+4.2%])</span> 299 | <span style='color: red'>(+12 [+4.2%])</span> 299 | <span style='color: red'>(+12 [+4.2%])</span> 299 |
| `main_trace_commit_time_ms` |  37 |  37 |  37 |  37 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+23.1%])</span> 48 | <span style='color: red'>(+9 [+23.1%])</span> 48 | <span style='color: red'>(+9 [+23.1%])</span> 48 | <span style='color: red'>(+9 [+23.1%])</span> 48 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.2%])</span> 34.15 | <span style='color: green'>(-0 [-0.2%])</span> 34.15 | <span style='color: green'>(-0 [-0.2%])</span> 34.15 | <span style='color: green'>(-0 [-0.2%])</span> 34.15 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-0 [-0.3%])</span> 104.81 | <span style='color: green'>(-0 [-0.3%])</span> 104.81 | <span style='color: green'>(-0 [-0.3%])</span> 104.81 | <span style='color: green'>(-0 [-0.3%])</span> 104.81 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-0 [-0.6%])</span> 9.13 | <span style='color: green'>(-0 [-0.6%])</span> 9.13 | <span style='color: green'>(-0 [-0.6%])</span> 9.13 | <span style='color: green'>(-0 [-0.6%])</span> 9.13 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3 [+5.6%])</span> 57 | <span style='color: red'>(+3 [+5.6%])</span> 57 | <span style='color: red'>(+3 [+5.6%])</span> 57 | <span style='color: red'>(+3 [+5.6%])</span> 57 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 996 | 751 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 739 | 1 | 6 | 122,859 | 19.51 | 0 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | 11 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 56 | 166 | 7,104 | 
| ecrecover_program | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 320 | 263 | 9,328 | 
| ecrecover_program | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 192 | 199 | 6,256 | 
| ecrecover_program | 14 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | 16 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 192 | 199 | 6,256 | 
| ecrecover_program | 18 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | 19 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 64 |  | 52 | 31 | 5,312 | 
| ecrecover_program | 2 | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | 20 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | 21 | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | 22 | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | 23 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| ecrecover_program | 24 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| ecrecover_program | 25 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| ecrecover_program | 26 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8,192 |  | 32 | 32 | 524,288 | 
| ecrecover_program | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | 28 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | 29 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | 3 | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | 30 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 8,192 |  | 52 | 53 | 860,160 | 
| ecrecover_program | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | 33 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | 34 | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | 35 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | 36 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | 6 | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| ecrecover_program | 7 | AccessAdapterAir<16> | 0 | 4,096 |  | 16 | 25 | 167,936 | 
| ecrecover_program | 8 | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 210 | 739 | 11,346,486 | 31,789,298 | 210 | 299 | 0 | 104.81 | 9.13 | 7 | 34.15 | 57 | 89 | 57 | 0 | 37 | 2,265,100 | 48 | 61 | 122,859 | 2.26 | 21 | 114 | 2 | 57 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 375,956 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,169,840 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 187,978 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,534,916 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 446,696 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 320 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,694,650 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/5c70cc73ac58ddb7c9b76b90f0c9baf66591d161

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20904656972)
