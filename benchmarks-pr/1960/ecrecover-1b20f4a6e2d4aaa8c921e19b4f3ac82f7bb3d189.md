| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-2.4%])</span> 1.36 | <span style='color: green'>(-0 [-2.4%])</span> 1.36 |
| ecrecover_program | <span style='color: green'>(-0 [-2.4%])</span> 1.36 | <span style='color: green'>(-0 [-2.4%])</span> 1.36 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-34 [-2.4%])</span> 1,358 | <span style='color: green'>(-34 [-2.4%])</span> 1,358 | <span style='color: green'>(-34 [-2.4%])</span> 1,358 | <span style='color: green'>(-34 [-2.4%])</span> 1,358 |
| `main_cells_used     ` |  8,866,644 |  8,866,644 |  8,866,644 |  8,866,644 |
| `total_cells_used    ` |  26,518,202 |  26,518,202 |  26,518,202 |  26,518,202 |
| `insns               ` |  140,487 |  280,974 |  140,487 |  140,487 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+0.2%])</span> 20.54 | -          | <span style='color: red'>(+0 [+0.2%])</span> 20.54 | <span style='color: red'>(+0 [+0.2%])</span> 20.54 |
| `execute_preflight_time_ms` | <span style='color: green'>(-1 [-1.2%])</span> 81 | <span style='color: green'>(-1 [-1.2%])</span> 81 | <span style='color: green'>(-1 [-1.2%])</span> 81 | <span style='color: green'>(-1 [-1.2%])</span> 81 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+0.5%])</span> 1.86 | -          | <span style='color: red'>(+0 [+0.5%])</span> 1.86 | <span style='color: red'>(+0 [+0.5%])</span> 1.86 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-2.2%])</span> 91 | <span style='color: green'>(-2 [-2.2%])</span> 91 | <span style='color: green'>(-2 [-2.2%])</span> 91 | <span style='color: green'>(-2 [-2.2%])</span> 91 |
| `memory_finalize_time_ms` |  4 |  4 |  4 |  4 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-31 [-3.2%])</span> 939 | <span style='color: green'>(-31 [-3.2%])</span> 939 | <span style='color: green'>(-31 [-3.2%])</span> 939 | <span style='color: green'>(-31 [-3.2%])</span> 939 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 |
| `generate_perm_trace_time_ms` |  30 |  30 |  30 |  30 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-2 [-1.6%])</span> 127 | <span style='color: green'>(-2 [-1.6%])</span> 127 | <span style='color: green'>(-2 [-1.6%])</span> 127 | <span style='color: green'>(-2 [-1.6%])</span> 127 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-2.4%])</span> 80 | <span style='color: green'>(-2 [-2.4%])</span> 80 | <span style='color: green'>(-2 [-2.4%])</span> 80 | <span style='color: green'>(-2 [-2.4%])</span> 80 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-0.8%])</span> 128 | <span style='color: green'>(-1 [-0.8%])</span> 128 | <span style='color: green'>(-1 [-0.8%])</span> 128 | <span style='color: green'>(-1 [-0.8%])</span> 128 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-29 [-6.5%])</span> 420 | <span style='color: green'>(-29 [-6.5%])</span> 420 | <span style='color: green'>(-29 [-6.5%])</span> 420 | <span style='color: green'>(-29 [-6.5%])</span> 420 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 868 | 8 | 1,410 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 1,358 | 6 | 140,487 | 1 | 6 | 20.54 | 39 | 

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
| ecrecover_program | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 8,192 |  | 52 | 36 | 720,896 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 64 |  | 52 | 31 | 5,312 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 91 | 1,358 | 26,518,202 | 32,927,986 | 91 | 939 | 0 | 80 | 128 | 127 | 420 | 7 | 4 | 138 | 8,866,644 | 140,487 | 30 | 81 | 1.86 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 396,436 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,239,472 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 198,218 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,663,940 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 471,272 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 320 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,948,602 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/1b20f4a6e2d4aaa8c921e19b4f3ac82f7bb3d189

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16894050316)
