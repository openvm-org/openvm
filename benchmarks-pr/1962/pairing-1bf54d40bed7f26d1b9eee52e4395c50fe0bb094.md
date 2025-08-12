| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.6%])</span> 4.05 | <span style='color: red'>(+0 [+3.6%])</span> 4.05 |
| pairing | <span style='color: red'>(+0 [+3.7%])</span> 3.93 | <span style='color: red'>(+0 [+3.7%])</span> 3.93 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+142 [+3.7%])</span> 3,933 | <span style='color: red'>(+142 [+3.7%])</span> 3,933 | <span style='color: red'>(+142 [+3.7%])</span> 3,933 | <span style='color: red'>(+142 [+3.7%])</span> 3,933 |
| `main_cells_used     ` |  98,834,293 |  98,834,293 |  98,834,293 |  98,834,293 |
| `total_cells_used    ` |  219,937,463 |  219,937,463 |  219,937,463 |  219,937,463 |
| `insns               ` |  1,882,939 |  3,765,878 |  1,882,939 |  1,882,939 |
| `execute_metered_time_ms` |  114 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  16.39 | -          |  16.39 |  16.39 |
| `execute_preflight_time_ms` | <span style='color: green'>(-1 [-0.5%])</span> 212 | <span style='color: green'>(-1 [-0.5%])</span> 212 | <span style='color: green'>(-1 [-0.5%])</span> 212 | <span style='color: green'>(-1 [-0.5%])</span> 212 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-0.2%])</span> 9.14 | -          | <span style='color: green'>(-0 [-0.2%])</span> 9.14 | <span style='color: green'>(-0 [-0.2%])</span> 9.14 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-9 [-2.4%])</span> 368 | <span style='color: green'>(-9 [-2.4%])</span> 368 | <span style='color: green'>(-9 [-2.4%])</span> 368 | <span style='color: green'>(-9 [-2.4%])</span> 368 |
| `memory_finalize_time_ms` |  4 |  4 |  4 |  4 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+153 [+5.2%])</span> 3,105 | <span style='color: red'>(+153 [+5.2%])</span> 3,105 | <span style='color: red'>(+153 [+5.2%])</span> 3,105 | <span style='color: red'>(+153 [+5.2%])</span> 3,105 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+52 [+8.7%])</span> 648 | <span style='color: red'>(+52 [+8.7%])</span> 648 | <span style='color: red'>(+52 [+8.7%])</span> 648 | <span style='color: red'>(+52 [+8.7%])</span> 648 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-14 [-5.3%])</span> 248 | <span style='color: green'>(-14 [-5.3%])</span> 248 | <span style='color: green'>(-14 [-5.3%])</span> 248 | <span style='color: green'>(-14 [-5.3%])</span> 248 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+66 [+11.6%])</span> 634 | <span style='color: red'>(+66 [+11.6%])</span> 634 | <span style='color: red'>(+66 [+11.6%])</span> 634 | <span style='color: red'>(+66 [+11.6%])</span> 634 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+6 [+1.7%])</span> 357 | <span style='color: red'>(+6 [+1.7%])</span> 357 | <span style='color: red'>(+6 [+1.7%])</span> 357 | <span style='color: red'>(+6 [+1.7%])</span> 357 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+11 [+4.0%])</span> 289 | <span style='color: red'>(+11 [+4.0%])</span> 289 | <span style='color: red'>(+11 [+4.0%])</span> 289 | <span style='color: red'>(+11 [+4.0%])</span> 289 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+32 [+3.6%])</span> 920 | <span style='color: red'>(+32 [+3.6%])</span> 920 | <span style='color: red'>(+32 [+3.6%])</span> 920 | <span style='color: red'>(+32 [+3.6%])</span> 920 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 720 | 8 | 4,092 | 

| group | vm_create_initial_state_time_ms | prove_segment_time_ms | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 1 | 3,933 | 6 | 1,882,939 | 1 | 114 | 16.39 | 37 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| pairing | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| pairing | PhantomAir | 2 | 3 | 5 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| pairing | ProgramAir | 1 | 1 | 4 | 
| pairing | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| pairing | Rv32HintStoreAir | 2 | 18 | 28 | 
| pairing | VariableRangeCheckerAir | 1 | 1 | 4 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| pairing | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<32> | 0 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 32,768 |  | 16 | 32 | 1,572,864 | 
| pairing | PersistentBoundaryAir<8> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32,768 |  | 8 | 300 | 10,092,544 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 56 | 166 | 7,104 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 65,536 |  | 36 | 28 | 4,194,304 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 320 | 263 | 596,992 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 16,384 |  | 604 | 497 | 18,038,784 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 368 | 3,933 | 219,937,463 | 304,931,516 | 368 | 3,105 | 0 | 357 | 289 | 634 | 920 | 7 | 4 | 648 | 98,834,293 | 1,882,939 | 248 | 212 | 9.14 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 5,382,342 | 2,013,265,921 | 
| pairing | 0 | 1 | 18,152,512 | 2,013,265,921 | 
| pairing | 0 | 2 | 2,691,171 | 2,013,265,921 | 
| pairing | 0 | 3 | 25,000,068 | 2,013,265,921 | 
| pairing | 0 | 4 | 131,072 | 2,013,265,921 | 
| pairing | 0 | 5 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 6 | 6,016,192 | 2,013,265,921 | 
| pairing | 0 | 7 | 4,096 | 2,013,265,921 | 
| pairing | 0 | 8 | 58,426,029 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/1bf54d40bed7f26d1b9eee52e4395c50fe0bb094

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16897009067)
