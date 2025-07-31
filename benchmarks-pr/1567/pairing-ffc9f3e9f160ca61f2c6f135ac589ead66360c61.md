| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  8.21 |  8.21 |
| pairing |  3.66 |  3.66 |
| leaf |  4.41 |  4.41 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,660 |  3,660 |  3,660 |  3,660 |
| `main_cells_used     ` | <span style='color: green'>(-4172678 [-4.3%])</span> 93,105,105 | <span style='color: green'>(-4172678 [-4.3%])</span> 93,105,105 | <span style='color: green'>(-4172678 [-4.3%])</span> 93,105,105 | <span style='color: green'>(-4172678 [-4.3%])</span> 93,105,105 |
| `total_cells_used    ` |  209,833,619 |  209,833,619 |  209,833,619 |  209,833,619 |
| `insns               ` |  1,862,965 |  3,725,930 |  1,862,965 |  1,862,965 |
| `execute_metered_time_ms` |  147 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  12.63 | -          |  12.63 |  12.63 |
| `execute_e3_time_ms  ` |  304 |  304 |  304 |  304 |
| `execute_e3_insn_mi/s` |  6.11 | -          |  6.11 |  6.11 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-610 [-61.9%])</span> 376 | <span style='color: green'>(-610 [-61.9%])</span> 376 | <span style='color: green'>(-610 [-61.9%])</span> 376 | <span style='color: green'>(-610 [-61.9%])</span> 376 |
| `memory_finalize_time_ms` |  6 |  6 |  6 |  6 |
| `boundary_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `merkle_finalize_time_ms` |  105 |  105 |  105 |  105 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+117 [+4.1%])</span> 2,980 | <span style='color: red'>(+117 [+4.1%])</span> 2,980 | <span style='color: red'>(+117 [+4.1%])</span> 2,980 | <span style='color: red'>(+117 [+4.1%])</span> 2,980 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+23 [+4.0%])</span> 601 | <span style='color: red'>(+23 [+4.0%])</span> 601 | <span style='color: red'>(+23 [+4.0%])</span> 601 | <span style='color: red'>(+23 [+4.0%])</span> 601 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+2.6%])</span> 237 | <span style='color: red'>(+6 [+2.6%])</span> 237 | <span style='color: red'>(+6 [+2.6%])</span> 237 | <span style='color: red'>(+6 [+2.6%])</span> 237 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+29 [+5.3%])</span> 577 | <span style='color: red'>(+29 [+5.3%])</span> 577 | <span style='color: red'>(+29 [+5.3%])</span> 577 | <span style='color: red'>(+29 [+5.3%])</span> 577 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+23 [+7.0%])</span> 352 | <span style='color: red'>(+23 [+7.0%])</span> 352 | <span style='color: red'>(+23 [+7.0%])</span> 352 | <span style='color: red'>(+23 [+7.0%])</span> 352 |
| `quotient_poly_commit_time_ms` |  280 |  280 |  280 |  280 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+42 [+4.8%])</span> 925 | <span style='color: red'>(+42 [+4.8%])</span> 925 | <span style='color: red'>(+42 [+4.8%])</span> 925 | <span style='color: red'>(+42 [+4.8%])</span> 925 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,407 |  4,407 |  4,407 |  4,407 |
| `main_cells_used     ` | <span style='color: green'>(-66693637 [-32.5%])</span> 138,831,697 | <span style='color: green'>(-66693637 [-32.5%])</span> 138,831,697 | <span style='color: green'>(-66693637 [-32.5%])</span> 138,831,697 | <span style='color: green'>(-66693637 [-32.5%])</span> 138,831,697 |
| `total_cells_used    ` |  348,737,475 |  348,737,475 |  348,737,475 |  348,737,475 |
| `insns               ` |  2,010,480 |  2,010,480 |  2,010,480 |  2,010,480 |
| `execute_e3_time_ms  ` |  671 |  671 |  671 |  671 |
| `execute_e3_insn_mi/s` |  2.100 | -          |  2.100 |  2.100 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1603 [-82.3%])</span> 344 | <span style='color: green'>(-1603 [-82.3%])</span> 344 | <span style='color: green'>(-1603 [-82.3%])</span> 344 | <span style='color: green'>(-1603 [-82.3%])</span> 344 |
| `memory_finalize_time_ms` |  10 |  10 |  10 |  10 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-2357 [-41.0%])</span> 3,392 | <span style='color: green'>(-2357 [-41.0%])</span> 3,392 | <span style='color: green'>(-2357 [-41.0%])</span> 3,392 | <span style='color: green'>(-2357 [-41.0%])</span> 3,392 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-469 [-42.4%])</span> 637 | <span style='color: green'>(-469 [-42.4%])</span> 637 | <span style='color: green'>(-469 [-42.4%])</span> 637 | <span style='color: green'>(-469 [-42.4%])</span> 637 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-241 [-43.2%])</span> 317 | <span style='color: green'>(-241 [-43.2%])</span> 317 | <span style='color: green'>(-241 [-43.2%])</span> 317 | <span style='color: green'>(-241 [-43.2%])</span> 317 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-636 [-44.6%])</span> 789 | <span style='color: green'>(-636 [-44.6%])</span> 789 | <span style='color: green'>(-636 [-44.6%])</span> 789 | <span style='color: green'>(-636 [-44.6%])</span> 789 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-286 [-40.6%])</span> 418 | <span style='color: green'>(-286 [-40.6%])</span> 418 | <span style='color: green'>(-286 [-40.6%])</span> 418 | <span style='color: green'>(-286 [-40.6%])</span> 418 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-169 [-36.8%])</span> 290 | <span style='color: green'>(-169 [-36.8%])</span> 290 | <span style='color: green'>(-169 [-36.8%])</span> 290 | <span style='color: green'>(-169 [-36.8%])</span> 290 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-557 [-37.3%])</span> 936 | <span style='color: green'>(-557 [-37.3%])</span> 936 | <span style='color: green'>(-557 [-37.3%])</span> 936 | <span style='color: green'>(-557 [-37.3%])</span> 936 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 50 | 9 | 4,105 | 5,531 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 5,530 |  | 1 |  |  | 1 |  |  |  | 
| pairing |  | 3,911 |  | 6 | 1,862,965 | 1 | 147 | 12.63 | 38 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 572 | 
| leaf | PhantomAir | 2 | 3 | 5 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 11 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 
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

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 344 | 4,407 | 348,737,475 | 418,598,378 | 342 | 3,392 | 2 | 418 | 290 | 789 | 936 | 10 | 637 | 138,831,697 | 2,010,480 | 317 | 671 | 2.100 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 7,274,628 | 2,013,265,921 | 
| leaf | 0 | 1 | 45,531,392 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,637,314 | 2,013,265,921 | 
| leaf | 0 | 3 | 44,859,652 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 102,351,562 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 376 | 3,660 | 209,833,619 | 304,931,516 | 375 | 2,980 | 2 | 352 | 280 | 577 | 925 | 105 | 7 | 6 | 601 | 93,105,105 | 1,862,965 | 237 | 304 | 6.11 | 2 | 

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


Commit: https://github.com/openvm-org/openvm/commit/ffc9f3e9f160ca61f2c6f135ac589ead66360c61

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16658358121)
