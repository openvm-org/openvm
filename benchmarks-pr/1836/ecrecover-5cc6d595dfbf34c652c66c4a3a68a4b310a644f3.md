| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+1.4%])</span> 75 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: green'>(-2 [-2.4%])</span> 82 | <span style='color: green'>(-2 [-2.4%])</span> 82 | <span style='color: green'>(-2 [-2.4%])</span> 82 | <span style='color: green'>(-2 [-2.4%])</span> 82 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+0 [+2.4%])</span> 1.65 | -          | <span style='color: red'>(+0 [+2.4%])</span> 1.65 | <span style='color: red'>(+0 [+2.4%])</span> 1.65 |
| `memory_finalize_time_ms` | <span style='color: green'>(-75 [-96.2%])</span> 3 | <span style='color: green'>(-75 [-96.2%])</span> 3 | <span style='color: green'>(-75 [-96.2%])</span> 3 | <span style='color: green'>(-75 [-96.2%])</span> 3 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-2 [-2.8%])</span> 70 | <span style='color: green'>(-2 [-2.8%])</span> 70 | <span style='color: green'>(-2 [-2.8%])</span> 70 | <span style='color: green'>(-2 [-2.8%])</span> 70 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+17 [+1.7%])</span> 1,004 | <span style='color: red'>(+17 [+1.7%])</span> 1,004 | <span style='color: red'>(+17 [+1.7%])</span> 1,004 | <span style='color: red'>(+17 [+1.7%])</span> 1,004 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-14 [-9.8%])</span> 129 | <span style='color: green'>(-14 [-9.8%])</span> 129 | <span style='color: green'>(-14 [-9.8%])</span> 129 | <span style='color: green'>(-14 [-9.8%])</span> 129 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+8 [+28.6%])</span> 36 | <span style='color: red'>(+8 [+28.6%])</span> 36 | <span style='color: red'>(+8 [+28.6%])</span> 36 | <span style='color: red'>(+8 [+28.6%])</span> 36 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+33 [+20.0%])</span> 198 | <span style='color: red'>(+33 [+20.0%])</span> 198 | <span style='color: red'>(+33 [+20.0%])</span> 198 | <span style='color: red'>(+33 [+20.0%])</span> 198 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-2.5%])</span> 79 | <span style='color: green'>(-2 [-2.5%])</span> 79 | <span style='color: green'>(-2 [-2.5%])</span> 79 | <span style='color: green'>(-2 [-2.5%])</span> 79 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-11 [-8.2%])</span> 123 | <span style='color: green'>(-11 [-8.2%])</span> 123 | <span style='color: green'>(-11 [-8.2%])</span> 123 | <span style='color: green'>(-11 [-8.2%])</span> 123 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+2 [+0.5%])</span> 423 | <span style='color: red'>(+2 [+0.5%])</span> 423 | <span style='color: red'>(+2 [+0.5%])</span> 423 | <span style='color: red'>(+2 [+0.5%])</span> 423 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `execute_e3_time_ms  ` |  1,131 |  1,131 |  1,131 |  1,131 |
| `execute_e3_insn_mi/s` |  2.86 | -          |  2.86 |  2.86 |
| `memory_finalize_time_ms` |  13 |  13 |  13 |  13 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  10,429 |  10,429 |  10,429 |  10,429 |
| `main_trace_commit_time_ms` |  1,578 |  1,578 |  1,578 |  1,578 |
| `generate_perm_trace_time_ms` |  968 |  968 |  968 |  968 |
| `perm_trace_commit_time_ms` |  3,689 |  3,689 |  3,689 |  3,689 |
| `quotient_poly_compute_time_ms` |  1,097 |  1,097 |  1,097 |  1,097 |
| `quotient_poly_commit_time_ms` |  690 |  690 |  690 |  690 |
| `pcs_opening_time_ms ` |  2,402 |  2,402 |  2,402 |  2,402 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 50 | 9 | 2,703 | 13,354 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program |  | 2,557 |  | 24 | 1 | 75 | 
| leaf | 13,353 |  | 1 |  | 1 |  | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 1 | 2 | 5 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,571 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 8 | 
| ecrecover_program | PhantomAir | 1 | 3 | 6 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 36 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 45 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 49 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 103 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 237 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 28 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 39 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 45 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 92 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 687 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 269 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 428 | 671 | 
| ecrecover_program | VmConnectorAir | 1 | 5 | 13 | 
| leaf | AccessAdapterAir<2> | 2 | 5 | 14 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 14 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 14 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 90 | 
| leaf | JalRangeCheckAir | 2 | 9 | 17 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 631 | 
| leaf | PhantomAir | 1 | 3 | 6 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 34 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 35 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 26 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 26 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 34 | 
| leaf | VmConnectorAir | 1 | 5 | 13 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 22 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 2,097,152 |  | 24 | 11 | 73,400,320 | 
| leaf | AccessAdapterAir<4> | 0 | 1,048,576 |  | 24 | 13 | 38,797,312 | 
| leaf | AccessAdapterAir<8> | 0 | 32,768 |  | 24 | 17 | 1,343,488 | 
| leaf | FriReducedOpeningAir | 0 | 4,194,304 |  | 160 | 27 | 784,334,848 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 40 | 12 | 3,407,872 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 548 | 398 | 247,988,224 | 
| leaf | PhantomAir | 0 | 32,768 |  | 16 | 6 | 720,896 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 64 | 29 | 195,035,136 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 48 | 23 | 37,224,448 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 48 | 27 | 4,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 64 | 21 | 89,128,960 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 64 | 27 | 23,855,104 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 64 | 38 | 26,738,688 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 24 | 5 | 58 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 32 | 12 | 46,137,344 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 4,096 |  | 24 | 25 | 200,704 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 2,048 |  | 24 | 41 | 133,120 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 16,384 |  | 24 | 17 | 671,744 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,288 | 3,163 | 569,728 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 20 | 32 | 212,992 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 16 | 20 | 147,456 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 16 | 6 | 352 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 76 | 32 | 27,648 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 84 | 36 | 7,864,320 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 76 | 37 | 462,848 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 100 | 53 | 2,506,752 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 48 | 26 | 1,212,416 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 56 | 32 | 360,448 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 44 | 18 | 253,952 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 104 | 166 | 1,105,920 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 68 | 28 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 8,192 |  | 76 | 36 | 917,504 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 72 | 41 | 7,405,568 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 100 | 39 | 1,112 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 80 | 31 | 3,552 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 52 | 20 | 147,456 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 1,664 | 547 | 4,528,128 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 636 | 263 | 28,768 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 1,716 | 625 | 2,397,184 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 24 | 5 | 58 | 

| group | idx | tracegen_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 705 | 1,579,913,978 | 10,429 | 1,097 | 690 | 3,689 | 2,402 | 13 | 1,578 | 3,233,414 | 968 | 1,131 | 2.86 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 18,022,532 | 2,013,265,921 | 
| leaf | 0 | 1 | 128,155,904 | 2,013,265,921 | 
| leaf | 0 | 2 | 9,011,266 | 2,013,265,921 | 
| leaf | 0 | 3 | 128,254,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 0 | 5 | 284,754,634 | 2,013,265,921 | 

| group | segment | tracegen_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 107 | 41,441,282 | 1,004 | 79 | 123 | 198 | 423 | 70 | 24 | 3 | 129 | 137,284 | 36 | 82 | 1.65 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 396,372 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,239,280 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 198,186 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,663,748 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 471,272 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 192 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,947,994 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/5cc6d595dfbf34c652c66c4a3a68a4b310a644f3

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16355724310)
