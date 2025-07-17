| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+7.7%])</span> 14 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: green'>(-21 [-22.1%])</span> 74 | <span style='color: green'>(-21 [-22.1%])</span> 74 | <span style='color: green'>(-21 [-22.1%])</span> 74 | <span style='color: green'>(-21 [-22.1%])</span> 74 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+5 [+28.9%])</span> 20.26 | -          | <span style='color: red'>(+5 [+28.9%])</span> 20.26 | <span style='color: red'>(+5 [+28.9%])</span> 20.26 |
| `memory_finalize_time_ms` | <span style='color: green'>(-62 [-95.4%])</span> 3 | <span style='color: green'>(-62 [-95.4%])</span> 3 | <span style='color: green'>(-62 [-95.4%])</span> 3 | <span style='color: green'>(-62 [-95.4%])</span> 3 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: red'>(+4 [+6.5%])</span> 66 | <span style='color: red'>(+4 [+6.5%])</span> 66 | <span style='color: red'>(+4 [+6.5%])</span> 66 | <span style='color: red'>(+4 [+6.5%])</span> 66 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+342 [+19.2%])</span> 2,122 | <span style='color: red'>(+342 [+19.2%])</span> 2,122 | <span style='color: red'>(+342 [+19.2%])</span> 2,122 | <span style='color: red'>(+342 [+19.2%])</span> 2,122 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+40 [+12.5%])</span> 360 | <span style='color: red'>(+40 [+12.5%])</span> 360 | <span style='color: red'>(+40 [+12.5%])</span> 360 | <span style='color: red'>(+40 [+12.5%])</span> 360 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+13 [+9.2%])</span> 154 | <span style='color: red'>(+13 [+9.2%])</span> 154 | <span style='color: red'>(+13 [+9.2%])</span> 154 | <span style='color: red'>(+13 [+9.2%])</span> 154 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+247 [+71.8%])</span> 591 | <span style='color: red'>(+247 [+71.8%])</span> 591 | <span style='color: red'>(+247 [+71.8%])</span> 591 | <span style='color: red'>(+247 [+71.8%])</span> 591 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+13 [+7.8%])</span> 180 | <span style='color: red'>(+13 [+7.8%])</span> 180 | <span style='color: red'>(+13 [+7.8%])</span> 180 | <span style='color: red'>(+13 [+7.8%])</span> 180 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+18 [+9.8%])</span> 202 | <span style='color: red'>(+18 [+9.8%])</span> 202 | <span style='color: red'>(+18 [+9.8%])</span> 202 | <span style='color: red'>(+18 [+9.8%])</span> 202 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+15 [+2.4%])</span> 630 | <span style='color: red'>(+15 [+2.4%])</span> 630 | <span style='color: red'>(+15 [+2.4%])</span> 630 | <span style='color: red'>(+15 [+2.4%])</span> 630 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `execute_e3_time_ms  ` |  472 |  472 |  472 |  472 |
| `execute_e3_insn_mi/s` |  2.65 | -          |  2.65 |  2.65 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  3,171 |  3,171 |  3,171 |  3,171 |
| `main_trace_commit_time_ms` |  563 |  563 |  563 |  563 |
| `generate_perm_trace_time_ms` |  250 |  250 |  250 |  250 |
| `perm_trace_commit_time_ms` |  1,021 |  1,021 |  1,021 |  1,021 |
| `quotient_poly_compute_time_ms` |  334 |  334 |  334 |  334 |
| `quotient_poly_commit_time_ms` |  245 |  245 |  245 |  245 |
| `pcs_opening_time_ms ` |  752 |  752 |  752 |  752 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 48 | 5 | 3,820 | 4,914 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 3,740 |  | 25 | 1 | 14 | 
| leaf | 4,913 |  | 1 |  | 1 |  | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 1 | 2 | 5 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 8 | 
| fibonacci_program | PhantomAir | 1 | 3 | 6 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 36 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 45 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 49 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 103 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 28 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 45 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 92 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 20 | 
| fibonacci_program | VmConnectorAir | 1 | 5 | 13 | 
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
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 24 | 11 | 9,175,040 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 24 | 13 | 4,849,664 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 24 | 17 | 167,936 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 160 | 27 | 98,041,856 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 40 | 12 | 3,407,872 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 548 | 398 | 123,994,112 | 
| leaf | PhantomAir | 0 | 32,768 |  | 16 | 6 | 720,896 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 64 | 29 | 97,517,568 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 48 | 23 | 9,306,112 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 48 | 27 | 4,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 64 | 21 | 44,564,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 64 | 27 | 11,927,552 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 64 | 38 | 13,369,344 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 24 | 5 | 58 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 32 | 12 | 5,767,168 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 128 |  | 24 | 17 | 5,248 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 128 |  | 16 | 20 | 4,608 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 16 | 6 | 22 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 76 | 32 | 432 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 84 | 36 | 125,829,120 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 76 | 37 | 59,244,544 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32 |  | 68 | 28 | 3,072 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 72 | 41 | 14,464 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 52 | 20 | 1,152 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 24 | 5 | 58 | 

| group | idx | tracegen_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 188 | 427,533,050 | 3,171 | 334 | 245 | 1,021 | 752 | 7 | 563 | 1,254,728 | 250 | 472 | 2.65 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,701,764 | 2,013,265,921 | 
| leaf | 0 | 1 | 30,945,536 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,850,882 | 2,013,265,921 | 
| leaf | 0 | 3 | 31,072,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 71,226,058 | 2,013,265,921 | 

| group | segment | tracegen_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 230 | 220,876,864 | 2,122 | 180 | 202 | 591 | 630 | 66 | 24 | 3 | 360 | 1,500,278 | 154 | 74 | 20.26 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,542 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,749,400 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,271 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,749,532 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 1,664 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 640 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,100 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,535,101 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/5cc6d595dfbf34c652c66c4a3a68a4b310a644f3

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16355724310)
