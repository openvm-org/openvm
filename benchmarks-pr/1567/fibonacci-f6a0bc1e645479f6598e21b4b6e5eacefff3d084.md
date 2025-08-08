| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  5.24 |  5.24 |
| fibonacci_program |  2.20 |  2.20 |
| leaf |  3.03 |  3.03 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,199 |  2,199 |  2,199 |  2,199 |
| `main_cells_used     ` | <span style='color: red'>(+915004 [+1.8%])</span> 51,504,507 | <span style='color: red'>(+915004 [+1.8%])</span> 51,504,507 | <span style='color: red'>(+915004 [+1.8%])</span> 51,504,507 | <span style='color: red'>(+915004 [+1.8%])</span> 51,504,507 |
| `total_cells_used    ` |  127,360,509 |  127,360,509 |  127,360,509 |  127,360,509 |
| `insns               ` |  1,500,210 |  3,000,420 |  1,500,210 |  1,500,210 |
| `execute_metered_time_ms` |  7 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  205.37 | -          |  205.37 |  205.37 |
| `execute_preflight_time_ms` |  48 |  48 |  48 |  48 |
| `execute_preflight_insn_mi/s` |  32.18 | -          |  32.18 |  32.18 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-327 [-59.7%])</span> 221 | <span style='color: green'>(-327 [-59.7%])</span> 221 | <span style='color: green'>(-327 [-59.7%])</span> 221 | <span style='color: green'>(-327 [-59.7%])</span> 221 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+65 [+3.5%])</span> 1,930 | <span style='color: red'>(+65 [+3.5%])</span> 1,930 | <span style='color: red'>(+65 [+3.5%])</span> 1,930 | <span style='color: red'>(+65 [+3.5%])</span> 1,930 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+13 [+3.7%])</span> 360 | <span style='color: red'>(+13 [+3.7%])</span> 360 | <span style='color: red'>(+13 [+3.7%])</span> 360 | <span style='color: red'>(+13 [+3.7%])</span> 360 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+7.1%])</span> 136 | <span style='color: red'>(+9 [+7.1%])</span> 136 | <span style='color: red'>(+9 [+7.1%])</span> 136 | <span style='color: red'>(+9 [+7.1%])</span> 136 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+11 [+2.9%])</span> 392 | <span style='color: red'>(+11 [+2.9%])</span> 392 | <span style='color: red'>(+11 [+2.9%])</span> 392 | <span style='color: red'>(+11 [+2.9%])</span> 392 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+10 [+5.7%])</span> 184 | <span style='color: red'>(+10 [+5.7%])</span> 184 | <span style='color: red'>(+10 [+5.7%])</span> 184 | <span style='color: red'>(+10 [+5.7%])</span> 184 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+1.0%])</span> 200 | <span style='color: red'>(+2 [+1.0%])</span> 200 | <span style='color: red'>(+2 [+1.0%])</span> 200 | <span style='color: red'>(+2 [+1.0%])</span> 200 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+20 [+3.2%])</span> 653 | <span style='color: red'>(+20 [+3.2%])</span> 653 | <span style='color: red'>(+20 [+3.2%])</span> 653 | <span style='color: red'>(+20 [+3.2%])</span> 653 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,033 |  3,033 |  3,033 |  3,033 |
| `main_cells_used     ` | <span style='color: red'>(+1052170 [+1.5%])</span> 70,886,848 | <span style='color: red'>(+1052170 [+1.5%])</span> 70,886,848 | <span style='color: red'>(+1052170 [+1.5%])</span> 70,886,848 | <span style='color: red'>(+1052170 [+1.5%])</span> 70,886,848 |
| `total_cells_used    ` |  168,012,978 |  168,012,978 |  168,012,978 |  168,012,978 |
| `insns               ` |  1,248,063 |  1,248,063 |  1,248,063 |  1,248,063 |
| `execute_preflight_time_ms` |  371 |  371 |  371 |  371 |
| `execute_preflight_insn_mi/s` |  3.73 | -          |  3.73 |  3.73 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-487 [-72.6%])</span> 184 | <span style='color: green'>(-487 [-72.6%])</span> 184 | <span style='color: green'>(-487 [-72.6%])</span> 184 | <span style='color: green'>(-487 [-72.6%])</span> 184 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+41 [+1.7%])</span> 2,478 | <span style='color: red'>(+41 [+1.7%])</span> 2,478 | <span style='color: red'>(+41 [+1.7%])</span> 2,478 | <span style='color: red'>(+41 [+1.7%])</span> 2,478 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+6 [+1.3%])</span> 463 | <span style='color: red'>(+6 [+1.3%])</span> 463 | <span style='color: red'>(+6 [+1.3%])</span> 463 | <span style='color: red'>(+6 [+1.3%])</span> 463 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+23 [+12.8%])</span> 202 | <span style='color: red'>(+23 [+12.8%])</span> 202 | <span style='color: red'>(+23 [+12.8%])</span> 202 | <span style='color: red'>(+23 [+12.8%])</span> 202 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.2%])</span> 572 | <span style='color: green'>(-1 [-0.2%])</span> 572 | <span style='color: green'>(-1 [-0.2%])</span> 572 | <span style='color: green'>(-1 [-0.2%])</span> 572 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+4 [+1.5%])</span> 263 | <span style='color: red'>(+4 [+1.5%])</span> 263 | <span style='color: red'>(+4 [+1.5%])</span> 263 | <span style='color: red'>(+4 [+1.5%])</span> 263 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+4 [+1.7%])</span> 244 | <span style='color: red'>(+4 [+1.7%])</span> 244 | <span style='color: red'>(+4 [+1.7%])</span> 244 | <span style='color: red'>(+4 [+1.7%])</span> 244 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+6 [+0.8%])</span> 729 | <span style='color: red'>(+6 [+0.8%])</span> 729 | <span style='color: red'>(+6 [+0.8%])</span> 729 | <span style='color: red'>(+6 [+0.8%])</span> 729 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 48 | 5 | 2,487 | 4,095 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 2,440 |  | 6 | 1,500,210 | 1 | 7 | 205.37 | 36 | 
| leaf | 4,094 |  | 1 |  |  | 1 |  |  |  | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| fibonacci_program | VmConnectorAir | 2 | 5 | 11 | 
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

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 128 |  | 16 | 17 | 4,224 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 16 | 32 | 24,576 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 128 |  | 12 | 20 | 4,096 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 52 | 41 | 11,904 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 20 | 768 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 184 | 3,033 | 168,012,978 | 253,173,226 | 184 | 2,478 | 0 | 263 | 244 | 572 | 729 | 8 | 463 | 70,886,848 | 1,248,063 | 202 | 371 | 3.73 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 221 | 2,199 | 127,360,509 | 160,836,972 | 221 | 1,930 | 0 | 184 | 200 | 392 | 653 | 7 | 0 | 360 | 51,504,507 | 1,500,210 | 136 | 48 | 32.18 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,510 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,749,336 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,255 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,749,404 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 1,664 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 640 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,084 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,534,845 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f6a0bc1e645479f6598e21b4b6e5eacefff3d084

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16842092387)
