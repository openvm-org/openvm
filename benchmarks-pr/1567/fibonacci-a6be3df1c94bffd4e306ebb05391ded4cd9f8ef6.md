| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  5.82 |  5.82 |
| fibonacci_program |  2.33 |  2.33 |
| leaf |  3.40 |  3.40 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,328 |  2,328 |  2,328 |  2,328 |
| `main_cells_used     ` |  50,589,231 |  50,589,231 |  50,589,231 |  50,589,231 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_metered_time_ms` |  91 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  24.14 | -          |  24.14 |  24.14 |
| `execute_e3_time_ms  ` |  99 |  99 |  99 |  99 |
| `execute_e3_insn_mi/s` |  15.12 | -          |  15.12 |  15.12 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-102 [-18.5%])</span> 450 | <span style='color: green'>(-102 [-18.5%])</span> 450 | <span style='color: green'>(-102 [-18.5%])</span> 450 | <span style='color: green'>(-102 [-18.5%])</span> 450 |
| `memory_finalize_time_ms` |  65 |  65 |  65 |  65 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  63 |  63 |  63 |  63 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-103 [-5.5%])</span> 1,779 | <span style='color: green'>(-103 [-5.5%])</span> 1,779 | <span style='color: green'>(-103 [-5.5%])</span> 1,779 | <span style='color: green'>(-103 [-5.5%])</span> 1,779 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-22 [-6.4%])</span> 324 | <span style='color: green'>(-22 [-6.4%])</span> 324 | <span style='color: green'>(-22 [-6.4%])</span> 324 | <span style='color: green'>(-22 [-6.4%])</span> 324 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-8 [-5.5%])</span> 138 | <span style='color: green'>(-8 [-5.5%])</span> 138 | <span style='color: green'>(-8 [-5.5%])</span> 138 | <span style='color: green'>(-8 [-5.5%])</span> 138 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-60 [-14.9%])</span> 343 | <span style='color: green'>(-60 [-14.9%])</span> 343 | <span style='color: green'>(-60 [-14.9%])</span> 343 | <span style='color: green'>(-60 [-14.9%])</span> 343 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-4 [-2.3%])</span> 168 | <span style='color: green'>(-4 [-2.3%])</span> 168 | <span style='color: green'>(-4 [-2.3%])</span> 168 | <span style='color: green'>(-4 [-2.3%])</span> 168 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-16 [-8.1%])</span> 182 | <span style='color: green'>(-16 [-8.1%])</span> 182 | <span style='color: green'>(-16 [-8.1%])</span> 182 | <span style='color: green'>(-16 [-8.1%])</span> 182 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+2 [+0.3%])</span> 615 | <span style='color: red'>(+2 [+0.3%])</span> 615 | <span style='color: red'>(+2 [+0.3%])</span> 615 | <span style='color: red'>(+2 [+0.3%])</span> 615 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,401 |  3,401 |  3,401 |  3,401 |
| `main_cells_used     ` | <span style='color: red'>(+259148 [+0.4%])</span> 70,094,234 | <span style='color: red'>(+259148 [+0.4%])</span> 70,094,234 | <span style='color: red'>(+259148 [+0.4%])</span> 70,094,234 | <span style='color: red'>(+259148 [+0.4%])</span> 70,094,234 |
| `total_cycles        ` |  1,247,983 |  1,247,983 |  1,247,983 |  1,247,983 |
| `execute_metered_time_ms` |  426 |  426 |  426 |  426 |
| `execute_metered_insn_mi/s` |  2.93 | -          |  2.93 |  2.93 |
| `execute_e3_time_ms  ` |  481 |  481 |  481 |  481 |
| `execute_e3_insn_mi/s` |  2.59 | -          |  2.59 |  2.59 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-480 [-71.5%])</span> 191 | <span style='color: green'>(-480 [-71.5%])</span> 191 | <span style='color: green'>(-480 [-71.5%])</span> 191 | <span style='color: green'>(-480 [-71.5%])</span> 191 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-150 [-6.1%])</span> 2,303 | <span style='color: green'>(-150 [-6.1%])</span> 2,303 | <span style='color: green'>(-150 [-6.1%])</span> 2,303 | <span style='color: green'>(-150 [-6.1%])</span> 2,303 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-45 [-9.9%])</span> 410 | <span style='color: green'>(-45 [-9.9%])</span> 410 | <span style='color: green'>(-45 [-9.9%])</span> 410 | <span style='color: green'>(-45 [-9.9%])</span> 410 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+3.3%])</span> 190 | <span style='color: red'>(+6 [+3.3%])</span> 190 | <span style='color: red'>(+6 [+3.3%])</span> 190 | <span style='color: red'>(+6 [+3.3%])</span> 190 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-67 [-11.7%])</span> 507 | <span style='color: green'>(-67 [-11.7%])</span> 507 | <span style='color: green'>(-67 [-11.7%])</span> 507 | <span style='color: green'>(-67 [-11.7%])</span> 507 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-14 [-5.3%])</span> 252 | <span style='color: green'>(-14 [-5.3%])</span> 252 | <span style='color: green'>(-14 [-5.3%])</span> 252 | <span style='color: green'>(-14 [-5.3%])</span> 252 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-16 [-6.7%])</span> 222 | <span style='color: green'>(-16 [-6.7%])</span> 222 | <span style='color: green'>(-16 [-6.7%])</span> 222 | <span style='color: green'>(-16 [-6.7%])</span> 222 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-13 [-1.8%])</span> 716 | <span style='color: green'>(-13 [-1.8%])</span> 716 | <span style='color: green'>(-13 [-1.8%])</span> 716 | <span style='color: green'>(-13 [-1.8%])</span> 716 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 47 | 5 | 6,453 | 3,440 | 

| group | single_leaf_agg_time_ms | num_segments | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 1 |  | 24 | 1,500,278 | 1 | 5,926 | 91 | 24.14 | 
| leaf | 3,439 |  | 1 |  |  | 1 |  |  |  | 

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
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32 |  | 36 | 28 | 2,048 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 52 | 41 | 11,904 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 20 | 768 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 191 | 3,401 | 1,247,983 | 253,173,226 | 2,303 | 252 | 222 | 507 | 716 | 8 | 410 | 70,094,234 | 1,247,984 | 190 | 426 | 2.93 | 481 | 2.59 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 450 | 2,328 | 1,500,277 | 160,837,996 | 1,779 | 168 | 182 | 1,953 | 343 | 615 | 63 | 25 | 65 | 324 | 50,589,231 | 1,500,278 | 138 | 99 | 15.12 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/a6be3df1c94bffd4e306ebb05391ded4cd9f8ef6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16224742693)
