| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  1.04 |  1.04 |
| verify_fibair |  1.04 |  1.04 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,043 |  1,043 |  1,043 |  1,043 |
| `main_cells_used     ` | <span style='color: red'>(+332276 [+1.9%])</span> 17,672,408 | <span style='color: red'>(+332276 [+1.9%])</span> 17,672,408 | <span style='color: red'>(+332276 [+1.9%])</span> 17,672,408 | <span style='color: red'>(+332276 [+1.9%])</span> 17,672,408 |
| `total_cells_used    ` |  39,989,126 |  39,989,126 |  39,989,126 |  39,989,126 |
| `insns               ` |  322,700 |  322,700 |  322,700 |  322,700 |
| `execute_e3_time_ms  ` |  148 |  148 |  148 |  148 |
| `execute_e3_insn_mi/s` |  2.17 | -          |  2.17 |  2.17 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-129 [-72.1%])</span> 50 | <span style='color: green'>(-129 [-72.1%])</span> 50 | <span style='color: green'>(-129 [-72.1%])</span> 50 | <span style='color: green'>(-129 [-72.1%])</span> 50 |
| `memory_finalize_time_ms` |  5 |  5 |  5 |  5 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-18 [-2.1%])</span> 845 | <span style='color: green'>(-18 [-2.1%])</span> 845 | <span style='color: green'>(-18 [-2.1%])</span> 845 | <span style='color: green'>(-18 [-2.1%])</span> 845 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+8 [+5.1%])</span> 164 | <span style='color: red'>(+8 [+5.1%])</span> 164 | <span style='color: red'>(+8 [+5.1%])</span> 164 | <span style='color: red'>(+8 [+5.1%])</span> 164 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.0%])</span> 49 | <span style='color: green'>(-1 [-2.0%])</span> 49 | <span style='color: green'>(-1 [-2.0%])</span> 49 | <span style='color: green'>(-1 [-2.0%])</span> 49 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+3 [+2.0%])</span> 150 | <span style='color: red'>(+3 [+2.0%])</span> 150 | <span style='color: red'>(+3 [+2.0%])</span> 150 | <span style='color: red'>(+3 [+2.0%])</span> 150 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-5.6%])</span> 85 | <span style='color: green'>(-5 [-5.6%])</span> 85 | <span style='color: green'>(-5 [-5.6%])</span> 85 | <span style='color: green'>(-5 [-5.6%])</span> 85 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+1.0%])</span> 99 | <span style='color: red'>(+1 [+1.0%])</span> 99 | <span style='color: red'>(+1 [+1.0%])</span> 99 | <span style='color: red'>(+1 [+1.0%])</span> 99 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-24 [-7.6%])</span> 292 | <span style='color: green'>(-24 [-7.6%])</span> 292 | <span style='color: green'>(-24 [-7.6%])</span> 292 | <span style='color: green'>(-24 [-7.6%])</span> 292 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | app proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 38 | 1 | 6 | 0 | 22 | 7 | 2,117 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 71 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 14 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 572 |  | 
| PhantomAir |  | 2 |  | 3 | 5 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 25 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 29 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmConnectorAir |  | 2 |  | 5 | 11 |  | 
| VolatileBoundaryAir |  | 2 |  | 7 | 19 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 50 | 1,043 | 39,989,126 | 62,474,410 | 50 | 845 | 2 | 85 | 99 | 150 | 292 | 5 | 164 | 17,672,408 | 322,700 | 49 | 1 | 148 | 2.17 | 0 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 84 | 27 | 227,328 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 28 | 12 | 1,310,720 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 312 | 398 | 23,265,280 | 
| verify_fibair | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 36 | 29 | 17,039,360 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 40 | 21 | 3,997,696 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 40 | 27 | 2,195,456 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 36 | 38 | 2,424,832 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 20 | 12 | 2,097,152 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| verify_fibair | 0 | 1,085,444 | 2,013,265,921 | 
| verify_fibair | 1 | 5,411,200 | 2,013,265,921 | 
| verify_fibair | 2 | 542,722 | 2,013,265,921 | 
| verify_fibair | 3 | 5,476,612 | 2,013,265,921 | 
| verify_fibair | 4 | 65,536 | 2,013,265,921 | 
| verify_fibair | 5 | 12,851,850 | 2,013,265,921 | 

| trace_height_constraint | threshold |
| --- | --- |
| 0 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/3fb163efeba26c73aa393485bcd61e9f143a39a1

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16764337392)
