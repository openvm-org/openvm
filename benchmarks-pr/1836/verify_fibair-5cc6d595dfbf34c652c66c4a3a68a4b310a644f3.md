| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `execute_e3_time_ms  ` |  177 |  177 |  177 |  177 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-0.2%])</span> 1.82 | -          | <span style='color: green'>(-0 [-0.2%])</span> 1.82 | <span style='color: green'>(-0 [-0.2%])</span> 1.82 |
| `memory_finalize_time_ms` | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+84 [+9.8%])</span> 939 | <span style='color: red'>(+84 [+9.8%])</span> 939 | <span style='color: red'>(+84 [+9.8%])</span> 939 | <span style='color: red'>(+84 [+9.8%])</span> 939 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+4 [+2.5%])</span> 165 | <span style='color: red'>(+4 [+2.5%])</span> 165 | <span style='color: red'>(+4 [+2.5%])</span> 165 | <span style='color: red'>(+4 [+2.5%])</span> 165 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+3.6%])</span> 57 | <span style='color: red'>(+2 [+3.6%])</span> 57 | <span style='color: red'>(+2 [+3.6%])</span> 57 | <span style='color: red'>(+2 [+3.6%])</span> 57 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+55 [+35.0%])</span> 212 | <span style='color: red'>(+55 [+35.0%])</span> 212 | <span style='color: red'>(+55 [+35.0%])</span> 212 | <span style='color: red'>(+55 [+35.0%])</span> 212 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+7 [+8.3%])</span> 91 | <span style='color: red'>(+7 [+8.3%])</span> 91 | <span style='color: red'>(+7 [+8.3%])</span> 91 | <span style='color: red'>(+7 [+8.3%])</span> 91 |
| `quotient_poly_commit_time_ms` |  100 |  100 |  100 |  100 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+17 [+5.8%])</span> 309 | <span style='color: red'>(+17 [+5.8%])</span> 309 | <span style='color: red'>(+17 [+5.8%])</span> 309 | <span style='color: red'>(+17 [+5.8%])</span> 309 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | app proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 37 | 1 | 6 | 0 | 21 | 7 | 2,200 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 14 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 14 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 14 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 90 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 17 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 631 |  | 
| PhantomAir |  | 1 |  | 3 | 6 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 34 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 30 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 34 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 26 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 26 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 34 |  | 
| VmConnectorAir |  | 1 |  | 5 | 13 |  | 
| VolatileBoundaryAir |  | 2 |  | 7 | 22 |  | 

| group | tracegen_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 48 | 84,454,586 | 939 | 91 | 100 | 212 | 309 | 6 | 165 | 322,700 | 57 | 1 | 177 | 1.82 | 0 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 24 | 11 | 4,587,520 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 24 | 13 | 2,424,832 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 24 | 17 | 5,248 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 160 | 27 | 382,976 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 40 | 12 | 1,703,936 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 548 | 398 | 30,998,528 | 
| verify_fibair | PhantomAir | 16,384 |  | 16 | 6 | 360,448 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 64 | 29 | 24,379,392 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 48 | 23 | 2,326,528 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 64 | 21 | 5,570,560 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 64 | 27 | 2,981,888 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 64 | 38 | 3,342,336 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 24 | 5 | 58 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 32 | 12 | 2,883,584 | 

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


Commit: https://github.com/openvm-org/openvm/commit/5cc6d595dfbf34c652c66c4a3a68a4b310a644f3

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16355724310)
