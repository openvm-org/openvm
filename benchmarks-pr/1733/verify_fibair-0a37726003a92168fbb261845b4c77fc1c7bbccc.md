| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+4.7%])</span> 1.51 | <span style='color: red'>(+0 [+4.7%])</span> 1.51 |
| verify_fibair | <span style='color: red'>(+0 [+5.6%])</span> 1.35 | <span style='color: red'>(+0 [+5.6%])</span> 1.35 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+72 [+5.6%])</span> 1,348 | <span style='color: red'>(+72 [+5.6%])</span> 1,348 | <span style='color: red'>(+72 [+5.6%])</span> 1,348 | <span style='color: red'>(+72 [+5.6%])</span> 1,348 |
| `main_cells_used     ` | <span style='color: red'>(+2248062 [+13.0%])</span> 19,587,304 | <span style='color: red'>(+2248062 [+13.0%])</span> 19,587,304 | <span style='color: red'>(+2248062 [+13.0%])</span> 19,587,304 | <span style='color: red'>(+2248062 [+13.0%])</span> 19,587,304 |
| `total_cycles        ` |  322,676 |  322,676 |  322,676 |  322,676 |
| `execute_metered_time_ms` | <span style='color: green'>(-4 [-2.4%])</span> 166 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+2.5%])</span> 1.94 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: red'>(+37 [+19.5%])</span> 227 | <span style='color: red'>(+37 [+19.5%])</span> 227 | <span style='color: red'>(+37 [+19.5%])</span> 227 | <span style='color: red'>(+37 [+19.5%])</span> 227 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-16.4%])</span> 1.42 | -          | <span style='color: green'>(-0 [-16.4%])</span> 1.42 | <span style='color: green'>(-0 [-16.4%])</span> 1.42 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+85 [+128.8%])</span> 151 | <span style='color: red'>(+85 [+128.8%])</span> 151 | <span style='color: red'>(+85 [+128.8%])</span> 151 | <span style='color: red'>(+85 [+128.8%])</span> 151 |
| `memory_finalize_time_ms` | <span style='color: red'>(+34 [+283.3%])</span> 46 | <span style='color: red'>(+34 [+283.3%])</span> 46 | <span style='color: red'>(+34 [+283.3%])</span> 46 | <span style='color: red'>(+34 [+283.3%])</span> 46 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-46 [-5.4%])</span> 804 | <span style='color: green'>(-46 [-5.4%])</span> 804 | <span style='color: green'>(-46 [-5.4%])</span> 804 | <span style='color: green'>(-46 [-5.4%])</span> 804 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-10 [-6.5%])</span> 145 | <span style='color: green'>(-10 [-6.5%])</span> 145 | <span style='color: green'>(-10 [-6.5%])</span> 145 | <span style='color: green'>(-10 [-6.5%])</span> 145 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-15 [-9.9%])</span> 137 | <span style='color: green'>(-15 [-9.9%])</span> 137 | <span style='color: green'>(-15 [-9.9%])</span> 137 | <span style='color: green'>(-15 [-9.9%])</span> 137 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-1.2%])</span> 84 | <span style='color: green'>(-1 [-1.2%])</span> 84 | <span style='color: green'>(-1 [-1.2%])</span> 84 | <span style='color: green'>(-1 [-1.2%])</span> 84 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-8 [-7.8%])</span> 95 | <span style='color: green'>(-8 [-7.8%])</span> 95 | <span style='color: green'>(-8 [-7.8%])</span> 95 | <span style='color: green'>(-8 [-7.8%])</span> 95 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-2.7%])</span> 287 | <span style='color: green'>(-8 [-2.7%])</span> 287 | <span style='color: green'>(-8 [-2.7%])</span> 287 | <span style='color: green'>(-8 [-2.7%])</span> 287 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 41 | 1 | 6 | 0 | 25 | 7 | 

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

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 151 | 1,348 | 322,676 | 62,474,410 | 804 | 84 | 95 | 137 | 287 | 46 | 145 | 19,587,304 | 322,677 | 52 | 1 | 166 | 1.94 | 227 | 1.42 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/0a37726003a92168fbb261845b4c77fc1c7bbccc

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15860691963)
