| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+20 [+1408.3%])</span> 21.87 | <span style='color: red'>(+20 [+1408.3%])</span> 21.87 |
| verify_fibair | <span style='color: red'>(+20 [+1591.6%])</span> 21.70 | <span style='color: red'>(+20 [+1591.6%])</span> 21.70 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+20420 [+1591.6%])</span> 21,703 | <span style='color: red'>(+20420 [+1591.6%])</span> 21,703 | <span style='color: red'>(+20420 [+1591.6%])</span> 21,703 | <span style='color: red'>(+20420 [+1591.6%])</span> 21,703 |
| `main_cells_used     ` | <span style='color: red'>(+2246814 [+13.0%])</span> 19,586,968 | <span style='color: red'>(+2246814 [+13.0%])</span> 19,586,968 | <span style='color: red'>(+2246814 [+13.0%])</span> 19,586,968 | <span style='color: red'>(+2246814 [+13.0%])</span> 19,586,968 |
| `total_cycles        ` |  322,648 |  322,648 |  322,648 |  322,648 |
| `execute_metered_time_ms` |  167 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  1.93 | -          | -          | -          |
| `execute_e3_time_ms  ` | <span style='color: red'>(+5852 [+3032.1%])</span> 6,045 | <span style='color: red'>(+5852 [+3032.1%])</span> 6,045 | <span style='color: red'>(+5852 [+3032.1%])</span> 6,045 | <span style='color: red'>(+5852 [+3032.1%])</span> 6,045 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-2 [-96.8%])</span> 0.05 | -          | <span style='color: green'>(-2 [-96.8%])</span> 0.05 | <span style='color: green'>(-2 [-96.8%])</span> 0.05 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+14084 [+21667.7%])</span> 14,149 | <span style='color: red'>(+14084 [+21667.7%])</span> 14,149 | <span style='color: red'>(+14084 [+21667.7%])</span> 14,149 | <span style='color: red'>(+14084 [+21667.7%])</span> 14,149 |
| `memory_finalize_time_ms` | <span style='color: red'>(+1536 [+12800.0%])</span> 1,548 | <span style='color: red'>(+1536 [+12800.0%])</span> 1,548 | <span style='color: red'>(+1536 [+12800.0%])</span> 1,548 | <span style='color: red'>(+1536 [+12800.0%])</span> 1,548 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+484 [+56.4%])</span> 1,342 | <span style='color: red'>(+484 [+56.4%])</span> 1,342 | <span style='color: red'>(+484 [+56.4%])</span> 1,342 | <span style='color: red'>(+484 [+56.4%])</span> 1,342 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+66 [+42.6%])</span> 221 | <span style='color: red'>(+66 [+42.6%])</span> 221 | <span style='color: red'>(+66 [+42.6%])</span> 221 | <span style='color: red'>(+66 [+42.6%])</span> 221 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+27 [+52.9%])</span> 78 | <span style='color: red'>(+27 [+52.9%])</span> 78 | <span style='color: red'>(+27 [+52.9%])</span> 78 | <span style='color: red'>(+27 [+52.9%])</span> 78 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+74 [+48.7%])</span> 226 | <span style='color: red'>(+74 [+48.7%])</span> 226 | <span style='color: red'>(+74 [+48.7%])</span> 226 | <span style='color: red'>(+74 [+48.7%])</span> 226 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+38 [+41.3%])</span> 130 | <span style='color: red'>(+38 [+41.3%])</span> 130 | <span style='color: red'>(+38 [+41.3%])</span> 130 | <span style='color: red'>(+38 [+41.3%])</span> 130 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+85 [+83.3%])</span> 187 | <span style='color: red'>(+85 [+83.3%])</span> 187 | <span style='color: red'>(+85 [+83.3%])</span> 187 | <span style='color: red'>(+85 [+83.3%])</span> 187 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+193 [+64.1%])</span> 494 | <span style='color: red'>(+193 [+64.1%])</span> 494 | <span style='color: red'>(+193 [+64.1%])</span> 494 | <span style='color: red'>(+193 [+64.1%])</span> 494 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 34 | 1 | 6 | 0 | 19 | 6 | 

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
| verify_fibair | 14,149 | 21,703 | 322,648 | 78,923,818 | 1,342 | 130 | 187 | 226 | 494 | 1,548 | 221 | 19,586,968 | 322,649 | 78 | 1 | 167 | 1.93 | 6,045 | 0.05 | 0 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 524,288 |  | 16 | 11 | 14,155,776 | 
| verify_fibair | AccessAdapterAir<4> | 262,144 |  | 16 | 13 | 7,602,176 | 
| verify_fibair | AccessAdapterAir<8> | 4,096 |  | 16 | 17 | 135,168 | 
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
| verify_fibair | 1 | 7,192,576 | 2,013,265,921 | 
| verify_fibair | 2 | 542,722 | 2,013,265,921 | 
| verify_fibair | 3 | 6,664,196 | 2,013,265,921 | 
| verify_fibair | 4 | 65,536 | 2,013,265,921 | 
| verify_fibair | 5 | 15,820,810 | 2,013,265,921 | 

| trace_height_constraint | threshold |
| --- | --- |
| 0 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/d4a85fe8c3e748f81c21dfc4d8ff71cd0ef67c19

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15822061104)
