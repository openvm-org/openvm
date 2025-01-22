| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.1%])</span> 2.27 | <span style='color: green'>(-0 [-1.1%])</span> 2.27 |
| verify_fibair | <span style='color: green'>(-0 [-1.1%])</span> 2.27 | <span style='color: green'>(-0 [-1.1%])</span> 2.27 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-26 [-1.1%])</span> 2,270 | <span style='color: green'>(-26 [-1.1%])</span> 2,270 | <span style='color: green'>(-26 [-1.1%])</span> 2,270 | <span style='color: green'>(-26 [-1.1%])</span> 2,270 |
| `main_cells_used     ` | <span style='color: green'>(-76254 [-0.4%])</span> 19,309,709 | <span style='color: green'>(-76254 [-0.4%])</span> 19,309,709 | <span style='color: green'>(-76254 [-0.4%])</span> 19,309,709 | <span style='color: green'>(-76254 [-0.4%])</span> 19,309,709 |
| `total_cycles        ` | <span style='color: green'>(-1964 [-0.4%])</span> 511,804 | <span style='color: green'>(-1964 [-0.4%])</span> 511,804 | <span style='color: green'>(-1964 [-0.4%])</span> 511,804 | <span style='color: green'>(-1964 [-0.4%])</span> 511,804 |
| `execute_time_ms     ` | <span style='color: green'>(-11 [-10.1%])</span> 98 | <span style='color: green'>(-11 [-10.1%])</span> 98 | <span style='color: green'>(-11 [-10.1%])</span> 98 | <span style='color: green'>(-11 [-10.1%])</span> 98 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-30 [-9.0%])</span> 303 | <span style='color: green'>(-30 [-9.0%])</span> 303 | <span style='color: green'>(-30 [-9.0%])</span> 303 | <span style='color: green'>(-30 [-9.0%])</span> 303 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+15 [+0.8%])</span> 1,869 | <span style='color: red'>(+15 [+0.8%])</span> 1,869 | <span style='color: red'>(+15 [+0.8%])</span> 1,869 | <span style='color: red'>(+15 [+0.8%])</span> 1,869 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+28 [+9.6%])</span> 319 | <span style='color: red'>(+28 [+9.6%])</span> 319 | <span style='color: red'>(+28 [+9.6%])</span> 319 | <span style='color: red'>(+28 [+9.6%])</span> 319 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+1.9%])</span> 55 | <span style='color: red'>(+1 [+1.9%])</span> 55 | <span style='color: red'>(+1 [+1.9%])</span> 55 | <span style='color: red'>(+1 [+1.9%])</span> 55 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-8 [-2.4%])</span> 329 | <span style='color: green'>(-8 [-2.4%])</span> 329 | <span style='color: green'>(-8 [-2.4%])</span> 329 | <span style='color: green'>(-8 [-2.4%])</span> 329 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-17 [-4.3%])</span> 378 | <span style='color: green'>(-17 [-4.3%])</span> 378 | <span style='color: green'>(-17 [-4.3%])</span> 378 | <span style='color: green'>(-17 [-4.3%])</span> 378 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+11 [+3.3%])</span> 348 | <span style='color: red'>(+11 [+3.3%])</span> 348 | <span style='color: red'>(+11 [+3.3%])</span> 348 | <span style='color: red'>(+11 [+3.3%])</span> 348 |
| `pcs_opening_time_ms ` |  437 |  437 |  437 |  437 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 | 65,536 | 65 | 3 | 13 | 0 | 32 | 16 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 176 | 590 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 23 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 303 | 2,270 | 511,804 | 50,178,200 | 1,869 | 378 | 348 | 329 | 437 | 319 | 19,309,709 | 55 | 98 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 16 | 11 | 1,769,472 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 16 | 13 | 950,272 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 356 | 399 | 12,369,920 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 28 | 23 | 6,684,672 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 16,384 |  | 12 | 10 | 360,448 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 262,144 |  | 20 | 30 | 13,107,200 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 36 | 25 | 7,995,392 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/401610adc3efa2e19821e5f6fcd8c9746dde723b

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12919031954)
