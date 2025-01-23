| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.2%])</span> 2.34 | <span style='color: red'>(+0 [+0.2%])</span> 2.34 |
| verify_fibair | <span style='color: red'>(+0 [+0.2%])</span> 2.34 | <span style='color: red'>(+0 [+0.2%])</span> 2.34 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+5 [+0.2%])</span> 2,342 | <span style='color: red'>(+5 [+0.2%])</span> 2,342 | <span style='color: red'>(+5 [+0.2%])</span> 2,342 | <span style='color: red'>(+5 [+0.2%])</span> 2,342 |
| `main_cells_used     ` |  19,358,022 |  19,358,022 |  19,358,022 |  19,358,022 |
| `total_cycles        ` |  513,343 |  513,343 |  513,343 |  513,343 |
| `execute_time_ms     ` | <span style='color: red'>(+6 [+5.5%])</span> 115 | <span style='color: red'>(+6 [+5.5%])</span> 115 | <span style='color: red'>(+6 [+5.5%])</span> 115 | <span style='color: red'>(+6 [+5.5%])</span> 115 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.3%])</span> 332 | <span style='color: green'>(-1 [-0.3%])</span> 332 | <span style='color: green'>(-1 [-0.3%])</span> 332 | <span style='color: green'>(-1 [-0.3%])</span> 332 |
| `stark_prove_excluding_trace_time_ms` |  1,895 |  1,895 |  1,895 |  1,895 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+4 [+1.2%])</span> 328 | <span style='color: red'>(+4 [+1.2%])</span> 328 | <span style='color: red'>(+4 [+1.2%])</span> 328 | <span style='color: red'>(+4 [+1.2%])</span> 328 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.3%])</span> 352 | <span style='color: green'>(-1 [-0.3%])</span> 352 | <span style='color: green'>(-1 [-0.3%])</span> 352 | <span style='color: green'>(-1 [-0.3%])</span> 352 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+3 [+0.8%])</span> 375 | <span style='color: red'>(+3 [+0.8%])</span> 375 | <span style='color: red'>(+3 [+0.8%])</span> 375 | <span style='color: red'>(+3 [+0.8%])</span> 375 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+7 [+2.0%])</span> 360 | <span style='color: red'>(+7 [+2.0%])</span> 360 | <span style='color: red'>(+7 [+2.0%])</span> 360 | <span style='color: red'>(+7 [+2.0%])</span> 360 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-16 [-3.6%])</span> 434 | <span style='color: green'>(-16 [-3.6%])</span> 434 | <span style='color: green'>(-16 [-3.6%])</span> 434 | <span style='color: green'>(-16 [-3.6%])</span> 434 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 64 | 3 | 13 | 0 | 31 | 15 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 53 |  | 
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
| verify_fibair | 332 | 2,342 | 513,343 | 50,170,008 | 1,895 | 375 | 360 | 352 | 434 | 328 | 19,358,022 | 43 | 115 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 16 | 11 | 1,769,472 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 16 | 13 | 950,272 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 36 | 26 | 63,488 | 
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


Commit: https://github.com/openvm-org/openvm/commit/8c7deb8f2a845a16d7a581cae3f127a68b65c2c5

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12936252106)
