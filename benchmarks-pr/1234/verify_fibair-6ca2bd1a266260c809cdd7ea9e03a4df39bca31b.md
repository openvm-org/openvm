| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.4%])</span> 3.38 | <span style='color: red'>(+0 [+0.4%])</span> 3.38 |
| verify_fibair | <span style='color: red'>(+0 [+0.4%])</span> 3.38 | <span style='color: red'>(+0 [+0.4%])</span> 3.38 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+12 [+0.4%])</span> 3,385 | <span style='color: red'>(+12 [+0.4%])</span> 3,385 | <span style='color: red'>(+12 [+0.4%])</span> 3,385 | <span style='color: red'>(+12 [+0.4%])</span> 3,385 |
| `main_cells_used     ` |  25,523,334 |  25,523,334 |  25,523,334 |  25,523,334 |
| `total_cycles        ` |  711,682 |  711,682 |  711,682 |  711,682 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.7%])</span> 147 | <span style='color: red'>(+1 [+0.7%])</span> 147 | <span style='color: red'>(+1 [+0.7%])</span> 147 | <span style='color: red'>(+1 [+0.7%])</span> 147 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+3 [+0.7%])</span> 424 | <span style='color: red'>(+3 [+0.7%])</span> 424 | <span style='color: red'>(+3 [+0.7%])</span> 424 | <span style='color: red'>(+3 [+0.7%])</span> 424 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+8 [+0.3%])</span> 2,814 | <span style='color: red'>(+8 [+0.3%])</span> 2,814 | <span style='color: red'>(+8 [+0.3%])</span> 2,814 | <span style='color: red'>(+8 [+0.3%])</span> 2,814 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+1 [+0.2%])</span> 475 | <span style='color: red'>(+1 [+0.2%])</span> 475 | <span style='color: red'>(+1 [+0.2%])</span> 475 | <span style='color: red'>(+1 [+0.2%])</span> 475 |
| `generate_perm_trace_time_ms` |  65 |  65 |  65 |  65 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.6%])</span> 488 | <span style='color: red'>(+3 [+0.6%])</span> 488 | <span style='color: red'>(+3 [+0.6%])</span> 488 | <span style='color: red'>(+3 [+0.6%])</span> 488 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+5 [+0.9%])</span> 575 | <span style='color: red'>(+5 [+0.9%])</span> 575 | <span style='color: red'>(+5 [+0.9%])</span> 575 | <span style='color: red'>(+5 [+0.9%])</span> 575 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+8 [+1.5%])</span> 533 | <span style='color: red'>(+8 [+1.5%])</span> 533 | <span style='color: red'>(+8 [+1.5%])</span> 533 | <span style='color: red'>(+8 [+1.5%])</span> 533 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-7 [-1.0%])</span> 677 | <span style='color: green'>(-7 [-1.0%])</span> 677 | <span style='color: green'>(-7 [-1.0%])</span> 677 | <span style='color: green'>(-7 [-1.0%])</span> 677 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 66 | 3 | 13 | 0 | 33 | 15 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 31 | 302 |  | 
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
| verify_fibair | 424 | 3,385 | 711,682 | 72,898,584 | 2,814 | 575 | 533 | 488 | 677 | 475 | 25,523,334 | 65 | 147 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 32,768 |  | 16 | 17 | 1,081,344 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 8,192 |  | 36 | 348 | 3,145,728 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 262,144 |  | 28 | 23 | 13,369,344 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 32,768 |  | 12 | 10 | 720,896 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 524,288 |  | 20 | 30 | 26,214,400 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 36 | 25 | 15,990,784 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/6ca2bd1a266260c809cdd7ea9e03a4df39bca31b

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12840306070)