| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+5.6%])</span> 2.38 | <span style='color: red'>(+0 [+5.6%])</span> 2.38 |
| verify_fibair | <span style='color: red'>(+0 [+5.6%])</span> 2.38 | <span style='color: red'>(+0 [+5.6%])</span> 2.38 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+126 [+5.6%])</span> 2,378 | <span style='color: red'>(+126 [+5.6%])</span> 2,378 | <span style='color: red'>(+126 [+5.6%])</span> 2,378 | <span style='color: red'>(+126 [+5.6%])</span> 2,378 |
| `main_cells_used     ` | <span style='color: green'>(-25991 [-0.1%])</span> 19,359,162 | <span style='color: green'>(-25991 [-0.1%])</span> 19,359,162 | <span style='color: green'>(-25991 [-0.1%])</span> 19,359,162 | <span style='color: green'>(-25991 [-0.1%])</span> 19,359,162 |
| `total_cycles        ` |  513,457 |  513,457 |  513,457 |  513,457 |
| `execute_time_ms     ` |  113 |  113 |  113 |  113 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+4 [+1.2%])</span> 338 | <span style='color: red'>(+4 [+1.2%])</span> 338 | <span style='color: red'>(+4 [+1.2%])</span> 338 | <span style='color: red'>(+4 [+1.2%])</span> 338 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+122 [+6.8%])</span> 1,927 | <span style='color: red'>(+122 [+6.8%])</span> 1,927 | <span style='color: red'>(+122 [+6.8%])</span> 1,927 | <span style='color: red'>(+122 [+6.8%])</span> 1,927 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+18 [+5.8%])</span> 326 | <span style='color: red'>(+18 [+5.8%])</span> 326 | <span style='color: red'>(+18 [+5.8%])</span> 326 | <span style='color: red'>(+18 [+5.8%])</span> 326 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-8.9%])</span> 41 | <span style='color: green'>(-4 [-8.9%])</span> 41 | <span style='color: green'>(-4 [-8.9%])</span> 41 | <span style='color: green'>(-4 [-8.9%])</span> 41 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+37 [+11.9%])</span> 349 | <span style='color: red'>(+37 [+11.9%])</span> 349 | <span style='color: red'>(+37 [+11.9%])</span> 349 | <span style='color: red'>(+37 [+11.9%])</span> 349 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+0.5%])</span> 370 | <span style='color: red'>(+2 [+0.5%])</span> 370 | <span style='color: red'>(+2 [+0.5%])</span> 370 | <span style='color: red'>(+2 [+0.5%])</span> 370 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+58 [+17.4%])</span> 392 | <span style='color: red'>(+58 [+17.4%])</span> 392 | <span style='color: red'>(+58 [+17.4%])</span> 392 | <span style='color: red'>(+58 [+17.4%])</span> 392 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+11 [+2.5%])</span> 446 | <span style='color: red'>(+11 [+2.5%])</span> 446 | <span style='color: red'>(+11 [+2.5%])</span> 446 | <span style='color: red'>(+11 [+2.5%])</span> 446 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 67 | 3 | 14 | 0 | 32 | 17 | 

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
| verify_fibair | 338 | 2,378 | 513,457 | 50,170,008 | 1,927 | 370 | 392 | 349 | 446 | 326 | 19,359,162 | 41 | 113 | 

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


Commit: https://github.com/openvm-org/openvm/commit/0efba66c1bd00b1586ecbbba20e5c693e04e8900

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12934974892)