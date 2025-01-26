| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+1.9%])</span> 2.22 | <span style='color: red'>(+0 [+1.9%])</span> 2.22 |
| verify_fibair | <span style='color: red'>(+0 [+1.9%])</span> 2.22 | <span style='color: red'>(+0 [+1.9%])</span> 2.22 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+41 [+1.9%])</span> 2,216 | <span style='color: red'>(+41 [+1.9%])</span> 2,216 | <span style='color: red'>(+41 [+1.9%])</span> 2,216 | <span style='color: red'>(+41 [+1.9%])</span> 2,216 |
| `main_cells_used     ` |  19,377,061 |  19,377,061 |  19,377,061 |  19,377,061 |
| `total_cycles        ` |  513,854 |  513,854 |  513,854 |  513,854 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-1.8%])</span> 110 | <span style='color: green'>(-2 [-1.8%])</span> 110 | <span style='color: green'>(-2 [-1.8%])</span> 110 | <span style='color: green'>(-2 [-1.8%])</span> 110 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-5 [-1.5%])</span> 334 | <span style='color: green'>(-5 [-1.5%])</span> 334 | <span style='color: green'>(-5 [-1.5%])</span> 334 | <span style='color: green'>(-5 [-1.5%])</span> 334 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+48 [+2.8%])</span> 1,772 | <span style='color: red'>(+48 [+2.8%])</span> 1,772 | <span style='color: red'>(+48 [+2.8%])</span> 1,772 | <span style='color: red'>(+48 [+2.8%])</span> 1,772 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-11 [-3.4%])</span> 315 | <span style='color: green'>(-11 [-3.4%])</span> 315 | <span style='color: green'>(-11 [-3.4%])</span> 315 | <span style='color: green'>(-11 [-3.4%])</span> 315 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+8 [+21.1%])</span> 46 | <span style='color: red'>(+8 [+21.1%])</span> 46 | <span style='color: red'>(+8 [+21.1%])</span> 46 | <span style='color: red'>(+8 [+21.1%])</span> 46 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+13 [+4.3%])</span> 315 | <span style='color: red'>(+13 [+4.3%])</span> 315 | <span style='color: red'>(+13 [+4.3%])</span> 315 | <span style='color: red'>(+13 [+4.3%])</span> 315 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+7 [+3.1%])</span> 233 | <span style='color: red'>(+7 [+3.1%])</span> 233 | <span style='color: red'>(+7 [+3.1%])</span> 233 | <span style='color: red'>(+7 [+3.1%])</span> 233 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+23 [+6.0%])</span> 408 | <span style='color: red'>(+23 [+6.0%])</span> 408 | <span style='color: red'>(+23 [+6.0%])</span> 408 | <span style='color: red'>(+23 [+6.0%])</span> 408 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+8 [+1.8%])</span> 451 | <span style='color: red'>(+8 [+1.8%])</span> 451 | <span style='color: red'>(+8 [+1.8%])</span> 451 | <span style='color: red'>(+8 [+1.8%])</span> 451 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 63 | 2 | 13 | 0 | 33 | 13 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 53 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 176 | 555 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 4 |  | 11 | 20 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 17 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 17 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 334 | 2,216 | 513,854 | 44,140,184 | 1,772 | 233 | 408 | 315 | 451 | 315 | 19,377,061 | 46 | 110 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 36 | 26 | 63,488 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 216 | 399 | 10,076,160 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 16 | 23 | 5,111,808 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 16,384 |  | 12 | 10 | 360,448 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 262,144 |  | 20 | 30 | 13,107,200 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 24 | 25 | 6,422,528 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 34 | 950,272 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/96a4fd7100308cf23270b0ab8cb65ef6bc4c709c

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12978507037)
