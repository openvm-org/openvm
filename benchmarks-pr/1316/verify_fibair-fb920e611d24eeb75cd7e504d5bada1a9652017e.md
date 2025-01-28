| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+1.5%])</span> 2.21 | <span style='color: red'>(+0 [+1.5%])</span> 2.21 |
| verify_fibair | <span style='color: red'>(+0 [+1.5%])</span> 2.21 | <span style='color: red'>(+0 [+1.5%])</span> 2.21 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+32 [+1.5%])</span> 2,207 | <span style='color: red'>(+32 [+1.5%])</span> 2,207 | <span style='color: red'>(+32 [+1.5%])</span> 2,207 | <span style='color: red'>(+32 [+1.5%])</span> 2,207 |
| `main_cells_used     ` |  18,710,170 |  18,710,170 |  18,710,170 |  18,710,170 |
| `total_cycles        ` |  513,761 |  513,761 |  513,761 |  513,761 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 | <span style='color: red'>(+2 [+1.8%])</span> 111 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-9 [-2.6%])</span> 331 | <span style='color: green'>(-9 [-2.6%])</span> 331 | <span style='color: green'>(-9 [-2.6%])</span> 331 | <span style='color: green'>(-9 [-2.6%])</span> 331 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+39 [+2.3%])</span> 1,765 | <span style='color: red'>(+39 [+2.3%])</span> 1,765 | <span style='color: red'>(+39 [+2.3%])</span> 1,765 | <span style='color: red'>(+39 [+2.3%])</span> 1,765 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+14 [+4.5%])</span> 325 | <span style='color: red'>(+14 [+4.5%])</span> 325 | <span style='color: red'>(+14 [+4.5%])</span> 325 | <span style='color: red'>(+14 [+4.5%])</span> 325 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+7 [+19.4%])</span> 43 | <span style='color: red'>(+7 [+19.4%])</span> 43 | <span style='color: red'>(+7 [+19.4%])</span> 43 | <span style='color: red'>(+7 [+19.4%])</span> 43 |
| `perm_trace_commit_time_ms` |  310 |  310 |  310 |  310 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+3 [+1.3%])</span> 230 | <span style='color: red'>(+3 [+1.3%])</span> 230 | <span style='color: red'>(+3 [+1.3%])</span> 230 | <span style='color: red'>(+3 [+1.3%])</span> 230 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+4 [+1.0%])</span> 401 | <span style='color: red'>(+4 [+1.0%])</span> 401 | <span style='color: red'>(+4 [+1.0%])</span> 401 | <span style='color: red'>(+4 [+1.0%])</span> 401 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+11 [+2.5%])</span> 453 | <span style='color: red'>(+11 [+2.5%])</span> 453 | <span style='color: red'>(+11 [+2.5%])</span> 453 | <span style='color: red'>(+11 [+2.5%])</span> 453 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 67 | 3 | 13 | 0 | 36 | 14 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 52 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 176 | 555 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 331 | 2,207 | 513,761 | 43,401,880 | 1,765 | 230 | 401 | 310 | 453 | 325 | 18,710,170 | 43 | 111 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 36 | 25 | 62,464 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 216 | 399 | 10,076,160 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 20 | 29 | 12,845,056 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 16 | 23 | 5,111,808 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 16,384 |  | 12 | 9 | 344,064 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 24 | 22 | 6,029,312 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/fb920e611d24eeb75cd7e504d5bada1a9652017e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13017202212)
