| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.2%])</span> 2.31 | <span style='color: green'>(-0 [-1.2%])</span> 2.31 |
| verify_fibair | <span style='color: green'>(-0 [-1.2%])</span> 2.31 | <span style='color: green'>(-0 [-1.2%])</span> 2.31 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-27 [-1.2%])</span> 2,310 | <span style='color: green'>(-27 [-1.2%])</span> 2,310 | <span style='color: green'>(-27 [-1.2%])</span> 2,310 | <span style='color: green'>(-27 [-1.2%])</span> 2,310 |
| `main_cells_used     ` | <span style='color: red'>(+27941 [+0.1%])</span> 19,385,753 | <span style='color: red'>(+27941 [+0.1%])</span> 19,385,753 | <span style='color: red'>(+27941 [+0.1%])</span> 19,385,753 | <span style='color: red'>(+27941 [+0.1%])</span> 19,385,753 |
| `total_cycles        ` |  513,747 |  513,747 |  513,747 |  513,747 |
| `execute_time_ms     ` |  109 |  109 |  109 |  109 |
| `trace_gen_time_ms   ` |  333 |  333 |  333 |  333 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-27 [-1.4%])</span> 1,868 | <span style='color: green'>(-27 [-1.4%])</span> 1,868 | <span style='color: green'>(-27 [-1.4%])</span> 1,868 | <span style='color: green'>(-27 [-1.4%])</span> 1,868 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-13 [-4.0%])</span> 311 | <span style='color: green'>(-13 [-4.0%])</span> 311 | <span style='color: green'>(-13 [-4.0%])</span> 311 | <span style='color: green'>(-13 [-4.0%])</span> 311 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 | <span style='color: red'>(+3 [+7.5%])</span> 43 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-35 [-9.9%])</span> 318 | <span style='color: green'>(-35 [-9.9%])</span> 318 | <span style='color: green'>(-35 [-9.9%])</span> 318 | <span style='color: green'>(-35 [-9.9%])</span> 318 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-6 [-1.6%])</span> 366 | <span style='color: green'>(-6 [-1.6%])</span> 366 | <span style='color: green'>(-6 [-1.6%])</span> 366 | <span style='color: green'>(-6 [-1.6%])</span> 366 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+13 [+3.7%])</span> 366 | <span style='color: red'>(+13 [+3.7%])</span> 366 | <span style='color: red'>(+13 [+3.7%])</span> 366 | <span style='color: red'>(+13 [+3.7%])</span> 366 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+11 [+2.4%])</span> 461 | <span style='color: red'>(+11 [+2.4%])</span> 461 | <span style='color: red'>(+11 [+2.4%])</span> 461 | <span style='color: red'>(+11 [+2.4%])</span> 461 |



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
| verify_fibair | 333 | 2,310 | 513,747 | 50,178,200 | 1,868 | 366 | 366 | 318 | 461 | 311 | 19,385,753 | 43 | 109 | 

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


Commit: https://github.com/openvm-org/openvm/commit/e6fa95eb8d4bdb86279aaf2b852239691aa21149

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12936072689)
