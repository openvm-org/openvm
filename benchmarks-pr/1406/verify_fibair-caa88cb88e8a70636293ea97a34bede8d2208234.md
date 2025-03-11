| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.9%])</span> 1.36 | <span style='color: green'>(-0 [-1.9%])</span> 1.36 |
| verify_fibair | <span style='color: green'>(-0 [-1.9%])</span> 1.36 | <span style='color: green'>(-0 [-1.9%])</span> 1.36 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-27 [-1.9%])</span> 1,362 | <span style='color: green'>(-27 [-1.9%])</span> 1,362 | <span style='color: green'>(-27 [-1.9%])</span> 1,362 | <span style='color: green'>(-27 [-1.9%])</span> 1,362 |
| `main_cells_used     ` |  8,194,904 |  8,194,904 |  8,194,904 |  8,194,904 |
| `total_cycles        ` |  147,040 |  147,040 |  147,040 |  147,040 |
| `execute_time_ms     ` | <span style='color: green'>(-1 [-1.1%])</span> 94 | <span style='color: green'>(-1 [-1.1%])</span> 94 | <span style='color: green'>(-1 [-1.1%])</span> 94 | <span style='color: green'>(-1 [-1.1%])</span> 94 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-25 [-2.1%])</span> 1,164 | <span style='color: green'>(-25 [-2.1%])</span> 1,164 | <span style='color: green'>(-25 [-2.1%])</span> 1,164 | <span style='color: green'>(-25 [-2.1%])</span> 1,164 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-12.0%])</span> 22 | <span style='color: green'>(-3 [-12.0%])</span> 22 | <span style='color: green'>(-3 [-12.0%])</span> 22 | <span style='color: green'>(-3 [-12.0%])</span> 22 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-4 [-2.1%])</span> 184 | <span style='color: green'>(-4 [-2.1%])</span> 184 | <span style='color: green'>(-4 [-2.1%])</span> 184 | <span style='color: green'>(-4 [-2.1%])</span> 184 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.8%])</span> 132 | <span style='color: red'>(+1 [+0.8%])</span> 132 | <span style='color: red'>(+1 [+0.8%])</span> 132 | <span style='color: red'>(+1 [+0.8%])</span> 132 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.8%])</span> 263 | <span style='color: red'>(+2 [+0.8%])</span> 263 | <span style='color: red'>(+2 [+0.8%])</span> 263 | <span style='color: red'>(+2 [+0.8%])</span> 263 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-14 [-3.8%])</span> 351 | <span style='color: green'>(-14 [-3.8%])</span> 351 | <span style='color: green'>(-14 [-3.8%])</span> 351 | <span style='color: green'>(-14 [-3.8%])</span> 351 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 62 | 3 | 13 | 0 | 31 | 13 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 39 | 60 |  | 
| JalRangeCheckAir |  | 4 |  | 9 | 11 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 136 | 533 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 5 | 9 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 104 | 1,362 | 147,040 | 23,947,938 | 1,164 | 132 | 263 | 184 | 351 | 210 | 8,194,904 | 22 | 94 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 32,768 |  | 12 | 11 | 753,664 | 
| verify_fibair | AccessAdapterAir<4> | 16,384 |  | 12 | 13 | 409,600 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 44 | 27 | 72,704 | 
| verify_fibair | JalRangeCheckAir | 16,384 |  | 16 | 12 | 458,752 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 8,192 |  | 8 | 6 | 114,688 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 16,384 |  | 16 | 23 | 638,976 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 21 | 1,474,560 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 27 | 835,584 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 12 | 5 | 34 | 
| verify_fibair | VolatileBoundaryAir | 32,768 |  | 8 | 11 | 622,592 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/caa88cb88e8a70636293ea97a34bede8d2208234

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13792586411)
