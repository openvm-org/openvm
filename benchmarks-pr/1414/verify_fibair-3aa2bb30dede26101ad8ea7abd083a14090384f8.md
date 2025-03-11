| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.9%])</span> 1.38 | <span style='color: green'>(-0 [-0.9%])</span> 1.38 |
| verify_fibair | <span style='color: green'>(-0 [-0.9%])</span> 1.38 | <span style='color: green'>(-0 [-0.9%])</span> 1.38 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-12 [-0.9%])</span> 1,377 | <span style='color: green'>(-12 [-0.9%])</span> 1,377 | <span style='color: green'>(-12 [-0.9%])</span> 1,377 | <span style='color: green'>(-12 [-0.9%])</span> 1,377 |
| `main_cells_used     ` |  8,194,904 |  8,194,904 |  8,194,904 |  8,194,904 |
| `total_cycles        ` |  147,040 |  147,040 |  147,040 |  147,040 |
| `execute_time_ms     ` |  95 |  95 |  95 |  95 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 | <span style='color: green'>(-1 [-1.0%])</span> 104 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-11 [-0.9%])</span> 1,178 | <span style='color: green'>(-11 [-0.9%])</span> 1,178 | <span style='color: green'>(-11 [-0.9%])</span> 1,178 | <span style='color: green'>(-11 [-0.9%])</span> 1,178 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 | <span style='color: green'>(-5 [-2.3%])</span> 210 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-16.0%])</span> 21 | <span style='color: green'>(-4 [-16.0%])</span> 21 | <span style='color: green'>(-4 [-16.0%])</span> 21 | <span style='color: green'>(-4 [-16.0%])</span> 21 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+2.1%])</span> 192 | <span style='color: red'>(+4 [+2.1%])</span> 192 | <span style='color: red'>(+4 [+2.1%])</span> 192 | <span style='color: red'>(+4 [+2.1%])</span> 192 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-9 [-6.9%])</span> 122 | <span style='color: green'>(-9 [-6.9%])</span> 122 | <span style='color: green'>(-9 [-6.9%])</span> 122 | <span style='color: green'>(-9 [-6.9%])</span> 122 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+11 [+4.2%])</span> 272 | <span style='color: red'>(+11 [+4.2%])</span> 272 | <span style='color: red'>(+11 [+4.2%])</span> 272 | <span style='color: red'>(+11 [+4.2%])</span> 272 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-2.2%])</span> 357 | <span style='color: green'>(-8 [-2.2%])</span> 357 | <span style='color: green'>(-8 [-2.2%])</span> 357 | <span style='color: green'>(-8 [-2.2%])</span> 357 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 62 | 2 | 14 | 0 | 31 | 13 | 

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
| verify_fibair | 104 | 1,377 | 147,040 | 23,947,938 | 1,178 | 122 | 272 | 192 | 357 | 210 | 8,194,904 | 21 | 95 | 

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


Commit: https://github.com/openvm-org/openvm/commit/3aa2bb30dede26101ad8ea7abd083a14090384f8

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13799121581)
