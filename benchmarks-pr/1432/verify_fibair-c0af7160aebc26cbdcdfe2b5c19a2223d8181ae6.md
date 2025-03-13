| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.2%])</span> 1.26 | <span style='color: green'>(-0 [-1.2%])</span> 1.26 |
| verify_fibair | <span style='color: green'>(-0 [-1.2%])</span> 1.26 | <span style='color: green'>(-0 [-1.2%])</span> 1.26 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-15 [-1.2%])</span> 1,259 | <span style='color: green'>(-15 [-1.2%])</span> 1,259 | <span style='color: green'>(-15 [-1.2%])</span> 1,259 | <span style='color: green'>(-15 [-1.2%])</span> 1,259 |
| `main_cells_used     ` |  17,943,801 |  17,943,801 |  17,943,801 |  17,943,801 |
| `total_cycles        ` |  334,753 |  334,753 |  334,753 |  334,753 |
| `execute_time_ms     ` | <span style='color: red'>(+3 [+1.6%])</span> 193 | <span style='color: red'>(+3 [+1.6%])</span> 193 | <span style='color: red'>(+3 [+1.6%])</span> 193 | <span style='color: red'>(+3 [+1.6%])</span> 193 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-3 [-1.6%])</span> 181 | <span style='color: green'>(-3 [-1.6%])</span> 181 | <span style='color: green'>(-3 [-1.6%])</span> 181 | <span style='color: green'>(-3 [-1.6%])</span> 181 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-15 [-1.7%])</span> 885 | <span style='color: green'>(-15 [-1.7%])</span> 885 | <span style='color: green'>(-15 [-1.7%])</span> 885 | <span style='color: green'>(-15 [-1.7%])</span> 885 |
| `main_trace_commit_time_ms` |  164 |  164 |  164 |  164 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-5.3%])</span> 36 | <span style='color: green'>(-2 [-5.3%])</span> 36 | <span style='color: green'>(-2 [-5.3%])</span> 36 | <span style='color: green'>(-2 [-5.3%])</span> 36 |
| `perm_trace_commit_time_ms` |  200 |  200 |  200 |  200 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1 [-0.9%])</span> 110 | <span style='color: green'>(-1 [-0.9%])</span> 110 | <span style='color: green'>(-1 [-0.9%])</span> 110 | <span style='color: green'>(-1 [-0.9%])</span> 110 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-2 [-1.7%])</span> 114 | <span style='color: green'>(-2 [-1.7%])</span> 114 | <span style='color: green'>(-2 [-1.7%])</span> 114 | <span style='color: green'>(-2 [-1.7%])</span> 114 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-7 [-2.6%])</span> 258 | <span style='color: green'>(-7 [-2.6%])</span> 258 | <span style='color: green'>(-7 [-2.6%])</span> 258 | <span style='color: green'>(-7 [-2.6%])</span> 258 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 40 | 2 | 7 | 0 | 22 | 7 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 71 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 14 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 571 |  | 
| PhantomAir |  | 2 |  | 3 | 5 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 25 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 29 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmConnectorAir |  | 2 |  | 5 | 10 |  | 
| VolatileBoundaryAir |  | 2 |  | 4 | 17 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 181 | 1,259 | 334,753 | 61,917,354 | 885 | 110 | 114 | 200 | 258 | 164 | 17,943,801 | 36 | 193 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 84 | 27 | 227,328 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 28 | 12 | 1,310,720 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 312 | 399 | 23,298,048 | 
| verify_fibair | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 36 | 29 | 17,039,360 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 40 | 21 | 3,997,696 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 40 | 27 | 2,195,456 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 36 | 38 | 2,424,832 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 12 | 11 | 1,507,328 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/c0af7160aebc26cbdcdfe2b5c19a2223d8181ae6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13824388885)
