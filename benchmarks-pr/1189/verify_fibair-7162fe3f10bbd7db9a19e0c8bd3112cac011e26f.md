| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-4.9%])</span> 1.42 | <span style='color: green'>(-0 [-4.9%])</span> 1.42 |
| verify_fibair | <span style='color: green'>(-0 [-4.9%])</span> 1.42 | <span style='color: green'>(-0 [-4.9%])</span> 1.42 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-73 [-4.9%])</span> 1,422 | <span style='color: green'>(-73 [-4.9%])</span> 1,422 | <span style='color: green'>(-73 [-4.9%])</span> 1,422 | <span style='color: green'>(-73 [-4.9%])</span> 1,422 |
| `main_cells_used     ` |  8,025,052 |  8,025,052 |  8,025,052 |  8,025,052 |
| `total_cycles        ` |  195,006 |  195,006 |  195,006 |  195,006 |
| `execute_time_ms     ` |  61 |  61 |  61 |  61 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-1.4%])</span> 139 | <span style='color: green'>(-2 [-1.4%])</span> 139 | <span style='color: green'>(-2 [-1.4%])</span> 139 | <span style='color: green'>(-2 [-1.4%])</span> 139 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-71 [-5.5%])</span> 1,222 | <span style='color: green'>(-71 [-5.5%])</span> 1,222 | <span style='color: green'>(-71 [-5.5%])</span> 1,222 | <span style='color: green'>(-71 [-5.5%])</span> 1,222 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.5%])</span> 211 | <span style='color: green'>(-1 [-0.5%])</span> 211 | <span style='color: green'>(-1 [-0.5%])</span> 211 | <span style='color: green'>(-1 [-0.5%])</span> 211 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+4 [+17.4%])</span> 27 | <span style='color: red'>(+4 [+17.4%])</span> 27 | <span style='color: red'>(+4 [+17.4%])</span> 27 | <span style='color: red'>(+4 [+17.4%])</span> 27 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-5 [-2.7%])</span> 178 | <span style='color: green'>(-5 [-2.7%])</span> 178 | <span style='color: green'>(-5 [-2.7%])</span> 178 | <span style='color: green'>(-5 [-2.7%])</span> 178 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-4 [-1.8%])</span> 217 | <span style='color: green'>(-4 [-1.8%])</span> 217 | <span style='color: green'>(-4 [-1.8%])</span> 217 | <span style='color: green'>(-4 [-1.8%])</span> 217 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-62 [-20.5%])</span> 241 | <span style='color: green'>(-62 [-20.5%])</span> 241 | <span style='color: green'>(-62 [-20.5%])</span> 241 | <span style='color: green'>(-62 [-20.5%])</span> 241 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-3 [-0.9%])</span> 345 | <span style='color: green'>(-3 [-0.9%])</span> 345 | <span style='color: green'>(-3 [-0.9%])</span> 345 | <span style='color: green'>(-3 [-0.9%])</span> 345 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 32 | 9 | 0 | 1 | 0 | 2 | 5 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 16 | 1 | 2 |  | 5 | 32 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 31 | 302 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 23 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 19 | 31 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 139 | 1,422 | 195,006 | 23,304,216 | 1,222 | 217 | 241 | 178 | 345 | 211 | 8,025,052 | 27 | 61 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 32,768 |  | 16 | 11 | 884,736 | 
| verify_fibair | AccessAdapterAir<4> | 16,384 |  | 16 | 13 | 475,136 | 
| verify_fibair | AccessAdapterAir<8> | 4,096 |  | 16 | 17 | 135,168 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 2,048 |  | 36 | 348 | 786,432 | 
| verify_fibair | PhantomAir | 2,048 |  | 8 | 6 | 28,672 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 8,192 |  | 12 | 10 | 180,224 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 131,072 |  | 20 | 30 | 6,553,600 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 24 | 41 | 8,519,680 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4,096 |  | 20 | 40 | 245,760 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 8 | 11 | 1,245,184 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/7162fe3f10bbd7db9a19e0c8bd3112cac011e26f

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12659659762)