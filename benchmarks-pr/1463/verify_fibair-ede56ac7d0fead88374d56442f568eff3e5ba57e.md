| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+8.1%])</span> 1.34 | <span style='color: red'>(+0 [+8.1%])</span> 1.34 |
| verify_fibair | <span style='color: red'>(+0 [+8.1%])</span> 1.34 | <span style='color: red'>(+0 [+8.1%])</span> 1.34 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+100 [+8.1%])</span> 1,339 | <span style='color: red'>(+100 [+8.1%])</span> 1,339 | <span style='color: red'>(+100 [+8.1%])</span> 1,339 | <span style='color: red'>(+100 [+8.1%])</span> 1,339 |
| `main_cells_used     ` | <span style='color: green'>(-9611346 [-53.7%])</span> 8,290,183 | <span style='color: green'>(-9611346 [-53.7%])</span> 8,290,183 | <span style='color: green'>(-9611346 [-53.7%])</span> 8,290,183 | <span style='color: green'>(-9611346 [-53.7%])</span> 8,290,183 |
| `total_cycles        ` | <span style='color: green'>(-190402 [-57.0%])</span> 143,584 | <span style='color: green'>(-190402 [-57.0%])</span> 143,584 | <span style='color: green'>(-190402 [-57.0%])</span> 143,584 | <span style='color: green'>(-190402 [-57.0%])</span> 143,584 |
| `execute_time_ms     ` | <span style='color: green'>(-99 [-50.5%])</span> 97 | <span style='color: green'>(-99 [-50.5%])</span> 97 | <span style='color: green'>(-99 [-50.5%])</span> 97 | <span style='color: green'>(-99 [-50.5%])</span> 97 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-74 [-41.1%])</span> 106 | <span style='color: green'>(-74 [-41.1%])</span> 106 | <span style='color: green'>(-74 [-41.1%])</span> 106 | <span style='color: green'>(-74 [-41.1%])</span> 106 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+273 [+31.6%])</span> 1,136 | <span style='color: red'>(+273 [+31.6%])</span> 1,136 | <span style='color: red'>(+273 [+31.6%])</span> 1,136 | <span style='color: red'>(+273 [+31.6%])</span> 1,136 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+56 [+37.1%])</span> 207 | <span style='color: red'>(+56 [+37.1%])</span> 207 | <span style='color: red'>(+56 [+37.1%])</span> 207 | <span style='color: red'>(+56 [+37.1%])</span> 207 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-12 [-34.3%])</span> 23 | <span style='color: green'>(-12 [-34.3%])</span> 23 | <span style='color: green'>(-12 [-34.3%])</span> 23 | <span style='color: green'>(-12 [-34.3%])</span> 23 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-5 [-2.7%])</span> 181 | <span style='color: green'>(-5 [-2.7%])</span> 181 | <span style='color: green'>(-5 [-2.7%])</span> 181 | <span style='color: green'>(-5 [-2.7%])</span> 181 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+14 [+12.6%])</span> 125 | <span style='color: red'>(+14 [+12.6%])</span> 125 | <span style='color: red'>(+14 [+12.6%])</span> 125 | <span style='color: red'>(+14 [+12.6%])</span> 125 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+135 [+120.5%])</span> 247 | <span style='color: red'>(+135 [+120.5%])</span> 247 | <span style='color: red'>(+135 [+120.5%])</span> 247 | <span style='color: red'>(+135 [+120.5%])</span> 247 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+85 [+32.2%])</span> 349 | <span style='color: red'>(+85 [+32.2%])</span> 349 | <span style='color: red'>(+85 [+32.2%])</span> 349 | <span style='color: red'>(+85 [+32.2%])</span> 349 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 62 | 3 | 13 | 0 | 31 | 13 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 39 | 60 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 136 | 530 |  | 
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
| verify_fibair | 106 | 1,339 | 143,584 | 23,616,152 | 1,136 | 125 | 247 | 181 | 349 | 207 | 8,290,183 | 23 | 97 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 32,768 |  | 12 | 11 | 753,664 | 
| verify_fibair | AccessAdapterAir<4> | 16,384 |  | 12 | 13 | 409,600 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 44 | 27 | 72,704 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 16,384 |  | 16 | 23 | 638,976 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4,096 |  | 12 | 9 | 86,016 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 32,768 |  | 8 | 11 | 622,592 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ede56ac7d0fead88374d56442f568eff3e5ba57e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13909079269)
