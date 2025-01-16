| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-10.4%])</span> 3.40 | <span style='color: green'>(-0 [-10.4%])</span> 3.40 |
| verify_fibair | <span style='color: green'>(-0 [-10.4%])</span> 3.40 | <span style='color: green'>(-0 [-10.4%])</span> 3.40 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-393 [-10.4%])</span> 3,401 | <span style='color: green'>(-393 [-10.4%])</span> 3,401 | <span style='color: green'>(-393 [-10.4%])</span> 3,401 | <span style='color: green'>(-393 [-10.4%])</span> 3,401 |
| `main_cells_used     ` | <span style='color: green'>(-3525909 [-11.7%])</span> 26,487,405 | <span style='color: green'>(-3525909 [-11.7%])</span> 26,487,405 | <span style='color: green'>(-3525909 [-11.7%])</span> 26,487,405 | <span style='color: green'>(-3525909 [-11.7%])</span> 26,487,405 |
| `total_cycles        ` | <span style='color: red'>(+3486 [+0.5%])</span> 749,819 | <span style='color: red'>(+3486 [+0.5%])</span> 749,819 | <span style='color: red'>(+3486 [+0.5%])</span> 749,819 | <span style='color: red'>(+3486 [+0.5%])</span> 749,819 |
| `execute_time_ms     ` | <span style='color: green'>(-7 [-4.5%])</span> 150 | <span style='color: green'>(-7 [-4.5%])</span> 150 | <span style='color: green'>(-7 [-4.5%])</span> 150 | <span style='color: green'>(-7 [-4.5%])</span> 150 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-16 [-3.6%])</span> 429 | <span style='color: green'>(-16 [-3.6%])</span> 429 | <span style='color: green'>(-16 [-3.6%])</span> 429 | <span style='color: green'>(-16 [-3.6%])</span> 429 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-370 [-11.6%])</span> 2,822 | <span style='color: green'>(-370 [-11.6%])</span> 2,822 | <span style='color: green'>(-370 [-11.6%])</span> 2,822 | <span style='color: green'>(-370 [-11.6%])</span> 2,822 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-95 [-15.4%])</span> 522 | <span style='color: green'>(-95 [-15.4%])</span> 522 | <span style='color: green'>(-95 [-15.4%])</span> 522 | <span style='color: green'>(-95 [-15.4%])</span> 522 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-25 [-28.4%])</span> 63 | <span style='color: green'>(-25 [-28.4%])</span> 63 | <span style='color: green'>(-25 [-28.4%])</span> 63 | <span style='color: green'>(-25 [-28.4%])</span> 63 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-77 [-14.4%])</span> 459 | <span style='color: green'>(-77 [-14.4%])</span> 459 | <span style='color: green'>(-77 [-14.4%])</span> 459 | <span style='color: green'>(-77 [-14.4%])</span> 459 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-64 [-10.0%])</span> 575 | <span style='color: green'>(-64 [-10.0%])</span> 575 | <span style='color: green'>(-64 [-10.0%])</span> 575 | <span style='color: green'>(-64 [-10.0%])</span> 575 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-44 [-7.7%])</span> 529 | <span style='color: green'>(-44 [-7.7%])</span> 529 | <span style='color: green'>(-44 [-7.7%])</span> 529 | <span style='color: green'>(-44 [-7.7%])</span> 529 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-65 [-8.8%])</span> 671 | <span style='color: green'>(-65 [-8.8%])</span> 671 | <span style='color: green'>(-65 [-8.8%])</span> 671 | <span style='color: green'>(-65 [-8.8%])</span> 671 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 66 | 3 | 13 | 0 | 32 | 17 | 

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
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 18 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 429 | 3,401 | 749,819 | 68,081,688 | 2,822 | 575 | 529 | 459 | 671 | 522 | 26,487,405 | 63 | 150 | 

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
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 20 | 27 | 12,320,768 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f745778b48fe4b2a01a999b58fdd540ebe7f32bf

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12706008592)