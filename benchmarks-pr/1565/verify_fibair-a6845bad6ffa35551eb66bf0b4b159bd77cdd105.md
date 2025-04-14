| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+86.6%])</span> 2.41 | <span style='color: red'>(+1 [+86.6%])</span> 2.41 |
| verify_fibair | <span style='color: red'>(+1 [+86.6%])</span> 2.41 | <span style='color: red'>(+1 [+86.6%])</span> 2.41 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1118 [+86.6%])</span> 2,409 | <span style='color: red'>(+1118 [+86.6%])</span> 2,409 | <span style='color: red'>(+1118 [+86.6%])</span> 2,409 | <span style='color: red'>(+1118 [+86.6%])</span> 2,409 |
| `main_cells_used     ` |  17,675,690 |  17,675,690 |  17,675,690 |  17,675,690 |
| `total_cycles        ` |  334,008 |  334,008 |  334,008 |  334,008 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-1.1%])</span> 187 | <span style='color: green'>(-2 [-1.1%])</span> 187 | <span style='color: green'>(-2 [-1.1%])</span> 187 | <span style='color: green'>(-2 [-1.1%])</span> 187 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.5%])</span> 183 | <span style='color: green'>(-1 [-0.5%])</span> 183 | <span style='color: green'>(-1 [-0.5%])</span> 183 | <span style='color: green'>(-1 [-0.5%])</span> 183 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+1121 [+122.1%])</span> 2,039 | <span style='color: red'>(+1121 [+122.1%])</span> 2,039 | <span style='color: red'>(+1121 [+122.1%])</span> 2,039 | <span style='color: red'>(+1121 [+122.1%])</span> 2,039 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-10 [-6.1%])</span> 153 | <span style='color: green'>(-10 [-6.1%])</span> 153 | <span style='color: green'>(-10 [-6.1%])</span> 153 | <span style='color: green'>(-10 [-6.1%])</span> 153 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+2.5%])</span> 41 | <span style='color: red'>(+1 [+2.5%])</span> 41 | <span style='color: red'>(+1 [+2.5%])</span> 41 | <span style='color: red'>(+1 [+2.5%])</span> 41 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-105 [-52.8%])</span> 94 | <span style='color: green'>(-105 [-52.8%])</span> 94 | <span style='color: green'>(-105 [-52.8%])</span> 94 | <span style='color: green'>(-105 [-52.8%])</span> 94 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+195 [+171.1%])</span> 309 | <span style='color: red'>(+195 [+171.1%])</span> 309 | <span style='color: red'>(+195 [+171.1%])</span> 309 | <span style='color: red'>(+195 [+171.1%])</span> 309 |
| `quotient_poly_commit_time_ms` |  119 |  119 |  119 |  119 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+394 [+141.2%])</span> 673 | <span style='color: red'>(+394 [+141.2%])</span> 673 | <span style='color: red'>(+394 [+141.2%])</span> 673 | <span style='color: red'>(+394 [+141.2%])</span> 673 |
| `sumcheck_prove_batch_ms` |  449 |  449 |  449 |  449 |
| `gkr_prove_batch_ms  ` |  559 |  559 |  559 |  559 |
| `gkr_gen_layers_ms   ` |  57 |  57 |  57 |  57 |
| `gkr_generate_aux    ` |  76 |  76 |  76 |  76 |
| `gkr_build_instances_ms` |  20 |  20 |  20 |  20 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 42 | 5 | 8 | 0 | 22 | 6 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 10 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 10 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 10 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 52 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 9 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 496 |  | 
| PhantomAir |  | 2 |  | 3 | 4 |  | 
| ProgramAir |  | 2 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 2 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 20 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 24 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 12 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 12 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 20 |  | 
| VmConnectorAir |  | 2 |  | 5 | 9 |  | 
| VolatileBoundaryAir |  | 2 |  | 7 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_generate_aux | gkr_gen_layers_ms | gkr_build_instances_ms | generate_perm_trace_time_ms | execute_time_ms | build_gkr_input_layer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 183 | 2,409 | 334,008 | 41,387,682 | 449 | 2,039 | 309 | 119 | 94 | 673 | 153 | 17,675,690 | 559 | 76 | 57 | 20 | 41 | 187 | 30 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 12 | 11 | 3,014,656 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 12 | 13 | 1,638,400 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 12 | 27 | 79,872 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 12 | 12 | 786,432 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 12 | 398 | 13,434,880 | 
| verify_fibair | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | ProgramAir | 8,192 |  | 12 | 10 | 180,224 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 12 | 29 | 10,747,904 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 12 | 23 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 12 | 21 | 2,162,688 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 12 | 27 | 1,277,952 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 12 | 38 | 1,638,400 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 12 | 5 | 34 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 12 | 12 | 1,572,864 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| verify_fibair | 0 | 1,085,444 | 2,013,265,921 | 
| verify_fibair | 1 | 5,411,200 | 2,013,265,921 | 
| verify_fibair | 2 | 542,722 | 2,013,265,921 | 
| verify_fibair | 3 | 5,476,612 | 2,013,265,921 | 
| verify_fibair | 4 | 65,536 | 2,013,265,921 | 
| verify_fibair | 5 | 12,851,850 | 2,013,265,921 | 

| trace_height_constraint | threshold |
| --- | --- |
| 0 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/a6845bad6ffa35551eb66bf0b4b159bd77cdd105

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14449987733)
