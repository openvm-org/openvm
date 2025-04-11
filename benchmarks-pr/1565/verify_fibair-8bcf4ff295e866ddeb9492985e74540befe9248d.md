| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+84.3%])</span> 2.38 | <span style='color: red'>(+1 [+84.3%])</span> 2.38 |
| verify_fibair | <span style='color: red'>(+1 [+84.3%])</span> 2.38 | <span style='color: red'>(+1 [+84.3%])</span> 2.38 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1088 [+84.3%])</span> 2,378 | <span style='color: red'>(+1088 [+84.3%])</span> 2,378 | <span style='color: red'>(+1088 [+84.3%])</span> 2,378 | <span style='color: red'>(+1088 [+84.3%])</span> 2,378 |
| `main_cells_used     ` |  17,676,626 |  17,676,626 |  17,676,626 |  17,676,626 |
| `total_cycles        ` |  334,086 |  334,086 |  334,086 |  334,086 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+1.1%])</span> 190 | <span style='color: red'>(+2 [+1.1%])</span> 190 | <span style='color: red'>(+2 [+1.1%])</span> 190 | <span style='color: red'>(+2 [+1.1%])</span> 190 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+3 [+1.6%])</span> 185 | <span style='color: red'>(+3 [+1.6%])</span> 185 | <span style='color: red'>(+3 [+1.6%])</span> 185 | <span style='color: red'>(+3 [+1.6%])</span> 185 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+1083 [+117.7%])</span> 2,003 | <span style='color: red'>(+1083 [+117.7%])</span> 2,003 | <span style='color: red'>(+1083 [+117.7%])</span> 2,003 | <span style='color: red'>(+1083 [+117.7%])</span> 2,003 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-9 [-5.5%])</span> 154 | <span style='color: green'>(-9 [-5.5%])</span> 154 | <span style='color: green'>(-9 [-5.5%])</span> 154 | <span style='color: green'>(-9 [-5.5%])</span> 154 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-7.9%])</span> 35 | <span style='color: green'>(-3 [-7.9%])</span> 35 | <span style='color: green'>(-3 [-7.9%])</span> 35 | <span style='color: green'>(-3 [-7.9%])</span> 35 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-107 [-52.7%])</span> 96 | <span style='color: green'>(-107 [-52.7%])</span> 96 | <span style='color: green'>(-107 [-52.7%])</span> 96 | <span style='color: green'>(-107 [-52.7%])</span> 96 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+202 [+174.1%])</span> 318 | <span style='color: red'>(+202 [+174.1%])</span> 318 | <span style='color: red'>(+202 [+174.1%])</span> 318 | <span style='color: red'>(+202 [+174.1%])</span> 318 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-3 [-2.4%])</span> 121 | <span style='color: green'>(-3 [-2.4%])</span> 121 | <span style='color: green'>(-3 [-2.4%])</span> 121 | <span style='color: green'>(-3 [-2.4%])</span> 121 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+433 [+159.8%])</span> 704 | <span style='color: red'>(+433 [+159.8%])</span> 704 | <span style='color: red'>(+433 [+159.8%])</span> 704 | <span style='color: red'>(+433 [+159.8%])</span> 704 |
| `sumcheck_prove_batch_ms` |  386 |  386 |  386 |  386 |
| `gkr_prove_batch_ms  ` |  482 |  482 |  482 |  482 |
| `gkr_gen_layers_ms   ` |  52 |  52 |  52 |  52 |
| `gkr_generate_aux    ` |  70 |  70 |  70 |  70 |
| `gkr_build_instances_ms` |  19 |  19 |  19 |  19 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 43 | 5 | 7 | 0 | 23 | 6 | 

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
| verify_fibair | 185 | 2,378 | 334,086 | 41,387,682 | 386 | 2,003 | 318 | 121 | 96 | 704 | 154 | 17,676,626 | 482 | 70 | 52 | 19 | 35 | 190 | 32 | 

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


Commit: https://github.com/openvm-org/openvm/commit/8bcf4ff295e866ddeb9492985e74540befe9248d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14406709434)
