| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: green'>(-0 [-8.6%])</span> 1.03 | <span style='color: green'>(-0 [-8.6%])</span> 1.03 | 1.03 |
| verify_fibair | <span style='color: green'>(-0 [-8.6%])</span> 1.03 | <span style='color: green'>(-0 [-8.6%])</span> 1.03 |  1.03 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-97 [-8.6%])</span> 1,031 | <span style='color: green'>(-97 [-8.6%])</span> 1,031 | <span style='color: green'>(-97 [-8.6%])</span> 1,031 | <span style='color: green'>(-97 [-8.6%])</span> 1,031 |
| `main_cells_used     ` |  3,615,122 |  3,615,122 |  3,615,122 |  3,615,122 |
| `total_cells_used    ` |  11,078,572 |  11,078,572 |  11,078,572 |  11,078,572 |
| `execute_preflight_insns` |  622,084 |  622,084 |  622,084 |  622,084 |
| `execute_preflight_time_ms` | <span style='color: red'>(+1 [+0.8%])</span> 128 | <span style='color: red'>(+1 [+0.8%])</span> 128 | <span style='color: red'>(+1 [+0.8%])</span> 128 | <span style='color: red'>(+1 [+0.8%])</span> 128 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-0.8%])</span> 5.09 | -          | <span style='color: green'>(-0 [-0.8%])</span> 5.09 | <span style='color: green'>(-0 [-0.8%])</span> 5.09 |
| `trace_gen_time_ms   ` |  37 |  37 |  37 |  37 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-20.0%])</span> 4 | <span style='color: green'>(-1 [-20.0%])</span> 4 | <span style='color: green'>(-1 [-20.0%])</span> 4 | <span style='color: green'>(-1 [-20.0%])</span> 4 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-98 [-10.2%])</span> 865 | <span style='color: green'>(-98 [-10.2%])</span> 865 | <span style='color: green'>(-98 [-10.2%])</span> 865 | <span style='color: green'>(-98 [-10.2%])</span> 865 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-16 [-44.4%])</span> 20 | <span style='color: green'>(-16 [-44.4%])</span> 20 | <span style='color: green'>(-16 [-44.4%])</span> 20 | <span style='color: green'>(-16 [-44.4%])</span> 20 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.1%])</span> 52.29 | <span style='color: green'>(-1 [-1.1%])</span> 52.29 | <span style='color: green'>(-1 [-1.1%])</span> 52.29 | <span style='color: green'>(-1 [-1.1%])</span> 52.29 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-10.0%])</span> 41.86 | <span style='color: green'>(-5 [-10.0%])</span> 41.86 | <span style='color: green'>(-5 [-10.0%])</span> 41.86 | <span style='color: green'>(-5 [-10.0%])</span> 41.86 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-4.4%])</span> 13.37 | <span style='color: green'>(-1 [-4.4%])</span> 13.37 | <span style='color: green'>(-1 [-4.4%])</span> 13.37 | <span style='color: green'>(-1 [-4.4%])</span> 13.37 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-75 [-9.8%])</span> 688 | <span style='color: green'>(-75 [-9.8%])</span> 688 | <span style='color: green'>(-75 [-9.8%])</span> 688 | <span style='color: green'>(-75 [-9.8%])</span> 688 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | verify_fibair_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | main_trace_commit_time_ms | generate_perm_trace_time_ms | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 9 | 1,031 | 65,536 | 600 | 0.13 | 0.73 | 2 | 0 | 594 | 0 | 594 | 3 | 0 | 1 | 0 | 0 | 594 | 

| air_id | air_name | rows | main_cols | cells |
| --- | --- | --- | --- | --- |
| 0 | FibonacciAir | 32,768 | 2 | 65,536 | 

| air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- |
| AccessAdapterAir<2> | 2 | 5 | 12 | 
| AccessAdapterAir<4> | 2 | 5 | 12 | 
| AccessAdapterAir<8> | 2 | 5 | 12 | 
| FibonacciAir | 1 |  | 5 | 
| FriReducedOpeningAir | 2 | 39 | 71 | 
| JalRangeCheckAir | 2 | 9 | 14 | 
| NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 574 | 
| PhantomAir | 2 | 3 | 6 | 
| ProgramAir | 1 | 1 | 4 | 
| VariableRangeCheckerAir | 1 | 1 | 4 | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 29 | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| VmConnectorAir | 2 | 5 | 12 | 
| VolatileBoundaryAir | 2 | 7 | 19 | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 37 | 1,031 | 11,078,572 | 120,541,482 | 37 | 865 | 0 | 41.86 | 13.37 | 5 | 52.29 | 688 | 73 | 688 | 4 | 46 | 3,615,122 | 20 | 1 | 128 | 622,084 | 5.09 | 15 | 55 | 0 | 688 | 

| group | air_id | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 0 | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | 1 | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 524,288 |  | 36 | 29 | 34,078,720 | 
| verify_fibair | 11 | JalRangeCheckAir | 65,536 |  | 28 | 12 | 2,621,440 | 
| verify_fibair | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 65,536 |  | 28 | 23 | 3,342,336 | 
| verify_fibair | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 65,536 |  | 40 | 27 | 4,390,912 | 
| verify_fibair | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 40 | 21 | 7,995,392 | 
| verify_fibair | 15 | PhantomAir | 32,768 |  | 12 | 6 | 589,824 | 
| verify_fibair | 16 | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | 3 | VolatileBoundaryAir | 131,072 |  | 20 | 12 | 4,194,304 | 
| verify_fibair | 4 | AccessAdapterAir<2> | 262,144 |  | 16 | 11 | 7,077,888 | 
| verify_fibair | 5 | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | 6 | AccessAdapterAir<8> | 256 |  | 16 | 17 | 8,448 | 
| verify_fibair | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 65,536 |  | 312 | 398 | 46,530,560 | 
| verify_fibair | 8 | FriReducedOpeningAir | 4,096 |  | 84 | 27 | 454,656 | 
| verify_fibair | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 65,536 |  | 36 | 38 | 4,849,664 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| verify_fibair | 0 | 2,170,884 | 2,013,265,921 | 
| verify_fibair | 1 | 10,625,792 | 2,013,265,921 | 
| verify_fibair | 2 | 1,085,442 | 2,013,265,921 | 
| verify_fibair | 3 | 10,822,148 | 2,013,265,921 | 
| verify_fibair | 4 | 131,072 | 2,013,265,921 | 
| verify_fibair | 5 | 25,105,674 | 2,013,265,921 | 

| trace_height_constraint | threshold |
| --- | --- |
| 0 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/eda30f68985a549ec1a04df3d5c77cdea666976b

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25931777645)
