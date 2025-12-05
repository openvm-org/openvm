| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+256.9%])</span> 0.80 | <span style='color: red'>(+1 [+256.9%])</span> 0.80 |
| verify_fibair | <span style='color: red'>(+1 [+256.9%])</span> 0.80 | <span style='color: red'>(+1 [+256.9%])</span> 0.80 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+578 [+256.9%])</span> 803 | <span style='color: red'>(+578 [+256.9%])</span> 803 | <span style='color: red'>(+578 [+256.9%])</span> 803 | <span style='color: red'>(+578 [+256.9%])</span> 803 |
| `main_cells_used     ` | <span style='color: red'>(+1538076 [+74.7%])</span> 3,596,730 | <span style='color: red'>(+1538076 [+74.7%])</span> 3,596,730 | <span style='color: red'>(+1538076 [+74.7%])</span> 3,596,730 | <span style='color: red'>(+1538076 [+74.7%])</span> 3,596,730 |
| `total_cells_used    ` | <span style='color: red'>(+3780948 [+52.1%])</span> 11,036,840 | <span style='color: red'>(+3780948 [+52.1%])</span> 11,036,840 | <span style='color: red'>(+3780948 [+52.1%])</span> 11,036,840 | <span style='color: red'>(+3780948 [+52.1%])</span> 11,036,840 |
| `execute_preflight_insns` | <span style='color: red'>(+292460 [+90.7%])</span> 615,070 | <span style='color: red'>(+292460 [+90.7%])</span> 615,070 | <span style='color: red'>(+292460 [+90.7%])</span> 615,070 | <span style='color: red'>(+292460 [+90.7%])</span> 615,070 |
| `execute_preflight_time_ms` | <span style='color: red'>(+66 [+91.7%])</span> 138 | <span style='color: red'>(+66 [+91.7%])</span> 138 | <span style='color: red'>(+66 [+91.7%])</span> 138 | <span style='color: red'>(+66 [+91.7%])</span> 138 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-0.8%])</span> 4.65 | -          | <span style='color: green'>(-0 [-0.8%])</span> 4.65 | <span style='color: green'>(-0 [-0.8%])</span> 4.65 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+18 [+81.8%])</span> 40 | <span style='color: red'>(+18 [+81.8%])</span> 40 | <span style='color: red'>(+18 [+81.8%])</span> 40 | <span style='color: red'>(+18 [+81.8%])</span> 40 |
| `memory_finalize_time_ms` | <span style='color: red'>(+2 [+100.0%])</span> 4 | <span style='color: red'>(+2 [+100.0%])</span> 4 | <span style='color: red'>(+2 [+100.0%])</span> 4 | <span style='color: red'>(+2 [+100.0%])</span> 4 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+494 [+382.9%])</span> 623 | <span style='color: red'>(+494 [+382.9%])</span> 623 | <span style='color: red'>(+494 [+382.9%])</span> 623 | <span style='color: red'>(+494 [+382.9%])</span> 623 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+23 [+143.8%])</span> 39 | <span style='color: red'>(+23 [+143.8%])</span> 39 | <span style='color: red'>(+23 [+143.8%])</span> 39 | <span style='color: red'>(+23 [+143.8%])</span> 39 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+26 [+97.2%])</span> 52.73 | <span style='color: red'>(+26 [+97.2%])</span> 52.73 | <span style='color: red'>(+26 [+97.2%])</span> 52.73 | <span style='color: red'>(+26 [+97.2%])</span> 52.73 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+28 [+130.8%])</span> 48.54 | <span style='color: red'>(+28 [+130.8%])</span> 48.54 | <span style='color: red'>(+28 [+130.8%])</span> 48.54 | <span style='color: red'>(+28 [+130.8%])</span> 48.54 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+7 [+87.7%])</span> 14.76 | <span style='color: red'>(+7 [+87.7%])</span> 14.76 | <span style='color: red'>(+7 [+87.7%])</span> 14.76 | <span style='color: red'>(+7 [+87.7%])</span> 14.76 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+388 [+1251.6%])</span> 419 | <span style='color: red'>(+388 [+1251.6%])</span> 419 | <span style='color: red'>(+388 [+1251.6%])</span> 419 | <span style='color: red'>(+388 [+1251.6%])</span> 419 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | verify_fibair_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | main_trace_commit_time_ms | generate_perm_trace_time_ms | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 10 | 803 | 65,536 | 47 | 0.13 | 0.75 | 2 | 0 | 43 | 0 | 42 | 3 | 0 | 1 | 0 | 0 | 42 | 

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
| NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 572 | 
| PhantomAir | 2 | 3 | 5 | 
| ProgramAir | 1 | 1 | 4 | 
| VariableRangeCheckerAir | 1 | 1 | 4 | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 29 | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| VmConnectorAir | 2 | 5 | 11 | 
| VolatileBoundaryAir | 2 | 7 | 19 | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 40 | 803 | 11,036,840 | 120,541,482 | 40 | 623 | 0 | 48.54 | 14.76 | 6 | 52.73 | 419 | 92 | 419 | 4 | 47 | 3,596,730 | 39 | 1 | 138 | 615,070 | 4.65 | 15 | 63 | 0 | 419 | 

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


Commit: https://github.com/openvm-org/openvm/commit/df6399ca01d48f46ae6807d1217b84cb4b93b84d

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/19968228435)
