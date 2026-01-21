| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: red'>(+4 [+1794.7%])</span> 4.30 | <span style='color: red'>(+4 [+1794.7%])</span> 4.30 | 4.30 |
| verify_fibair | <span style='color: red'>(+4 [+1794.7%])</span> 4.30 | <span style='color: red'>(+4 [+1794.7%])</span> 4.30 |  4.30 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4074 [+1794.7%])</span> 4,301 | <span style='color: red'>(+4074 [+1794.7%])</span> 4,301 | <span style='color: red'>(+4074 [+1794.7%])</span> 4,301 | <span style='color: red'>(+4074 [+1794.7%])</span> 4,301 |
| `main_cells_used     ` | <span style='color: red'>(+1555558 [+75.6%])</span> 3,614,212 | <span style='color: red'>(+1555558 [+75.6%])</span> 3,614,212 | <span style='color: red'>(+1555558 [+75.6%])</span> 3,614,212 | <span style='color: red'>(+1555558 [+75.6%])</span> 3,614,212 |
| `total_cells_used    ` | <span style='color: red'>(+3820990 [+52.7%])</span> 11,076,882 | <span style='color: red'>(+3820990 [+52.7%])</span> 11,076,882 | <span style='color: red'>(+3820990 [+52.7%])</span> 11,076,882 | <span style='color: red'>(+3820990 [+52.7%])</span> 11,076,882 |
| `execute_preflight_insns` | <span style='color: red'>(+299370 [+92.8%])</span> 621,980 | <span style='color: red'>(+299370 [+92.8%])</span> 621,980 | <span style='color: red'>(+299370 [+92.8%])</span> 621,980 | <span style='color: red'>(+299370 [+92.8%])</span> 621,980 |
| `execute_preflight_time_ms` | <span style='color: red'>(+63 [+88.7%])</span> 134 | <span style='color: red'>(+63 [+88.7%])</span> 134 | <span style='color: red'>(+63 [+88.7%])</span> 134 | <span style='color: red'>(+63 [+88.7%])</span> 134 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-0.5%])</span> 4.83 | -          | <span style='color: green'>(-0 [-0.5%])</span> 4.83 | <span style='color: green'>(-0 [-0.5%])</span> 4.83 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+15 [+65.2%])</span> 38 | <span style='color: red'>(+15 [+65.2%])</span> 38 | <span style='color: red'>(+15 [+65.2%])</span> 38 | <span style='color: red'>(+15 [+65.2%])</span> 38 |
| `memory_finalize_time_ms` | <span style='color: red'>(+1 [+33.3%])</span> 4 | <span style='color: red'>(+1 [+33.3%])</span> 4 | <span style='color: red'>(+1 [+33.3%])</span> 4 | <span style='color: red'>(+1 [+33.3%])</span> 4 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+3994 [+3003.0%])</span> 4,127 | <span style='color: red'>(+3994 [+3003.0%])</span> 4,127 | <span style='color: red'>(+3994 [+3003.0%])</span> 4,127 | <span style='color: red'>(+3994 [+3003.0%])</span> 4,127 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 | <span style='color: red'>(+23 [+95.8%])</span> 47 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+39 [+243.8%])</span> 55 | <span style='color: red'>(+39 [+243.8%])</span> 55 | <span style='color: red'>(+39 [+243.8%])</span> 55 | <span style='color: red'>(+39 [+243.8%])</span> 55 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+24 [+89.8%])</span> 51.76 | <span style='color: red'>(+24 [+89.8%])</span> 51.76 | <span style='color: red'>(+24 [+89.8%])</span> 51.76 | <span style='color: red'>(+24 [+89.8%])</span> 51.76 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+27 [+120.7%])</span> 48.76 | <span style='color: red'>(+27 [+120.7%])</span> 48.76 | <span style='color: red'>(+27 [+120.7%])</span> 48.76 | <span style='color: red'>(+27 [+120.7%])</span> 48.76 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+5 [+66.9%])</span> 13.59 | <span style='color: red'>(+5 [+66.9%])</span> 13.59 | <span style='color: red'>(+5 [+66.9%])</span> 13.59 | <span style='color: red'>(+5 [+66.9%])</span> 13.59 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3876 [+11745.5%])</span> 3,909 | <span style='color: red'>(+3876 [+11745.5%])</span> 3,909 | <span style='color: red'>(+3876 [+11745.5%])</span> 3,909 | <span style='color: red'>(+3876 [+11745.5%])</span> 3,909 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | verify_fibair_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | main_trace_commit_time_ms | generate_perm_trace_time_ms | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 9 | 4,301 | 65,536 | 3,093 | 0.18 | 0.73 | 2 | 0 | 3,088 | 0 | 3,088 | 3 | 0 | 1 | 0 | 0 | 3,088 | 

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
| verify_fibair | 38 | 4,301 | 11,076,882 | 120,541,482 | 38 | 4,127 | 0 | 48.76 | 13.59 | 6 | 51.76 | 3,909 | 107 | 3,908 | 4 | 47 | 3,614,212 | 55 | 1 | 134 | 621,980 | 4.83 | 15 | 62 | 0 | 3,908 | 

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


Commit: https://github.com/openvm-org/openvm/commit/1280f8eb498b75d37a8ae6df0d2d34d7b1d36430

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/21192229467)
