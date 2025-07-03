| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-2.8%])</span> 1.42 | <span style='color: green'>(-0 [-2.8%])</span> 1.42 |
| verify_fibair | <span style='color: green'>(-0 [-3.2%])</span> 1.25 | <span style='color: green'>(-0 [-3.2%])</span> 1.25 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-42 [-3.2%])</span> 1,254 | <span style='color: green'>(-42 [-3.2%])</span> 1,254 | <span style='color: green'>(-42 [-3.2%])</span> 1,254 | <span style='color: green'>(-42 [-3.2%])</span> 1,254 |
| `main_cells_used     ` |  17,271,190 |  17,271,190 |  17,271,190 |  17,271,190 |
| `total_cycles        ` |  322,699 |  322,699 |  322,699 |  322,699 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+0.6%])</span> 162 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-0.8%])</span> 1.98 | -          | <span style='color: green'>(-0 [-0.8%])</span> 1.98 | <span style='color: green'>(-0 [-0.8%])</span> 1.98 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-1 [-0.6%])</span> 174 | <span style='color: green'>(-1 [-0.6%])</span> 174 | <span style='color: green'>(-1 [-0.6%])</span> 174 | <span style='color: green'>(-1 [-0.6%])</span> 174 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+0 [+0.8%])</span> 1.85 | -          | <span style='color: red'>(+0 [+0.8%])</span> 1.85 | <span style='color: red'>(+0 [+0.8%])</span> 1.85 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 | <span style='color: red'>(+2 [+4.0%])</span> 52 |
| `memory_finalize_time_ms` | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 | <span style='color: red'>(+1 [+20.0%])</span> 6 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-44 [-4.8%])</span> 866 | <span style='color: green'>(-44 [-4.8%])</span> 866 | <span style='color: green'>(-44 [-4.8%])</span> 866 | <span style='color: green'>(-44 [-4.8%])</span> 866 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+1 [+0.6%])</span> 161 | <span style='color: red'>(+1 [+0.6%])</span> 161 | <span style='color: red'>(+1 [+0.6%])</span> 161 | <span style='color: red'>(+1 [+0.6%])</span> 161 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-11 [-16.4%])</span> 56 | <span style='color: green'>(-11 [-16.4%])</span> 56 | <span style='color: green'>(-11 [-16.4%])</span> 56 | <span style='color: green'>(-11 [-16.4%])</span> 56 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-34 [-17.5%])</span> 160 | <span style='color: green'>(-34 [-17.5%])</span> 160 | <span style='color: green'>(-34 [-17.5%])</span> 160 | <span style='color: green'>(-34 [-17.5%])</span> 160 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+2.3%])</span> 89 | <span style='color: red'>(+2 [+2.3%])</span> 89 | <span style='color: red'>(+2 [+2.3%])</span> 89 | <span style='color: red'>(+2 [+2.3%])</span> 89 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+5 [+5.1%])</span> 104 | <span style='color: red'>(+5 [+5.1%])</span> 104 | <span style='color: red'>(+5 [+5.1%])</span> 104 | <span style='color: red'>(+5 [+5.1%])</span> 104 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-4 [-1.4%])</span> 291 | <span style='color: green'>(-4 [-1.4%])</span> 291 | <span style='color: green'>(-4 [-1.4%])</span> 291 | <span style='color: green'>(-4 [-1.4%])</span> 291 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | app proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 34 | 1 | 6 | 0 | 19 | 7 | 1,264 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 71 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 14 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 572 |  | 
| PhantomAir |  | 2 |  | 3 | 5 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 25 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 29 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmConnectorAir |  | 2 |  | 5 | 11 |  | 
| VolatileBoundaryAir |  | 2 |  | 7 | 19 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 52 | 1,254 | 322,699 | 62,474,410 | 866 | 89 | 104 | 160 | 291 | 6 | 161 | 17,271,190 | 322,700 | 56 | 1 | 162 | 1.98 | 174 | 1.85 | 0 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 84 | 27 | 227,328 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 28 | 12 | 1,310,720 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 312 | 398 | 23,265,280 | 
| verify_fibair | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 36 | 29 | 17,039,360 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 40 | 21 | 3,997,696 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 40 | 27 | 2,195,456 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 36 | 38 | 2,424,832 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 20 | 12 | 2,097,152 | 

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


Commit: https://github.com/openvm-org/openvm/commit/16b7a62a16edf26cc66d2921eb0a3352934c66be

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16061914981)
