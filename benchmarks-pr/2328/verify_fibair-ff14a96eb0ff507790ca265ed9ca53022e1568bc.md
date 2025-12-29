| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  0.28 |  0.28 | 0.28 |
| verify_fibair |  0.28 |  0.28 |  0.28 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  277 |  277 |  277 |  277 |
| `main_cells_used     ` |  2,058,654 |  2,058,654 |  2,058,654 |  2,058,654 |
| `total_cells_used    ` |  7,255,892 |  7,255,892 |  7,255,892 |  7,255,892 |
| `execute_preflight_insns` |  322,610 |  322,610 |  322,610 |  322,610 |
| `execute_preflight_time_ms` |  70 |  70 |  70 |  70 |
| `execute_preflight_insn_mi/s` |  4.84 | -          |  4.84 |  4.84 |
| `trace_gen_time_ms   ` |  22 |  22 |  22 |  22 |
| `memory_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `stark_prove_excluding_trace_time_ms` |  184 |  184 |  184 |  184 |
| `main_trace_commit_time_ms` |  24 |  24 |  24 |  24 |
| `generate_perm_trace_time_ms` |  16 |  16 |  16 |  16 |
| `perm_trace_commit_time_ms` |  26.54 |  26.54 |  26.54 |  26.54 |
| `quotient_poly_compute_time_ms` |  20.76 |  20.76 |  20.76 |  20.76 |
| `quotient_poly_commit_time_ms` |  7.49 |  7.49 |  7.49 |  7.49 |
| `pcs_opening_time_ms ` |  87 |  87 |  87 |  87 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | verify_fibair_time_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | main_trace_commit_time_ms | generate_perm_trace_time_ms | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 9 | 277 | 65,536 | 23 | 0.13 | 0.72 | 1 | 0 | 19 | 0 | 19 | 3 | 0 | 1 | 0 | 0 | 19 | 

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
| verify_fibair | 22 | 277 | 7,255,892 | 62,474,410 | 22 | 184 | 0 | 20.76 | 7.49 | 4 | 26.54 | 87 | 43 | 86 | 2 | 24 | 2,058,654 | 16 | 1 | 70 | 322,610 | 4.84 | 11 | 28 | 0 | 86 | 

| group | air_id | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 0 | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | 1 | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 36 | 29 | 17,039,360 | 
| verify_fibair | 11 | JalRangeCheckAir | 32,768 |  | 28 | 12 | 1,310,720 | 
| verify_fibair | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 40 | 27 | 2,195,456 | 
| verify_fibair | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 40 | 21 | 3,997,696 | 
| verify_fibair | 15 | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | 16 | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | 3 | VolatileBoundaryAir | 65,536 |  | 20 | 12 | 2,097,152 | 
| verify_fibair | 4 | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | 5 | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | 6 | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 312 | 398 | 23,265,280 | 
| verify_fibair | 8 | FriReducedOpeningAir | 2,048 |  | 84 | 27 | 227,328 | 
| verify_fibair | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 36 | 38 | 2,424,832 | 

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


Commit: https://github.com/openvm-org/openvm/commit/ff14a96eb0ff507790ca265ed9ca53022e1568bc

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20580722795)
