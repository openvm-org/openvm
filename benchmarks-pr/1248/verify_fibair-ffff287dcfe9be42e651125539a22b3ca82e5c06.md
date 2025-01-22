| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+4.6%])</span> 2.35 | <span style='color: red'>(+0 [+4.6%])</span> 2.35 |
| verify_fibair | <span style='color: red'>(+0 [+4.6%])</span> 2.35 | <span style='color: red'>(+0 [+4.6%])</span> 2.35 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+103 [+4.6%])</span> 2,350 | <span style='color: red'>(+103 [+4.6%])</span> 2,350 | <span style='color: red'>(+103 [+4.6%])</span> 2,350 | <span style='color: red'>(+103 [+4.6%])</span> 2,350 |
| `main_cells_used     ` | <span style='color: green'>(-28011 [-0.1%])</span> 19,357,742 | <span style='color: green'>(-28011 [-0.1%])</span> 19,357,742 | <span style='color: green'>(-28011 [-0.1%])</span> 19,357,742 | <span style='color: green'>(-28011 [-0.1%])</span> 19,357,742 |
| `total_cycles        ` |  513,315 |  513,315 |  513,315 |  513,315 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-2.0%])</span> 100 | <span style='color: green'>(-2 [-2.0%])</span> 100 | <span style='color: green'>(-2 [-2.0%])</span> 100 | <span style='color: green'>(-2 [-2.0%])</span> 100 |
| `trace_gen_time_ms   ` |  302 |  302 |  302 |  302 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+105 [+5.7%])</span> 1,948 | <span style='color: red'>(+105 [+5.7%])</span> 1,948 | <span style='color: red'>(+105 [+5.7%])</span> 1,948 | <span style='color: red'>(+105 [+5.7%])</span> 1,948 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+27 [+8.6%])</span> 342 | <span style='color: red'>(+27 [+8.6%])</span> 342 | <span style='color: red'>(+27 [+8.6%])</span> 342 | <span style='color: red'>(+27 [+8.6%])</span> 342 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.2%])</span> 44 | <span style='color: green'>(-1 [-2.2%])</span> 44 | <span style='color: green'>(-1 [-2.2%])</span> 44 | <span style='color: green'>(-1 [-2.2%])</span> 44 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+20 [+5.9%])</span> 359 | <span style='color: red'>(+20 [+5.9%])</span> 359 | <span style='color: red'>(+20 [+5.9%])</span> 359 | <span style='color: red'>(+20 [+5.9%])</span> 359 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+29 [+7.9%])</span> 396 | <span style='color: red'>(+29 [+7.9%])</span> 396 | <span style='color: red'>(+29 [+7.9%])</span> 396 | <span style='color: red'>(+29 [+7.9%])</span> 396 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+33 [+9.9%])</span> 368 | <span style='color: red'>(+33 [+9.9%])</span> 368 | <span style='color: red'>(+33 [+9.9%])</span> 368 | <span style='color: red'>(+33 [+9.9%])</span> 368 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-4 [-0.9%])</span> 436 | <span style='color: green'>(-4 [-0.9%])</span> 436 | <span style='color: green'>(-4 [-0.9%])</span> 436 | <span style='color: green'>(-4 [-0.9%])</span> 436 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 71 | 3 | 13 | 0 | 36 | 17 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 55 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 176 | 590 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 23 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 302 | 2,350 | 513,315 | 50,182,296 | 1,948 | 396 | 368 | 359 | 436 | 342 | 19,357,742 | 44 | 100 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 16 | 11 | 1,769,472 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 16 | 13 | 950,272 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 48 | 26 | 75,776 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 356 | 399 | 12,369,920 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 28 | 23 | 6,684,672 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 16,384 |  | 12 | 10 | 360,448 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 262,144 |  | 20 | 30 | 13,107,200 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 36 | 25 | 7,995,392 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ffff287dcfe9be42e651125539a22b3ca82e5c06

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12913459700)