| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.6%])</span> 2.26 | <span style='color: green'>(-0 [-0.6%])</span> 2.26 |
| verify_fibair | <span style='color: green'>(-0 [-0.6%])</span> 2.26 | <span style='color: green'>(-0 [-0.6%])</span> 2.26 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-14 [-0.6%])</span> 2,256 | <span style='color: green'>(-14 [-0.6%])</span> 2,256 | <span style='color: green'>(-14 [-0.6%])</span> 2,256 | <span style='color: green'>(-14 [-0.6%])</span> 2,256 |
| `main_cells_used     ` |  19,310,589 |  19,310,589 |  19,310,589 |  19,310,589 |
| `total_cycles        ` |  511,892 |  511,892 |  511,892 |  511,892 |
| `execute_time_ms     ` |  99 |  99 |  99 |  99 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+3 [+1.0%])</span> 298 | <span style='color: red'>(+3 [+1.0%])</span> 298 | <span style='color: red'>(+3 [+1.0%])</span> 298 | <span style='color: red'>(+3 [+1.0%])</span> 298 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-17 [-0.9%])</span> 1,859 | <span style='color: green'>(-17 [-0.9%])</span> 1,859 | <span style='color: green'>(-17 [-0.9%])</span> 1,859 | <span style='color: green'>(-17 [-0.9%])</span> 1,859 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-11 [-3.3%])</span> 320 | <span style='color: green'>(-11 [-3.3%])</span> 320 | <span style='color: green'>(-11 [-3.3%])</span> 320 | <span style='color: green'>(-11 [-3.3%])</span> 320 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 | <span style='color: green'>(-1 [-2.1%])</span> 46 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.3%])</span> 335 | <span style='color: green'>(-1 [-0.3%])</span> 335 | <span style='color: green'>(-1 [-0.3%])</span> 335 | <span style='color: green'>(-1 [-0.3%])</span> 335 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-8 [-2.1%])</span> 370 | <span style='color: green'>(-8 [-2.1%])</span> 370 | <span style='color: green'>(-8 [-2.1%])</span> 370 | <span style='color: green'>(-8 [-2.1%])</span> 370 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+7 [+2.1%])</span> 344 | <span style='color: red'>(+7 [+2.1%])</span> 344 | <span style='color: red'>(+7 [+2.1%])</span> 344 | <span style='color: red'>(+7 [+2.1%])</span> 344 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-0.5%])</span> 441 | <span style='color: green'>(-2 [-0.5%])</span> 441 | <span style='color: green'>(-2 [-0.5%])</span> 441 | <span style='color: green'>(-2 [-0.5%])</span> 441 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 | 65,536 | 64 | 3 | 13 | 0 | 31 | 16 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
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
| verify_fibair | 298 | 2,256 | 511,892 | 50,178,200 | 1,859 | 370 | 344 | 335 | 441 | 320 | 19,310,589 | 46 | 99 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 16 | 11 | 1,769,472 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 16 | 13 | 950,272 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
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


Commit: https://github.com/openvm-org/openvm/commit/eb571c9de61cbda4e651351270dd7e5be4702d5d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12901511757)