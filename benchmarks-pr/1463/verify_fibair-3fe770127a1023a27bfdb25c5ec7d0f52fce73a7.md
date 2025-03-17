| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+7.2%])</span> 1.36 | <span style='color: red'>(+0 [+7.2%])</span> 1.36 |
| verify_fibair | <span style='color: red'>(+0 [+7.2%])</span> 1.36 | <span style='color: red'>(+0 [+7.2%])</span> 1.36 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+91 [+7.2%])</span> 1,363 | <span style='color: red'>(+91 [+7.2%])</span> 1,363 | <span style='color: red'>(+91 [+7.2%])</span> 1,363 | <span style='color: red'>(+91 [+7.2%])</span> 1,363 |
| `main_cells_used     ` | <span style='color: green'>(-9611817 [-53.7%])</span> 8,290,048 | <span style='color: green'>(-9611817 [-53.7%])</span> 8,290,048 | <span style='color: green'>(-9611817 [-53.7%])</span> 8,290,048 | <span style='color: green'>(-9611817 [-53.7%])</span> 8,290,048 |
| `total_cycles        ` | <span style='color: green'>(-190445 [-57.0%])</span> 143,569 | <span style='color: green'>(-190445 [-57.0%])</span> 143,569 | <span style='color: green'>(-190445 [-57.0%])</span> 143,569 | <span style='color: green'>(-190445 [-57.0%])</span> 143,569 |
| `execute_time_ms     ` | <span style='color: green'>(-86 [-46.2%])</span> 100 | <span style='color: green'>(-86 [-46.2%])</span> 100 | <span style='color: green'>(-86 [-46.2%])</span> 100 | <span style='color: green'>(-86 [-46.2%])</span> 100 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-79 [-43.6%])</span> 102 | <span style='color: green'>(-79 [-43.6%])</span> 102 | <span style='color: green'>(-79 [-43.6%])</span> 102 | <span style='color: green'>(-79 [-43.6%])</span> 102 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+256 [+28.3%])</span> 1,161 | <span style='color: red'>(+256 [+28.3%])</span> 1,161 | <span style='color: red'>(+256 [+28.3%])</span> 1,161 | <span style='color: red'>(+256 [+28.3%])</span> 1,161 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+39 [+23.8%])</span> 203 | <span style='color: red'>(+39 [+23.8%])</span> 203 | <span style='color: red'>(+39 [+23.8%])</span> 203 | <span style='color: red'>(+39 [+23.8%])</span> 203 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-11 [-28.9%])</span> 27 | <span style='color: green'>(-11 [-28.9%])</span> 27 | <span style='color: green'>(-11 [-28.9%])</span> 27 | <span style='color: green'>(-11 [-28.9%])</span> 27 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-7 [-3.5%])</span> 193 | <span style='color: green'>(-7 [-3.5%])</span> 193 | <span style='color: green'>(-7 [-3.5%])</span> 193 | <span style='color: green'>(-7 [-3.5%])</span> 193 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+16 [+14.5%])</span> 126 | <span style='color: red'>(+16 [+14.5%])</span> 126 | <span style='color: red'>(+16 [+14.5%])</span> 126 | <span style='color: red'>(+16 [+14.5%])</span> 126 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+133 [+112.7%])</span> 251 | <span style='color: red'>(+133 [+112.7%])</span> 251 | <span style='color: red'>(+133 [+112.7%])</span> 251 | <span style='color: red'>(+133 [+112.7%])</span> 251 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+89 [+33.2%])</span> 357 | <span style='color: red'>(+89 [+33.2%])</span> 357 | <span style='color: red'>(+89 [+33.2%])</span> 357 | <span style='color: red'>(+89 [+33.2%])</span> 357 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 62 | 3 | 13 | 0 | 32 | 14 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 39 | 60 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 136 | 530 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 102 | 1,363 | 143,569 | 23,616,152 | 1,161 | 126 | 251 | 193 | 357 | 203 | 8,290,048 | 27 | 100 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 32,768 |  | 12 | 11 | 753,664 | 
| verify_fibair | AccessAdapterAir<4> | 16,384 |  | 12 | 13 | 409,600 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 44 | 27 | 72,704 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 16,384 |  | 16 | 23 | 638,976 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4,096 |  | 12 | 9 | 86,016 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 32,768 |  | 8 | 11 | 622,592 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/3fe770127a1023a27bfdb25c5ec7d0f52fce73a7

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13908477001)
