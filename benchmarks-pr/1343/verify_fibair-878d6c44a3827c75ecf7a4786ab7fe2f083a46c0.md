| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.5%])</span> 1.52 | <span style='color: green'>(-0 [-0.5%])</span> 1.52 |
| verify_fibair | <span style='color: green'>(-0 [-0.5%])</span> 1.52 | <span style='color: green'>(-0 [-0.5%])</span> 1.52 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-8 [-0.5%])</span> 1,521 | <span style='color: green'>(-8 [-0.5%])</span> 1,521 | <span style='color: green'>(-8 [-0.5%])</span> 1,521 | <span style='color: green'>(-8 [-0.5%])</span> 1,521 |
| `main_cells_used     ` |  9,765,248 |  9,765,248 |  9,765,248 |  9,765,248 |
| `total_cycles        ` |  187,412 |  187,412 |  187,412 |  187,412 |
| `execute_time_ms     ` | <span style='color: red'>(+6 [+5.9%])</span> 108 | <span style='color: red'>(+6 [+5.9%])</span> 108 | <span style='color: red'>(+6 [+5.9%])</span> 108 | <span style='color: red'>(+6 [+5.9%])</span> 108 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+3 [+2.3%])</span> 136 | <span style='color: red'>(+3 [+2.3%])</span> 136 | <span style='color: red'>(+3 [+2.3%])</span> 136 | <span style='color: red'>(+3 [+2.3%])</span> 136 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-17 [-1.3%])</span> 1,277 | <span style='color: green'>(-17 [-1.3%])</span> 1,277 | <span style='color: green'>(-17 [-1.3%])</span> 1,277 | <span style='color: green'>(-17 [-1.3%])</span> 1,277 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+18 [+8.1%])</span> 239 | <span style='color: red'>(+18 [+8.1%])</span> 239 | <span style='color: red'>(+18 [+8.1%])</span> 239 | <span style='color: red'>(+18 [+8.1%])</span> 239 |
| `generate_perm_trace_time_ms` |  22 |  22 |  22 |  22 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+10 [+5.1%])</span> 208 | <span style='color: red'>(+10 [+5.1%])</span> 208 | <span style='color: red'>(+10 [+5.1%])</span> 208 | <span style='color: red'>(+10 [+5.1%])</span> 208 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-2.0%])</span> 145 | <span style='color: green'>(-3 [-2.0%])</span> 145 | <span style='color: green'>(-3 [-2.0%])</span> 145 | <span style='color: green'>(-3 [-2.0%])</span> 145 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-21 [-6.7%])</span> 294 | <span style='color: green'>(-21 [-6.7%])</span> 294 | <span style='color: green'>(-21 [-6.7%])</span> 294 | <span style='color: green'>(-21 [-6.7%])</span> 294 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-21 [-5.4%])</span> 365 | <span style='color: green'>(-21 [-5.4%])</span> 365 | <span style='color: green'>(-21 [-5.4%])</span> 365 | <span style='color: green'>(-21 [-5.4%])</span> 365 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 61 | 3 | 14 | 0 | 31 | 12 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 52 |  | 
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
| verify_fibair | 136 | 1,521 | 187,412 | 26,116,760 | 1,277 | 145 | 294 | 208 | 365 | 239 | 9,765,248 | 22 | 108 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 36 | 25 | 62,464 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 16 | 23 | 1,277,952 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 8,192 |  | 12 | 9 | 172,032 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 8 | 11 | 1,245,184 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/878d6c44a3827c75ecf7a4786ab7fe2f083a46c0

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13147434416)
