| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.5%])</span> 1.59 | <span style='color: red'>(+0 [+3.5%])</span> 1.59 |
| verify_fibair | <span style='color: red'>(+0 [+3.5%])</span> 1.59 | <span style='color: red'>(+0 [+3.5%])</span> 1.59 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+54 [+3.5%])</span> 1,586 | <span style='color: red'>(+54 [+3.5%])</span> 1,586 | <span style='color: red'>(+54 [+3.5%])</span> 1,586 | <span style='color: red'>(+54 [+3.5%])</span> 1,586 |
| `main_cells_used     ` | <span style='color: red'>(+479717 [+4.9%])</span> 10,244,506 | <span style='color: red'>(+479717 [+4.9%])</span> 10,244,506 | <span style='color: red'>(+479717 [+4.9%])</span> 10,244,506 | <span style='color: red'>(+479717 [+4.9%])</span> 10,244,506 |
| `total_cycles        ` | <span style='color: red'>(+18324 [+9.8%])</span> 205,685 | <span style='color: red'>(+18324 [+9.8%])</span> 205,685 | <span style='color: red'>(+18324 [+9.8%])</span> 205,685 | <span style='color: red'>(+18324 [+9.8%])</span> 205,685 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+1.9%])</span> 110 | <span style='color: red'>(+2 [+1.9%])</span> 110 | <span style='color: red'>(+2 [+1.9%])</span> 110 | <span style='color: red'>(+2 [+1.9%])</span> 110 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+5 [+3.7%])</span> 139 | <span style='color: red'>(+5 [+3.7%])</span> 139 | <span style='color: red'>(+5 [+3.7%])</span> 139 | <span style='color: red'>(+5 [+3.7%])</span> 139 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+47 [+3.6%])</span> 1,337 | <span style='color: red'>(+47 [+3.6%])</span> 1,337 | <span style='color: red'>(+47 [+3.6%])</span> 1,337 | <span style='color: red'>(+47 [+3.6%])</span> 1,337 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+12 [+5.1%])</span> 246 | <span style='color: red'>(+12 [+5.1%])</span> 246 | <span style='color: red'>(+12 [+5.1%])</span> 246 | <span style='color: red'>(+12 [+5.1%])</span> 246 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+8 [+34.8%])</span> 31 | <span style='color: red'>(+8 [+34.8%])</span> 31 | <span style='color: red'>(+8 [+34.8%])</span> 31 | <span style='color: red'>(+8 [+34.8%])</span> 31 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+17 [+8.2%])</span> 224 | <span style='color: red'>(+17 [+8.2%])</span> 224 | <span style='color: red'>(+17 [+8.2%])</span> 224 | <span style='color: red'>(+17 [+8.2%])</span> 224 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+5 [+3.4%])</span> 150 | <span style='color: red'>(+5 [+3.4%])</span> 150 | <span style='color: red'>(+5 [+3.4%])</span> 150 | <span style='color: red'>(+5 [+3.4%])</span> 150 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+7 [+2.4%])</span> 303 | <span style='color: red'>(+7 [+2.4%])</span> 303 | <span style='color: red'>(+7 [+2.4%])</span> 303 | <span style='color: red'>(+7 [+2.4%])</span> 303 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-0.5%])</span> 380 | <span style='color: green'>(-2 [-0.5%])</span> 380 | <span style='color: green'>(-2 [-0.5%])</span> 380 | <span style='color: green'>(-2 [-0.5%])</span> 380 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 64 | 3 | 13 | 0 | 34 | 13 | 

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
| verify_fibair | 139 | 1,586 | 205,685 | 27,624,088 | 1,337 | 150 | 303 | 224 | 380 | 246 | 10,244,506 | 31 | 110 | 

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
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 24 | 22 | 3,014,656 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 8 | 11 | 1,245,184 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/27eb1494e7cda849a55cfbbdb9e41350f0e266cc

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13136034372)
