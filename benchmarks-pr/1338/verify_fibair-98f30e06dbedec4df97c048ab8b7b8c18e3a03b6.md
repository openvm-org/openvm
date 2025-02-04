| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+2.0%])</span> 1.56 | <span style='color: red'>(+0 [+2.0%])</span> 1.56 |
| verify_fibair | <span style='color: red'>(+0 [+2.0%])</span> 1.56 | <span style='color: red'>(+0 [+2.0%])</span> 1.56 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+30 [+2.0%])</span> 1,562 | <span style='color: red'>(+30 [+2.0%])</span> 1,562 | <span style='color: red'>(+30 [+2.0%])</span> 1,562 | <span style='color: red'>(+30 [+2.0%])</span> 1,562 |
| `main_cells_used     ` | <span style='color: red'>(+480608 [+4.9%])</span> 10,245,397 | <span style='color: red'>(+480608 [+4.9%])</span> 10,245,397 | <span style='color: red'>(+480608 [+4.9%])</span> 10,245,397 | <span style='color: red'>(+480608 [+4.9%])</span> 10,245,397 |
| `total_cycles        ` | <span style='color: red'>(+18423 [+9.8%])</span> 205,784 | <span style='color: red'>(+18423 [+9.8%])</span> 205,784 | <span style='color: red'>(+18423 [+9.8%])</span> 205,784 | <span style='color: red'>(+18423 [+9.8%])</span> 205,784 |
| `execute_time_ms     ` | <span style='color: red'>(+3 [+2.8%])</span> 111 | <span style='color: red'>(+3 [+2.8%])</span> 111 | <span style='color: red'>(+3 [+2.8%])</span> 111 | <span style='color: red'>(+3 [+2.8%])</span> 111 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+7 [+5.2%])</span> 141 | <span style='color: red'>(+7 [+5.2%])</span> 141 | <span style='color: red'>(+7 [+5.2%])</span> 141 | <span style='color: red'>(+7 [+5.2%])</span> 141 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+20 [+1.6%])</span> 1,310 | <span style='color: red'>(+20 [+1.6%])</span> 1,310 | <span style='color: red'>(+20 [+1.6%])</span> 1,310 | <span style='color: red'>(+20 [+1.6%])</span> 1,310 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-5 [-2.1%])</span> 229 | <span style='color: green'>(-5 [-2.1%])</span> 229 | <span style='color: green'>(-5 [-2.1%])</span> 229 | <span style='color: green'>(-5 [-2.1%])</span> 229 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+13.0%])</span> 26 | <span style='color: red'>(+3 [+13.0%])</span> 26 | <span style='color: red'>(+3 [+13.0%])</span> 26 | <span style='color: red'>(+3 [+13.0%])</span> 26 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+12 [+5.8%])</span> 219 | <span style='color: red'>(+12 [+5.8%])</span> 219 | <span style='color: red'>(+12 [+5.8%])</span> 219 | <span style='color: red'>(+12 [+5.8%])</span> 219 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-2.1%])</span> 142 | <span style='color: green'>(-3 [-2.1%])</span> 142 | <span style='color: green'>(-3 [-2.1%])</span> 142 | <span style='color: green'>(-3 [-2.1%])</span> 142 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+10 [+3.4%])</span> 306 | <span style='color: red'>(+10 [+3.4%])</span> 306 | <span style='color: red'>(+10 [+3.4%])</span> 306 | <span style='color: red'>(+10 [+3.4%])</span> 306 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+2 [+0.5%])</span> 384 | <span style='color: red'>(+2 [+0.5%])</span> 384 | <span style='color: red'>(+2 [+0.5%])</span> 384 | <span style='color: red'>(+2 [+0.5%])</span> 384 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 62 | 2 | 13 | 0 | 32 | 13 | 

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
| verify_fibair | 141 | 1,562 | 205,784 | 27,624,088 | 1,310 | 142 | 306 | 219 | 384 | 229 | 10,245,397 | 26 | 111 | 

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


Commit: https://github.com/openvm-org/openvm/commit/98f30e06dbedec4df97c048ab8b7b8c18e3a03b6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13137078202)
