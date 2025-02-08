| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.3%])</span> 1.45 | <span style='color: red'>(+0 [+0.3%])</span> 1.45 |
| verify_fibair | <span style='color: red'>(+0 [+0.3%])</span> 1.45 | <span style='color: red'>(+0 [+0.3%])</span> 1.45 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+5 [+0.3%])</span> 1,450 | <span style='color: red'>(+5 [+0.3%])</span> 1,450 | <span style='color: red'>(+5 [+0.3%])</span> 1,450 | <span style='color: red'>(+5 [+0.3%])</span> 1,450 |
| `main_cells_used     ` | <span style='color: red'>(+285009 [+3.1%])</span> 9,395,270 | <span style='color: red'>(+285009 [+3.1%])</span> 9,395,270 | <span style='color: red'>(+285009 [+3.1%])</span> 9,395,270 | <span style='color: red'>(+285009 [+3.1%])</span> 9,395,270 |
| `total_cycles        ` | <span style='color: red'>(+14375 [+8.8%])</span> 177,494 | <span style='color: red'>(+14375 [+8.8%])</span> 177,494 | <span style='color: red'>(+14375 [+8.8%])</span> 177,494 | <span style='color: red'>(+14375 [+8.8%])</span> 177,494 |
| `execute_time_ms     ` |  107 |  107 |  107 |  107 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+1 [+0.8%])</span> 120 | <span style='color: red'>(+1 [+0.8%])</span> 120 | <span style='color: red'>(+1 [+0.8%])</span> 120 | <span style='color: red'>(+1 [+0.8%])</span> 120 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+4 [+0.3%])</span> 1,223 | <span style='color: red'>(+4 [+0.3%])</span> 1,223 | <span style='color: red'>(+4 [+0.3%])</span> 1,223 | <span style='color: red'>(+4 [+0.3%])</span> 1,223 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-3 [-1.4%])</span> 217 | <span style='color: green'>(-3 [-1.4%])</span> 217 | <span style='color: green'>(-3 [-1.4%])</span> 217 | <span style='color: green'>(-3 [-1.4%])</span> 217 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-4.2%])</span> 23 | <span style='color: green'>(-1 [-4.2%])</span> 23 | <span style='color: green'>(-1 [-4.2%])</span> 23 | <span style='color: green'>(-1 [-4.2%])</span> 23 |
| `perm_trace_commit_time_ms` |  199 |  199 |  199 |  199 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 | <span style='color: red'>(+3 [+2.2%])</span> 138 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+0.4%])</span> 277 | <span style='color: red'>(+1 [+0.4%])</span> 277 | <span style='color: red'>(+1 [+0.4%])</span> 277 | <span style='color: red'>(+1 [+0.4%])</span> 277 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+4 [+1.1%])</span> 366 | <span style='color: red'>(+4 [+1.1%])</span> 366 | <span style='color: red'>(+4 [+1.1%])</span> 366 | <span style='color: red'>(+4 [+1.1%])</span> 366 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 63 | 3 | 13 | 0 | 32 | 13 | 

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
| verify_fibair | 120 | 1,450 | 177,494 | 25,418,392 | 1,223 | 138 | 277 | 199 | 366 | 217 | 9,395,270 | 23 | 107 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 44 | 27 | 72,704 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 16 | 23 | 1,277,952 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4,096 |  | 12 | 9 | 86,016 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 32,768 |  | 8 | 11 | 622,592 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/0dead0c92cd71cf5fe8de3bfa4a3567110679702

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13212834972)
