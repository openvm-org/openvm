| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.9%])</span> 3.55 | <span style='color: red'>(+0 [+3.9%])</span> 3.55 |
| verify_fibair | <span style='color: red'>(+0 [+3.9%])</span> 3.55 | <span style='color: red'>(+0 [+3.9%])</span> 3.55 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+135 [+3.9%])</span> 3,553 | <span style='color: red'>(+135 [+3.9%])</span> 3,553 | <span style='color: red'>(+135 [+3.9%])</span> 3,553 | <span style='color: red'>(+135 [+3.9%])</span> 3,553 |
| `main_cells_used     ` | <span style='color: red'>(+26359 [+0.1%])</span> 25,536,574 | <span style='color: red'>(+26359 [+0.1%])</span> 25,536,574 | <span style='color: red'>(+26359 [+0.1%])</span> 25,536,574 | <span style='color: red'>(+26359 [+0.1%])</span> 25,536,574 |
| `total_cycles        ` | <span style='color: red'>(+933 [+0.1%])</span> 712,144 | <span style='color: red'>(+933 [+0.1%])</span> 712,144 | <span style='color: red'>(+933 [+0.1%])</span> 712,144 | <span style='color: red'>(+933 [+0.1%])</span> 712,144 |
| `execute_time_ms     ` |  146 |  146 |  146 |  146 |
| `trace_gen_time_ms   ` |  432 |  432 |  432 |  432 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+135 [+4.8%])</span> 2,975 | <span style='color: red'>(+135 [+4.8%])</span> 2,975 | <span style='color: red'>(+135 [+4.8%])</span> 2,975 | <span style='color: red'>(+135 [+4.8%])</span> 2,975 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+35 [+7.2%])</span> 518 | <span style='color: red'>(+35 [+7.2%])</span> 518 | <span style='color: red'>(+35 [+7.2%])</span> 518 | <span style='color: red'>(+35 [+7.2%])</span> 518 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-11 [-14.1%])</span> 67 | <span style='color: green'>(-11 [-14.1%])</span> 67 | <span style='color: green'>(-11 [-14.1%])</span> 67 | <span style='color: green'>(-11 [-14.1%])</span> 67 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+33 [+6.7%])</span> 522 | <span style='color: red'>(+33 [+6.7%])</span> 522 | <span style='color: red'>(+33 [+6.7%])</span> 522 | <span style='color: red'>(+33 [+6.7%])</span> 522 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-6 [-1.0%])</span> 573 | <span style='color: green'>(-6 [-1.0%])</span> 573 | <span style='color: green'>(-6 [-1.0%])</span> 573 | <span style='color: green'>(-6 [-1.0%])</span> 573 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+68 [+12.7%])</span> 604 | <span style='color: red'>(+68 [+12.7%])</span> 604 | <span style='color: red'>(+68 [+12.7%])</span> 604 | <span style='color: red'>(+68 [+12.7%])</span> 604 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+16 [+2.4%])</span> 688 | <span style='color: red'>(+16 [+2.4%])</span> 688 | <span style='color: red'>(+16 [+2.4%])</span> 688 | <span style='color: red'>(+16 [+2.4%])</span> 688 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 67 | 3 | 13 | 0 | 32 | 17 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 31 | 302 |  | 
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
| verify_fibair | 432 | 3,553 | 712,144 | 72,898,584 | 2,975 | 573 | 604 | 522 | 688 | 518 | 25,536,574 | 67 | 146 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 32,768 |  | 16 | 17 | 1,081,344 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 8,192 |  | 36 | 348 | 3,145,728 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 262,144 |  | 28 | 23 | 13,369,344 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 32,768 |  | 12 | 10 | 720,896 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 524,288 |  | 20 | 30 | 26,214,400 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 36 | 25 | 15,990,784 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4d21ab35d4ac419a4bf5ef2e5b938bc30dcdafca

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12824473669)
