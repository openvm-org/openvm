| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,631 |  1,631 |  1,631 |  1,631 |
| `total_cells_used    ` | <span style='color: red'>(+110402 [+1.4%])</span> 8,123,164 | <span style='color: red'>(+110402 [+1.4%])</span> 8,123,164 | <span style='color: red'>(+110402 [+1.4%])</span> 8,123,164 | <span style='color: red'>(+110402 [+1.4%])</span> 8,123,164 |
| `total_cycles        ` | <span style='color: red'>(+698 [+0.4%])</span> 195,437 | <span style='color: red'>(+698 [+0.4%])</span> 195,437 | <span style='color: red'>(+698 [+0.4%])</span> 195,437 | <span style='color: red'>(+698 [+0.4%])</span> 195,437 |
| `execute_time_ms     ` | <span style='color: green'>(-20 [-10.0%])</span> 181 | <span style='color: green'>(-20 [-10.0%])</span> 181 | <span style='color: green'>(-20 [-10.0%])</span> 181 | <span style='color: green'>(-20 [-10.0%])</span> 181 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-4 [-7.5%])</span> 49 | <span style='color: green'>(-4 [-7.5%])</span> 49 | <span style='color: green'>(-4 [-7.5%])</span> 49 | <span style='color: green'>(-4 [-7.5%])</span> 49 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-49 [-3.4%])</span> 1,401 | <span style='color: green'>(-49 [-3.4%])</span> 1,401 | <span style='color: green'>(-49 [-3.4%])</span> 1,401 | <span style='color: green'>(-49 [-3.4%])</span> 1,401 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-7 [-3.0%])</span> 229 | <span style='color: green'>(-7 [-3.0%])</span> 229 | <span style='color: green'>(-7 [-3.0%])</span> 229 | <span style='color: green'>(-7 [-3.0%])</span> 229 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-11.1%])</span> 24 | <span style='color: green'>(-3 [-11.1%])</span> 24 | <span style='color: green'>(-3 [-11.1%])</span> 24 | <span style='color: green'>(-3 [-11.1%])</span> 24 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-19 [-9.7%])</span> 177 | <span style='color: green'>(-19 [-9.7%])</span> 177 | <span style='color: green'>(-19 [-9.7%])</span> 177 | <span style='color: green'>(-19 [-9.7%])</span> 177 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-7 [-2.7%])</span> 251 | <span style='color: green'>(-7 [-2.7%])</span> 251 | <span style='color: green'>(-7 [-2.7%])</span> 251 | <span style='color: green'>(-7 [-2.7%])</span> 251 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-12 [-4.5%])</span> 255 | <span style='color: green'>(-12 [-4.5%])</span> 255 | <span style='color: green'>(-12 [-4.5%])</span> 255 | <span style='color: green'>(-12 [-4.5%])</span> 255 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-1 [-0.2%])</span> 462 | <span style='color: green'>(-1 [-0.2%])</span> 462 | <span style='color: green'>(-1 [-0.2%])</span> 462 | <span style='color: green'>(-1 [-0.2%])</span> 462 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 15 | 32 | 11 | 0 | 1 | 0 | 3 | 5 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 16 | 1 | 2 |  | 5 | 32 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 31 | 302 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 23 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 19 | 31 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells_used | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | generate_perm_trace_time_ms | fri.log_blowup | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 49 | 1,631 | 195,437 | 8,123,164 | 23,451,672 | 1,401 | 251 | 255 | 177 | 462 | 229 | 24 | 2 | 181 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 32,768 |  | 16 | 11 | 884,736 | 
| verify_fibair | AccessAdapterAir<4> | 16,384 |  | 16 | 13 | 475,136 | 
| verify_fibair | AccessAdapterAir<8> | 4,096 |  | 16 | 17 | 135,168 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 2,048 |  | 36 | 348 | 786,432 | 
| verify_fibair | PhantomAir | 2,048 |  | 8 | 6 | 28,672 | 
| verify_fibair | ProgramAir | 16,384 |  | 8 | 10 | 294,912 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 8,192 |  | 12 | 10 | 180,224 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 131,072 |  | 20 | 30 | 6,553,600 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 24 | 41 | 8,519,680 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4,096 |  | 20 | 40 | 245,760 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 8 | 11 | 1,245,184 | 

</details>
