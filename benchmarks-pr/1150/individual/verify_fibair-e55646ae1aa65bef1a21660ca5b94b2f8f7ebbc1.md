| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| verify_fibair | <div style='text-align: right'>2</div>  | <span style="color: green">(-420 [-0.0%])</span> <div style='text-align: right'>8,122,024</div>  | <span style="color: green">(-56 [-0.0%])</span> <div style='text-align: right'>195,372</div>  | <span style="color: red">(+14.0 [+1.0%])</span> <div style='text-align: right'>1,464.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| air_name | cells | constraints | interactions | main_cols | quotient_deg | rows |
| --- | --- | --- | --- | --- | --- | --- |
| FibonacciAir | <div style='text-align: right'>32</div>  | <div style='text-align: right'>5</div>  |  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>16</div>  |
| ProgramAir |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  |  | <div style='text-align: right'>1</div>  |  |
| VmConnectorAir |  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  |  | <div style='text-align: right'>4</div>  |  |
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | <div style='text-align: right'>22</div>  | <div style='text-align: right'>11</div>  |  | <div style='text-align: right'>4</div>  |  |
| VolatileBoundaryAir |  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>4</div>  |  | <div style='text-align: right'>4</div>  |  |
| AccessAdapterAir<2> |  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  |  | <div style='text-align: right'>4</div>  |  |
| AccessAdapterAir<4> |  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  |  | <div style='text-align: right'>4</div>  |  |
| AccessAdapterAir<8> |  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  |  | <div style='text-align: right'>4</div>  |  |
| NativePoseidon2Air<BabyBearParameters>, 1> |  | <div style='text-align: right'>302</div>  | <div style='text-align: right'>31</div>  |  | <div style='text-align: right'>4</div>  |  |
| FriReducedOpeningAir |  | <div style='text-align: right'>59</div>  | <div style='text-align: right'>35</div>  |  | <div style='text-align: right'>4</div>  |  |
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  |  | <div style='text-align: right'>4</div>  |  |
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  |  | <div style='text-align: right'>4</div>  |  |
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>7</div>  |  | <div style='text-align: right'>4</div>  |  |
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  |  | <div style='text-align: right'>2</div>  |  |
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>19</div>  |  | <div style='text-align: right'>4</div>  |  |
| PhantomAir |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>3</div>  |  | <div style='text-align: right'>4</div>  |  |
| VariableRangeCheckerAir |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  |  | <div style='text-align: right'>1</div>  |  |

| main_trace_commit_time_ms | pcs_opening_time_ms | perm_trace_commit_time_ms | quotient_poly_commit_time_ms | quotient_poly_compute_time_ms | stark_prove_excluding_trace_time_ms | total_cells | verify_program_compile_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <div style='text-align: right'>5.0</div>  | <span style="color: red">(+3.0 [+300.0%])</span> <div style='text-align: right'>4.0</div>  | <div style='text-align: right'>0.0</div>  | <div style='text-align: right'>1.0</div>  | <div style='text-align: right'>0.0</div>  | <span style="color: red">(+2.0 [+22.2%])</span> <div style='text-align: right'>11.0</div>  | <div style='text-align: right'>32</div>  | <span style="color: red">(+1.0 [+6.7%])</span> <div style='text-align: right'>16.0</div>  |

| group | fri.log_blowup | generate_perm_trace_time_ms | main_trace_commit_time_ms | pcs_opening_time_ms | perm_trace_commit_time_ms | quotient_poly_commit_time_ms | quotient_poly_compute_time_ms | stark_prove_excluding_trace_time_ms | total_cells | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | <div style='text-align: right'>2</div>  | <span style="color: red">(+4.0 [+16.0%])</span> <div style='text-align: right'>29.0</div>  | <span style="color: red">(+3.0 [+1.3%])</span> <div style='text-align: right'>231.0</div>  | <span style="color: red">(+2.0 [+0.4%])</span> <div style='text-align: right'>471.0</div>  | <span style="color: green">(-5.0 [-2.6%])</span> <div style='text-align: right'>191.0</div>  | <span style="color: red">(+19.0 [+7.3%])</span> <div style='text-align: right'>281.0</div>  | <span style="color: green">(-7.0 [-2.6%])</span> <div style='text-align: right'>260.0</div>  | <span style="color: red">(+14.0 [+1.0%])</span> <div style='text-align: right'>1,464.0</div>  | <div style='text-align: right'>23,451,672</div>  | <span style="color: green">(-420 [-0.0%])</span> <div style='text-align: right'>8,122,024</div>  | <span style="color: green">(-56 [-0.0%])</span> <div style='text-align: right'>195,372</div>  | <span style="color: red">(+14.0 [+1.0%])</span> <div style='text-align: right'>1,464.0</div>  |

| group | segment | execute_time_ms | total_cells_used | total_cycles | trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- |
| verify_fibair | 0 | <span style="color: green">(-1.0 [-0.5%])</span> <div style='text-align: right'>202.0</div>  | <span style="color: green">(-420 [-0.0%])</span> <div style='text-align: right'>8,122,024</div>  | <span style="color: green">(-56 [-0.0%])</span> <div style='text-align: right'>195,372</div>  | <span style="color: green">(-1.0 [-2.0%])</span> <div style='text-align: right'>49.0</div>  |

| group | air_name | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | ProgramAir | <div style='text-align: right'>294,912</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>16,384</div>  |
| verify_fibair | VmConnectorAir | <div style='text-align: right'>24</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| verify_fibair | VolatileBoundaryAir | <div style='text-align: right'>1,245,184</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>65,536</div>  |
| verify_fibair | AccessAdapterAir<2> | <div style='text-align: right'>884,736</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>32,768</div>  |
| verify_fibair | AccessAdapterAir<4> | <div style='text-align: right'>475,136</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>16,384</div>  |
| verify_fibair | AccessAdapterAir<8> | <div style='text-align: right'>135,168</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4,096</div>  |
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | <div style='text-align: right'>786,432</div>  | <div style='text-align: right'>348</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>2,048</div>  |
| verify_fibair | FriReducedOpeningAir | <div style='text-align: right'>71,680</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>512</div>  |
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | <div style='text-align: right'>245,760</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4,096</div>  |
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | <div style='text-align: right'>6,553,600</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>131,072</div>  |
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | <div style='text-align: right'>180,224</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>8,192</div>  |
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | <div style='text-align: right'>1,671,168</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>32,768</div>  |
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | <div style='text-align: right'>8,519,680</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>131,072</div>  |
| verify_fibair | PhantomAir | <div style='text-align: right'>28,672</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>2,048</div>  |
| verify_fibair | VariableRangeCheckerAir | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

</details>



Commit: https://github.com/openvm-org/openvm/commit/e55646ae1aa65bef1a21660ca5b94b2f8f7ebbc1

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12564146335)