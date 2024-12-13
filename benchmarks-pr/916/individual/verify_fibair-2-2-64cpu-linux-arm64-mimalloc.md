| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| verify_fibair | <div style='text-align: right'>2</div>  | <span style="color: red">(+2,909 [+0.0%])</span> <div style='text-align: right'>8,428,736</div>  | <span style="color: red">(+148 [+0.1%])</span> <div style='text-align: right'>198,645</div>  | <span style="color: green">(-43.0 [-2.6%])</span> <div style='text-align: right'>1,632.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| air_name | cells | constraints | main_cols | quotient_deg | rows |
| --- | --- | --- | --- | --- | --- |
| FibonacciAir | <div style='text-align: right'>32</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>16</div>  |

| stark_prove_excluding_trace_time_ms | total_cells |
| --- | --- |
| <div style='text-align: right'>12.0</div>  | <div style='text-align: right'>32</div>  |

| group | collect_metrics | execute_time_ms | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- |
| verify_fibair | true | <span style="color: red">(+1.0 [+0.1%])</span> <div style='text-align: right'>1,005.0</div>  | <span style="color: red">(+2,909 [+0.0%])</span> <div style='text-align: right'>8,428,736</div>  | <span style="color: red">(+148 [+0.1%])</span> <div style='text-align: right'>198,645</div>  |

| group | chip_name | collect_metrics | rows_used |
| --- | --- | --- | --- |
| verify_fibair | ProgramChip | true | <span style="color: red">(+20 [+0.1%])</span> <div style='text-align: right'>16,315</div>  |
| verify_fibair | VmConnectorAir | true | <div style='text-align: right'>2</div>  |
| verify_fibair | Boundary | true | <span style="color: red">(+1 [+0.0%])</span> <div style='text-align: right'>44,591</div>  |
| verify_fibair | AccessAdapter<2> | true | <span style="color: red">(+56 [+0.3%])</span> <div style='text-align: right'>22,052</div>  |
| verify_fibair | AccessAdapter<4> | true | <span style="color: red">(+28 [+0.3%])</span> <div style='text-align: right'>11,026</div>  |
| verify_fibair | AccessAdapter<8> | true | <div style='text-align: right'>3,220</div>  |
| verify_fibair | Poseidon2VmAir<BabyBearParameters> | true | <div style='text-align: right'>1,357</div>  |
| verify_fibair | FriReducedOpeningAir | true | <div style='text-align: right'>336</div>  |
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | <div style='text-align: right'>2,186</div>  |
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | <span style="color: red">(+5 [+0.0%])</span> <div style='text-align: right'>68,142</div>  |
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | true | <span style="color: red">(+131 [+2.6%])</span> <div style='text-align: right'>5,169</div>  |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | <span style="color: red">(+3 [+0.0%])</span> <div style='text-align: right'>30,558</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | <span style="color: red">(+9 [+0.0%])</span> <div style='text-align: right'>85,891</div>  |
| verify_fibair | PhantomAir | true | <div style='text-align: right'>5,216</div>  |
| verify_fibair | VariableRangeCheckerAir | true | <div style='text-align: right'>262,144</div>  |

| group | collect_metrics | dsl_ir | opcode | frequency |
| --- | --- | --- | --- | --- |
| verify_fibair | true |  | ADD | <span style="color: red">(+5 [+0.0%])</span> <div style='text-align: right'>54,982</div>  |
| verify_fibair | true |  | BBE4DIV | <div style='text-align: right'>297</div>  |
| verify_fibair | true |  | BBE4MUL | <div style='text-align: right'>891</div>  |
| verify_fibair | true |  | BEQ | <div style='text-align: right'>1,418</div>  |
| verify_fibair | true |  | BNE | <span style="color: red">(+3 [+0.0%])</span> <div style='text-align: right'>29,140</div>  |
| verify_fibair | true |  | COMP_POS2 | <div style='text-align: right'>1,092</div>  |
| verify_fibair | true |  | DIV | <div style='text-align: right'>3</div>  |
| verify_fibair | true |  | FE4ADD | <div style='text-align: right'>492</div>  |
| verify_fibair | true |  | FE4SUB | <div style='text-align: right'>506</div>  |
| verify_fibair | true |  | FRI_REDUCED_OPENING | <div style='text-align: right'>126</div>  |
| verify_fibair | true |  | JAL | <span style="color: red">(+131 [+2.6%])</span> <div style='text-align: right'>5,169</div>  |
| verify_fibair | true |  | LOADW | <div style='text-align: right'>18,438</div>  |
| verify_fibair | true |  | LOADW2 | <span style="color: red">(+6 [+0.0%])</span> <div style='text-align: right'>14,569</div>  |
| verify_fibair | true |  | MUL | <div style='text-align: right'>9,857</div>  |
| verify_fibair | true |  | PERM_POS2 | <div style='text-align: right'>265</div>  |
| verify_fibair | true |  | PHANTOM | <div style='text-align: right'>5,216</div>  |
| verify_fibair | true |  | SHINTW | <div style='text-align: right'>13,651</div>  |
| verify_fibair | true |  | STOREW | <span style="color: red">(+2 [+0.0%])</span> <div style='text-align: right'>30,347</div>  |
| verify_fibair | true |  | STOREW2 | <span style="color: red">(+1 [+0.0%])</span> <div style='text-align: right'>8,886</div>  |
| verify_fibair | true |  | SUB | <div style='text-align: right'>3,300</div>  |

| group | air_name | collect_metrics | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true |  | ADD | <span style="color: red">(+150 [+0.0%])</span> <div style='text-align: right'>1,649,460</div>  |
| verify_fibair | AccessAdapter<2> | true |  | ADD | <span style="color: red">(+308 [+2.6%])</span> <div style='text-align: right'>12,210</div>  |
| verify_fibair | AccessAdapter<4> | true |  | ADD | <span style="color: red">(+182 [+2.6%])</span> <div style='text-align: right'>7,215</div>  |
| verify_fibair | Boundary | true |  | ADD | <span style="color: red">(+11 [+0.6%])</span> <div style='text-align: right'>1,738</div>  |
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true |  | BBE4DIV | <div style='text-align: right'>11,880</div>  |
| verify_fibair | AccessAdapter<2> | true |  | BBE4DIV | <div style='text-align: right'>2,904</div>  |
| verify_fibair | AccessAdapter<4> | true |  | BBE4DIV | <div style='text-align: right'>1,716</div>  |
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true |  | BBE4MUL | <div style='text-align: right'>35,640</div>  |
| verify_fibair | AccessAdapter<2> | true |  | BBE4MUL | <span style="color: red">(+308 [+2.0%])</span> <div style='text-align: right'>16,038</div>  |
| verify_fibair | AccessAdapter<4> | true |  | BBE4MUL | <span style="color: red">(+182 [+2.0%])</span> <div style='text-align: right'>9,477</div>  |
| verify_fibair | Boundary | true |  | BBE4MUL | <div style='text-align: right'>1,496</div>  |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true |  | BEQ | <div style='text-align: right'>32,614</div>  |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true |  | BNE | <span style="color: red">(+69 [+0.0%])</span> <div style='text-align: right'>670,220</div>  |
| verify_fibair | AccessAdapter<2> | true |  | BNE | <div style='text-align: right'>946</div>  |
| verify_fibair | AccessAdapter<4> | true |  | BNE | <div style='text-align: right'>559</div>  |
| verify_fibair | AccessAdapter<2> | true |  | COMP_POS2 | <div style='text-align: right'>48,048</div>  |
| verify_fibair | AccessAdapter<4> | true |  | COMP_POS2 | <div style='text-align: right'>28,392</div>  |
| verify_fibair | AccessAdapter<8> | true |  | COMP_POS2 | <div style='text-align: right'>18,564</div>  |
| verify_fibair | Poseidon2VmAir<BabyBearParameters> | true |  | COMP_POS2 | <div style='text-align: right'>610,428</div>  |
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true |  | DIV | <div style='text-align: right'>90</div>  |
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true |  | FE4ADD | <div style='text-align: right'>19,680</div>  |
| verify_fibair | AccessAdapter<2> | true |  | FE4ADD | <div style='text-align: right'>10,846</div>  |
| verify_fibair | AccessAdapter<4> | true |  | FE4ADD | <div style='text-align: right'>6,409</div>  |
| verify_fibair | Boundary | true |  | FE4ADD | <div style='text-align: right'>792</div>  |
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true |  | FE4SUB | <div style='text-align: right'>20,240</div>  |
| verify_fibair | AccessAdapter<2> | true |  | FE4SUB | <div style='text-align: right'>18,656</div>  |
| verify_fibair | AccessAdapter<4> | true |  | FE4SUB | <div style='text-align: right'>11,024</div>  |
| verify_fibair | Boundary | true |  | FE4SUB | <div style='text-align: right'>220</div>  |
| verify_fibair | AccessAdapter<2> | true |  | FRI_REDUCED_OPENING | <div style='text-align: right'>2,024</div>  |
| verify_fibair | AccessAdapter<4> | true |  | FRI_REDUCED_OPENING | <div style='text-align: right'>1,196</div>  |
| verify_fibair | FriReducedOpeningAir | true |  | FRI_REDUCED_OPENING | <div style='text-align: right'>21,504</div>  |
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | true |  | JAL | <span style="color: red">(+1,310 [+2.6%])</span> <div style='text-align: right'>51,690</div>  |
| verify_fibair | Boundary | true |  | JAL | <div style='text-align: right'>11</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | LOADW | <div style='text-align: right'>755,958</div>  |
| verify_fibair | AccessAdapter<2> | true |  | LOADW | <div style='text-align: right'>20,295</div>  |
| verify_fibair | AccessAdapter<4> | true |  | LOADW | <div style='text-align: right'>11,232</div>  |
| verify_fibair | AccessAdapter<8> | true |  | LOADW | <div style='text-align: right'>4,284</div>  |
| verify_fibair | Boundary | true |  | LOADW | <div style='text-align: right'>17,424</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | LOADW2 | <span style="color: red">(+246 [+0.0%])</span> <div style='text-align: right'>597,329</div>  |
| verify_fibair | AccessAdapter<2> | true |  | LOADW2 | <div style='text-align: right'>12,452</div>  |
| verify_fibair | AccessAdapter<4> | true |  | LOADW2 | <div style='text-align: right'>7,358</div>  |
| verify_fibair | AccessAdapter<8> | true |  | LOADW2 | <div style='text-align: right'>204</div>  |
| verify_fibair | Boundary | true |  | LOADW2 | <div style='text-align: right'>1,650</div>  |
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true |  | MUL | <div style='text-align: right'>295,710</div>  |
| verify_fibair | AccessAdapter<2> | true |  | MUL | <div style='text-align: right'>3,751</div>  |
| verify_fibair | AccessAdapter<4> | true |  | MUL | <div style='text-align: right'>2,236</div>  |
| verify_fibair | Boundary | true |  | MUL | <div style='text-align: right'>29,348</div>  |
| verify_fibair | AccessAdapter<2> | true |  | PERM_POS2 | <div style='text-align: right'>22,770</div>  |
| verify_fibair | AccessAdapter<4> | true |  | PERM_POS2 | <div style='text-align: right'>13,455</div>  |
| verify_fibair | AccessAdapter<8> | true |  | PERM_POS2 | <div style='text-align: right'>8,806</div>  |
| verify_fibair | Poseidon2VmAir<BabyBearParameters> | true |  | PERM_POS2 | <div style='text-align: right'>148,135</div>  |
| verify_fibair | PhantomAir | true |  | PHANTOM | <div style='text-align: right'>31,296</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | SHINTW | <div style='text-align: right'>559,691</div>  |
| verify_fibair | Boundary | true |  | SHINTW | <div style='text-align: right'>150,161</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | STOREW | <span style="color: red">(+82 [+0.0%])</span> <div style='text-align: right'>1,244,227</div>  |
| verify_fibair | AccessAdapter<2> | true |  | STOREW | <div style='text-align: right'>5,445</div>  |
| verify_fibair | AccessAdapter<4> | true |  | STOREW | <div style='text-align: right'>3,120</div>  |
| verify_fibair | Boundary | true |  | STOREW | <div style='text-align: right'>204,556</div>  |
| verify_fibair | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | STOREW2 | <span style="color: red">(+41 [+0.0%])</span> <div style='text-align: right'>364,326</div>  |
| verify_fibair | AccessAdapter<2> | true |  | STOREW2 | <div style='text-align: right'>3,828</div>  |
| verify_fibair | AccessAdapter<4> | true |  | STOREW2 | <div style='text-align: right'>2,262</div>  |
| verify_fibair | AccessAdapter<8> | true |  | STOREW2 | <div style='text-align: right'>17</div>  |
| verify_fibair | Boundary | true |  | STOREW2 | <div style='text-align: right'>67,848</div>  |
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true |  | SUB | <div style='text-align: right'>99,000</div>  |
| verify_fibair | AccessAdapter<2> | true |  | SUB | <div style='text-align: right'>1,419</div>  |
| verify_fibair | AccessAdapter<4> | true |  | SUB | <div style='text-align: right'>1,677</div>  |
| verify_fibair | Boundary | true |  | SUB | <div style='text-align: right'>15,257</div>  |

| group | commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms | num_segments | total_cells_used | total_cycles | total_proof_time_ms | verify_program_compile_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | <span style="color: red">(+1.0 [+11.1%])</span> <div style='text-align: right'>10.0</div>  | <span style="color: green">(-1.0 [-0.4%])</span> <div style='text-align: right'>236.0</div>  | <span style="color: green">(-1.0 [-0.5%])</span> <div style='text-align: right'>185.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: red">(+1.0 [+1.9%])</span> <div style='text-align: right'>55.0</div>  | <div style='text-align: right'>1</div>  | <span style="color: red">(+2,909 [+0.0%])</span> <div style='text-align: right'>8,428,736</div>  | <span style="color: red">(+148 [+0.1%])</span> <div style='text-align: right'>198,645</div>  | <span style="color: green">(-43.0 [-2.6%])</span> <div style='text-align: right'>1,632.0</div>  | <div style='text-align: right'>14.0</div>  |

| group | air_name | constraints | interactions | quotient_deg |
| --- | --- | --- | --- | --- |
| verify_fibair | ProgramAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| verify_fibair | VmConnectorAir | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VolatileBoundaryAir | <div style='text-align: right'>16</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | AccessAdapterAir<2> | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | AccessAdapterAir<4> | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | AccessAdapterAir<8> | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | Poseidon2VmAir<BabyBearParameters> | <div style='text-align: right'>517</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | FriReducedOpeningAir | <div style='text-align: right'>59</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>7</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, KernelLoadStoreCoreAir<1> | <div style='text-align: right'>31</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | PhantomAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  |
| verify_fibair | VariableRangeCheckerAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |

| group | air_name | segment | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | ProgramAir | 0 | <div style='text-align: right'>294,912</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>16,384</div>  |
| verify_fibair | VmConnectorAir | 0 | <div style='text-align: right'>24</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| verify_fibair | VolatileBoundaryAir | 0 | <div style='text-align: right'>1,245,184</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>65,536</div>  |
| verify_fibair | AccessAdapterAir<2> | 0 | <div style='text-align: right'>884,736</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>32,768</div>  |
| verify_fibair | AccessAdapterAir<4> | 0 | <div style='text-align: right'>475,136</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>16,384</div>  |
| verify_fibair | AccessAdapterAir<8> | 0 | <div style='text-align: right'>135,168</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4,096</div>  |
| verify_fibair | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>1,218,560</div>  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>2,048</div>  |
| verify_fibair | FriReducedOpeningAir | 0 | <div style='text-align: right'>71,680</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>512</div>  |
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | <div style='text-align: right'>245,760</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4,096</div>  |
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>6,553,600</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>131,072</div>  |
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | <div style='text-align: right'>180,224</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>8,192</div>  |
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | <div style='text-align: right'>1,671,168</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>32,768</div>  |
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, KernelLoadStoreCoreAir<1> | 0 | <div style='text-align: right'>8,519,680</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>131,072</div>  |
| verify_fibair | PhantomAir | 0 | <div style='text-align: right'>114,688</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>8,192</div>  |
| verify_fibair | VariableRangeCheckerAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

| group | segment | execute_and_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | total_cells |
| --- | --- | --- | --- | --- |
| verify_fibair | 0 | <span style="color: green">(-1.0 [-2.0%])</span> <div style='text-align: right'>49.0</div>  | <span style="color: green">(-41.0 [-3.0%])</span> <div style='text-align: right'>1,347.0</div>  | <div style='text-align: right'>23,969,816</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/797ff538a0f968d37ad747d03437a7bdcca1ef13/verify_fibair-2-2-64cpu-linux-arm64-mimalloc-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/axiom-crypto/afs-prototype/commit/797ff538a0f968d37ad747d03437a7bdcca1ef13

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12128742304)