| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| fibonacci_program | <div style='text-align: right'>2</div>  | <div style='text-align: right'>51,505,102</div>  | <div style='text-align: right'>3,000,274</div>  | <span style="color: green">(-61.0 [-1.1%])</span> <div style='text-align: right'>5,463.0</div>  |
| leaf | <div style='text-align: right'>2</div>  | <div style='text-align: right'>129,937,051</div>  | <div style='text-align: right'>6,670,188</div>  | <span style="color: green">(-13.0 [-0.1%])</span> <div style='text-align: right'>13,849.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| commit_exe_time_ms | fri.log_blowup | keygen_time_ms |
| --- | --- | --- |
| <span style="color: green">(-1.0 [-25.0%])</span> <div style='text-align: right'>3.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-2.0 [-0.5%])</span> <div style='text-align: right'>374.0</div>  |

| air_name | constraints | interactions | quotient_deg |
| --- | --- | --- | --- |
| ProgramAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| VmConnectorAir | <div style='text-align: right'>9</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  |
| PersistentBoundaryAir<8> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| MemoryMerkleAir<8> | <div style='text-align: right'>40</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<2> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| AccessAdapterAir<4> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| AccessAdapterAir<8> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>4</div>  |
| AccessAdapterAir<16> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<32> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<64> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | <div style='text-align: right'>17</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | <div style='text-align: right'>88</div>  | <div style='text-align: right'>25</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>24</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | <div style='text-align: right'>26</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| RangeTupleCheckerAir<2> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | <div style='text-align: right'>15</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | <div style='text-align: right'>20</div>  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | <div style='text-align: right'>22</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | <div style='text-align: right'>41</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | <div style='text-align: right'>25</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | <div style='text-align: right'>33</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | <div style='text-align: right'>90</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | <div style='text-align: right'>39</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | <div style='text-align: right'>43</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| BitwiseOperationLookupAir<8> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>2</div>  |
| PhantomAir | <div style='text-align: right'>5</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  |
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | <div style='text-align: right'>286</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| VariableRangeCheckerAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>4</div>  |
| VolatileBoundaryAir | <div style='text-align: right'>16</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>4</div>  |
| NativePoseidon2Air<BabyBearParameters>, 1> | <div style='text-align: right'>302</div>  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>4</div>  |
| FriReducedOpeningAir | <div style='text-align: right'>59</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>4</div>  |
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>4</div>  |
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>4</div>  |
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>7</div>  | <div style='text-align: right'>4</div>  |
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | <div style='text-align: right'>31</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>4</div>  |

| group | segment | stark_prove_excluding_trace_time_ms | total_cells | total_cells_used | total_cycles | trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | <span style="color: green">(-61.0 [-1.1%])</span> <div style='text-align: right'>5,463.0</div>  | <div style='text-align: right'>197,453,854</div>  | <div style='text-align: right'>51,505,102</div>  | <div style='text-align: right'>3,000,274</div>  | <span style="color: green">(-7.0 [-2.8%])</span> <div style='text-align: right'>241.0</div>  |

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | ProgramChip | 0 | <div style='text-align: right'>3,335</div>  |
| fibonacci_program | VmConnectorAir | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program | Boundary | 0 | <div style='text-align: right'>38</div>  |
| fibonacci_program | Merkle | 0 | <div style='text-align: right'>284</div>  |
| fibonacci_program | AccessAdapter<8> | 0 | <div style='text-align: right'>38</div>  |
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>3</div>  |
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>9</div>  |
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | <div style='text-align: right'>13</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>100,010</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>5</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | <div style='text-align: right'>200,009</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | <div style='text-align: right'>28</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>300,002</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | <div style='text-align: right'>900,054</div>  |
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>65,536</div>  |
| fibonacci_program | PhantomAir | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>231</div>  |
| fibonacci_program | VariableRangeCheckerAir | 0 | <div style='text-align: right'>262,144</div>  |

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program |  | ADD | 0 | <div style='text-align: right'>900,045</div>  |
| fibonacci_program |  | AND | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program |  | AUIPC | 0 | <div style='text-align: right'>9</div>  |
| fibonacci_program |  | BEQ | 0 | <div style='text-align: right'>100,004</div>  |
| fibonacci_program |  | BGEU | 0 | <div style='text-align: right'>3</div>  |
| fibonacci_program |  | BLTU | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program |  | BNE | 0 | <div style='text-align: right'>100,005</div>  |
| fibonacci_program |  | HINT_STOREW | 0 | <div style='text-align: right'>3</div>  |
| fibonacci_program |  | JAL | 0 | <div style='text-align: right'>100,001</div>  |
| fibonacci_program |  | JALR | 0 | <div style='text-align: right'>13</div>  |
| fibonacci_program |  | LOADW | 0 | <div style='text-align: right'>13</div>  |
| fibonacci_program |  | LUI | 0 | <div style='text-align: right'>9</div>  |
| fibonacci_program |  | OR | 0 | <div style='text-align: right'>1</div>  |
| fibonacci_program |  | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program |  | SLL | 0 | <div style='text-align: right'>2</div>  |
| fibonacci_program |  | SLTU | 0 | <div style='text-align: right'>300,002</div>  |
| fibonacci_program |  | STOREW | 0 | <div style='text-align: right'>15</div>  |
| fibonacci_program |  | SUB | 0 | <div style='text-align: right'>4</div>  |
| fibonacci_program |  | XOR | 0 | <div style='text-align: right'>2</div>  |

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | <div style='text-align: right'>32,401,620</div>  |
| fibonacci_program | AccessAdapter<8> |  | ADD | 0 | <div style='text-align: right'>68</div>  |
| fibonacci_program | Boundary |  | ADD | 0 | <div style='text-align: right'>160</div>  |
| fibonacci_program | Merkle |  | ADD | 0 | <div style='text-align: right'>320</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | <div style='text-align: right'>72</div>  |
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | <div style='text-align: right'>189</div>  |
| fibonacci_program | AccessAdapter<8> |  | AUIPC | 0 | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary |  | AUIPC | 0 | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle |  | AUIPC | 0 | <div style='text-align: right'>3,456</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | <div style='text-align: right'>2,600,104</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | <div style='text-align: right'>96</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | <div style='text-align: right'>64</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | <div style='text-align: right'>2,600,130</div>  |
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | <div style='text-align: right'>78</div>  |
| fibonacci_program | AccessAdapter<8> |  | HINT_STOREW | 0 | <div style='text-align: right'>17</div>  |
| fibonacci_program | Boundary |  | HINT_STOREW | 0 | <div style='text-align: right'>40</div>  |
| fibonacci_program | Merkle |  | HINT_STOREW | 0 | <div style='text-align: right'>128</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | <div style='text-align: right'>1,800,018</div>  |
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | <div style='text-align: right'>364</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | <div style='text-align: right'>520</div>  |
| fibonacci_program | AccessAdapter<8> |  | LOADW | 0 | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary |  | LOADW | 0 | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle |  | LOADW | 0 | <div style='text-align: right'>2,304</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | <div style='text-align: right'>162</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | <div style='text-align: right'>36</div>  |
| fibonacci_program | PhantomAir |  | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | <div style='text-align: right'>106</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | <div style='text-align: right'>11,100,074</div>  |
| fibonacci_program | AccessAdapter<8> |  | SLTU | 0 | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary |  | SLTU | 0 | <div style='text-align: right'>80</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | <div style='text-align: right'>600</div>  |
| fibonacci_program | AccessAdapter<8> |  | STOREW | 0 | <div style='text-align: right'>136</div>  |
| fibonacci_program | Boundary |  | STOREW | 0 | <div style='text-align: right'>320</div>  |
| fibonacci_program | Merkle |  | STOREW | 0 | <div style='text-align: right'>2,816</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | <div style='text-align: right'>144</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | <div style='text-align: right'>72</div>  |

| group | execute_time_ms | fri.log_blowup | num_segments | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | <span style="color: green">(-19.0 [-1.2%])</span> <div style='text-align: right'>1,630.0</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>51,505,102</div>  | <div style='text-align: right'>3,000,274</div>  | <span style="color: green">(-61.0 [-1.1%])</span> <div style='text-align: right'>5,463.0</div>  |
| leaf |  | <div style='text-align: right'>2</div>  |  | <div style='text-align: right'>129,937,051</div>  | <div style='text-align: right'>6,670,188</div>  | <span style="color: green">(-13.0 [-0.1%])</span> <div style='text-align: right'>13,849.0</div>  |

| group | air_name | segment | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | ProgramAir | 0 | <div style='text-align: right'>73,728</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>4,096</div>  |
| fibonacci_program | VmConnectorAir | 0 | <div style='text-align: right'>32</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | PersistentBoundaryAir<8> | 0 | <div style='text-align: right'>2,048</div>  | <div style='text-align: right'>20</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>64</div>  |
| fibonacci_program | MemoryMerkleAir<8> | 0 | <div style='text-align: right'>26,624</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>512</div>  |
| fibonacci_program | AccessAdapterAir<8> | 0 | <div style='text-align: right'>2,624</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>64</div>  |
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>248</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>4</div>  |
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>4,718,592</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>784</div>  | <div style='text-align: right'>21</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>16</div>  |
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | <div style='text-align: right'>1,024</div>  | <div style='text-align: right'>28</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>16</div>  |
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>8,126,464</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>44</div>  |  | <div style='text-align: right'>131,072</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>704</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>56</div>  |  | <div style='text-align: right'>8</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | <div style='text-align: right'>19,398,656</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>48</div>  |  | <div style='text-align: right'>262,144</div>  |
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | <div style='text-align: right'>3,584</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>72</div>  |  | <div style='text-align: right'>32</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | <div style='text-align: right'>210</div>  | <div style='text-align: right'>53</div>  | <div style='text-align: right'>52</div>  |  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>40,370,176</div>  | <div style='text-align: right'>37</div>  | <div style='text-align: right'>40</div>  |  | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | <div style='text-align: right'>121,634,816</div>  | <div style='text-align: right'>36</div>  | <div style='text-align: right'>80</div>  |  | <div style='text-align: right'>1,048,576</div>  |
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>655,360</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>65,536</div>  |
| fibonacci_program | PhantomAir | 0 | <div style='text-align: right'>36</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>2</div>  |
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>78,848</div>  | <div style='text-align: right'>300</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>256</div>  |
| fibonacci_program | VariableRangeCheckerAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

| group | idx | execute_time_ms | stark_prove_excluding_trace_time_ms | total_cells | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | <span style="color: red">(+4.0 [+0.1%])</span> <div style='text-align: right'>3,808.0</div>  | <span style="color: green">(-13.0 [-0.1%])</span> <div style='text-align: right'>13,849.0</div>  | <div style='text-align: right'>372,705,752</div>  | <div style='text-align: right'>129,937,051</div>  | <div style='text-align: right'>3,335,094</div>  |

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | ProgramChip | 0 | <div style='text-align: right'>98,171</div>  |
| leaf | VmConnectorAir | 0 | <div style='text-align: right'>2</div>  |
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | <div style='text-align: right'>36</div>  |
| leaf | Boundary | 0 | <div style='text-align: right'>391,367</div>  |
| leaf | AccessAdapter<2> | 0 | <div style='text-align: right'>370,556</div>  |
| leaf | AccessAdapter<4> | 0 | <div style='text-align: right'>185,530</div>  |
| leaf | AccessAdapter<8> | 0 | <div style='text-align: right'>55,690</div>  |
| leaf | Arc<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>26,562</div>  |
| leaf | FriReducedOpeningAir | 0 | <div style='text-align: right'>117,936</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | <div style='text-align: right'>32,068</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>1,280,999</div>  |
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | <div style='text-align: right'>74,795</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | <div style='text-align: right'>637,176</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | <div style='text-align: right'>1,093,149</div>  |
| leaf | PhantomAir | 0 | <div style='text-align: right'>184,975</div>  |
| leaf | VariableRangeCheckerAir | 0 | <div style='text-align: right'>262,144</div>  |

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | <div style='text-align: right'>1,079,077</div>  |
| leaf |  | 0 | BBE4DIV | <div style='text-align: right'>6,268</div>  |
| leaf |  | 0 | BBE4MUL | <div style='text-align: right'>10,061</div>  |
| leaf |  | 0 | BEQ | <div style='text-align: right'>18,683</div>  |
| leaf |  | 0 | BNE | <div style='text-align: right'>618,493</div>  |
| leaf |  | 0 | COMP_POS2 | <div style='text-align: right'>17,315</div>  |
| leaf |  | 0 | DIV | <div style='text-align: right'>128</div>  |
| leaf |  | 0 | FE4ADD | <div style='text-align: right'>12,433</div>  |
| leaf |  | 0 | FE4SUB | <div style='text-align: right'>3,306</div>  |
| leaf |  | 0 | FRI_REDUCED_OPENING | <div style='text-align: right'>5,334</div>  |
| leaf |  | 0 | JAL | <div style='text-align: right'>74,795</div>  |
| leaf |  | 0 | LOADW | <div style='text-align: right'>153,607</div>  |
| leaf |  | 0 | LOADW2 | <div style='text-align: right'>349,066</div>  |
| leaf |  | 0 | MUL | <div style='text-align: right'>143,689</div>  |
| leaf |  | 0 | PERM_POS2 | <div style='text-align: right'>9,247</div>  |
| leaf |  | 0 | PHANTOM | <div style='text-align: right'>184,975</div>  |
| leaf |  | 0 | PUBLISH | <div style='text-align: right'>36</div>  |
| leaf |  | 0 | SHINTW | <div style='text-align: right'>229,632</div>  |
| leaf |  | 0 | STOREW | <div style='text-align: right'>194,238</div>  |
| leaf |  | 0 | STOREW2 | <div style='text-align: right'>166,606</div>  |
| leaf |  | 0 | SUB | <div style='text-align: right'>58,105</div>  |

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | <div style='text-align: right'>32,372,310</div>  |
| leaf | AccessAdapter<2> |  | 0 | ADD | <div style='text-align: right'>187,154</div>  |
| leaf | AccessAdapter<4> |  | 0 | ADD | <div style='text-align: right'>110,591</div>  |
| leaf | Boundary |  | 0 | ADD | <div style='text-align: right'>86,339</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | BBE4DIV | <div style='text-align: right'>250,720</div>  |
| leaf | AccessAdapter<2> |  | 0 | BBE4DIV | <div style='text-align: right'>121,044</div>  |
| leaf | AccessAdapter<4> |  | 0 | BBE4DIV | <div style='text-align: right'>71,526</div>  |
| leaf | Boundary |  | 0 | BBE4DIV | <div style='text-align: right'>704</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | BBE4MUL | <div style='text-align: right'>402,440</div>  |
| leaf | AccessAdapter<2> |  | 0 | BBE4MUL | <div style='text-align: right'>237,666</div>  |
| leaf | AccessAdapter<4> |  | 0 | BBE4MUL | <div style='text-align: right'>140,439</div>  |
| leaf | Boundary |  | 0 | BBE4MUL | <div style='text-align: right'>60,544</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> |  | 0 | BEQ | <div style='text-align: right'>429,709</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> |  | 0 | BNE | <div style='text-align: right'>14,225,339</div>  |
| leaf | AccessAdapter<2> |  | 0 | BNE | <div style='text-align: right'>1,386</div>  |
| leaf | AccessAdapter<4> |  | 0 | BNE | <div style='text-align: right'>819</div>  |
| leaf | AccessAdapter<2> |  | 0 | COMP_POS2 | <div style='text-align: right'>694,452</div>  |
| leaf | AccessAdapter<4> |  | 0 | COMP_POS2 | <div style='text-align: right'>410,358</div>  |
| leaf | AccessAdapter<8> |  | 0 | COMP_POS2 | <div style='text-align: right'>268,311</div>  |
| leaf | Arc<BabyBearParameters>, 1> |  | 0 | COMP_POS2 | <div style='text-align: right'>6,025,620</div>  |
| leaf | Boundary |  | 0 | COMP_POS2 | <div style='text-align: right'>88</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | DIV | <div style='text-align: right'>3,840</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | FE4ADD | <div style='text-align: right'>497,320</div>  |
| leaf | AccessAdapter<2> |  | 0 | FE4ADD | <div style='text-align: right'>218,856</div>  |
| leaf | AccessAdapter<4> |  | 0 | FE4ADD | <div style='text-align: right'>129,324</div>  |
| leaf | Boundary |  | 0 | FE4ADD | <div style='text-align: right'>64,416</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | FE4SUB | <div style='text-align: right'>132,240</div>  |
| leaf | AccessAdapter<2> |  | 0 | FE4SUB | <div style='text-align: right'>113,476</div>  |
| leaf | AccessAdapter<4> |  | 0 | FE4SUB | <div style='text-align: right'>67,054</div>  |
| leaf | Boundary |  | 0 | FE4SUB | <div style='text-align: right'>11,396</div>  |
| leaf | AccessAdapter<2> |  | 0 | FRI_REDUCED_OPENING | <div style='text-align: right'>137,544</div>  |
| leaf | AccessAdapter<4> |  | 0 | FRI_REDUCED_OPENING | <div style='text-align: right'>81,276</div>  |
| leaf | FriReducedOpeningAir |  | 0 | FRI_REDUCED_OPENING | <div style='text-align: right'>7,547,904</div>  |
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | <div style='text-align: right'>747,950</div>  |
| leaf | AccessAdapter<2> |  | 0 | JAL | <div style='text-align: right'>418</div>  |
| leaf | AccessAdapter<4> |  | 0 | JAL | <div style='text-align: right'>494</div>  |
| leaf | Boundary |  | 0 | JAL | <div style='text-align: right'>11</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | LOADW | <div style='text-align: right'>6,297,887</div>  |
| leaf | AccessAdapter<2> |  | 0 | LOADW | <div style='text-align: right'>260,744</div>  |
| leaf | AccessAdapter<4> |  | 0 | LOADW | <div style='text-align: right'>119,730</div>  |
| leaf | AccessAdapter<8> |  | 0 | LOADW | <div style='text-align: right'>20,893</div>  |
| leaf | Boundary |  | 0 | LOADW | <div style='text-align: right'>21,681</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | LOADW2 | <div style='text-align: right'>14,311,706</div>  |
| leaf | AccessAdapter<2> |  | 0 | LOADW2 | <div style='text-align: right'>57,200</div>  |
| leaf | AccessAdapter<4> |  | 0 | LOADW2 | <div style='text-align: right'>33,800</div>  |
| leaf | AccessAdapter<8> |  | 0 | LOADW2 | <div style='text-align: right'>493</div>  |
| leaf | Boundary |  | 0 | LOADW2 | <div style='text-align: right'>1,397</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | MUL | <div style='text-align: right'>4,310,670</div>  |
| leaf | AccessAdapter<2> |  | 0 | MUL | <div style='text-align: right'>23,661</div>  |
| leaf | AccessAdapter<4> |  | 0 | MUL | <div style='text-align: right'>14,001</div>  |
| leaf | Boundary |  | 0 | MUL | <div style='text-align: right'>31,856</div>  |
| leaf | AccessAdapter<2> |  | 0 | PERM_POS2 | <div style='text-align: right'>515,020</div>  |
| leaf | AccessAdapter<4> |  | 0 | PERM_POS2 | <div style='text-align: right'>305,968</div>  |
| leaf | AccessAdapter<8> |  | 0 | PERM_POS2 | <div style='text-align: right'>205,054</div>  |
| leaf | Arc<BabyBearParameters>, 1> |  | 0 | PERM_POS2 | <div style='text-align: right'>3,217,956</div>  |
| leaf | PhantomAir |  | 0 | PHANTOM | <div style='text-align: right'>1,109,850</div>  |
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> |  | 0 | PUBLISH | <div style='text-align: right'>828</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | SHINTW | <div style='text-align: right'>9,414,912</div>  |
| leaf | AccessAdapter<2> |  | 0 | SHINTW | <div style='text-align: right'>22</div>  |
| leaf | AccessAdapter<4> |  | 0 | SHINTW | <div style='text-align: right'>26</div>  |
| leaf | AccessAdapter<8> |  | 0 | SHINTW | <div style='text-align: right'>17</div>  |
| leaf | Boundary |  | 0 | SHINTW | <div style='text-align: right'>2,521,420</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | STOREW | <div style='text-align: right'>7,963,758</div>  |
| leaf | AccessAdapter<2> |  | 0 | STOREW | <div style='text-align: right'>67,573</div>  |
| leaf | AccessAdapter<4> |  | 0 | STOREW | <div style='text-align: right'>39,013</div>  |
| leaf | AccessAdapter<8> |  | 0 | STOREW | <div style='text-align: right'>1,768</div>  |
| leaf | Boundary |  | 0 | STOREW | <div style='text-align: right'>750,321</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | STOREW2 | <div style='text-align: right'>6,830,846</div>  |
| leaf | AccessAdapter<2> |  | 0 | STOREW2 | <div style='text-align: right'>418,066</div>  |
| leaf | AccessAdapter<4> |  | 0 | STOREW2 | <div style='text-align: right'>248,677</div>  |
| leaf | AccessAdapter<8> |  | 0 | STOREW2 | <div style='text-align: right'>108,868</div>  |
| leaf | Boundary |  | 0 | STOREW2 | <div style='text-align: right'>739,684</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | SUB | <div style='text-align: right'>1,743,150</div>  |
| leaf | AccessAdapter<2> |  | 0 | SUB | <div style='text-align: right'>59,235</div>  |
| leaf | AccessAdapter<4> |  | 0 | SUB | <div style='text-align: right'>70,005</div>  |
| leaf | Boundary |  | 0 | SUB | <div style='text-align: right'>15,180</div>  |

| group | idx | segment | total_cycles | trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | <div style='text-align: right'>3,335,094</div>  | <span style="color: green">(-8.0 [-1.2%])</span> <div style='text-align: right'>643.0</div>  |

| group | air_name | idx | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | ProgramAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | VmConnectorAir | 0 | <div style='text-align: right'>24</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | <div style='text-align: right'>2,496</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>64</div>  |
| leaf | VolatileBoundaryAir | 0 | <div style='text-align: right'>9,961,472</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>524,288</div>  |
| leaf | AccessAdapterAir<2> | 0 | <div style='text-align: right'>14,155,776</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>524,288</div>  |
| leaf | AccessAdapterAir<4> | 0 | <div style='text-align: right'>7,602,176</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>262,144</div>  |
| leaf | AccessAdapterAir<8> | 0 | <div style='text-align: right'>2,162,688</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>65,536</div>  |
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>12,582,912</div>  | <div style='text-align: right'>348</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>32,768</div>  |
| leaf | FriReducedOpeningAir | 0 | <div style='text-align: right'>18,350,080</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | <div style='text-align: right'>1,966,080</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>32,768</div>  |
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>104,857,600</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | <div style='text-align: right'>2,883,584</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | <div style='text-align: right'>53,477,376</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>1,048,576</div>  |
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | <div style='text-align: right'>136,314,880</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| leaf | PhantomAir | 0 | <div style='text-align: right'>3,670,016</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>262,144</div>  |
| leaf | VariableRangeCheckerAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1942aef9493a0e123bbf5ea00d0cf43fb3efe430/fibonacci-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/1942aef9493a0e123bbf5ea00d0cf43fb3efe430

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12477575228)