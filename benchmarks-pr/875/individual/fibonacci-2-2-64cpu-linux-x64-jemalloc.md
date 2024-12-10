| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| fibonacci_program | <div style='text-align: right'>2</div>  | <span style="color: green">(-131,072 [-0.3%])</span> <div style='text-align: right'>51,516,960</div>  | <div style='text-align: right'>1,500,219</div>  | <span style="color: green">(-28.0 [-0.4%])</span> <div style='text-align: right'>6,955.0</div>  |
| leaf_aggregation | <div style='text-align: right'>2</div>  | <span style="color: green">(-254,033 [-0.2%])</span> <div style='text-align: right'>143,362,100</div>  | <span style="color: green">(-1,504 [-0.0%])</span> <div style='text-align: right'>3,504,783</div>  | <span style="color: red">(+719.0 [+3.8%])</span> <div style='text-align: right'>19,598.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| group | collect_metrics | execute_time_ms | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- |
| fibonacci_program | true | <span style="color: red">(+752.0 [+14.0%])</span> <div style='text-align: right'>6,127.0</div>  | <span style="color: green">(-131,072 [-0.3%])</span> <div style='text-align: right'>51,516,960</div>  | <div style='text-align: right'>1,500,219</div>  |

| group | chip_name | collect_metrics | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | ProgramChip | true | <div style='text-align: right'>6,551</div>  |
| fibonacci_program | VmConnectorAir | true | <div style='text-align: right'>2</div>  |
| fibonacci_program | Boundary | true | <div style='text-align: right'>56</div>  |
| fibonacci_program | Merkle | true | <div style='text-align: right'>310</div>  |
| fibonacci_program | AccessAdapter<8> | true | <div style='text-align: right'>56</div>  |
| fibonacci_program | PhantomAir | true | <div style='text-align: right'>3</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true | <div style='text-align: right'>900,085</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | true | <div style='text-align: right'>300,004</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true | <div style='text-align: right'>4</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true | <div style='text-align: right'>57</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true | <div style='text-align: right'>200,012</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true | <div style='text-align: right'>11</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true | <div style='text-align: right'>100,012</div>  |
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | true | <div style='text-align: right'>17</div>  |
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | true | <div style='text-align: right'>11</div>  |
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | true | <div style='text-align: right'>3</div>  |
| fibonacci_program | Poseidon2VmAir<BabyBearParameters> | true | <div style='text-align: right'>366</div>  |
| fibonacci_program | BitwiseOperationLookupAir<8> | true | <div style='text-align: right'>65,536</div>  |
| fibonacci_program | RangeTupleCheckerAir<2> | true | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | VariableRangeCheckerAir | true | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | collect_metrics | dsl_ir | opcode | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program | true |  | ADD | <div style='text-align: right'>900,068</div>  |
| fibonacci_program | true |  | AND | <div style='text-align: right'>5</div>  |
| fibonacci_program | true |  | AUIPC | <div style='text-align: right'>11</div>  |
| fibonacci_program | true |  | BEQ | <div style='text-align: right'>100,005</div>  |
| fibonacci_program | true |  | BGEU | <div style='text-align: right'>3</div>  |
| fibonacci_program | true |  | BLT | <div style='text-align: right'>1</div>  |
| fibonacci_program | true |  | BLTU | <div style='text-align: right'>7</div>  |
| fibonacci_program | true |  | BNE | <div style='text-align: right'>100,007</div>  |
| fibonacci_program | true |  | HINT_STOREW | <div style='text-align: right'>3</div>  |
| fibonacci_program | true |  | JAL | <div style='text-align: right'>100,002</div>  |
| fibonacci_program | true |  | JALR | <div style='text-align: right'>17</div>  |
| fibonacci_program | true |  | LOADBU | <div style='text-align: right'>6</div>  |
| fibonacci_program | true |  | LOADW | <div style='text-align: right'>22</div>  |
| fibonacci_program | true |  | LUI | <div style='text-align: right'>10</div>  |
| fibonacci_program | true |  | OR | <div style='text-align: right'>4</div>  |
| fibonacci_program | true |  | PHANTOM | <div style='text-align: right'>3</div>  |
| fibonacci_program | true |  | SLL | <div style='text-align: right'>3</div>  |
| fibonacci_program | true |  | SLTU | <div style='text-align: right'>300,004</div>  |
| fibonacci_program | true |  | SRL | <div style='text-align: right'>1</div>  |
| fibonacci_program | true |  | STOREB | <div style='text-align: right'>1</div>  |
| fibonacci_program | true |  | STOREW | <div style='text-align: right'>28</div>  |
| fibonacci_program | true |  | SUB | <div style='text-align: right'>4</div>  |
| fibonacci_program | true |  | XOR | <div style='text-align: right'>4</div>  |

| group | air_name | collect_metrics | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | ADD | <div style='text-align: right'>32,402,448</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | ADD | <div style='text-align: right'>51</div>  |
| fibonacci_program | Boundary | true |  | ADD | <div style='text-align: right'>120</div>  |
| fibonacci_program | Merkle | true |  | ADD | <div style='text-align: right'>64</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | AND | <div style='text-align: right'>180</div>  |
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | true |  | AUIPC | <div style='text-align: right'>231</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | AUIPC | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary | true |  | AUIPC | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle | true |  | AUIPC | <div style='text-align: right'>3,456</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true |  | BEQ | <div style='text-align: right'>2,600,130</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BGEU | <div style='text-align: right'>96</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BLT | <div style='text-align: right'>32</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BLTU | <div style='text-align: right'>224</div>  |
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true |  | BNE | <div style='text-align: right'>2,600,182</div>  |
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | true |  | HINT_STOREW | <div style='text-align: right'>78</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | HINT_STOREW | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary | true |  | HINT_STOREW | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle | true |  | HINT_STOREW | <div style='text-align: right'>192</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true |  | JAL | <div style='text-align: right'>1,800,036</div>  |
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | true |  | JALR | <div style='text-align: right'>476</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | LOADBU | <div style='text-align: right'>240</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | LOADW | <div style='text-align: right'>880</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | LOADW | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary | true |  | LOADW | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle | true |  | LOADW | <div style='text-align: right'>2,304</div>  |
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true |  | LUI | <div style='text-align: right'>180</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | OR | <div style='text-align: right'>144</div>  |
| fibonacci_program | PhantomAir | true |  | PHANTOM | <div style='text-align: right'>18</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true |  | SLL | <div style='text-align: right'>159</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | true |  | SLTU | <div style='text-align: right'>11,100,148</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | SLTU | <div style='text-align: right'>34</div>  |
| fibonacci_program | Boundary | true |  | SLTU | <div style='text-align: right'>80</div>  |
| fibonacci_program | Merkle | true |  | SLTU | <div style='text-align: right'>64</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true |  | SRL | <div style='text-align: right'>53</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | STOREB | <div style='text-align: right'>40</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | STOREB | <div style='text-align: right'>17</div>  |
| fibonacci_program | Boundary | true |  | STOREB | <div style='text-align: right'>40</div>  |
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | STOREW | <div style='text-align: right'>1,120</div>  |
| fibonacci_program | AccessAdapter<8> | true |  | STOREW | <div style='text-align: right'>272</div>  |
| fibonacci_program | Boundary | true |  | STOREW | <div style='text-align: right'>640</div>  |
| fibonacci_program | Merkle | true |  | STOREW | <div style='text-align: right'>3,776</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | SUB | <div style='text-align: right'>144</div>  |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | XOR | <div style='text-align: right'>144</div>  |

| group | commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms | num_segments | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | <div style='text-align: right'>7.0</div>  | <span style="color: red">(+103.0 [+10.6%])</span> <div style='text-align: right'>1,072.0</div>  | <span style="color: red">(+100.0 [+12.4%])</span> <div style='text-align: right'>906.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-6.0 [-2.8%])</span> <div style='text-align: right'>207.0</div>  | <div style='text-align: right'>1</div>  | <span style="color: green">(-131,072 [-0.3%])</span> <div style='text-align: right'>51,516,960</div>  | <div style='text-align: right'>1,500,219</div>  | <span style="color: green">(-28.0 [-0.4%])</span> <div style='text-align: right'>6,955.0</div>  |
| leaf_aggregation |  |  |  | <div style='text-align: right'>2</div>  |  |  | <span style="color: green">(-254,033 [-0.2%])</span> <div style='text-align: right'>143,362,100</div>  | <span style="color: green">(-1,504 [-0.0%])</span> <div style='text-align: right'>3,504,783</div>  | <span style="color: red">(+719.0 [+3.8%])</span> <div style='text-align: right'>19,598.0</div>  |

| group | air_name | constraints | interactions | quotient_deg |
| --- | --- | --- | --- | --- |
| fibonacci_program | ProgramAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| fibonacci_program | VmConnectorAir | <div style='text-align: right'>9</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | PersistentBoundaryAir<8> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | MemoryMerkleAir<8> | <div style='text-align: right'>40</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<2> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<4> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<8> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<16> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<32> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | AccessAdapterAir<64> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | PhantomAir | <div style='text-align: right'>5</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | <div style='text-align: right'>43</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | <div style='text-align: right'>39</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | <div style='text-align: right'>90</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | <div style='text-align: right'>33</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | <div style='text-align: right'>25</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | <div style='text-align: right'>41</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | <div style='text-align: right'>22</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | <div style='text-align: right'>20</div>  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | <div style='text-align: right'>15</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | <div style='text-align: right'>26</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>24</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | <div style='text-align: right'>88</div>  | <div style='text-align: right'>25</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | <div style='text-align: right'>17</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | Poseidon2VmAir<BabyBearParameters> | <div style='text-align: right'>525</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | BitwiseOperationLookupAir<8> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>2</div>  |
| fibonacci_program | RangeTupleCheckerAir<2> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| fibonacci_program | VariableRangeCheckerAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |

| group | air_name | segment | cells | constraints | interactions | main_cols | perm_cols | prep_cols | quotient_deg | rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | ProgramAir | 0 | <div style='text-align: right'>147,456</div>  |  |  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  |  | <div style='text-align: right'>8,192</div>  |
| fibonacci_program | VmConnectorAir | 0 | <div style='text-align: right'>32</div>  |  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>1</div>  |  | <div style='text-align: right'>2</div>  |
| fibonacci_program | PersistentBoundaryAir<8> | 0 | <div style='text-align: right'>2,048</div>  |  |  | <div style='text-align: right'>20</div>  | <div style='text-align: right'>12</div>  |  |  | <div style='text-align: right'>64</div>  |
| fibonacci_program | MemoryMerkleAir<8> | 0 | <div style='text-align: right'>26,624</div>  |  |  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>20</div>  |  |  | <div style='text-align: right'>512</div>  |
| fibonacci_program | AccessAdapterAir<8> | 0 | <div style='text-align: right'>2,624</div>  |  |  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>24</div>  |  |  | <div style='text-align: right'>64</div>  |
| fibonacci_program | PhantomAir | 0 | <div style='text-align: right'>72</div>  |  |  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>12</div>  |  |  | <div style='text-align: right'>4</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | <div style='text-align: right'>121,634,816</div>  |  |  | <div style='text-align: right'>36</div>  | <div style='text-align: right'>80</div>  |  |  | <div style='text-align: right'>1,048,576</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>40,370,176</div>  |  |  | <div style='text-align: right'>37</div>  | <div style='text-align: right'>40</div>  |  |  | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | <div style='text-align: right'>420</div>  |  |  | <div style='text-align: right'>53</div>  | <div style='text-align: right'>52</div>  |  |  | <div style='text-align: right'>4</div>  |
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | <div style='text-align: right'>7,168</div>  |  |  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>72</div>  |  |  | <div style='text-align: right'>64</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | <div style='text-align: right'>19,398,656</div>  |  |  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>48</div>  |  |  | <div style='text-align: right'>262,144</div>  |
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>1,408</div>  |  |  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>56</div>  |  |  | <div style='text-align: right'>16</div>  |
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>8,126,464</div>  |  |  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>44</div>  |  |  | <div style='text-align: right'>131,072</div>  |
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | <div style='text-align: right'>2,048</div>  |  |  | <div style='text-align: right'>28</div>  | <div style='text-align: right'>36</div>  |  |  | <div style='text-align: right'>32</div>  |
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>784</div>  |  |  | <div style='text-align: right'>21</div>  | <div style='text-align: right'>28</div>  |  |  | <div style='text-align: right'>16</div>  |
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>248</div>  |  |  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>36</div>  |  |  | <div style='text-align: right'>4</div>  |
| fibonacci_program | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>321,024</div>  |  |  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>68</div>  |  |  | <div style='text-align: right'>512</div>  |
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>655,360</div>  |  |  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  |  | <div style='text-align: right'>65,536</div>  |
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>4,718,592</div>  |  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  |  | <div style='text-align: right'>524,288</div>  |
| fibonacci_program | VariableRangeCheckerAir | 0 | <span style="color: green">(-1,179,648 [-50.0%])</span> <div style='text-align: right'>1,179,648</div>  |  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  |  | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | ProgramAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | VmConnectorAir | 0 | <div style='text-align: right'>24</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| leaf_aggregation | VolatileBoundaryAir | 0 | <div style='text-align: right'>9,961,472</div>  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>524,288</div>  |
| leaf_aggregation | AccessAdapterAir<2> | 0 | <div style='text-align: right'>14,155,776</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>524,288</div>  |
| leaf_aggregation | AccessAdapterAir<4> | 0 | <div style='text-align: right'>7,602,176</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>262,144</div>  |
| leaf_aggregation | AccessAdapterAir<8> | 0 | <div style='text-align: right'>2,162,688</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>65,536</div>  |
| leaf_aggregation | PhantomAir | 0 | <div style='text-align: right'>3,670,016</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>262,144</div>  |
| leaf_aggregation | VmAirWrapper<NativeLoadStoreAdapterAir<1>, KernelLoadStoreCoreAir<1> | 0 | <div style='text-align: right'>136,314,880</div>  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2,097,152</div>  |
| leaf_aggregation | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | <div style='text-align: right'>53,477,376</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1,048,576</div>  |
| leaf_aggregation | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | <div style='text-align: right'>2,883,584</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>7</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>104,857,600</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2,097,152</div>  |
| leaf_aggregation | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | <div style='text-align: right'>3,932,160</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>65,536</div>  |
| leaf_aggregation | FriReducedOpeningAir | 0 | <div style='text-align: right'>36,700,160</div>  | <div style='text-align: right'>59</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>262,144</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>19,496,960</div>  | <div style='text-align: right'>517</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>32,768</div>  |
| leaf_aggregation | VariableRangeCheckerAir | 0 | <span style="color: green">(-1,179,648 [-50.0%])</span> <div style='text-align: right'>1,179,648</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | segment | commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms | num_segments | stark_prove_excluding_trace_time_ms | total_cells | verify_program_compile_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 |  | <span style="color: red">(+2.0 [+1.2%])</span> <div style='text-align: right'>163.0</div>  |  |  |  |  | <span style="color: green">(-133.0 [-2.3%])</span> <div style='text-align: right'>5,720.0</div>  | <span style="color: green">(-1,179,648 [-0.6%])</span> <div style='text-align: right'>196,595,668</div>  |  |
| leaf_aggregation | 0 | <span style="color: green">(-6.0 [-12.5%])</span> <div style='text-align: right'>42.0</div>  | <span style="color: red">(+152.0 [+4.0%])</span> <div style='text-align: right'>3,912.0</div>  | <span style="color: red">(+130.0 [+4.1%])</span> <div style='text-align: right'>3,306.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-34.0 [-44.2%])</span> <div style='text-align: right'>43.0</div>  | <div style='text-align: right'>1</div>  | <span style="color: red">(+567.0 [+3.8%])</span> <div style='text-align: right'>15,686.0</div>  | <span style="color: green">(-1,179,648 [-0.3%])</span> <div style='text-align: right'>398,753,816</div>  | <span style="color: green">(-44.0 [-17.0%])</span> <div style='text-align: right'>215.0</div>  |

| group | collect_metrics | segment | execute_time_ms | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- | --- |
| leaf_aggregation | true | 0 | <span style="color: red">(+1,441.0 [+9.6%])</span> <div style='text-align: right'>16,387.0</div>  | <span style="color: green">(-254,033 [-0.2%])</span> <div style='text-align: right'>143,362,100</div>  | <span style="color: green">(-1,504 [-0.0%])</span> <div style='text-align: right'>3,504,783</div>  |

| group | chip_name | collect_metrics | segment | rows_used |
| --- | --- | --- | --- | --- |
| leaf_aggregation | ProgramChip | true | 0 | <span style="color: green">(-20 [-0.0%])</span> <div style='text-align: right'>104,705</div>  |
| leaf_aggregation | VmConnectorAir | true | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | Boundary | true | 0 | <span style="color: green">(-462 [-0.1%])</span> <div style='text-align: right'>421,443</div>  |
| leaf_aggregation | AccessAdapter<2> | true | 0 | <span style="color: green">(-288 [-0.1%])</span> <div style='text-align: right'>400,972</div>  |
| leaf_aggregation | AccessAdapter<4> | true | 0 | <span style="color: green">(-144 [-0.1%])</span> <div style='text-align: right'>200,738</div>  |
| leaf_aggregation | AccessAdapter<8> | true | 0 | <span style="color: green">(-84 [-0.1%])</span> <div style='text-align: right'>58,224</div>  |
| leaf_aggregation | PhantomAir | true | 0 | <span style="color: green">(-42 [-0.0%])</span> <div style='text-align: right'>209,823</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | 0 | <span style="color: green">(-772 [-0.1%])</span> <div style='text-align: right'>1,123,644</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | 0 | <span style="color: green">(-633 [-0.1%])</span> <div style='text-align: right'>673,771</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | 0 | <span style="color: red">(+2,048 [+2.8%])</span> <div style='text-align: right'>75,071</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | 0 | <span style="color: green">(-2,059 [-0.2%])</span> <div style='text-align: right'>1,354,215</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | 0 | <span style="color: green">(-4 [-0.0%])</span> <div style='text-align: right'>34,990</div>  |
| leaf_aggregation | FriReducedOpeningAir | true | 0 | <div style='text-align: right'>144,732</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>27,935</div>  |
| leaf_aggregation | VariableRangeCheckerAir | true | 0 | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | collect_metrics | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- | --- |
| leaf_aggregation | true |  | JAL | 0 | <div style='text-align: right'>1</div>  |
| leaf_aggregation | true |  | STOREW | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | AddE | FE4ADD | 0 | <div style='text-align: right'>13,124</div>  |
| leaf_aggregation | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>176</div>  |
| leaf_aggregation | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>528</div>  |
| leaf_aggregation | true | AddEFI | ADD | 0 | <div style='text-align: right'>312</div>  |
| leaf_aggregation | true | AddEI | ADD | 0 | <span style="color: red">(+4 [+0.0%])</span> <div style='text-align: right'>30,728</div>  |
| leaf_aggregation | true | AddF | ADD | 0 | <div style='text-align: right'>1,333</div>  |
| leaf_aggregation | true | AddFI | ADD | 0 | <span style="color: green">(-168 [-0.4%])</span> <div style='text-align: right'>44,208</div>  |
| leaf_aggregation | true | AddV | ADD | 0 | <span style="color: green">(-2 [-0.0%])</span> <div style='text-align: right'>14,806</div>  |
| leaf_aggregation | true | AddVI | ADD | 0 | <span style="color: green">(-420 [-0.1%])</span> <div style='text-align: right'>352,358</div>  |
| leaf_aggregation | true | Alloc | ADD | 0 | <span style="color: green">(-84 [-0.1%])</span> <div style='text-align: right'>56,236</div>  |
| leaf_aggregation | true | Alloc | LOADW | 0 | <span style="color: green">(-84 [-0.1%])</span> <div style='text-align: right'>56,236</div>  |
| leaf_aggregation | true | Alloc | MUL | 0 | <span style="color: green">(-42 [-0.1%])</span> <div style='text-align: right'>33,458</div>  |
| leaf_aggregation | true | AssertEqE | BNE | 0 | <div style='text-align: right'>248</div>  |
| leaf_aggregation | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>4</div>  |
| leaf_aggregation | true | AssertEqF | BNE | 0 | <div style='text-align: right'>10,784</div>  |
| leaf_aggregation | true | AssertEqV | BNE | 0 | <div style='text-align: right'>1,073</div>  |
| leaf_aggregation | true | AssertEqVI | BNE | 0 | <div style='text-align: right'>237</div>  |
| leaf_aggregation | true | CT-InitializePcsConst | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-ReadingProofFromInput | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-VerifierProgram | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-compute-reduced-opening | PHANTOM | 0 | <div style='text-align: right'>672</div>  |
| leaf_aggregation | true | CT-exp-reverse-bits-len | PHANTOM | 0 | <div style='text-align: right'>6,888</div>  |
| leaf_aggregation | true | CT-poseidon2-hash | PHANTOM | 0 | <div style='text-align: right'>3,444</div>  |
| leaf_aggregation | true | CT-poseidon2-hash-ext | PHANTOM | 0 | <div style='text-align: right'>1,680</div>  |
| leaf_aggregation | true | CT-poseidon2-hash-setup | PHANTOM | 0 | <div style='text-align: right'>150,948</div>  |
| leaf_aggregation | true | CT-single-reduced-opening-eval | PHANTOM | 0 | <div style='text-align: right'>10,668</div>  |
| leaf_aggregation | true | CT-stage-c-build-rounds | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-1-verify-shape-and-sample-challenges | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-2-fri-fold | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-3-verify-challenges | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-verify-pcs | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-e-verify-constraints | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-verify-batch | PHANTOM | 0 | <div style='text-align: right'>672</div>  |
| leaf_aggregation | true | CT-verify-batch-ext | PHANTOM | 0 | <div style='text-align: right'>1,680</div>  |
| leaf_aggregation | true | CT-verify-batch-reduce-fast | PHANTOM | 0 | <div style='text-align: right'>5,124</div>  |
| leaf_aggregation | true | CT-verify-batch-reduce-fast-setup | PHANTOM | 0 | <div style='text-align: right'>5,124</div>  |
| leaf_aggregation | true | CT-verify-query | PHANTOM | 0 | <div style='text-align: right'>84</div>  |
| leaf_aggregation | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>6,214</div>  |
| leaf_aggregation | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>54</div>  |
| leaf_aggregation | true | DivEIN | STOREW | 0 | <div style='text-align: right'>216</div>  |
| leaf_aggregation | true | DivFIN | DIV | 0 | <div style='text-align: right'>128</div>  |
| leaf_aggregation | true | For | ADD | 0 | <span style="color: green">(-465 [-0.1%])</span> <div style='text-align: right'>428,500</div>  |
| leaf_aggregation | true | For | BNE | 0 | <span style="color: green">(-507 [-0.1%])</span> <div style='text-align: right'>472,386</div>  |
| leaf_aggregation | true | For | JAL | 0 | <span style="color: green">(-42 [-0.1%])</span> <div style='text-align: right'>43,886</div>  |
| leaf_aggregation | true | For | LOADW | 0 | <div style='text-align: right'>2,604</div>  |
| leaf_aggregation | true | For | STOREW | 0 | <span style="color: green">(-42 [-0.1%])</span> <div style='text-align: right'>41,282</div>  |
| leaf_aggregation | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>5,334</div>  |
| leaf_aggregation | true | HintBitsF | PHANTOM | 0 | <div style='text-align: right'>43</div>  |
| leaf_aggregation | true | HintInputVec | PHANTOM | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>22,778</div>  |
| leaf_aggregation | true | IfEq | BNE | 0 | <span style="color: red">(+126 [+0.5%])</span> <div style='text-align: right'>26,433</div>  |
| leaf_aggregation | true | IfEqI | BNE | 0 | <span style="color: green">(-210 [-0.1%])</span> <div style='text-align: right'>144,176</div>  |
| leaf_aggregation | true | IfEqI | JAL | 0 | <span style="color: red">(+2,090 [+7.2%])</span> <div style='text-align: right'>31,182</div>  |
| leaf_aggregation | true | IfNe | BEQ | 0 | <span style="color: green">(-42 [-0.3%])</span> <div style='text-align: right'>15,767</div>  |
| leaf_aggregation | true | IfNe | JAL | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | IfNeI | BEQ | 0 | <div style='text-align: right'>2,663</div>  |
| leaf_aggregation | true | ImmE | STOREW | 0 | <span style="color: green">(-16 [-0.4%])</span> <div style='text-align: right'>3,832</div>  |
| leaf_aggregation | true | ImmF | STOREW | 0 | <div style='text-align: right'>42,709</div>  |
| leaf_aggregation | true | ImmV | STOREW | 0 | <div style='text-align: right'>29,283</div>  |
| leaf_aggregation | true | LoadE | LOADW | 0 | <div style='text-align: right'>24,036</div>  |
| leaf_aggregation | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>65,768</div>  |
| leaf_aggregation | true | LoadF | LOADW | 0 | <div style='text-align: right'>27,954</div>  |
| leaf_aggregation | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>93,977</div>  |
| leaf_aggregation | true | LoadV | LOADW | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>25,818</div>  |
| leaf_aggregation | true | LoadV | LOADW2 | 0 | <span style="color: green">(-126 [-0.1%])</span> <div style='text-align: right'>200,346</div>  |
| leaf_aggregation | true | MulE | BBE4MUL | 0 | <span style="color: green">(-4 [-0.0%])</span> <div style='text-align: right'>10,395</div>  |
| leaf_aggregation | true | MulEF | MUL | 0 | <div style='text-align: right'>3,792</div>  |
| leaf_aggregation | true | MulEFI | MUL | 0 | <div style='text-align: right'>556</div>  |
| leaf_aggregation | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>1,646</div>  |
| leaf_aggregation | true | MulEI | STOREW | 0 | <div style='text-align: right'>6,584</div>  |
| leaf_aggregation | true | MulF | MUL | 0 | <span style="color: green">(-336 [-0.4%])</span> <div style='text-align: right'>85,683</div>  |
| leaf_aggregation | true | MulFI | MUL | 0 | <div style='text-align: right'>1,353</div>  |
| leaf_aggregation | true | MulVI | MUL | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>20,056</div>  |
| leaf_aggregation | true | NegE | MUL | 0 | <div style='text-align: right'>204</div>  |
| leaf_aggregation | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>17,010</div>  |
| leaf_aggregation | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <div style='text-align: right'>10,925</div>  |
| leaf_aggregation | true | StoreE | STOREW | 0 | <div style='text-align: right'>24,448</div>  |
| leaf_aggregation | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>34,352</div>  |
| leaf_aggregation | true | StoreF | STOREW | 0 | <div style='text-align: right'>34,398</div>  |
| leaf_aggregation | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>82,847</div>  |
| leaf_aggregation | true | StoreHintWord | ADD | 0 | <span style="color: green">(-336 [-0.2%])</span> <div style='text-align: right'>220,603</div>  |
| leaf_aggregation | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-378 [-0.2%])</span> <div style='text-align: right'>244,714</div>  |
| leaf_aggregation | true | StoreV | STOREW | 0 | <div style='text-align: right'>3,018</div>  |
| leaf_aggregation | true | StoreV | STOREW2 | 0 | <span style="color: green">(-84 [-0.1%])</span> <div style='text-align: right'>62,334</div>  |
| leaf_aggregation | true | SubE | FE4SUB | 0 | <div style='text-align: right'>3,557</div>  |
| leaf_aggregation | true | SubEF | LOADW | 0 | <div style='text-align: right'>16,182</div>  |
| leaf_aggregation | true | SubEF | SUB | 0 | <div style='text-align: right'>5,394</div>  |
| leaf_aggregation | true | SubEFI | ADD | 0 | <div style='text-align: right'>356</div>  |
| leaf_aggregation | true | SubEI | ADD | 0 | <div style='text-align: right'>432</div>  |
| leaf_aggregation | true | SubFI | SUB | 0 | <div style='text-align: right'>1,333</div>  |
| leaf_aggregation | true | SubV | SUB | 0 | <span style="color: green">(-168 [-0.3%])</span> <div style='text-align: right'>50,546</div>  |
| leaf_aggregation | true | SubVI | SUB | 0 | <div style='text-align: right'>1,000</div>  |
| leaf_aggregation | true | SubVIN | SUB | 0 | <div style='text-align: right'>840</div>  |

| group | air_name | collect_metrics | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- | --- |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true |  | JAL | 0 | <div style='text-align: right'>10</div>  |
| leaf_aggregation | Boundary | true |  | JAL | 0 | <div style='text-align: right'>11</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | STOREW | 0 | <div style='text-align: right'>82</div>  |
| leaf_aggregation | Boundary | true |  | STOREW | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | AddE | FE4ADD | 0 | <div style='text-align: right'>524,960</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddE | FE4ADD | 0 | <span style="color: red">(+132 [+0.1%])</span> <div style='text-align: right'>237,248</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddE | FE4ADD | 0 | <span style="color: red">(+78 [+0.1%])</span> <div style='text-align: right'>140,192</div>  |
| leaf_aggregation | Boundary | true | AddE | FE4ADD | 0 | <div style='text-align: right'>112,508</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>7,216</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>1,089</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>1,287</div>  |
| leaf_aggregation | Boundary | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>308</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>21,648</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>1,089</div>  |
| leaf_aggregation | Boundary | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>924</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddEFI | ADD | 0 | <div style='text-align: right'>9,360</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFI | ADD | 0 | <div style='text-align: right'>1,298</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEFI | ADD | 0 | <div style='text-align: right'>767</div>  |
| leaf_aggregation | Boundary | true | AddEFI | ADD | 0 | <div style='text-align: right'>1,364</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddEI | ADD | 0 | <span style="color: red">(+120 [+0.0%])</span> <div style='text-align: right'>921,840</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEI | ADD | 0 | <span style="color: red">(+242 [+0.1%])</span> <div style='text-align: right'>190,124</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEI | ADD | 0 | <span style="color: red">(+143 [+0.1%])</span> <div style='text-align: right'>112,346</div>  |
| leaf_aggregation | Boundary | true | AddEI | ADD | 0 | <span style="color: red">(+44 [+0.0%])</span> <div style='text-align: right'>136,708</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddF | ADD | 0 | <div style='text-align: right'>39,990</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddFI | ADD | 0 | <span style="color: green">(-5,040 [-0.4%])</span> <div style='text-align: right'>1,326,240</div>  |
| leaf_aggregation | Boundary | true | AddFI | ADD | 0 | <div style='text-align: right'>253</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddV | ADD | 0 | <span style="color: green">(-60 [-0.0%])</span> <div style='text-align: right'>444,180</div>  |
| leaf_aggregation | Boundary | true | AddV | ADD | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddVI | ADD | 0 | <span style="color: green">(-12,600 [-0.1%])</span> <div style='text-align: right'>10,570,740</div>  |
| leaf_aggregation | Boundary | true | AddVI | ADD | 0 | <div style='text-align: right'>1,232</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | Alloc | ADD | 0 | <span style="color: green">(-2,520 [-0.1%])</span> <div style='text-align: right'>1,687,080</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | Alloc | LOADW | 0 | <span style="color: green">(-3,444 [-0.1%])</span> <div style='text-align: right'>2,305,676</div>  |
| leaf_aggregation | Boundary | true | Alloc | LOADW | 0 | <div style='text-align: right'>1,133</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | Alloc | MUL | 0 | <span style="color: green">(-1,260 [-0.1%])</span> <div style='text-align: right'>1,003,740</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Alloc | MUL | 0 | <div style='text-align: right'>33</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Alloc | MUL | 0 | <div style='text-align: right'>39</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>5,704</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>1,364</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>806</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>92</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>13</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqF | BNE | 0 | <div style='text-align: right'>248,032</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqV | BNE | 0 | <div style='text-align: right'>24,679</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqVI | BNE | 0 | <div style='text-align: right'>5,451</div>  |
| leaf_aggregation | PhantomAir | true | CT-InitializePcsConst | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-ReadingProofFromInput | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-VerifierProgram | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-compute-reduced-opening | PHANTOM | 0 | <div style='text-align: right'>4,032</div>  |
| leaf_aggregation | PhantomAir | true | CT-exp-reverse-bits-len | PHANTOM | 0 | <div style='text-align: right'>41,328</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash | PHANTOM | 0 | <div style='text-align: right'>20,664</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash-ext | PHANTOM | 0 | <div style='text-align: right'>10,080</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash-setup | PHANTOM | 0 | <div style='text-align: right'>905,688</div>  |
| leaf_aggregation | PhantomAir | true | CT-single-reduced-opening-eval | PHANTOM | 0 | <div style='text-align: right'>64,008</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-c-build-rounds | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-1-verify-shape-and-sample-challenges | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-2-fri-fold | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-3-verify-challenges | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-verify-pcs | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-e-verify-constraints | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch | PHANTOM | 0 | <div style='text-align: right'>4,032</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-ext | PHANTOM | 0 | <div style='text-align: right'>10,080</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-reduce-fast | PHANTOM | 0 | <div style='text-align: right'>30,744</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-reduce-fast-setup | PHANTOM | 0 | <div style='text-align: right'>30,744</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-query | PHANTOM | 0 | <div style='text-align: right'>504</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>248,560</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>118,690</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>70,135</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>2,160</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>2,398</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>1,417</div>  |
| leaf_aggregation | Boundary | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>528</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>8,856</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>781</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>221</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | DivFIN | DIV | 0 | <div style='text-align: right'>3,840</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | For | ADD | 0 | <span style="color: green">(-13,950 [-0.1%])</span> <div style='text-align: right'>12,855,000</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | For | BNE | 0 | <span style="color: green">(-11,661 [-0.1%])</span> <div style='text-align: right'>10,864,878</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | For | JAL | 0 | <span style="color: green">(-420 [-0.1%])</span> <div style='text-align: right'>438,860</div>  |
| leaf_aggregation | AccessAdapter<2> | true | For | JAL | 0 | <div style='text-align: right'>407</div>  |
| leaf_aggregation | AccessAdapter<4> | true | For | JAL | 0 | <div style='text-align: right'>481</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | For | LOADW | 0 | <div style='text-align: right'>106,764</div>  |
| leaf_aggregation | Boundary | true | For | LOADW | 0 | <div style='text-align: right'>473</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | For | STOREW | 0 | <span style="color: green">(-1,722 [-0.1%])</span> <div style='text-align: right'>1,692,562</div>  |
| leaf_aggregation | Boundary | true | For | STOREW | 0 | <div style='text-align: right'>748</div>  |
| leaf_aggregation | AccessAdapter<2> | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>151,580</div>  |
| leaf_aggregation | AccessAdapter<4> | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>89,570</div>  |
| leaf_aggregation | FriReducedOpeningAir | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>9,262,848</div>  |
| leaf_aggregation | PhantomAir | true | HintBitsF | PHANTOM | 0 | <div style='text-align: right'>258</div>  |
| leaf_aggregation | PhantomAir | true | HintInputVec | PHANTOM | 0 | <span style="color: green">(-252 [-0.2%])</span> <div style='text-align: right'>136,668</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfEq | BNE | 0 | <span style="color: red">(+2,898 [+0.5%])</span> <div style='text-align: right'>607,959</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfEqI | BNE | 0 | <span style="color: green">(-4,830 [-0.1%])</span> <div style='text-align: right'>3,316,048</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | IfEqI | JAL | 0 | <span style="color: red">(+20,900 [+7.2%])</span> <div style='text-align: right'>311,820</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfNe | BEQ | 0 | <span style="color: green">(-966 [-0.3%])</span> <div style='text-align: right'>362,641</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | IfNe | JAL | 0 | <div style='text-align: right'>20</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfNeI | BEQ | 0 | <div style='text-align: right'>61,249</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmE | STOREW | 0 | <span style="color: green">(-656 [-0.4%])</span> <div style='text-align: right'>157,112</div>  |
| leaf_aggregation | AccessAdapter<2> | true | ImmE | STOREW | 0 | <span style="color: red">(+88 [+1.0%])</span> <div style='text-align: right'>8,866</div>  |
| leaf_aggregation | AccessAdapter<4> | true | ImmE | STOREW | 0 | <span style="color: red">(+52 [+1.0%])</span> <div style='text-align: right'>5,239</div>  |
| leaf_aggregation | Boundary | true | ImmE | STOREW | 0 | <span style="color: green">(-44 [-0.3%])</span> <div style='text-align: right'>13,772</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmF | STOREW | 0 | <div style='text-align: right'>1,751,069</div>  |
| leaf_aggregation | Boundary | true | ImmF | STOREW | 0 | <div style='text-align: right'>16,291</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmV | STOREW | 0 | <div style='text-align: right'>1,200,603</div>  |
| leaf_aggregation | Boundary | true | ImmV | STOREW | 0 | <div style='text-align: right'>1,716</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadE | LOADW | 0 | <div style='text-align: right'>985,476</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadE | LOADW | 0 | <div style='text-align: right'>170,280</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadE | LOADW | 0 | <div style='text-align: right'>100,620</div>  |
| leaf_aggregation | Boundary | true | LoadE | LOADW | 0 | <div style='text-align: right'>3,740</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>2,696,488</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>56,408</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>33,332</div>  |
| leaf_aggregation | Boundary | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>44</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadF | LOADW | 0 | <div style='text-align: right'>1,146,114</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadF | LOADW | 0 | <div style='text-align: right'>51,744</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadF | LOADW | 0 | <div style='text-align: right'>30,576</div>  |
| leaf_aggregation | AccessAdapter<8> | true | LoadF | LOADW | 0 | <div style='text-align: right'>19,992</div>  |
| leaf_aggregation | Boundary | true | LoadF | LOADW | 0 | <div style='text-align: right'>14,905</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>3,853,057</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>792</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>468</div>  |
| leaf_aggregation | AccessAdapter<8> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>493</div>  |
| leaf_aggregation | Boundary | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>638</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadV | LOADW | 0 | <span style="color: green">(-1,722 [-0.2%])</span> <div style='text-align: right'>1,058,538</div>  |
| leaf_aggregation | Boundary | true | LoadV | LOADW | 0 | <div style='text-align: right'>440</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadV | LOADW2 | 0 | <span style="color: green">(-5,166 [-0.1%])</span> <div style='text-align: right'>8,214,186</div>  |
| leaf_aggregation | Boundary | true | LoadV | LOADW2 | 0 | <div style='text-align: right'>979</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | MulE | BBE4MUL | 0 | <span style="color: green">(-160 [-0.0%])</span> <div style='text-align: right'>415,800</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulE | BBE4MUL | 0 | <span style="color: red">(+176 [+0.1%])</span> <div style='text-align: right'>229,284</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulE | BBE4MUL | 0 | <span style="color: red">(+104 [+0.1%])</span> <div style='text-align: right'>135,486</div>  |
| leaf_aggregation | Boundary | true | MulE | BBE4MUL | 0 | <div style='text-align: right'>136,752</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulEF | MUL | 0 | <div style='text-align: right'>113,760</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEF | MUL | 0 | <span style="color: green">(-66 [-0.3%])</span> <div style='text-align: right'>19,118</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEF | MUL | 0 | <span style="color: green">(-39 [-0.3%])</span> <div style='text-align: right'>11,297</div>  |
| leaf_aggregation | Boundary | true | MulEF | MUL | 0 | <div style='text-align: right'>1,056</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulEFI | MUL | 0 | <div style='text-align: right'>16,680</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEFI | MUL | 0 | <span style="color: red">(+22 [+0.7%])</span> <div style='text-align: right'>3,036</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEFI | MUL | 0 | <span style="color: red">(+13 [+0.7%])</span> <div style='text-align: right'>1,794</div>  |
| leaf_aggregation | Boundary | true | MulEFI | MUL | 0 | <div style='text-align: right'>1,496</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>65,840</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEI | BBE4MUL | 0 | <span style="color: green">(-44 [-0.1%])</span> <div style='text-align: right'>74,140</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEI | BBE4MUL | 0 | <span style="color: green">(-26 [-0.1%])</span> <div style='text-align: right'>43,810</div>  |
| leaf_aggregation | Boundary | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>4,312</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | MulEI | STOREW | 0 | <div style='text-align: right'>269,944</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEI | STOREW | 0 | <div style='text-align: right'>36,157</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEI | STOREW | 0 | <div style='text-align: right'>21,346</div>  |
| leaf_aggregation | Boundary | true | MulEI | STOREW | 0 | <div style='text-align: right'>33</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulF | MUL | 0 | <span style="color: green">(-10,080 [-0.4%])</span> <div style='text-align: right'>2,570,490</div>  |
| leaf_aggregation | Boundary | true | MulF | MUL | 0 | <div style='text-align: right'>14,630</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulFI | MUL | 0 | <div style='text-align: right'>40,590</div>  |
| leaf_aggregation | Boundary | true | MulFI | MUL | 0 | <div style='text-align: right'>14,641</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulVI | MUL | 0 | <span style="color: green">(-1,260 [-0.2%])</span> <div style='text-align: right'>601,680</div>  |
| leaf_aggregation | Boundary | true | MulVI | MUL | 0 | <div style='text-align: right'>77</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | NegE | MUL | 0 | <div style='text-align: right'>6,120</div>  |
| leaf_aggregation | AccessAdapter<2> | true | NegE | MUL | 0 | <span style="color: green">(-22 [-1.4%])</span> <div style='text-align: right'>1,518</div>  |
| leaf_aggregation | AccessAdapter<4> | true | NegE | MUL | 0 | <span style="color: green">(-13 [-1.4%])</span> <div style='text-align: right'>897</div>  |
| leaf_aggregation | Boundary | true | NegE | MUL | 0 | <div style='text-align: right'>792</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-1,848 [-0.3%])</span> <div style='text-align: right'>687,456</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-1,092 [-0.3%])</span> <div style='text-align: right'>406,224</div>  |
| leaf_aggregation | AccessAdapter<8> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-714 [-0.3%])</span> <div style='text-align: right'>265,608</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-23,478 [-0.2%])</span> <div style='text-align: right'>9,508,590</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <div style='text-align: right'>578,666</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <div style='text-align: right'>343,577</div>  |
| leaf_aggregation | AccessAdapter<8> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <div style='text-align: right'>229,296</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <div style='text-align: right'>6,107,075</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreE | STOREW | 0 | <div style='text-align: right'>1,002,368</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreE | STOREW | 0 | <div style='text-align: right'>18,524</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreE | STOREW | 0 | <div style='text-align: right'>10,946</div>  |
| leaf_aggregation | Boundary | true | StoreE | STOREW | 0 | <div style='text-align: right'>268,928</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>1,408,432</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>151,536</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>89,544</div>  |
| leaf_aggregation | Boundary | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>37,840</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreF | STOREW | 0 | <div style='text-align: right'>1,410,318</div>  |
| leaf_aggregation | Boundary | true | StoreF | STOREW | 0 | <div style='text-align: right'>378,378</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>3,396,727</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>352,352</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>209,846</div>  |
| leaf_aggregation | AccessAdapter<8> | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>141,678</div>  |
| leaf_aggregation | Boundary | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>77,484</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | StoreHintWord | ADD | 0 | <span style="color: green">(-10,080 [-0.2%])</span> <div style='text-align: right'>6,618,090</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-15,498 [-0.2%])</span> <div style='text-align: right'>10,033,274</div>  |
| leaf_aggregation | Boundary | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-4,158 [-0.2%])</span> <div style='text-align: right'>2,691,854</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreV | STOREW | 0 | <div style='text-align: right'>123,738</div>  |
| leaf_aggregation | Boundary | true | StoreV | STOREW | 0 | <div style='text-align: right'>33,198</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreV | STOREW2 | 0 | <span style="color: green">(-3,444 [-0.1%])</span> <div style='text-align: right'>2,555,694</div>  |
| leaf_aggregation | Boundary | true | StoreV | STOREW2 | 0 | <span style="color: green">(-924 [-0.1%])</span> <div style='text-align: right'>621,291</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | SubE | FE4SUB | 0 | <div style='text-align: right'>142,280</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubE | FE4SUB | 0 | <div style='text-align: right'>125,884</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubE | FE4SUB | 0 | <div style='text-align: right'>74,386</div>  |
| leaf_aggregation | Boundary | true | SubE | FE4SUB | 0 | <div style='text-align: right'>27,896</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | SubEF | LOADW | 0 | <div style='text-align: right'>663,462</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEF | LOADW | 0 | <div style='text-align: right'>59,213</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEF | SUB | 0 | <div style='text-align: right'>161,820</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEF | SUB | 0 | <div style='text-align: right'>59,213</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEF | SUB | 0 | <div style='text-align: right'>69,979</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEFI | ADD | 0 | <div style='text-align: right'>10,680</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEFI | ADD | 0 | <div style='text-align: right'>1,738</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEFI | ADD | 0 | <div style='text-align: right'>1,027</div>  |
| leaf_aggregation | Boundary | true | SubEFI | ADD | 0 | <div style='text-align: right'>220</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEI | ADD | 0 | <div style='text-align: right'>12,960</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEI | ADD | 0 | <div style='text-align: right'>3,432</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEI | ADD | 0 | <div style='text-align: right'>2,028</div>  |
| leaf_aggregation | Boundary | true | SubEI | ADD | 0 | <div style='text-align: right'>1,056</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubFI | SUB | 0 | <div style='text-align: right'>39,990</div>  |
| leaf_aggregation | Boundary | true | SubFI | SUB | 0 | <div style='text-align: right'>14,630</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubV | SUB | 0 | <span style="color: green">(-5,040 [-0.3%])</span> <div style='text-align: right'>1,516,380</div>  |
| leaf_aggregation | Boundary | true | SubV | SUB | 0 | <div style='text-align: right'>44</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubVI | SUB | 0 | <div style='text-align: right'>30,000</div>  |
| leaf_aggregation | Boundary | true | SubVI | SUB | 0 | <div style='text-align: right'>506</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubVIN | SUB | 0 | <div style='text-align: right'>25,200</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/fibonacci-2-2-64cpu-linux-x64-jemalloc-leaf_aggregation.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/axiom-crypto/afs-prototype/commit/e5471991fa80e66c6365fc7ebd0ed0061ee10820

Instance Type: 64cpu-linux-x64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12089310696)