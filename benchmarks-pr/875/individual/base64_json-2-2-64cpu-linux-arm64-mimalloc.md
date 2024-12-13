| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| base64_json_program | <div style='text-align: right'>2</div>  | <span style="color: green">(-131,072 [-0.9%])</span> <div style='text-align: right'>14,987,538</div>  | <div style='text-align: right'>217,352</div>  | <span style="color: green">(-145.0 [-5.5%])</span> <div style='text-align: right'>2,508.0</div>  |
| leaf_aggregation | <div style='text-align: right'>2</div>  | <span style="color: green">(-1,122,065 [-0.4%])</span> <div style='text-align: right'>293,297,336</div>  | <span style="color: green">(-27,319 [-0.4%])</span> <div style='text-align: right'>6,748,697</div>  | <span style="color: red">(+25.0 [+0.1%])</span> <div style='text-align: right'>35,365.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| group | collect_metrics | execute_time_ms | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- |
| base64_json_program | true | <span style="color: green">(-5.0 [-0.4%])</span> <div style='text-align: right'>1,390.0</div>  | <span style="color: green">(-131,072 [-0.9%])</span> <div style='text-align: right'>14,987,538</div>  | <div style='text-align: right'>217,352</div>  |

| group | chip_name | collect_metrics | rows_used |
| --- | --- | --- | --- |
| base64_json_program | ProgramChip | true | <div style='text-align: right'>19,414</div>  |
| base64_json_program | VmConnectorAir | true | <div style='text-align: right'>2</div>  |
| base64_json_program | Boundary | true | <div style='text-align: right'>5,180</div>  |
| base64_json_program | Merkle | true | <div style='text-align: right'>5,524</div>  |
| base64_json_program | AccessAdapter<8> | true | <div style='text-align: right'>5,180</div>  |
| base64_json_program | PhantomAir | true | <div style='text-align: right'>5</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true | <div style='text-align: right'>89,113</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | true | <div style='text-align: right'>575</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true | <div style='text-align: right'>16,188</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true | <div style='text-align: right'>55,121</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | true | <div style='text-align: right'>1,236</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true | <div style='text-align: right'>27,336</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true | <div style='text-align: right'>16,738</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true | <div style='text-align: right'>5,004</div>  |
| base64_json_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | true | <div style='text-align: right'>2,940</div>  |
| base64_json_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | true | <div style='text-align: right'>1,331</div>  |
| base64_json_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | true | <div style='text-align: right'>116</div>  |
| base64_json_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | true | <div style='text-align: right'>86</div>  |
| base64_json_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | true | <div style='text-align: right'>1,563</div>  |
| base64_json_program | Poseidon2VmAir<BabyBearParameters> | true | <div style='text-align: right'>10,704</div>  |
| base64_json_program | BitwiseOperationLookupAir<8> | true | <div style='text-align: right'>65,536</div>  |
| base64_json_program | RangeTupleCheckerAir<2> | true | <div style='text-align: right'>524,288</div>  |
| base64_json_program | VariableRangeCheckerAir | true | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | collect_metrics | dsl_ir | opcode | frequency |
| --- | --- | --- | --- | --- |
| base64_json_program | true |  | ADD | <div style='text-align: right'>69,777</div>  |
| base64_json_program | true |  | AND | <div style='text-align: right'>10,124</div>  |
| base64_json_program | true |  | AUIPC | <div style='text-align: right'>1,331</div>  |
| base64_json_program | true |  | BEQ | <div style='text-align: right'>15,568</div>  |
| base64_json_program | true |  | BGE | <div style='text-align: right'>704</div>  |
| base64_json_program | true |  | BGEU | <div style='text-align: right'>6,863</div>  |
| base64_json_program | true |  | BLT | <div style='text-align: right'>3,353</div>  |
| base64_json_program | true |  | BLTU | <div style='text-align: right'>5,818</div>  |
| base64_json_program | true |  | BNE | <div style='text-align: right'>11,768</div>  |
| base64_json_program | true |  | HINT_STOREW | <div style='text-align: right'>1,563</div>  |
| base64_json_program | true |  | JAL | <div style='text-align: right'>3,686</div>  |
| base64_json_program | true |  | JALR | <div style='text-align: right'>2,940</div>  |
| base64_json_program | true |  | LOADB | <div style='text-align: right'>1,236</div>  |
| base64_json_program | true |  | LOADBU | <div style='text-align: right'>23,858</div>  |
| base64_json_program | true |  | LOADHU | <div style='text-align: right'>3</div>  |
| base64_json_program | true |  | LOADW | <div style='text-align: right'>13,465</div>  |
| base64_json_program | true |  | LUI | <div style='text-align: right'>1,318</div>  |
| base64_json_program | true |  | MUL | <div style='text-align: right'>116</div>  |
| base64_json_program | true |  | MULHU | <div style='text-align: right'>86</div>  |
| base64_json_program | true |  | OR | <div style='text-align: right'>7,608</div>  |
| base64_json_program | true |  | PHANTOM | <div style='text-align: right'>5</div>  |
| base64_json_program | true |  | SLL | <div style='text-align: right'>7,118</div>  |
| base64_json_program | true |  | SLT | <div style='text-align: right'>5</div>  |
| base64_json_program | true |  | SLTU | <div style='text-align: right'>570</div>  |
| base64_json_program | true |  | SRA | <div style='text-align: right'>8</div>  |
| base64_json_program | true |  | SRL | <div style='text-align: right'>9,062</div>  |
| base64_json_program | true |  | STOREB | <div style='text-align: right'>5,133</div>  |
| base64_json_program | true |  | STOREH | <div style='text-align: right'>10</div>  |
| base64_json_program | true |  | STOREW | <div style='text-align: right'>12,652</div>  |
| base64_json_program | true |  | SUB | <div style='text-align: right'>1,416</div>  |
| base64_json_program | true |  | XOR | <div style='text-align: right'>188</div>  |

| group | air_name | collect_metrics | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | ADD | <div style='text-align: right'>2,511,972</div>  |
| base64_json_program | AccessAdapter<8> | true |  | ADD | <div style='text-align: right'>85</div>  |
| base64_json_program | Boundary | true |  | ADD | <div style='text-align: right'>200</div>  |
| base64_json_program | Merkle | true |  | ADD | <div style='text-align: right'>128</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | AND | <div style='text-align: right'>364,464</div>  |
| base64_json_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | true |  | AUIPC | <div style='text-align: right'>27,951</div>  |
| base64_json_program | AccessAdapter<8> | true |  | AUIPC | <div style='text-align: right'>51</div>  |
| base64_json_program | Boundary | true |  | AUIPC | <div style='text-align: right'>120</div>  |
| base64_json_program | Merkle | true |  | AUIPC | <div style='text-align: right'>3,520</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true |  | BEQ | <div style='text-align: right'>404,768</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BGE | <div style='text-align: right'>22,528</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BGEU | <div style='text-align: right'>219,616</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BLT | <div style='text-align: right'>107,296</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | true |  | BLTU | <div style='text-align: right'>186,176</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | true |  | BNE | <div style='text-align: right'>305,968</div>  |
| base64_json_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | true |  | HINT_STOREW | <div style='text-align: right'>40,638</div>  |
| base64_json_program | AccessAdapter<8> | true |  | HINT_STOREW | <div style='text-align: right'>13,277</div>  |
| base64_json_program | Boundary | true |  | HINT_STOREW | <div style='text-align: right'>31,240</div>  |
| base64_json_program | Merkle | true |  | HINT_STOREW | <div style='text-align: right'>49,920</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true |  | JAL | <div style='text-align: right'>66,348</div>  |
| base64_json_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | true |  | JALR | <div style='text-align: right'>82,320</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | true |  | LOADB | <div style='text-align: right'>43,260</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | LOADBU | <div style='text-align: right'>954,320</div>  |
| base64_json_program | AccessAdapter<8> | true |  | LOADBU | <div style='text-align: right'>2,873</div>  |
| base64_json_program | Boundary | true |  | LOADBU | <div style='text-align: right'>6,760</div>  |
| base64_json_program | Merkle | true |  | LOADBU | <div style='text-align: right'>12,096</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | LOADHU | <div style='text-align: right'>120</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | LOADW | <div style='text-align: right'>538,600</div>  |
| base64_json_program | AccessAdapter<8> | true |  | LOADW | <div style='text-align: right'>1,921</div>  |
| base64_json_program | Boundary | true |  | LOADW | <div style='text-align: right'>4,520</div>  |
| base64_json_program | Merkle | true |  | LOADW | <div style='text-align: right'>12,416</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | true |  | LUI | <div style='text-align: right'>23,724</div>  |
| base64_json_program | AccessAdapter<8> | true |  | LUI | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary | true |  | LUI | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | true |  | MUL | <div style='text-align: right'>3,596</div>  |
| base64_json_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | true |  | MULHU | <div style='text-align: right'>3,354</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | OR | <div style='text-align: right'>273,888</div>  |
| base64_json_program | PhantomAir | true |  | PHANTOM | <div style='text-align: right'>30</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true |  | SLL | <div style='text-align: right'>377,254</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | true |  | SLT | <div style='text-align: right'>185</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | true |  | SLTU | <div style='text-align: right'>21,090</div>  |
| base64_json_program | AccessAdapter<8> | true |  | SLTU | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary | true |  | SLTU | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true |  | SRA | <div style='text-align: right'>424</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | true |  | SRL | <div style='text-align: right'>480,286</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | STOREB | <div style='text-align: right'>205,320</div>  |
| base64_json_program | AccessAdapter<8> | true |  | STOREB | <div style='text-align: right'>10,472</div>  |
| base64_json_program | Boundary | true |  | STOREB | <div style='text-align: right'>24,640</div>  |
| base64_json_program | Merkle | true |  | STOREB | <div style='text-align: right'>39,488</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | STOREH | <div style='text-align: right'>400</div>  |
| base64_json_program | AccessAdapter<8> | true |  | STOREH | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary | true |  | STOREH | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | true |  | STOREW | <div style='text-align: right'>506,080</div>  |
| base64_json_program | AccessAdapter<8> | true |  | STOREW | <div style='text-align: right'>15,300</div>  |
| base64_json_program | Boundary | true |  | STOREW | <div style='text-align: right'>36,000</div>  |
| base64_json_program | Merkle | true |  | STOREW | <div style='text-align: right'>59,136</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | SUB | <div style='text-align: right'>50,976</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | true |  | XOR | <div style='text-align: right'>6,768</div>  |

| group | commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms | num_segments | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | <span style="color: red">(+1.0 [+7.7%])</span> <div style='text-align: right'>14.0</div>  | <span style="color: green">(-13.0 [-2.5%])</span> <div style='text-align: right'>500.0</div>  | <span style="color: green">(-6.0 [-1.8%])</span> <div style='text-align: right'>333.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-17.0 [-8.1%])</span> <div style='text-align: right'>192.0</div>  | <div style='text-align: right'>1</div>  | <span style="color: green">(-131,072 [-0.9%])</span> <div style='text-align: right'>14,987,538</div>  | <div style='text-align: right'>217,352</div>  | <span style="color: green">(-145.0 [-5.5%])</span> <div style='text-align: right'>2,508.0</div>  |
| leaf_aggregation |  |  |  | <div style='text-align: right'>2</div>  |  |  | <span style="color: green">(-1,122,065 [-0.4%])</span> <div style='text-align: right'>293,297,336</div>  | <span style="color: green">(-27,319 [-0.4%])</span> <div style='text-align: right'>6,748,697</div>  | <span style="color: red">(+25.0 [+0.1%])</span> <div style='text-align: right'>35,365.0</div>  |

| group | air_name | constraints | interactions | quotient_deg |
| --- | --- | --- | --- | --- |
| base64_json_program | ProgramAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| base64_json_program | VmConnectorAir | <div style='text-align: right'>9</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | PersistentBoundaryAir<8> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | MemoryMerkleAir<8> | <div style='text-align: right'>40</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<2> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<4> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<8> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<16> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<32> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | AccessAdapterAir<64> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | PhantomAir | <div style='text-align: right'>5</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | <div style='text-align: right'>43</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | <div style='text-align: right'>39</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | <div style='text-align: right'>90</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | <div style='text-align: right'>33</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | <div style='text-align: right'>25</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | <div style='text-align: right'>41</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | <div style='text-align: right'>22</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | <div style='text-align: right'>20</div>  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | <div style='text-align: right'>15</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | <div style='text-align: right'>26</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | <div style='text-align: right'>38</div>  | <div style='text-align: right'>24</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | <div style='text-align: right'>88</div>  | <div style='text-align: right'>25</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | <div style='text-align: right'>17</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | KeccakVmAir | <div style='text-align: right'>4,571</div>  | <div style='text-align: right'>321</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | Poseidon2VmAir<BabyBearParameters> | <div style='text-align: right'>525</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | BitwiseOperationLookupAir<8> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | RangeTupleCheckerAir<2> | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| base64_json_program | VariableRangeCheckerAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |

| group | air_name | segment | cells | constraints | interactions | main_cols | perm_cols | prep_cols | quotient_deg | rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | ProgramAir | 0 | <div style='text-align: right'>589,824</div>  |  |  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmConnectorAir | 0 | <div style='text-align: right'>32</div>  |  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>1</div>  |  | <div style='text-align: right'>2</div>  |
| base64_json_program | PersistentBoundaryAir<8> | 0 | <div style='text-align: right'>262,144</div>  |  |  | <div style='text-align: right'>20</div>  | <div style='text-align: right'>12</div>  |  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | MemoryMerkleAir<8> | 0 | <div style='text-align: right'>425,984</div>  |  |  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>20</div>  |  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | AccessAdapterAir<8> | 0 | <div style='text-align: right'>335,872</div>  |  |  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>24</div>  |  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | PhantomAir | 0 | <div style='text-align: right'>144</div>  |  |  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>12</div>  |  |  | <div style='text-align: right'>8</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | <div style='text-align: right'>15,204,352</div>  |  |  | <div style='text-align: right'>36</div>  | <div style='text-align: right'>80</div>  |  |  | <div style='text-align: right'>131,072</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>78,848</div>  |  |  | <div style='text-align: right'>37</div>  | <div style='text-align: right'>40</div>  |  |  | <div style='text-align: right'>1,024</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | <div style='text-align: right'>1,720,320</div>  |  |  | <div style='text-align: right'>53</div>  | <div style='text-align: right'>52</div>  |  |  | <div style='text-align: right'>16,384</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | <div style='text-align: right'>7,340,032</div>  |  |  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>72</div>  |  |  | <div style='text-align: right'>65,536</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | <div style='text-align: right'>227,328</div>  |  |  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>76</div>  |  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | <div style='text-align: right'>2,424,832</div>  |  |  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>48</div>  |  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>2,883,584</div>  |  |  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>56</div>  |  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>507,904</div>  |  |  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>44</div>  |  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | <div style='text-align: right'>262,144</div>  |  |  | <div style='text-align: right'>28</div>  | <div style='text-align: right'>36</div>  |  |  | <div style='text-align: right'>4,096</div>  |
| base64_json_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>100,352</div>  |  |  | <div style='text-align: right'>21</div>  | <div style='text-align: right'>28</div>  |  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | <div style='text-align: right'>14,208</div>  |  |  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>80</div>  |  |  | <div style='text-align: right'>128</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | <div style='text-align: right'>17,792</div>  |  |  | <div style='text-align: right'>39</div>  | <div style='text-align: right'>100</div>  |  |  | <div style='text-align: right'>128</div>  |
| base64_json_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>126,976</div>  |  |  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>36</div>  |  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | KeccakVmAir | 0 | <div style='text-align: right'>4,452</div>  |  |  | <div style='text-align: right'>3,164</div>  | <div style='text-align: right'>1,288</div>  |  |  | <div style='text-align: right'>1</div>  |
| base64_json_program | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>10,272,768</div>  |  |  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>68</div>  |  |  | <div style='text-align: right'>16,384</div>  |
| base64_json_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>655,360</div>  |  |  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  |  | <div style='text-align: right'>65,536</div>  |
| base64_json_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>4,718,592</div>  |  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  |  | <div style='text-align: right'>524,288</div>  |
| base64_json_program | VariableRangeCheckerAir | 0 | <span style="color: green">(-1,179,648 [-50.0%])</span> <div style='text-align: right'>1,179,648</div>  |  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  |  | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | ProgramAir | 0 | <div style='text-align: right'>9,437,184</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>524,288</div>  |
| leaf_aggregation | VmConnectorAir | 0 | <div style='text-align: right'>24</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| leaf_aggregation | VolatileBoundaryAir | 0 | <div style='text-align: right'>19,922,944</div>  | <div style='text-align: right'>16</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1,048,576</div>  |
| leaf_aggregation | AccessAdapterAir<2> | 0 | <div style='text-align: right'>56,623,104</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2,097,152</div>  |
| leaf_aggregation | AccessAdapterAir<4> | 0 | <div style='text-align: right'>30,408,704</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1,048,576</div>  |
| leaf_aggregation | AccessAdapterAir<8> | 0 | <div style='text-align: right'>4,325,376</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | PhantomAir | 0 | <div style='text-align: right'>14,680,064</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1,048,576</div>  |
| leaf_aggregation | VmAirWrapper<NativeLoadStoreAdapterAir<1>, KernelLoadStoreCoreAir<1> | 0 | <div style='text-align: right'>136,314,880</div>  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>19</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2,097,152</div>  |
| leaf_aggregation | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | <div style='text-align: right'>106,954,752</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>2,097,152</div>  |
| leaf_aggregation | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | <div style='text-align: right'>2,883,584</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>7</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>209,715,200</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>4,194,304</div>  |
| leaf_aggregation | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | <div style='text-align: right'>7,864,320</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>15</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>131,072</div>  |
| leaf_aggregation | FriReducedOpeningAir | 0 | <div style='text-align: right'>146,800,640</div>  | <div style='text-align: right'>59</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1,048,576</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>38,993,920</div>  | <div style='text-align: right'>517</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>65,536</div>  |
| leaf_aggregation | VariableRangeCheckerAir | 0 | <span style="color: green">(-1,179,648 [-50.0%])</span> <div style='text-align: right'>1,179,648</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | segment | commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms | num_segments | stark_prove_excluding_trace_time_ms | total_cells | verify_program_compile_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | 0 |  | <span style="color: green">(-7.0 [-4.1%])</span> <div style='text-align: right'>165.0</div>  |  |  |  |  | <span style="color: green">(-125.0 [-6.4%])</span> <div style='text-align: right'>1,843.0</div>  | <span style="color: green">(-1,179,648 [-2.3%])</span> <div style='text-align: right'>49,353,492</div>  |  |
| leaf_aggregation | 0 | <span style="color: green">(-6.0 [-5.3%])</span> <div style='text-align: right'>107.0</div>  | <span style="color: green">(-90.0 [-1.0%])</span> <div style='text-align: right'>9,382.0</div>  | <span style="color: green">(-83.0 [-1.1%])</span> <div style='text-align: right'>7,693.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-18.0 [-23.1%])</span> <div style='text-align: right'>60.0</div>  | <div style='text-align: right'>1</div>  | <span style="color: red">(+115.0 [+0.4%])</span> <div style='text-align: right'>25,983.0</div>  | <span style="color: green">(-1,179,648 [-0.1%])</span> <div style='text-align: right'>786,104,344</div>  | <span style="color: green">(-23.0 [-0.0%])</span> <div style='text-align: right'>73,031.0</div>  |

| group | collect_metrics | segment | execute_time_ms | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- | --- |
| leaf_aggregation | true | 0 | <span style="color: red">(+18.0 [+0.0%])</span> <div style='text-align: right'>36,592.0</div>  | <span style="color: green">(-1,122,065 [-0.4%])</span> <div style='text-align: right'>293,297,336</div>  | <span style="color: green">(-27,319 [-0.4%])</span> <div style='text-align: right'>6,748,697</div>  |

| group | chip_name | collect_metrics | segment | rows_used |
| --- | --- | --- | --- | --- |
| leaf_aggregation | ProgramChip | true | 0 | <span style="color: green">(-20 [-0.0%])</span> <div style='text-align: right'>305,302</div>  |
| leaf_aggregation | VmConnectorAir | true | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | Boundary | true | 0 | <span style="color: green">(-2,478 [-0.2%])</span> <div style='text-align: right'>1,009,881</div>  |
| leaf_aggregation | AccessAdapter<2> | true | 0 | <span style="color: green">(-1,844 [-0.2%])</span> <div style='text-align: right'>1,059,820</div>  |
| leaf_aggregation | AccessAdapter<4> | true | 0 | <span style="color: green">(-880 [-0.2%])</span> <div style='text-align: right'>530,162</div>  |
| leaf_aggregation | AccessAdapter<8> | true | 0 | <span style="color: green">(-420 [-0.4%])</span> <div style='text-align: right'>107,366</div>  |
| leaf_aggregation | PhantomAir | true | 0 | <span style="color: green">(-798 [-0.1%])</span> <div style='text-align: right'>620,280</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | 0 | <span style="color: green">(-9,470 [-0.5%])</span> <div style='text-align: right'>1,913,247</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | 0 | <span style="color: green">(-8,571 [-0.6%])</span> <div style='text-align: right'>1,354,088</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | 0 | <span style="color: green">(-1,544 [-1.8%])</span> <div style='text-align: right'>83,080</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | 0 | <span style="color: green">(-6,721 [-0.3%])</span> <div style='text-align: right'>2,612,508</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | 0 | <span style="color: green">(-5 [-0.0%])</span> <div style='text-align: right'>106,604</div>  |
| leaf_aggregation | FriReducedOpeningAir | true | 0 | <div style='text-align: right'>550,368</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | 0 | <span style="color: green">(-210 [-0.4%])</span> <div style='text-align: right'>52,548</div>  |
| leaf_aggregation | VariableRangeCheckerAir | true | 0 | <span style="color: green">(-131,072 [-50.0%])</span> <div style='text-align: right'>131,072</div>  |

| group | collect_metrics | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- | --- |
| leaf_aggregation | true |  | JAL | 0 | <div style='text-align: right'>1</div>  |
| leaf_aggregation | true |  | STOREW | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | AddE | FE4ADD | 0 | <div style='text-align: right'>46,074</div>  |
| leaf_aggregation | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>180</div>  |
| leaf_aggregation | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>540</div>  |
| leaf_aggregation | true | AddEFI | ADD | 0 | <div style='text-align: right'>528</div>  |
| leaf_aggregation | true | AddEI | ADD | 0 | <span style="color: red">(+4 [+0.0%])</span> <div style='text-align: right'>91,016</div>  |
| leaf_aggregation | true | AddF | ADD | 0 | <div style='text-align: right'>1,333</div>  |
| leaf_aggregation | true | AddFI | ADD | 0 | <span style="color: green">(-168 [-0.3%])</span> <div style='text-align: right'>61,024</div>  |
| leaf_aggregation | true | AddV | ADD | 0 | <span style="color: green">(-2 [-0.0%])</span> <div style='text-align: right'>14,758</div>  |
| leaf_aggregation | true | AddVI | ADD | 0 | <span style="color: green">(-1,050 [-0.1%])</span> <div style='text-align: right'>774,124</div>  |
| leaf_aggregation | true | Alloc | ADD | 0 | <span style="color: green">(-210 [-0.4%])</span> <div style='text-align: right'>55,480</div>  |
| leaf_aggregation | true | Alloc | LOADW | 0 | <span style="color: green">(-210 [-0.4%])</span> <div style='text-align: right'>55,480</div>  |
| leaf_aggregation | true | Alloc | MUL | 0 | <span style="color: green">(-168 [-0.5%])</span> <div style='text-align: right'>33,005</div>  |
| leaf_aggregation | true | AssertEqE | BNE | 0 | <div style='text-align: right'>264</div>  |
| leaf_aggregation | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>4</div>  |
| leaf_aggregation | true | AssertEqF | BNE | 0 | <div style='text-align: right'>10,448</div>  |
| leaf_aggregation | true | AssertEqV | BNE | 0 | <div style='text-align: right'>1,075</div>  |
| leaf_aggregation | true | AssertEqVI | BNE | 0 | <div style='text-align: right'>281</div>  |
| leaf_aggregation | true | CT-InitializePcsConst | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-ReadingProofFromInput | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-VerifierProgram | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-compute-reduced-opening | PHANTOM | 0 | <div style='text-align: right'>672</div>  |
| leaf_aggregation | true | CT-exp-reverse-bits-len | PHANTOM | 0 | <div style='text-align: right'>8,232</div>  |
| leaf_aggregation | true | CT-poseidon2-hash | PHANTOM | 0 | <span style="color: green">(-252 [-6.4%])</span> <div style='text-align: right'>3,696</div>  |
| leaf_aggregation | true | CT-poseidon2-hash-ext | PHANTOM | 0 | <div style='text-align: right'>1,596</div>  |
| leaf_aggregation | true | CT-poseidon2-hash-setup | PHANTOM | 0 | <div style='text-align: right'>557,928</div>  |
| leaf_aggregation | true | CT-single-reduced-opening-eval | PHANTOM | 0 | <div style='text-align: right'>12,684</div>  |
| leaf_aggregation | true | CT-stage-c-build-rounds | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-1-verify-shape-and-sample-challenges | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-2-fri-fold | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-3-verify-challenges | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-d-verify-pcs | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-stage-e-verify-constraints | PHANTOM | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | CT-verify-batch | PHANTOM | 0 | <div style='text-align: right'>672</div>  |
| leaf_aggregation | true | CT-verify-batch-ext | PHANTOM | 0 | <div style='text-align: right'>1,596</div>  |
| leaf_aggregation | true | CT-verify-batch-reduce-fast | PHANTOM | 0 | <span style="color: green">(-252 [-4.5%])</span> <div style='text-align: right'>5,292</div>  |
| leaf_aggregation | true | CT-verify-batch-reduce-fast-setup | PHANTOM | 0 | <span style="color: green">(-252 [-4.5%])</span> <div style='text-align: right'>5,292</div>  |
| leaf_aggregation | true | CT-verify-query | PHANTOM | 0 | <div style='text-align: right'>84</div>  |
| leaf_aggregation | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>7,188</div>  |
| leaf_aggregation | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>66</div>  |
| leaf_aggregation | true | DivEIN | STOREW | 0 | <div style='text-align: right'>264</div>  |
| leaf_aggregation | true | DivFIN | DIV | 0 | <div style='text-align: right'>156</div>  |
| leaf_aggregation | true | For | ADD | 0 | <span style="color: green">(-4,245 [-0.5%])</span> <div style='text-align: right'>886,650</div>  |
| leaf_aggregation | true | For | BNE | 0 | <span style="color: green">(-4,539 [-0.5%])</span> <div style='text-align: right'>932,317</div>  |
| leaf_aggregation | true | For | JAL | 0 | <span style="color: green">(-294 [-0.6%])</span> <div style='text-align: right'>45,667</div>  |
| leaf_aggregation | true | For | LOADW | 0 | <span style="color: green">(-126 [-4.5%])</span> <div style='text-align: right'>2,688</div>  |
| leaf_aggregation | true | For | STOREW | 0 | <span style="color: green">(-168 [-0.4%])</span> <div style='text-align: right'>42,979</div>  |
| leaf_aggregation | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>6,342</div>  |
| leaf_aggregation | true | HintBitsF | PHANTOM | 0 | <div style='text-align: right'>43</div>  |
| leaf_aggregation | true | HintInputVec | PHANTOM | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>22,475</div>  |
| leaf_aggregation | true | IfEq | BNE | 0 | <span style="color: green">(-3,654 [-11.2%])</span> <div style='text-align: right'>28,949</div>  |
| leaf_aggregation | true | IfEqI | BNE | 0 | <span style="color: green">(-210 [-0.1%])</span> <div style='text-align: right'>363,144</div>  |
| leaf_aggregation | true | IfEqI | JAL | 0 | <span style="color: green">(-1,250 [-3.2%])</span> <div style='text-align: right'>37,410</div>  |
| leaf_aggregation | true | IfNe | BEQ | 0 | <span style="color: green">(-42 [-0.3%])</span> <div style='text-align: right'>14,839</div>  |
| leaf_aggregation | true | IfNe | JAL | 0 | <div style='text-align: right'>2</div>  |
| leaf_aggregation | true | IfNeI | BEQ | 0 | <span style="color: green">(-126 [-4.4%])</span> <div style='text-align: right'>2,767</div>  |
| leaf_aggregation | true | ImmE | STOREW | 0 | <span style="color: green">(-20 [-0.1%])</span> <div style='text-align: right'>16,176</div>  |
| leaf_aggregation | true | ImmF | STOREW | 0 | <span style="color: green">(-2,016 [-4.2%])</span> <div style='text-align: right'>45,741</div>  |
| leaf_aggregation | true | ImmV | STOREW | 0 | <span style="color: green">(-504 [-0.9%])</span> <div style='text-align: right'>54,791</div>  |
| leaf_aggregation | true | LoadE | LOADW | 0 | <div style='text-align: right'>62,196</div>  |
| leaf_aggregation | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>74,352</div>  |
| leaf_aggregation | true | LoadF | LOADW | 0 | <div style='text-align: right'>26,896</div>  |
| leaf_aggregation | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>300,903</div>  |
| leaf_aggregation | true | LoadV | LOADW | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>25,515</div>  |
| leaf_aggregation | true | LoadV | LOADW2 | 0 | <span style="color: green">(-3,906 [-1.6%])</span> <div style='text-align: right'>233,588</div>  |
| leaf_aggregation | true | MulE | BBE4MUL | 0 | <span style="color: green">(-5 [-0.0%])</span> <div style='text-align: right'>31,902</div>  |
| leaf_aggregation | true | MulEF | MUL | 0 | <div style='text-align: right'>3,720</div>  |
| leaf_aggregation | true | MulEFI | MUL | 0 | <div style='text-align: right'>7,976</div>  |
| leaf_aggregation | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>4,774</div>  |
| leaf_aggregation | true | MulEI | STOREW | 0 | <div style='text-align: right'>19,096</div>  |
| leaf_aggregation | true | MulF | MUL | 0 | <span style="color: green">(-336 [-0.3%])</span> <div style='text-align: right'>118,651</div>  |
| leaf_aggregation | true | MulFI | MUL | 0 | <div style='text-align: right'>1,357</div>  |
| leaf_aggregation | true | MulVI | MUL | 0 | <span style="color: green">(-42 [-0.2%])</span> <div style='text-align: right'>19,790</div>  |
| leaf_aggregation | true | NegE | MUL | 0 | <div style='text-align: right'>436</div>  |
| leaf_aggregation | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-168 [-1.0%])</span> <div style='text-align: right'>16,212</div>  |
| leaf_aggregation | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <span style="color: green">(-42 [-0.1%])</span> <div style='text-align: right'>36,336</div>  |
| leaf_aggregation | true | StoreE | STOREW | 0 | <div style='text-align: right'>23,824</div>  |
| leaf_aggregation | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>39,388</div>  |
| leaf_aggregation | true | StoreF | STOREW | 0 | <span style="color: green">(-2,016 [-5.3%])</span> <div style='text-align: right'>36,078</div>  |
| leaf_aggregation | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>286,073</div>  |
| leaf_aggregation | true | StoreHintWord | ADD | 0 | <span style="color: green">(-336 [-0.1%])</span> <div style='text-align: right'>455,137</div>  |
| leaf_aggregation | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-378 [-0.1%])</span> <div style='text-align: right'>478,945</div>  |
| leaf_aggregation | true | StoreV | STOREW | 0 | <div style='text-align: right'>2,952</div>  |
| leaf_aggregation | true | StoreV | STOREW2 | 0 | <span style="color: green">(-84 [-0.1%])</span> <div style='text-align: right'>65,370</div>  |
| leaf_aggregation | true | SubE | FE4SUB | 0 | <div style='text-align: right'>16,600</div>  |
| leaf_aggregation | true | SubEF | LOADW | 0 | <div style='text-align: right'>19,230</div>  |
| leaf_aggregation | true | SubEF | SUB | 0 | <div style='text-align: right'>6,410</div>  |
| leaf_aggregation | true | SubEFI | ADD | 0 | <div style='text-align: right'>9,444</div>  |
| leaf_aggregation | true | SubEI | ADD | 0 | <div style='text-align: right'>528</div>  |
| leaf_aggregation | true | SubFI | SUB | 0 | <div style='text-align: right'>1,333</div>  |
| leaf_aggregation | true | SubV | SUB | 0 | <span style="color: green">(-168 [-0.2%])</span> <div style='text-align: right'>67,896</div>  |
| leaf_aggregation | true | SubVI | SUB | 0 | <div style='text-align: right'>958</div>  |
| leaf_aggregation | true | SubVIN | SUB | 0 | <div style='text-align: right'>798</div>  |

| group | air_name | collect_metrics | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- | --- |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true |  | JAL | 0 | <div style='text-align: right'>10</div>  |
| leaf_aggregation | Boundary | true |  | JAL | 0 | <div style='text-align: right'>11</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true |  | STOREW | 0 | <div style='text-align: right'>82</div>  |
| leaf_aggregation | Boundary | true |  | STOREW | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | AddE | FE4ADD | 0 | <div style='text-align: right'>1,842,960</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddE | FE4ADD | 0 | <span style="color: red">(+66 [+0.0%])</span> <div style='text-align: right'>1,321,914</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddE | FE4ADD | 0 | <span style="color: red">(+39 [+0.0%])</span> <div style='text-align: right'>781,131</div>  |
| leaf_aggregation | Boundary | true | AddE | FE4ADD | 0 | <div style='text-align: right'>1,385,648</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>7,380</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFFI | LOADW | 0 | <span style="color: green">(-22 [-1.9%])</span> <div style='text-align: right'>1,111</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEFFI | LOADW | 0 | <span style="color: green">(-26 [-1.9%])</span> <div style='text-align: right'>1,313</div>  |
| leaf_aggregation | Boundary | true | AddEFFI | LOADW | 0 | <div style='text-align: right'>308</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>22,140</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFFI | STOREW | 0 | <span style="color: green">(-22 [-1.9%])</span> <div style='text-align: right'>1,111</div>  |
| leaf_aggregation | Boundary | true | AddEFFI | STOREW | 0 | <div style='text-align: right'>924</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddEFI | ADD | 0 | <div style='text-align: right'>15,840</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEFI | ADD | 0 | <span style="color: green">(-22 [-1.0%])</span> <div style='text-align: right'>2,266</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEFI | ADD | 0 | <span style="color: green">(-13 [-1.0%])</span> <div style='text-align: right'>1,339</div>  |
| leaf_aggregation | Boundary | true | AddEFI | ADD | 0 | <div style='text-align: right'>2,112</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddEI | ADD | 0 | <span style="color: red">(+120 [+0.0%])</span> <div style='text-align: right'>2,730,480</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AddEI | ADD | 0 | <span style="color: green">(-396 [-0.1%])</span> <div style='text-align: right'>574,992</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AddEI | ADD | 0 | <span style="color: green">(-234 [-0.1%])</span> <div style='text-align: right'>339,768</div>  |
| leaf_aggregation | Boundary | true | AddEI | ADD | 0 | <span style="color: red">(+44 [+0.0%])</span> <div style='text-align: right'>665,368</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddF | ADD | 0 | <div style='text-align: right'>39,990</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddFI | ADD | 0 | <span style="color: green">(-5,040 [-0.3%])</span> <div style='text-align: right'>1,830,720</div>  |
| leaf_aggregation | Boundary | true | AddFI | ADD | 0 | <div style='text-align: right'>253</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddV | ADD | 0 | <span style="color: green">(-60 [-0.0%])</span> <div style='text-align: right'>442,740</div>  |
| leaf_aggregation | Boundary | true | AddV | ADD | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | AddVI | ADD | 0 | <span style="color: green">(-31,500 [-0.1%])</span> <div style='text-align: right'>23,223,720</div>  |
| leaf_aggregation | Boundary | true | AddVI | ADD | 0 | <div style='text-align: right'>1,276</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | Alloc | ADD | 0 | <span style="color: green">(-6,300 [-0.4%])</span> <div style='text-align: right'>1,664,400</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | Alloc | LOADW | 0 | <span style="color: green">(-8,610 [-0.4%])</span> <div style='text-align: right'>2,274,680</div>  |
| leaf_aggregation | Boundary | true | Alloc | LOADW | 0 | <div style='text-align: right'>1,122</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | Alloc | MUL | 0 | <span style="color: green">(-5,040 [-0.5%])</span> <div style='text-align: right'>990,150</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Alloc | MUL | 0 | <div style='text-align: right'>33</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Alloc | MUL | 0 | <div style='text-align: right'>39</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>6,072</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>1,452</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AssertEqE | BNE | 0 | <div style='text-align: right'>858</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>92</div>  |
| leaf_aggregation | AccessAdapter<2> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>22</div>  |
| leaf_aggregation | AccessAdapter<4> | true | AssertEqEI | BNE | 0 | <div style='text-align: right'>13</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqF | BNE | 0 | <div style='text-align: right'>240,304</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqV | BNE | 0 | <div style='text-align: right'>24,725</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | AssertEqVI | BNE | 0 | <div style='text-align: right'>6,463</div>  |
| leaf_aggregation | PhantomAir | true | CT-InitializePcsConst | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-ReadingProofFromInput | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-VerifierProgram | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-compute-reduced-opening | PHANTOM | 0 | <div style='text-align: right'>4,032</div>  |
| leaf_aggregation | PhantomAir | true | CT-exp-reverse-bits-len | PHANTOM | 0 | <div style='text-align: right'>49,392</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash | PHANTOM | 0 | <span style="color: green">(-1,512 [-6.4%])</span> <div style='text-align: right'>22,176</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash-ext | PHANTOM | 0 | <div style='text-align: right'>9,576</div>  |
| leaf_aggregation | PhantomAir | true | CT-poseidon2-hash-setup | PHANTOM | 0 | <div style='text-align: right'>3,347,568</div>  |
| leaf_aggregation | PhantomAir | true | CT-single-reduced-opening-eval | PHANTOM | 0 | <div style='text-align: right'>76,104</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-c-build-rounds | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-1-verify-shape-and-sample-challenges | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-2-fri-fold | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-3-verify-challenges | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-d-verify-pcs | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-stage-e-verify-constraints | PHANTOM | 0 | <div style='text-align: right'>12</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch | PHANTOM | 0 | <div style='text-align: right'>4,032</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-ext | PHANTOM | 0 | <div style='text-align: right'>9,576</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-reduce-fast | PHANTOM | 0 | <span style="color: green">(-1,512 [-4.5%])</span> <div style='text-align: right'>31,752</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-batch-reduce-fast-setup | PHANTOM | 0 | <span style="color: green">(-1,512 [-4.5%])</span> <div style='text-align: right'>31,752</div>  |
| leaf_aggregation | PhantomAir | true | CT-verify-query | PHANTOM | 0 | <div style='text-align: right'>504</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>287,520</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>141,130</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivE | BBE4DIV | 0 | <div style='text-align: right'>83,395</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>2,640</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivEIN | BBE4DIV | 0 | <span style="color: green">(-88 [-3.1%])</span> <div style='text-align: right'>2,794</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivEIN | BBE4DIV | 0 | <span style="color: green">(-52 [-3.1%])</span> <div style='text-align: right'>1,651</div>  |
| leaf_aggregation | Boundary | true | DivEIN | BBE4DIV | 0 | <div style='text-align: right'>528</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>10,824</div>  |
| leaf_aggregation | AccessAdapter<2> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>957</div>  |
| leaf_aggregation | AccessAdapter<4> | true | DivEIN | STOREW | 0 | <div style='text-align: right'>273</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | DivFIN | DIV | 0 | <div style='text-align: right'>4,680</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | For | ADD | 0 | <span style="color: green">(-127,350 [-0.5%])</span> <div style='text-align: right'>26,599,500</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | For | BNE | 0 | <span style="color: green">(-104,397 [-0.5%])</span> <div style='text-align: right'>21,443,291</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | For | JAL | 0 | <span style="color: green">(-2,940 [-0.6%])</span> <div style='text-align: right'>456,670</div>  |
| leaf_aggregation | AccessAdapter<2> | true | For | JAL | 0 | <div style='text-align: right'>495</div>  |
| leaf_aggregation | AccessAdapter<4> | true | For | JAL | 0 | <div style='text-align: right'>585</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | For | LOADW | 0 | <span style="color: green">(-5,166 [-4.5%])</span> <div style='text-align: right'>110,208</div>  |
| leaf_aggregation | Boundary | true | For | LOADW | 0 | <div style='text-align: right'>473</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | For | STOREW | 0 | <span style="color: green">(-6,888 [-0.4%])</span> <div style='text-align: right'>1,762,139</div>  |
| leaf_aggregation | Boundary | true | For | STOREW | 0 | <div style='text-align: right'>814</div>  |
| leaf_aggregation | AccessAdapter<2> | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>378,840</div>  |
| leaf_aggregation | AccessAdapter<4> | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>223,860</div>  |
| leaf_aggregation | FriReducedOpeningAir | true | FriReducedOpening | FRI_REDUCED_OPENING | 0 | <div style='text-align: right'>35,223,552</div>  |
| leaf_aggregation | PhantomAir | true | HintBitsF | PHANTOM | 0 | <div style='text-align: right'>258</div>  |
| leaf_aggregation | PhantomAir | true | HintInputVec | PHANTOM | 0 | <span style="color: green">(-252 [-0.2%])</span> <div style='text-align: right'>134,850</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfEq | BNE | 0 | <span style="color: green">(-84,042 [-11.2%])</span> <div style='text-align: right'>665,827</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfEqI | BNE | 0 | <span style="color: green">(-4,830 [-0.1%])</span> <div style='text-align: right'>8,352,312</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | IfEqI | JAL | 0 | <span style="color: green">(-12,500 [-3.2%])</span> <div style='text-align: right'>374,100</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfNe | BEQ | 0 | <span style="color: green">(-966 [-0.3%])</span> <div style='text-align: right'>341,297</div>  |
| leaf_aggregation | <JalNativeAdapterAir,JalCoreAir> | true | IfNe | JAL | 0 | <div style='text-align: right'>20</div>  |
| leaf_aggregation | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | true | IfNeI | BEQ | 0 | <span style="color: green">(-2,898 [-4.4%])</span> <div style='text-align: right'>63,641</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmE | STOREW | 0 | <span style="color: green">(-820 [-0.1%])</span> <div style='text-align: right'>663,216</div>  |
| leaf_aggregation | AccessAdapter<2> | true | ImmE | STOREW | 0 | <span style="color: green">(-22 [-0.2%])</span> <div style='text-align: right'>14,102</div>  |
| leaf_aggregation | AccessAdapter<4> | true | ImmE | STOREW | 0 | <span style="color: green">(-13 [-0.2%])</span> <div style='text-align: right'>8,333</div>  |
| leaf_aggregation | Boundary | true | ImmE | STOREW | 0 | <span style="color: green">(-44 [-0.0%])</span> <div style='text-align: right'>134,728</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmF | STOREW | 0 | <span style="color: green">(-82,656 [-4.2%])</span> <div style='text-align: right'>1,875,381</div>  |
| leaf_aggregation | Boundary | true | ImmF | STOREW | 0 | <div style='text-align: right'>16,291</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | ImmV | STOREW | 0 | <span style="color: green">(-20,664 [-0.9%])</span> <div style='text-align: right'>2,246,431</div>  |
| leaf_aggregation | Boundary | true | ImmV | STOREW | 0 | <div style='text-align: right'>1,694</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadE | LOADW | 0 | <div style='text-align: right'>2,550,036</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadE | LOADW | 0 | <span style="color: red">(+22 [+0.0%])</span> <div style='text-align: right'>427,152</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadE | LOADW | 0 | <span style="color: red">(+13 [+0.0%])</span> <div style='text-align: right'>252,408</div>  |
| leaf_aggregation | Boundary | true | LoadE | LOADW | 0 | <div style='text-align: right'>302,808</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>3,048,432</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>53,636</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>31,694</div>  |
| leaf_aggregation | Boundary | true | LoadE | LOADW2 | 0 | <div style='text-align: right'>44</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadF | LOADW | 0 | <div style='text-align: right'>1,102,736</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadF | LOADW | 0 | <div style='text-align: right'>49,896</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadF | LOADW | 0 | <div style='text-align: right'>29,484</div>  |
| leaf_aggregation | AccessAdapter<8> | true | LoadF | LOADW | 0 | <div style='text-align: right'>19,278</div>  |
| leaf_aggregation | Boundary | true | LoadF | LOADW | 0 | <div style='text-align: right'>14,905</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>12,337,023</div>  |
| leaf_aggregation | AccessAdapter<2> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>770</div>  |
| leaf_aggregation | AccessAdapter<4> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>455</div>  |
| leaf_aggregation | AccessAdapter<8> | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>476</div>  |
| leaf_aggregation | Boundary | true | LoadF | LOADW2 | 0 | <div style='text-align: right'>638</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadV | LOADW | 0 | <span style="color: green">(-1,722 [-0.2%])</span> <div style='text-align: right'>1,046,115</div>  |
| leaf_aggregation | Boundary | true | LoadV | LOADW | 0 | <div style='text-align: right'>396</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | LoadV | LOADW2 | 0 | <span style="color: green">(-160,146 [-1.6%])</span> <div style='text-align: right'>9,577,108</div>  |
| leaf_aggregation | Boundary | true | LoadV | LOADW2 | 0 | <div style='text-align: right'>979</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | MulE | BBE4MUL | 0 | <span style="color: green">(-200 [-0.0%])</span> <div style='text-align: right'>1,276,080</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulE | BBE4MUL | 0 | <span style="color: green">(-440 [-0.1%])</span> <div style='text-align: right'>827,420</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulE | BBE4MUL | 0 | <span style="color: green">(-260 [-0.1%])</span> <div style='text-align: right'>488,930</div>  |
| leaf_aggregation | Boundary | true | MulE | BBE4MUL | 0 | <div style='text-align: right'>938,476</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulEF | MUL | 0 | <div style='text-align: right'>111,600</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEF | MUL | 0 | <span style="color: red">(+22 [+0.1%])</span> <div style='text-align: right'>18,524</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEF | MUL | 0 | <span style="color: red">(+13 [+0.1%])</span> <div style='text-align: right'>10,946</div>  |
| leaf_aggregation | Boundary | true | MulEF | MUL | 0 | <div style='text-align: right'>1,056</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulEFI | MUL | 0 | <div style='text-align: right'>239,280</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEFI | MUL | 0 | <div style='text-align: right'>4,840</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEFI | MUL | 0 | <div style='text-align: right'>2,860</div>  |
| leaf_aggregation | Boundary | true | MulEFI | MUL | 0 | <div style='text-align: right'>81,092</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>190,960</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEI | BBE4MUL | 0 | <span style="color: red">(+66 [+0.0%])</span> <div style='text-align: right'>271,458</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEI | BBE4MUL | 0 | <span style="color: red">(+39 [+0.0%])</span> <div style='text-align: right'>160,407</div>  |
| leaf_aggregation | Boundary | true | MulEI | BBE4MUL | 0 | <div style='text-align: right'>117,656</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | MulEI | STOREW | 0 | <div style='text-align: right'>782,936</div>  |
| leaf_aggregation | AccessAdapter<2> | true | MulEI | STOREW | 0 | <div style='text-align: right'>104,973</div>  |
| leaf_aggregation | AccessAdapter<4> | true | MulEI | STOREW | 0 | <div style='text-align: right'>62,010</div>  |
| leaf_aggregation | Boundary | true | MulEI | STOREW | 0 | <div style='text-align: right'>33</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulF | MUL | 0 | <span style="color: green">(-10,080 [-0.3%])</span> <div style='text-align: right'>3,559,530</div>  |
| leaf_aggregation | Boundary | true | MulF | MUL | 0 | <div style='text-align: right'>14,630</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulFI | MUL | 0 | <div style='text-align: right'>40,710</div>  |
| leaf_aggregation | Boundary | true | MulFI | MUL | 0 | <div style='text-align: right'>14,641</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | MulVI | MUL | 0 | <span style="color: green">(-1,260 [-0.2%])</span> <div style='text-align: right'>593,700</div>  |
| leaf_aggregation | Boundary | true | MulVI | MUL | 0 | <div style='text-align: right'>77</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | NegE | MUL | 0 | <div style='text-align: right'>13,080</div>  |
| leaf_aggregation | AccessAdapter<2> | true | NegE | MUL | 0 | <div style='text-align: right'>3,168</div>  |
| leaf_aggregation | AccessAdapter<4> | true | NegE | MUL | 0 | <div style='text-align: right'>1,872</div>  |
| leaf_aggregation | Boundary | true | NegE | MUL | 0 | <div style='text-align: right'>2,420</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-1,848 [-0.3%])</span> <div style='text-align: right'>646,800</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-1,092 [-0.3%])</span> <div style='text-align: right'>382,200</div>  |
| leaf_aggregation | AccessAdapter<8> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-714 [-0.3%])</span> <div style='text-align: right'>249,900</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | Poseidon2CompressBabyBear | COMP_POS2 | 0 | <span style="color: green">(-93,912 [-1.0%])</span> <div style='text-align: right'>9,062,508</div>  |
| leaf_aggregation | AccessAdapter<2> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <span style="color: green">(-7,854 [-0.5%])</span> <div style='text-align: right'>1,698,598</div>  |
| leaf_aggregation | AccessAdapter<4> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <span style="color: green">(-4,368 [-0.4%])</span> <div style='text-align: right'>1,005,355</div>  |
| leaf_aggregation | AccessAdapter<8> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <span style="color: green">(-2,856 [-0.4%])</span> <div style='text-align: right'>662,711</div>  |
| leaf_aggregation | Poseidon2VmAir<BabyBearParameters> | true | Poseidon2PermuteBabyBear | PERM_POS2 | 0 | <span style="color: green">(-23,478 [-0.1%])</span> <div style='text-align: right'>20,311,824</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreE | STOREW | 0 | <div style='text-align: right'>976,784</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreE | STOREW | 0 | <div style='text-align: right'>17,600</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreE | STOREW | 0 | <div style='text-align: right'>10,400</div>  |
| leaf_aggregation | Boundary | true | StoreE | STOREW | 0 | <div style='text-align: right'>262,064</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>1,614,908</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>181,104</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>107,016</div>  |
| leaf_aggregation | Boundary | true | StoreE | STOREW2 | 0 | <div style='text-align: right'>35,948</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreF | STOREW | 0 | <span style="color: green">(-82,656 [-5.3%])</span> <div style='text-align: right'>1,479,198</div>  |
| leaf_aggregation | Boundary | true | StoreF | STOREW | 0 | <span style="color: green">(-22,176 [-5.3%])</span> <div style='text-align: right'>396,858</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>11,728,993</div>  |
| leaf_aggregation | AccessAdapter<2> | true | StoreF | STOREW2 | 0 | <span style="color: red">(+3,234 [+0.2%])</span> <div style='text-align: right'>1,464,914</div>  |
| leaf_aggregation | AccessAdapter<4> | true | StoreF | STOREW2 | 0 | <span style="color: red">(+2,184 [+0.3%])</span> <div style='text-align: right'>867,269</div>  |
| leaf_aggregation | AccessAdapter<8> | true | StoreF | STOREW2 | 0 | <span style="color: red">(+1,428 [+0.3%])</span> <div style='text-align: right'>572,254</div>  |
| leaf_aggregation | Boundary | true | StoreF | STOREW2 | 0 | <div style='text-align: right'>74,492</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | StoreHintWord | ADD | 0 | <span style="color: green">(-10,080 [-0.1%])</span> <div style='text-align: right'>13,654,110</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-15,498 [-0.1%])</span> <div style='text-align: right'>19,636,745</div>  |
| leaf_aggregation | Boundary | true | StoreHintWord | SHINTW | 0 | <span style="color: green">(-4,158 [-0.1%])</span> <div style='text-align: right'>5,268,395</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreV | STOREW | 0 | <div style='text-align: right'>121,032</div>  |
| leaf_aggregation | Boundary | true | StoreV | STOREW | 0 | <div style='text-align: right'>32,472</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | StoreV | STOREW2 | 0 | <span style="color: green">(-3,444 [-0.1%])</span> <div style='text-align: right'>2,680,170</div>  |
| leaf_aggregation | Boundary | true | StoreV | STOREW2 | 0 | <span style="color: green">(-924 [-0.1%])</span> <div style='text-align: right'>643,137</div>  |
| leaf_aggregation | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | true | SubE | FE4SUB | 0 | <div style='text-align: right'>664,000</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubE | FE4SUB | 0 | <span style="color: green">(-44 [-0.0%])</span> <div style='text-align: right'>540,056</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubE | FE4SUB | 0 | <span style="color: green">(-26 [-0.0%])</span> <div style='text-align: right'>319,124</div>  |
| leaf_aggregation | Boundary | true | SubE | FE4SUB | 0 | <div style='text-align: right'>578,028</div>  |
| leaf_aggregation | <NativeLoadStoreAdapterAir<1>,KernelLoadStoreCoreAir<1>> | true | SubEF | LOADW | 0 | <div style='text-align: right'>788,430</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEF | LOADW | 0 | <div style='text-align: right'>70,389</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEF | SUB | 0 | <div style='text-align: right'>192,300</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEF | SUB | 0 | <div style='text-align: right'>70,389</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEF | SUB | 0 | <div style='text-align: right'>83,187</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEFI | ADD | 0 | <div style='text-align: right'>283,320</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEFI | ADD | 0 | <div style='text-align: right'>7,700</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEFI | ADD | 0 | <div style='text-align: right'>4,550</div>  |
| leaf_aggregation | Boundary | true | SubEFI | ADD | 0 | <div style='text-align: right'>99,616</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubEI | ADD | 0 | <div style='text-align: right'>15,840</div>  |
| leaf_aggregation | AccessAdapter<2> | true | SubEI | ADD | 0 | <div style='text-align: right'>4,114</div>  |
| leaf_aggregation | AccessAdapter<4> | true | SubEI | ADD | 0 | <div style='text-align: right'>2,431</div>  |
| leaf_aggregation | Boundary | true | SubEI | ADD | 0 | <div style='text-align: right'>1,056</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubFI | SUB | 0 | <div style='text-align: right'>39,990</div>  |
| leaf_aggregation | Boundary | true | SubFI | SUB | 0 | <div style='text-align: right'>14,630</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubV | SUB | 0 | <span style="color: green">(-5,040 [-0.2%])</span> <div style='text-align: right'>2,036,880</div>  |
| leaf_aggregation | Boundary | true | SubV | SUB | 0 | <div style='text-align: right'>44</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubVI | SUB | 0 | <div style='text-align: right'>28,740</div>  |
| leaf_aggregation | Boundary | true | SubVI | SUB | 0 | <div style='text-align: right'>506</div>  |
| leaf_aggregation | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | true | SubVIN | SUB | 0 | <div style='text-align: right'>23,940</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-base64_json_program.dsl_ir.opcode.frequency.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e5471991fa80e66c6365fc7ebd0ed0061ee10820/base64_json-2-2-64cpu-linux-arm64-mimalloc-leaf_aggregation.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/axiom-crypto/afs-prototype/commit/e5471991fa80e66c6365fc7ebd0ed0061ee10820

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12089310696)