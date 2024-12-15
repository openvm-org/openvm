| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| base64_json_program | <div style='text-align: right'>2</div>  | <span style="color: red">(+6,156,724 [+68.7%])</span> <div style='text-align: right'>15,116,803</div>  | <span style="color: green">(-2 [-0.0%])</span> <div style='text-align: right'>217,347</div>  | <span style="color: green">(-865.0 [-31.1%])</span> <div style='text-align: right'>1,917.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| commit_exe_time_ms | execute_and_trace_gen_time_ms | execute_time_ms | fri.log_blowup | keygen_time_ms |
| --- | --- | --- | --- | --- |
| <div style='text-align: right'>16.0</div>  | <div style='text-align: right'>499.0</div>  | <div style='text-align: right'>324.0</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>74,716.0</div>  |

| air_name | constraints | interactions | quotient_deg |
| --- | --- | --- | --- |
| ProgramAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |
| VmConnectorAir | <div style='text-align: right'>9</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| PersistentBoundaryAir<8> | <div style='text-align: right'>6</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| MemoryMerkleAir<8> | <div style='text-align: right'>40</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<2> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<4> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<8> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<16> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<32> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| AccessAdapterAir<64> | <div style='text-align: right'>14</div>  | <div style='text-align: right'>5</div>  | <div style='text-align: right'>2</div>  |
| KeccakVmAir | <div style='text-align: right'>4,571</div>  | <div style='text-align: right'>321</div>  | <div style='text-align: right'>2</div>  |
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
| PhantomAir | <div style='text-align: right'>5</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>2</div>  |
| Poseidon2VmAir<BabyBearParameters> | <div style='text-align: right'>525</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>2</div>  |
| VariableRangeCheckerAir | <div style='text-align: right'>4</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>1</div>  |

| group | segment | stark_prove_excluding_trace_time_ms | total_cells | total_cells_used | total_cycles | trace_gen_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | 0 | <span style="color: green">(-126.0 [-6.2%])</span> <div style='text-align: right'>1,917.0</div>  | <span style="color: red">(+1,179,648 [+2.4%])</span> <div style='text-align: right'>50,533,140</div>  | <div style='text-align: right'>15,116,803</div>  | <div style='text-align: right'>217,347</div>  | <div style='text-align: right'>175.0</div>  |

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| base64_json_program | ProgramChip | 0 | <div style='text-align: right'>18,961</div>  |
| base64_json_program | VmConnectorAir | 0 | <div style='text-align: right'>2</div>  |
| base64_json_program | Boundary | 0 | <div style='text-align: right'>5,178</div>  |
| base64_json_program | Merkle | 0 | <div style='text-align: right'>5,524</div>  |
| base64_json_program | AccessAdapter<8> | 0 | <div style='text-align: right'>5,178</div>  |
| base64_json_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>1,563</div>  |
| base64_json_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | <div style='text-align: right'>86</div>  |
| base64_json_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | <div style='text-align: right'>116</div>  |
| base64_json_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>524,288</div>  |
| base64_json_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>1,331</div>  |
| base64_json_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | <div style='text-align: right'>2,940</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>5,003</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>16,738</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | <div style='text-align: right'>27,336</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | <div style='text-align: right'>1,236</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | <div style='text-align: right'>55,121</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | <div style='text-align: right'>16,188</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>575</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | <div style='text-align: right'>89,109</div>  |
| base64_json_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>65,536</div>  |
| base64_json_program | PhantomAir | 0 | <div style='text-align: right'>5</div>  |
| base64_json_program | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>10,702</div>  |
| base64_json_program | VariableRangeCheckerAir | 0 | <div style='text-align: right'>262,144</div>  |

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| base64_json_program |  | ADD | 0 | <div style='text-align: right'>69,773</div>  |
| base64_json_program |  | AND | 0 | <div style='text-align: right'>10,124</div>  |
| base64_json_program |  | AUIPC | 0 | <div style='text-align: right'>1,331</div>  |
| base64_json_program |  | BEQ | 0 | <div style='text-align: right'>15,568</div>  |
| base64_json_program |  | BGE | 0 | <div style='text-align: right'>703</div>  |
| base64_json_program |  | BGEU | 0 | <div style='text-align: right'>6,863</div>  |
| base64_json_program |  | BLT | 0 | <div style='text-align: right'>3,354</div>  |
| base64_json_program |  | BLTU | 0 | <div style='text-align: right'>5,818</div>  |
| base64_json_program |  | BNE | 0 | <div style='text-align: right'>11,768</div>  |
| base64_json_program |  | HINT_STOREW | 0 | <div style='text-align: right'>1,563</div>  |
| base64_json_program |  | JAL | 0 | <div style='text-align: right'>3,685</div>  |
| base64_json_program |  | JALR | 0 | <div style='text-align: right'>2,940</div>  |
| base64_json_program |  | LOADB | 0 | <div style='text-align: right'>1,236</div>  |
| base64_json_program |  | LOADBU | 0 | <div style='text-align: right'>23,858</div>  |
| base64_json_program |  | LOADHU | 0 | <div style='text-align: right'>3</div>  |
| base64_json_program |  | LOADW | 0 | <div style='text-align: right'>13,465</div>  |
| base64_json_program |  | LUI | 0 | <div style='text-align: right'>1,318</div>  |
| base64_json_program |  | MUL | 0 | <div style='text-align: right'>116</div>  |
| base64_json_program |  | MULHU | 0 | <div style='text-align: right'>86</div>  |
| base64_json_program |  | OR | 0 | <div style='text-align: right'>7,608</div>  |
| base64_json_program |  | PHANTOM | 0 | <div style='text-align: right'>5</div>  |
| base64_json_program |  | SLL | 0 | <div style='text-align: right'>7,118</div>  |
| base64_json_program |  | SLT | 0 | <div style='text-align: right'>5</div>  |
| base64_json_program |  | SLTU | 0 | <div style='text-align: right'>570</div>  |
| base64_json_program |  | SRA | 0 | <div style='text-align: right'>8</div>  |
| base64_json_program |  | SRL | 0 | <div style='text-align: right'>9,062</div>  |
| base64_json_program |  | STOREB | 0 | <div style='text-align: right'>5,133</div>  |
| base64_json_program |  | STOREH | 0 | <div style='text-align: right'>10</div>  |
| base64_json_program |  | STOREW | 0 | <div style='text-align: right'>12,652</div>  |
| base64_json_program |  | SUB | 0 | <div style='text-align: right'>1,416</div>  |
| base64_json_program |  | XOR | 0 | <div style='text-align: right'>188</div>  |

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | <div style='text-align: right'>2,511,828</div>  |
| base64_json_program | AccessAdapter<8> |  | ADD | 0 | <div style='text-align: right'>85</div>  |
| base64_json_program | Boundary |  | ADD | 0 | <div style='text-align: right'>200</div>  |
| base64_json_program | Merkle |  | ADD | 0 | <div style='text-align: right'>128</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | <div style='text-align: right'>364,464</div>  |
| base64_json_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | <div style='text-align: right'>27,951</div>  |
| base64_json_program | AccessAdapter<8> |  | AUIPC | 0 | <div style='text-align: right'>51</div>  |
| base64_json_program | Boundary |  | AUIPC | 0 | <div style='text-align: right'>120</div>  |
| base64_json_program | Merkle |  | AUIPC | 0 | <div style='text-align: right'>3,520</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | <div style='text-align: right'>404,768</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGE | 0 | <div style='text-align: right'>22,496</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | <div style='text-align: right'>219,616</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | <div style='text-align: right'>107,328</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | <div style='text-align: right'>186,176</div>  |
| base64_json_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | <div style='text-align: right'>305,968</div>  |
| base64_json_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | <div style='text-align: right'>40,638</div>  |
| base64_json_program | AccessAdapter<8> |  | HINT_STOREW | 0 | <div style='text-align: right'>13,277</div>  |
| base64_json_program | Boundary |  | HINT_STOREW | 0 | <div style='text-align: right'>31,240</div>  |
| base64_json_program | Merkle |  | HINT_STOREW | 0 | <div style='text-align: right'>50,240</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | <div style='text-align: right'>66,330</div>  |
| base64_json_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | <div style='text-align: right'>82,320</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | <div style='text-align: right'>43,260</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | <div style='text-align: right'>954,320</div>  |
| base64_json_program | AccessAdapter<8> |  | LOADBU | 0 | <div style='text-align: right'>2,856</div>  |
| base64_json_program | Boundary |  | LOADBU | 0 | <div style='text-align: right'>6,720</div>  |
| base64_json_program | Merkle |  | LOADBU | 0 | <div style='text-align: right'>12,288</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADHU | 0 | <div style='text-align: right'>120</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | <div style='text-align: right'>538,600</div>  |
| base64_json_program | AccessAdapter<8> |  | LOADW | 0 | <div style='text-align: right'>1,921</div>  |
| base64_json_program | Boundary |  | LOADW | 0 | <div style='text-align: right'>4,520</div>  |
| base64_json_program | Merkle |  | LOADW | 0 | <div style='text-align: right'>12,224</div>  |
| base64_json_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | <div style='text-align: right'>23,724</div>  |
| base64_json_program | AccessAdapter<8> |  | LUI | 0 | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary |  | LUI | 0 | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | <div style='text-align: right'>3,596</div>  |
| base64_json_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | <div style='text-align: right'>3,354</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | <div style='text-align: right'>273,888</div>  |
| base64_json_program | PhantomAir |  | PHANTOM | 0 | <div style='text-align: right'>30</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | <div style='text-align: right'>377,254</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLT | 0 | <div style='text-align: right'>185</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | <div style='text-align: right'>21,090</div>  |
| base64_json_program | AccessAdapter<8> |  | SLTU | 0 | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary |  | SLTU | 0 | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRA | 0 | <div style='text-align: right'>424</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | <div style='text-align: right'>480,286</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | <div style='text-align: right'>205,320</div>  |
| base64_json_program | AccessAdapter<8> |  | STOREB | 0 | <div style='text-align: right'>10,472</div>  |
| base64_json_program | Boundary |  | STOREB | 0 | <div style='text-align: right'>24,640</div>  |
| base64_json_program | Merkle |  | STOREB | 0 | <div style='text-align: right'>39,232</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREH | 0 | <div style='text-align: right'>400</div>  |
| base64_json_program | AccessAdapter<8> |  | STOREH | 0 | <div style='text-align: right'>17</div>  |
| base64_json_program | Boundary |  | STOREH | 0 | <div style='text-align: right'>40</div>  |
| base64_json_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | <div style='text-align: right'>506,080</div>  |
| base64_json_program | AccessAdapter<8> |  | STOREW | 0 | <div style='text-align: right'>15,300</div>  |
| base64_json_program | Boundary |  | STOREW | 0 | <div style='text-align: right'>36,000</div>  |
| base64_json_program | Merkle |  | STOREW | 0 | <div style='text-align: right'>59,072</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | <div style='text-align: right'>50,976</div>  |
| base64_json_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | <div style='text-align: right'>6,768</div>  |

| group | execute_time_ms | fri.log_blowup | num_segments | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | <span style="color: green">(-111.0 [-25.5%])</span> <div style='text-align: right'>325.0</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <span style="color: red">(+6,156,724 [+68.7%])</span> <div style='text-align: right'>15,116,803</div>  | <span style="color: green">(-2 [-0.0%])</span> <div style='text-align: right'>217,347</div>  | <span style="color: green">(-865.0 [-31.1%])</span> <div style='text-align: right'>1,917.0</div>  |

| group | air_name | segment | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base64_json_program | ProgramAir | 0 | <div style='text-align: right'>589,824</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmConnectorAir | 0 | <div style='text-align: right'>32</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| base64_json_program | PersistentBoundaryAir<8> | 0 | <div style='text-align: right'>262,144</div>  | <div style='text-align: right'>20</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | MemoryMerkleAir<8> | 0 | <div style='text-align: right'>425,984</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | AccessAdapterAir<8> | 0 | <div style='text-align: right'>335,872</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | KeccakVmAir | 0 | <div style='text-align: right'>4,452</div>  | <div style='text-align: right'>3,164</div>  | <div style='text-align: right'>1,288</div>  |  | <div style='text-align: right'>1</div>  |
| base64_json_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>126,976</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | <div style='text-align: right'>17,792</div>  | <div style='text-align: right'>39</div>  | <div style='text-align: right'>100</div>  |  | <div style='text-align: right'>128</div>  |
| base64_json_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | <div style='text-align: right'>14,208</div>  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>80</div>  |  | <div style='text-align: right'>128</div>  |
| base64_json_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>4,718,592</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>524,288</div>  |
| base64_json_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>100,352</div>  | <div style='text-align: right'>21</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | <div style='text-align: right'>262,144</div>  | <div style='text-align: right'>28</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>4,096</div>  |
| base64_json_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>507,904</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>44</div>  |  | <div style='text-align: right'>8,192</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>2,883,584</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>56</div>  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | <div style='text-align: right'>2,424,832</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>48</div>  |  | <div style='text-align: right'>32,768</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | <div style='text-align: right'>227,328</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>2,048</div>  |
| base64_json_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | <div style='text-align: right'>7,340,032</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>72</div>  |  | <div style='text-align: right'>65,536</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | <div style='text-align: right'>1,720,320</div>  | <div style='text-align: right'>53</div>  | <div style='text-align: right'>52</div>  |  | <div style='text-align: right'>16,384</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>78,848</div>  | <div style='text-align: right'>37</div>  | <div style='text-align: right'>40</div>  |  | <div style='text-align: right'>1,024</div>  |
| base64_json_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | <div style='text-align: right'>15,204,352</div>  | <div style='text-align: right'>36</div>  | <div style='text-align: right'>80</div>  |  | <div style='text-align: right'>131,072</div>  |
| base64_json_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>655,360</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>65,536</div>  |
| base64_json_program | PhantomAir | 0 | <div style='text-align: right'>144</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>8</div>  |
| base64_json_program | Poseidon2VmAir<BabyBearParameters> | 0 | <div style='text-align: right'>10,272,768</div>  | <div style='text-align: right'>559</div>  | <div style='text-align: right'>68</div>  |  | <div style='text-align: right'>16,384</div>  |
| base64_json_program | VariableRangeCheckerAir | 0 | <span style="color: red">(+1,179,648 [+100.0%])</span> <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <span style="color: red">(+131,072 [+100.0%])</span> <div style='text-align: right'>262,144</div>  |

| segment | trace_gen_time_ms |
| --- | --- |
| 0 | <div style='text-align: right'>173.0</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.air_name.cells_used.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.frequency.reverse.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.frequency.svg)](https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f/base64_json-base64_json_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/axiom-crypto/afs-prototype/commit/f612ccbfbd8c47c8c5a5b00744f1ba4f0c8b698f

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12291633663)