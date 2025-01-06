| group | fri.log_blowup | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- |
| regex_program | <div style='text-align: right'>2</div>  | <span style="color: green">(-73,862,276 [-30.9%])</span> <div style='text-align: right'>165,028,173</div>  | <div style='text-align: right'>8,381,808</div>  | <span style="color: green">(-1,473.0 [-8.4%])</span> <div style='text-align: right'>16,064.0</div>  |
| leaf | <div style='text-align: right'>2</div>  | <span style="color: green">(-20,463,557 [-6.5%])</span> <div style='text-align: right'>294,978,850</div>  | <span style="color: green">(-402,372 [-2.7%])</span> <div style='text-align: right'>14,240,672</div>  | <span style="color: green">(-1,607.0 [-5.3%])</span> <div style='text-align: right'>28,432.0</div>  |


<details>
<summary>Detailed Metrics</summary>

| commit_exe_time_ms | fri.log_blowup | keygen_time_ms |
| --- | --- | --- |
| <span style="color: red">(+1.0 [+2.0%])</span> <div style='text-align: right'>50.0</div>  | <div style='text-align: right'>2</div>  | <span style="color: green">(-44,823.0 [-98.3%])</span> <div style='text-align: right'>752.0</div>  |

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
| regex_program | 0 | <span style="color: green">(-1,473.0 [-8.4%])</span> <div style='text-align: right'>16,064.0</div>  | <span style="color: green">(-159,318,016 [-20.1%])</span> <div style='text-align: right'>632,452,480</div>  | <span style="color: green">(-73,862,276 [-30.9%])</span> <div style='text-align: right'>165,028,173</div>  | <div style='text-align: right'>8,381,808</div>  | <span style="color: green">(-1,826.0 [-70.9%])</span> <div style='text-align: right'>750.0</div>  |

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| regex_program | ProgramChip | 0 | <div style='text-align: right'>89,914</div>  |
| regex_program | VmConnectorAir | 0 | <div style='text-align: right'>2</div>  |
| regex_program | Boundary | 0 | <div style='text-align: right'>69,164</div>  |
| regex_program | Merkle | 0 | <div style='text-align: right'>70,500</div>  |
| regex_program | AccessAdapter<2> | 0 | <div style='text-align: right'>42</div>  |
| regex_program | AccessAdapter<4> | 0 | <div style='text-align: right'>22</div>  |
| regex_program | AccessAdapter<8> | 0 | <div style='text-align: right'>69,164</div>  |
| regex_program | KeccakVmAir | 0 | <div style='text-align: right'>24</div>  |
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>12,767</div>  |
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | <div style='text-align: right'>114</div>  |
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | <div style='text-align: right'>244</div>  |
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | <div style='text-align: right'>52,087</div>  |
| regex_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>524,288</div>  |
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>39,557</div>  |
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | <div style='text-align: right'>130,444</div>  |
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>106,072</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>198,078</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | <div style='text-align: right'>282,074</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | <div style='text-align: right'>687</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | <div style='text-align: right'>1,961,387</div>  |
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | <div style='text-align: right'>218,625</div>  |
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | <div style='text-align: right'>38,005</div>  |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | <div style='text-align: right'>1,150,473</div>  |
| regex_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>65,536</div>  |
| regex_program | PhantomAir | 0 | <div style='text-align: right'>289</div>  |
| regex_program | Arc<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>14,033</div>  |
| regex_program | VariableRangeCheckerAir | 0 | <div style='text-align: right'>262,144</div>  |

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| regex_program |  | ADD | 0 | <div style='text-align: right'>1,008,001</div>  |
| regex_program |  | AND | 0 | <div style='text-align: right'>66,789</div>  |
| regex_program |  | AUIPC | 0 | <div style='text-align: right'>39,557</div>  |
| regex_program |  | BEQ | 0 | <div style='text-align: right'>178,501</div>  |
| regex_program |  | BGE | 0 | <div style='text-align: right'>294</div>  |
| regex_program |  | BGEU | 0 | <div style='text-align: right'>121,597</div>  |
| regex_program |  | BLT | 0 | <div style='text-align: right'>5,141</div>  |
| regex_program |  | BLTU | 0 | <div style='text-align: right'>71,046</div>  |
| regex_program |  | BNE | 0 | <div style='text-align: right'>103,573</div>  |
| regex_program |  | DIVU | 0 | <div style='text-align: right'>114</div>  |
| regex_program |  | HINT_STOREW | 0 | <div style='text-align: right'>12,767</div>  |
| regex_program |  | JAL | 0 | <div style='text-align: right'>61,576</div>  |
| regex_program |  | JALR | 0 | <div style='text-align: right'>130,444</div>  |
| regex_program |  | KECCAK256 | 0 | <div style='text-align: right'>1</div>  |
| regex_program |  | LOADB | 0 | <div style='text-align: right'>679</div>  |
| regex_program |  | LOADBU | 0 | <div style='text-align: right'>27,294</div>  |
| regex_program |  | LOADH | 0 | <div style='text-align: right'>8</div>  |
| regex_program |  | LOADHU | 0 | <div style='text-align: right'>95</div>  |
| regex_program |  | LOADW | 0 | <div style='text-align: right'>1,142,838</div>  |
| regex_program |  | LUI | 0 | <div style='text-align: right'>44,496</div>  |
| regex_program |  | MUL | 0 | <div style='text-align: right'>52,087</div>  |
| regex_program |  | MULHU | 0 | <div style='text-align: right'>244</div>  |
| regex_program |  | OR | 0 | <div style='text-align: right'>23,536</div>  |
| regex_program |  | PHANTOM | 0 | <div style='text-align: right'>289</div>  |
| regex_program |  | SLL | 0 | <div style='text-align: right'>213,542</div>  |
| regex_program |  | SLT | 0 | <div style='text-align: right'>5</div>  |
| regex_program |  | SLTU | 0 | <div style='text-align: right'>38,000</div>  |
| regex_program |  | SRA | 0 | <div style='text-align: right'>1</div>  |
| regex_program |  | SRL | 0 | <div style='text-align: right'>5,082</div>  |
| regex_program |  | STOREB | 0 | <div style='text-align: right'>12,721</div>  |
| regex_program |  | STOREH | 0 | <div style='text-align: right'>10,074</div>  |
| regex_program |  | STOREW | 0 | <div style='text-align: right'>768,365</div>  |
| regex_program |  | SUB | 0 | <div style='text-align: right'>42,583</div>  |
| regex_program |  | XOR | 0 | <div style='text-align: right'>9,564</div>  |

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | <div style='text-align: right'>36,288,036</div>  |
| regex_program | AccessAdapter<8> |  | ADD | 0 | <div style='text-align: right'>102</div>  |
| regex_program | Boundary |  | ADD | 0 | <div style='text-align: right'>240</div>  |
| regex_program | Merkle |  | ADD | 0 | <div style='text-align: right'>128</div>  |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | <div style='text-align: right'>2,404,404</div>  |
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | <div style='text-align: right'>830,697</div>  |
| regex_program | AccessAdapter<8> |  | AUIPC | 0 | <div style='text-align: right'>34</div>  |
| regex_program | Boundary |  | AUIPC | 0 | <div style='text-align: right'>80</div>  |
| regex_program | Merkle |  | AUIPC | 0 | <div style='text-align: right'>3,456</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | <div style='text-align: right'>4,641,026</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGE | 0 | <div style='text-align: right'>9,408</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | <div style='text-align: right'>3,891,104</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | <div style='text-align: right'>164,512</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | <div style='text-align: right'>2,273,472</div>  |
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | <div style='text-align: right'>2,692,898</div>  |
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | <div style='text-align: right'>6,498</div>  |
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | <div style='text-align: right'>331,942</div>  |
| regex_program | AccessAdapter<8> |  | HINT_STOREW | 0 | <div style='text-align: right'>108,528</div>  |
| regex_program | Boundary |  | HINT_STOREW | 0 | <div style='text-align: right'>255,360</div>  |
| regex_program | Merkle |  | HINT_STOREW | 0 | <div style='text-align: right'>408,576</div>  |
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | <div style='text-align: right'>1,108,368</div>  |
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | <div style='text-align: right'>3,652,432</div>  |
| regex_program | AccessAdapter<2> |  | KECCAK256 | 0 | <div style='text-align: right'>231</div>  |
| regex_program | AccessAdapter<4> |  | KECCAK256 | 0 | <div style='text-align: right'>143</div>  |
| regex_program | KeccakVmAir |  | KECCAK256 | 0 | <div style='text-align: right'>75,936</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | <div style='text-align: right'>23,765</div>  |
| regex_program | AccessAdapter<8> |  | LOADB | 0 | <div style='text-align: right'>17</div>  |
| regex_program | Boundary |  | LOADB | 0 | <div style='text-align: right'>40</div>  |
| regex_program | Merkle |  | LOADB | 0 | <div style='text-align: right'>64</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | <div style='text-align: right'>1,091,760</div>  |
| regex_program | AccessAdapter<8> |  | LOADBU | 0 | <div style='text-align: right'>187</div>  |
| regex_program | Boundary |  | LOADBU | 0 | <div style='text-align: right'>440</div>  |
| regex_program | Merkle |  | LOADBU | 0 | <div style='text-align: right'>2,624</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADH | 0 | <div style='text-align: right'>280</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADHU | 0 | <div style='text-align: right'>3,800</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | <div style='text-align: right'>45,713,520</div>  |
| regex_program | AccessAdapter<8> |  | LOADW | 0 | <div style='text-align: right'>3,026</div>  |
| regex_program | Boundary |  | LOADW | 0 | <div style='text-align: right'>7,120</div>  |
| regex_program | Merkle |  | LOADW | 0 | <div style='text-align: right'>26,112</div>  |
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | <div style='text-align: right'>800,928</div>  |
| regex_program | AccessAdapter<8> |  | LUI | 0 | <div style='text-align: right'>17</div>  |
| regex_program | Boundary |  | LUI | 0 | <div style='text-align: right'>40</div>  |
| regex_program | Merkle |  | LUI | 0 | <div style='text-align: right'>64</div>  |
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | <div style='text-align: right'>1,614,697</div>  |
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | <div style='text-align: right'>9,516</div>  |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | <div style='text-align: right'>847,296</div>  |
| regex_program | PhantomAir |  | PHANTOM | 0 | <div style='text-align: right'>1,734</div>  |
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | <div style='text-align: right'>11,317,726</div>  |
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLT | 0 | <div style='text-align: right'>185</div>  |
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | <div style='text-align: right'>1,406,000</div>  |
| regex_program | AccessAdapter<8> |  | SLTU | 0 | <div style='text-align: right'>17</div>  |
| regex_program | Boundary |  | SLTU | 0 | <div style='text-align: right'>40</div>  |
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRA | 0 | <div style='text-align: right'>53</div>  |
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | <div style='text-align: right'>269,346</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | <div style='text-align: right'>508,840</div>  |
| regex_program | AccessAdapter<8> |  | STOREB | 0 | <div style='text-align: right'>1,105</div>  |
| regex_program | Boundary |  | STOREB | 0 | <div style='text-align: right'>2,600</div>  |
| regex_program | Merkle |  | STOREB | 0 | <div style='text-align: right'>8,320</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREH | 0 | <div style='text-align: right'>402,960</div>  |
| regex_program | AccessAdapter<8> |  | STOREH | 0 | <div style='text-align: right'>85,221</div>  |
| regex_program | Boundary |  | STOREH | 0 | <div style='text-align: right'>200,520</div>  |
| regex_program | Merkle |  | STOREH | 0 | <div style='text-align: right'>321,024</div>  |
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | <div style='text-align: right'>30,734,600</div>  |
| regex_program | AccessAdapter<8> |  | STOREW | 0 | <div style='text-align: right'>389,640</div>  |
| regex_program | Boundary |  | STOREW | 0 | <div style='text-align: right'>916,800</div>  |
| regex_program | Merkle |  | STOREW | 0 | <div style='text-align: right'>1,485,568</div>  |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | <div style='text-align: right'>1,532,988</div>  |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | <div style='text-align: right'>344,304</div>  |

| group | execute_time_ms | fri.log_blowup | num_segments | total_cells_used | total_cycles | total_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| regex_program | <span style="color: green">(-758.0 [-11.2%])</span> <div style='text-align: right'>6,019.0</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>1</div>  | <span style="color: green">(-73,862,276 [-30.9%])</span> <div style='text-align: right'>165,028,173</div>  | <div style='text-align: right'>8,381,808</div>  | <span style="color: green">(-1,473.0 [-8.4%])</span> <div style='text-align: right'>16,064.0</div>  |
| leaf |  | <div style='text-align: right'>2</div>  |  | <span style="color: green">(-20,463,557 [-6.5%])</span> <div style='text-align: right'>294,978,850</div>  | <span style="color: green">(-402,372 [-2.7%])</span> <div style='text-align: right'>14,240,672</div>  | <span style="color: green">(-1,607.0 [-5.3%])</span> <div style='text-align: right'>28,432.0</div>  |

| group | air_name | segment | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | ProgramAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | VmConnectorAir | 0 | <div style='text-align: right'>32</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>12</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| regex_program | PersistentBoundaryAir<8> | 0 | <div style='text-align: right'>4,194,304</div>  | <div style='text-align: right'>20</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | MemoryMerkleAir<8> | 0 | <div style='text-align: right'>6,815,744</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | AccessAdapterAir<2> | 0 | <div style='text-align: right'>2,240</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>64</div>  |
| regex_program | AccessAdapterAir<4> | 0 | <div style='text-align: right'>1,184</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>32</div>  |
| regex_program | AccessAdapterAir<8> | 0 | <div style='text-align: right'>5,373,952</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | KeccakVmAir | 0 | <div style='text-align: right'>142,464</div>  | <div style='text-align: right'>3,164</div>  | <div style='text-align: right'>1,288</div>  |  | <div style='text-align: right'>32</div>  |
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | <div style='text-align: right'>1,015,808</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>16,384</div>  |
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | <div style='text-align: right'>20,608</div>  | <div style='text-align: right'>57</div>  | <div style='text-align: right'>104</div>  |  | <div style='text-align: right'>128</div>  |
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | <div style='text-align: right'>35,584</div>  | <div style='text-align: right'>39</div>  | <div style='text-align: right'>100</div>  |  | <div style='text-align: right'>256</div>  |
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | <div style='text-align: right'>7,274,496</div>  | <div style='text-align: right'>31</div>  | <div style='text-align: right'>80</div>  |  | <div style='text-align: right'>65,536</div>  |
| regex_program | RangeTupleCheckerAir<2> | 0 | <div style='text-align: right'>4,718,592</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>524,288</div>  |
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | <div style='text-align: right'>3,211,264</div>  | <div style='text-align: right'>21</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>65,536</div>  |
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | <div style='text-align: right'>8,388,608</div>  | <div style='text-align: right'>28</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | <div style='text-align: right'>8,126,464</div>  | <div style='text-align: right'>18</div>  | <div style='text-align: right'>44</div>  |  | <div style='text-align: right'>131,072</div>  |
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>23,068,672</div>  | <div style='text-align: right'>32</div>  | <div style='text-align: right'>56</div>  |  | <div style='text-align: right'>262,144</div>  |
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | <div style='text-align: right'>38,797,312</div>  | <div style='text-align: right'>26</div>  | <div style='text-align: right'>48</div>  |  | <div style='text-align: right'>524,288</div>  |
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | <div style='text-align: right'>113,664</div>  | <div style='text-align: right'>35</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>1,024</div>  |
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | <div style='text-align: right'>234,881,024</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>72</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | <div style='text-align: right'>27,525,120</div>  | <div style='text-align: right'>53</div>  | <div style='text-align: right'>52</div>  |  | <div style='text-align: right'>262,144</div>  |
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | <div style='text-align: right'>5,046,272</div>  | <div style='text-align: right'>37</div>  | <div style='text-align: right'>40</div>  |  | <div style='text-align: right'>65,536</div>  |
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | <div style='text-align: right'>243,269,632</div>  | <div style='text-align: right'>36</div>  | <div style='text-align: right'>80</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| regex_program | BitwiseOperationLookupAir<8> | 0 | <div style='text-align: right'>655,360</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>3</div>  | <div style='text-align: right'>65,536</div>  |
| regex_program | PhantomAir | 0 | <div style='text-align: right'>9,216</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>512</div>  |
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>5,046,272</div>  | <div style='text-align: right'>300</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>16,384</div>  |
| regex_program | VariableRangeCheckerAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

| group | idx | execute_time_ms | stark_prove_excluding_trace_time_ms | total_cells | total_cells_used | total_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | <span style="color: green">(-696.0 [-7.7%])</span> <div style='text-align: right'>8,342.0</div>  | <span style="color: green">(-1,607.0 [-5.3%])</span> <div style='text-align: right'>28,432.0</div>  | <span style="color: green">(-33,751,040 [-4.2%])</span> <div style='text-align: right'>773,458,392</div>  | <span style="color: green">(-20,463,557 [-6.5%])</span> <div style='text-align: right'>294,978,850</div>  | <span style="color: green">(-201,186 [-2.7%])</span> <div style='text-align: right'>7,120,336</div>  |

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | ProgramChip | 0 | <span style="color: green">(-10,757 [-3.5%])</span> <div style='text-align: right'>300,364</div>  |
| leaf | VmConnectorAir | 0 | <div style='text-align: right'>2</div>  |
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | <div style='text-align: right'>36</div>  |
| leaf | Boundary | 0 | <span style="color: green">(-15,950 [-1.5%])</span> <div style='text-align: right'>1,042,375</div>  |
| leaf | AccessAdapter<2> | 0 | <span style="color: green">(-23,888 [-2.1%])</span> <div style='text-align: right'>1,087,668</div>  |
| leaf | AccessAdapter<4> | 0 | <span style="color: green">(-11,944 [-2.1%])</span> <div style='text-align: right'>544,044</div>  |
| leaf | AccessAdapter<8> | 0 | <span style="color: green">(-3,360 [-2.9%])</span> <div style='text-align: right'>111,814</div>  |
| leaf | Arc<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>54,582</div>  |
| leaf | FriReducedOpeningAir | 0 | <span style="color: green">(-26,796 [-4.7%])</span> <div style='text-align: right'>544,152</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | <span style="color: green">(-3,021 [-2.7%])</span> <div style='text-align: right'>108,742</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | <span style="color: green">(-79,742 [-2.8%])</span> <div style='text-align: right'>2,791,074</div>  |
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | <span style="color: red">(+1,471 [+1.5%])</span> <div style='text-align: right'>98,840</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | <span style="color: green">(-43,110 [-2.9%])</span> <div style='text-align: right'>1,421,000</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | <span style="color: green">(-48,308 [-2.3%])</span> <div style='text-align: right'>2,017,265</div>  |
| leaf | PhantomAir | 0 | <span style="color: green">(-26,796 [-4.1%])</span> <div style='text-align: right'>621,699</div>  |
| leaf | VariableRangeCheckerAir | 0 | <div style='text-align: right'>262,144</div>  |

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | <span style="color: green">(-77,638 [-3.1%])</span> <div style='text-align: right'>2,467,026</div>  |
| leaf |  | 0 | BBE4DIV | <div style='text-align: right'>8,109</div>  |
| leaf |  | 0 | BBE4MUL | <span style="color: green">(-1,774 [-4.7%])</span> <div style='text-align: right'>36,358</div>  |
| leaf |  | 0 | BEQ | <div style='text-align: right'>19,898</div>  |
| leaf |  | 0 | BNE | <span style="color: green">(-43,110 [-3.0%])</span> <div style='text-align: right'>1,401,102</div>  |
| leaf |  | 0 | COMP_POS2 | <div style='text-align: right'>18,449</div>  |
| leaf |  | 0 | DIV | <div style='text-align: right'>177</div>  |
| leaf |  | 0 | FE4ADD | <span style="color: green">(-996 [-2.1%])</span> <div style='text-align: right'>47,552</div>  |
| leaf |  | 0 | FE4SUB | <span style="color: green">(-251 [-1.5%])</span> <div style='text-align: right'>16,723</div>  |
| leaf |  | 0 | FRI_REDUCED_OPENING | <div style='text-align: right'>7,098</div>  |
| leaf |  | 0 | JAL | <span style="color: red">(+1,471 [+1.5%])</span> <div style='text-align: right'>98,840</div>  |
| leaf |  | 0 | LOADW | <span style="color: green">(-2,552 [-1.2%])</span> <div style='text-align: right'>209,719</div>  |
| leaf |  | 0 | LOADW2 | <span style="color: green">(-13,566 [-2.0%])</span> <div style='text-align: right'>653,000</div>  |
| leaf |  | 0 | MUL | <span style="color: green">(-1,432 [-0.6%])</span> <div style='text-align: right'>227,136</div>  |
| leaf |  | 0 | PERM_POS2 | <span style="color: green">(-1,680 [-4.4%])</span> <div style='text-align: right'>36,133</div>  |
| leaf |  | 0 | PHANTOM | <span style="color: green">(-26,796 [-4.1%])</span> <div style='text-align: right'>621,699</div>  |
| leaf |  | 0 | PUBLISH | <div style='text-align: right'>36</div>  |
| leaf |  | 0 | SHINTW | <span style="color: green">(-15,950 [-3.1%])</span> <div style='text-align: right'>497,656</div>  |
| leaf |  | 0 | STOREW | <span style="color: green">(-2,842 [-1.1%])</span> <div style='text-align: right'>254,342</div>  |
| leaf |  | 0 | STOREW2 | <span style="color: green">(-13,398 [-3.2%])</span> <div style='text-align: right'>402,548</div>  |
| leaf |  | 0 | SUB | <span style="color: green">(-672 [-0.7%])</span> <div style='text-align: right'>96,735</div>  |

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | <span style="color: green">(-2,329,140 [-3.1%])</span> <div style='text-align: right'>74,010,780</div>  |
| leaf | AccessAdapter<2> |  | 0 | ADD | <span style="color: green">(-49,896 [-7.9%])</span> <div style='text-align: right'>584,826</div>  |
| leaf | AccessAdapter<4> |  | 0 | ADD | <span style="color: green">(-29,484 [-7.9%])</span> <div style='text-align: right'>345,579</div>  |
| leaf | Boundary |  | 0 | ADD | <div style='text-align: right'>767,943</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | BBE4DIV | <div style='text-align: right'>324,360</div>  |
| leaf | AccessAdapter<2> |  | 0 | BBE4DIV | <div style='text-align: right'>161,084</div>  |
| leaf | AccessAdapter<4> |  | 0 | BBE4DIV | <div style='text-align: right'>95,186</div>  |
| leaf | Boundary |  | 0 | BBE4DIV | <div style='text-align: right'>352</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | BBE4MUL | <span style="color: green">(-70,960 [-4.7%])</span> <div style='text-align: right'>1,454,320</div>  |
| leaf | AccessAdapter<2> |  | 0 | BBE4MUL | <span style="color: green">(-36,498 [-3.3%])</span> <div style='text-align: right'>1,079,958</div>  |
| leaf | AccessAdapter<4> |  | 0 | BBE4MUL | <span style="color: green">(-21,567 [-3.3%])</span> <div style='text-align: right'>638,157</div>  |
| leaf | Boundary |  | 0 | BBE4MUL | <div style='text-align: right'>1,037,080</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> |  | 0 | BEQ | <div style='text-align: right'>457,654</div>  |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> |  | 0 | BNE | <span style="color: green">(-991,530 [-3.0%])</span> <div style='text-align: right'>32,225,346</div>  |
| leaf | AccessAdapter<2> |  | 0 | BNE | <div style='text-align: right'>1,540</div>  |
| leaf | AccessAdapter<4> |  | 0 | BNE | <div style='text-align: right'>910</div>  |
| leaf | AccessAdapter<2> |  | 0 | COMP_POS2 | <div style='text-align: right'>749,892</div>  |
| leaf | AccessAdapter<4> |  | 0 | COMP_POS2 | <div style='text-align: right'>443,118</div>  |
| leaf | AccessAdapter<8> |  | 0 | COMP_POS2 | <div style='text-align: right'>289,731</div>  |
| leaf | Arc<BabyBearParameters>, 1> |  | 0 | COMP_POS2 | <div style='text-align: right'>6,420,252</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | DIV | <div style='text-align: right'>5,310</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | FE4ADD | <span style="color: green">(-39,840 [-2.1%])</span> <div style='text-align: right'>1,902,080</div>  |
| leaf | AccessAdapter<2> |  | 0 | FE4ADD | <span style="color: green">(-1,078 [-0.1%])</span> <div style='text-align: right'>1,369,566</div>  |
| leaf | AccessAdapter<4> |  | 0 | FE4ADD | <span style="color: green">(-637 [-0.1%])</span> <div style='text-align: right'>809,289</div>  |
| leaf | Boundary |  | 0 | FE4ADD | <div style='text-align: right'>1,380,324</div>  |
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> |  | 0 | FE4SUB | <span style="color: green">(-10,040 [-1.5%])</span> <div style='text-align: right'>668,920</div>  |
| leaf | AccessAdapter<2> |  | 0 | FE4SUB | <span style="color: green">(-5,852 [-1.1%])</span> <div style='text-align: right'>544,874</div>  |
| leaf | AccessAdapter<4> |  | 0 | FE4SUB | <span style="color: green">(-3,458 [-1.1%])</span> <div style='text-align: right'>321,971</div>  |
| leaf | Boundary |  | 0 | FE4SUB | <div style='text-align: right'>574,816</div>  |
| leaf | AccessAdapter<2> |  | 0 | FRI_REDUCED_OPENING | <span style="color: green">(-14,036 [-3.5%])</span> <div style='text-align: right'>386,672</div>  |
| leaf | AccessAdapter<4> |  | 0 | FRI_REDUCED_OPENING | <span style="color: green">(-8,294 [-3.5%])</span> <div style='text-align: right'>228,488</div>  |
| leaf | FriReducedOpeningAir |  | 0 | FRI_REDUCED_OPENING | <span style="color: green">(-1,714,944 [-4.7%])</span> <div style='text-align: right'>34,825,728</div>  |
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | <span style="color: red">(+14,710 [+1.5%])</span> <div style='text-align: right'>988,400</div>  |
| leaf | AccessAdapter<2> |  | 0 | JAL | <div style='text-align: right'>572</div>  |
| leaf | AccessAdapter<4> |  | 0 | JAL | <div style='text-align: right'>676</div>  |
| leaf | Boundary |  | 0 | JAL | <div style='text-align: right'>11</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | LOADW | <span style="color: green">(-104,632 [-1.2%])</span> <div style='text-align: right'>8,598,479</div>  |
| leaf | AccessAdapter<2> |  | 0 | LOADW | <span style="color: green">(-24,849 [-4.4%])</span> <div style='text-align: right'>542,498</div>  |
| leaf | AccessAdapter<4> |  | 0 | LOADW | <span style="color: green">(-14,677 [-5.1%])</span> <div style='text-align: right'>274,768</div>  |
| leaf | AccessAdapter<8> |  | 0 | LOADW | <div style='text-align: right'>21,607</div>  |
| leaf | Boundary |  | 0 | LOADW | <div style='text-align: right'>382,239</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | LOADW2 | <span style="color: green">(-556,206 [-2.0%])</span> <div style='text-align: right'>26,773,000</div>  |
| leaf | AccessAdapter<2> |  | 0 | LOADW2 | <div style='text-align: right'>59,994</div>  |
| leaf | AccessAdapter<4> |  | 0 | LOADW2 | <div style='text-align: right'>35,451</div>  |
| leaf | AccessAdapter<8> |  | 0 | LOADW2 | <div style='text-align: right'>510</div>  |
| leaf | Boundary |  | 0 | LOADW2 | <div style='text-align: right'>1,408</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | MUL | <span style="color: green">(-42,960 [-0.6%])</span> <div style='text-align: right'>6,814,080</div>  |
| leaf | AccessAdapter<2> |  | 0 | MUL | <span style="color: green">(-726 [-2.4%])</span> <div style='text-align: right'>29,315</div>  |
| leaf | AccessAdapter<4> |  | 0 | MUL | <span style="color: green">(-429 [-2.4%])</span> <div style='text-align: right'>17,342</div>  |
| leaf | Boundary |  | 0 | MUL | <div style='text-align: right'>112,376</div>  |
| leaf | AccessAdapter<2> |  | 0 | PERM_POS2 | <span style="color: green">(-73,920 [-4.2%])</span> <div style='text-align: right'>1,690,128</div>  |
| leaf | AccessAdapter<4> |  | 0 | PERM_POS2 | <span style="color: green">(-43,680 [-4.2%])</span> <div style='text-align: right'>1,000,077</div>  |
| leaf | AccessAdapter<8> |  | 0 | PERM_POS2 | <span style="color: green">(-28,560 [-4.1%])</span> <div style='text-align: right'>660,688</div>  |
| leaf | Arc<BabyBearParameters>, 1> |  | 0 | PERM_POS2 | <div style='text-align: right'>12,574,284</div>  |
| leaf | PhantomAir |  | 0 | PHANTOM | <span style="color: green">(-160,776 [-4.1%])</span> <div style='text-align: right'>3,730,194</div>  |
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> |  | 0 | PUBLISH | <div style='text-align: right'>828</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | SHINTW | <span style="color: green">(-653,950 [-3.1%])</span> <div style='text-align: right'>20,403,896</div>  |
| leaf | AccessAdapter<2> |  | 0 | SHINTW | <div style='text-align: right'>22</div>  |
| leaf | AccessAdapter<4> |  | 0 | SHINTW | <div style='text-align: right'>26</div>  |
| leaf | AccessAdapter<8> |  | 0 | SHINTW | <div style='text-align: right'>17</div>  |
| leaf | Boundary |  | 0 | SHINTW | <span style="color: green">(-175,450 [-3.1%])</span> <div style='text-align: right'>5,468,628</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | STOREW | <span style="color: green">(-116,522 [-1.1%])</span> <div style='text-align: right'>10,428,022</div>  |
| leaf | AccessAdapter<2> |  | 0 | STOREW | <span style="color: green">(-5,203 [-3.3%])</span> <div style='text-align: right'>150,337</div>  |
| leaf | AccessAdapter<4> |  | 0 | STOREW | <span style="color: green">(-3,081 [-3.4%])</span> <div style='text-align: right'>87,724</div>  |
| leaf | AccessAdapter<8> |  | 0 | STOREW | <div style='text-align: right'>1,768</div>  |
| leaf | Boundary |  | 0 | STOREW | <div style='text-align: right'>868,362</div>  |
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | STOREW2 | <span style="color: green">(-549,318 [-3.2%])</span> <div style='text-align: right'>16,504,468</div>  |
| leaf | AccessAdapter<2> |  | 0 | STOREW2 | <span style="color: green">(-73,920 [-4.3%])</span> <div style='text-align: right'>1,652,288</div>  |
| leaf | AccessAdapter<4> |  | 0 | STOREW2 | <span style="color: green">(-43,680 [-4.3%])</span> <div style='text-align: right'>977,717</div>  |
| leaf | AccessAdapter<8> |  | 0 | STOREW2 | <span style="color: green">(-28,560 [-4.8%])</span> <div style='text-align: right'>567,341</div>  |
| leaf | Boundary |  | 0 | STOREW2 | <div style='text-align: right'>857,406</div>  |
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | SUB | <span style="color: green">(-20,160 [-0.7%])</span> <div style='text-align: right'>2,902,050</div>  |
| leaf | AccessAdapter<2> |  | 0 | SUB | <div style='text-align: right'>78,793</div>  |
| leaf | AccessAdapter<4> |  | 0 | SUB | <div style='text-align: right'>93,119</div>  |
| leaf | Boundary |  | 0 | SUB | <div style='text-align: right'>15,180</div>  |

| group | idx | segment | total_cycles | trace_gen_time_ms |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | <span style="color: green">(-201,186 [-2.7%])</span> <div style='text-align: right'>7,120,336</div>  | <span style="color: green">(-506.0 [-25.8%])</span> <div style='text-align: right'>1,456.0</div>  |

| group | air_name | idx | cells | main_cols | perm_cols | prep_cols | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | ProgramAir | 0 | <div style='text-align: right'>9,437,184</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>524,288</div>  |
| leaf | VmConnectorAir | 0 | <div style='text-align: right'>24</div>  | <div style='text-align: right'>4</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>2</div>  |
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | <div style='text-align: right'>2,496</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>64</div>  |
| leaf | VolatileBoundaryAir | 0 | <span style="color: green">(-19,922,944 [-50.0%])</span> <div style='text-align: right'>19,922,944</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>8</div>  |  | <span style="color: green">(-1,048,576 [-50.0%])</span> <div style='text-align: right'>1,048,576</div>  |
| leaf | AccessAdapterAir<2> | 0 | <div style='text-align: right'>56,623,104</div>  | <div style='text-align: right'>11</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| leaf | AccessAdapterAir<4> | 0 | <div style='text-align: right'>30,408,704</div>  | <div style='text-align: right'>13</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>1,048,576</div>  |
| leaf | AccessAdapterAir<8> | 0 | <div style='text-align: right'>4,325,376</div>  | <div style='text-align: right'>17</div>  | <div style='text-align: right'>16</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | <div style='text-align: right'>25,165,824</div>  | <div style='text-align: right'>348</div>  | <div style='text-align: right'>36</div>  |  | <div style='text-align: right'>65,536</div>  |
| leaf | FriReducedOpeningAir | 0 | <div style='text-align: right'>146,800,640</div>  | <div style='text-align: right'>64</div>  | <div style='text-align: right'>76</div>  |  | <div style='text-align: right'>1,048,576</div>  |
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | <div style='text-align: right'>7,864,320</div>  | <div style='text-align: right'>40</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | <div style='text-align: right'>209,715,200</div>  | <div style='text-align: right'>30</div>  | <div style='text-align: right'>20</div>  |  | <div style='text-align: right'>4,194,304</div>  |
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | <div style='text-align: right'>2,883,584</div>  | <div style='text-align: right'>10</div>  | <div style='text-align: right'>12</div>  |  | <div style='text-align: right'>131,072</div>  |
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | <div style='text-align: right'>106,954,752</div>  | <div style='text-align: right'>23</div>  | <div style='text-align: right'>28</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | <div style='text-align: right'>136,314,880</div>  | <div style='text-align: right'>41</div>  | <div style='text-align: right'>24</div>  |  | <div style='text-align: right'>2,097,152</div>  |
| leaf | PhantomAir | 0 | <div style='text-align: right'>14,680,064</div>  | <div style='text-align: right'>6</div>  | <div style='text-align: right'>8</div>  |  | <div style='text-align: right'>1,048,576</div>  |
| leaf | VariableRangeCheckerAir | 0 | <div style='text-align: right'>2,359,296</div>  | <div style='text-align: right'>1</div>  | <div style='text-align: right'>8</div>  | <div style='text-align: right'>2</div>  | <div style='text-align: right'>262,144</div>  |

</details>



<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c/regex-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/ecfbe293075fbe8a2c20ebce8f99bb90ecf5da5c

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12451828017)