| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+69 [+185.1%])</span> 105.64 | <span style='color: red'>(+69 [+185.1%])</span> 105.64 |
| regex_program | <span style='color: red'>(+59 [+329.1%])</span> 76.66 | <span style='color: red'>(+59 [+329.1%])</span> 76.66 |
| leaf | <span style='color: red'>(+10 [+51.1%])</span> 28.99 | <span style='color: red'>(+10 [+51.1%])</span> 28.99 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+58792 [+329.1%])</span> 76,655 | <span style='color: red'>(+58792 [+329.1%])</span> 76,655 | <span style='color: red'>(+58792 [+329.1%])</span> 76,655 | <span style='color: red'>(+58792 [+329.1%])</span> 76,655 |
| `main_cells_used     ` | <span style='color: red'>(+169837 [+0.1%])</span> 165,180,746 | <span style='color: red'>(+169837 [+0.1%])</span> 165,180,746 | <span style='color: red'>(+169837 [+0.1%])</span> 165,180,746 | <span style='color: red'>(+169837 [+0.1%])</span> 165,180,746 |
| `total_cycles        ` | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 |
| `execute_time_ms     ` | <span style='color: red'>(+58646 [+6015.0%])</span> 59,621 | <span style='color: red'>(+58646 [+6015.0%])</span> 59,621 | <span style='color: red'>(+58646 [+6015.0%])</span> 59,621 | <span style='color: red'>(+58646 [+6015.0%])</span> 59,621 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+41 [+1.3%])</span> 3,099 | <span style='color: red'>(+41 [+1.3%])</span> 3,099 | <span style='color: red'>(+41 [+1.3%])</span> 3,099 | <span style='color: red'>(+41 [+1.3%])</span> 3,099 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+105 [+0.8%])</span> 13,935 | <span style='color: red'>(+105 [+0.8%])</span> 13,935 | <span style='color: red'>(+105 [+0.8%])</span> 13,935 | <span style='color: red'>(+105 [+0.8%])</span> 13,935 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+5 [+0.2%])</span> 2,405 | <span style='color: red'>(+5 [+0.2%])</span> 2,405 | <span style='color: red'>(+5 [+0.2%])</span> 2,405 | <span style='color: red'>(+5 [+0.2%])</span> 2,405 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-0.6%])</span> 495 | <span style='color: green'>(-3 [-0.6%])</span> 495 | <span style='color: green'>(-3 [-0.6%])</span> 495 | <span style='color: green'>(-3 [-0.6%])</span> 495 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+32 [+0.6%])</span> 5,225 | <span style='color: red'>(+32 [+0.6%])</span> 5,225 | <span style='color: red'>(+32 [+0.6%])</span> 5,225 | <span style='color: red'>(+32 [+0.6%])</span> 5,225 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+19 [+1.1%])</span> 1,819 | <span style='color: red'>(+19 [+1.1%])</span> 1,819 | <span style='color: red'>(+19 [+1.1%])</span> 1,819 | <span style='color: red'>(+19 [+1.1%])</span> 1,819 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+51 [+4.2%])</span> 1,266 | <span style='color: red'>(+51 [+4.2%])</span> 1,266 | <span style='color: red'>(+51 [+4.2%])</span> 1,266 | <span style='color: red'>(+51 [+4.2%])</span> 1,266 |
| `pcs_opening_time_ms ` |  2,722 |  2,722 |  2,722 |  2,722 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+9798 [+51.1%])</span> 28,990 | <span style='color: red'>(+9798 [+51.1%])</span> 28,990 | <span style='color: red'>(+9798 [+51.1%])</span> 28,990 | <span style='color: red'>(+9798 [+51.1%])</span> 28,990 |
| `main_cells_used     ` | <span style='color: red'>(+169256 [+0.1%])</span> 163,454,497 | <span style='color: red'>(+169256 [+0.1%])</span> 163,454,497 | <span style='color: red'>(+169256 [+0.1%])</span> 163,454,497 | <span style='color: red'>(+169256 [+0.1%])</span> 163,454,497 |
| `total_cycles        ` | <span style='color: red'>(+27849 [+0.9%])</span> 3,056,532 | <span style='color: red'>(+27849 [+0.9%])</span> 3,056,532 | <span style='color: red'>(+27849 [+0.9%])</span> 3,056,532 | <span style='color: red'>(+27849 [+0.9%])</span> 3,056,532 |
| `execute_time_ms     ` | <span style='color: red'>(+9784 [+1418.0%])</span> 10,474 | <span style='color: red'>(+9784 [+1418.0%])</span> 10,474 | <span style='color: red'>(+9784 [+1418.0%])</span> 10,474 | <span style='color: red'>(+9784 [+1418.0%])</span> 10,474 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+57 [+2.0%])</span> 2,891 | <span style='color: red'>(+57 [+2.0%])</span> 2,891 | <span style='color: red'>(+57 [+2.0%])</span> 2,891 | <span style='color: red'>(+57 [+2.0%])</span> 2,891 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-43 [-0.3%])</span> 15,625 | <span style='color: green'>(-43 [-0.3%])</span> 15,625 | <span style='color: green'>(-43 [-0.3%])</span> 15,625 | <span style='color: green'>(-43 [-0.3%])</span> 15,625 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+13 [+0.5%])</span> 2,853 | <span style='color: red'>(+13 [+0.5%])</span> 2,853 | <span style='color: red'>(+13 [+0.5%])</span> 2,853 | <span style='color: red'>(+13 [+0.5%])</span> 2,853 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-8 [-2.2%])</span> 353 | <span style='color: green'>(-8 [-2.2%])</span> 353 | <span style='color: green'>(-8 [-2.2%])</span> 353 | <span style='color: green'>(-8 [-2.2%])</span> 353 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+21 [+0.6%])</span> 3,484 | <span style='color: red'>(+21 [+0.6%])</span> 3,484 | <span style='color: red'>(+21 [+0.6%])</span> 3,484 | <span style='color: red'>(+21 [+0.6%])</span> 3,484 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-62 [-1.5%])</span> 4,005 | <span style='color: green'>(-62 [-1.5%])</span> 4,005 | <span style='color: green'>(-62 [-1.5%])</span> 4,005 | <span style='color: green'>(-62 [-1.5%])</span> 4,005 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-11 [-0.5%])</span> 2,292 | <span style='color: green'>(-11 [-0.5%])</span> 2,292 | <span style='color: green'>(-11 [-0.5%])</span> 2,292 | <span style='color: green'>(-11 [-0.5%])</span> 2,292 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3 [+0.1%])</span> 2,634 | <span style='color: red'>(+3 [+0.1%])</span> 2,634 | <span style='color: red'>(+3 [+0.1%])</span> 2,634 | <span style='color: red'>(+3 [+0.1%])</span> 2,634 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 639 | 43 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 176 | 590 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 23 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 
| regex_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,571 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| regex_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 659,364 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 72,312 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 36,110 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 7,337 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,312 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 566,191 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,381 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 3,105 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 13,254,785 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 92,350 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 30 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 348,480 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 23,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 25,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,780,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 94,650 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 490,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,379,730 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 3,119,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,443,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 950,910 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 4,620 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 9,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 5,310 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 492,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 139,110 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 171,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 356,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 356,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 464,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 310,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,582,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,420,020 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 123,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 240,600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 599,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,024,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 90,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 373,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 12,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 303,660 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 303,660 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 31,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 19,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 471,570 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 319,290 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 645,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 215,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 286,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 18,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 89,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 212,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 31,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 26,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 4,110 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 18,206,820 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 755,850 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 3,918,575 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 233,575 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 12,472,550 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,646,750 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,343,680 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 533,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,902,280 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 321,360 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 3,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,546,280 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 199,960 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 668,920 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 34,825,728 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 55,440 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 85,176 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,584 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 918 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 154,200 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,773 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 22,743 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,926,852 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 17,294,256 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 36,618,768 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 1,912,104 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 847,584 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 1,532,952 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 344,232 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLT | 0 | 185 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 1,237,798 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 11,318,044 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRA | 0 | 53 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 269,770 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 4,880,538 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,691,832 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGE | 0 | 9,408 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 3,890,944 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 164,512 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 2,273,600 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,190,322 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 800,964 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 331,942 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 3,652,404 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 24,255 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADH | 0 | 280 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 1,093,200 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADHU | 0 | 3,800 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 45,715,640 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 509,480 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREH | 0 | 402,960 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 30,916,880 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 6,498 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 9,516 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 1,614,697 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 830,676 | 
| regex_program | KeccakVmAir |  | KECCAK256 | 0 | 75,936 | 
| regex_program | PhantomAir |  | PHANTOM | 0 | 1,734 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 356 | 399 | 49,479,680 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 40 | 7,864,320 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 0 | 64 |  | 24 | 11 | 2,240 | 
| regex_program | AccessAdapterAir<4> | 0 | 32 |  | 24 | 13 | 1,184 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 24 | 17 | 5,373,952 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 1,288 | 3,164 | 142,464 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PhantomAir | 0 | 512 |  | 12 | 6 | 9,216 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 80 | 36 | 243,269,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 48 | 26 | 38,797,312 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 56 | 32 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 16,384 |  | 36 | 26 | 1,015,808 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 76 | 35 | 113,664 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 72 | 40 | 234,881,024 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 104 | 57 | 20,608 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 100 | 39 | 35,584 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 80 | 31 | 7,274,496 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 28 | 21 | 3,211,264 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 635,044 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 44,087 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,383,449 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 761,092 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 55,210 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 116,045 | 
| leaf | AccessAdapter<2> | 0 | 720,870 | 
| leaf | AccessAdapter<4> | 0 | 354,934 | 
| leaf | AccessAdapter<8> | 0 | 336 | 
| leaf | Boundary | 0 | 990,486 | 
| leaf | FriReducedOpeningAir | 0 | 544,152 | 
| leaf | PhantomAir | 0 | 53,169 | 
| leaf | ProgramChip | 0 | 250,789 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 55,776 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,145,990 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 33,459 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 218,639 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 291,245 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 198,077 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 110,627 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 12,767 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 130,443 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 701 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 1,966,049 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 114 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 244 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 52,087 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 39,557 | 
| regex_program | AccessAdapter<2> | 0 | 42 | 
| regex_program | AccessAdapter<4> | 0 | 22 | 
| regex_program | AccessAdapter<8> | 0 | 69,206 | 
| regex_program | Arc<BabyBearParameters>, 1> | 0 | 13,953 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| regex_program | Boundary | 0 | 69,206 | 
| regex_program | KeccakVmAir | 0 | 24 | 
| regex_program | Merkle | 0 | 70,392 | 
| regex_program | PhantomAir | 0 | 289 | 
| regex_program | ProgramChip | 0 | 89,891 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 
| regex_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 47,557 | 
| leaf | AddEFFI | 0 | ADD | 776 | 
| leaf | AddEFI | 0 | ADD | 860 | 
| leaf | AddEI | 0 | ADD | 92,672 | 
| leaf | AddF | 0 | ADD | 3,155 | 
| leaf | AddFI | 0 | ADD | 16,362 | 
| leaf | AddV | 0 | ADD | 45,991 | 
| leaf | AddVI | 0 | ADD | 103,988 | 
| leaf | Alloc | 0 | ADD | 114,794 | 
| leaf | Alloc | 0 | MUL | 31,697 | 
| leaf | AssertEqE | 0 | BNE | 28,668 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 3,144 | 
| leaf | AssertEqV | 0 | BNE | 1,570 | 
| leaf | AssertEqVI | 0 | BNE | 319 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 9,240 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 14,196 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,764 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 154 | 
| leaf | DivE | 0 | BBE4DIV | 8,034 | 
| leaf | DivEIN | 0 | ADD | 300 | 
| leaf | DivEIN | 0 | BBE4DIV | 75 | 
| leaf | DivF | 0 | DIV | 7,728 | 
| leaf | DivFIN | 0 | DIV | 177 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,098 | 
| leaf | HintBitsF | 0 | PHANTOM | 153 | 
| leaf | HintInputVec | 0 | PHANTOM | 25,700 | 
| leaf | IfEq | 0 | BNE | 144 | 
| leaf | IfEqI | 0 | BNE | 24,617 | 
| leaf | IfEqI | 0 | JAL | 9,235 | 
| leaf | IfNe | 0 | BEQ | 147 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNeI | 0 | BEQ | 135 | 
| leaf | ImmE | 0 | ADD | 16,412 | 
| leaf | ImmF | 0 | ADD | 4,637 | 
| leaf | ImmV | 0 | ADD | 5,721 | 
| leaf | LoadE | 0 | ADD | 11,886 | 
| leaf | LoadE | 0 | LOADW | 39,520 | 
| leaf | LoadE | 0 | MUL | 11,886 | 
| leaf | LoadF | 0 | ADD | 15,492 | 
| leaf | LoadF | 0 | LOADW | 30,234 | 
| leaf | LoadF | 0 | MUL | 10,347 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 52,741 | 
| leaf | LoadV | 0 | LOADW | 156,743 | 
| leaf | LoadV | 0 | MUL | 47,334 | 
| leaf | MulE | 0 | BBE4MUL | 38,657 | 
| leaf | MulEF | 0 | MUL | 4,128 | 
| leaf | MulEFI | 0 | MUL | 8,020 | 
| leaf | MulEI | 0 | ADD | 19,996 | 
| leaf | MulEI | 0 | BBE4MUL | 4,999 | 
| leaf | MulF | 0 | MUL | 34,147 | 
| leaf | MulFI | 0 | MUL | 3,010 | 
| leaf | MulVI | 0 | MUL | 12,444 | 
| leaf | NegE | 0 | MUL | 428 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 57 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 10,122 | 
| leaf | StoreE | 0 | MUL | 10,122 | 
| leaf | StoreE | 0 | STOREW | 15,690 | 
| leaf | StoreF | 0 | ADD | 1,056 | 
| leaf | StoreF | 0 | MUL | 636 | 
| leaf | StoreF | 0 | STOREW | 9,343 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 498,902 | 
| leaf | StoreV | 0 | ADD | 15,719 | 
| leaf | StoreV | 0 | MUL | 10,643 | 
| leaf | StoreV | 0 | STOREW | 65,870 | 
| leaf | SubE | 0 | FE4SUB | 16,723 | 
| leaf | SubEF | 0 | ADD | 21,516 | 
| leaf | SubEF | 0 | SUB | 7,172 | 
| leaf | SubEFI | 0 | ADD | 9,544 | 
| leaf | SubEI | 0 | ADD | 600 | 
| leaf | SubFI | 0 | SUB | 2,983 | 
| leaf | SubV | 0 | SUB | 7,083 | 
| leaf | SubVI | 0 | SUB | 1,053 | 
| leaf | SubVIN | 0 | SUB | 882 | 
| leaf | UnsafeCastVF | 0 | ADD | 137 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 882 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 606,894 | 
| leaf | ZipFor | 0 | BNE | 576,295 | 
| leaf | ZipFor | 0 | JAL | 34,848 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| regex_program |  | ADD | 0 | 1,017,188 | 
| regex_program |  | AND | 0 | 53,114 | 
| regex_program |  | AUIPC | 0 | 39,557 | 
| regex_program |  | BEQ | 0 | 187,713 | 
| regex_program |  | BGE | 0 | 294 | 
| regex_program |  | BGEU | 0 | 121,592 | 
| regex_program |  | BLT | 0 | 5,141 | 
| regex_program |  | BLTU | 0 | 71,050 | 
| regex_program |  | BNE | 0 | 103,532 | 
| regex_program |  | DIVU | 0 | 114 | 
| regex_program |  | HINT_STOREW | 0 | 12,767 | 
| regex_program |  | JAL | 0 | 66,129 | 
| regex_program |  | JALR | 0 | 130,443 | 
| regex_program |  | KECCAK256 | 0 | 1 | 
| regex_program |  | LOADB | 0 | 693 | 
| regex_program |  | LOADBU | 0 | 27,330 | 
| regex_program |  | LOADH | 0 | 8 | 
| regex_program |  | LOADHU | 0 | 95 | 
| regex_program |  | LOADW | 0 | 1,142,891 | 
| regex_program |  | LUI | 0 | 44,498 | 
| regex_program |  | MUL | 0 | 52,087 | 
| regex_program |  | MULHU | 0 | 244 | 
| regex_program |  | OR | 0 | 23,544 | 
| regex_program |  | PHANTOM | 0 | 289 | 
| regex_program |  | SLL | 0 | 213,548 | 
| regex_program |  | SLT | 0 | 5 | 
| regex_program |  | SLTU | 0 | 33,454 | 
| regex_program |  | SRA | 0 | 1 | 
| regex_program |  | SRL | 0 | 5,090 | 
| regex_program |  | STOREB | 0 | 12,737 | 
| regex_program |  | STOREH | 0 | 10,074 | 
| regex_program |  | STOREW | 0 | 772,922 | 
| regex_program |  | SUB | 0 | 42,582 | 
| regex_program |  | XOR | 0 | 9,562 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 2,891 | 28,990 | 3,056,532 | 503,925,720 | 15,625 | 4,005 | 2,292 | 3,484 | 2,634 | 2,853 | 163,454,497 | 353 | 10,474 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,099 | 76,655 | 4,200,289 | 632,452,480 | 13,935 | 1,819 | 1,266 | 5,225 | 2,722 | 2,405 | 165,180,746 | 495 | 59,621 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87/regex-765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/765f1a740f97c87f0787edf4e2c7ccb1dfc2ba87

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12915781220)
