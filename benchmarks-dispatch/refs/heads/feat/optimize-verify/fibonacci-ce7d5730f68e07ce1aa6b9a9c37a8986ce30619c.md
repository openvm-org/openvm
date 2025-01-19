| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+5 [+23.5%])</span> 23.97 | <span style='color: red'>(+5 [+23.5%])</span> 23.97 |
| fibonacci_program | <span style='color: red'>(+5 [+74.8%])</span> 10.66 | <span style='color: red'>(+5 [+74.8%])</span> 10.66 |
| leaf |  13.31 |  13.31 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4560 [+74.8%])</span> 10,657 | <span style='color: red'>(+4560 [+74.8%])</span> 10,657 | <span style='color: red'>(+4560 [+74.8%])</span> 10,657 | <span style='color: red'>(+4560 [+74.8%])</span> 10,657 |
| `main_cells_used     ` |  51,503,940 |  51,503,940 |  51,503,940 |  51,503,940 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+4541 [+1469.6%])</span> 4,850 | <span style='color: red'>(+4541 [+1469.6%])</span> 4,850 | <span style='color: red'>(+4541 [+1469.6%])</span> 4,850 | <span style='color: red'>(+4541 [+1469.6%])</span> 4,850 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+10 [+1.2%])</span> 829 | <span style='color: red'>(+10 [+1.2%])</span> 829 | <span style='color: red'>(+10 [+1.2%])</span> 829 | <span style='color: red'>(+10 [+1.2%])</span> 829 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+9 [+0.2%])</span> 4,978 | <span style='color: red'>(+9 [+0.2%])</span> 4,978 | <span style='color: red'>(+9 [+0.2%])</span> 4,978 | <span style='color: red'>(+9 [+0.2%])</span> 4,978 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+11 [+1.4%])</span> 817 | <span style='color: red'>(+11 [+1.4%])</span> 817 | <span style='color: red'>(+11 [+1.4%])</span> 817 | <span style='color: red'>(+11 [+1.4%])</span> 817 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-2.2%])</span> 174 | <span style='color: green'>(-4 [-2.2%])</span> 174 | <span style='color: green'>(-4 [-2.2%])</span> 174 | <span style='color: green'>(-4 [-2.2%])</span> 174 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+13 [+0.8%])</span> 1,612 | <span style='color: red'>(+13 [+0.8%])</span> 1,612 | <span style='color: red'>(+13 [+0.8%])</span> 1,612 | <span style='color: red'>(+13 [+0.8%])</span> 1,612 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-4 [-0.5%])</span> 783 | <span style='color: green'>(-4 [-0.5%])</span> 783 | <span style='color: green'>(-4 [-0.5%])</span> 783 | <span style='color: green'>(-4 [-0.5%])</span> 783 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.4%])</span> 508 | <span style='color: red'>(+2 [+0.4%])</span> 508 | <span style='color: red'>(+2 [+0.4%])</span> 508 | <span style='color: red'>(+2 [+0.4%])</span> 508 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-0.7%])</span> 1,082 | <span style='color: green'>(-8 [-0.7%])</span> 1,082 | <span style='color: green'>(-8 [-0.7%])</span> 1,082 | <span style='color: green'>(-8 [-0.7%])</span> 1,082 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  13,313 |  13,313 |  13,313 |  13,313 |
| `main_cells_used     ` | <span style='color: green'>(-37118793 [-33.5%])</span> 73,592,345 | <span style='color: green'>(-37118793 [-33.5%])</span> 73,592,345 | <span style='color: green'>(-37118793 [-33.5%])</span> 73,592,345 | <span style='color: green'>(-37118793 [-33.5%])</span> 73,592,345 |
| `total_cycles        ` | <span style='color: green'>(-1276539 [-41.4%])</span> 1,807,926 | <span style='color: green'>(-1276539 [-41.4%])</span> 1,807,926 | <span style='color: green'>(-1276539 [-41.4%])</span> 1,807,926 | <span style='color: green'>(-1276539 [-41.4%])</span> 1,807,926 |
| `execute_time_ms     ` | <span style='color: red'>(+5421 [+936.3%])</span> 6,000 | <span style='color: red'>(+5421 [+936.3%])</span> 6,000 | <span style='color: red'>(+5421 [+936.3%])</span> 6,000 | <span style='color: red'>(+5421 [+936.3%])</span> 6,000 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-548 [-29.9%])</span> 1,282 | <span style='color: green'>(-548 [-29.9%])</span> 1,282 | <span style='color: green'>(-548 [-29.9%])</span> 1,282 | <span style='color: green'>(-548 [-29.9%])</span> 1,282 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-4876 [-44.7%])</span> 6,031 | <span style='color: green'>(-4876 [-44.7%])</span> 6,031 | <span style='color: green'>(-4876 [-44.7%])</span> 6,031 | <span style='color: green'>(-4876 [-44.7%])</span> 6,031 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-864 [-43.9%])</span> 1,105 | <span style='color: green'>(-864 [-43.9%])</span> 1,105 | <span style='color: green'>(-864 [-43.9%])</span> 1,105 | <span style='color: green'>(-864 [-43.9%])</span> 1,105 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-105 [-44.7%])</span> 130 | <span style='color: green'>(-105 [-44.7%])</span> 130 | <span style='color: green'>(-105 [-44.7%])</span> 130 | <span style='color: green'>(-105 [-44.7%])</span> 130 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-788 [-40.0%])</span> 1,181 | <span style='color: green'>(-788 [-40.0%])</span> 1,181 | <span style='color: green'>(-788 [-40.0%])</span> 1,181 | <span style='color: green'>(-788 [-40.0%])</span> 1,181 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1200 [-47.7%])</span> 1,318 | <span style='color: green'>(-1200 [-47.7%])</span> 1,318 | <span style='color: green'>(-1200 [-47.7%])</span> 1,318 | <span style='color: green'>(-1200 [-47.7%])</span> 1,318 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-870 [-47.1%])</span> 978 | <span style='color: green'>(-870 [-47.1%])</span> 978 | <span style='color: green'>(-870 [-47.1%])</span> 978 | <span style='color: green'>(-870 [-47.1%])</span> 978 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-1048 [-44.3%])</span> 1,317 | <span style='color: green'>(-1048 [-44.3%])</span> 1,317 | <span style='color: green'>(-1048 [-44.3%])</span> 1,317 | <span style='color: green'>(-1048 [-44.3%])</span> 1,317 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 340 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 
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

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,704 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 31,832 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 24,679 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,543 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 606,119 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,266 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 478,768 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,323 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,235,300 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 19,520 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 85,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 290,200 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 21,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 18,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 829,560 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 467,700 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,225,170 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 2,768,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,121,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 864,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 6,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 790,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 96,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 76,710 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 157,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 282,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 282,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 355,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 236,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,220,970 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,095,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 113,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 15,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 183,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 932,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 40,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 297,870 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 5,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 33,270 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 21,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 355,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 238,020 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 485,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 161,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 10,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 12,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 174,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 30,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 8,753,550 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 572,850 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 3,123,150 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 236,900 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,741,100 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,416,000 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 686,052 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 442,782 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 497,320 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 248,560 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 2,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 341,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 61,040 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 132,240 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,547,904 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 41,328 | 
| leaf | PhantomAir | CT-initialize-opening-vals | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-sample-challenges | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-setup-and-permute-openings | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 64,008 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 139,272 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 21,147 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 20,349 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 6,502,104 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 32,401,620 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 144 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 11,100,074 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 106 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 2,600,104 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,600,130 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 96 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 64 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,800,018 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 162 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 78 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 364 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 520 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 600 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 168 | 
| fibonacci_program | PhantomAir |  | PHANTOM | 0 | 12 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 321,606 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 39,489 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 885,947 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 443,600 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 33,201 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 32,068 | 
| leaf | AccessAdapter<2> | 0 | 161,598 | 
| leaf | AccessAdapter<4> | 0 | 76,516 | 
| leaf | AccessAdapter<8> | 0 | 478 | 
| leaf | Boundary | 0 | 337,555 | 
| leaf | FriReducedOpeningAir | 0 | 117,936 | 
| leaf | PhantomAir | 0 | 45,365 | 
| leaf | ProgramChip | 0 | 75,702 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 27,740 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 900,054 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 300,002 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 2 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 200,009 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 5 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 100,010 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 3 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 13 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 28 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 9 | 
| fibonacci_program | AccessAdapter<8> | 0 | 36 | 
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | 228 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fibonacci_program | Boundary | 0 | 36 | 
| fibonacci_program | Merkle | 0 | 280 | 
| fibonacci_program | PhantomAir | 0 | 2 | 
| fibonacci_program | ProgramChip | 0 | 3,275 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 12,433 | 
| leaf | AddEFFI | 0 | ADD | 704 | 
| leaf | AddEFI | 0 | ADD | 616 | 
| leaf | AddEI | 0 | ADD | 27,652 | 
| leaf | AddF | 0 | ADD | 1,333 | 
| leaf | AddFI | 0 | ADD | 15,590 | 
| leaf | AddV | 0 | ADD | 40,839 | 
| leaf | AddVI | 0 | ADD | 92,288 | 
| leaf | Alloc | 0 | ADD | 104,056 | 
| leaf | Alloc | 0 | MUL | 28,816 | 
| leaf | AssertEqE | 0 | BNE | 248 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 1,384 | 
| leaf | AssertEqV | 0 | BNE | 1,073 | 
| leaf | AssertEqVI | 0 | BNE | 241 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 6,888 | 
| leaf | CT-initialize-opening-vals | 0 | PHANTOM | 84 | 
| leaf | CT-sample-challenges | 0 | PHANTOM | 2 | 
| leaf | CT-setup-and-permute-openings | 0 | PHANTOM | 672 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 10,668 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 1 | 
| leaf | DivE | 0 | BBE4DIV | 6,214 | 
| leaf | DivEIN | 0 | ADD | 216 | 
| leaf | DivEIN | 0 | BBE4DIV | 54 | 
| leaf | DivF | 0 | DIV | 7,392 | 
| leaf | DivFIN | 0 | DIV | 128 | 
| leaf | For | 0 | ADD | 26,353 | 
| leaf | For | 0 | BNE | 26,353 | 
| leaf | For | 0 | JAL | 1,952 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,334 | 
| leaf | HintBitsF | 0 | PHANTOM | 43 | 
| leaf | HintInputVec | 0 | PHANTOM | 23,212 | 
| leaf | IfEq | 0 | BNE | 142 | 
| leaf | IfEqI | 0 | BNE | 20,816 | 
| leaf | IfEqI | 0 | JAL | 8,515 | 
| leaf | IfNe | 0 | BEQ | 143 | 
| leaf | IfNe | 0 | JAL | 1 | 
| leaf | IfNeI | 0 | BEQ | 101 | 
| leaf | ImmE | 0 | ADD | 3,208 | 
| leaf | ImmF | 0 | ADD | 2,557 | 
| leaf | ImmV | 0 | ADD | 5,264 | 
| leaf | LoadE | 0 | ADD | 9,408 | 
| leaf | LoadE | 0 | LOADW | 20,178 | 
| leaf | LoadE | 0 | MUL | 9,408 | 
| leaf | LoadF | 0 | ADD | 11,848 | 
| leaf | LoadF | 0 | LOADW | 22,914 | 
| leaf | LoadF | 0 | MUL | 7,883 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 40,699 | 
| leaf | LoadV | 0 | LOADW | 124,926 | 
| leaf | LoadV | 0 | MUL | 36,526 | 
| leaf | MulE | 0 | BBE4MUL | 8,535 | 
| leaf | MulEF | 0 | MUL | 3,792 | 
| leaf | MulEFI | 0 | MUL | 500 | 
| leaf | MulEI | 0 | ADD | 6,104 | 
| leaf | MulEI | 0 | BBE4MUL | 1,526 | 
| leaf | MulF | 0 | MUL | 31,083 | 
| leaf | MulFI | 0 | MUL | 1,353 | 
| leaf | MulVI | 0 | MUL | 9,929 | 
| leaf | NegE | 0 | MUL | 172 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 53 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 51 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 7,728 | 
| leaf | StoreE | 0 | MUL | 7,728 | 
| leaf | StoreE | 0 | STOREW | 13,023 | 
| leaf | StoreF | 0 | ADD | 1,109 | 
| leaf | StoreF | 0 | MUL | 732 | 
| leaf | StoreF | 0 | STOREW | 9,476 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 229,644 | 
| leaf | StoreV | 0 | ADD | 11,848 | 
| leaf | StoreV | 0 | MUL | 7,934 | 
| leaf | StoreV | 0 | STOREW | 56,640 | 
| leaf | SubE | 0 | FE4SUB | 3,306 | 
| leaf | SubEF | 0 | ADD | 16,182 | 
| leaf | SubEF | 0 | SUB | 5,394 | 
| leaf | SubEFI | 0 | ADD | 344 | 
| leaf | SubEI | 0 | ADD | 432 | 
| leaf | SubFI | 0 | SUB | 1,333 | 
| leaf | SubV | 0 | SUB | 5,816 | 
| leaf | SubVI | 0 | SUB | 1,000 | 
| leaf | SubVIN | 0 | SUB | 840 | 
| leaf | UnsafeCastVF | 0 | ADD | 20 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 291,785 | 
| leaf | ZipFor | 0 | BNE | 271,100 | 
| leaf | ZipFor | 0 | JAL | 29,020 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program |  | ADD | 0 | 900,045 | 
| fibonacci_program |  | AND | 0 | 2 | 
| fibonacci_program |  | AUIPC | 0 | 9 | 
| fibonacci_program |  | BEQ | 0 | 100,004 | 
| fibonacci_program |  | BGEU | 0 | 3 | 
| fibonacci_program |  | BLTU | 0 | 2 | 
| fibonacci_program |  | BNE | 0 | 100,005 | 
| fibonacci_program |  | HINT_STOREW | 0 | 3 | 
| fibonacci_program |  | JAL | 0 | 100,001 | 
| fibonacci_program |  | JALR | 0 | 13 | 
| fibonacci_program |  | LOADW | 0 | 13 | 
| fibonacci_program |  | LUI | 0 | 9 | 
| fibonacci_program |  | OR | 0 | 1 | 
| fibonacci_program |  | PHANTOM | 0 | 2 | 
| fibonacci_program |  | SLL | 0 | 2 | 
| fibonacci_program |  | SLTU | 0 | 300,002 | 
| fibonacci_program |  | STOREW | 0 | 15 | 
| fibonacci_program |  | SUB | 0 | 4 | 
| fibonacci_program |  | XOR | 0 | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,282 | 13,313 | 1,807,926 | 188,730,328 | 6,031 | 1,318 | 978 | 1,181 | 1,317 | 1,105 | 73,592,345 | 130 | 6,000 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 829 | 10,657 | 1,500,137 | 197,453,854 | 4,978 | 783 | 508 | 1,612 | 1,082 | 817 | 51,503,940 | 174 | 4,850 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/fibonacci-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12848892486)
