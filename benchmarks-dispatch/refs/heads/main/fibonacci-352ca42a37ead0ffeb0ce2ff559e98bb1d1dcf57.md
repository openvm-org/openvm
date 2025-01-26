| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+10 [+82.8%])</span> 23 | <span style='color: red'>(+10 [+82.8%])</span> 23 |
| fibonacci_program | <span style='color: red'>(+5 [+82.9%])</span> 10.15 | <span style='color: red'>(+5 [+82.9%])</span> 10.15 |
| leaf | <span style='color: red'>(+6 [+82.8%])</span> 12.85 | <span style='color: red'>(+6 [+82.8%])</span> 12.85 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4601 [+82.9%])</span> 10,151 | <span style='color: red'>(+4601 [+82.9%])</span> 10,151 | <span style='color: red'>(+4601 [+82.9%])</span> 10,151 | <span style='color: red'>(+4601 [+82.9%])</span> 10,151 |
| `main_cells_used     ` |  51,486,676 |  51,486,676 |  51,486,676 |  51,486,676 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+4598 [+1464.3%])</span> 4,912 | <span style='color: red'>(+4598 [+1464.3%])</span> 4,912 | <span style='color: red'>(+4598 [+1464.3%])</span> 4,912 | <span style='color: red'>(+4598 [+1464.3%])</span> 4,912 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-6 [-0.7%])</span> 831 | <span style='color: green'>(-6 [-0.7%])</span> 831 | <span style='color: green'>(-6 [-0.7%])</span> 831 | <span style='color: green'>(-6 [-0.7%])</span> 831 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+9 [+0.2%])</span> 4,408 | <span style='color: red'>(+9 [+0.2%])</span> 4,408 | <span style='color: red'>(+9 [+0.2%])</span> 4,408 | <span style='color: red'>(+9 [+0.2%])</span> 4,408 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.4%])</span> 809 | <span style='color: red'>(+3 [+0.4%])</span> 809 | <span style='color: red'>(+3 [+0.4%])</span> 809 | <span style='color: red'>(+3 [+0.4%])</span> 809 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+1.7%])</span> 180 | <span style='color: red'>(+3 [+1.7%])</span> 180 | <span style='color: red'>(+3 [+1.7%])</span> 180 | <span style='color: red'>(+3 [+1.7%])</span> 180 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+30 [+1.9%])</span> 1,593 | <span style='color: red'>(+30 [+1.9%])</span> 1,593 | <span style='color: red'>(+30 [+1.9%])</span> 1,593 | <span style='color: red'>(+30 [+1.9%])</span> 1,593 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+5 [+1.6%])</span> 321 | <span style='color: red'>(+5 [+1.6%])</span> 321 | <span style='color: red'>(+5 [+1.6%])</span> 321 | <span style='color: red'>(+5 [+1.6%])</span> 321 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 | <span style='color: green'>(-11 [-2.2%])</span> 500 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-22 [-2.2%])</span> 1,001 | <span style='color: green'>(-22 [-2.2%])</span> 1,001 | <span style='color: green'>(-22 [-2.2%])</span> 1,001 | <span style='color: green'>(-22 [-2.2%])</span> 1,001 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+5822 [+82.8%])</span> 12,854 | <span style='color: red'>(+5822 [+82.8%])</span> 12,854 | <span style='color: red'>(+5822 [+82.8%])</span> 12,854 | <span style='color: red'>(+5822 [+82.8%])</span> 12,854 |
| `main_cells_used     ` | <span style='color: red'>(+126526 [+0.2%])</span> 70,818,267 | <span style='color: red'>(+126526 [+0.2%])</span> 70,818,267 | <span style='color: red'>(+126526 [+0.2%])</span> 70,818,267 | <span style='color: red'>(+126526 [+0.2%])</span> 70,818,267 |
| `total_cycles        ` | <span style='color: red'>(+21191 [+1.2%])</span> 1,853,828 | <span style='color: red'>(+21191 [+1.2%])</span> 1,853,828 | <span style='color: red'>(+21191 [+1.2%])</span> 1,853,828 | <span style='color: red'>(+21191 [+1.2%])</span> 1,853,828 |
| `execute_time_ms     ` | <span style='color: red'>(+5828 [+1529.7%])</span> 6,209 | <span style='color: red'>(+5828 [+1529.7%])</span> 6,209 | <span style='color: red'>(+5828 [+1529.7%])</span> 6,209 | <span style='color: red'>(+5828 [+1529.7%])</span> 6,209 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-9 [-0.7%])</span> 1,268 | <span style='color: green'>(-9 [-0.7%])</span> 1,268 | <span style='color: green'>(-9 [-0.7%])</span> 1,268 | <span style='color: green'>(-9 [-0.7%])</span> 1,268 |
| `stark_prove_excluding_trace_time_ms` |  5,377 |  5,377 |  5,377 |  5,377 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-5 [-0.5%])</span> 1,047 | <span style='color: green'>(-5 [-0.5%])</span> 1,047 | <span style='color: green'>(-5 [-0.5%])</span> 1,047 | <span style='color: green'>(-5 [-0.5%])</span> 1,047 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+1.5%])</span> 135 | <span style='color: red'>(+2 [+1.5%])</span> 135 | <span style='color: red'>(+2 [+1.5%])</span> 135 | <span style='color: red'>(+2 [+1.5%])</span> 135 |
| `perm_trace_commit_time_ms` |  1,166 |  1,166 |  1,166 |  1,166 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+8 [+1.1%])</span> 736 | <span style='color: red'>(+8 [+1.1%])</span> 736 | <span style='color: red'>(+8 [+1.1%])</span> 736 | <span style='color: red'>(+8 [+1.1%])</span> 736 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+11 [+1.1%])</span> 1,001 | <span style='color: red'>(+11 [+1.1%])</span> 1,001 | <span style='color: red'>(+11 [+1.1%])</span> 1,001 | <span style='color: red'>(+11 [+1.1%])</span> 1,001 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-18 [-1.4%])</span> 1,285 | <span style='color: green'>(-18 [-1.4%])</span> 1,285 | <span style='color: green'>(-18 [-1.4%])</span> 1,285 | <span style='color: green'>(-18 [-1.4%])</span> 1,285 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 376 | 5 | 

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
| leaf | FriReducedOpeningAir | 4 | 31 | 53 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 92,184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 33,373 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,566 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,266 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 540,109 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,300 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,833,369 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 86,310 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 307,730 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 21,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 18,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 843,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 131,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 471,450 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,219,140 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 2,842,650 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,102,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 861,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 3,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 6,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 106,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 162,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 160,410 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,008,750 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 116,850 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 298,140 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 5,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 27,030 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 15,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 355,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 238,020 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 485,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 161,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 10,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 12,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 116,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 174,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 30,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 9,661,350 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 624,900 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 3,215,900 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 221,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,794,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,416,400 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 807,398 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 446,624 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 497,480 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 248,560 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 2,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 774,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 61,040 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 132,240 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 3,343,704 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 41,328 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 64,008 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 750 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 138,024 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,773 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 20,349 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 6,451,830 | 

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
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 26 | 8,126,464 | 
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
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 20 | 32 | 13,312 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 326,925 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 39,406 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 905,706 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 450,910 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 36,883 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 42,891 | 
| leaf | AccessAdapter<2> | 0 | 162,570 | 
| leaf | AccessAdapter<4> | 0 | 77,002 | 
| leaf | AccessAdapter<8> | 0 | 322 | 
| leaf | Boundary | 0 | 339,983 | 
| leaf | FriReducedOpeningAir | 0 | 128,604 | 
| leaf | PhantomAir | 0 | 44,483 | 
| leaf | ProgramChip | 0 | 75,563 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 27,588 | 
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
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | 176 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fibonacci_program | Boundary | 0 | 36 | 
| fibonacci_program | Merkle | 0 | 228 | 
| fibonacci_program | PhantomAir | 0 | 2 | 
| fibonacci_program | ProgramChip | 0 | 3,275 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 12,437 | 
| leaf | AddEFFI | 0 | ADD | 704 | 
| leaf | AddEFI | 0 | ADD | 616 | 
| leaf | AddEI | 0 | ADD | 28,124 | 
| leaf | AddF | 0 | ADD | 4,375 | 
| leaf | AddFI | 0 | ADD | 15,715 | 
| leaf | AddV | 0 | ADD | 40,638 | 
| leaf | AddVI | 0 | ADD | 94,755 | 
| leaf | Alloc | 0 | ADD | 103,408 | 
| leaf | Alloc | 0 | MUL | 28,700 | 
| leaf | AssertEqE | 0 | BNE | 248 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 4,008 | 
| leaf | AssertEqV | 0 | BNE | 1,451 | 
| leaf | AssertEqVI | 0 | BNE | 242 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 6,888 | 
| leaf | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 10,668 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 126 | 
| leaf | DivE | 0 | BBE4DIV | 6,214 | 
| leaf | DivEIN | 0 | ADD | 216 | 
| leaf | DivEIN | 0 | BBE4DIV | 54 | 
| leaf | DivF | 0 | DIV | 7,392 | 
| leaf | DivFIN | 0 | DIV | 128 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,334 | 
| leaf | HintBitsF | 0 | PHANTOM | 125 | 
| leaf | HintInputVec | 0 | PHANTOM | 23,004 | 
| leaf | IfEq | 0 | BNE | 142 | 
| leaf | IfEqI | 0 | BNE | 23,483 | 
| leaf | IfEqI | 0 | JAL | 8,631 | 
| leaf | IfNe | 0 | BEQ | 143 | 
| leaf | IfNe | 0 | JAL | 1 | 
| leaf | IfNeI | 0 | BEQ | 100 | 
| leaf | ImmE | 0 | ADD | 3,536 | 
| leaf | ImmF | 0 | ADD | 5,431 | 
| leaf | ImmV | 0 | ADD | 5,347 | 
| leaf | LoadE | 0 | ADD | 9,408 | 
| leaf | LoadE | 0 | LOADW | 23,747 | 
| leaf | LoadE | 0 | MUL | 9,408 | 
| leaf | LoadF | 0 | ADD | 11,848 | 
| leaf | LoadF | 0 | LOADW | 24,996 | 
| leaf | LoadF | 0 | MUL | 7,883 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 40,699 | 
| leaf | LoadV | 0 | LOADW | 128,636 | 
| leaf | LoadV | 0 | MUL | 36,526 | 
| leaf | MulE | 0 | BBE4MUL | 19,354 | 
| leaf | MulEF | 0 | MUL | 3,792 | 
| leaf | MulEFI | 0 | MUL | 500 | 
| leaf | MulEI | 0 | ADD | 6,104 | 
| leaf | MulEI | 0 | BBE4MUL | 1,526 | 
| leaf | MulF | 0 | MUL | 33,625 | 
| leaf | MulFI | 0 | MUL | 3,895 | 
| leaf | MulVI | 0 | MUL | 9,938 | 
| leaf | NegE | 0 | MUL | 172 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 51 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 7,728 | 
| leaf | StoreE | 0 | MUL | 7,728 | 
| leaf | StoreE | 0 | STOREW | 13,136 | 
| leaf | StoreF | 0 | ADD | 901 | 
| leaf | StoreF | 0 | MUL | 524 | 
| leaf | StoreF | 0 | STOREW | 8,852 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 231,770 | 
| leaf | StoreV | 0 | ADD | 11,848 | 
| leaf | StoreV | 0 | MUL | 7,934 | 
| leaf | StoreV | 0 | STOREW | 56,656 | 
| leaf | SubE | 0 | FE4SUB | 3,306 | 
| leaf | SubEF | 0 | ADD | 16,182 | 
| leaf | SubEF | 0 | SUB | 5,394 | 
| leaf | SubEFI | 0 | ADD | 344 | 
| leaf | SubEI | 0 | ADD | 432 | 
| leaf | SubFI | 0 | SUB | 3,875 | 
| leaf | SubV | 0 | SUB | 5,816 | 
| leaf | SubVI | 0 | SUB | 1,000 | 
| leaf | SubVIN | 0 | SUB | 840 | 
| leaf | UnsafeCastVF | 0 | ADD | 102 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 322,045 | 
| leaf | ZipFor | 0 | BNE | 297,103 | 
| leaf | ZipFor | 0 | JAL | 30,773 | 

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
| leaf | 0 | 1,268 | 12,854 | 1,853,828 | 180,472,792 | 5,377 | 736 | 1,001 | 1,166 | 1,285 | 1,047 | 70,818,267 | 135 | 6,209 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 831 | 10,151 | 1,500,137 | 197,440,542 | 4,408 | 321 | 500 | 1,593 | 1,001 | 809 | 51,486,676 | 180 | 4,912 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fibonacci-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12970394348)
