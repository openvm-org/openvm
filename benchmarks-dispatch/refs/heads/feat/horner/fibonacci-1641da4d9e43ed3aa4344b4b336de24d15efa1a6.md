| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+15 [+79.3%])</span> 34.66 | <span style='color: red'>(+15 [+79.3%])</span> 34.66 |
| fibonacci_program | <span style='color: red'>(+5 [+75.0%])</span> 10.65 | <span style='color: red'>(+5 [+75.0%])</span> 10.65 |
| leaf | <span style='color: red'>(+11 [+81.3%])</span> 24.01 | <span style='color: red'>(+11 [+81.3%])</span> 24.01 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4565 [+75.0%])</span> 10,648 | <span style='color: red'>(+4565 [+75.0%])</span> 10,648 | <span style='color: red'>(+4565 [+75.0%])</span> 10,648 | <span style='color: red'>(+4565 [+75.0%])</span> 10,648 |
| `main_cells_used     ` |  51,503,940 |  51,503,940 |  51,503,940 |  51,503,940 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+4544 [+1470.6%])</span> 4,853 | <span style='color: red'>(+4544 [+1470.6%])</span> 4,853 | <span style='color: red'>(+4544 [+1470.6%])</span> 4,853 | <span style='color: red'>(+4544 [+1470.6%])</span> 4,853 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-7 [-0.9%])</span> 813 | <span style='color: green'>(-7 [-0.9%])</span> 813 | <span style='color: green'>(-7 [-0.9%])</span> 813 | <span style='color: green'>(-7 [-0.9%])</span> 813 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+28 [+0.6%])</span> 4,982 | <span style='color: red'>(+28 [+0.6%])</span> 4,982 | <span style='color: red'>(+28 [+0.6%])</span> 4,982 | <span style='color: red'>(+28 [+0.6%])</span> 4,982 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-7 [-0.9%])</span> 805 | <span style='color: green'>(-7 [-0.9%])</span> 805 | <span style='color: green'>(-7 [-0.9%])</span> 805 | <span style='color: green'>(-7 [-0.9%])</span> 805 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.6%])</span> 179 | <span style='color: green'>(-1 [-0.6%])</span> 179 | <span style='color: green'>(-1 [-0.6%])</span> 179 | <span style='color: green'>(-1 [-0.6%])</span> 179 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+28 [+1.7%])</span> 1,629 | <span style='color: red'>(+28 [+1.7%])</span> 1,629 | <span style='color: red'>(+28 [+1.7%])</span> 1,629 | <span style='color: red'>(+28 [+1.7%])</span> 1,629 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-11 [-1.4%])</span> 773 | <span style='color: green'>(-11 [-1.4%])</span> 773 | <span style='color: green'>(-11 [-1.4%])</span> 773 | <span style='color: green'>(-11 [-1.4%])</span> 773 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+25 [+5.1%])</span> 518 | <span style='color: red'>(+25 [+5.1%])</span> 518 | <span style='color: red'>(+25 [+5.1%])</span> 518 | <span style='color: red'>(+25 [+5.1%])</span> 518 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-5 [-0.5%])</span> 1,076 | <span style='color: green'>(-5 [-0.5%])</span> 1,076 | <span style='color: green'>(-5 [-0.5%])</span> 1,076 | <span style='color: green'>(-5 [-0.5%])</span> 1,076 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+10765 [+81.3%])</span> 24,010 | <span style='color: red'>(+10765 [+81.3%])</span> 24,010 | <span style='color: red'>(+10765 [+81.3%])</span> 24,010 | <span style='color: red'>(+10765 [+81.3%])</span> 24,010 |
| `main_cells_used     ` | <span style='color: red'>(+1354401 [+1.2%])</span> 112,091,909 | <span style='color: red'>(+1354401 [+1.2%])</span> 112,091,909 | <span style='color: red'>(+1354401 [+1.2%])</span> 112,091,909 | <span style='color: red'>(+1354401 [+1.2%])</span> 112,091,909 |
| `total_cycles        ` | <span style='color: red'>(+173125 [+5.6%])</span> 3,260,227 | <span style='color: red'>(+173125 [+5.6%])</span> 3,260,227 | <span style='color: red'>(+173125 [+5.6%])</span> 3,260,227 | <span style='color: red'>(+173125 [+5.6%])</span> 3,260,227 |
| `execute_time_ms     ` | <span style='color: red'>(+10675 [+1882.7%])</span> 11,242 | <span style='color: red'>(+10675 [+1882.7%])</span> 11,242 | <span style='color: red'>(+10675 [+1882.7%])</span> 11,242 | <span style='color: red'>(+10675 [+1882.7%])</span> 11,242 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-8 [-0.4%])</span> 1,804 | <span style='color: green'>(-8 [-0.4%])</span> 1,804 | <span style='color: green'>(-8 [-0.4%])</span> 1,804 | <span style='color: green'>(-8 [-0.4%])</span> 1,804 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+98 [+0.9%])</span> 10,964 | <span style='color: red'>(+98 [+0.9%])</span> 10,964 | <span style='color: red'>(+98 [+0.9%])</span> 10,964 | <span style='color: red'>(+98 [+0.9%])</span> 10,964 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+31 [+1.6%])</span> 2,005 | <span style='color: red'>(+31 [+1.6%])</span> 2,005 | <span style='color: red'>(+31 [+1.6%])</span> 2,005 | <span style='color: red'>(+31 [+1.6%])</span> 2,005 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-1.7%])</span> 231 | <span style='color: green'>(-4 [-1.7%])</span> 231 | <span style='color: green'>(-4 [-1.7%])</span> 231 | <span style='color: green'>(-4 [-1.7%])</span> 231 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-26 [-1.3%])</span> 1,965 | <span style='color: green'>(-26 [-1.3%])</span> 1,965 | <span style='color: green'>(-26 [-1.3%])</span> 1,965 | <span style='color: green'>(-26 [-1.3%])</span> 1,965 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+21 [+0.8%])</span> 2,514 | <span style='color: red'>(+21 [+0.8%])</span> 2,514 | <span style='color: red'>(+21 [+0.8%])</span> 2,514 | <span style='color: red'>(+21 [+0.8%])</span> 2,514 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+44 [+2.4%])</span> 1,888 | <span style='color: red'>(+44 [+2.4%])</span> 1,888 | <span style='color: red'>(+44 [+2.4%])</span> 1,888 | <span style='color: red'>(+44 [+2.4%])</span> 1,888 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+33 [+1.4%])</span> 2,358 | <span style='color: red'>(+33 [+1.4%])</span> 2,358 | <span style='color: red'>(+33 [+1.4%])</span> 2,358 | <span style='color: red'>(+33 [+1.4%])</span> 2,358 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 347 | 5 | 

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
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 31 | 302 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 248,216 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 24,679 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,543 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 3,140,903 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 2,110,112 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 1,557,790 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 408,043 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 21,643 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 7,132,714 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 121,580 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 283,660 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 326,740 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 21,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 18,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 829,560 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 1,298,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,947,150 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 8,243,340 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,441,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,024,500 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 6,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 4,096,830 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 509,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 1,402,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 582,540 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 332,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 332,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 345,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 236,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,378,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,146,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 113,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 15,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 183,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 2,510,010 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 40,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 945,510 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 5,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 234,870 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 21,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 612,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 495,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 485,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 161,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 10,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 12,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 446,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 30,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 11,094,630 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 2,746,350 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 6,105,150 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 2,729,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,741,100 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,630,200 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW4 | 743,172 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW4 | 499,902 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 497,320 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 248,560 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 2,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 341,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 61,040 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 132,240 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 6,025,620 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 3,218,652 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,547,904 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 41,328 | 
| leaf | PhantomAir | CT-poseidon2-hash | 0 | PHANTOM | 23,688 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 0 | PHANTOM | 744,912 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 64,008 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 0 | PHANTOM | 33,768 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 33,768 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 139,272 | 

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
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | PhantomAir | 0 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 637,194 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 73,200 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,506,203 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 758,096 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 36,561 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 32,068 | 
| leaf | AccessAdapter<2> | 0 | 377,918 | 
| leaf | AccessAdapter<4> | 0 | 189,212 | 
| leaf | AccessAdapter<8> | 0 | 55,694 | 
| leaf | Arc<BabyBearParameters>, 1> | 0 | 26,564 | 
| leaf | Boundary | 0 | 385,029 | 
| leaf | FriReducedOpeningAir | 0 | 117,936 | 
| leaf | PhantomAir | 0 | 184,971 | 
| leaf | ProgramChip | 0 | 76,210 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
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
| leaf | AddFI | 0 | ADD | 43,268 | 
| leaf | AddV | 0 | ADD | 64,905 | 
| leaf | AddVI | 0 | ADD | 274,778 | 
| leaf | Alloc | 0 | ADD | 114,724 | 
| leaf | Alloc | 0 | MUL | 34,150 | 
| leaf | AssertEqE | 0 | BNE | 248 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 10,792 | 
| leaf | AssertEqV | 0 | BNE | 1,073 | 
| leaf | AssertEqVI | 0 | BNE | 241 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 6,888 | 
| leaf | CT-poseidon2-hash | 0 | PHANTOM | 3,948 | 
| leaf | CT-poseidon2-hash-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-setup | 0 | PHANTOM | 124,152 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 10,668 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-reduce-fast | 0 | PHANTOM | 5,628 | 
| leaf | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 5,628 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 1 | 
| leaf | DivE | 0 | BBE4DIV | 6,214 | 
| leaf | DivEIN | 0 | ADD | 216 | 
| leaf | DivEIN | 0 | BBE4DIV | 54 | 
| leaf | DivFIN | 0 | DIV | 128 | 
| leaf | For | 0 | ADD | 136,561 | 
| leaf | For | 0 | BNE | 136,561 | 
| leaf | For | 0 | JAL | 12,158 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,334 | 
| leaf | HintBitsF | 0 | PHANTOM | 43 | 
| leaf | HintInputVec | 0 | PHANTOM | 23,212 | 
| leaf | IfEq | 0 | BNE | 91,744 | 
| leaf | IfEqI | 0 | BNE | 67,730 | 
| leaf | IfEqI | 0 | JAL | 28,366 | 
| leaf | IfNe | 0 | BEQ | 17,741 | 
| leaf | IfNe | 0 | JAL | 1 | 
| leaf | IfNeI | 0 | BEQ | 941 | 
| leaf | ImmE | 0 | ADD | 16,984 | 
| leaf | ImmF | 0 | ADD | 46,741 | 
| leaf | ImmV | 0 | ADD | 19,418 | 
| leaf | LoadE | 0 | ADD | 11,088 | 
| leaf | LoadE | 0 | LOADW4 | 21,858 | 
| leaf | LoadE | 0 | MUL | 11,088 | 
| leaf | LoadF | 0 | ADD | 11,512 | 
| leaf | LoadF | 0 | LOADW | 109,854 | 
| leaf | LoadF | 0 | MUL | 7,883 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 45,949 | 
| leaf | LoadV | 0 | LOADW | 244,206 | 
| leaf | LoadV | 0 | MUL | 38,206 | 
| leaf | MulE | 0 | BBE4MUL | 8,535 | 
| leaf | MulEF | 0 | MUL | 3,792 | 
| leaf | MulEFI | 0 | MUL | 500 | 
| leaf | MulEI | 0 | ADD | 6,104 | 
| leaf | MulEI | 0 | BBE4MUL | 1,526 | 
| leaf | MulF | 0 | MUL | 83,667 | 
| leaf | MulFI | 0 | MUL | 1,353 | 
| leaf | MulVI | 0 | MUL | 31,517 | 
| leaf | NegE | 0 | MUL | 172 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 17,315 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 9,249 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 7,728 | 
| leaf | StoreE | 0 | MUL | 7,728 | 
| leaf | StoreE | 0 | STOREW4 | 14,703 | 
| leaf | StoreF | 0 | ADD | 7,829 | 
| leaf | StoreF | 0 | MUL | 732 | 
| leaf | StoreF | 0 | STOREW | 109,184 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 229,644 | 
| leaf | StoreV | 0 | ADD | 20,416 | 
| leaf | StoreV | 0 | MUL | 16,502 | 
| leaf | StoreV | 0 | STOREW | 65,208 | 
| leaf | SubE | 0 | FE4SUB | 3,306 | 
| leaf | SubEF | 0 | ADD | 16,182 | 
| leaf | SubEF | 0 | SUB | 5,394 | 
| leaf | SubEFI | 0 | ADD | 344 | 
| leaf | SubEI | 0 | ADD | 432 | 
| leaf | SubFI | 0 | SUB | 1,333 | 
| leaf | SubV | 0 | SUB | 14,888 | 
| leaf | SubVI | 0 | SUB | 1,000 | 
| leaf | SubVIN | 0 | SUB | 840 | 
| leaf | UnsafeCastVF | 0 | ADD | 20 | 
| leaf | ZipFor | 0 | ADD | 369,821 | 
| leaf | ZipFor | 0 | BNE | 310,118 | 
| leaf | ZipFor | 0 | JAL | 32,674 | 

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
| leaf | 0 | 1,804 | 24,010 | 3,260,227 | 304,941,528 | 10,964 | 2,514 | 1,888 | 1,965 | 2,358 | 2,005 | 112,091,909 | 231 | 11,242 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 813 | 10,648 | 1,500,137 | 197,453,854 | 4,982 | 773 | 518 | 1,629 | 1,076 | 805 | 51,503,940 | 179 | 4,853 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/fibonacci-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/1641da4d9e43ed3aa4344b4b336de24d15efa1a6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12849658015)
