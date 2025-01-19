| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-2 [-5.1%])</span> 42.14 | <span style='color: green'>(-2 [-5.1%])</span> 42.14 |
| ecrecover_program | <span style='color: red'>(+2 [+74.0%])</span> 4.58 | <span style='color: red'>(+2 [+74.0%])</span> 4.58 |
| leaf | <span style='color: green'>(-4 [-10.0%])</span> 37.56 | <span style='color: green'>(-4 [-10.0%])</span> 37.56 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1948 [+74.0%])</span> 4,582 | <span style='color: red'>(+1948 [+74.0%])</span> 4,582 | <span style='color: red'>(+1948 [+74.0%])</span> 4,582 | <span style='color: red'>(+1948 [+74.0%])</span> 4,582 |
| `main_cells_used     ` | <span style='color: red'>(+155632 [+1.0%])</span> 15,247,929 | <span style='color: red'>(+155632 [+1.0%])</span> 15,247,929 | <span style='color: red'>(+155632 [+1.0%])</span> 15,247,929 | <span style='color: red'>(+155632 [+1.0%])</span> 15,247,929 |
| `total_cycles        ` | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 | <span style='color: red'>(+4847 [+1.7%])</span> 290,248 |
| `execute_time_ms     ` | <span style='color: red'>(+1953 [+1284.9%])</span> 2,105 | <span style='color: red'>(+1953 [+1284.9%])</span> 2,105 | <span style='color: red'>(+1953 [+1284.9%])</span> 2,105 | <span style='color: red'>(+1953 [+1284.9%])</span> 2,105 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+9 [+3.5%])</span> 266 | <span style='color: red'>(+9 [+3.5%])</span> 266 | <span style='color: red'>(+9 [+3.5%])</span> 266 | <span style='color: red'>(+9 [+3.5%])</span> 266 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-14 [-0.6%])</span> 2,211 | <span style='color: green'>(-14 [-0.6%])</span> 2,211 | <span style='color: green'>(-14 [-0.6%])</span> 2,211 | <span style='color: green'>(-14 [-0.6%])</span> 2,211 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 | <span style='color: green'>(-2 [-0.5%])</span> 394 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 | <span style='color: green'>(-1 [-2.4%])</span> 40 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+0.8%])</span> 534 | <span style='color: red'>(+4 [+0.8%])</span> 534 | <span style='color: red'>(+4 [+0.8%])</span> 534 | <span style='color: red'>(+4 [+0.8%])</span> 534 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-7 [-2.3%])</span> 294 | <span style='color: green'>(-7 [-2.3%])</span> 294 | <span style='color: green'>(-7 [-2.3%])</span> 294 | <span style='color: green'>(-7 [-2.3%])</span> 294 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-24 [-7.2%])</span> 308 | <span style='color: green'>(-24 [-7.2%])</span> 308 | <span style='color: green'>(-24 [-7.2%])</span> 308 | <span style='color: green'>(-24 [-7.2%])</span> 308 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+15 [+2.4%])</span> 637 | <span style='color: red'>(+15 [+2.4%])</span> 637 | <span style='color: red'>(+15 [+2.4%])</span> 637 | <span style='color: red'>(+15 [+2.4%])</span> 637 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-4193 [-10.0%])</span> 37,562 | <span style='color: green'>(-4193 [-10.0%])</span> 37,562 | <span style='color: green'>(-4193 [-10.0%])</span> 37,562 | <span style='color: green'>(-4193 [-10.0%])</span> 37,562 |
| `main_cells_used     ` | <span style='color: green'>(-125130972 [-34.2%])</span> 240,780,125 | <span style='color: green'>(-125130972 [-34.2%])</span> 240,780,125 | <span style='color: green'>(-125130972 [-34.2%])</span> 240,780,125 | <span style='color: green'>(-125130972 [-34.2%])</span> 240,780,125 |
| `total_cycles        ` | <span style='color: green'>(-4487668 [-51.9%])</span> 4,167,337 | <span style='color: green'>(-4487668 [-51.9%])</span> 4,167,337 | <span style='color: green'>(-4487668 [-51.9%])</span> 4,167,337 | <span style='color: green'>(-4487668 [-51.9%])</span> 4,167,337 |
| `execute_time_ms     ` | <span style='color: red'>(+12100 [+495.9%])</span> 14,540 | <span style='color: red'>(+12100 [+495.9%])</span> 14,540 | <span style='color: red'>(+12100 [+495.9%])</span> 14,540 | <span style='color: red'>(+12100 [+495.9%])</span> 14,540 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2173 [-32.8%])</span> 4,444 | <span style='color: green'>(-2173 [-32.8%])</span> 4,444 | <span style='color: green'>(-2173 [-32.8%])</span> 4,444 | <span style='color: green'>(-2173 [-32.8%])</span> 4,444 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-14120 [-43.2%])</span> 18,578 | <span style='color: green'>(-14120 [-43.2%])</span> 18,578 | <span style='color: green'>(-14120 [-43.2%])</span> 18,578 | <span style='color: green'>(-14120 [-43.2%])</span> 18,578 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2689 [-46.4%])</span> 3,109 | <span style='color: green'>(-2689 [-46.4%])</span> 3,109 | <span style='color: green'>(-2689 [-46.4%])</span> 3,109 | <span style='color: green'>(-2689 [-46.4%])</span> 3,109 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-271 [-38.1%])</span> 441 | <span style='color: green'>(-271 [-38.1%])</span> 441 | <span style='color: green'>(-271 [-38.1%])</span> 441 | <span style='color: green'>(-271 [-38.1%])</span> 441 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-3152 [-43.5%])</span> 4,089 | <span style='color: green'>(-3152 [-43.5%])</span> 4,089 | <span style='color: green'>(-3152 [-43.5%])</span> 4,089 | <span style='color: green'>(-3152 [-43.5%])</span> 4,089 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2649 [-35.6%])</span> 4,786 | <span style='color: green'>(-2649 [-35.6%])</span> 4,786 | <span style='color: green'>(-2649 [-35.6%])</span> 4,786 | <span style='color: green'>(-2649 [-35.6%])</span> 4,786 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-2349 [-45.7%])</span> 2,793 | <span style='color: green'>(-2349 [-45.7%])</span> 2,793 | <span style='color: green'>(-2349 [-45.7%])</span> 2,793 | <span style='color: green'>(-2349 [-45.7%])</span> 2,793 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-3008 [-47.3%])</span> 3,358 | <span style='color: green'>(-3008 [-47.3%])</span> 3,358 | <span style='color: green'>(-3008 [-47.3%])</span> 3,358 | <span style='color: green'>(-3008 [-47.3%])</span> 3,358 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 998 | 12 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,571 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 223 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 2 | 411 | 514 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 156 | 190 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 422 | 457 | 
| ecrecover_program | VmConnectorAir | 2 | 3 | 9 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 7,268 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 31,832 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 28,014 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 9,844 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 688,413 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,128 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 530,426 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,197 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 4,278 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 18,461,249 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 21,650 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 79,340 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 30 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 339,960 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 24,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 239,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 5,539,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 450,870 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,364,220 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 3,031,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,382,260 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 930,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 12,600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 211,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 7,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 897,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 918,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 77,730 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 168,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 449,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 449,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 620,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 416,010 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 2,077,110 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,865,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 120,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 1,359,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 1,296,600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 897,270 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 41,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 471,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 19,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 401,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 401,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 43,740 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 30,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 641,790 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 435,390 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 874,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 291,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 357,600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 258,150 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 29,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 23,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 1,110 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 24,990,210 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 937,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 4,673,475 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 237,225 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 18,738,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,761,650 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,729,410 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 632,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 3,681,880 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 419,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 4,200 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 2,386,680 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 432,200 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 813,960 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 63,705,600 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 75,600 | 
| leaf | PhantomAir | CT-initialize-opening-vals | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-sample-challenges | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-setup-and-permute-openings | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 115,416 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 9,576 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 152,040 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 21,147 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 24,738 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,139,226 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 28,807,002 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 2,645,532 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 559,512 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 250,740 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 318,600 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 900 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 74,407 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 228,536 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 238,023 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 275,912 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 124,202 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 29,600 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 384 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 719,648 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 22,734 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 50,292 | 
| ecrecover_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 5,564 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 531,698 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 186,060 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 132,300 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 98,000 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 553,840 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 1,037,520 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 2,702,880 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 285 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 195 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 79,329 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 71,022 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,EcDoubleCoreAir> |  | EcDouble | 0 | 690,153 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularAddSub | 0 | 2,388 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 8,352 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcAddNe | 0 | 449,394 | 
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,680 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 270 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 356 | 399 | 98,959,360 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 36 | 34 | 9,175,040 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 40 | 15,728,640 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 2,097,152 |  | 8 | 11 | 39,845,888 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 24 | 25 | 802,816 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 24 | 11 | 8,960 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 24 | 41 | 532,480 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 24 | 13 | 4,736 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 24 | 17 | 1,343,488 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,288 | 3,164 | 569,856 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 20 | 32 | 212,992 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 12 | 6 | 1,152 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 80 | 36 | 15,204,352 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 48 | 26 | 1,212,416 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 56 | 32 | 2,883,584 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 44 | 18 | 253,952 | 
| ecrecover_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 256 |  | 36 | 26 | 15,872 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 36 | 28 | 524,288 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 76 | 35 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 72 | 40 | 14,680,064 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 104 | 57 | 1,288 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 100 | 39 | 1,112 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 80 | 31 | 454,656 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 28 | 21 | 200,704 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, EcDoubleCoreAir> | 0 | 2,048 |  | 828 | 543 | 2,807,808 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 316 | 261 | 18,464 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 848 | 619 | 1,502,208 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 859,468 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 44,099 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,874,311 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 1,053,922 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 69,482 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 193,463 | 
| leaf | AccessAdapter<2> | 0 | 939,468 | 
| leaf | AccessAdapter<4> | 0 | 462,638 | 
| leaf | AccessAdapter<8> | 0 | 498 | 
| leaf | Boundary | 0 | 1,244,023 | 
| leaf | FriReducedOpeningAir | 0 | 995,400 | 
| leaf | PhantomAir | 0 | 61,689 | 
| leaf | ProgramChip | 0 | 521,095 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 82,687 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 104,869 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,011 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 8,803 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 15,389 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 23,426 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 4,057 | 
| ecrecover_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 214 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 3,194 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 6,645 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 3,780 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 109,806 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 2,559 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 3,383 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,EcDoubleCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 21 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 13,306 | 
| ecrecover_program | AccessAdapter<2> | 0 | 132 | 
| ecrecover_program | AccessAdapter<32> | 0 | 6,654 | 
| ecrecover_program | AccessAdapter<4> | 0 | 68 | 
| ecrecover_program | AccessAdapter<8> | 0 | 27,216 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,060 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 2,990 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,288 | 
| ecrecover_program | PhantomAir | 0 | 45 | 
| ecrecover_program | ProgramChip | 0 | 8,624 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 92,047 | 
| leaf | AddEFFI | 0 | ADD | 824 | 
| leaf | AddEFI | 0 | ADD | 7,988 | 
| leaf | AddEI | 0 | ADD | 184,648 | 
| leaf | AddF | 0 | ADD | 1,333 | 
| leaf | AddFI | 0 | ADD | 15,029 | 
| leaf | AddV | 0 | ADD | 45,474 | 
| leaf | AddVI | 0 | ADD | 101,041 | 
| leaf | Alloc | 0 | ADD | 112,742 | 
| leaf | Alloc | 0 | MUL | 31,031 | 
| leaf | AssertEqE | 0 | BNE | 316 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 1,384 | 
| leaf | AssertEqV | 0 | BNE | 1,218 | 
| leaf | AssertEqVI | 0 | BNE | 428 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 12,600 | 
| leaf | CT-initialize-opening-vals | 0 | PHANTOM | 84 | 
| leaf | CT-sample-challenges | 0 | PHANTOM | 2 | 
| leaf | CT-setup-and-permute-openings | 0 | PHANTOM | 672 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 19,236 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,596 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 1 | 
| leaf | DivE | 0 | BBE4DIV | 10,490 | 
| leaf | DivEIN | 0 | ADD | 420 | 
| leaf | DivEIN | 0 | BBE4DIV | 105 | 
| leaf | DivF | 0 | DIV | 7,056 | 
| leaf | DivFIN | 0 | DIV | 247 | 
| leaf | For | 0 | ADD | 29,931 | 
| leaf | For | 0 | BNE | 29,931 | 
| leaf | For | 0 | JAL | 2,165 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 9,618 | 
| leaf | HintBitsF | 0 | PHANTOM | 43 | 
| leaf | HintInputVec | 0 | PHANTOM | 25,340 | 
| leaf | IfEq | 0 | BNE | 136 | 
| leaf | IfEqI | 0 | BNE | 23,062 | 
| leaf | IfEqI | 0 | JAL | 7,934 | 
| leaf | IfNe | 0 | BEQ | 139 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNeI | 0 | BEQ | 186 | 
| leaf | ImmE | 0 | ADD | 30,604 | 
| leaf | ImmF | 0 | ADD | 2,591 | 
| leaf | ImmV | 0 | ADD | 5,624 | 
| leaf | LoadE | 0 | ADD | 14,994 | 
| leaf | LoadE | 0 | LOADW | 50,865 | 
| leaf | LoadE | 0 | MUL | 14,994 | 
| leaf | LoadF | 0 | ADD | 20,684 | 
| leaf | LoadF | 0 | LOADW | 37,480 | 
| leaf | LoadF | 0 | MUL | 13,867 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 69,237 | 
| leaf | LoadV | 0 | LOADW | 186,939 | 
| leaf | LoadV | 0 | MUL | 62,198 | 
| leaf | MulE | 0 | BBE4MUL | 59,667 | 
| leaf | MulEF | 0 | MUL | 4,032 | 
| leaf | MulEFI | 0 | MUL | 45,304 | 
| leaf | MulEI | 0 | ADD | 43,220 | 
| leaf | MulEI | 0 | BBE4MUL | 10,805 | 
| leaf | MulF | 0 | MUL | 29,909 | 
| leaf | MulFI | 0 | MUL | 1,370 | 
| leaf | MulVI | 0 | MUL | 15,727 | 
| leaf | NegE | 0 | MUL | 664 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 53 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 62 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 13,398 | 
| leaf | StoreE | 0 | MUL | 13,398 | 
| leaf | StoreE | 0 | STOREW | 18,617 | 
| leaf | StoreF | 0 | ADD | 1,458 | 
| leaf | StoreF | 0 | MUL | 1,004 | 
| leaf | StoreF | 0 | STOREW | 9,489 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 749,548 | 
| leaf | StoreV | 0 | ADD | 21,393 | 
| leaf | StoreV | 0 | MUL | 14,513 | 
| leaf | StoreV | 0 | STOREW | 70,466 | 
| leaf | SubE | 0 | FE4SUB | 20,349 | 
| leaf | SubEF | 0 | ADD | 29,136 | 
| leaf | SubEF | 0 | SUB | 9,712 | 
| leaf | SubEFI | 0 | ADD | 11,920 | 
| leaf | SubEI | 0 | ADD | 840 | 
| leaf | SubFI | 0 | SUB | 1,333 | 
| leaf | SubV | 0 | SUB | 8,605 | 
| leaf | SubVI | 0 | SUB | 971 | 
| leaf | SubVIN | 0 | SUB | 798 | 
| leaf | UnsafeCastVF | 0 | ADD | 37 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 798 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 833,007 | 
| leaf | ZipFor | 0 | BNE | 802,663 | 
| leaf | ZipFor | 0 | JAL | 33,996 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 73,487 | 
| ecrecover_program |  | AND | 0 | 15,542 | 
| ecrecover_program |  | AUIPC | 0 | 3,383 | 
| ecrecover_program |  | BEQ | 0 | 10,612 | 
| ecrecover_program |  | BGEU | 0 | 925 | 
| ecrecover_program |  | BLT | 0 | 12 | 
| ecrecover_program |  | BLTU | 0 | 22,489 | 
| ecrecover_program |  | BNE | 0 | 4,777 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_STOREW | 0 | 214 | 
| ecrecover_program |  | IS_EQ | 0 | 3,203 | 
| ecrecover_program |  | JAL | 0 | 1,263 | 
| ecrecover_program |  | JALR | 0 | 6,645 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 3,780 | 
| ecrecover_program |  | LOADBU | 0 | 2,450 | 
| ecrecover_program |  | LOADW | 0 | 13,846 | 
| ecrecover_program |  | LUI | 0 | 2,794 | 
| ecrecover_program |  | MUL | 0 | 2,559 | 
| ecrecover_program |  | MULHU | 0 | 5 | 
| ecrecover_program |  | ModularAddSub | 0 | 12 | 
| ecrecover_program |  | ModularMulDiv | 0 | 32 | 
| ecrecover_program |  | OR | 0 | 6,965 | 
| ecrecover_program |  | PHANTOM | 0 | 45 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 4,312 | 
| ecrecover_program |  | SLTU | 0 | 2,011 | 
| ecrecover_program |  | SRL | 0 | 4,491 | 
| ecrecover_program |  | STOREB | 0 | 25,938 | 
| ecrecover_program |  | STOREW | 0 | 67,572 | 
| ecrecover_program |  | SUB | 0 | 8,850 | 
| ecrecover_program |  | XOR | 0 | 25 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 4,444 | 37,562 | 4,167,337 | 654,461,912 | 18,578 | 4,786 | 2,793 | 4,089 | 3,358 | 3,109 | 240,780,125 | 441 | 14,540 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 266 | 4,582 | 290,248 | 55,919,495 | 2,211 | 294 | 308 | 534 | 637 | 394 | 15,247,929 | 40 | 2,105 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-ecrecover_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c/ecrecover-ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/ce7d5730f68e07ce1aa6b9a9c37a8986ce30619c

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12848892486)
