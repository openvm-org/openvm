| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+2 [+64.3%])</span> 5.57 | <span style='color: red'>(+2 [+64.3%])</span> 5.57 |
| verify_fibair | <span style='color: red'>(+2 [+64.3%])</span> 5.57 | <span style='color: red'>(+2 [+64.3%])</span> 5.57 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+2182 [+64.3%])</span> 5,573 | <span style='color: red'>(+2182 [+64.3%])</span> 5,573 | <span style='color: red'>(+2182 [+64.3%])</span> 5,573 | <span style='color: red'>(+2182 [+64.3%])</span> 5,573 |
| `main_cells_used     ` |  25,526,369 |  25,526,369 |  25,526,369 |  25,526,369 |
| `total_cycles        ` |  711,874 |  711,874 |  711,874 |  711,874 |
| `execute_time_ms     ` | <span style='color: red'>(+2028 [+1370.3%])</span> 2,176 | <span style='color: red'>(+2028 [+1370.3%])</span> 2,176 | <span style='color: red'>(+2028 [+1370.3%])</span> 2,176 | <span style='color: red'>(+2028 [+1370.3%])</span> 2,176 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+6 [+1.4%])</span> 437 | <span style='color: red'>(+6 [+1.4%])</span> 437 | <span style='color: red'>(+6 [+1.4%])</span> 437 | <span style='color: red'>(+6 [+1.4%])</span> 437 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+148 [+5.3%])</span> 2,960 | <span style='color: red'>(+148 [+5.3%])</span> 2,960 | <span style='color: red'>(+148 [+5.3%])</span> 2,960 | <span style='color: red'>(+148 [+5.3%])</span> 2,960 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+46 [+9.7%])</span> 520 | <span style='color: red'>(+46 [+9.7%])</span> 520 | <span style='color: red'>(+46 [+9.7%])</span> 520 | <span style='color: red'>(+46 [+9.7%])</span> 520 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-1.5%])</span> 67 | <span style='color: green'>(-1 [-1.5%])</span> 67 | <span style='color: green'>(-1 [-1.5%])</span> 67 | <span style='color: green'>(-1 [-1.5%])</span> 67 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+34 [+7.0%])</span> 521 | <span style='color: red'>(+34 [+7.0%])</span> 521 | <span style='color: red'>(+34 [+7.0%])</span> 521 | <span style='color: red'>(+34 [+7.0%])</span> 521 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+10 [+1.8%])</span> 572 | <span style='color: red'>(+10 [+1.8%])</span> 572 | <span style='color: red'>(+10 [+1.8%])</span> 572 | <span style='color: red'>(+10 [+1.8%])</span> 572 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+39 [+7.2%])</span> 578 | <span style='color: red'>(+39 [+7.2%])</span> 578 | <span style='color: red'>(+39 [+7.2%])</span> 578 | <span style='color: red'>(+39 [+7.2%])</span> 578 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+19 [+2.8%])</span> 699 | <span style='color: red'>(+19 [+2.8%])</span> 699 | <span style='color: red'>(+19 [+2.8%])</span> 699 | <span style='color: red'>(+19 [+2.8%])</span> 699 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 68 | 3 | 14 | 0 | 31 | 18 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 31 | 302 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 23 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 437 | 5,573 | 711,874 | 72,898,584 | 2,960 | 572 | 578 | 521 | 699 | 520 | 25,526,369 | 67 | 2,176 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 32,768 |  | 16 | 17 | 1,081,344 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 8,192 |  | 36 | 348 | 3,145,728 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 262,144 |  | 28 | 23 | 13,369,344 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 32,768 |  | 12 | 10 | 720,896 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 524,288 |  | 20 | 30 | 26,214,400 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  | 36 | 25 | 15,990,784 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

| group | air_name | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | BNE | 3,956 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | BNE | 92 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | BNE | 163,024 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | BNE | 14,697 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | BNE | 460 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | BNE | 319,493 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | BNE | 24,817 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | BNE | 350,543 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | BEQ | 167,831 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | BEQ | 14,559 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | BNE | 2,052,842 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> |  | JAL | 10 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | For | JAL | 23,820 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfEqI | JAL | 51,880 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfNe | JAL | 20 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | ZipFor | JAL | 118,990 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | ADD | 13,680 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | ADD | 246,360 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | ADD | 39,990 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | ADD | 50,160 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | ADD | 625,590 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | ADD | 2,033,610 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | ADD | 1,471,920 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | MUL | 454,980 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | ADD | 120 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | DIV | 90 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | ADD | 416,730 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | ADD | 35,520 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | ADD | 240,930 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | ADD | 292,890 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | ADD | 99,540 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | MUL | 99,540 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | ADD | 17,670 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | MUL | 10,440 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | ADD | 195,270 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | MUL | 152,340 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | MUL | 75,840 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | ADD | 9,240 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | MUL | 128,310 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | MUL | 40,020 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | MUL | 313,140 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | ADD | 23,940 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | MUL | 23,940 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | ADD | 156,690 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | MUL | 300 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | ADD | 70,050 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | MUL | 49,350 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | ADD | 11,790 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | SUB | 3,930 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | ADD | 240 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | SUB | 39,990 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | SUB | 90,750 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | SUB | 22,350 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | SUB | 18,900 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | ADD | 2,970,420 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | LOADW | 489,875 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | LOADW | 1,130,975 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | STOREW | 298,575 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | HINT_STOREW | 1,799,350 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | STOREW | 511,575 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | LOADW4 | 183,192 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | STOREW4 | 226,372 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | FE4ADD | 57,960 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | BBE4DIV | 30,320 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | BBE4DIV | 40 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | BBE4MUL | 108,680 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | BBE4MUL | 3,080 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | FE4SUB | 75,680 | 
| verify_fibair | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | COMP_POS2 | 2,470,104 | 
| verify_fibair | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | PERM_POS2 | 257,520 | 
| verify_fibair | FriReducedOpeningAir | FriReducedOpening | FRI_REDUCED_OPENING | 21,504 | 
| verify_fibair | PhantomAir | HintBitsF | PHANTOM | 258 | 
| verify_fibair | PhantomAir | HintInputVec | PHANTOM | 56,196 | 

| group | chip_name | rows_used |
| --- | --- | --- |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 135,318 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | 19,472 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 351,557 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 169,214 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 12,046 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6,894 | 
| verify_fibair | AccessAdapter<2> | 113,358 | 
| verify_fibair | AccessAdapter<4> | 56,680 | 
| verify_fibair | AccessAdapter<8> | 17,106 | 
| verify_fibair | Arc<BabyBearParameters>, 1> | 7,838 | 
| verify_fibair | Boundary | 128,104 | 
| verify_fibair | FriReducedOpeningAir | 336 | 
| verify_fibair | PhantomAir | 9,409 | 
| verify_fibair | ProgramChip | 5,493 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 
| verify_fibair | VmConnectorAir | 2 | 

| group | dsl_ir | opcode | frequency |
| --- | --- | --- | --- |
| verify_fibair |  | ADD | 2 | 
| verify_fibair |  | JAL | 1 | 
| verify_fibair | AddE | FE4ADD | 1,449 | 
| verify_fibair | AddEFFI | ADD | 456 | 
| verify_fibair | AddEI | ADD | 8,212 | 
| verify_fibair | AddF | ADD | 1,333 | 
| verify_fibair | AddFI | ADD | 1,672 | 
| verify_fibair | AddV | ADD | 20,853 | 
| verify_fibair | AddVI | ADD | 67,787 | 
| verify_fibair | Alloc | ADD | 49,064 | 
| verify_fibair | Alloc | MUL | 15,166 | 
| verify_fibair | AssertEqE | BNE | 172 | 
| verify_fibair | AssertEqEI | BNE | 4 | 
| verify_fibair | AssertEqF | BNE | 7,088 | 
| verify_fibair | AssertEqV | BNE | 639 | 
| verify_fibair | AssertEqVI | BNE | 20 | 
| verify_fibair | DivE | BBE4DIV | 758 | 
| verify_fibair | DivEIN | ADD | 4 | 
| verify_fibair | DivEIN | BBE4DIV | 1 | 
| verify_fibair | DivFIN | DIV | 3 | 
| verify_fibair | For | ADD | 13,891 | 
| verify_fibair | For | BNE | 13,891 | 
| verify_fibair | For | JAL | 2,382 | 
| verify_fibair | FriReducedOpening | FRI_REDUCED_OPENING | 126 | 
| verify_fibair | HintBitsF | PHANTOM | 43 | 
| verify_fibair | HintInputVec | PHANTOM | 9,366 | 
| verify_fibair | IfEq | BNE | 1,079 | 
| verify_fibair | IfEqI | BNE | 15,241 | 
| verify_fibair | IfEqI | JAL | 5,188 | 
| verify_fibair | IfNe | BEQ | 7,297 | 
| verify_fibair | IfNe | JAL | 2 | 
| verify_fibair | IfNeI | BEQ | 633 | 
| verify_fibair | ImmE | ADD | 1,184 | 
| verify_fibair | ImmF | ADD | 8,031 | 
| verify_fibair | ImmV | ADD | 9,763 | 
| verify_fibair | LoadE | ADD | 3,318 | 
| verify_fibair | LoadE | LOADW4 | 5,388 | 
| verify_fibair | LoadE | MUL | 3,318 | 
| verify_fibair | LoadF | ADD | 589 | 
| verify_fibair | LoadF | LOADW | 19,595 | 
| verify_fibair | LoadF | MUL | 348 | 
| verify_fibair | LoadHeapPtr | ADD | 1 | 
| verify_fibair | LoadV | ADD | 6,509 | 
| verify_fibair | LoadV | LOADW | 45,239 | 
| verify_fibair | LoadV | MUL | 5,078 | 
| verify_fibair | MulE | BBE4MUL | 2,717 | 
| verify_fibair | MulEF | MUL | 2,528 | 
| verify_fibair | MulEI | ADD | 308 | 
| verify_fibair | MulEI | BBE4MUL | 77 | 
| verify_fibair | MulF | MUL | 4,277 | 
| verify_fibair | MulFI | MUL | 1,334 | 
| verify_fibair | MulVI | MUL | 10,438 | 
| verify_fibair | Poseidon2CompressBabyBear | COMP_POS2 | 7,098 | 
| verify_fibair | Poseidon2PermuteBabyBear | PERM_POS2 | 740 | 
| verify_fibair | StoreE | ADD | 798 | 
| verify_fibair | StoreE | MUL | 798 | 
| verify_fibair | StoreE | STOREW4 | 6,658 | 
| verify_fibair | StoreF | ADD | 5,223 | 
| verify_fibair | StoreF | MUL | 10 | 
| verify_fibair | StoreF | STOREW | 11,943 | 
| verify_fibair | StoreHeapPtr | ADD | 1 | 
| verify_fibair | StoreHintWord | HINT_STOREW | 71,974 | 
| verify_fibair | StoreV | ADD | 2,335 | 
| verify_fibair | StoreV | MUL | 1,645 | 
| verify_fibair | StoreV | STOREW | 20,463 | 
| verify_fibair | SubE | FE4SUB | 1,892 | 
| verify_fibair | SubEF | ADD | 393 | 
| verify_fibair | SubEF | SUB | 131 | 
| verify_fibair | SubEI | ADD | 8 | 
| verify_fibair | SubFI | SUB | 1,333 | 
| verify_fibair | SubV | SUB | 3,025 | 
| verify_fibair | SubVI | SUB | 745 | 
| verify_fibair | SubVIN | SUB | 630 | 
| verify_fibair | UnsafeCastVF | ADD | 1 | 
| verify_fibair | ZipFor | ADD | 99,014 | 
| verify_fibair | ZipFor | BNE | 89,254 | 
| verify_fibair | ZipFor | JAL | 11,899 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/1641da4d9e43ed3aa4344b4b336de24d15efa1a6/verify_fibair-1641da4d9e43ed3aa4344b4b336de24d15efa1a6-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/1641da4d9e43ed3aa4344b4b336de24d15efa1a6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12849658015)
