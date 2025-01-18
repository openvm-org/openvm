| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+11.4%])</span> 3.79 | <span style='color: red'>(+0 [+11.4%])</span> 3.79 |
| verify_fibair | <span style='color: red'>(+0 [+11.4%])</span> 3.79 | <span style='color: red'>(+0 [+11.4%])</span> 3.79 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+387 [+11.4%])</span> 3,790 | <span style='color: red'>(+387 [+11.4%])</span> 3,790 | <span style='color: red'>(+387 [+11.4%])</span> 3,790 | <span style='color: red'>(+387 [+11.4%])</span> 3,790 |
| `main_cells_used     ` | <span style='color: green'>(-6223482 [-24.4%])</span> 19,300,172 | <span style='color: green'>(-6223482 [-24.4%])</span> 19,300,172 | <span style='color: green'>(-6223482 [-24.4%])</span> 19,300,172 | <span style='color: green'>(-6223482 [-24.4%])</span> 19,300,172 |
| `total_cycles        ` | <span style='color: green'>(-200192 [-28.1%])</span> 511,522 | <span style='color: green'>(-200192 [-28.1%])</span> 511,522 | <span style='color: green'>(-200192 [-28.1%])</span> 511,522 | <span style='color: green'>(-200192 [-28.1%])</span> 511,522 |
| `execute_time_ms     ` | <span style='color: red'>(+1393 [+954.1%])</span> 1,539 | <span style='color: red'>(+1393 [+954.1%])</span> 1,539 | <span style='color: red'>(+1393 [+954.1%])</span> 1,539 | <span style='color: red'>(+1393 [+954.1%])</span> 1,539 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-79 [-18.5%])</span> 348 | <span style='color: green'>(-79 [-18.5%])</span> 348 | <span style='color: green'>(-79 [-18.5%])</span> 348 | <span style='color: green'>(-79 [-18.5%])</span> 348 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-927 [-32.8%])</span> 1,903 | <span style='color: green'>(-927 [-32.8%])</span> 1,903 | <span style='color: green'>(-927 [-32.8%])</span> 1,903 | <span style='color: green'>(-927 [-32.8%])</span> 1,903 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-165 [-33.9%])</span> 322 | <span style='color: green'>(-165 [-33.9%])</span> 322 | <span style='color: green'>(-165 [-33.9%])</span> 322 | <span style='color: green'>(-165 [-33.9%])</span> 322 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-19 [-28.8%])</span> 47 | <span style='color: green'>(-19 [-28.8%])</span> 47 | <span style='color: green'>(-19 [-28.8%])</span> 47 | <span style='color: green'>(-19 [-28.8%])</span> 47 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-150 [-31.0%])</span> 334 | <span style='color: green'>(-150 [-31.0%])</span> 334 | <span style='color: green'>(-150 [-31.0%])</span> 334 | <span style='color: green'>(-150 [-31.0%])</span> 334 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-165 [-29.3%])</span> 399 | <span style='color: green'>(-165 [-29.3%])</span> 399 | <span style='color: green'>(-165 [-29.3%])</span> 399 | <span style='color: green'>(-165 [-29.3%])</span> 399 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-193 [-35.9%])</span> 344 | <span style='color: green'>(-193 [-35.9%])</span> 344 | <span style='color: green'>(-193 [-35.9%])</span> 344 | <span style='color: green'>(-193 [-35.9%])</span> 344 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-235 [-34.1%])</span> 454 | <span style='color: green'>(-235 [-34.1%])</span> 454 | <span style='color: green'>(-235 [-34.1%])</span> 454 | <span style='color: green'>(-235 [-34.1%])</span> 454 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 | 65,536 | 66 | 3 | 13 | 0 | 32 | 16 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 35 | 59 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 176 | 590 |  | 
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
| verify_fibair | 348 | 3,790 | 511,522 | 50,178,200 | 1,903 | 399 | 344 | 334 | 454 | 322 | 19,300,172 | 47 | 1,539 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 16 | 11 | 1,769,472 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 16 | 13 | 950,272 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 512 |  | 76 | 64 | 71,680 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 356 | 399 | 12,369,920 | 
| verify_fibair | PhantomAir | 16,384 |  | 8 | 6 | 229,376 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  | 28 | 23 | 6,684,672 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 16,384 |  | 12 | 10 | 360,448 | 
| verify_fibair | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 262,144 |  | 20 | 30 | 13,107,200 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 131,072 |  | 36 | 25 | 7,995,392 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 36 | 34 | 1,146,880 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 40 | 491,520 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 131,072 |  | 8 | 11 | 2,490,368 | 

| group | air_name | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | BNE | 3,956 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | BNE | 92 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | BNE | 31,648 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | BNE | 14,697 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | BNE | 460 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | BNE | 235,451 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | BNE | 2,599 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | BNE | 104,213 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | BEQ | 2,645 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | BEQ | 69 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | BNE | 1,875,098 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> |  | JAL | 10 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | For | JAL | 9,540 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfEqI | JAL | 21,100 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfNe | JAL | 20 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | ZipFor | JAL | 105,550 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | ADD | 13,680 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | ADD | 246,360 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | ADD | 39,990 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | ADD | 135,840 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | ADD | 513,450 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | ADD | 996,630 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | ADD | 1,315,680 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | MUL | 376,860 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | ADD | 120 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | DIV | 42,840 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | DIV | 90 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | ADD | 307,110 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | ADD | 25,440 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | ADD | 51,930 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | ADD | 110,190 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | ADD | 61,740 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | MUL | 61,740 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | ADD | 20,190 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | MUL | 10,440 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | ADD | 136,050 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | MUL | 114,540 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | MUL | 75,840 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | ADD | 9,240 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | MUL | 171,150 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | MUL | 40,020 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | MUL | 54,840 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | ADD | 23,940 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | MUL | 23,940 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | ADD | 5,490 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | MUL | 300 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | ADD | 27,210 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | MUL | 6,510 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | ADD | 11,790 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | SUB | 3,930 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | ADD | 240 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | SUB | 39,990 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | SUB | 45,390 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | SUB | 22,350 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | SUB | 18,900 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | ADD | 30 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | ADD | 2,506,740 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | LOADW | 76,175 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | LOADW | 527,225 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | STOREW | 42,375 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | HINT_STOREW | 1,799,350 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | STOREW | 475,875 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | LOADW | 140,352 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | STOREW | 183,532 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | FE4ADD | 57,960 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | BBE4DIV | 30,320 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | BBE4DIV | 40 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | BBE4MUL | 108,680 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | BBE4MUL | 3,080 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | FE4SUB | 75,680 | 
| verify_fibair | FriReducedOpeningAir | FriReducedOpening | FRI_REDUCED_OPENING | 21,504 | 
| verify_fibair | PhantomAir | HintBitsF | PHANTOM | 258 | 
| verify_fibair | PhantomAir | HintInputVec | PHANTOM | 56,196 | 
| verify_fibair | VerifyBatchAir | Poseidon2PermuteBabyBear | PERM_POS2 | 10,374 | 
| verify_fibair | VerifyBatchAir | VerifyBatchExt | VERIFY_BATCH | 2,765,070 | 
| verify_fibair | VerifyBatchAir | VerifyBatchFelt | VERIFY_BATCH | 636,804 | 

| group | chip_name | rows_used |
| --- | --- | --- |
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 98,736 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | 13,622 | 
| verify_fibair | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 255,629 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 116,840 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 9,526 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6,894 | 
| verify_fibair | AccessAdapter<2> | 41,514 | 
| verify_fibair | AccessAdapter<4> | 20,044 | 
| verify_fibair | AccessAdapter<8> | 88 | 
| verify_fibair | Boundary | 111,584 | 
| verify_fibair | FriReducedOpeningAir | 336 | 
| verify_fibair | PhantomAir | 9,409 | 
| verify_fibair | ProgramChip | 5,006 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 
| verify_fibair | VerifyBatchAir | 8,552 | 
| verify_fibair | VmConnectorAir | 2 | 

| group | dsl_ir | opcode | frequency |
| --- | --- | --- | --- |
| verify_fibair |  | ADD | 2 | 
| verify_fibair |  | JAL | 1 | 
| verify_fibair | AddE | FE4ADD | 1,449 | 
| verify_fibair | AddEFFI | ADD | 456 | 
| verify_fibair | AddEI | ADD | 8,212 | 
| verify_fibair | AddF | ADD | 1,333 | 
| verify_fibair | AddFI | ADD | 4,528 | 
| verify_fibair | AddV | ADD | 17,115 | 
| verify_fibair | AddVI | ADD | 33,221 | 
| verify_fibair | Alloc | ADD | 43,856 | 
| verify_fibair | Alloc | MUL | 12,562 | 
| verify_fibair | AssertEqE | BNE | 172 | 
| verify_fibair | AssertEqEI | BNE | 4 | 
| verify_fibair | AssertEqF | BNE | 1,376 | 
| verify_fibair | AssertEqV | BNE | 639 | 
| verify_fibair | AssertEqVI | BNE | 20 | 
| verify_fibair | DivE | BBE4DIV | 758 | 
| verify_fibair | DivEIN | ADD | 4 | 
| verify_fibair | DivEIN | BBE4DIV | 1 | 
| verify_fibair | DivF | DIV | 1,428 | 
| verify_fibair | DivFIN | DIV | 3 | 
| verify_fibair | For | ADD | 10,237 | 
| verify_fibair | For | BNE | 10,237 | 
| verify_fibair | For | JAL | 954 | 
| verify_fibair | FriReducedOpening | FRI_REDUCED_OPENING | 126 | 
| verify_fibair | HintBitsF | PHANTOM | 43 | 
| verify_fibair | HintInputVec | PHANTOM | 9,366 | 
| verify_fibair | IfEq | BNE | 113 | 
| verify_fibair | IfEqI | BNE | 4,531 | 
| verify_fibair | IfEqI | JAL | 2,110 | 
| verify_fibair | IfNe | BEQ | 115 | 
| verify_fibair | IfNe | JAL | 2 | 
| verify_fibair | IfNeI | BEQ | 3 | 
| verify_fibair | ImmE | ADD | 848 | 
| verify_fibair | ImmF | ADD | 1,731 | 
| verify_fibair | ImmV | ADD | 3,673 | 
| verify_fibair | LoadE | ADD | 2,058 | 
| verify_fibair | LoadE | LOADW | 4,128 | 
| verify_fibair | LoadE | MUL | 2,058 | 
| verify_fibair | LoadF | ADD | 673 | 
| verify_fibair | LoadF | LOADW | 3,047 | 
| verify_fibair | LoadF | MUL | 348 | 
| verify_fibair | LoadHeapPtr | ADD | 1 | 
| verify_fibair | LoadV | ADD | 4,535 | 
| verify_fibair | LoadV | LOADW | 21,089 | 
| verify_fibair | LoadV | MUL | 3,818 | 
| verify_fibair | MulE | BBE4MUL | 2,717 | 
| verify_fibair | MulEF | MUL | 2,528 | 
| verify_fibair | MulEI | ADD | 308 | 
| verify_fibair | MulEI | BBE4MUL | 77 | 
| verify_fibair | MulF | MUL | 5,705 | 
| verify_fibair | MulFI | MUL | 1,334 | 
| verify_fibair | MulVI | MUL | 1,828 | 
| verify_fibair | Poseidon2PermuteBabyBear | PERM_POS2 | 26 | 
| verify_fibair | StoreE | ADD | 798 | 
| verify_fibair | StoreE | MUL | 798 | 
| verify_fibair | StoreE | STOREW | 5,398 | 
| verify_fibair | StoreF | ADD | 183 | 
| verify_fibair | StoreF | MUL | 10 | 
| verify_fibair | StoreF | STOREW | 1,695 | 
| verify_fibair | StoreHeapPtr | ADD | 1 | 
| verify_fibair | StoreHintWord | HINT_STOREW | 71,974 | 
| verify_fibair | StoreV | ADD | 907 | 
| verify_fibair | StoreV | MUL | 217 | 
| verify_fibair | StoreV | STOREW | 19,035 | 
| verify_fibair | SubE | FE4SUB | 1,892 | 
| verify_fibair | SubEF | ADD | 393 | 
| verify_fibair | SubEF | SUB | 131 | 
| verify_fibair | SubEI | ADD | 8 | 
| verify_fibair | SubFI | SUB | 1,333 | 
| verify_fibair | SubV | SUB | 1,513 | 
| verify_fibair | SubVI | SUB | 745 | 
| verify_fibair | SubVIN | SUB | 630 | 
| verify_fibair | UnsafeCastVF | ADD | 1 | 
| verify_fibair | VerifyBatchExt | VERIFY_BATCH | 630 | 
| verify_fibair | VerifyBatchFelt | VERIFY_BATCH | 84 | 
| verify_fibair | ZipFor | ADD | 83,558 | 
| verify_fibair | ZipFor | BNE | 81,526 | 
| verify_fibair | ZipFor | JAL | 10,555 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/e648d7cf909099ea6cd6465428ad18caf9cb79ba/verify_fibair-e648d7cf909099ea6cd6465428ad18caf9cb79ba-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/e648d7cf909099ea6cd6465428ad18caf9cb79ba

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12848088425)
