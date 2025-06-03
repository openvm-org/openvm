| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+77.6%])</span> 2.09 | <span style='color: red'>(+1 [+77.6%])</span> 2.09 |
| verify_fibair | <span style='color: red'>(+1 [+77.6%])</span> 2.09 | <span style='color: red'>(+1 [+77.6%])</span> 2.09 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+915 [+77.6%])</span> 2,094 | <span style='color: red'>(+915 [+77.6%])</span> 2,094 | <span style='color: red'>(+915 [+77.6%])</span> 2,094 | <span style='color: red'>(+915 [+77.6%])</span> 2,094 |
| `main_cells_used     ` |  17,339,520 |  17,339,520 |  17,339,520 |  17,339,520 |
| `total_cycles        ` |  322,648 |  322,648 |  322,648 |  322,648 |
| `execute_time_ms     ` | <span style='color: red'>(+899 [+475.7%])</span> 1,088 | <span style='color: red'>(+899 [+475.7%])</span> 1,088 | <span style='color: red'>(+899 [+475.7%])</span> 1,088 | <span style='color: red'>(+899 [+475.7%])</span> 1,088 |
| `trace_gen_time_ms   ` |  180 |  180 |  180 |  180 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+16 [+2.0%])</span> 826 | <span style='color: red'>(+16 [+2.0%])</span> 826 | <span style='color: red'>(+16 [+2.0%])</span> 826 | <span style='color: red'>(+16 [+2.0%])</span> 826 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+9 [+6.1%])</span> 157 | <span style='color: red'>(+9 [+6.1%])</span> 157 | <span style='color: red'>(+9 [+6.1%])</span> 157 | <span style='color: red'>(+9 [+6.1%])</span> 157 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+5 [+12.8%])</span> 44 | <span style='color: red'>(+5 [+12.8%])</span> 44 | <span style='color: red'>(+5 [+12.8%])</span> 44 | <span style='color: red'>(+5 [+12.8%])</span> 44 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+10 [+7.3%])</span> 147 | <span style='color: red'>(+10 [+7.3%])</span> 147 | <span style='color: red'>(+10 [+7.3%])</span> 147 | <span style='color: red'>(+10 [+7.3%])</span> 147 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-9 [-9.7%])</span> 84 | <span style='color: green'>(-9 [-9.7%])</span> 84 | <span style='color: green'>(-9 [-9.7%])</span> 84 | <span style='color: green'>(-9 [-9.7%])</span> 84 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1 [-1.0%])</span> 97 | <span style='color: green'>(-1 [-1.0%])</span> 97 | <span style='color: green'>(-1 [-1.0%])</span> 97 | <span style='color: green'>(-1 [-1.0%])</span> 97 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-0.7%])</span> 290 | <span style='color: green'>(-2 [-0.7%])</span> 290 | <span style='color: green'>(-2 [-0.7%])</span> 290 | <span style='color: green'>(-2 [-0.7%])</span> 290 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7 | 65,536 | 35 | 1 | 6 | 0 | 20 | 7 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<4> |  | 2 |  | 5 | 12 |  | 
| AccessAdapterAir<8> |  | 2 |  | 5 | 12 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 2 |  | 39 | 71 |  | 
| JalRangeCheckAir |  | 2 |  | 9 | 14 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 2 |  | 136 | 572 |  | 
| PhantomAir |  | 2 |  | 3 | 5 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 2 |  | 11 | 25 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 2 |  | 11 | 29 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 2 |  | 15 | 20 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 2 |  | 15 | 27 |  | 
| VmConnectorAir |  | 2 |  | 5 | 11 |  | 
| VolatileBoundaryAir |  | 2 |  | 7 | 19 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 180 | 2,094 | 322,648 | 62,474,410 | 826 | 84 | 97 | 147 | 290 | 157 | 17,339,520 | 44 | 1,088 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 131,072 |  | 16 | 11 | 3,538,944 | 
| verify_fibair | AccessAdapterAir<4> | 65,536 |  | 16 | 13 | 1,900,544 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 16 | 17 | 4,224 | 
| verify_fibair | FriReducedOpeningAir | 2,048 |  | 84 | 27 | 227,328 | 
| verify_fibair | JalRangeCheckAir | 32,768 |  | 28 | 12 | 1,310,720 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  | 312 | 398 | 23,265,280 | 
| verify_fibair | PhantomAir | 16,384 |  | 12 | 6 | 294,912 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 262,144 |  | 36 | 29 | 17,039,360 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 28 | 23 | 1,671,168 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 65,536 |  | 40 | 21 | 3,997,696 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 32,768 |  | 40 | 27 | 2,195,456 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 32,768 |  | 36 | 38 | 2,424,832 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 20 | 12 | 2,097,152 | 

| group | air_name | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | ADD | 19,952 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | ADD | 553,436 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | ADD | 106,575 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | ADD | 104,516 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | ADD | 127,542 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | ADD | 396,430 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | ADD | 604,244 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | MUL | 185,368 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | ADD | 3,045 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | ADD | 116 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | DIV | 46,400 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | DIV | 87 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | ADD | 46,748 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | ADD | 112,578 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | ADD | 238,351 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | ADD | 58,000 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | MUL | 58,000 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | ADD | 9,309 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | MUL | 580 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | ADD | 4,698 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | MUL | 4,466 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | MUL | 174,232 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | ADD | 8,932 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | MUL | 280,111 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | MUL | 94,424 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | MUL | 783 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | MUL | 24,186 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | ADD | 55,100 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | MUL | 55,100 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | ADD | 754 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | MUL | 290 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | ADD | 45,501 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | MUL | 1,392 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | ADD | 26,535 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | SUB | 8,845 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | ADD | 232 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | SUB | 94,395 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | SUB | 105,183 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | SUB | 5,017 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | SUB | 43,500 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | ADD | 203 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | ADD | 906,975 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | BNE | 9,292 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | BNE | 77,280 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | BNE | 9,637 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | BNE | 598 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | BNE | 8,234 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | BNE | 149,017 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | BEQ | 4,715 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | BEQ | 69 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | BNE | 450,432 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | LOADW | 122,766 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | LOADW | 513,408 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | STOREW | 40,005 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | HINT_STOREW | 282,597 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | STOREW | 218,778 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | LOADW | 310,824 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | STOREW | 325,809 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | FE4ADD | 127,794 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | BBE4DIV | 68,476 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | BBE4DIV | 38 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | BBE4MUL | 318,326 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | BBE4MUL | 2,926 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | FE4SUB | 171,076 | 
| verify_fibair | FriReducedOpeningAir | FriReducedOpening | FRI_REDUCED_OPENING | 37,800 | 
| verify_fibair | JalRangeCheck |  | JAL | 12 | 
| verify_fibair | JalRangeCheck | Alloc | RANGE_CHECK | 201,720 | 
| verify_fibair | JalRangeCheck | IfEqI | JAL | 37,140 | 
| verify_fibair | JalRangeCheck | IfNe | JAL | 24 | 
| verify_fibair | JalRangeCheck | ZipFor | JAL | 33,372 | 
| verify_fibair | PhantomAir | HintBitsF | PHANTOM | 630 | 
| verify_fibair | PhantomAir | HintFelt | PHANTOM | 49,536 | 
| verify_fibair | PhantomAir | HintInputVec | PHANTOM | 36 | 
| verify_fibair | PhantomAir | HintLoad | PHANTOM | 11,400 | 
| verify_fibair | VerifyBatchAir | Poseidon2PermuteBabyBear | PERM_POS2 | 14,328 | 
| verify_fibair | VerifyBatchAir | VerifyBatchExt | VERIFY_BATCH | 5,970,000 | 
| verify_fibair | VerifyBatchAir | VerifyBatchFelt | VERIFY_BATCH | 1,432,800 | 

| group | chip_name | rows_used |
| --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 159,043 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 30,838 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 56,074 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 23,579 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 18,122 | 
| verify_fibair | AccessAdapter<2> | 70,610 | 
| verify_fibair | AccessAdapter<4> | 33,804 | 
| verify_fibair | AccessAdapter<8> | 108 | 
| verify_fibair | Boundary | 42,365 | 
| verify_fibair | FriReducedOpeningAir | 1,400 | 
| verify_fibair | JalRangeCheck | 22,689 | 
| verify_fibair | PhantomAir | 10,267 | 
| verify_fibair | ProgramChip | 7,010 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 
| verify_fibair | VerifyBatchAir | 18,636 | 
| verify_fibair | VmConnectorAir | 2 | 

| group | dsl_ir | opcode | frequency |
| --- | --- | --- | --- |
| verify_fibair |  | ADD | 2 | 
| verify_fibair |  | JAL | 1 | 
| verify_fibair | AddE | FE4ADD | 3,363 | 
| verify_fibair | AddEFFI | ADD | 688 | 
| verify_fibair | AddEI | ADD | 19,084 | 
| verify_fibair | AddF | ADD | 3,675 | 
| verify_fibair | AddFI | ADD | 3,604 | 
| verify_fibair | AddV | ADD | 4,398 | 
| verify_fibair | AddVI | ADD | 13,670 | 
| verify_fibair | Alloc | ADD | 20,836 | 
| verify_fibair | Alloc | MUL | 6,392 | 
| verify_fibair | Alloc | RANGE_CHECK | 16,810 | 
| verify_fibair | AssertEqE | BNE | 404 | 
| verify_fibair | AssertEqF | BNE | 3,360 | 
| verify_fibair | AssertEqV | BNE | 419 | 
| verify_fibair | AssertEqVI | BNE | 26 | 
| verify_fibair | CastFV | ADD | 105 | 
| verify_fibair | DivE | BBE4DIV | 1,802 | 
| verify_fibair | DivEIN | ADD | 4 | 
| verify_fibair | DivEIN | BBE4DIV | 1 | 
| verify_fibair | DivF | DIV | 1,600 | 
| verify_fibair | DivFIN | DIV | 3 | 
| verify_fibair | FriReducedOpening | FRI_REDUCED_OPENING | 300 | 
| verify_fibair | HintBitsF | PHANTOM | 105 | 
| verify_fibair | HintFelt | PHANTOM | 8,256 | 
| verify_fibair | HintInputVec | PHANTOM | 6 | 
| verify_fibair | HintLoad | PHANTOM | 1,900 | 
| verify_fibair | IfEq | BNE | 358 | 
| verify_fibair | IfEqI | BNE | 6,479 | 
| verify_fibair | IfEqI | JAL | 3,095 | 
| verify_fibair | IfNe | BEQ | 205 | 
| verify_fibair | IfNe | JAL | 2 | 
| verify_fibair | IfNeI | BEQ | 3 | 
| verify_fibair | ImmE | ADD | 1,612 | 
| verify_fibair | ImmF | ADD | 3,882 | 
| verify_fibair | ImmV | ADD | 8,219 | 
| verify_fibair | LoadE | ADD | 2,000 | 
| verify_fibair | LoadE | LOADW | 11,512 | 
| verify_fibair | LoadE | MUL | 2,000 | 
| verify_fibair | LoadF | ADD | 321 | 
| verify_fibair | LoadF | LOADW | 5,846 | 
| verify_fibair | LoadF | MUL | 20 | 
| verify_fibair | LoadHeapPtr | ADD | 1 | 
| verify_fibair | LoadV | ADD | 162 | 
| verify_fibair | LoadV | LOADW | 24,448 | 
| verify_fibair | LoadV | MUL | 154 | 
| verify_fibair | MulE | BBE4MUL | 8,377 | 
| verify_fibair | MulEF | MUL | 6,008 | 
| verify_fibair | MulEI | ADD | 308 | 
| verify_fibair | MulEI | BBE4MUL | 77 | 
| verify_fibair | MulF | MUL | 9,659 | 
| verify_fibair | MulFI | MUL | 3,256 | 
| verify_fibair | MulV | MUL | 27 | 
| verify_fibair | MulVI | MUL | 834 | 
| verify_fibair | Poseidon2PermuteBabyBear | PERM_POS2 | 36 | 
| verify_fibair | StoreE | ADD | 1,900 | 
| verify_fibair | StoreE | MUL | 1,900 | 
| verify_fibair | StoreE | STOREW | 12,067 | 
| verify_fibair | StoreF | ADD | 26 | 
| verify_fibair | StoreF | MUL | 10 | 
| verify_fibair | StoreF | STOREW | 1,905 | 
| verify_fibair | StoreHeapPtr | ADD | 1 | 
| verify_fibair | StoreHintWord | HINT_STOREW | 13,457 | 
| verify_fibair | StoreV | ADD | 1,569 | 
| verify_fibair | StoreV | MUL | 48 | 
| verify_fibair | StoreV | STOREW | 10,418 | 
| verify_fibair | SubE | FE4SUB | 4,502 | 
| verify_fibair | SubEF | ADD | 915 | 
| verify_fibair | SubEF | SUB | 305 | 
| verify_fibair | SubEI | ADD | 8 | 
| verify_fibair | SubFI | SUB | 3,255 | 
| verify_fibair | SubV | SUB | 3,627 | 
| verify_fibair | SubVI | SUB | 173 | 
| verify_fibair | SubVIN | SUB | 1,500 | 
| verify_fibair | UnsafeCastVF | ADD | 7 | 
| verify_fibair | VerifyBatchExt | VERIFY_BATCH | 1,500 | 
| verify_fibair | VerifyBatchFelt | VERIFY_BATCH | 200 | 
| verify_fibair | ZipFor | ADD | 31,275 | 
| verify_fibair | ZipFor | BNE | 19,584 | 
| verify_fibair | ZipFor | JAL | 2,781 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| verify_fibair | 0 | 1,085,444 | 2,013,265,921 | 
| verify_fibair | 1 | 5,411,200 | 2,013,265,921 | 
| verify_fibair | 2 | 542,722 | 2,013,265,921 | 
| verify_fibair | 3 | 5,476,612 | 2,013,265,921 | 
| verify_fibair | 4 | 65,536 | 2,013,265,921 | 
| verify_fibair | 5 | 12,851,850 | 2,013,265,921 | 

| trace_height_constraint | threshold |
| --- | --- |
| 0 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/verify_fibair-4a42ad1e761d20fd80dfcf60c5498951d41f308b/verify_fibair-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/4a42ad1e761d20fd80dfcf60c5498951d41f308b

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15429437790)
