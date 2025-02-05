| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+33.5%])</span> 2.06 | <span style='color: red'>(+1 [+33.5%])</span> 2.06 |
| verify_fibair | <span style='color: red'>(+1 [+33.5%])</span> 2.06 | <span style='color: red'>(+1 [+33.5%])</span> 2.06 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+518 [+33.5%])</span> 2,062 | <span style='color: red'>(+518 [+33.5%])</span> 2,062 | <span style='color: red'>(+518 [+33.5%])</span> 2,062 | <span style='color: red'>(+518 [+33.5%])</span> 2,062 |
| `main_cells_used     ` |  9,765,248 |  9,765,248 |  9,765,248 |  9,765,248 |
| `total_cycles        ` |  187,412 |  187,412 |  187,412 |  187,412 |
| `execute_time_ms     ` | <span style='color: red'>(+549 [+517.9%])</span> 655 | <span style='color: red'>(+549 [+517.9%])</span> 655 | <span style='color: red'>(+549 [+517.9%])</span> 655 | <span style='color: red'>(+549 [+517.9%])</span> 655 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+1 [+0.8%])</span> 134 | <span style='color: red'>(+1 [+0.8%])</span> 134 | <span style='color: red'>(+1 [+0.8%])</span> 134 | <span style='color: red'>(+1 [+0.8%])</span> 134 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-32 [-2.5%])</span> 1,273 | <span style='color: green'>(-32 [-2.5%])</span> 1,273 | <span style='color: green'>(-32 [-2.5%])</span> 1,273 | <span style='color: green'>(-32 [-2.5%])</span> 1,273 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-10 [-4.4%])</span> 219 | <span style='color: green'>(-10 [-4.4%])</span> 219 | <span style='color: green'>(-10 [-4.4%])</span> 219 | <span style='color: green'>(-10 [-4.4%])</span> 219 |
| `generate_perm_trace_time_ms` |  22 |  22 |  22 |  22 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-4 [-1.9%])</span> 203 | <span style='color: green'>(-4 [-1.9%])</span> 203 | <span style='color: green'>(-4 [-1.9%])</span> 203 | <span style='color: green'>(-4 [-1.9%])</span> 203 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-3.4%])</span> 141 | <span style='color: green'>(-5 [-3.4%])</span> 141 | <span style='color: green'>(-5 [-3.4%])</span> 141 | <span style='color: green'>(-5 [-3.4%])</span> 141 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.7%])</span> 307 | <span style='color: red'>(+2 [+0.7%])</span> 307 | <span style='color: red'>(+2 [+0.7%])</span> 307 | <span style='color: red'>(+2 [+0.7%])</span> 307 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-16 [-4.1%])</span> 377 | <span style='color: green'>(-16 [-4.1%])</span> 377 | <span style='color: green'>(-16 [-4.1%])</span> 377 | <span style='color: green'>(-16 [-4.1%])</span> 377 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 5 | 65,536 | 62 | 3 | 13 | 0 | 32 | 13 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 31 | 52 |  | 
| NativePoseidon2Air<BabyBearParameters>, 1> |  | 4 |  | 136 | 530 |  | 
| PhantomAir |  | 4 |  | 3 | 4 |  | 
| ProgramAir |  | 1 |  | 1 | 4 |  | 
| VariableRangeCheckerAir |  | 1 |  | 1 | 4 |  | 
| VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<JalNativeAdapterAir, JalCoreAir> |  | 4 |  | 7 | 6 |  | 
| VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> |  | 4 |  | 11 | 22 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> |  | 4 |  | 15 | 16 |  | 
| VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> |  | 4 |  | 15 | 23 |  | 
| VmConnectorAir |  | 4 |  | 3 | 8 |  | 
| VolatileBoundaryAir |  | 4 |  | 4 | 16 |  | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | 134 | 2,062 | 187,412 | 26,116,760 | 1,273 | 141 | 307 | 203 | 377 | 219 | 9,765,248 | 22 | 655 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 36 | 25 | 62,464 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 16 | 23 | 1,277,952 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 8,192 |  | 12 | 9 | 172,032 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 65,536 |  | 8 | 11 | 1,245,184 | 

| group | air_name | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | ADD | 13,224 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | ADD | 242,092 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | ADD | 45,675 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | ADD | 91,205 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | ADD | 225,475 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | ADD | 283,939 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | ADD | 360,006 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | MUL | 93,525 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | ADD | 1,305 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | ADD | 116 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | DIV | 41,412 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | DIV | 87 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | ADD | 24,824 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | ADD | 54,665 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | ADD | 106,604 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | ADD | 59,682 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | MUL | 59,682 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | ADD | 19,517 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | MUL | 10,092 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | ADD | 131,515 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | MUL | 110,722 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | MUL | 73,312 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | ADD | 8,932 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | MUL | 206,219 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | MUL | 40,484 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | MUL | 32,393 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | ADD | 23,142 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | MUL | 23,142 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | ADD | 5,307 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | MUL | 290 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | ADD | 26,303 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | MUL | 6,293 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | ADD | 11,397 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | SUB | 3,799 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | ADD | 232 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | SUB | 40,455 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | SUB | 43,877 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | SUB | 21,605 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | SUB | 18,270 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | ADD | 87 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | ADD | 561,063 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | BNE | 3,956 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | BNE | 92 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | BNE | 33,120 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | BNE | 17,595 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | BNE | 483 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | BNE | 2,599 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | BNE | 106,674 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | BEQ | 2,645 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | BEQ | 46 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | BNE | 391,322 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> |  | JAL | 9 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfEqI | JAL | 19,449 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfNe | JAL | 18 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | ZipFor | JAL | 19,521 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | LOADW | 68,486 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | LOADW | 295,504 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | STOREW | 37,290 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | HINT_STOREW | 161,414 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | STOREW | 106,546 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | LOADW | 130,634 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | STOREW | 168,361 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | FE4ADD | 55,062 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | BBE4DIV | 28,804 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | BBE4DIV | 38 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | BBE4MUL | 114,076 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | BBE4MUL | 2,926 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | FE4SUB | 71,896 | 
| verify_fibair | FriReducedOpeningAir | FriReducedOpening | FRI_REDUCED_OPENING | 14,700 | 
| verify_fibair | PhantomAir | HintBitsF | PHANTOM | 270 | 
| verify_fibair | PhantomAir | HintFelt | PHANTOM | 9,186 | 
| verify_fibair | PhantomAir | HintInputVec | PHANTOM | 4,422 | 
| verify_fibair | PhantomAir | HintLoad | PHANTOM | 4,284 | 
| verify_fibair | VerifyBatchAir | Poseidon2PermuteBabyBear | PERM_POS2 | 10,374 | 
| verify_fibair | VerifyBatchAir | VerifyBatchExt | VERIFY_BATCH | 2,765,070 | 
| verify_fibair | VerifyBatchAir | VerifyBatchFelt | VERIFY_BATCH | 636,804 | 

| group | chip_name | rows_used |
| --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 107,658 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 24,284 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | 4,333 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 30,420 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 9,645 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 7,179 | 
| verify_fibair | AccessAdapter<2> | 42,114 | 
| verify_fibair | AccessAdapter<4> | 20,344 | 
| verify_fibair | AccessAdapter<8> | 88 | 
| verify_fibair | Boundary | 32,982 | 
| verify_fibair | FriReducedOpeningAir | 588 | 
| verify_fibair | PhantomAir | 3,027 | 
| verify_fibair | ProgramChip | 5,314 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 
| verify_fibair | VerifyBatchAir | 8,552 | 
| verify_fibair | VmConnectorAir | 2 | 

| group | dsl_ir | opcode | frequency |
| --- | --- | --- | --- |
| verify_fibair |  | ADD | 2 | 
| verify_fibair |  | JAL | 1 | 
| verify_fibair | AddE | FE4ADD | 1,449 | 
| verify_fibair | AddEFFI | ADD | 456 | 
| verify_fibair | AddEI | ADD | 8,348 | 
| verify_fibair | AddF | ADD | 1,575 | 
| verify_fibair | AddFI | ADD | 3,145 | 
| verify_fibair | AddV | ADD | 7,775 | 
| verify_fibair | AddVI | ADD | 9,791 | 
| verify_fibair | Alloc | ADD | 12,414 | 
| verify_fibair | Alloc | MUL | 3,225 | 
| verify_fibair | AssertEqE | BNE | 172 | 
| verify_fibair | AssertEqEI | BNE | 4 | 
| verify_fibair | AssertEqF | BNE | 1,440 | 
| verify_fibair | AssertEqV | BNE | 765 | 
| verify_fibair | AssertEqVI | BNE | 21 | 
| verify_fibair | CastFV | ADD | 45 | 
| verify_fibair | DivE | BBE4DIV | 758 | 
| verify_fibair | DivEIN | ADD | 4 | 
| verify_fibair | DivEIN | BBE4DIV | 1 | 
| verify_fibair | DivF | DIV | 1,428 | 
| verify_fibair | DivFIN | DIV | 3 | 
| verify_fibair | FriReducedOpening | FRI_REDUCED_OPENING | 126 | 
| verify_fibair | HintBitsF | PHANTOM | 45 | 
| verify_fibair | HintFelt | PHANTOM | 1,531 | 
| verify_fibair | HintInputVec | PHANTOM | 737 | 
| verify_fibair | HintLoad | PHANTOM | 714 | 
| verify_fibair | IfEq | BNE | 113 | 
| verify_fibair | IfEqI | BNE | 4,638 | 
| verify_fibair | IfEqI | JAL | 2,161 | 
| verify_fibair | IfNe | BEQ | 115 | 
| verify_fibair | IfNe | JAL | 2 | 
| verify_fibair | IfNeI | BEQ | 2 | 
| verify_fibair | ImmE | ADD | 856 | 
| verify_fibair | ImmF | ADD | 1,885 | 
| verify_fibair | ImmV | ADD | 3,676 | 
| verify_fibair | LoadE | ADD | 2,058 | 
| verify_fibair | LoadE | LOADW | 4,214 | 
| verify_fibair | LoadE | MUL | 2,058 | 
| verify_fibair | LoadF | ADD | 673 | 
| verify_fibair | LoadF | LOADW | 3,113 | 
| verify_fibair | LoadF | MUL | 348 | 
| verify_fibair | LoadHeapPtr | ADD | 1 | 
| verify_fibair | LoadV | ADD | 4,535 | 
| verify_fibair | LoadV | LOADW | 13,432 | 
| verify_fibair | LoadV | MUL | 3,818 | 
| verify_fibair | MulE | BBE4MUL | 3,002 | 
| verify_fibair | MulEF | MUL | 2,528 | 
| verify_fibair | MulEI | ADD | 308 | 
| verify_fibair | MulEI | BBE4MUL | 77 | 
| verify_fibair | MulF | MUL | 7,111 | 
| verify_fibair | MulFI | MUL | 1,396 | 
| verify_fibair | MulVI | MUL | 1,117 | 
| verify_fibair | Poseidon2PermuteBabyBear | PERM_POS2 | 26 | 
| verify_fibair | StoreE | ADD | 798 | 
| verify_fibair | StoreE | MUL | 798 | 
| verify_fibair | StoreE | STOREW | 5,431 | 
| verify_fibair | StoreF | ADD | 183 | 
| verify_fibair | StoreF | MUL | 10 | 
| verify_fibair | StoreF | STOREW | 1,695 | 
| verify_fibair | StoreHeapPtr | ADD | 1 | 
| verify_fibair | StoreHintWord | HINT_STOREW | 7,337 | 
| verify_fibair | StoreV | ADD | 907 | 
| verify_fibair | StoreV | MUL | 217 | 
| verify_fibair | StoreV | STOREW | 4,843 | 
| verify_fibair | SubE | FE4SUB | 1,892 | 
| verify_fibair | SubEF | ADD | 393 | 
| verify_fibair | SubEF | SUB | 131 | 
| verify_fibair | SubEI | ADD | 8 | 
| verify_fibair | SubFI | SUB | 1,395 | 
| verify_fibair | SubV | SUB | 1,513 | 
| verify_fibair | SubVI | SUB | 745 | 
| verify_fibair | SubVIN | SUB | 630 | 
| verify_fibair | UnsafeCastVF | ADD | 3 | 
| verify_fibair | VerifyBatchExt | VERIFY_BATCH | 630 | 
| verify_fibair | VerifyBatchFelt | VERIFY_BATCH | 84 | 
| verify_fibair | ZipFor | ADD | 19,347 | 
| verify_fibair | ZipFor | BNE | 17,014 | 
| verify_fibair | ZipFor | JAL | 2,169 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/verify_fibair-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/9d1c71639c516da305d7ec13e7ee959df8b5b3ee

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13163606004)
