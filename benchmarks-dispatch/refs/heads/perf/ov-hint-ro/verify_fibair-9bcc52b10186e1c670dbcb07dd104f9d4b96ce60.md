| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+32.4%])</span> 1.98 | <span style='color: red'>(+0 [+32.4%])</span> 1.98 |
| verify_fibair | <span style='color: red'>(+0 [+32.4%])</span> 1.98 | <span style='color: red'>(+0 [+32.4%])</span> 1.98 |


| verify_fibair |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+484 [+32.4%])</span> 1,980 | <span style='color: red'>(+484 [+32.4%])</span> 1,980 | <span style='color: red'>(+484 [+32.4%])</span> 1,980 | <span style='color: red'>(+484 [+32.4%])</span> 1,980 |
| `main_cells_used     ` | <span style='color: green'>(-269615 [-2.8%])</span> 9,495,390 | <span style='color: green'>(-269615 [-2.8%])</span> 9,495,390 | <span style='color: green'>(-269615 [-2.8%])</span> 9,495,390 | <span style='color: green'>(-269615 [-2.8%])</span> 9,495,390 |
| `total_cycles        ` | <span style='color: green'>(-9299 [-5.0%])</span> 178,086 | <span style='color: green'>(-9299 [-5.0%])</span> 178,086 | <span style='color: green'>(-9299 [-5.0%])</span> 178,086 | <span style='color: green'>(-9299 [-5.0%])</span> 178,086 |
| `execute_time_ms     ` | <span style='color: red'>(+519 [+489.6%])</span> 625 | <span style='color: red'>(+519 [+489.6%])</span> 625 | <span style='color: red'>(+519 [+489.6%])</span> 625 | <span style='color: red'>(+519 [+489.6%])</span> 625 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-5 [-3.8%])</span> 125 | <span style='color: green'>(-5 [-3.8%])</span> 125 | <span style='color: green'>(-5 [-3.8%])</span> 125 | <span style='color: green'>(-5 [-3.8%])</span> 125 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-30 [-2.4%])</span> 1,230 | <span style='color: green'>(-30 [-2.4%])</span> 1,230 | <span style='color: green'>(-30 [-2.4%])</span> 1,230 | <span style='color: green'>(-30 [-2.4%])</span> 1,230 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.9%])</span> 219 | <span style='color: green'>(-2 [-0.9%])</span> 219 | <span style='color: green'>(-2 [-0.9%])</span> 219 | <span style='color: green'>(-2 [-0.9%])</span> 219 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+9.1%])</span> 24 | <span style='color: red'>(+2 [+9.1%])</span> 24 | <span style='color: red'>(+2 [+9.1%])</span> 24 | <span style='color: red'>(+2 [+9.1%])</span> 24 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.5%])</span> 203 | <span style='color: green'>(-1 [-0.5%])</span> 203 | <span style='color: green'>(-1 [-0.5%])</span> 203 | <span style='color: green'>(-1 [-0.5%])</span> 203 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-9 [-6.2%])</span> 136 | <span style='color: green'>(-9 [-6.2%])</span> 136 | <span style='color: green'>(-9 [-6.2%])</span> 136 | <span style='color: green'>(-9 [-6.2%])</span> 136 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-7 [-2.4%])</span> 284 | <span style='color: green'>(-7 [-2.4%])</span> 284 | <span style='color: green'>(-7 [-2.4%])</span> 284 | <span style='color: green'>(-7 [-2.4%])</span> 284 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-14 [-3.7%])</span> 360 | <span style='color: green'>(-14 [-3.7%])</span> 360 | <span style='color: green'>(-14 [-3.7%])</span> 360 | <span style='color: green'>(-14 [-3.7%])</span> 360 |



<details>
<summary>Detailed Metrics</summary>

|  | verify_program_compile_ms | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 4 | 65,536 | 58 | 2 | 11 | 0 | 31 | 12 | 

| air_name | rows | quotient_deg | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- |
| AccessAdapterAir<2> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<4> |  | 4 |  | 5 | 11 |  | 
| AccessAdapterAir<8> |  | 4 |  | 5 | 11 |  | 
| FibonacciAir | 32,768 | 1 | 2 |  | 5 | 65,536 | 
| FriReducedOpeningAir |  | 4 |  | 39 | 59 |  | 
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
| verify_fibair | 125 | 1,980 | 178,086 | 25,418,392 | 1,230 | 136 | 284 | 203 | 360 | 219 | 9,495,390 | 24 | 625 | 

| group | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- |
| verify_fibair | AccessAdapterAir<2> | 65,536 |  | 12 | 11 | 1,507,328 | 
| verify_fibair | AccessAdapterAir<4> | 32,768 |  | 12 | 13 | 819,200 | 
| verify_fibair | AccessAdapterAir<8> | 128 |  | 12 | 17 | 3,712 | 
| verify_fibair | FriReducedOpeningAir | 1,024 |  | 44 | 27 | 72,704 | 
| verify_fibair | NativePoseidon2Air<BabyBearParameters>, 1> | 16,384 |  | 160 | 399 | 9,158,656 | 
| verify_fibair | PhantomAir | 4,096 |  | 8 | 6 | 57,344 | 
| verify_fibair | ProgramAir | 8,192 |  | 8 | 10 | 147,456 | 
| verify_fibair | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| verify_fibair | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 131,072 |  | 20 | 29 | 6,422,528 | 
| verify_fibair | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 32,768 |  | 16 | 23 | 1,277,952 | 
| verify_fibair | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4,096 |  | 12 | 9 | 86,016 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 32,768 |  | 24 | 22 | 1,507,328 | 
| verify_fibair | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 16,384 |  | 24 | 31 | 901,120 | 
| verify_fibair | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8,192 |  | 20 | 38 | 475,136 | 
| verify_fibair | VmConnectorAir | 2 | 1 | 8 | 4 | 24 | 
| verify_fibair | VolatileBoundaryAir | 32,768 |  | 8 | 11 | 622,592 | 

| group | air_name | dsl_ir | opcode | cells_used |
| --- | --- | --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | ADD | 13,224 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | ADD | 242,092 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | ADD | 45,675 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | ADD | 91,205 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | ADD | 220,603 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | ADD | 264,306 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | ADD | 336,168 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | MUL | 81,606 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | ADD | 1,305 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | ADD | 116 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | DIV | 41,412 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | DIV | 87 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | ADD | 24,824 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | ADD | 54,665 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | ADD | 109,040 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | ADD | 59,682 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | MUL | 59,682 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | ADD | 7,598 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | MUL | 580 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | ADD | 95,845 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | MUL | 77,430 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | MUL | 73,312 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | ADD | 8,932 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | MUL | 206,219 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | MUL | 40,484 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | MUL | 27,579 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | ADD | 23,142 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | MUL | 23,142 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | ADD | 5,307 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | MUL | 290 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | ADD | 29 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | ADD | 19,169 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | MUL | 1,537 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | ADD | 11,397 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | SUB | 3,799 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | ADD | 232 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | SUB | 40,455 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | SUB | 43,877 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | SUB | 21,605 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | SUB | 18,270 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | ADD | 87 | 
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | ADD | 541,807 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | BNE | 3,956 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | BNE | 92 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | BNE | 33,120 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | BNE | 17,595 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | BNE | 483 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | BNE | 2,599 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | BNE | 102,902 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | BEQ | 2,645 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | BEQ | 46 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | BNE | 375,958 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> |  | JAL | 9 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfEqI | JAL | 19,206 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | IfNe | JAL | 18 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | ZipFor | JAL | 17,271 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | LOADW | 57,618 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | LOADW | 279,576 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | STOREW | 37,290 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | HINT_STOREW | 155,870 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | STOREW | 97,834 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | LOADW | 130,634 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | STOREW | 168,361 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | FE4ADD | 55,062 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | BBE4DIV | 28,804 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | BBE4DIV | 38 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | BBE4MUL | 114,076 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | BBE4MUL | 2,926 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | FE4SUB | 71,896 | 
| verify_fibair | FriReducedOpeningAir | FriReducedOpening | FRI_REDUCED_OPENING | 15,876 | 
| verify_fibair | PhantomAir | HintBitsF | PHANTOM | 270 | 
| verify_fibair | PhantomAir | HintFelt | PHANTOM | 9,186 | 
| verify_fibair | PhantomAir | HintInputVec | PHANTOM | 3,918 | 
| verify_fibair | PhantomAir | HintLoad | PHANTOM | 4,788 | 
| verify_fibair | VerifyBatchAir | Poseidon2PermuteBabyBear | PERM_POS2 | 10,374 | 
| verify_fibair | VerifyBatchAir | VerifyBatchExt | VERIFY_BATCH | 2,765,070 | 
| verify_fibair | VerifyBatchAir | VerifyBatchFelt | VERIFY_BATCH | 636,804 | 

| group | chip_name | rows_used |
| --- | --- | --- |
| verify_fibair | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 101,307 | 
| verify_fibair | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 23,452 | 
| verify_fibair | <JalNativeAdapterAir,JalCoreAir> | 4,056 | 
| verify_fibair | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 28,554 | 
| verify_fibair | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 9,645 | 
| verify_fibair | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 7,179 | 
| verify_fibair | AccessAdapter<2> | 41,950 | 
| verify_fibair | AccessAdapter<4> | 20,344 | 
| verify_fibair | AccessAdapter<8> | 88 | 
| verify_fibair | Boundary | 30,947 | 
| verify_fibair | FriReducedOpeningAir | 588 | 
| verify_fibair | PhantomAir | 3,027 | 
| verify_fibair | ProgramChip | 5,329 | 
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
| verify_fibair | AddV | ADD | 7,607 | 
| verify_fibair | AddVI | ADD | 9,114 | 
| verify_fibair | Alloc | ADD | 11,592 | 
| verify_fibair | Alloc | MUL | 2,814 | 
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
| verify_fibair | HintInputVec | PHANTOM | 653 | 
| verify_fibair | HintLoad | PHANTOM | 798 | 
| verify_fibair | IfEq | BNE | 113 | 
| verify_fibair | IfEqI | BNE | 4,474 | 
| verify_fibair | IfEqI | JAL | 2,134 | 
| verify_fibair | IfNe | BEQ | 115 | 
| verify_fibair | IfNe | JAL | 2 | 
| verify_fibair | IfNeI | BEQ | 2 | 
| verify_fibair | ImmE | ADD | 856 | 
| verify_fibair | ImmF | ADD | 1,885 | 
| verify_fibair | ImmV | ADD | 3,760 | 
| verify_fibair | LoadE | ADD | 2,058 | 
| verify_fibair | LoadE | LOADW | 4,214 | 
| verify_fibair | LoadE | MUL | 2,058 | 
| verify_fibair | LoadF | ADD | 262 | 
| verify_fibair | LoadF | LOADW | 2,619 | 
| verify_fibair | LoadF | MUL | 20 | 
| verify_fibair | LoadHeapPtr | ADD | 1 | 
| verify_fibair | LoadV | ADD | 3,305 | 
| verify_fibair | LoadV | LOADW | 12,708 | 
| verify_fibair | LoadV | MUL | 2,670 | 
| verify_fibair | MulE | BBE4MUL | 3,002 | 
| verify_fibair | MulEF | MUL | 2,528 | 
| verify_fibair | MulEI | ADD | 308 | 
| verify_fibair | MulEI | BBE4MUL | 77 | 
| verify_fibair | MulF | MUL | 7,111 | 
| verify_fibair | MulFI | MUL | 1,396 | 
| verify_fibair | MulVI | MUL | 951 | 
| verify_fibair | Poseidon2PermuteBabyBear | PERM_POS2 | 26 | 
| verify_fibair | StoreE | ADD | 798 | 
| verify_fibair | StoreE | MUL | 798 | 
| verify_fibair | StoreE | STOREW | 5,431 | 
| verify_fibair | StoreF | ADD | 183 | 
| verify_fibair | StoreF | MUL | 10 | 
| verify_fibair | StoreF | STOREW | 1,695 | 
| verify_fibair | StoreHeapPtr | ADD | 1 | 
| verify_fibair | StoreHintWord | HINT_STOREW | 7,085 | 
| verify_fibair | StoreV | ADD | 661 | 
| verify_fibair | StoreV | MUL | 53 | 
| verify_fibair | StoreV | STOREW | 4,447 | 
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
| verify_fibair | ZipFor | ADD | 18,683 | 
| verify_fibair | ZipFor | BNE | 16,346 | 
| verify_fibair | ZipFor | JAL | 1,919 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60/verify_fibair-9bcc52b10186e1c670dbcb07dd104f9d4b96ce60-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/9bcc52b10186e1c670dbcb07dd104f9d4b96ce60

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13200174488)
