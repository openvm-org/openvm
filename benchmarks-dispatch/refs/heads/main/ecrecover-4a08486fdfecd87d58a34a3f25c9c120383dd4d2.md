| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+6 [+48.0%])</span> 19.80 | <span style='color: red'>(+6 [+48.0%])</span> 19.80 |
| ecrecover_program | <span style='color: red'>(+1 [+36.0%])</span> 3.37 | <span style='color: red'>(+1 [+36.0%])</span> 3.37 |
| leaf | <span style='color: red'>(+6 [+50.8%])</span> 16.43 | <span style='color: red'>(+6 [+50.8%])</span> 16.43 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+892 [+36.0%])</span> 3,372 | <span style='color: red'>(+892 [+36.0%])</span> 3,372 | <span style='color: red'>(+892 [+36.0%])</span> 3,372 | <span style='color: red'>(+892 [+36.0%])</span> 3,372 |
| `main_cells_used     ` | <span style='color: red'>(+154384 [+1.0%])</span> 15,210,107 | <span style='color: red'>(+154384 [+1.0%])</span> 15,210,107 | <span style='color: red'>(+154384 [+1.0%])</span> 15,210,107 | <span style='color: red'>(+154384 [+1.0%])</span> 15,210,107 |
| `total_cycles        ` | <span style='color: red'>(+4847 [+1.7%])</span> 289,414 | <span style='color: red'>(+4847 [+1.7%])</span> 289,414 | <span style='color: red'>(+4847 [+1.7%])</span> 289,414 | <span style='color: red'>(+4847 [+1.7%])</span> 289,414 |
| `execute_time_ms     ` | <span style='color: red'>(+916 [+623.1%])</span> 1,063 | <span style='color: red'>(+916 [+623.1%])</span> 1,063 | <span style='color: red'>(+916 [+623.1%])</span> 1,063 | <span style='color: red'>(+916 [+623.1%])</span> 1,063 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-5 [-2.7%])</span> 179 | <span style='color: green'>(-5 [-2.7%])</span> 179 | <span style='color: green'>(-5 [-2.7%])</span> 179 | <span style='color: green'>(-5 [-2.7%])</span> 179 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-19 [-0.9%])</span> 2,130 | <span style='color: green'>(-19 [-0.9%])</span> 2,130 | <span style='color: green'>(-19 [-0.9%])</span> 2,130 | <span style='color: green'>(-19 [-0.9%])</span> 2,130 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+2 [+0.5%])</span> 385 | <span style='color: red'>(+2 [+0.5%])</span> 385 | <span style='color: red'>(+2 [+0.5%])</span> 385 | <span style='color: red'>(+2 [+0.5%])</span> 385 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-2.9%])</span> 33 | <span style='color: green'>(-1 [-2.9%])</span> 33 | <span style='color: green'>(-1 [-2.9%])</span> 33 | <span style='color: green'>(-1 [-2.9%])</span> 33 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-6 [-1.7%])</span> 355 | <span style='color: green'>(-6 [-1.7%])</span> 355 | <span style='color: green'>(-6 [-1.7%])</span> 355 | <span style='color: green'>(-6 [-1.7%])</span> 355 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-17 [-7.3%])</span> 216 | <span style='color: green'>(-17 [-7.3%])</span> 216 | <span style='color: green'>(-17 [-7.3%])</span> 216 | <span style='color: green'>(-17 [-7.3%])</span> 216 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+9 [+1.9%])</span> 476 | <span style='color: red'>(+9 [+1.9%])</span> 476 | <span style='color: red'>(+9 [+1.9%])</span> 476 | <span style='color: red'>(+9 [+1.9%])</span> 476 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-7 [-1.1%])</span> 652 | <span style='color: green'>(-7 [-1.1%])</span> 652 | <span style='color: green'>(-7 [-1.1%])</span> 652 | <span style='color: green'>(-7 [-1.1%])</span> 652 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+5532 [+50.8%])</span> 16,429 | <span style='color: red'>(+5532 [+50.8%])</span> 16,429 | <span style='color: red'>(+5532 [+50.8%])</span> 16,429 | <span style='color: red'>(+5532 [+50.8%])</span> 16,429 |
| `main_cells_used     ` | <span style='color: red'>(+1674206 [+1.4%])</span> 118,660,584 | <span style='color: red'>(+1674206 [+1.4%])</span> 118,660,584 | <span style='color: red'>(+1674206 [+1.4%])</span> 118,660,584 | <span style='color: red'>(+1674206 [+1.4%])</span> 118,660,584 |
| `total_cycles        ` | <span style='color: red'>(+73610 [+4.6%])</span> 1,665,197 | <span style='color: red'>(+73610 [+4.6%])</span> 1,665,197 | <span style='color: red'>(+73610 [+4.6%])</span> 1,665,197 | <span style='color: red'>(+73610 [+4.6%])</span> 1,665,197 |
| `execute_time_ms     ` | <span style='color: red'>(+5541 [+993.0%])</span> 6,099 | <span style='color: red'>(+5541 [+993.0%])</span> 6,099 | <span style='color: red'>(+5541 [+993.0%])</span> 6,099 | <span style='color: red'>(+5541 [+993.0%])</span> 6,099 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-21 [-1.4%])</span> 1,515 | <span style='color: green'>(-21 [-1.4%])</span> 1,515 | <span style='color: green'>(-21 [-1.4%])</span> 1,515 | <span style='color: green'>(-21 [-1.4%])</span> 1,515 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+12 [+0.1%])</span> 8,815 | <span style='color: red'>(+12 [+0.1%])</span> 8,815 | <span style='color: red'>(+12 [+0.1%])</span> 8,815 | <span style='color: red'>(+12 [+0.1%])</span> 8,815 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-18 [-1.0%])</span> 1,767 | <span style='color: green'>(-18 [-1.0%])</span> 1,767 | <span style='color: green'>(-18 [-1.0%])</span> 1,767 | <span style='color: green'>(-18 [-1.0%])</span> 1,767 |
| `generate_perm_trace_time_ms` |  265 |  265 |  265 |  265 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-7 [-0.4%])</span> 1,904 | <span style='color: green'>(-7 [-0.4%])</span> 1,904 | <span style='color: green'>(-7 [-0.4%])</span> 1,904 | <span style='color: green'>(-7 [-0.4%])</span> 1,904 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-21 [-1.4%])</span> 1,446 | <span style='color: green'>(-21 [-1.4%])</span> 1,446 | <span style='color: green'>(-21 [-1.4%])</span> 1,446 | <span style='color: green'>(-21 [-1.4%])</span> 1,446 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+55 [+3.5%])</span> 1,626 | <span style='color: red'>(+55 [+3.5%])</span> 1,626 | <span style='color: red'>(+55 [+3.5%])</span> 1,626 | <span style='color: red'>(+55 [+3.5%])</span> 1,626 |
| `pcs_opening_time_ms ` |  1,802 |  1,802 |  1,802 |  1,802 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 1,161 | 12 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| ecrecover_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 4 | 321 | 4,380 | 
| ecrecover_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| ecrecover_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| ecrecover_program | PhantomAir | 4 | 3 | 4 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 4 | 25 | 217 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 411 | 378 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 4 | 156 | 150 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 422 | 351 | 
| ecrecover_program | VmConnectorAir | 4 | 3 | 8 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 11 | 
| leaf | FriReducedOpeningAir | 4 | 39 | 60 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 136 | 530 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 11 | 22 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 61,016 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 250,560 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 6,168,648 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 260,855 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 76,676 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 362,703 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 876,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 613,524 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 140,592 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 7,482 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 49,300 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 25,578 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 25,723 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 789,380 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 261,870 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 754,290 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 590,730 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 590,730 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 299,773 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 33,843 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 358,817 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 325,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 191,168 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 1,336,320 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 662,244 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 352,843 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 232,116 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 310,822 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 18,792 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 544,446 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 544,446 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 43,674 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 30,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 95,497 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 70,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,078,800 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 359,600 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 346,956 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 98,600 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 231,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 327,497 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 28,159 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 23,142 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 7,279 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 4,887,573 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 7,268 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 189,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 38,180 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 11,339 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,128 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 257,393 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,197 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 4,255 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 2,908,166 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 15,723 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 27 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 108,810 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 849,464 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 2,732,664 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 64,834 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 2,012,868 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 243,958 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,955,883 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 740,962 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 3,516,292 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 500,764 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 16,150 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 3,625,010 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 216,942 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 761,368 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 21,700,224 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 107,856 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 147,672 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 9,576 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 1,542 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 18,420 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 7,140 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 8,820 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,773 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 24,738 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,139,226 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 23,628,780 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 2,631,528 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 559,512 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 250,740 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 318,600 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 900 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 74,407 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 228,536 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 238,023 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 275,912 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 119,834 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 29,984 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 384 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 719,264 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 22,734 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 50,292 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | IS_EQ | 0 | 531,698 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 186,060 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 132,300 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 98,000 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 552,240 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 1,037,520 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 2,701,280 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 285 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 195 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 79,329 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 71,022 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcDouble | 0 | 690,153 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularAddSub | 0 | 2,388 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 8,352 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> |  | EcAddNe | 0 | 449,394 | 
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,560 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 270 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 6,656 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 192 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 160 | 399 | 73,269,248 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | ProgramAir | 0 | 1,048,576 |  | 8 | 10 | 18,874,368 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 16,384 |  | 12 | 9 | 344,064 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 31 | 7,208,960 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 12 | 25 | 606,208 | 
| ecrecover_program | AccessAdapterAir<2> | 0 | 256 |  | 12 | 11 | 5,888 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 12 | 41 | 434,176 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 128 |  | 12 | 13 | 3,200 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 12 | 17 | 950,272 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 532 | 3,163 | 472,960 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 12 | 32 | 180,224 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 8 | 20 | 114,688 | 
| ecrecover_program | PhantomAir | 0 | 64 |  | 8 | 6 | 896 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 2,048 |  | 8 | 300 | 630,784 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 24 | 32 | 14,336 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 28 | 36 | 8,388,608 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 24 | 37 | 124,928 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 28 | 53 | 1,327,104 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 16 | 26 | 688,128 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 20 | 32 | 1,703,936 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 16 | 18 | 139,264 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 32 | 166 | 811,008 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 20 | 28 | 393,216 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 28 | 35 | 258,048 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 28 | 40 | 8,912,896 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 8 |  | 40 | 57 | 776 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 40 | 39 | 632 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 28 | 31 | 241,664 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 16 | 21 | 151,552 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 416 | 543 | 1,964,032 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 160 | 261 | 13,472 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 428 | 619 | 1,072,128 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 853,317 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 148,799 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 13,841 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 268,354 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 86,995 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 227,277 | 
| leaf | AccessAdapter<2> | 0 | 946,348 | 
| leaf | AccessAdapter<4> | 0 | 472,164 | 
| leaf | AccessAdapter<8> | 0 | 342 | 
| leaf | Boundary | 0 | 542,084 | 
| leaf | FriReducedOpeningAir | 0 | 803,712 | 
| leaf | PhantomAir | 0 | 53,049 | 
| leaf | ProgramChip | 0 | 528,573 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 69,683 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 104,480 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 2,011 | 
| ecrecover_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 8,803 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 15,221 | 
| ecrecover_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 23,426 | 
| ecrecover_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 4,057 | 
| ecrecover_program | <Rv32IsEqualModAdapterAir<2, 1, 32, 32>,ModularIsEqualCoreAir<32, 4, 8>> | 0 | 3,194 | 
| ecrecover_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 6,645 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 3,780 | 
| ecrecover_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 109,726 | 
| ecrecover_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 5 | 
| ecrecover_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 2,559 | 
| ecrecover_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 3,383 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 1,271 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>,FieldExpressionCoreAir> | 0 | 21 | 
| ecrecover_program | <Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>,FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | AccessAdapter<16> | 0 | 13,306 | 
| ecrecover_program | AccessAdapter<2> | 0 | 132 | 
| ecrecover_program | AccessAdapter<32> | 0 | 6,654 | 
| ecrecover_program | AccessAdapter<4> | 0 | 68 | 
| ecrecover_program | AccessAdapter<8> | 0 | 27,210 | 
| ecrecover_program | Arc<BabyBearParameters>, 1> | 0 | 2,009 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 2,990 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,226 | 
| ecrecover_program | PhantomAir | 0 | 45 | 
| ecrecover_program | ProgramChip | 0 | 8,596 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 214 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 92,534 | 
| leaf | AddEFFI | 0 | ADD | 2,104 | 
| leaf | AddEFI | 0 | ADD | 8,640 | 
| leaf | AddEI | 0 | ADD | 212,712 | 
| leaf | AddF | 0 | ADD | 8,995 | 
| leaf | AddFI | 0 | ADD | 2,644 | 
| leaf | AddV | 0 | ADD | 12,507 | 
| leaf | AddVI | 0 | ADD | 30,228 | 
| leaf | Alloc | 0 | ADD | 21,156 | 
| leaf | Alloc | 0 | MUL | 4,848 | 
| leaf | AssertEqE | 0 | BNE | 316 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 8,232 | 
| leaf | AssertEqV | 0 | BNE | 1,660 | 
| leaf | AssertEqVI | 0 | BNE | 493 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 672 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 674 | 
| leaf | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 84 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 17,976 | 
| leaf | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 24,612 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,596 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 258 | 
| leaf | DivE | 0 | BBE4DIV | 13,178 | 
| leaf | DivEIN | 0 | ADD | 1,700 | 
| leaf | DivEIN | 0 | BBE4DIV | 425 | 
| leaf | DivF | 0 | DIV | 882 | 
| leaf | DivFIN | 0 | DIV | 887 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 12,306 | 
| leaf | HintBitsF | 0 | PHANTOM | 257 | 
| leaf | HintFelt | 0 | PHANTOM | 3,070 | 
| leaf | HintInputVec | 0 | PHANTOM | 1,190 | 
| leaf | HintLoad | 0 | PHANTOM | 1,470 | 
| leaf | IfEq | 0 | BNE | 136 | 
| leaf | IfEqI | 0 | BNE | 11,191 | 
| leaf | IfEqI | 0 | JAL | 1,747 | 
| leaf | IfNe | 0 | BEQ | 139 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNeI | 0 | BEQ | 185 | 
| leaf | ImmE | 0 | ADD | 27,220 | 
| leaf | ImmF | 0 | ADD | 9,030 | 
| leaf | ImmV | 0 | ADD | 26,010 | 
| leaf | LoadE | 0 | ADD | 20,370 | 
| leaf | LoadE | 0 | LOADW | 63,093 | 
| leaf | LoadE | 0 | MUL | 20,370 | 
| leaf | LoadF | 0 | ADD | 10,337 | 
| leaf | LoadF | 0 | LOADW | 38,612 | 
| leaf | LoadF | 0 | MUL | 1,167 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 12,373 | 
| leaf | LoadV | 0 | LOADW | 124,212 | 
| leaf | LoadV | 0 | MUL | 11,215 | 
| leaf | MulE | 0 | BBE4MUL | 95,395 | 
| leaf | MulEF | 0 | MUL | 6,592 | 
| leaf | MulEFI | 0 | MUL | 46,080 | 
| leaf | MulEI | 0 | ADD | 22,836 | 
| leaf | MulEI | 0 | BBE4MUL | 5,709 | 
| leaf | MulF | 0 | MUL | 12,167 | 
| leaf | MulFI | 0 | MUL | 8,004 | 
| leaf | MulVI | 0 | MUL | 10,718 | 
| leaf | NegE | 0 | MUL | 648 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 62 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 18,774 | 
| leaf | StoreE | 0 | MUL | 18,774 | 
| leaf | StoreE | 0 | STOREW | 23,902 | 
| leaf | StoreF | 0 | ADD | 1,506 | 
| leaf | StoreF | 0 | MUL | 1,052 | 
| leaf | StoreF | 0 | STOREW | 2,947 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 91,494 | 
| leaf | StoreV | 0 | ADD | 3,293 | 
| leaf | StoreV | 0 | MUL | 2,435 | 
| leaf | StoreV | 0 | STOREW | 11,089 | 
| leaf | SubE | 0 | FE4SUB | 20,036 | 
| leaf | SubEF | 0 | ADD | 37,200 | 
| leaf | SubEF | 0 | SUB | 12,400 | 
| leaf | SubEFI | 0 | ADD | 11,964 | 
| leaf | SubEI | 0 | ADD | 3,400 | 
| leaf | SubFI | 0 | SUB | 7,967 | 
| leaf | SubV | 0 | SUB | 11,293 | 
| leaf | SubVI | 0 | SUB | 971 | 
| leaf | SubVIN | 0 | SUB | 798 | 
| leaf | UnsafeCastVF | 0 | ADD | 251 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 798 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 168,537 | 
| leaf | ZipFor | 0 | BNE | 126,442 | 
| leaf | ZipFor | 0 | JAL | 12,090 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 73,098 | 
| ecrecover_program |  | AND | 0 | 15,542 | 
| ecrecover_program |  | AUIPC | 0 | 3,383 | 
| ecrecover_program |  | BEQ | 0 | 10,612 | 
| ecrecover_program |  | BGEU | 0 | 937 | 
| ecrecover_program |  | BLT | 0 | 12 | 
| ecrecover_program |  | BLTU | 0 | 22,477 | 
| ecrecover_program |  | BNE | 0 | 4,609 | 
| ecrecover_program |  | DIVU | 0 | 5 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_BUFFER | 0 | 11 | 
| ecrecover_program |  | HINT_STOREW | 0 | 6 | 
| ecrecover_program |  | IS_EQ | 0 | 3,203 | 
| ecrecover_program |  | JAL | 0 | 1,263 | 
| ecrecover_program |  | JALR | 0 | 6,645 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 3,780 | 
| ecrecover_program |  | LOADBU | 0 | 2,450 | 
| ecrecover_program |  | LOADW | 0 | 13,806 | 
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
| ecrecover_program |  | STOREW | 0 | 67,532 | 
| ecrecover_program |  | SUB | 0 | 8,850 | 
| ecrecover_program |  | XOR | 0 | 25 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,515 | 16,429 | 1,665,197 | 335,512,536 | 8,815 | 1,446 | 1,626 | 1,904 | 1,802 | 1,767 | 118,660,584 | 265 | 6,099 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 179 | 3,372 | 289,414 | 37,648,195 | 2,130 | 216 | 476 | 355 | 652 | 385 | 15,210,107 | 33 | 1,063 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-ecrecover_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/4a08486fdfecd87d58a34a3f25c9c120383dd4d2/ecrecover-4a08486fdfecd87d58a34a3f25c9c120383dd4d2-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/4a08486fdfecd87d58a34a3f25c9c120383dd4d2

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13229712473)
