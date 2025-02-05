| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+9 [+78.4%])</span> 19.41 | <span style='color: red'>(+9 [+78.4%])</span> 19.41 |
| fibonacci_program | <span style='color: red'>(+4 [+89.1%])</span> 9.33 | <span style='color: red'>(+4 [+89.1%])</span> 9.33 |
| leaf | <span style='color: red'>(+4 [+69.5%])</span> 10.08 | <span style='color: red'>(+4 [+69.5%])</span> 10.08 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4398 [+89.1%])</span> 9,334 | <span style='color: red'>(+4398 [+89.1%])</span> 9,334 | <span style='color: red'>(+4398 [+89.1%])</span> 9,334 | <span style='color: red'>(+4398 [+89.1%])</span> 9,334 |
| `main_cells_used     ` |  51,484,646 |  51,484,646 |  51,484,646 |  51,484,646 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` | <span style='color: red'>(+4399 [+1423.6%])</span> 4,708 | <span style='color: red'>(+4399 [+1423.6%])</span> 4,708 | <span style='color: red'>(+4399 [+1423.6%])</span> 4,708 | <span style='color: red'>(+4399 [+1423.6%])</span> 4,708 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+5 [+0.8%])</span> 646 | <span style='color: red'>(+5 [+0.8%])</span> 646 | <span style='color: red'>(+5 [+0.8%])</span> 646 | <span style='color: red'>(+5 [+0.8%])</span> 646 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-6 [-0.2%])</span> 3,980 | <span style='color: green'>(-6 [-0.2%])</span> 3,980 | <span style='color: green'>(-6 [-0.2%])</span> 3,980 | <span style='color: green'>(-6 [-0.2%])</span> 3,980 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-3 [-0.4%])</span> 793 | <span style='color: green'>(-3 [-0.4%])</span> 793 | <span style='color: green'>(-3 [-0.4%])</span> 793 | <span style='color: green'>(-3 [-0.4%])</span> 793 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.7%])</span> 140 | <span style='color: green'>(-1 [-0.7%])</span> 140 | <span style='color: green'>(-1 [-0.7%])</span> 140 | <span style='color: green'>(-1 [-0.7%])</span> 140 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+9 [+1.2%])</span> 751 | <span style='color: red'>(+9 [+1.2%])</span> 751 | <span style='color: red'>(+9 [+1.2%])</span> 751 | <span style='color: red'>(+9 [+1.2%])</span> 751 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-6 [-1.2%])</span> 511 | <span style='color: green'>(-6 [-1.2%])</span> 511 | <span style='color: green'>(-6 [-1.2%])</span> 511 | <span style='color: green'>(-6 [-1.2%])</span> 511 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-26 [-3.5%])</span> 715 | <span style='color: green'>(-26 [-3.5%])</span> 715 | <span style='color: green'>(-26 [-3.5%])</span> 715 | <span style='color: green'>(-26 [-3.5%])</span> 715 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+22 [+2.1%])</span> 1,067 | <span style='color: red'>(+22 [+2.1%])</span> 1,067 | <span style='color: red'>(+22 [+2.1%])</span> 1,067 | <span style='color: red'>(+22 [+2.1%])</span> 1,067 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4132 [+69.5%])</span> 10,076 | <span style='color: red'>(+4132 [+69.5%])</span> 10,076 | <span style='color: red'>(+4132 [+69.5%])</span> 10,076 | <span style='color: red'>(+4132 [+69.5%])</span> 10,076 |
| `main_cells_used     ` | <span style='color: red'>(+163585 [+0.3%])</span> 50,075,889 | <span style='color: red'>(+163585 [+0.3%])</span> 50,075,889 | <span style='color: red'>(+163585 [+0.3%])</span> 50,075,889 | <span style='color: red'>(+163585 [+0.3%])</span> 50,075,889 |
| `total_cycles        ` | <span style='color: red'>(+27027 [+2.2%])</span> 1,239,863 | <span style='color: red'>(+27027 [+2.2%])</span> 1,239,863 | <span style='color: red'>(+27027 [+2.2%])</span> 1,239,863 | <span style='color: red'>(+27027 [+2.2%])</span> 1,239,863 |
| `execute_time_ms     ` | <span style='color: red'>(+4050 [+1238.5%])</span> 4,377 | <span style='color: red'>(+4050 [+1238.5%])</span> 4,377 | <span style='color: red'>(+4050 [+1238.5%])</span> 4,377 | <span style='color: red'>(+4050 [+1238.5%])</span> 4,377 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+1 [+0.1%])</span> 757 | <span style='color: red'>(+1 [+0.1%])</span> 757 | <span style='color: red'>(+1 [+0.1%])</span> 757 | <span style='color: red'>(+1 [+0.1%])</span> 757 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+81 [+1.7%])</span> 4,942 | <span style='color: red'>(+81 [+1.7%])</span> 4,942 | <span style='color: red'>(+81 [+1.7%])</span> 4,942 | <span style='color: red'>(+81 [+1.7%])</span> 4,942 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.3%])</span> 970 | <span style='color: red'>(+3 [+0.3%])</span> 970 | <span style='color: red'>(+3 [+0.3%])</span> 970 | <span style='color: red'>(+3 [+0.3%])</span> 970 |
| `generate_perm_trace_time_ms` |  118 |  118 |  118 |  118 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 | <span style='color: red'>(+16 [+1.8%])</span> 926 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+16 [+2.4%])</span> 681 | <span style='color: red'>(+16 [+2.4%])</span> 681 | <span style='color: red'>(+16 [+2.4%])</span> 681 | <span style='color: red'>(+16 [+2.4%])</span> 681 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+19 [+2.0%])</span> 990 | <span style='color: red'>(+19 [+2.0%])</span> 990 | <span style='color: red'>(+19 [+2.0%])</span> 990 | <span style='color: red'>(+19 [+2.0%])</span> 990 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+27 [+2.2%])</span> 1,253 | <span style='color: red'>(+27 [+2.2%])</span> 1,253 | <span style='color: red'>(+27 [+2.2%])</span> 1,253 | <span style='color: red'>(+27 [+2.2%])</span> 1,253 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 399 | 5 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| fibonacci_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| fibonacci_program | PhantomAir | 4 | 3 | 4 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| fibonacci_program | VmConnectorAir | 4 | 3 | 8 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 11 | 
| leaf | FriReducedOpeningAir | 4 | 31 | 52 | 
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
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 36,424 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 19,604 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 872,320 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 456,228 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 666,652 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,411,024 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,140,454 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 291,798 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 4,350 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,156 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,629 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 94,192 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 181,105 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 155,759 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 331,296 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 331,296 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 432,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 288,231 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,441,445 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,291,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 141,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 14,848 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 114,956 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,000,471 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 134,502 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 313,084 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 282,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 282,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 28,768 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 17,980 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 437,001 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 293,567 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 549,492 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 183,164 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 8,700 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 44,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 133,951 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 197,867 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 28,971 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,625 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 4,556,335 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,612 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 109,848 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 33,764 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,957 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,220 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 580,773 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,185 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 2,948,623 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 78,975 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 27 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 120,321 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 679,008 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 2,908,576 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 196,746 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 1,694,990 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 675,840 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 838,488 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 470,363 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 488,072 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 271,168 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,258 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 915,154 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 37,658 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 120,574 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 2,574,600 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 53,424 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 75,096 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 894 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 17,520 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 33,030 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 7,056 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,773 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 20,349 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 5,815,026 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 32,401,224 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 108 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 11,100,074 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 2,600,078 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,600,026 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 32 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 96 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,800,018 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 108 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 252 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 280 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 320 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 126 | 
| fibonacci_program | PhantomAir |  | PHANTOM | 0 | 12 | 
| fibonacci_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 64 | 
| fibonacci_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 32 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 8 | 6 | 28 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 20 | 32 | 208 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 20 | 28 | 768 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 40 | 1,088 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 21 | 296 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 642,850 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 160,582 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 22,148 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 279,780 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 42,221 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 48,418 | 
| leaf | AccessAdapter<2> | 0 | 173,102 | 
| leaf | AccessAdapter<4> | 0 | 81,260 | 
| leaf | AccessAdapter<8> | 0 | 322 | 
| leaf | Boundary | 0 | 159,632 | 
| leaf | FriReducedOpeningAir | 0 | 102,984 | 
| leaf | PhantomAir | 0 | 36,316 | 
| leaf | ProgramChip | 0 | 88,414 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 25,992 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 900,042 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 300,002 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 200,004 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 4 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 100,007 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 9 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 15 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 7 | 
| fibonacci_program | AccessAdapter<8> | 0 | 30 | 
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | 175 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fibonacci_program | Boundary | 0 | 30 | 
| fibonacci_program | Merkle | 0 | 226 | 
| fibonacci_program | PhantomAir | 0 | 2 | 
| fibonacci_program | ProgramChip | 0 | 3,241 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 3 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 12,844 | 
| leaf | AddEFFI | 0 | ADD | 1,256 | 
| leaf | AddEFI | 0 | ADD | 676 | 
| leaf | AddEI | 0 | ADD | 30,080 | 
| leaf | AddF | 0 | ADD | 5,215 | 
| leaf | AddFI | 0 | ADD | 15,732 | 
| leaf | AddV | 0 | ADD | 22,988 | 
| leaf | AddVI | 0 | ADD | 48,656 | 
| leaf | Alloc | 0 | ADD | 39,326 | 
| leaf | Alloc | 0 | MUL | 10,062 | 
| leaf | AssertEqE | 0 | BNE | 244 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 4,776 | 
| leaf | AssertEqV | 0 | BNE | 1,468 | 
| leaf | AssertEqVI | 0 | BNE | 259 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 672 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 674 | 
| leaf | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,904 | 
| leaf | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 12,516 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 150 | 
| leaf | DivE | 0 | BBE4DIV | 7,136 | 
| leaf | DivEIN | 0 | ADD | 764 | 
| leaf | DivEIN | 0 | BBE4DIV | 191 | 
| leaf | DivF | 0 | DIV | 7,392 | 
| leaf | DivFIN | 0 | DIV | 401 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 6,258 | 
| leaf | HintBitsF | 0 | PHANTOM | 149 | 
| leaf | HintFelt | 0 | PHANTOM | 2,920 | 
| leaf | HintInputVec | 0 | PHANTOM | 5,505 | 
| leaf | HintLoad | 0 | PHANTOM | 1,176 | 
| leaf | IfEq | 0 | BNE | 140 | 
| leaf | IfEqI | 0 | BNE | 25,251 | 
| leaf | IfEqI | 0 | JAL | 8,775 | 
| leaf | IfNe | 0 | BEQ | 143 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNeI | 0 | BEQ | 95 | 
| leaf | ImmE | 0 | ADD | 3,248 | 
| leaf | ImmF | 0 | ADD | 6,245 | 
| leaf | ImmV | 0 | ADD | 5,371 | 
| leaf | LoadE | 0 | ADD | 11,424 | 
| leaf | LoadE | 0 | LOADW | 27,048 | 
| leaf | LoadE | 0 | MUL | 11,424 | 
| leaf | LoadF | 0 | ADD | 14,912 | 
| leaf | LoadF | 0 | LOADW | 30,864 | 
| leaf | LoadF | 0 | MUL | 9,939 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 49,705 | 
| leaf | LoadV | 0 | LOADW | 132,208 | 
| leaf | LoadV | 0 | MUL | 44,526 | 
| leaf | MulE | 0 | BBE4MUL | 24,083 | 
| leaf | MulEF | 0 | MUL | 4,888 | 
| leaf | MulEFI | 0 | MUL | 512 | 
| leaf | MulEI | 0 | ADD | 3,964 | 
| leaf | MulEI | 0 | BBE4MUL | 991 | 
| leaf | MulF | 0 | MUL | 34,499 | 
| leaf | MulFI | 0 | MUL | 4,638 | 
| leaf | MulVI | 0 | MUL | 10,796 | 
| leaf | NegE | 0 | MUL | 152 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 51 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 9,744 | 
| leaf | StoreE | 0 | MUL | 9,744 | 
| leaf | StoreE | 0 | STOREW | 15,173 | 
| leaf | StoreF | 0 | ADD | 992 | 
| leaf | StoreF | 0 | MUL | 620 | 
| leaf | StoreF | 0 | STOREW | 8,943 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 77,045 | 
| leaf | StoreV | 0 | ADD | 15,069 | 
| leaf | StoreV | 0 | MUL | 10,123 | 
| leaf | StoreV | 0 | STOREW | 30,720 | 
| leaf | SubE | 0 | FE4SUB | 3,173 | 
| leaf | SubEF | 0 | ADD | 18,948 | 
| leaf | SubEF | 0 | SUB | 6,316 | 
| leaf | SubEFI | 0 | ADD | 300 | 
| leaf | SubEI | 0 | ADD | 1,528 | 
| leaf | SubFI | 0 | SUB | 4,619 | 
| leaf | SubV | 0 | SUB | 6,823 | 
| leaf | SubVI | 0 | SUB | 999 | 
| leaf | SubVIN | 0 | SUB | 840 | 
| leaf | UnsafeCastVF | 0 | ADD | 125 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 157,115 | 
| leaf | ZipFor | 0 | BNE | 128,201 | 
| leaf | ZipFor | 0 | JAL | 13,369 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program |  | ADD | 0 | 900,034 | 
| fibonacci_program |  | AND | 0 | 3 | 
| fibonacci_program |  | AUIPC | 0 | 7 | 
| fibonacci_program |  | BEQ | 0 | 100,003 | 
| fibonacci_program |  | BGEU | 0 | 1 | 
| fibonacci_program |  | BLTU | 0 | 3 | 
| fibonacci_program |  | BNE | 0 | 100,001 | 
| fibonacci_program |  | HINT_BUFFER | 0 | 2 | 
| fibonacci_program |  | HINT_STOREW | 0 | 1 | 
| fibonacci_program |  | JAL | 0 | 100,001 | 
| fibonacci_program |  | JALR | 0 | 9 | 
| fibonacci_program |  | LOADW | 0 | 7 | 
| fibonacci_program |  | LUI | 0 | 6 | 
| fibonacci_program |  | OR | 0 | 1 | 
| fibonacci_program |  | PHANTOM | 0 | 2 | 
| fibonacci_program |  | SLTU | 0 | 300,002 | 
| fibonacci_program |  | STOREW | 0 | 8 | 
| fibonacci_program |  | SUB | 0 | 2 | 
| fibonacci_program |  | XOR | 0 | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 757 | 10,076 | 1,239,863 | 140,067,800 | 4,942 | 681 | 990 | 926 | 1,253 | 970 | 50,075,889 | 118 | 4,377 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 646 | 9,334 | 1,500,095 | 122,458,476 | 3,980 | 511 | 715 | 751 | 1,067 | 793 | 51,484,646 | 140 | 4,708 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/9d1c71639c516da305d7ec13e7ee959df8b5b3ee/fibonacci-9d1c71639c516da305d7ec13e7ee959df8b5b3ee-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/9d1c71639c516da305d7ec13e7ee959df8b5b3ee

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13163606004)
