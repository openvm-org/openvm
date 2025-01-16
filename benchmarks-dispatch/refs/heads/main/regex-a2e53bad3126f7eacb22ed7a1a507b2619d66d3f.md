| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+84 [+163.6%])</span> 134.57 | <span style='color: red'>(+84 [+163.6%])</span> 134.57 |
| regex_program | <span style='color: red'>(+59 [+300.9%])</span> 79.15 | <span style='color: red'>(+59 [+300.9%])</span> 79.15 |
| leaf | <span style='color: red'>(+24 [+77.0%])</span> 55.41 | <span style='color: red'>(+24 [+77.0%])</span> 55.41 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+59406 [+300.9%])</span> 79,152 | <span style='color: red'>(+59406 [+300.9%])</span> 79,152 | <span style='color: red'>(+59406 [+300.9%])</span> 79,152 | <span style='color: red'>(+59406 [+300.9%])</span> 79,152 |
| `main_cells_used     ` | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 | <span style='color: red'>(+169837 [+0.1%])</span> 165,198,010 |
| `total_cycles        ` | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 | <span style='color: red'>(+9385 [+0.2%])</span> 4,200,289 |
| `execute_time_ms     ` | <span style='color: red'>(+59445 [+3680.8%])</span> 61,060 | <span style='color: red'>(+59445 [+3680.8%])</span> 61,060 | <span style='color: red'>(+59445 [+3680.8%])</span> 61,060 | <span style='color: red'>(+59445 [+3680.8%])</span> 61,060 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+23 [+0.6%])</span> 3,665 | <span style='color: red'>(+23 [+0.6%])</span> 3,665 | <span style='color: red'>(+23 [+0.6%])</span> 3,665 | <span style='color: red'>(+23 [+0.6%])</span> 3,665 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-62 [-0.4%])</span> 14,427 | <span style='color: green'>(-62 [-0.4%])</span> 14,427 | <span style='color: green'>(-62 [-0.4%])</span> 14,427 | <span style='color: green'>(-62 [-0.4%])</span> 14,427 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+24 [+0.9%])</span> 2,578 | <span style='color: red'>(+24 [+0.9%])</span> 2,578 | <span style='color: red'>(+24 [+0.9%])</span> 2,578 | <span style='color: red'>(+24 [+0.9%])</span> 2,578 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-14 [-2.8%])</span> 494 | <span style='color: green'>(-14 [-2.8%])</span> 494 | <span style='color: green'>(-14 [-2.8%])</span> 494 | <span style='color: green'>(-14 [-2.8%])</span> 494 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+7 [+0.1%])</span> 5,492 | <span style='color: red'>(+7 [+0.1%])</span> 5,492 | <span style='color: red'>(+7 [+0.1%])</span> 5,492 | <span style='color: red'>(+7 [+0.1%])</span> 5,492 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-40 [-2.1%])</span> 1,870 | <span style='color: green'>(-40 [-2.1%])</span> 1,870 | <span style='color: green'>(-40 [-2.1%])</span> 1,870 | <span style='color: green'>(-40 [-2.1%])</span> 1,870 |
| `quotient_poly_commit_time_ms` |  1,246 |  1,246 |  1,246 |  1,246 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-38 [-1.4%])</span> 2,743 | <span style='color: green'>(-38 [-1.4%])</span> 2,743 | <span style='color: green'>(-38 [-1.4%])</span> 2,743 | <span style='color: green'>(-38 [-1.4%])</span> 2,743 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+24108 [+77.0%])</span> 55,415 | <span style='color: red'>(+24108 [+77.0%])</span> 55,415 | <span style='color: red'>(+24108 [+77.0%])</span> 55,415 | <span style='color: red'>(+24108 [+77.0%])</span> 55,415 |
| `main_cells_used     ` | <span style='color: red'>(+3563830 [+1.2%])</span> 294,878,589 | <span style='color: red'>(+3563830 [+1.2%])</span> 294,878,589 | <span style='color: red'>(+3563830 [+1.2%])</span> 294,878,589 | <span style='color: red'>(+3563830 [+1.2%])</span> 294,878,589 |
| `total_cycles        ` | <span style='color: red'>(+594766 [+9.1%])</span> 7,119,104 | <span style='color: red'>(+594766 [+9.1%])</span> 7,119,104 | <span style='color: red'>(+594766 [+9.1%])</span> 7,119,104 | <span style='color: red'>(+594766 [+9.1%])</span> 7,119,104 |
| `execute_time_ms     ` | <span style='color: red'>(+23228 [+895.1%])</span> 25,823 | <span style='color: red'>(+23228 [+895.1%])</span> 25,823 | <span style='color: red'>(+23228 [+895.1%])</span> 25,823 | <span style='color: red'>(+23228 [+895.1%])</span> 25,823 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-119 [-2.2%])</span> 5,395 | <span style='color: green'>(-119 [-2.2%])</span> 5,395 | <span style='color: green'>(-119 [-2.2%])</span> 5,395 | <span style='color: green'>(-119 [-2.2%])</span> 5,395 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+999 [+4.3%])</span> 24,197 | <span style='color: red'>(+999 [+4.3%])</span> 24,197 | <span style='color: red'>(+999 [+4.3%])</span> 24,197 | <span style='color: red'>(+999 [+4.3%])</span> 24,197 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+72 [+1.6%])</span> 4,548 | <span style='color: red'>(+72 [+1.6%])</span> 4,548 | <span style='color: red'>(+72 [+1.6%])</span> 4,548 | <span style='color: red'>(+72 [+1.6%])</span> 4,548 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-5 [-0.9%])</span> 570 | <span style='color: green'>(-5 [-0.9%])</span> 570 | <span style='color: green'>(-5 [-0.9%])</span> 570 | <span style='color: green'>(-5 [-0.9%])</span> 570 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+171 [+3.9%])</span> 4,582 | <span style='color: red'>(+171 [+3.9%])</span> 4,582 | <span style='color: red'>(+171 [+3.9%])</span> 4,582 | <span style='color: red'>(+171 [+3.9%])</span> 4,582 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+487 [+10.4%])</span> 5,166 | <span style='color: red'>(+487 [+10.4%])</span> 5,166 | <span style='color: red'>(+487 [+10.4%])</span> 5,166 | <span style='color: red'>(+487 [+10.4%])</span> 5,166 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+170 [+4.4%])</span> 4,067 | <span style='color: red'>(+170 [+4.4%])</span> 4,067 | <span style='color: red'>(+170 [+4.4%])</span> 4,067 | <span style='color: red'>(+170 [+4.4%])</span> 4,067 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+104 [+2.0%])</span> 5,262 | <span style='color: red'>(+104 [+2.0%])</span> 5,262 | <span style='color: red'>(+104 [+2.0%])</span> 5,262 | <span style='color: red'>(+104 [+2.0%])</span> 5,262 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 720 | 47 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
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
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 19 | 31 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 
| regex_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,571 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| regex_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 6,348 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 255,944 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 27,416 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 7,314 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNeVI | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 22,489,837 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 668,886 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 8,769,532 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 391,713 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 65,918 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 511,740 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 467,420 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 30 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 25,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,764,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 2,363,190 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 520,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 24,187,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,892,070 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,114,830 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 5,310 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 27,799,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 123,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 240,600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 4,603,710 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 40,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 677,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 12,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHintWord | 0 | ADD | 14,112,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 215,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 286,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 18,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 2,588,850 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 31,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 26,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 810 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> |  | 0 | STOREW | 41 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | AddEFFI | 0 | LOADW | 7,954 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | AddEFFI | 0 | STOREW | 23,862 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | Alloc | 0 | LOADW | 2,585,829 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | DivEIN | 0 | STOREW | 12,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | For | 0 | LOADW | 113,652 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | For | 0 | STOREW | 1,984,482 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | ImmE | 0 | STOREW | 654,852 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | ImmF | 0 | STOREW | 1,944,507 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | ImmV | 0 | STOREW | 2,280,871 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadE | 0 | LOADW | 2,574,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadE | 0 | LOADW2 | 3,407,756 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 1,243,120 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW2 | 12,336,900 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 1,197,733 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW2 | 11,012,190 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | MulEI | 0 | STOREW | 819,836 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreE | 0 | STOREW | 1,033,856 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreE | 0 | STOREW2 | 1,808,100 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 1,541,518 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW2 | 11,671,921 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | SHINTW | 20,404,388 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 129,765 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW2 | 3,021,413 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | SubEF | 0 | LOADW | 882,156 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,902,080 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 321,360 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 3,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,254,360 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 199,960 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 668,920 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 6,420,252 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 12,574,980 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 34,825,728 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 55,440 | 
| leaf | PhantomAir | CT-poseidon2-hash | 0 | PHANTOM | 22,176 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 0 | PHANTOM | 10,584 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 0 | PHANTOM | 3,316,320 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 85,176 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,584 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 0 | PHANTOM | 32,760 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 32,760 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 155,448 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 36,618,768 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 1,912,104 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 847,584 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 1,532,952 | 
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 344,232 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLT | 0 | 185 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 1,237,798 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 11,318,044 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRA | 0 | 53 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SRL | 0 | 269,770 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 4,880,538 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,691,832 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGE | 0 | 9,408 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 3,890,944 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 0 | 164,512 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 2,273,600 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,190,322 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 800,964 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 331,942 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 3,652,404 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADB | 0 | 24,255 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> |  | LOADH | 0 | 280 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 1,093,200 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADHU | 0 | 3,800 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 45,715,640 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 0 | 509,480 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREH | 0 | 402,960 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 30,916,880 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> |  | DIVU | 0 | 6,498 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> |  | MULHU | 0 | 9,516 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> |  | MUL | 0 | 1,614,697 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 830,676 | 
| regex_program | KeccakVmAir |  | KECCAK256 | 0 | 75,936 | 
| regex_program | PhantomAir |  | PHANTOM | 0 | 1,734 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 2,097,152 |  | 16 | 11 | 56,623,104 | 
| leaf | AccessAdapterAir<4> | 0 | 1,048,576 |  | 16 | 13 | 30,408,704 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 36 | 348 | 25,165,824 | 
| leaf | PhantomAir | 0 | 1,048,576 |  | 8 | 6 | 14,680,064 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 24 | 41 | 136,314,880 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 40 | 7,864,320 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 0 | 64 |  | 24 | 11 | 2,240 | 
| regex_program | AccessAdapterAir<4> | 0 | 32 |  | 24 | 13 | 1,184 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 24 | 17 | 5,373,952 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 1,288 | 3,164 | 142,464 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PhantomAir | 0 | 512 |  | 12 | 6 | 9,216 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 80 | 36 | 243,269,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 48 | 26 | 38,797,312 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 56 | 32 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 16,384 |  | 36 | 26 | 1,015,808 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 76 | 35 | 113,664 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 72 | 40 | 234,881,024 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 104 | 57 | 20,608 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 100 | 39 | 35,584 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 80 | 31 | 7,274,496 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 28 | 21 | 3,211,264 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 1,421,001 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 97,920 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 2,791,105 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 2,016,923 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 108,742 | 
| leaf | AccessAdapter<2> | 0 | 1,087,798 | 
| leaf | AccessAdapter<4> | 0 | 544,110 | 
| leaf | AccessAdapter<8> | 0 | 111,818 | 
| leaf | Arc<BabyBearParameters>, 1> | 0 | 54,584 | 
| leaf | Boundary | 0 | 1,036,041 | 
| leaf | FriReducedOpeningAir | 0 | 544,152 | 
| leaf | PhantomAir | 0 | 621,695 | 
| leaf | ProgramChip | 0 | 289,018 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| regex_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,145,990 | 
| regex_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 33,459 | 
| regex_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 218,639 | 
| regex_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 291,245 | 
| regex_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 198,077 | 
| regex_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 110,627 | 
| regex_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 12,767 | 
| regex_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 130,443 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadSignExtendCoreAir<4, 8>> | 0 | 701 | 
| regex_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 1,966,049 | 
| regex_program | <Rv32MultAdapterAir,DivRemCoreAir<4, 8>> | 0 | 114 | 
| regex_program | <Rv32MultAdapterAir,MulHCoreAir<4, 8>> | 0 | 244 | 
| regex_program | <Rv32MultAdapterAir,MultiplicationCoreAir<4, 8>> | 0 | 52,087 | 
| regex_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 39,557 | 
| regex_program | AccessAdapter<2> | 0 | 42 | 
| regex_program | AccessAdapter<4> | 0 | 22 | 
| regex_program | AccessAdapter<8> | 0 | 69,206 | 
| regex_program | Arc<BabyBearParameters>, 1> | 0 | 14,005 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| regex_program | Boundary | 0 | 69,206 | 
| regex_program | KeccakVmAir | 0 | 24 | 
| regex_program | Merkle | 0 | 70,444 | 
| regex_program | PhantomAir | 0 | 289 | 
| regex_program | ProgramChip | 0 | 89,891 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 
| regex_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | JAL | 1 | 
| leaf |  | 0 | STOREW | 2 | 
| leaf | AddE | 0 | FE4ADD | 47,552 | 
| leaf | AddEFFI | 0 | LOADW | 194 | 
| leaf | AddEFFI | 0 | STOREW | 582 | 
| leaf | AddEFI | 0 | ADD | 860 | 
| leaf | AddEI | 0 | ADD | 92,148 | 
| leaf | AddF | 0 | ADD | 1,333 | 
| leaf | AddFI | 0 | ADD | 78,773 | 
| leaf | AddV | 0 | ADD | 17,365 | 
| leaf | AddVI | 0 | ADD | 806,262 | 
| leaf | Alloc | 0 | ADD | 63,069 | 
| leaf | Alloc | 0 | LOADW | 63,069 | 
| leaf | Alloc | 0 | MUL | 37,161 | 
| leaf | AssertEqE | 0 | BNE | 276 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 11,128 | 
| leaf | AssertEqV | 0 | BNE | 1,192 | 
| leaf | AssertEqVI | 0 | BNE | 318 | 
| leaf | AssertNeVI | 0 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 9,240 | 
| leaf | CT-poseidon2-hash | 0 | PHANTOM | 3,696 | 
| leaf | CT-poseidon2-hash-ext | 0 | PHANTOM | 1,764 | 
| leaf | CT-poseidon2-hash-setup | 0 | PHANTOM | 552,720 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 14,196 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,764 | 
| leaf | CT-verify-batch-reduce-fast | 0 | PHANTOM | 5,460 | 
| leaf | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 5,460 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 1 | 
| leaf | DivE | 0 | BBE4DIV | 8,034 | 
| leaf | DivEIN | 0 | BBE4DIV | 75 | 
| leaf | DivEIN | 0 | STOREW | 300 | 
| leaf | DivFIN | 0 | DIV | 177 | 
| leaf | For | 0 | ADD | 926,645 | 
| leaf | For | 0 | BNE | 977,819 | 
| leaf | For | 0 | JAL | 51,174 | 
| leaf | For | 0 | LOADW | 2,772 | 
| leaf | For | 0 | STOREW | 48,402 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,098 | 
| leaf | HintBitsF | 0 | PHANTOM | 43 | 
| leaf | HintInputVec | 0 | PHANTOM | 25,908 | 
| leaf | IfEq | 0 | BNE | 29,082 | 
| leaf | IfEqI | 0 | BNE | 381,284 | 
| leaf | IfEqI | 0 | JAL | 46,742 | 
| leaf | IfNe | 0 | BEQ | 17,031 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNeI | 0 | BEQ | 2,866 | 
| leaf | ImmE | 0 | STOREW | 15,972 | 
| leaf | ImmF | 0 | STOREW | 47,427 | 
| leaf | ImmV | 0 | STOREW | 55,631 | 
| leaf | LoadE | 0 | LOADW | 62,800 | 
| leaf | LoadE | 0 | LOADW2 | 83,116 | 
| leaf | LoadF | 0 | LOADW | 30,320 | 
| leaf | LoadF | 0 | LOADW2 | 300,900 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | LOADW | 29,213 | 
| leaf | LoadV | 0 | LOADW2 | 268,590 | 
| leaf | MulE | 0 | BBE4MUL | 31,359 | 
| leaf | MulEF | 0 | MUL | 4,128 | 
| leaf | MulEFI | 0 | MUL | 8,020 | 
| leaf | MulEI | 0 | BBE4MUL | 4,999 | 
| leaf | MulEI | 0 | STOREW | 19,996 | 
| leaf | MulF | 0 | MUL | 153,457 | 
| leaf | MulFI | 0 | MUL | 1,360 | 
| leaf | MulVI | 0 | MUL | 22,583 | 
| leaf | NegE | 0 | MUL | 428 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 18,449 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 36,135 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | STOREW | 25,216 | 
| leaf | StoreE | 0 | STOREW2 | 44,100 | 
| leaf | StoreF | 0 | STOREW | 37,598 | 
| leaf | StoreF | 0 | STOREW2 | 284,681 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | ADD | 470,427 | 
| leaf | StoreHintWord | 0 | SHINTW | 497,668 | 
| leaf | StoreV | 0 | STOREW | 3,165 | 
| leaf | StoreV | 0 | STOREW2 | 73,693 | 
| leaf | SubE | 0 | FE4SUB | 16,723 | 
| leaf | SubEF | 0 | LOADW | 21,516 | 
| leaf | SubEF | 0 | SUB | 7,172 | 
| leaf | SubEFI | 0 | ADD | 9,544 | 
| leaf | SubEI | 0 | ADD | 600 | 
| leaf | SubFI | 0 | SUB | 1,333 | 
| leaf | SubV | 0 | SUB | 86,295 | 
| leaf | SubVI | 0 | SUB | 1,053 | 
| leaf | SubVIN | 0 | SUB | 882 | 
| leaf | UnsafeCastVF | 0 | ADD | 27 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| regex_program |  | ADD | 0 | 1,017,188 | 
| regex_program |  | AND | 0 | 53,114 | 
| regex_program |  | AUIPC | 0 | 39,557 | 
| regex_program |  | BEQ | 0 | 187,713 | 
| regex_program |  | BGE | 0 | 294 | 
| regex_program |  | BGEU | 0 | 121,592 | 
| regex_program |  | BLT | 0 | 5,141 | 
| regex_program |  | BLTU | 0 | 71,050 | 
| regex_program |  | BNE | 0 | 103,532 | 
| regex_program |  | DIVU | 0 | 114 | 
| regex_program |  | HINT_STOREW | 0 | 12,767 | 
| regex_program |  | JAL | 0 | 66,129 | 
| regex_program |  | JALR | 0 | 130,443 | 
| regex_program |  | KECCAK256 | 0 | 1 | 
| regex_program |  | LOADB | 0 | 693 | 
| regex_program |  | LOADBU | 0 | 27,330 | 
| regex_program |  | LOADH | 0 | 8 | 
| regex_program |  | LOADHU | 0 | 95 | 
| regex_program |  | LOADW | 0 | 1,142,891 | 
| regex_program |  | LUI | 0 | 44,498 | 
| regex_program |  | MUL | 0 | 52,087 | 
| regex_program |  | MULHU | 0 | 244 | 
| regex_program |  | OR | 0 | 23,544 | 
| regex_program |  | PHANTOM | 0 | 289 | 
| regex_program |  | SLL | 0 | 213,548 | 
| regex_program |  | SLT | 0 | 5 | 
| regex_program |  | SLTU | 0 | 33,454 | 
| regex_program |  | SRA | 0 | 1 | 
| regex_program |  | SRL | 0 | 5,090 | 
| regex_program |  | STOREB | 0 | 12,737 | 
| regex_program |  | STOREH | 0 | 10,074 | 
| regex_program |  | STOREW | 0 | 772,922 | 
| regex_program |  | SUB | 0 | 42,582 | 
| regex_program |  | XOR | 0 | 9,562 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 5,395 | 55,415 | 7,119,104 | 773,458,392 | 24,197 | 5,166 | 4,067 | 4,582 | 5,262 | 4,548 | 294,878,589 | 570 | 25,823 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,665 | 79,152 | 4,200,289 | 632,452,480 | 14,427 | 1,870 | 1,246 | 5,492 | 2,743 | 2,578 | 165,198,010 | 494 | 61,060 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f/regex-a2e53bad3126f7eacb22ed7a1a507b2619d66d3f-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/a2e53bad3126f7eacb22ed7a1a507b2619d66d3f

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12657226496)