| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-2 [-4.9%])</span> 41.60 | <span style='color: green'>(-1 [-4.8%])</span> 21.82 |
| regex_program | <span style='color: green'>(-0 [-0.5%])</span> 14.82 | <span style='color: green'>(-0 [-0.1%])</span> 8.32 |
| leaf | <span style='color: green'>(-2 [-7.2%])</span> 26.79 | <span style='color: green'>(-1 [-7.5%])</span> 13.50 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-38 [-0.5%])</span> 7,409 | <span style='color: green'>(-77 [-0.5%])</span> 14,818 | <span style='color: green'>(-12 [-0.1%])</span> 8,322 | <span style='color: green'>(-65 [-1.0%])</span> 6,496 |
| `main_cells_used     ` |  82,727,686.50 |  165,455,373 |  92,686,348 |  72,769,025 |
| `total_cycles        ` |  1,914,103 |  1,914,103 |  1,914,103 |  1,914,103 |
| `execute_time_ms     ` | <span style='color: red'>(+4 [+0.8%])</span> 453 | <span style='color: red'>(+7 [+0.8%])</span> 906 | <span style='color: red'>(+4 [+0.8%])</span> 487 | <span style='color: red'>(+3 [+0.7%])</span> 419 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+3 [+0.2%])</span> 1,453 | <span style='color: red'>(+6 [+0.2%])</span> 2,906 | <span style='color: green'>(-9 [-0.6%])</span> 1,602 | <span style='color: red'>(+15 [+1.2%])</span> 1,304 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-45 [-0.8%])</span> 5,503 | <span style='color: green'>(-90 [-0.8%])</span> 11,006 | <span style='color: green'>(-7 [-0.1%])</span> 6,233 | <span style='color: green'>(-83 [-1.7%])</span> 4,773 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-24 [-2.2%])</span> 1,092 | <span style='color: green'>(-49 [-2.2%])</span> 2,184 | <span style='color: green'>(-6 [-0.4%])</span> 1,335 | <span style='color: green'>(-43 [-4.8%])</span> 849 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-3.6%])</span> 173 | <span style='color: green'>(-13 [-3.6%])</span> 346 | <span style='color: green'>(-5 [-2.6%])</span> 189 | <span style='color: green'>(-8 [-4.8%])</span> 157 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-21 [-1.8%])</span> 1,170 | <span style='color: green'>(-42 [-1.8%])</span> 2,340 |  1,251 | <span style='color: green'>(-41 [-3.6%])</span> 1,089 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-0.7%])</span> 734.50 | <span style='color: green'>(-10 [-0.7%])</span> 1,469 | <span style='color: green'>(-3 [-0.3%])</span> 856 | <span style='color: green'>(-7 [-1.1%])</span> 613 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+16 [+1.7%])</span> 995 | <span style='color: red'>(+33 [+1.7%])</span> 1,990 | <span style='color: red'>(+13 [+1.1%])</span> 1,157 | <span style='color: red'>(+20 [+2.5%])</span> 833 |
| `pcs_opening_time_ms ` |  1,328 |  2,656 | <span style='color: green'>(-4 [-0.3%])</span> 1,436 | <span style='color: red'>(+2 [+0.2%])</span> 1,220 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1032 [-7.2%])</span> 13,393.50 | <span style='color: green'>(-2063 [-7.2%])</span> 26,787 | <span style='color: green'>(-1100 [-7.5%])</span> 13,496 | <span style='color: green'>(-963 [-6.8%])</span> 13,291 |
| `main_cells_used     ` | <span style='color: green'>(-18836072 [-14.6%])</span> 110,611,902 | <span style='color: green'>(-37672143 [-14.6%])</span> 221,223,804 | <span style='color: green'>(-18839145 [-14.4%])</span> 112,210,199 | <span style='color: green'>(-18832998 [-14.7%])</span> 109,013,605 |
| `total_cycles        ` | <span style='color: green'>(-685528 [-23.3%])</span> 2,256,186 | <span style='color: green'>(-1371055 [-23.3%])</span> 4,512,372 | <span style='color: green'>(-685869 [-23.0%])</span> 2,292,354 | <span style='color: green'>(-685186 [-23.6%])</span> 2,220,018 |
| `execute_time_ms     ` | <span style='color: green'>(-16 [-2.4%])</span> 625 | <span style='color: green'>(-31 [-2.4%])</span> 1,250 | <span style='color: green'>(-10 [-1.5%])</span> 666 | <span style='color: green'>(-21 [-3.5%])</span> 584 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-384 [-16.2%])</span> 1,996 | <span style='color: green'>(-769 [-16.2%])</span> 3,992 | <span style='color: green'>(-400 [-15.7%])</span> 2,155 | <span style='color: green'>(-369 [-16.7%])</span> 1,837 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-632 [-5.5%])</span> 10,772.50 | <span style='color: green'>(-1263 [-5.5%])</span> 21,545 | <span style='color: green'>(-648 [-5.7%])</span> 10,788 | <span style='color: green'>(-615 [-5.4%])</span> 10,757 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-123 [-5.2%])</span> 2,223 | <span style='color: green'>(-246 [-5.2%])</span> 4,446 | <span style='color: green'>(-135 [-5.7%])</span> 2,231 | <span style='color: green'>(-111 [-4.8%])</span> 2,215 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-26 [-9.0%])</span> 262 | <span style='color: green'>(-52 [-9.0%])</span> 524 | <span style='color: green'>(-28 [-9.6%])</span> 264 | <span style='color: green'>(-24 [-8.5%])</span> 260 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-176 [-7.7%])</span> 2,129.50 | <span style='color: green'>(-353 [-7.7%])</span> 4,259 | <span style='color: green'>(-179 [-7.8%])</span> 2,130 | <span style='color: green'>(-174 [-7.6%])</span> 2,129 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-100 [-6.1%])</span> 1,539.50 | <span style='color: green'>(-201 [-6.1%])</span> 3,079 | <span style='color: green'>(-98 [-6.0%])</span> 1,546 | <span style='color: green'>(-103 [-6.3%])</span> 1,533 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-76 [-3.4%])</span> 2,145 | <span style='color: green'>(-153 [-3.4%])</span> 4,290 | <span style='color: green'>(-85 [-3.8%])</span> 2,169 | <span style='color: green'>(-68 [-3.1%])</span> 2,121 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-129 [-5.0%])</span> 2,470 | <span style='color: green'>(-258 [-5.0%])</span> 4,940 | <span style='color: green'>(-140 [-5.4%])</span> 2,475 | <span style='color: green'>(-118 [-4.6%])</span> 2,465 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 759 | 40 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
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
| regex_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 4 | 321 | 4,380 | 
| regex_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| regex_program | PhantomAir | 4 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| regex_program | VmConnectorAir | 4 | 3 | 8 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<8> | 0 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 1 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | FriReducedOpeningAir | 1 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | PhantomAir | 0 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | PhantomAir | 1 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | ProgramAir | 1 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| leaf | VolatileBoundaryAir | 1 | 1,048,576 |  | 8 | 11 | 19,922,944 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 12 | 17 | 59,392 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 532 | 3,163 | 3,695 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 532 | 3,163 | 118,240 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 12 | 32 | 180,224 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 8 | 20 | 3,670,016 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 8 | 20 | 57,344 | 
| regex_program | PhantomAir | 0 | 512 |  | 8 | 6 | 7,168 | 
| regex_program | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 24 | 32 | 917,504 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 28 | 36 | 33,554,432 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 16 | 26 | 5,505,024 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 20 | 28 | 6,291,456 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 20 | 28 | 3,145,728 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 28 | 35 | 64,512 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 28 | 35 | 126 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 28 | 40 | 71,303,168 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 28 | 40 | 71,303,168 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 40 | 57 | 12,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 40 | 39 | 20,224 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 16 | 21 | 1,212,416 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 16 | 21 | 1,212,416 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,837 | 13,291 | 2,292,354 | 328,738,520 | 10,788 | 1,546 | 2,169 | 2,129 | 2,465 | 2,215 | 112,210,199 | 260 | 666 | 
| leaf | 1 | 2,155 | 13,496 | 2,220,018 | 328,745,944 | 10,757 | 1,533 | 2,121 | 2,130 | 2,475 | 2,231 | 109,013,605 | 264 | 584 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,602 | 8,322 |  | 209,921,543 | 6,233 | 856 | 1,157 | 1,251 | 1,436 | 1,335 | 92,686,348 | 189 | 487 | 
| regex_program | 1 | 1,304 | 6,496 | 1,914,103 | 149,454,692 | 4,773 | 613 | 833 | 1,089 | 1,220 | 849 | 72,769,025 | 157 | 419 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b42be8bb6fd2acc0830eb8833ae0b2c07c8eec69

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13097254898)
