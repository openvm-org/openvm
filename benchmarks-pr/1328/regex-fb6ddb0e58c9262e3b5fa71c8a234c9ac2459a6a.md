| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-2 [-4.8%])</span> 41.49 | <span style='color: green'>(-1 [-4.5%])</span> 21.79 |
| regex_program |  14.79 | <span style='color: red'>(+0 [+0.2%])</span> 8.29 |
| leaf | <span style='color: green'>(-2 [-7.2%])</span> 26.70 | <span style='color: green'>(-1 [-7.2%])</span> 13.50 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,394.50 |  14,789 | <span style='color: red'>(+13 [+0.2%])</span> 8,293 | <span style='color: green'>(-17 [-0.3%])</span> 6,496 |
| `main_cells_used     ` |  82,727,686.50 |  165,455,373 |  92,686,348 |  72,769,025 |
| `total_cycles        ` |  1,914,103 |  1,914,103 |  1,914,103 |  1,914,103 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-0.4%])</span> 449.50 | <span style='color: green'>(-4 [-0.4%])</span> 899 | <span style='color: green'>(-2 [-0.4%])</span> 484 | <span style='color: green'>(-2 [-0.5%])</span> 415 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-0.1%])</span> 1,444.50 | <span style='color: green'>(-3 [-0.1%])</span> 2,889 |  1,610 | <span style='color: green'>(-2 [-0.2%])</span> 1,279 |
| `stark_prove_excluding_trace_time_ms` |  5,500.50 |  11,001 | <span style='color: red'>(+16 [+0.3%])</span> 6,199 | <span style='color: green'>(-13 [-0.3%])</span> 4,802 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-22 [-2.0%])</span> 1,080.50 | <span style='color: green'>(-45 [-2.0%])</span> 2,161 | <span style='color: green'>(-8 [-0.6%])</span> 1,327 | <span style='color: green'>(-37 [-4.2%])</span> 834 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+18 [+10.7%])</span> 186.50 | <span style='color: red'>(+36 [+10.7%])</span> 373 | <span style='color: green'>(-2 [-1.1%])</span> 187 | <span style='color: red'>(+38 [+25.7%])</span> 186 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+2 [+0.2%])</span> 1,191 | <span style='color: red'>(+5 [+0.2%])</span> 2,382 | <span style='color: green'>(-3 [-0.2%])</span> 1,256 | <span style='color: red'>(+8 [+0.7%])</span> 1,126 |
| `quotient_poly_compute_time_ms` |  732 |  1,464 | <span style='color: red'>(+9 [+1.1%])</span> 858 | <span style='color: green'>(-8 [-1.3%])</span> 606 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+6 [+0.6%])</span> 983 | <span style='color: red'>(+12 [+0.6%])</span> 1,966 | <span style='color: red'>(+17 [+1.5%])</span> 1,142 | <span style='color: green'>(-5 [-0.6%])</span> 824 |
| `pcs_opening_time_ms ` |  1,315 |  2,630 |  1,420 | <span style='color: green'>(-3 [-0.2%])</span> 1,210 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1034 [-7.2%])</span> 13,350.50 | <span style='color: green'>(-2069 [-7.2%])</span> 26,701 | <span style='color: green'>(-1041 [-7.2%])</span> 13,499 | <span style='color: green'>(-1028 [-7.2%])</span> 13,202 |
| `main_cells_used     ` | <span style='color: green'>(-18838965 [-14.6%])</span> 110,609,737.50 | <span style='color: green'>(-37677930 [-14.6%])</span> 221,219,475 | <span style='color: green'>(-18836004 [-14.4%])</span> 112,211,378 | <span style='color: green'>(-18841926 [-14.7%])</span> 109,008,097 |
| `total_cycles        ` | <span style='color: green'>(-685849 [-23.3%])</span> 2,255,945.50 | <span style='color: green'>(-1371698 [-23.3%])</span> 4,511,891 | <span style='color: green'>(-685520 [-23.0%])</span> 2,292,485 | <span style='color: green'>(-686178 [-23.6%])</span> 2,219,406 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+0.2%])</span> 633 | <span style='color: red'>(+3 [+0.2%])</span> 1,266 | <span style='color: red'>(+15 [+2.3%])</span> 678 | <span style='color: green'>(-12 [-2.0%])</span> 588 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-352 [-15.0%])</span> 1,999.50 | <span style='color: green'>(-704 [-15.0%])</span> 3,999 | <span style='color: green'>(-347 [-13.7%])</span> 2,185 | <span style='color: green'>(-357 [-16.4%])</span> 1,814 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-684 [-6.0%])</span> 10,718 | <span style='color: green'>(-1368 [-6.0%])</span> 21,436 | <span style='color: green'>(-682 [-6.0%])</span> 10,726 | <span style='color: green'>(-686 [-6.0%])</span> 10,710 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-144 [-6.1%])</span> 2,207.50 | <span style='color: green'>(-287 [-6.1%])</span> 4,415 | <span style='color: green'>(-149 [-6.3%])</span> 2,218 | <span style='color: green'>(-138 [-5.9%])</span> 2,197 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-22 [-7.6%])</span> 262 | <span style='color: green'>(-43 [-7.6%])</span> 524 | <span style='color: green'>(-22 [-7.7%])</span> 262 | <span style='color: green'>(-21 [-7.4%])</span> 262 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-165 [-7.2%])</span> 2,140.50 | <span style='color: green'>(-330 [-7.2%])</span> 4,281 | <span style='color: green'>(-179 [-7.7%])</span> 2,143 | <span style='color: green'>(-151 [-6.6%])</span> 2,138 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-132 [-8.0%])</span> 1,525.50 | <span style='color: green'>(-265 [-8.0%])</span> 3,051 | <span style='color: green'>(-124 [-7.5%])</span> 1,540 | <span style='color: green'>(-141 [-8.5%])</span> 1,511 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-106 [-4.8%])</span> 2,114 | <span style='color: green'>(-213 [-4.8%])</span> 4,228 | <span style='color: green'>(-136 [-6.0%])</span> 2,117 | <span style='color: green'>(-77 [-3.5%])</span> 2,111 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-114 [-4.4%])</span> 2,465 | <span style='color: green'>(-228 [-4.4%])</span> 4,930 | <span style='color: green'>(-102 [-4.0%])</span> 2,477 | <span style='color: green'>(-126 [-4.9%])</span> 2,453 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 748 | 48 | 

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
| leaf | 0 | 1,814 | 13,202 | 2,292,485 | 328,738,520 | 10,710 | 1,540 | 2,117 | 2,138 | 2,453 | 2,197 | 112,211,378 | 262 | 678 | 
| leaf | 1 | 2,185 | 13,499 | 2,219,406 | 328,745,944 | 10,726 | 1,511 | 2,111 | 2,143 | 2,477 | 2,218 | 109,008,097 | 262 | 588 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,610 | 8,293 |  | 209,921,543 | 6,199 | 858 | 1,142 | 1,256 | 1,420 | 1,327 | 92,686,348 | 187 | 484 | 
| regex_program | 1 | 1,279 | 6,496 | 1,914,103 | 149,454,692 | 4,802 | 606 | 824 | 1,126 | 1,210 | 834 | 72,769,025 | 186 | 415 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/fb6ddb0e58c9262e3b5fa71c8a234c9ac2459a6a

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13122487529)
