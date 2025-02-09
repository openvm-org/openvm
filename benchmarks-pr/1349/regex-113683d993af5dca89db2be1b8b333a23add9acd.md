| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+3.8%])</span> 30.85 | <span style='color: red'>(+0 [+2.8%])</span> 16.48 |
| regex_program | <span style='color: red'>(+0 [+0.6%])</span> 14.28 | <span style='color: red'>(+0 [+1.4%])</span> 8.17 |
| leaf | <span style='color: red'>(+1 [+6.8%])</span> 16.57 | <span style='color: red'>(+0 [+4.2%])</span> 8.31 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+40 [+0.6%])</span> 7,137.50 | <span style='color: red'>(+79 [+0.6%])</span> 14,275 | <span style='color: red'>(+109 [+1.4%])</span> 8,166 | <span style='color: green'>(-30 [-0.5%])</span> 6,109 |
| `main_cells_used     ` |  82,727,686.50 |  165,455,373 |  92,686,348 |  72,769,025 |
| `total_cycles        ` |  1,914,103 |  1,914,103 |  1,914,103 |  1,914,103 |
| `execute_time_ms     ` | <span style='color: green'>(-6 [-1.4%])</span> 435.50 | <span style='color: green'>(-12 [-1.4%])</span> 871 | <span style='color: green'>(-6 [-1.3%])</span> 469 | <span style='color: green'>(-6 [-1.5%])</span> 402 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+44 [+3.8%])</span> 1,216 | <span style='color: red'>(+88 [+3.8%])</span> 2,432 | <span style='color: red'>(+89 [+6.4%])</span> 1,490 | <span style='color: green'>(-1 [-0.1%])</span> 942 |
| `stark_prove_excluding_trace_time_ms` |  5,486 |  10,972 | <span style='color: red'>(+26 [+0.4%])</span> 6,207 | <span style='color: green'>(-23 [-0.5%])</span> 4,765 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.2%])</span> 1,082 | <span style='color: green'>(-4 [-0.2%])</span> 2,164 |  1,325 | <span style='color: green'>(-4 [-0.5%])</span> 839 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-8 [-4.8%])</span> 168 | <span style='color: green'>(-17 [-4.8%])</span> 336 | <span style='color: green'>(-2 [-1.1%])</span> 187 | <span style='color: green'>(-15 [-9.1%])</span> 149 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+4 [+0.3%])</span> 1,180 | <span style='color: red'>(+7 [+0.3%])</span> 2,360 | <span style='color: green'>(-7 [-0.6%])</span> 1,235 | <span style='color: red'>(+14 [+1.3%])</span> 1,125 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+6 [+0.7%])</span> 741.50 | <span style='color: red'>(+11 [+0.7%])</span> 1,483 | <span style='color: red'>(+17 [+2.0%])</span> 871 | <span style='color: green'>(-6 [-1.0%])</span> 612 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-6 [-0.7%])</span> 969 | <span style='color: green'>(-13 [-0.7%])</span> 1,938 | <span style='color: red'>(+5 [+0.4%])</span> 1,134 | <span style='color: green'>(-18 [-2.2%])</span> 804 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+7 [+0.5%])</span> 1,331.50 | <span style='color: red'>(+14 [+0.5%])</span> 2,663 | <span style='color: red'>(+12 [+0.8%])</span> 1,446 | <span style='color: red'>(+2 [+0.2%])</span> 1,217 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+526 [+6.8%])</span> 8,286 | <span style='color: red'>(+1053 [+6.8%])</span> 16,572 | <span style='color: red'>(+336 [+4.2%])</span> 8,309 | <span style='color: red'>(+717 [+9.5%])</span> 8,263 |
| `main_cells_used     ` | <span style='color: red'>(+1080485 [+1.4%])</span> 78,482,669 | <span style='color: red'>(+2160970 [+1.4%])</span> 156,965,338 | <span style='color: red'>(+1088721 [+1.4%])</span> 79,571,734 | <span style='color: red'>(+1072249 [+1.4%])</span> 77,393,604 |
| `total_cycles        ` | <span style='color: red'>(+41666 [+4.1%])</span> 1,062,827 | <span style='color: red'>(+83332 [+4.1%])</span> 2,125,654 | <span style='color: red'>(+41975 [+4.0%])</span> 1,080,322 | <span style='color: red'>(+41357 [+4.1%])</span> 1,045,332 |
| `execute_time_ms     ` | <span style='color: red'>(+0 [+0.1%])</span> 382.50 | <span style='color: red'>(+1 [+0.1%])</span> 765 | <span style='color: red'>(+6 [+1.4%])</span> 420 | <span style='color: green'>(-5 [-1.4%])</span> 345 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-3 [-0.3%])</span> 1,166.50 | <span style='color: green'>(-6 [-0.3%])</span> 2,333 | <span style='color: red'>(+21 [+1.7%])</span> 1,239 | <span style='color: green'>(-27 [-2.4%])</span> 1,094 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+529 [+8.5%])</span> 6,737 | <span style='color: red'>(+1058 [+8.5%])</span> 13,474 | <span style='color: red'>(+311 [+4.8%])</span> 6,749 | <span style='color: red'>(+747 [+12.5%])</span> 6,725 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+132 [+10.7%])</span> 1,370 | <span style='color: red'>(+265 [+10.7%])</span> 2,740 | <span style='color: red'>(+93 [+7.2%])</span> 1,377 | <span style='color: red'>(+172 [+14.4%])</span> 1,363 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+12 [+7.6%])</span> 177.50 | <span style='color: red'>(+25 [+7.6%])</span> 355 | <span style='color: red'>(+3 [+1.7%])</span> 178 | <span style='color: red'>(+22 [+14.2%])</span> 177 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+126 [+10.2%])</span> 1,356.50 | <span style='color: red'>(+252 [+10.2%])</span> 2,713 | <span style='color: red'>(+81 [+6.3%])</span> 1,364 | <span style='color: red'>(+171 [+14.5%])</span> 1,349 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+54 [+5.6%])</span> 1,021.50 | <span style='color: red'>(+108 [+5.6%])</span> 2,043 | <span style='color: red'>(+7 [+0.7%])</span> 1,026 | <span style='color: red'>(+101 [+11.0%])</span> 1,017 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+161 [+13.6%])</span> 1,342 | <span style='color: red'>(+322 [+13.6%])</span> 2,684 | <span style='color: red'>(+125 [+10.1%])</span> 1,359 | <span style='color: red'>(+197 [+17.5%])</span> 1,325 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+44 [+3.1%])</span> 1,467 | <span style='color: red'>(+89 [+3.1%])</span> 2,934 | <span style='color: red'>(+37 [+2.6%])</span> 1,476 | <span style='color: red'>(+52 [+3.7%])</span> 1,458 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 756 | 49 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
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
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 44 | 27 | 37,224,448 | 
| leaf | FriReducedOpeningAir | 1 | 524,288 |  | 44 | 27 | 37,224,448 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 1 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | ProgramAir | 1 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 16 | 23 | 5,111,808 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 131,072 |  | 16 | 23 | 5,111,808 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 16,384 |  | 12 | 9 | 344,064 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 16,384 |  | 12 | 9 | 344,064 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 

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
| leaf | 0 | 1,094 | 8,263 | 1,080,322 | 220,669,656 | 6,749 | 1,026 | 1,359 | 1,349 | 1,458 | 1,377 | 79,571,734 | 177 | 420 | 
| leaf | 1 | 1,239 | 8,309 | 1,045,332 | 220,677,080 | 6,725 | 1,017 | 1,325 | 1,364 | 1,476 | 1,363 | 77,393,604 | 178 | 345 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,490 | 8,166 |  | 209,921,543 | 6,207 | 871 | 1,134 | 1,235 | 1,446 | 1,325 | 92,686,348 | 187 | 469 | 
| regex_program | 1 | 942 | 6,109 | 1,914,103 | 149,454,692 | 4,765 | 612 | 804 | 1,125 | 1,217 | 839 | 72,769,025 | 149 | 402 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/113683d993af5dca89db2be1b8b333a23add9acd

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13223685582)
