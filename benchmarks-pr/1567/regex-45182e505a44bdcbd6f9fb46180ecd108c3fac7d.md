| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  18.46 |  11.38 |
| regex_program |  7.42 |  4.73 |
| leaf |  11.01 |  6.62 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,707.50 |  7,415 |  4,729 |  2,686 |
| `main_cells_used     ` | <span style='color: green'>(-888080 [-1.1%])</span> 82,367,496 | <span style='color: green'>(-1776160 [-1.1%])</span> 164,734,992 | <span style='color: green'>(-2219112 [-2.4%])</span> 91,281,687 | <span style='color: red'>(+442952 [+0.6%])</span> 73,453,305 |
| `total_cells_used    ` |  193,042,412 |  386,084,824 |  211,758,557 |  174,326,267 |
| `insns               ` |  2,739,064.67 |  8,217,194 |  4,108,597 |  1,897,597 |
| `execute_metered_time_ms` |  31 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  128.53 | -          |  128.53 |  128.53 |
| `execute_preflight_time_ms` |  97 |  194 |  105 |  89 |
| `execute_preflight_insn_mi/s` |  27.38 | -          |  29.74 |  25.02 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-578 [-59.4%])</span> 394.50 | <span style='color: green'>(-1155 [-59.4%])</span> 789 | <span style='color: green'>(-641 [-55.9%])</span> 506 | <span style='color: green'>(-514 [-64.5%])</span> 283 |
| `memory_finalize_time_ms` |  7.50 |  15 |  13 |  2 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+332 [+12.5%])</span> 2,983 | <span style='color: red'>(+663 [+12.5%])</span> 5,966 | <span style='color: red'>(+1054 [+37.3%])</span> 3,876 | <span style='color: green'>(-391 [-15.8%])</span> 2,090 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+73 [+14.6%])</span> 573.50 | <span style='color: red'>(+146 [+14.6%])</span> 1,147 | <span style='color: red'>(+213 [+38.2%])</span> 770 | <span style='color: green'>(-67 [-15.1%])</span> 377 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+3.0%])</span> 226 | <span style='color: red'>(+13 [+3.0%])</span> 452 | <span style='color: red'>(+67 [+28.9%])</span> 299 | <span style='color: green'>(-54 [-26.1%])</span> 153 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+31 [+5.6%])</span> 587.50 | <span style='color: red'>(+62 [+5.6%])</span> 1,175 | <span style='color: red'>(+194 [+33.7%])</span> 770 | <span style='color: green'>(-132 [-24.6%])</span> 405 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+41 [+14.6%])</span> 322 | <span style='color: red'>(+82 [+14.6%])</span> 644 | <span style='color: red'>(+132 [+43.6%])</span> 435 | <span style='color: green'>(-50 [-19.3%])</span> 209 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+32 [+13.2%])</span> 278 | <span style='color: red'>(+65 [+13.2%])</span> 556 | <span style='color: red'>(+93 [+33.1%])</span> 374 | <span style='color: green'>(-28 [-13.3%])</span> 182 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+150 [+17.9%])</span> 988 | <span style='color: red'>(+300 [+17.9%])</span> 1,976 | <span style='color: red'>(+358 [+41.5%])</span> 1,221 | <span style='color: green'>(-58 [-7.1%])</span> 755 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,505 |  11,010 |  6,615 |  4,395 |
| `main_cells_used     ` | <span style='color: green'>(-29558012 [-19.5%])</span> 122,270,001 | <span style='color: green'>(-59116024 [-19.5%])</span> 244,540,002 | <span style='color: green'>(-517914 [-0.3%])</span> 153,709,964 | <span style='color: green'>(-58598110 [-39.2%])</span> 90,830,038 |
| `total_cells_used    ` |  304,358,921 |  608,717,842 |  391,464,962 |  217,252,880 |
| `insns               ` |  1,663,341.50 |  3,326,683 |  1,830,117 |  1,496,566 |
| `execute_preflight_time_ms` |  580 |  1,160 |  735 |  425 |
| `execute_preflight_insn_mi/s` |  4.38 | -          |  4.54 |  4.22 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1212 [-80.8%])</span> 288 | <span style='color: green'>(-2425 [-80.8%])</span> 576 | <span style='color: green'>(-1171 [-76.7%])</span> 355 | <span style='color: green'>(-1254 [-85.0%])</span> 221 |
| `memory_finalize_time_ms` |  11 |  22 |  14 |  8 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1181 [-24.7%])</span> 3,601 | <span style='color: green'>(-2362 [-24.7%])</span> 7,202 | <span style='color: green'>(-309 [-6.4%])</span> 4,483 | <span style='color: green'>(-2053 [-43.0%])</span> 2,719 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-222 [-25.5%])</span> 645.50 | <span style='color: green'>(-443 [-25.5%])</span> 1,291 | <span style='color: green'>(-104 [-12.0%])</span> 763 | <span style='color: green'>(-339 [-39.1%])</span> 528 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-69 [-17.5%])</span> 325.50 | <span style='color: green'>(-138 [-17.5%])</span> 651 | <span style='color: red'>(+4 [+1.0%])</span> 421 | <span style='color: green'>(-142 [-38.2%])</span> 230 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-391 [-31.1%])</span> 865.50 | <span style='color: green'>(-782 [-31.1%])</span> 1,731 | <span style='color: green'>(-138 [-11.0%])</span> 1,121 | <span style='color: green'>(-644 [-51.4%])</span> 610 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-102 [-18.9%])</span> 437 | <span style='color: green'>(-204 [-18.9%])</span> 874 | <span style='color: red'>(+5 [+0.9%])</span> 546 | <span style='color: green'>(-209 [-38.9%])</span> 328 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-122 [-29.4%])</span> 294.50 | <span style='color: green'>(-245 [-29.4%])</span> 589 | <span style='color: green'>(-71 [-16.9%])</span> 349 | <span style='color: green'>(-174 [-42.0%])</span> 240 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-278 [-21.3%])</span> 1,026.50 | <span style='color: green'>(-555 [-21.3%])</span> 2,053 | <span style='color: green'>(-37 [-2.8%])</span> 1,276 | <span style='color: green'>(-518 [-40.0%])</span> 777 |



<details>
<summary>Detailed Metrics</summary>

|  | vm.create_initial_state_time_ms | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- |
|  | 0 | 51 | 19 | 7,505 | 11,016 | 

| group | vm.reset_state_time_ms | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf |  | 6,617 |  | 1 |  |  | 1 |  |  |  | 
| regex_program | 1 |  | 2,686 |  | 6 | 4,108,597 | 1 | 31 | 128.53 | 45 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 572 | 
| leaf | PhantomAir | 2 | 3 | 5 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 11 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 
| regex_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,513 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| regex_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | FriReducedOpeningAir | 1 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 20 | 12 | 16,777,216 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 52 | 41 | 195,035,136 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 256 |  | 72 | 59 | 33,536 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | vm.reset_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 221 | 4,395 | 217,252,880 | 319,626,730 | 221 | 2,719 | 0 | 328 | 240 | 610 | 777 | 8 | 528 | 90,830,038 | 1,496,566 | 230 | 425 | 4.22 | 
| leaf | 1 | 5 | 355 | 6,615 | 391,464,962 | 538,660,330 | 355 | 4,483 | 0 | 546 | 349 | 1,121 | 1,276 | 14 | 763 | 153,709,964 | 1,830,117 | 421 | 735 | 4.54 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,963,908 | 2,013,265,921 | 
| leaf | 0 | 1 | 32,649,472 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,981,954 | 2,013,265,921 | 
| leaf | 0 | 3 | 32,383,236 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 74,765,002 | 2,013,265,921 | 
| leaf | 1 | 0 | 9,371,780 | 2,013,265,921 | 
| leaf | 1 | 1 | 64,930,048 | 2,013,265,921 | 
| leaf | 1 | 2 | 4,685,890 | 2,013,265,921 | 
| leaf | 1 | 3 | 65,044,740 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 144,818,890 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 506 | 4,729 | 211,758,557 | 373,183,036 | 506 | 3,876 | 0 | 435 | 374 | 770 | 1,221 | 7 | 13 | 770 | 91,281,687 | 2,211,000 | 299 | 105 | 25.02 | 
| regex_program | 1 | 283 | 2,686 | 174,326,267 | 196,838,026 | 283 | 2,090 | 54 | 209 | 182 | 405 | 755 | 6 | 2 | 377 | 73,453,305 | 1,897,597 | 153 | 89 | 29.74 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 7,965,702 | 2,013,265,921 | 
| regex_program | 0 | 1 | 22,979,584 | 2,013,265,921 | 
| regex_program | 0 | 2 | 3,982,851 | 2,013,265,921 | 
| regex_program | 0 | 3 | 28,094,468 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,669,056 | 2,013,265,921 | 
| regex_program | 0 | 7 | 135,168 | 2,013,265,921 | 
| regex_program | 0 | 8 | 71,678,221 | 2,013,265,921 | 
| regex_program | 1 | 0 | 4,358,276 | 2,013,265,921 | 
| regex_program | 1 | 1 | 12,037,120 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,179,138 | 2,013,265,921 | 
| regex_program | 1 | 3 | 15,047,780 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,887,424 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 38,711,914 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/45182e505a44bdcbd6f9fb46180ecd108c3fac7d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16946927424)
