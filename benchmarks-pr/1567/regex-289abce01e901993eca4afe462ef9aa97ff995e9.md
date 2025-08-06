| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  15.80 |  9.97 |
| regex_program |  7.02 |  4.56 |
| leaf |  8.74 |  5.38 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,509.50 |  7,019 |  4,562 |  2,457 |
| `main_cells_used     ` | <span style='color: green'>(-888080 [-1.1%])</span> 82,367,496 | <span style='color: green'>(-1776160 [-1.1%])</span> 164,734,992 | <span style='color: green'>(-2219112 [-2.4%])</span> 91,281,687 | <span style='color: red'>(+442952 [+0.6%])</span> 73,453,305 |
| `total_cells_used    ` |  193,042,412 |  386,084,824 |  211,758,557 |  174,326,267 |
| `insns               ` |  2,739,064.67 |  8,217,194 |  4,108,597 |  1,897,597 |
| `execute_metered_time_ms` |  32 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  126.36 | -          |  126.36 |  126.36 |
| `execute_e3_time_ms  ` |  121.50 |  243 |  137 |  106 |
| `execute_e3_insn_mi/s` |  16.98 | -          |  17.85 |  16.10 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-592 [-60.9%])</span> 380 | <span style='color: green'>(-1184 [-60.9%])</span> 760 | <span style='color: green'>(-653 [-56.9%])</span> 494 | <span style='color: green'>(-531 [-66.6%])</span> 266 |
| `memory_finalize_time_ms` |  7 |  14 |  12 |  2 |
| `boundary_finalize_time_ms` |  2 |  4 |  4 |  0 |
| `merkle_finalize_time_ms` |  140.50 |  281 |  229 |  52 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+356 [+13.4%])</span> 3,008 | <span style='color: red'>(+713 [+13.4%])</span> 6,016 | <span style='color: red'>(+1109 [+39.3%])</span> 3,931 | <span style='color: green'>(-396 [-16.0%])</span> 2,085 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+80 [+15.9%])</span> 580 | <span style='color: red'>(+159 [+15.9%])</span> 1,160 | <span style='color: red'>(+220 [+39.5%])</span> 777 | <span style='color: green'>(-61 [-13.7%])</span> 383 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+41 [+18.7%])</span> 260.50 | <span style='color: red'>(+82 [+18.7%])</span> 521 | <span style='color: red'>(+127 [+54.7%])</span> 359 | <span style='color: green'>(-45 [-21.7%])</span> 162 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+33 [+5.9%])</span> 589.50 | <span style='color: red'>(+66 [+5.9%])</span> 1,179 | <span style='color: red'>(+195 [+33.9%])</span> 771 | <span style='color: green'>(-129 [-24.0%])</span> 408 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+34 [+12.3%])</span> 315.50 | <span style='color: red'>(+69 [+12.3%])</span> 631 | <span style='color: red'>(+128 [+42.2%])</span> 431 | <span style='color: green'>(-59 [-22.8%])</span> 200 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+38 [+15.3%])</span> 283 | <span style='color: red'>(+75 [+15.3%])</span> 566 | <span style='color: red'>(+90 [+32.0%])</span> 371 | <span style='color: green'>(-15 [-7.1%])</span> 195 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+133 [+15.9%])</span> 971 | <span style='color: red'>(+266 [+15.9%])</span> 1,942 | <span style='color: red'>(+352 [+40.8%])</span> 1,215 | <span style='color: green'>(-86 [-10.6%])</span> 727 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,372 |  8,744 |  5,380 |  3,364 |
| `main_cells_used     ` | <span style='color: green'>(-29558030 [-19.5%])</span> 122,269,983 | <span style='color: green'>(-59116060 [-19.5%])</span> 244,539,966 | <span style='color: green'>(-519234 [-0.3%])</span> 153,708,644 | <span style='color: green'>(-58596826 [-39.2%])</span> 90,831,322 |
| `total_cells_used    ` |  304,358,861 |  608,717,722 |  391,460,562 |  217,257,160 |
| `insns               ` |  1,663,340 |  3,326,680 |  1,830,007 |  1,496,673 |
| `execute_e3_time_ms  ` |  464 |  928 |  506 |  422 |
| `execute_e3_insn_mi/s` |  3.58 | -          |  3.61 |  3.54 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1206 [-80.4%])</span> 294 | <span style='color: green'>(-2413 [-80.4%])</span> 588 | <span style='color: green'>(-1163 [-76.2%])</span> 363 | <span style='color: green'>(-1250 [-84.7%])</span> 225 |
| `memory_finalize_time_ms` |  12 |  24 |  14 |  10 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1168 [-24.4%])</span> 3,614 | <span style='color: green'>(-2336 [-24.4%])</span> 7,228 | <span style='color: green'>(-281 [-5.9%])</span> 4,511 | <span style='color: green'>(-2055 [-43.1%])</span> 2,717 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-207 [-23.9%])</span> 660 | <span style='color: green'>(-414 [-23.9%])</span> 1,320 | <span style='color: green'>(-87 [-10.0%])</span> 780 | <span style='color: green'>(-327 [-37.7%])</span> 540 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-78 [-19.8%])</span> 316.50 | <span style='color: green'>(-156 [-19.8%])</span> 633 | <span style='color: green'>(-5 [-1.2%])</span> 412 | <span style='color: green'>(-151 [-40.6%])</span> 221 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-388 [-30.8%])</span> 869 | <span style='color: green'>(-775 [-30.8%])</span> 1,738 | <span style='color: green'>(-132 [-10.5%])</span> 1,127 | <span style='color: green'>(-643 [-51.3%])</span> 611 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-120 [-22.4%])</span> 418.50 | <span style='color: green'>(-241 [-22.4%])</span> 837 | <span style='color: green'>(-7 [-1.3%])</span> 534 | <span style='color: green'>(-234 [-43.6%])</span> 303 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-120 [-28.8%])</span> 297 | <span style='color: green'>(-240 [-28.8%])</span> 594 | <span style='color: green'>(-69 [-16.4%])</span> 351 | <span style='color: green'>(-171 [-41.3%])</span> 243 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-258 [-19.7%])</span> 1,046.50 | <span style='color: green'>(-515 [-19.7%])</span> 2,093 | <span style='color: green'>(-11 [-0.8%])</span> 1,302 | <span style='color: green'>(-504 [-38.9%])</span> 791 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 49 | 18 | 7,622 | 11,225 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 6,748 |  | 1 |  |  | 1 |  |  |  | 
| regex_program |  | 2,701 |  | 6 | 4,108,597 | 1 | 32 | 126.36 | 45 | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 225 | 3,364 | 217,257,160 | 319,626,730 | 225 | 2,717 | 2 | 303 | 243 | 611 | 791 | 10 | 540 | 90,831,322 | 1,496,673 | 221 | 422 | 3.54 | 0 | 
| leaf | 1 | 363 | 5,380 | 391,460,562 | 538,660,330 | 363 | 4,511 | 2 | 534 | 351 | 1,127 | 1,302 | 14 | 780 | 153,708,644 | 1,830,007 | 412 | 506 | 3.61 | 0 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 494 | 4,562 | 211,758,557 | 373,183,036 | 494 | 3,931 | 2 | 431 | 371 | 771 | 1,215 | 229 | 8 | 12 | 777 | 91,281,687 | 2,211,000 | 359 | 137 | 16.10 | 4 | 
| regex_program | 1 | 266 | 2,457 | 174,326,267 | 196,838,026 | 266 | 2,085 | 2 | 200 | 195 | 408 | 727 | 52 | 6 | 2 | 383 | 73,453,305 | 1,897,597 | 162 | 106 | 17.85 | 0 | 

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


Commit: https://github.com/openvm-org/openvm/commit/289abce01e901993eca4afe462ef9aa97ff995e9

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16779867517)
