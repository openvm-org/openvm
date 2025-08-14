| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  19.09 |  11.82 |
| regex_program |  7.63 |  4.90 |
| leaf |  11.43 |  6.89 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,817 |  7,634 |  4,904 |  2,730 |
| `main_cells_used     ` | <span style='color: green'>(-888080 [-1.1%])</span> 82,367,496 | <span style='color: green'>(-1776160 [-1.1%])</span> 164,734,992 | <span style='color: green'>(-2219112 [-2.4%])</span> 91,281,687 | <span style='color: red'>(+442952 [+0.6%])</span> 73,453,305 |
| `total_cells_used    ` |  193,042,412 |  386,084,824 |  211,758,557 |  174,326,267 |
| `execute_metered_time_ms` |  32 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  125.44 | -          |  125.44 |  125.44 |
| `execute_preflight_insns` |  2,054,298.50 |  4,108,597 |  2,211,000 |  1,897,597 |
| `execute_preflight_time_ms` |  94.50 |  189 |  104 |  85 |
| `execute_preflight_insn_mi/s` |  28.16 | -          |  30.58 |  25.73 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-592 [-60.9%])</span> 380 | <span style='color: green'>(-1184 [-60.9%])</span> 760 | <span style='color: green'>(-657 [-57.3%])</span> 490 | <span style='color: green'>(-527 [-66.1%])</span> 270 |
| `memory_finalize_time_ms` |  8 |  16 |  14 |  2 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+462 [+17.4%])</span> 3,114 | <span style='color: red'>(+925 [+17.4%])</span> 6,228 | <span style='color: red'>(+1243 [+44.0%])</span> 4,065 | <span style='color: green'>(-318 [-12.8%])</span> 2,163 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+125 [+25.0%])</span> 625.50 | <span style='color: red'>(+250 [+25.0%])</span> 1,251 | <span style='color: red'>(+284 [+51.0%])</span> 841 | <span style='color: green'>(-34 [-7.7%])</span> 410 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+16 [+7.1%])</span> 235 | <span style='color: red'>(+31 [+7.1%])</span> 470 | <span style='color: red'>(+88 [+37.9%])</span> 320 | <span style='color: green'>(-57 [-27.5%])</span> 150 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+102 [+18.3%])</span> 658.50 | <span style='color: red'>(+204 [+18.3%])</span> 1,317 | <span style='color: red'>(+274 [+47.6%])</span> 850 | <span style='color: green'>(-70 [-13.0%])</span> 467 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+43 [+15.3%])</span> 324 | <span style='color: red'>(+86 [+15.3%])</span> 648 | <span style='color: red'>(+140 [+46.2%])</span> 443 | <span style='color: green'>(-54 [-20.8%])</span> 205 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+50 [+20.2%])</span> 295 | <span style='color: red'>(+99 [+20.2%])</span> 590 | <span style='color: red'>(+113 [+40.2%])</span> 394 | <span style='color: green'>(-14 [-6.7%])</span> 196 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+130 [+15.6%])</span> 968.50 | <span style='color: red'>(+261 [+15.6%])</span> 1,937 | <span style='color: red'>(+349 [+40.4%])</span> 1,212 | <span style='color: green'>(-88 [-10.8%])</span> 725 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  5,713.50 |  11,427 |  6,887 |  4,540 |
| `main_cells_used     ` | <span style='color: green'>(-29558624 [-19.5%])</span> 122,269,389 | <span style='color: green'>(-59117248 [-19.5%])</span> 244,538,778 | <span style='color: green'>(-518970 [-0.3%])</span> 153,708,908 | <span style='color: green'>(-58598278 [-39.2%])</span> 90,829,870 |
| `total_cells_used    ` |  304,356,881 |  608,713,762 |  391,461,442 |  217,252,320 |
| `execute_preflight_insns` |  1,663,290.50 |  3,326,581 |  1,830,029 |  1,496,552 |
| `execute_preflight_time_ms` |  582.50 |  1,165 |  743 |  422 |
| `execute_preflight_insn_mi/s` |  4.43 | -          |  4.58 |  4.29 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1202 [-80.1%])</span> 299 | <span style='color: green'>(-2403 [-80.1%])</span> 598 | <span style='color: green'>(-1157 [-75.8%])</span> 369 | <span style='color: green'>(-1246 [-84.5%])</span> 229 |
| `memory_finalize_time_ms` |  11 |  22 |  14 |  8 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-982 [-20.5%])</span> 3,799.50 | <span style='color: green'>(-1965 [-20.5%])</span> 7,599 | <span style='color: green'>(-49 [-1.0%])</span> 4,743 | <span style='color: green'>(-1916 [-40.2%])</span> 2,856 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-160 [-18.4%])</span> 707.50 | <span style='color: green'>(-319 [-18.4%])</span> 1,415 | <span style='color: green'>(-29 [-3.3%])</span> 838 | <span style='color: green'>(-290 [-33.4%])</span> 577 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-44 [-11.3%])</span> 350 | <span style='color: green'>(-89 [-11.3%])</span> 700 | <span style='color: red'>(+34 [+8.2%])</span> 451 | <span style='color: green'>(-123 [-33.1%])</span> 249 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-300 [-23.8%])</span> 957 | <span style='color: green'>(-599 [-23.8%])</span> 1,914 | <span style='color: green'>(-16 [-1.3%])</span> 1,243 | <span style='color: green'>(-583 [-46.5%])</span> 671 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-118 [-21.8%])</span> 421.50 | <span style='color: green'>(-235 [-21.8%])</span> 843 | <span style='color: green'>(-5 [-0.9%])</span> 536 | <span style='color: green'>(-230 [-42.8%])</span> 307 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-102 [-24.6%])</span> 314.50 | <span style='color: green'>(-205 [-24.6%])</span> 629 | <span style='color: green'>(-45 [-10.7%])</span> 375 | <span style='color: green'>(-160 [-38.6%])</span> 254 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-260 [-19.9%])</span> 1,044 | <span style='color: green'>(-520 [-19.9%])</span> 2,088 | <span style='color: green'>(-17 [-1.3%])</span> 1,296 | <span style='color: green'>(-503 [-38.8%])</span> 792 |



<details>
<summary>Detailed Metrics</summary>

|  | vm.create_initial_state_time_ms | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- |
|  | 0 | 49 | 21 | 7,724 | 11,432 | 

| group | vm.reset_state_time_ms | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf |  | 6,889 |  | 1 |  | 1 |  |  |  |  | 
| regex_program | 1 |  | 2,730 |  | 8 | 1 | 32 | 4,108,597 | 125.44 | 46 | 

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

| group | idx | vm.reset_state_time_ms | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 229 | 4,540 | 217,252,320 | 319,626,730 | 229 | 2,856 | 0 | 307 | 254 | 671 | 792 | 8 | 577 | 90,829,870 | 249 | 422 | 1,496,552 | 4.29 | 
| leaf | 1 | 4 | 369 | 6,887 | 391,461,442 | 538,660,330 | 369 | 4,743 | 0 | 536 | 375 | 1,243 | 1,296 | 14 | 838 | 153,708,908 | 451 | 743 | 1,830,029 | 4.58 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 490 | 4,904 | 211,758,557 | 373,183,036 | 490 | 4,065 | 0 | 443 | 394 | 850 | 1,212 | 7 | 14 | 841 | 91,281,687 | 320 | 104 | 2,211,000 | 25.73 | 
| regex_program | 1 | 270 | 2,730 | 174,326,267 | 196,838,026 | 270 | 2,163 | 49 | 205 | 196 | 467 | 725 | 7 | 2 | 410 | 73,453,305 | 150 | 85 | 1,897,597 | 30.58 | 

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


Commit: https://github.com/openvm-org/openvm/commit/ed00aedaae96866ba69f76c17df20c507bba3ccb

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16972639651)
