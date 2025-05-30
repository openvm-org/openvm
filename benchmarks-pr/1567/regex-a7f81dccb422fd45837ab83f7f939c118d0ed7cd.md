| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-4 [-20.1%])</span> 16.48 | <span style='color: green'>(-1 [-10.3%])</span> 9.64 |
| regex_program | <span style='color: green'>(-0 [-5.0%])</span> 6.100 | <span style='color: green'>(-0 [-7.9%])</span> 3.76 |
| leaf | <span style='color: green'>(-4 [-28.5%])</span> 9.48 | <span style='color: green'>(-1 [-11.8%])</span> 5.89 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-184 [-5.0%])</span> 3,499 | <span style='color: green'>(-367 [-5.0%])</span> 6,998 | <span style='color: green'>(-323 [-7.9%])</span> 3,757 | <span style='color: green'>(-44 [-1.3%])</span> 3,241 |
| `main_cells_used     ` |  83,259,728 |  166,519,456 |  93,502,834 |  73,016,622 |
| `total_cycles        ` |  2,082,716 |  4,165,432 |  2,243,714 |  1,921,718 |
| `execute_time_ms     ` | <span style='color: green'>(-10 [-3.1%])</span> 326 | <span style='color: green'>(-21 [-3.1%])</span> 652 | <span style='color: green'>(-21 [-5.6%])</span> 354 |  298 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-328 [-34.3%])</span> 629 | <span style='color: green'>(-656 [-34.3%])</span> 1,258 | <span style='color: green'>(-447 [-39.9%])</span> 672 | <span style='color: green'>(-209 [-26.3%])</span> 586 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+155 [+6.5%])</span> 2,544 | <span style='color: red'>(+310 [+6.5%])</span> 5,088 | <span style='color: red'>(+145 [+5.6%])</span> 2,731 | <span style='color: red'>(+165 [+7.5%])</span> 2,357 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+62 [+13.7%])</span> 511.50 | <span style='color: red'>(+123 [+13.7%])</span> 1,023 | <span style='color: red'>(+60 [+11.9%])</span> 563 | <span style='color: red'>(+63 [+15.9%])</span> 460 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+1.5%])</span> 198.50 | <span style='color: red'>(+6 [+1.5%])</span> 397 | <span style='color: green'>(-3 [-1.4%])</span> 204 | <span style='color: red'>(+9 [+4.9%])</span> 193 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+58 [+11.9%])</span> 545.50 | <span style='color: red'>(+116 [+11.9%])</span> 1,091 | <span style='color: red'>(+59 [+11.5%])</span> 572 | <span style='color: red'>(+57 [+12.3%])</span> 519 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+11 [+3.3%])</span> 343 | <span style='color: red'>(+22 [+3.3%])</span> 686 | <span style='color: red'>(+11 [+3.1%])</span> 370 | <span style='color: red'>(+11 [+3.6%])</span> 316 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+18 [+7.9%])</span> 244.50 | <span style='color: red'>(+36 [+7.9%])</span> 489 | <span style='color: red'>(+18 [+6.8%])</span> 281 | <span style='color: red'>(+18 [+9.5%])</span> 208 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+6 [+0.9%])</span> 694.50 | <span style='color: red'>(+13 [+0.9%])</span> 1,389 | <span style='color: red'>(+4 [+0.5%])</span> 736 | <span style='color: red'>(+9 [+1.4%])</span> 653 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1888 [-28.5%])</span> 4,739 | <span style='color: green'>(-3777 [-28.5%])</span> 9,478 | <span style='color: green'>(-788 [-11.8%])</span> 5,886 | <span style='color: green'>(-2989 [-45.4%])</span> 3,592 |
| `main_cells_used     ` | <span style='color: green'>(-37368710 [-24.5%])</span> 114,909,903.50 | <span style='color: green'>(-74737419 [-24.5%])</span> 229,819,807 | <span style='color: green'>(-8033317 [-5.2%])</span> 146,644,957 | <span style='color: green'>(-66704102 [-44.5%])</span> 83,174,850 |
| `total_cycles        ` | <span style='color: green'>(-301218 [-15.1%])</span> 1,689,819.50 | <span style='color: green'>(-602435 [-15.1%])</span> 3,379,639 | <span style='color: green'>(-138950 [-6.9%])</span> 1,882,898 | <span style='color: green'>(-463485 [-23.6%])</span> 1,496,741 |
| `execute_time_ms     ` | <span style='color: green'>(-60 [-7.8%])</span> 718.50 | <span style='color: green'>(-121 [-7.8%])</span> 1,437 | <span style='color: red'>(+46 [+5.8%])</span> 834 | <span style='color: green'>(-167 [-21.7%])</span> 603 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1152 [-78.0%])</span> 324 | <span style='color: green'>(-2303 [-78.0%])</span> 648 | <span style='color: green'>(-1083 [-71.9%])</span> 423 | <span style='color: green'>(-1220 [-84.4%])</span> 225 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-676 [-15.5%])</span> 3,696.50 | <span style='color: green'>(-1353 [-15.5%])</span> 7,393 | <span style='color: red'>(+249 [+5.7%])</span> 4,629 | <span style='color: green'>(-1602 [-36.7%])</span> 2,764 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-56 [-7.1%])</span> 724.50 | <span style='color: green'>(-111 [-7.1%])</span> 1,449 | <span style='color: red'>(+86 [+11.0%])</span> 868 | <span style='color: green'>(-197 [-25.3%])</span> 581 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-104 [-27.2%])</span> 280 | <span style='color: green'>(-209 [-27.2%])</span> 560 | <span style='color: green'>(-36 [-9.0%])</span> 366 | <span style='color: green'>(-173 [-47.1%])</span> 194 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-159 [-14.3%])</span> 955 | <span style='color: green'>(-318 [-14.3%])</span> 1,910 | <span style='color: red'>(+124 [+11.1%])</span> 1,240 | <span style='color: green'>(-442 [-39.7%])</span> 670 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-129 [-19.6%])</span> 528 | <span style='color: green'>(-258 [-19.6%])</span> 1,056 | <span style='color: red'>(+13 [+2.0%])</span> 671 | <span style='color: green'>(-271 [-41.3%])</span> 385 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-50 [-12.9%])</span> 334 | <span style='color: green'>(-99 [-12.9%])</span> 668 | <span style='color: red'>(+28 [+7.3%])</span> 413 | <span style='color: green'>(-127 [-33.2%])</span> 255 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-178 [-16.9%])</span> 870.50 | <span style='color: green'>(-355 [-16.9%])</span> 1,741 | <span style='color: red'>(+14 [+1.3%])</span> 1,066 | <span style='color: green'>(-369 [-35.3%])</span> 675 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 580 | 20 | 

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
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
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
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
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
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 225 | 3,592 | 1,496,741 | 319,626,730 | 2,764 | 385 | 255 | 670 | 675 | 581 | 83,174,850 | 194 | 603 | 
| leaf | 1 | 423 | 5,886 | 1,882,898 | 538,660,330 | 4,629 | 671 | 413 | 1,240 | 1,066 | 868 | 146,644,957 | 366 | 834 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 672 | 3,757 | 2,243,714 | 275,648,700 | 2,731 | 370 | 281 | 572 | 736 | 563 | 93,502,834 | 204 | 354 | 
| regex_program | 1 | 586 | 3,241 | 1,921,718 | 242,975,388 | 2,357 | 316 | 208 | 519 | 653 | 460 | 73,016,622 | 193 | 298 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 5,868,294 | 2,013,265,921 | 
| regex_program | 0 | 1 | 16,687,360 | 2,013,265,921 | 
| regex_program | 0 | 2 | 2,934,147 | 2,013,265,921 | 
| regex_program | 0 | 3 | 19,705,092 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,800 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 53,849,229 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,406,854 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,182,848 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,703,427 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,193,508 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,508,864 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,197,677 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/a7f81dccb422fd45837ab83f7f939c118d0ed7cd

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15356879503)
