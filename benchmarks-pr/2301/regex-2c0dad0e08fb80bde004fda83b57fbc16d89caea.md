| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  15.62 |  5.82 | 5.82 |
| regex_program |  4.87 |  1.73 |  1.73 |
| leaf |  10.75 |  4.09 |  4.09 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,207.50 |  4,830 |  1,692 |  769 |
| `main_cells_used     ` |  4,423,804 |  17,695,216 |  10,899,170 |  2,177,334 |
| `total_cells_used    ` |  14,619,340 |  58,477,360 |  23,446,148 |  11,579,064 |
| `execute_metered_time_ms` |  37 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  109.75 | -          |  109.75 |  109.75 |
| `execute_preflight_insns` |  1,034,375.50 |  4,137,502 |  1,104,000 |  826,502 |
| `execute_preflight_time_ms` |  54.50 |  218 |  66 |  46 |
| `execute_preflight_insn_mi/s` |  30.31 | -          |  31.60 |  27.17 |
| `trace_gen_time_ms   ` |  169 |  676 |  182 |  154 |
| `memory_finalize_time_ms` |  1 |  4 |  4 |  0 |
| `stark_prove_excluding_trace_time_ms` |  910.50 |  3,642 |  1,296 |  500 |
| `main_trace_commit_time_ms` |  51.75 |  207 |  58 |  49 |
| `generate_perm_trace_time_ms` |  27.25 |  109 |  33 |  22 |
| `perm_trace_commit_time_ms` |  63.94 |  255.75 |  71.74 |  55.07 |
| `quotient_poly_compute_time_ms` |  60.60 |  242.41 |  74.73 |  53.56 |
| `quotient_poly_commit_time_ms` |  15.35 |  61.42 |  17.50 |  13.95 |
| `pcs_opening_time_ms ` |  688.50 |  2,754 |  1,078 |  279 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,687.25 |  10,749 |  4,094 |  1,859 |
| `main_cells_used     ` |  15,207,682.50 |  60,830,730 |  24,205,574 |  11,789,672 |
| `total_cells_used    ` |  37,868,441.50 |  151,473,766 |  60,115,460 |  29,451,830 |
| `execute_preflight_insns` |  2,313,070.50 |  9,252,282 |  2,693,974 |  2,073,371 |
| `execute_preflight_time_ms` |  1,075 |  4,300 |  1,403 |  391 |
| `execute_preflight_insn_mi/s` |  6.20 | -          |  6.69 |  5.64 |
| `trace_gen_time_ms   ` |  129.25 |  517 |  218 |  95 |
| `memory_finalize_time_ms` |  12.75 |  51 |  24 |  9 |
| `stark_prove_excluding_trace_time_ms` |  1,481.25 |  5,925 |  2,544 |  977 |
| `main_trace_commit_time_ms` |  213.75 |  855 |  378 |  144 |
| `generate_perm_trace_time_ms` |  92.75 |  371 |  173 |  60 |
| `perm_trace_commit_time_ms` |  357.33 |  1,429.32 |  697.70 |  226.65 |
| `quotient_poly_compute_time_ms` |  251.51 |  1,006.04 |  460.30 |  172.44 |
| `quotient_poly_commit_time_ms` |  54.49 |  217.96 |  105.22 |  30.76 |
| `pcs_opening_time_ms ` |  506.50 |  2,026 |  721 |  338 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 73 | 4,882 | 10,769 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 4,097 |  | 1 | 1 |  |  |  |  | 
| regex_program |  | 769 |  | 1 | 37 | 4,137,502 | 109.75 | 0 | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | 0 | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | 0 | ProgramAir | 2 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | 0 | ProgramAir | 3 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 1 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 2 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 3 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | 15 | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 2 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 3 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 2 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 3 | 524,288 |  | 20 | 12 | 16,777,216 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 3 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 3 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 2 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 3 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | 8 | FriReducedOpeningAir | 2 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | 8 | FriReducedOpeningAir | 3 | 4,194,304 |  | 84 | 27 | 465,567,744 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 1 | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 1 | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | 10 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 16,384 |  | 52 | 31 | 1,359,872 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 12 | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | 13 | KeccakVmAir | 3 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | 14 | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 16,384 |  | 28 | 20 | 786,432 | 
| regex_program | 15 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 8,192 |  | 28 | 20 | 393,216 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | 16 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 32,768 |  | 36 | 28 | 2,097,152 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 17 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 32,768 |  | 28 | 18 | 1,507,328 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 65,536 |  | 32 | 32 | 4,194,304 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 19 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 65,536 |  | 28 | 26 | 3,538,944 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 1 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 2 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 2 | PersistentBoundaryAir<8> | 3 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 3 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 21 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 524,288 |  | 52 | 41 | 48,758,784 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 65,536 |  | 52 | 53 | 6,881,280 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 8,192 |  | 40 | 37 | 630,784 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 524,288 |  | 52 | 36 | 46,137,344 | 
| regex_program | 24 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 262,144 |  | 52 | 36 | 23,068,672 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 25 | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | 26 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | 27 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | 28 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 28 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 28 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 28 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | 3 | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | 3 | MemoryMerkleAir<8> | 1 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | 3 | MemoryMerkleAir<8> | 2 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | 3 | MemoryMerkleAir<8> | 3 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | 6 | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | 6 | AccessAdapterAir<8> | 1 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | 6 | AccessAdapterAir<8> | 2 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | 6 | AccessAdapterAir<8> | 3 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | 9 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 256 |  | 72 | 59 | 33,536 | 

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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 109 | 1,859 | 32,454,646 | 471,277,034 | 109 | 1,357 | 0 | 198.93 | 50.92 | 6 | 277.44 | 562 | 355 | 561 | 9 | 188 | 13,045,812 | 76 | 391 | 2,411,532 | 6.69 | 52 | 251 | 4 | 561 | 
| leaf | 1 | 95 | 2,477 | 29,451,830 | 403,119,594 | 95 | 977 | 0 | 172.44 | 30.76 | 6 | 226.65 | 338 | 288 | 338 | 9 | 145 | 11,789,672 | 60 | 1,403 | 2,073,405 | 6.26 | 38 | 204 | 0 | 338 | 
| leaf | 2 | 95 | 2,319 | 29,451,830 | 403,119,594 | 95 | 1,047 | 0 | 174.36 | 31.06 | 6 | 227.53 | 405 | 290 | 404 | 9 | 144 | 11,789,672 | 62 | 1,175 | 2,073,371 | 6.22 | 38 | 206 | 0 | 404 | 
| leaf | 3 | 218 | 4,094 | 60,115,460 | 989,416,938 | 218 | 2,544 | 0 | 460.30 | 105.22 | 6 | 697.70 | 721 | 871 | 720 | 24 | 378 | 24,205,574 | 173 | 1,331 | 2,693,974 | 5.64 | 136 | 572 | 9 | 720 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 9,764,996 | 2,013,265,921 | 
| leaf | 0 | 1 | 50,344,192 | 2,013,265,921 | 
| leaf | 0 | 2 | 4,882,498 | 2,013,265,921 | 
| leaf | 0 | 3 | 50,602,244 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 116,380,362 | 2,013,265,921 | 
| leaf | 1 | 0 | 7,667,844 | 2,013,265,921 | 
| leaf | 1 | 1 | 44,052,736 | 2,013,265,921 | 
| leaf | 1 | 2 | 3,833,922 | 2,013,265,921 | 
| leaf | 1 | 3 | 44,310,788 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 100,651,722 | 2,013,265,921 | 
| leaf | 2 | 0 | 7,667,844 | 2,013,265,921 | 
| leaf | 2 | 1 | 44,052,736 | 2,013,265,921 | 
| leaf | 2 | 2 | 3,833,922 | 2,013,265,921 | 
| leaf | 2 | 3 | 44,310,788 | 2,013,265,921 | 
| leaf | 2 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 2 | 5 | 100,651,722 | 2,013,265,921 | 
| leaf | 3 | 0 | 18,153,604 | 2,013,265,921 | 
| leaf | 3 | 1 | 122,470,656 | 2,013,265,921 | 
| leaf | 3 | 2 | 9,076,802 | 2,013,265,921 | 
| leaf | 3 | 3 | 122,716,420 | 2,013,265,921 | 
| leaf | 3 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 3 | 5 | 273,466,058 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 179 | 1,692 | 23,446,148 | 150,778,428 | 179 | 1,296 | 0 | 59.53 | 17.50 | 8 | 71.74 | 1,062 | 98 | 1,062 | 4 | 58 | 10,899,170 | 25 | 47 | 1,103,000 | 27.17 | 21 | 77 | 0 | 1,062 | 
| regex_program | 1 | 154 | 1,554 | 11,598,276 | 128,513,066 | 154 | 1,293 | 1 | 53.56 | 14.93 | 6 | 64.28 | 1,078 | 94 | 1,078 | 0 | 51 | 2,196,106 | 29 | 66 | 1,104,000 | 31.02 | 16 | 69 | 0 | 1,078 | 
| regex_program | 2 | 161 | 815 | 11,579,064 | 128,513,066 | 161 | 553 | 1 | 54.59 | 15.04 | 6 | 64.66 | 335 | 98 | 335 | 0 | 49 | 2,177,334 | 33 | 59 | 1,104,000 | 31.60 | 16 | 70 | 0 | 335 | 
| regex_program | 3 | 182 | 769 | 11,853,872 | 103,456,394 | 182 | 500 | 0 | 74.73 | 13.95 | 7 | 55.07 | 279 | 80 | 279 | 0 | 49 | 2,422,606 | 22 | 46 | 826,502 | 31.46 | 19 | 89 | 0 | 279 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 2,870,278 | 2,013,265,921 | 
| regex_program | 0 | 1 | 8,479,744 | 2,013,265,921 | 
| regex_program | 0 | 2 | 1,435,139 | 2,013,265,921 | 
| regex_program | 0 | 3 | 9,728,004 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 3,302,144 | 2,013,265,921 | 
| regex_program | 0 | 7 | 69,632 | 2,013,265,921 | 
| regex_program | 0 | 8 | 27,736,333 | 2,013,265,921 | 
| regex_program | 1 | 0 | 2,768,900 | 2,013,265,921 | 
| regex_program | 1 | 1 | 7,720,960 | 2,013,265,921 | 
| regex_program | 1 | 2 | 1,384,450 | 2,013,265,921 | 
| regex_program | 1 | 3 | 9,357,316 | 2,013,265,921 | 
| regex_program | 1 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 1 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 1 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 1 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 1 | 8 | 25,637,898 | 2,013,265,921 | 
| regex_program | 2 | 0 | 2,768,900 | 2,013,265,921 | 
| regex_program | 2 | 1 | 7,720,960 | 2,013,265,921 | 
| regex_program | 2 | 2 | 1,384,450 | 2,013,265,921 | 
| regex_program | 2 | 3 | 9,357,316 | 2,013,265,921 | 
| regex_program | 2 | 4 | 4,096 | 2,013,265,921 | 
| regex_program | 2 | 5 | 2,048 | 2,013,265,921 | 
| regex_program | 2 | 6 | 3,284,992 | 2,013,265,921 | 
| regex_program | 2 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 2 | 8 | 25,637,898 | 2,013,265,921 | 
| regex_program | 3 | 0 | 2,162,820 | 2,013,265,921 | 
| regex_program | 3 | 1 | 6,003,712 | 2,013,265,921 | 
| regex_program | 3 | 2 | 1,081,410 | 2,013,265,921 | 
| regex_program | 3 | 3 | 7,509,092 | 2,013,265,921 | 
| regex_program | 3 | 4 | 7,168 | 2,013,265,921 | 
| regex_program | 3 | 5 | 3,072 | 2,013,265,921 | 
| regex_program | 3 | 6 | 1,904,960 | 2,013,265,921 | 
| regex_program | 3 | 7 | 65,536 | 2,013,265,921 | 
| regex_program | 3 | 8 | 19,788,394 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/2c0dad0e08fb80bde004fda83b57fbc16d89caea

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20448996466)
