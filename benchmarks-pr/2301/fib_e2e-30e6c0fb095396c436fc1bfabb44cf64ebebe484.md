| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  61.64 |  13.14 | 13.14 |
| fib_e2e |  5.81 |  0.74 |  0.74 |
| leaf |  26.93 |  2.06 |  2.06 |
| internal.0 |  20.62 |  4.65 |  4.65 |
| internal.1 |  5.82 |  3.23 |  3.23 |
| internal.2 |  2.46 |  2.46 |  2.46 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  408 |  5,712 |  647 |  314 |
| `main_cells_used     ` |  1,014,539.57 |  14,203,554 |  1,064,416 |  1,008,596 |
| `total_cells_used    ` |  9,653,739.29 |  135,152,350 |  9,708,170 |  9,647,206 |
| `execute_metered_time_ms` |  95 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  126.10 | -          |  126.10 |  126.10 |
| `execute_preflight_insns` |  857,157.79 |  12,000,209 |  873,000 |  651,209 |
| `execute_preflight_time_ms` |  39.64 |  555 |  42 |  31 |
| `execute_preflight_insn_mi/s` |  39.55 | -          |  40.03 |  37.05 |
| `trace_gen_time_ms   ` |  35.57 |  498 |  43 |  33 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  281.71 |  3,944 |  530 |  197 |
| `main_trace_commit_time_ms` |  10.07 |  141 |  11 |  10 |
| `generate_perm_trace_time_ms` |  14.14 |  198 |  49 |  5 |
| `perm_trace_commit_time_ms` |  14.34 |  200.73 |  14.93 |  13.58 |
| `quotient_poly_compute_time_ms` |  11.18 |  156.50 |  12.13 |  10.66 |
| `quotient_poly_commit_time_ms` |  4.32 |  60.41 |  4.51 |  4.19 |
| `pcs_opening_time_ms ` |  225.29 |  3,154 |  439 |  140 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,923.86 |  26,934 |  2,059 |  1,812 |
| `main_cells_used     ` |  8,902,705.43 |  124,637,876 |  10,121,356 |  8,734,702 |
| `total_cells_used    ` |  23,485,838.86 |  328,801,744 |  26,398,858 |  23,084,280 |
| `execute_preflight_insns` |  1,719,342.21 |  24,070,791 |  2,061,668 |  1,672,432 |
| `execute_preflight_time_ms` |  1,451.57 |  20,322 |  1,475 |  1,429 |
| `execute_preflight_insn_mi/s` |  2.80 | -          |  3.29 |  2.71 |
| `trace_gen_time_ms   ` |  66.64 |  933 |  77 |  63 |
| `memory_finalize_time_ms` |  7.07 |  99 |  8 |  7 |
| `stark_prove_excluding_trace_time_ms` |  404 |  5,656 |  547 |  311 |
| `main_trace_commit_time_ms` |  39.21 |  549 |  41 |  39 |
| `generate_perm_trace_time_ms` |  22.21 |  311 |  63 |  14 |
| `perm_trace_commit_time_ms` |  54.47 |  762.64 |  56.42 |  53.99 |
| `quotient_poly_compute_time_ms` |  41.37 |  579.14 |  44.43 |  40.43 |
| `quotient_poly_commit_time_ms` |  9.20 |  128.77 |  9.60 |  9.07 |
| `pcs_opening_time_ms ` |  235.50 |  3,297 |  379 |  144 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,124.40 |  20,622 |  4,646 |  3,053 |
| `main_cells_used     ` |  25,288,239.60 |  126,441,198 |  26,857,916 |  19,009,534 |
| `total_cells_used    ` |  53,225,136 |  266,125,680 |  56,402,102 |  40,517,272 |
| `execute_preflight_insns` |  5,408,065.40 |  27,040,327 |  5,794,335 |  3,863,208 |
| `execute_preflight_time_ms` |  2,547 |  12,735 |  2,689 |  2,037 |
| `execute_preflight_insn_mi/s` |  3.17 | -          |  3.19 |  3.17 |
| `trace_gen_time_ms   ` |  188.60 |  943 |  214 |  142 |
| `memory_finalize_time_ms` |  10 |  50 |  11 |  9 |
| `stark_prove_excluding_trace_time_ms` |  1,386.40 |  6,932 |  1,775 |  871 |
| `main_trace_commit_time_ms` |  328 |  1,640 |  430 |  191 |
| `generate_perm_trace_time_ms` |  56.40 |  282 |  64 |  53 |
| `perm_trace_commit_time_ms` |  254.65 |  1,273.26 |  335.88 |  161.18 |
| `quotient_poly_compute_time_ms` |  312.79 |  1,563.94 |  354.85 |  221.19 |
| `quotient_poly_commit_time_ms` |  134.85 |  674.26 |  208.75 |  71.30 |
| `pcs_opening_time_ms ` |  293.40 |  1,467 |  370 |  170 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,908.50 |  5,817 |  3,230 |  2,587 |
| `main_cells_used     ` |  18,226,184 |  36,452,368 |  21,143,358 |  15,309,010 |
| `total_cells_used    ` |  38,933,246 |  77,866,492 |  44,841,000 |  33,025,492 |
| `execute_preflight_insns` |  3,872,377.50 |  7,744,755 |  4,658,120 |  3,086,635 |
| `execute_preflight_time_ms` |  1,890.50 |  3,781 |  2,103 |  1,678 |
| `execute_preflight_insn_mi/s` |  3.71 | -          |  3.74 |  3.69 |
| `trace_gen_time_ms   ` |  116.50 |  233 |  137 |  96 |
| `memory_finalize_time_ms` |  8 |  16 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  899.50 |  1,799 |  988 |  811 |
| `main_trace_commit_time_ms` |  194 |  388 |  224 |  164 |
| `generate_perm_trace_time_ms` |  32.50 |  65 |  38 |  27 |
| `perm_trace_commit_time_ms` |  145.83 |  291.65 |  167.83 |  123.82 |
| `quotient_poly_compute_time_ms` |  190.96 |  381.91 |  209.96 |  171.95 |
| `quotient_poly_commit_time_ms` |  70.54 |  141.08 |  85.18 |  55.90 |
| `pcs_opening_time_ms ` |  262 |  524 |  265 |  259 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,461 |  2,461 |  2,461 |  2,461 |
| `main_cells_used     ` |  15,309,010 |  15,309,010 |  15,309,010 |  15,309,010 |
| `total_cells_used    ` |  33,025,492 |  33,025,492 |  33,025,492 |  33,025,492 |
| `execute_preflight_insns` |  3,085,515 |  3,085,515 |  3,085,515 |  3,085,515 |
| `execute_preflight_time_ms` |  1,678 |  1,678 |  1,678 |  1,678 |
| `execute_preflight_insn_mi/s` |  3.69 | -          |  3.69 |  3.69 |
| `trace_gen_time_ms   ` |  96 |  96 |  96 |  96 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  685 |  685 |  685 |  685 |
| `main_trace_commit_time_ms` |  164 |  164 |  164 |  164 |
| `generate_perm_trace_time_ms` |  31 |  31 |  31 |  31 |
| `perm_trace_commit_time_ms` |  123.95 |  123.95 |  123.95 |  123.95 |
| `quotient_poly_compute_time_ms` |  171.38 |  171.38 |  171.38 |  171.38 |
| `quotient_poly_commit_time_ms` |  55.98 |  55.98 |  55.98 |  55.98 |
| `pcs_opening_time_ms ` |  136 |  136 |  136 |  136 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,003.50 |  18,007 |  14,929 |  3,078 |
| `main_cells_used     ` |  79,635,476.50 |  159,270,953 |  158,351,573 |  919,380 |
| `total_cells_used    ` |  197,884,270.50 |  395,768,541 |  386,262,975 |  9,505,566 |
| `execute_metered_time_ms` |  0 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.03 | -          |  0.03 |  0.03 |
| `execute_preflight_insns` |  1,329,262 |  2,658,524 |  2,658,523 |  1 |
| `execute_preflight_time_ms` |  622.50 |  1,245 |  1,244 |  1 |
| `execute_preflight_insn_mi/s` |  9,223,372,036,854,775,807 | -          |  9,223,372,036,854,775,807 |  3.86 |
| `trace_gen_time_ms   ` |  106.50 |  213 |  113 |  100 |
| `memory_finalize_time_ms` |  3.50 |  7 |  7 |  0 |
| `stark_prove_excluding_trace_time_ms` |  6,054.50 |  12,109 |  11,237 |  872 |
| `main_trace_commit_time_ms` |  1,175 |  2,350 |  2,247 |  103 |
| `generate_perm_trace_time_ms` |  428.50 |  857 |  848 |  9 |
| `perm_trace_commit_time_ms` |  968 |  1,936 |  1,841 |  95 |
| `quotient_poly_compute_time_ms` |  1,562 |  3,124 |  3,085 |  39 |
| `quotient_poly_commit_time_ms` |  1,011 |  2,022 |  1,910 |  112 |
| `pcs_opening_time_ms ` |  899.50 |  1,799 |  1,303 |  496 |



<details>
<summary>Detailed Metrics</summary>

|  | dummy_proof_and_keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 40,385 | 5,818 | 2,466 | 

| air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- |
| AccessAdapterAir<16> | 2 | 5 | 12 | 
| AccessAdapterAir<2> | 2 | 5 | 12 | 
| AccessAdapterAir<32> | 2 | 5 | 12 | 
| AccessAdapterAir<4> | 2 | 5 | 12 | 
| AccessAdapterAir<8> | 2 | 5 | 12 | 
| BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| MemoryMerkleAir<8> | 2 | 4 | 39 | 
| PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| PhantomAir | 2 | 3 | 5 | 
| Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ProgramAir | 1 | 1 | 4 | 
| RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| Rv32HintStoreAir | 2 | 18 | 28 | 
| VariableRangeCheckerAir | 1 | 1 | 4 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| VmConnectorAir | 2 | 5 | 11 | 

| group | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_children | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 113 | 14,929 | 386,262,975 | 515,190,250 | 113 | 11,237 | 0 |  |  | 3,085 | 1,910 | 3,078 | 1,841 | 1,303 |  | 7 | 2,247 | 158,351,573 | 848 |  | 1,244 | 2,658,523 | 3.86 | 0 | 1 | 0.03 | 0 | 
| fib_e2e |  |  |  |  |  |  |  |  |  |  |  | 320 |  |  |  |  |  |  |  | 1 |  |  |  | 95 | 12,000,209 | 126.10 | 0 | 
| internal.0 |  |  |  |  |  |  |  |  | 3,057 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| internal.1 |  |  |  |  |  |  |  |  | 2,590 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| internal.2 |  |  |  |  |  |  |  |  | 2,464 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| leaf |  |  |  |  |  |  |  | 1,967 |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |  |  |  |  | 

| group | air_id | air_name | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | ProgramAir | 131,072 |  | 8 | 10 | 2,359,296 | 
| agg_keygen | 1 | VmConnectorAir | 2 | 1 | 16 | 5 | 42 | 
| agg_keygen | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| agg_keygen | 11 | JalRangeCheckAir | 131,072 |  | 28 | 12 | 5,242,880 | 
| agg_keygen | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 262,144 |  | 28 | 23 | 13,369,344 | 
| agg_keygen | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 262,144 |  | 40 | 27 | 17,563,648 | 
| agg_keygen | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| agg_keygen | 15 | PhantomAir | 32,768 |  | 12 | 6 | 589,824 | 
| agg_keygen | 16 | VariableRangeCheckerAir | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| agg_keygen | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 64 |  | 28 | 27 | 3,520 | 
| agg_keygen | 3 | VolatileBoundaryAir | 262,144 |  | 20 | 12 | 8,388,608 | 
| agg_keygen | 4 | AccessAdapterAir<2> | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| agg_keygen | 5 | AccessAdapterAir<4> | 262,144 |  | 16 | 13 | 7,602,176 | 
| agg_keygen | 6 | AccessAdapterAir<8> | 8,192 |  | 16 | 17 | 270,336 | 
| agg_keygen | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 131,072 |  | 312 | 398 | 93,061,120 | 
| agg_keygen | 8 | FriReducedOpeningAir | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| agg_keygen | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 0 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | 1 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 1 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 1 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 1 | VmConnectorAir | 3 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 1 | VmConnectorAir | 4 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.0 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | 11 | JalRangeCheckAir | 0 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 11 | JalRangeCheckAir | 1 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 11 | JalRangeCheckAir | 2 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 11 | JalRangeCheckAir | 3 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 11 | JalRangeCheckAir | 4 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 24 | 21 | 94,371,840 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 2,097,152 |  | 24 | 21 | 94,371,840 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 2,097,152 |  | 24 | 21 | 94,371,840 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 2,097,152 |  | 24 | 21 | 94,371,840 | 
| internal.0 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | 15 | PhantomAir | 0 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | 15 | PhantomAir | 1 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | 15 | PhantomAir | 2 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | 15 | PhantomAir | 3 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | 15 | PhantomAir | 4 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 16 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 16 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 16 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | 3 | VolatileBoundaryAir | 0 | 524,288 |  | 12 | 12 | 12,582,912 | 
| internal.0 | 3 | VolatileBoundaryAir | 1 | 524,288 |  | 12 | 12 | 12,582,912 | 
| internal.0 | 3 | VolatileBoundaryAir | 2 | 524,288 |  | 12 | 12 | 12,582,912 | 
| internal.0 | 3 | VolatileBoundaryAir | 3 | 524,288 |  | 12 | 12 | 12,582,912 | 
| internal.0 | 3 | VolatileBoundaryAir | 4 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | 4 | AccessAdapterAir<2> | 0 | 2,097,152 |  | 12 | 11 | 48,234,496 | 
| internal.0 | 4 | AccessAdapterAir<2> | 1 | 2,097,152 |  | 12 | 11 | 48,234,496 | 
| internal.0 | 4 | AccessAdapterAir<2> | 2 | 2,097,152 |  | 12 | 11 | 48,234,496 | 
| internal.0 | 4 | AccessAdapterAir<2> | 3 | 2,097,152 |  | 12 | 11 | 48,234,496 | 
| internal.0 | 4 | AccessAdapterAir<2> | 4 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | 5 | AccessAdapterAir<4> | 0 | 1,048,576 |  | 12 | 13 | 26,214,400 | 
| internal.0 | 5 | AccessAdapterAir<4> | 1 | 1,048,576 |  | 12 | 13 | 26,214,400 | 
| internal.0 | 5 | AccessAdapterAir<4> | 2 | 1,048,576 |  | 12 | 13 | 26,214,400 | 
| internal.0 | 5 | AccessAdapterAir<4> | 3 | 1,048,576 |  | 12 | 13 | 26,214,400 | 
| internal.0 | 5 | AccessAdapterAir<4> | 4 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | 6 | AccessAdapterAir<8> | 0 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | 6 | AccessAdapterAir<8> | 1 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | 6 | AccessAdapterAir<8> | 2 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | 6 | AccessAdapterAir<8> | 3 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | 6 | AccessAdapterAir<8> | 4 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 524,288 |  | 160 | 398 | 292,552,704 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 524,288 |  | 160 | 398 | 292,552,704 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 524,288 |  | 160 | 398 | 292,552,704 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 524,288 |  | 160 | 398 | 292,552,704 | 
| internal.0 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | 8 | FriReducedOpeningAir | 0 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 8 | FriReducedOpeningAir | 1 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 8 | FriReducedOpeningAir | 2 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 8 | FriReducedOpeningAir | 3 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 8 | FriReducedOpeningAir | 4 | 2,097,152 |  | 44 | 27 | 148,897,792 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.0 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.1 | 0 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | 0 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | 1 | VmConnectorAir | 5 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | 1 | VmConnectorAir | 6 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 4,194,304 |  | 20 | 29 | 205,520,896 | 
| internal.1 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | 11 | JalRangeCheckAir | 5 | 262,144 |  | 16 | 12 | 7,340,032 | 
| internal.1 | 11 | JalRangeCheckAir | 6 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.1 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 524,288 |  | 24 | 27 | 26,738,688 | 
| internal.1 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.1 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.1 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.1 | 15 | PhantomAir | 5 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | 15 | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | 16 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | 16 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | 3 | VolatileBoundaryAir | 5 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.1 | 3 | VolatileBoundaryAir | 6 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.1 | 4 | AccessAdapterAir<2> | 5 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.1 | 4 | AccessAdapterAir<2> | 6 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.1 | 5 | AccessAdapterAir<4> | 5 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.1 | 5 | AccessAdapterAir<4> | 6 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.1 | 6 | AccessAdapterAir<8> | 5 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | 6 | AccessAdapterAir<8> | 6 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.1 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.1 | 8 | FriReducedOpeningAir | 5 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.1 | 8 | FriReducedOpeningAir | 6 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.1 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 524,288 |  | 20 | 38 | 30,408,704 | 
| internal.1 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.2 | 0 | ProgramAir | 7 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.2 | 1 | VmConnectorAir | 7 | 2 | 1 | 12 | 5 | 34 | 
| internal.2 | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 7 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.2 | 11 | JalRangeCheckAir | 7 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.2 | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 7 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.2 | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 7 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.2 | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 7 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.2 | 15 | PhantomAir | 7 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.2 | 16 | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 7 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | 3 | VolatileBoundaryAir | 7 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.2 | 4 | AccessAdapterAir<2> | 7 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.2 | 5 | AccessAdapterAir<4> | 7 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.2 | 6 | AccessAdapterAir<8> | 7 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.2 | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 7 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.2 | 8 | FriReducedOpeningAir | 7 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.2 | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 7 | 262,144 |  | 20 | 38 | 15,204,352 | 
| leaf | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 10 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 11 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 12 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 13 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 7 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 8 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 9 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 10 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 11 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 12 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 13 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 7 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 8 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 9 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 1 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 10 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 11 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 12 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 13 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 2 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 3 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 4 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 5 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 6 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 7 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 8 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 9 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 10 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 11 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 12 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 13 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 7 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 8 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 9 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 10 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 11 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 12 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 13 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 7 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 8 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 9 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 10 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 11 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 12 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 13 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 7 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 8 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 9 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 15 | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 10 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 11 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 12 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 13 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 2 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 3 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 4 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 5 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 6 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 7 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 8 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 9 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 13 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 10 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 11 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 12 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 13 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 7 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 8 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 9 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 10 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 11 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 12 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 13 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 2 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 3 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 4 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 5 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 6 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 7 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 8 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 9 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 10 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 11 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 12 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 13 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 3 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 4 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 5 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 6 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 7 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 8 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 9 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 10 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 11 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 12 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 13 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 3 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 4 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 5 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 6 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 7 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 8 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 9 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 10 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 11 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 12 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 13 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 2 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 4 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 5 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 6 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 7 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 8 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 6 | AccessAdapterAir<8> | 9 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 10 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 11 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 12 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 13 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 7 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 8 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 9 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 10 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 11 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 12 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 13 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 2 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 3 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 4 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 5 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 6 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 7 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 8 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 9 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 10 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 11 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 12 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 13 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 7 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 9 | 131,072 |  | 36 | 38 | 9,699,328 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | ProgramAir | 0 | 1 |  | 8 | 10 | 18 | 
| agg_keygen | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| agg_keygen | 10 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 1 |  | 72 | 39 | 111 | 
| agg_keygen | 11 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 1 |  | 52 | 31 | 83 | 
| agg_keygen | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| agg_keygen | 13 | Rv32HintStoreAir | 0 | 1 |  | 44 | 32 | 76 | 
| agg_keygen | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1 |  | 28 | 20 | 48 | 
| agg_keygen | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 1 |  | 36 | 28 | 64 | 
| agg_keygen | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1 |  | 28 | 18 | 46 | 
| agg_keygen | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 1 |  | 32 | 32 | 64 | 
| agg_keygen | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1 |  | 28 | 26 | 54 | 
| agg_keygen | 19 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1 |  | 52 | 36 | 88 | 
| agg_keygen | 2 | PersistentBoundaryAir<8> | 0 | 1 |  | 12 | 20 | 32 | 
| agg_keygen | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1 |  | 52 | 41 | 93 | 
| agg_keygen | 21 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1 |  | 52 | 53 | 105 | 
| agg_keygen | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 |  | 40 | 37 | 77 | 
| agg_keygen | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1 |  | 52 | 36 | 88 | 
| agg_keygen | 24 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| agg_keygen | 25 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| agg_keygen | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32 |  | 8 | 300 | 9,856 | 
| agg_keygen | 27 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| agg_keygen | 3 | MemoryMerkleAir<8> | 0 | 64 |  | 16 | 32 | 3,072 | 
| agg_keygen | 4 | AccessAdapterAir<2> | 0 | 1 |  | 16 | 11 | 27 | 
| agg_keygen | 5 | AccessAdapterAir<4> | 0 | 1 |  | 16 | 13 | 29 | 
| agg_keygen | 6 | AccessAdapterAir<8> | 0 | 1 |  | 16 | 17 | 33 | 
| agg_keygen | 7 | AccessAdapterAir<16> | 0 | 1 |  | 16 | 25 | 41 | 
| agg_keygen | 8 | AccessAdapterAir<32> | 0 | 1 |  | 16 | 41 | 57 | 
| agg_keygen | 9 | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 1 |  | 72 | 59 | 131 | 
| fib_e2e | 0 | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 10 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 11 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 12 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 13 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 2 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 3 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 4 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 5 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 6 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 7 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 8 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 0 | ProgramAir | 9 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 1 | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 10 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 11 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 12 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 13 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 7 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 8 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 12 | RangeTupleCheckerAir<2> | 9 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | 13 | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fib_e2e | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 13 | 4 |  | 28 | 20 | 192 | 
| fib_e2e | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fib_e2e | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 13 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 13 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fib_e2e | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fib_e2e | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 2 |  | 32 | 32 | 128 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 10 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 11 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 12 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 13 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 6 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 7 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 8 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 2 | PersistentBoundaryAir<8> | 9 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fib_e2e | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 13 | 64 |  | 52 | 41 | 5,952 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 13 | 131,072 |  | 40 | 37 | 10,092,544 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 13 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 13 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 24 | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | 25 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 13 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 13 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 27 | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 1 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 10 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 11 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 12 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 13 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 2 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 3 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 4 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 5 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 6 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 7 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 8 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 3 | MemoryMerkleAir<8> | 9 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 10 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 11 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 12 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 13 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 6 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 7 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 8 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | 6 | AccessAdapterAir<8> | 9 | 16 |  | 16 | 17 | 528 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| agg_keygen | AccessAdapterAir<16> | 2 | 5 | 12 | 
| agg_keygen | AccessAdapterAir<2> | 8 | 5 | 12 | 
| agg_keygen | AccessAdapterAir<32> | 2 | 5 | 12 | 
| agg_keygen | AccessAdapterAir<4> | 8 | 5 | 12 | 
| agg_keygen | AccessAdapterAir<8> | 8 | 5 | 12 | 
| agg_keygen | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| agg_keygen | FriReducedOpeningAir | 8 | 39 | 71 | 
| agg_keygen | JalRangeCheckAir | 8 | 9 | 14 | 
| agg_keygen | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| agg_keygen | NativePoseidon2Air<BabyBearParameters>, 1> | 8 | 136 | 572 | 
| agg_keygen | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| agg_keygen | PhantomAir | 4 | 3 | 5 | 
| agg_keygen | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| agg_keygen | ProgramAir | 1 | 1 | 4 | 
| agg_keygen | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| agg_keygen | Rv32HintStoreAir | 2 | 18 | 28 | 
| agg_keygen | VariableRangeCheckerAir | 1 | 1 | 4 | 
| agg_keygen | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 8 | 15 | 27 | 
| agg_keygen | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 8 | 11 | 25 | 
| agg_keygen | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 8 | 11 | 30 | 
| agg_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 8 | 15 | 20 | 
| agg_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 8 | 15 | 20 | 
| agg_keygen | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 8 | 15 | 27 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| agg_keygen | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| agg_keygen | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| agg_keygen | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| agg_keygen | VmConnectorAir | 8 | 5 | 11 | 
| agg_keygen | VolatileBoundaryAir | 8 | 7 | 19 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 191 | 4,646 | 56,402,102 | 920,340,962 | 191 | 1,775 | 0 | 354.85 | 208.75 | 6 | 335.88 | 370 | 401 | 369 | 11 | 430 | 26,857,916 | 64 | 2,678 | 5,794,293 | 3.17 | 67 | 573 | 18 | 369 | 
| internal.0 | 1 | 214 | 4,330 | 56,402,102 | 920,340,962 | 214 | 1,425 | 0 | 329.74 | 129.99 | 6 | 258.37 | 308 | 316 | 308 | 11 | 337 | 26,857,916 | 56 | 2,689 | 5,794,173 | 3.17 | 58 | 462 | 0 | 308 | 
| internal.0 | 2 | 205 | 4,266 | 56,402,102 | 920,340,962 | 205 | 1,394 | 0 | 328.76 | 131.93 | 6 | 258.79 | 274 | 314 | 274 | 9 | 340 | 26,857,916 | 55 | 2,664 | 5,794,318 | 3.17 | 58 | 463 | 0 | 274 | 
| internal.0 | 3 | 191 | 4,327 | 56,402,102 | 920,340,962 | 191 | 1,467 | 0 | 329.39 | 132.28 | 5 | 259.04 | 345 | 314 | 345 | 10 | 342 | 26,857,916 | 54 | 2,667 | 5,794,335 | 3.19 | 59 | 464 | 0 | 345 | 
| internal.0 | 4 | 142 | 3,053 | 40,517,272 | 579,447,266 | 142 | 871 | 0 | 221.19 | 71.30 | 5 | 161.18 | 170 | 215 | 170 | 9 | 191 | 19,009,534 | 53 | 2,037 | 3,863,208 | 3.18 | 37 | 294 | 0 | 170 | 
| internal.1 | 5 | 137 | 3,230 | 44,841,000 | 607,758,818 | 137 | 988 | 0 | 209.96 | 85.18 | 5 | 167.83 | 259 | 206 | 259 | 8 | 224 | 21,143,358 | 38 | 2,103 | 4,658,120 | 3.74 | 41 | 297 | 0 | 259 | 
| internal.1 | 6 | 96 | 2,587 | 33,025,492 | 472,754,658 | 96 | 811 | 0 | 171.95 | 55.90 | 5 | 123.82 | 265 | 151 | 265 | 8 | 164 | 15,309,010 | 27 | 1,678 | 3,086,635 | 3.69 | 31 | 229 | 0 | 265 | 
| internal.2 | 7 | 96 | 2,461 | 33,025,492 | 472,754,658 | 96 | 685 | 0 | 171.38 | 55.98 | 5 | 123.95 | 136 | 155 | 135 | 8 | 164 | 15,309,010 | 31 | 1,678 | 3,085,515 | 3.69 | 32 | 228 | 0 | 135 | 
| leaf | 0 | 77 | 1,965 | 26,398,858 | 342,564,330 | 77 | 411 | 0 | 44.17 | 9.60 | 6 | 56.29 | 195 | 120 | 195 | 7 | 41 | 10,121,356 | 63 | 1,475 | 2,061,668 | 3.29 | 14 | 54 | 0 | 195 | 
| leaf | 1 | 68 | 1,921 | 23,084,280 | 324,015,594 | 68 | 392 | 0 | 41.11 | 9.14 | 6 | 54.20 | 226 | 76 | 225 | 7 | 39 | 8,734,702 | 20 | 1,459 | 1,672,551 | 2.72 | 13 | 50 | 0 | 225 | 
| leaf | 10 | 64 | 1,958 | 23,084,280 | 324,015,594 | 64 | 436 | 0 | 40.43 | 9.09 | 5 | 54.17 | 271 | 75 | 271 | 7 | 39 | 8,734,702 | 20 | 1,457 | 1,672,600 | 2.72 | 13 | 49 | 0 | 271 | 
| leaf | 11 | 67 | 1,924 | 23,084,280 | 324,015,594 | 67 | 421 | 0 | 40.77 | 9.14 | 5 | 54.26 | 255 | 76 | 255 | 8 | 39 | 8,734,702 | 21 | 1,434 | 1,672,571 | 2.71 | 13 | 50 | 0 | 255 | 
| leaf | 12 | 67 | 1,812 | 23,084,280 | 324,015,594 | 67 | 315 | 0 | 40.60 | 9.15 | 6 | 54.17 | 156 | 69 | 156 | 7 | 39 | 8,734,702 | 14 | 1,429 | 1,672,578 | 2.74 | 13 | 50 | 0 | 156 | 
| leaf | 13 | 69 | 1,965 | 25,391,526 | 342,564,330 | 69 | 438 | 0 | 44.43 | 9.58 | 6 | 56.42 | 271 | 71 | 271 | 7 | 40 | 9,700,096 | 14 | 1,457 | 1,937,918 | 3.05 | 14 | 54 | 0 | 271 | 
| leaf | 2 | 64 | 1,940 | 23,084,280 | 324,015,594 | 64 | 411 | 0 | 40.58 | 9.09 | 5 | 54.08 | 238 | 83 | 238 | 7 | 39 | 8,734,702 | 28 | 1,463 | 1,672,656 | 2.76 | 13 | 50 | 0 | 238 | 
| leaf | 3 | 64 | 2,053 | 23,084,280 | 324,015,594 | 64 | 527 | 0 | 41.88 | 9.23 | 5 | 54.16 | 364 | 72 | 364 | 7 | 39 | 8,734,702 | 17 | 1,460 | 1,672,555 | 2.74 | 14 | 51 | 0 | 363 | 
| leaf | 4 | 64 | 1,916 | 23,084,280 | 324,015,594 | 64 | 406 | 0 | 41.65 | 9.17 | 5 | 54.18 | 240 | 75 | 240 | 7 | 39 | 8,734,702 | 20 | 1,444 | 1,672,432 | 2.79 | 13 | 51 | 0 | 240 | 
| leaf | 5 | 64 | 1,845 | 23,084,280 | 324,015,594 | 64 | 344 | 0 | 40.51 | 9.07 | 5 | 53.99 | 180 | 74 | 180 | 7 | 39 | 8,734,702 | 19 | 1,436 | 1,672,588 | 2.75 | 13 | 50 | 0 | 180 | 
| leaf | 6 | 64 | 2,059 | 23,084,280 | 324,015,594 | 64 | 547 | 0 | 40.75 | 9.13 | 6 | 54.05 | 379 | 78 | 379 | 7 | 39 | 8,734,702 | 23 | 1,446 | 1,672,665 | 2.77 | 13 | 50 | 0 | 379 | 
| leaf | 7 | 71 | 1,892 | 23,084,280 | 324,015,594 | 71 | 354 | 0 | 40.73 | 9.15 | 5 | 54.14 | 195 | 69 | 194 | 7 | 39 | 8,734,702 | 15 | 1,465 | 1,672,629 | 2.75 | 13 | 50 | 0 | 194 | 
| leaf | 8 | 67 | 1,858 | 23,084,280 | 324,015,594 | 67 | 343 | 0 | 40.62 | 9.08 | 5 | 54.30 | 183 | 70 | 183 | 7 | 39 | 8,734,702 | 15 | 1,447 | 1,672,630 | 2.76 | 13 | 50 | 0 | 183 | 
| leaf | 9 | 63 | 1,826 | 23,084,280 | 324,015,594 | 63 | 311 | 0 | 40.92 | 9.14 | 6 | 54.22 | 144 | 77 | 143 | 7 | 39 | 8,734,702 | 22 | 1,450 | 1,672,750 | 2.72 | 13 | 50 | 0 | 143 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 22,806,660 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 128,499,968 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 11,403,330 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 127,435,012 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 1,048,576 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 291,586,762 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 22,806,660 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 128,499,968 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 11,403,330 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 127,435,012 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 1,048,576 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 291,586,762 | 2,013,265,921 | 
| internal.0 | 2 | 0 | 22,806,660 | 2,013,265,921 | 
| internal.0 | 2 | 1 | 128,499,968 | 2,013,265,921 | 
| internal.0 | 2 | 2 | 11,403,330 | 2,013,265,921 | 
| internal.0 | 2 | 3 | 127,435,012 | 2,013,265,921 | 
| internal.0 | 2 | 4 | 1,048,576 | 2,013,265,921 | 
| internal.0 | 2 | 5 | 291,586,762 | 2,013,265,921 | 
| internal.0 | 3 | 0 | 22,806,660 | 2,013,265,921 | 
| internal.0 | 3 | 1 | 128,499,968 | 2,013,265,921 | 
| internal.0 | 3 | 2 | 11,403,330 | 2,013,265,921 | 
| internal.0 | 3 | 3 | 127,435,012 | 2,013,265,921 | 
| internal.0 | 3 | 4 | 1,048,576 | 2,013,265,921 | 
| internal.0 | 3 | 5 | 291,586,762 | 2,013,265,921 | 
| internal.0 | 4 | 0 | 15,335,556 | 2,013,265,921 | 
| internal.0 | 4 | 1 | 87,580,928 | 2,013,265,921 | 
| internal.0 | 4 | 2 | 7,667,778 | 2,013,265,921 | 
| internal.0 | 4 | 3 | 87,310,596 | 2,013,265,921 | 
| internal.0 | 4 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 4 | 5 | 198,812,362 | 2,013,265,921 | 
| internal.1 | 5 | 0 | 17,432,708 | 2,013,265,921 | 
| internal.1 | 5 | 1 | 81,289,472 | 2,013,265,921 | 
| internal.1 | 5 | 2 | 8,716,354 | 2,013,265,921 | 
| internal.1 | 5 | 3 | 81,019,140 | 2,013,265,921 | 
| internal.1 | 5 | 4 | 524,288 | 2,013,265,921 | 
| internal.1 | 5 | 5 | 189,375,178 | 2,013,265,921 | 
| internal.1 | 6 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.1 | 6 | 1 | 65,298,688 | 2,013,265,921 | 
| internal.1 | 6 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.1 | 6 | 3 | 64,766,212 | 2,013,265,921 | 
| internal.1 | 6 | 4 | 524,288 | 2,013,265,921 | 
| internal.1 | 6 | 5 | 148,873,930 | 2,013,265,921 | 
| internal.2 | 7 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.2 | 7 | 1 | 65,298,688 | 2,013,265,921 | 
| internal.2 | 7 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.2 | 7 | 3 | 64,766,212 | 2,013,265,921 | 
| internal.2 | 7 | 4 | 524,288 | 2,013,265,921 | 
| internal.2 | 7 | 5 | 148,873,930 | 2,013,265,921 | 
| leaf | 0 | 0 | 6,619,268 | 2,013,265,921 | 
| leaf | 0 | 1 | 34,615,552 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,309,634 | 2,013,265,921 | 
| leaf | 0 | 3 | 34,873,604 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 80,073,418 | 2,013,265,921 | 
| leaf | 1 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 1 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 1 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 1 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 10 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 10 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 10 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 10 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 10 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 10 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 11 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 11 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 11 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 11 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 11 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 11 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 12 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 12 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 12 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 12 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 12 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 12 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 13 | 0 | 6,619,268 | 2,013,265,921 | 
| leaf | 13 | 1 | 34,615,552 | 2,013,265,921 | 
| leaf | 13 | 2 | 3,309,634 | 2,013,265,921 | 
| leaf | 13 | 3 | 34,873,604 | 2,013,265,921 | 
| leaf | 13 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 13 | 5 | 80,073,418 | 2,013,265,921 | 
| leaf | 2 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 2 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 2 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 2 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 2 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 2 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 3 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 3 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 3 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 3 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 3 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 3 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 4 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 4 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 4 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 4 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 4 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 4 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 5 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 5 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 5 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 5 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 5 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 5 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 6 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 6 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 6 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 6 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 6 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 6 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 7 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 7 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 7 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 7 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 7 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 7 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 8 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 8 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 8 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 8 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 8 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 8 | 5 | 76,131,018 | 2,013,265,921 | 
| leaf | 9 | 0 | 6,094,980 | 2,013,265,921 | 
| leaf | 9 | 1 | 33,036,544 | 2,013,265,921 | 
| leaf | 9 | 2 | 3,047,490 | 2,013,265,921 | 
| leaf | 9 | 3 | 33,296,644 | 2,013,265,921 | 
| leaf | 9 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 9 | 5 | 76,131,018 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | 100 | 3,078 | 9,505,566 | 7,747,601 | 100 | 872 |  | 39 | 112 |  | 95 | 496 |  |  | 96 | 0 | 103 | 919,380 | 9 | 1 | 1 | 9,223,372,036,854,775,807 |  |  |  |  | 
| fib_e2e | 0 | 38 | 544 | 9,677,708 | 84,395,212 | 38 | 292 | 0 | 12.13 | 4.51 | 6 | 14.93 | 241 | 21 | 241 |  | 0 | 11 | 1,035,986 | 6 | 41 | 873,000 | 39.78 | 9 | 17 | 0 | 241 | 
| fib_e2e | 1 | 33 | 315 | 9,647,206 | 84,342,330 | 33 | 199 | 2 | 11.15 | 4.19 | 5 | 14.32 | 143 | 28 | 143 |  | 0 | 10 | 1,008,596 | 14 | 42 | 873,000 | 37.05 | 6 | 15 | 0 | 143 | 
| fib_e2e | 10 | 35 | 436 | 9,647,206 | 84,342,330 | 35 | 320 | 2 | 11.14 | 4.32 | 5 | 14.35 | 253 | 39 | 253 |  | 0 | 10 | 1,008,596 | 24 | 40 | 873,000 | 39.67 | 6 | 15 | 0 | 253 | 
| fib_e2e | 11 | 35 | 314 | 9,647,206 | 84,342,330 | 35 | 197 | 2 | 11.14 | 4.31 | 5 | 14.35 | 140 | 30 | 139 |  | 0 | 10 | 1,008,596 | 15 | 40 | 873,000 | 40.01 | 6 | 15 | 0 | 139 | 
| fib_e2e | 12 | 35 | 647 | 9,647,206 | 84,342,330 | 35 | 530 | 2 | 11.14 | 4.32 | 5 | 14.36 | 439 | 64 | 439 |  | 0 | 10 | 1,008,596 | 49 | 40 | 873,000 | 39.57 | 6 | 15 | 0 | 439 | 
| fib_e2e | 13 | 43 | 320 | 9,708,170 | 74,305,770 | 43 | 204 | 1 | 10.66 | 4.20 | 5 | 13.58 | 154 | 23 | 154 |  | 0 | 10 | 1,064,416 | 9 | 31 | 651,209 | 39.91 | 7 | 15 | 0 | 154 | 
| fib_e2e | 2 | 35 | 447 | 9,647,206 | 84,342,330 | 35 | 330 | 2 | 11.13 | 4.31 | 5 | 14.34 | 275 | 28 | 275 |  | 0 | 10 | 1,008,596 | 13 | 40 | 873,000 | 40.03 | 6 | 15 | 0 | 275 | 
| fib_e2e | 3 | 35 | 361 | 9,647,206 | 84,342,330 | 35 | 244 | 2 | 11.14 | 4.33 | 4 | 14.34 | 188 | 29 | 188 |  | 0 | 10 | 1,008,596 | 14 | 40 | 873,000 | 39.80 | 6 | 15 | 0 | 188 | 
| fib_e2e | 4 | 34 | 368 | 9,647,206 | 84,342,330 | 34 | 251 | 2 | 11.14 | 4.31 | 5 | 14.37 | 197 | 27 | 197 |  | 0 | 10 | 1,008,596 | 12 | 41 | 873,000 | 37.88 | 6 | 15 | 0 | 197 | 
| fib_e2e | 5 | 35 | 339 | 9,647,206 | 84,342,330 | 35 | 222 | 2 | 11.14 | 4.32 | 4 | 14.34 | 167 | 27 | 167 |  | 0 | 10 | 1,008,596 | 13 | 40 | 873,000 | 40.01 | 6 | 15 | 0 | 167 | 
| fib_e2e | 6 | 35 | 350 | 9,647,206 | 84,342,330 | 34 | 233 | 2 | 11.16 | 4.31 | 5 | 14.35 | 180 | 26 | 180 |  | 0 | 10 | 1,008,596 | 11 | 40 | 873,000 | 40.02 | 6 | 15 | 0 | 180 | 
| fib_e2e | 7 | 35 | 458 | 9,647,206 | 84,342,330 | 35 | 342 | 2 | 11.15 | 4.33 | 5 | 14.36 | 292 | 22 | 292 |  | 0 | 10 | 1,008,596 | 8 | 40 | 873,000 | 39.96 | 6 | 15 | 0 | 292 | 
| fib_e2e | 8 | 35 | 427 | 9,647,206 | 84,342,330 | 35 | 310 | 2 | 11.14 | 4.33 | 5 | 14.37 | 263 | 19 | 263 |  | 0 | 10 | 1,008,596 | 5 | 40 | 873,000 | 40.01 | 6 | 15 | 0 | 263 | 
| fib_e2e | 9 | 35 | 386 | 9,647,206 | 84,342,330 | 35 | 270 | 2 | 11.14 | 4.32 | 5 | 14.37 | 222 | 20 | 222 |  | 0 | 10 | 1,008,596 | 5 | 40 | 873,000 | 40.02 | 6 | 15 | 0 | 222 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| agg_keygen | 0 | 0 | 34 | 2,013,265,921 | 
| agg_keygen | 0 | 1 | 86 | 2,013,265,921 | 
| agg_keygen | 0 | 2 | 17 | 2,013,265,921 | 
| agg_keygen | 0 | 3 | 98 | 2,013,265,921 | 
| agg_keygen | 0 | 4 | 193 | 2,013,265,921 | 
| agg_keygen | 0 | 5 | 65 | 2,013,265,921 | 
| agg_keygen | 0 | 6 | 29 | 2,013,265,921 | 
| agg_keygen | 0 | 7 | 20 | 2,013,265,921 | 
| agg_keygen | 0 | 8 | 918,079 | 2,013,265,921 | 
| fib_e2e | 0 | 0 | 1,966,190 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 5,374,472 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 983,095 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 5,374,428 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 3,604,544 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 18,229,833 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 10 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 10 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 10 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 10 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 10 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 10 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 10 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 10 | 7 |  | 2,013,265,921 | 
| fib_e2e | 10 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 11 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 11 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 11 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 11 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 11 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 11 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 11 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 11 | 7 |  | 2,013,265,921 | 
| fib_e2e | 11 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 12 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 12 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 12 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 12 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 12 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 12 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 12 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 12 | 7 |  | 2,013,265,921 | 
| fib_e2e | 12 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 13 | 0 | 1,704,112 | 2,013,265,921 | 
| fib_e2e | 13 | 1 | 4,588,240 | 2,013,265,921 | 
| fib_e2e | 13 | 2 | 852,056 | 2,013,265,921 | 
| fib_e2e | 13 | 3 | 4,588,308 | 2,013,265,921 | 
| fib_e2e | 13 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 13 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 13 | 6 | 3,211,304 | 2,013,265,921 | 
| fib_e2e | 13 | 7 |  | 2,013,265,921 | 
| fib_e2e | 13 | 8 | 15,871,124 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 7 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 7 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 7 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 7 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 7 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 7 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 7 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 7 | 7 |  | 2,013,265,921 | 
| fib_e2e | 7 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 8 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 8 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 8 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 8 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 8 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 8 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 8 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 8 | 7 |  | 2,013,265,921 | 
| fib_e2e | 8 | 8 | 18,227,978 | 2,013,265,921 | 
| fib_e2e | 9 | 0 | 1,966,084 | 2,013,265,921 | 
| fib_e2e | 9 | 1 | 5,374,016 | 2,013,265,921 | 
| fib_e2e | 9 | 2 | 983,042 | 2,013,265,921 | 
| fib_e2e | 9 | 3 | 5,373,988 | 2,013,265,921 | 
| fib_e2e | 9 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 9 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 9 | 6 | 3,604,480 | 2,013,265,921 | 
| fib_e2e | 9 | 7 |  | 2,013,265,921 | 
| fib_e2e | 9 | 8 | 18,227,978 | 2,013,265,921 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| agg_keygen | 0 | 10,813,572 | 2,013,265,921 | 
| agg_keygen | 1 | 55,075,072 | 2,013,265,921 | 
| agg_keygen | 2 | 5,406,786 | 2,013,265,921 | 
| agg_keygen | 3 | 54,804,740 | 2,013,265,921 | 
| agg_keygen | 4 | 262,144 | 2,013,265,921 | 
| agg_keygen | 5 | 126,755,530 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/30e6c0fb095396c436fc1bfabb44cf64ebebe484

Max Segment Length: 1048476

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20447330623)
