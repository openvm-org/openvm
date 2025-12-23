| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  60.20 |  12.99 | 12.99 |
| fib_e2e |  5.31 |  0.54 |  0.54 |
| leaf |  26.48 |  2.21 |  2.21 |
| internal.0 |  20.38 |  4.64 |  4.64 |
| internal.1 |  5.56 |  3.13 |  3.13 |
| internal.2 |  2.47 |  2.47 |  2.47 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  373.14 |  5,224 |  454 |  291 |
| `main_cells_used     ` |  1,014,539.57 |  14,203,554 |  1,064,416 |  1,008,596 |
| `total_cells_used    ` |  9,653,739.29 |  135,152,350 |  9,708,170 |  9,647,206 |
| `execute_metered_time_ms` |  87 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  137.04 | -          |  137.04 |  137.04 |
| `execute_preflight_insns` |  857,157.79 |  12,000,209 |  873,000 |  651,209 |
| `execute_preflight_time_ms` |  39.50 |  553 |  41 |  31 |
| `execute_preflight_insn_mi/s` |  39.65 | -          |  40.28 |  37.55 |
| `trace_gen_time_ms   ` |  35.57 |  498 |  43 |  34 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  246.50 |  3,451 |  302 |  174 |
| `main_trace_commit_time_ms` |  10.07 |  141 |  11 |  10 |
| `generate_perm_trace_time_ms` |  14.43 |  202 |  50 |  5 |
| `perm_trace_commit_time_ms` |  14.34 |  200.81 |  14.94 |  13.61 |
| `quotient_poly_compute_time_ms` |  11.22 |  157.03 |  12.17 |  10.71 |
| `quotient_poly_commit_time_ms` |  4.33 |  60.58 |  4.50 |  4.21 |
| `pcs_opening_time_ms ` |  189.93 |  2,659 |  252 |  117 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,891.07 |  26,475 |  2,213 |  1,796 |
| `main_cells_used     ` |  8,902,705.43 |  124,637,876 |  10,121,356 |  8,734,702 |
| `total_cells_used    ` |  23,485,838.86 |  328,801,744 |  26,398,858 |  23,084,280 |
| `execute_preflight_insns` |  1,719,375.36 |  24,071,255 |  2,061,736 |  1,672,525 |
| `execute_preflight_time_ms` |  1,431.14 |  20,036 |  1,454 |  1,404 |
| `execute_preflight_insn_mi/s` |  2.89 | -          |  3.32 |  2.78 |
| `trace_gen_time_ms   ` |  68.07 |  953 |  78 |  63 |
| `memory_finalize_time_ms` |  7 |  98 |  8 |  6 |
| `stark_prove_excluding_trace_time_ms` |  390.36 |  5,465 |  698 |  296 |
| `main_trace_commit_time_ms` |  39.14 |  548 |  40 |  39 |
| `generate_perm_trace_time_ms` |  17.79 |  249 |  25 |  14 |
| `perm_trace_commit_time_ms` |  54.48 |  762.77 |  56.40 |  54.05 |
| `quotient_poly_compute_time_ms` |  41.67 |  583.44 |  44.63 |  40.24 |
| `quotient_poly_commit_time_ms` |  9.33 |  130.57 |  9.66 |  9.10 |
| `pcs_opening_time_ms ` |  225.71 |  3,160 |  532 |  126 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,076.20 |  20,381 |  4,638 |  3,003 |
| `main_cells_used     ` |  25,288,239.60 |  126,441,198 |  26,857,916 |  19,009,534 |
| `total_cells_used    ` |  53,225,136 |  266,125,680 |  56,402,102 |  40,517,272 |
| `execute_preflight_insns` |  5,408,077.80 |  27,040,389 |  5,794,390 |  3,863,189 |
| `execute_preflight_time_ms` |  2,497.80 |  12,489 |  2,637 |  2,012 |
| `execute_preflight_insn_mi/s` |  3.26 | -          |  3.28 |  3.24 |
| `trace_gen_time_ms   ` |  187 |  935 |  216 |  130 |
| `memory_finalize_time_ms` |  9.80 |  49 |  11 |  8 |
| `stark_prove_excluding_trace_time_ms` |  1,388.60 |  6,943 |  1,808 |  859 |
| `main_trace_commit_time_ms` |  329.20 |  1,646 |  426 |  192 |
| `generate_perm_trace_time_ms` |  51.60 |  258 |  66 |  35 |
| `perm_trace_commit_time_ms` |  253.87 |  1,269.34 |  331.19 |  162.63 |
| `quotient_poly_compute_time_ms` |  315.27 |  1,576.37 |  355.26 |  220.90 |
| `quotient_poly_commit_time_ms` |  135.64 |  678.19 |  208.12 |  71.60 |
| `pcs_opening_time_ms ` |  297.20 |  1,486 |  410 |  173 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,780.50 |  5,561 |  3,128 |  2,433 |
| `main_cells_used     ` |  18,226,184 |  36,452,368 |  21,143,358 |  15,309,010 |
| `total_cells_used    ` |  38,933,246 |  77,866,492 |  44,841,000 |  33,025,492 |
| `execute_preflight_insns` |  3,872,356 |  7,744,712 |  4,658,059 |  3,086,653 |
| `execute_preflight_time_ms` |  1,856.50 |  3,713 |  2,062 |  1,651 |
| `execute_preflight_insn_mi/s` |  3.84 | -          |  3.86 |  3.83 |
| `trace_gen_time_ms   ` |  117.50 |  235 |  135 |  100 |
| `memory_finalize_time_ms` |  8.50 |  17 |  9 |  8 |
| `stark_prove_excluding_trace_time_ms` |  804.50 |  1,609 |  929 |  680 |
| `main_trace_commit_time_ms` |  195 |  390 |  224 |  166 |
| `generate_perm_trace_time_ms` |  31.50 |  63 |  36 |  27 |
| `perm_trace_commit_time_ms` |  146.36 |  292.71 |  168.16 |  124.55 |
| `quotient_poly_compute_time_ms` |  191.36 |  382.73 |  212.35 |  170.38 |
| `quotient_poly_commit_time_ms` |  70.82 |  141.64 |  85.32 |  56.32 |
| `pcs_opening_time_ms ` |  165.50 |  331 |  198 |  133 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,474 |  2,474 |  2,474 |  2,474 |
| `main_cells_used     ` |  15,309,010 |  15,309,010 |  15,309,010 |  15,309,010 |
| `total_cells_used    ` |  33,025,492 |  33,025,492 |  33,025,492 |  33,025,492 |
| `execute_preflight_insns` |  3,085,611 |  3,085,611 |  3,085,611 |  3,085,611 |
| `execute_preflight_time_ms` |  1,639 |  1,639 |  1,639 |  1,639 |
| `execute_preflight_insn_mi/s` |  3.83 | -          |  3.83 |  3.83 |
| `trace_gen_time_ms   ` |  94 |  94 |  94 |  94 |
| `memory_finalize_time_ms` |  7 |  7 |  7 |  7 |
| `stark_prove_excluding_trace_time_ms` |  739 |  739 |  739 |  739 |
| `main_trace_commit_time_ms` |  163 |  163 |  163 |  163 |
| `generate_perm_trace_time_ms` |  35 |  35 |  35 |  35 |
| `perm_trace_commit_time_ms` |  123.49 |  123.49 |  123.49 |  123.49 |
| `quotient_poly_compute_time_ms` |  172.93 |  172.93 |  172.93 |  172.93 |
| `quotient_poly_commit_time_ms` |  56.02 |  56.02 |  56.02 |  56.02 |
| `pcs_opening_time_ms ` |  184 |  184 |  184 |  184 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,122.50 |  18,245 |  15,206 |  3,039 |
| `main_cells_used     ` |  79,635,386.50 |  159,270,773 |  158,351,393 |  919,380 |
| `total_cells_used    ` |  197,883,970.50 |  395,767,941 |  386,262,375 |  9,505,566 |
| `execute_metered_time_ms` |  0 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.04 | -          |  0.04 |  0.04 |
| `execute_preflight_insns` |  1,329,254.50 |  2,658,509 |  2,658,508 |  1 |
| `execute_preflight_time_ms` |  621 |  1,242 |  1,242 |  0 |
| `execute_preflight_insn_mi/s` |  2.42 | -          |  3.83 |  1 |
| `trace_gen_time_ms   ` |  106.50 |  213 |  113 |  100 |
| `memory_finalize_time_ms` |  3.50 |  7 |  7 |  0 |
| `stark_prove_excluding_trace_time_ms` |  6,150 |  12,300 |  11,503 |  797 |
| `main_trace_commit_time_ms` |  1,171 |  2,342 |  2,241 |  101 |
| `generate_perm_trace_time_ms` |  427.50 |  855 |  844 |  11 |
| `perm_trace_commit_time_ms` |  970 |  1,940 |  1,841 |  99 |
| `quotient_poly_compute_time_ms` |  1,562 |  3,124 |  3,084 |  40 |
| `quotient_poly_commit_time_ms` |  1,065 |  2,130 |  2,021 |  109 |
| `pcs_opening_time_ms ` |  944.50 |  1,889 |  1,469 |  420 |



<details>
<summary>Detailed Metrics</summary>

|  | dummy_proof_and_keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 40,602 | 5,323 | 2,478 | 

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
| agg_keygen | 113 | 15,206 | 386,262,375 | 515,190,250 | 113 | 11,503 | 0 |  |  | 3,084 | 2,021 | 3,039 | 1,841 | 1,469 |  | 7 | 2,241 | 158,351,393 | 844 |  | 1,242 | 2,658,508 | 3.83 | 0 | 1 | 0.04 | 0 | 
| fib_e2e |  |  |  |  |  |  |  |  |  |  |  | 338 |  |  |  |  |  |  |  | 1 |  |  |  | 87 | 12,000,209 | 137.04 | 0 | 
| internal.0 |  |  |  |  |  |  |  |  | 3,007 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| internal.1 |  |  |  |  |  |  |  |  | 2,436 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| internal.2 |  |  |  |  |  |  |  |  | 2,477 |  |  |  |  |  | 3 |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| leaf |  |  |  |  |  |  |  | 2,215 |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |  |  |  |  | 

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
| internal.0 | 0 | 216 | 4,638 | 56,402,102 | 920,340,962 | 216 | 1,808 | 0 | 355.26 | 208.12 | 6 | 331.19 | 410 | 398 | 409 | 11 | 426 | 26,857,916 | 66 | 2,608 | 5,794,271 | 3.28 | 67 | 572 | 18 | 409 | 
| internal.0 | 1 | 188 | 4,267 | 56,402,102 | 920,340,962 | 188 | 1,440 | 0 | 332.73 | 132.61 | 6 | 256.69 | 318 | 310 | 318 | 10 | 343 | 26,857,916 | 52 | 2,637 | 5,794,290 | 3.24 | 58 | 468 | 0 | 318 | 
| internal.0 | 2 | 212 | 4,187 | 56,402,102 | 920,340,962 | 212 | 1,352 | 0 | 335.08 | 132.86 | 6 | 260.04 | 223 | 314 | 222 | 10 | 344 | 26,857,916 | 53 | 2,621 | 5,794,249 | 3.27 | 59 | 470 | 0 | 222 | 
| internal.0 | 3 | 189 | 4,286 | 56,402,102 | 920,340,962 | 189 | 1,484 | 0 | 332.40 | 132.100 | 6 | 258.79 | 362 | 311 | 361 | 10 | 341 | 26,857,916 | 52 | 2,611 | 5,794,390 | 3.25 | 59 | 468 | 0 | 361 | 
| internal.0 | 4 | 130 | 3,003 | 40,517,272 | 579,447,266 | 130 | 859 | 0 | 220.90 | 71.60 | 5 | 162.63 | 173 | 199 | 172 | 8 | 192 | 19,009,534 | 35 | 2,012 | 3,863,189 | 3.26 | 37 | 294 | 0 | 172 | 
| internal.1 | 5 | 135 | 3,128 | 44,841,000 | 607,758,818 | 135 | 929 | 0 | 212.35 | 85.32 | 6 | 168.16 | 198 | 205 | 198 | 9 | 224 | 21,143,358 | 36 | 2,062 | 4,658,059 | 3.86 | 41 | 299 | 0 | 198 | 
| internal.1 | 6 | 100 | 2,433 | 33,025,492 | 472,754,658 | 100 | 680 | 0 | 170.38 | 56.32 | 5 | 124.55 | 133 | 152 | 133 | 8 | 166 | 15,309,010 | 27 | 1,651 | 3,086,653 | 3.83 | 31 | 228 | 0 | 133 | 
| internal.2 | 7 | 94 | 2,474 | 33,025,492 | 472,754,658 | 94 | 739 | 0 | 172.93 | 56.02 | 5 | 123.49 | 184 | 160 | 184 | 7 | 163 | 15,309,010 | 35 | 1,639 | 3,085,611 | 3.83 | 31 | 230 | 0 | 184 | 
| leaf | 0 | 76 | 1,904 | 26,398,858 | 342,564,330 | 76 | 373 | 0 | 44.63 | 9.65 | 6 | 56.40 | 203 | 74 | 202 | 7 | 40 | 10,121,356 | 17 | 1,454 | 2,061,736 | 3.32 | 14 | 54 | 0 | 202 | 
| leaf | 1 | 78 | 1,858 | 23,084,280 | 324,015,594 | 78 | 350 | 0 | 42.74 | 9.49 | 5 | 54.20 | 187 | 71 | 187 | 7 | 39 | 8,734,702 | 16 | 1,428 | 1,672,676 | 2.82 | 14 | 52 | 0 | 187 | 
| leaf | 10 | 64 | 1,865 | 23,084,280 | 324,015,594 | 64 | 365 | 0 | 40.84 | 9.19 | 5 | 54.23 | 205 | 69 | 205 | 7 | 39 | 8,734,702 | 14 | 1,435 | 1,672,665 | 2.86 | 14 | 50 | 0 | 205 | 
| leaf | 11 | 66 | 1,891 | 23,084,280 | 324,015,594 | 66 | 388 | 0 | 41.42 | 9.27 | 5 | 54.27 | 225 | 72 | 224 | 7 | 39 | 8,734,702 | 17 | 1,436 | 1,672,661 | 2.80 | 13 | 51 | 0 | 224 | 
| leaf | 12 | 67 | 1,874 | 23,084,280 | 324,015,594 | 66 | 393 | 0 | 42.86 | 9.66 | 5 | 54.14 | 228 | 72 | 228 | 7 | 39 | 8,734,702 | 17 | 1,413 | 1,672,616 | 2.87 | 14 | 53 | 0 | 228 | 
| leaf | 13 | 69 | 2,213 | 25,391,526 | 342,564,330 | 68 | 698 | 0 | 42.59 | 9.54 | 5 | 56.34 | 532 | 72 | 532 | 8 | 40 | 9,700,096 | 15 | 1,445 | 1,937,839 | 3.17 | 14 | 52 | 0 | 532 | 
| leaf | 2 | 63 | 1,852 | 23,084,280 | 324,015,594 | 63 | 359 | 0 | 40.81 | 9.13 | 5 | 54.13 | 195 | 73 | 195 | 7 | 39 | 8,734,702 | 18 | 1,428 | 1,672,656 | 2.79 | 13 | 50 | 0 | 195 | 
| leaf | 3 | 64 | 1,796 | 23,084,280 | 324,015,594 | 64 | 296 | 0 | 40.35 | 9.10 | 5 | 54.22 | 126 | 80 | 126 | 7 | 39 | 8,734,702 | 25 | 1,435 | 1,672,564 | 2.80 | 13 | 49 | 0 | 126 | 
| leaf | 4 | 63 | 1,798 | 23,084,280 | 324,015,594 | 63 | 302 | 0 | 41.19 | 9.22 | 6 | 54.14 | 140 | 72 | 139 | 7 | 39 | 8,734,702 | 17 | 1,431 | 1,672,577 | 2.87 | 14 | 50 | 0 | 139 | 
| leaf | 5 | 66 | 1,861 | 23,084,280 | 324,015,594 | 66 | 358 | 0 | 42.16 | 9.31 | 5 | 54.13 | 189 | 77 | 189 | 7 | 39 | 8,734,702 | 22 | 1,436 | 1,672,760 | 2.78 | 14 | 51 | 0 | 189 | 
| leaf | 6 | 77 | 1,800 | 23,084,280 | 324,015,594 | 77 | 318 | 0 | 42.40 | 9.56 | 5 | 54.05 | 155 | 71 | 155 | 7 | 39 | 8,734,702 | 16 | 1,404 | 1,672,676 | 2.91 | 14 | 52 | 0 | 155 | 
| leaf | 7 | 65 | 1,940 | 23,084,280 | 324,015,594 | 65 | 433 | 0 | 40.24 | 9.10 | 5 | 54.19 | 268 | 75 | 268 | 7 | 39 | 8,734,702 | 20 | 1,441 | 1,672,688 | 2.78 | 13 | 49 | 0 | 268 | 
| leaf | 8 | 70 | 1,860 | 23,084,280 | 324,015,594 | 70 | 374 | 0 | 40.79 | 9.15 | 6 | 54.13 | 209 | 75 | 208 | 6 | 39 | 8,734,702 | 20 | 1,414 | 1,672,525 | 2.88 | 14 | 50 | 0 | 208 | 
| leaf | 9 | 65 | 1,963 | 23,084,280 | 324,015,594 | 65 | 458 | 0 | 40.41 | 9.20 | 6 | 54.20 | 298 | 70 | 298 | 7 | 39 | 8,734,702 | 15 | 1,436 | 1,672,616 | 2.81 | 13 | 50 | 0 | 298 | 

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
| agg_keygen | 0 | 100 | 3,039 | 9,505,566 | 7,747,601 | 100 | 797 |  | 40 | 109 |  | 99 | 420 |  |  | 96 | 0 | 101 | 919,380 | 11 | 0 | 1 | 1 |  |  |  |  | 
| fib_e2e | 0 | 38 | 454 | 9,677,708 | 84,395,212 | 38 | 198 | 0 | 12.17 | 4.50 | 6 | 14.94 | 147 | 21 | 147 |  | 0 | 11 | 1,035,986 | 6 | 40 | 873,000 | 40.21 | 9 | 17 | 0 | 147 | 
| fib_e2e | 1 | 34 | 394 | 9,647,206 | 84,342,330 | 34 | 277 | 2 | 11.18 | 4.21 | 5 | 14.35 | 221 | 29 | 221 |  | 0 | 10 | 1,008,596 | 14 | 40 | 873,000 | 39.62 | 6 | 15 | 0 | 221 | 
| fib_e2e | 10 | 35 | 342 | 9,647,206 | 84,342,330 | 35 | 225 | 2 | 11.17 | 4.35 | 5 | 14.37 | 177 | 21 | 177 |  | 0 | 10 | 1,008,596 | 6 | 40 | 873,000 | 40.06 | 6 | 15 | 0 | 177 | 
| fib_e2e | 11 | 35 | 296 | 9,647,206 | 84,342,330 | 35 | 180 | 2 | 11.17 | 4.35 | 5 | 14.36 | 120 | 32 | 120 |  | 0 | 10 | 1,008,596 | 18 | 40 | 873,000 | 40.05 | 6 | 15 | 0 | 120 | 
| fib_e2e | 12 | 35 | 415 | 9,647,206 | 84,342,330 | 35 | 298 | 2 | 11.17 | 4.32 | 5 | 14.34 | 207 | 64 | 207 |  | 0 | 10 | 1,008,596 | 50 | 40 | 873,000 | 40.28 | 6 | 15 | 0 | 207 | 
| fib_e2e | 13 | 43 | 338 | 9,708,170 | 74,305,770 | 43 | 223 | 1 | 10.71 | 4.23 | 5 | 13.61 | 172 | 23 | 172 |  | 0 | 10 | 1,064,416 | 9 | 31 | 651,209 | 39.44 | 7 | 15 | 0 | 172 | 
| fib_e2e | 2 | 34 | 386 | 9,647,206 | 84,342,330 | 34 | 270 | 2 | 11.16 | 4.35 | 5 | 14.35 | 207 | 35 | 207 |  | 0 | 10 | 1,008,596 | 20 | 41 | 873,000 | 38.40 | 6 | 15 | 0 | 207 | 
| fib_e2e | 3 | 35 | 419 | 9,647,206 | 84,342,330 | 35 | 302 | 2 | 11.22 | 4.33 | 5 | 14.33 | 251 | 23 | 251 |  | 0 | 10 | 1,008,596 | 9 | 40 | 873,000 | 40.26 | 6 | 15 | 0 | 251 | 
| fib_e2e | 4 | 35 | 352 | 9,647,206 | 84,342,330 | 35 | 235 | 2 | 11.22 | 4.32 | 5 | 14.37 | 175 | 33 | 175 |  | 0 | 10 | 1,008,596 | 18 | 40 | 873,000 | 40.09 | 6 | 15 | 0 | 175 | 
| fib_e2e | 5 | 34 | 371 | 9,647,206 | 84,342,330 | 34 | 254 | 2 | 11.16 | 4.31 | 5 | 14.35 | 199 | 28 | 199 |  | 0 | 10 | 1,008,596 | 13 | 41 | 873,000 | 37.55 | 6 | 15 | 0 | 199 | 
| fib_e2e | 6 | 35 | 412 | 9,647,206 | 84,342,330 | 35 | 295 | 2 | 11.18 | 4.32 | 5 | 14.36 | 242 | 26 | 242 |  | 0 | 10 | 1,008,596 | 11 | 40 | 873,000 | 39.46 | 6 | 15 | 0 | 242 | 
| fib_e2e | 7 | 35 | 339 | 9,647,206 | 84,342,330 | 35 | 222 | 2 | 11.18 | 4.35 | 5 | 14.39 | 172 | 23 | 172 |  | 0 | 10 | 1,008,596 | 8 | 40 | 873,000 | 39.87 | 6 | 15 | 0 | 172 | 
| fib_e2e | 8 | 35 | 291 | 9,647,206 | 84,342,330 | 35 | 174 | 2 | 11.16 | 4.32 | 5 | 14.34 | 117 | 29 | 117 |  | 0 | 10 | 1,008,596 | 15 | 40 | 873,000 | 40.07 | 6 | 15 | 0 | 117 | 
| fib_e2e | 9 | 35 | 415 | 9,647,206 | 84,342,330 | 35 | 298 | 2 | 11.18 | 4.32 | 5 | 14.34 | 252 | 19 | 251 |  | 0 | 10 | 1,008,596 | 5 | 40 | 873,000 | 39.78 | 6 | 15 | 0 | 251 | 

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


Commit: https://github.com/openvm-org/openvm/commit/2c0dad0e08fb80bde004fda83b57fbc16d89caea

Max Segment Length: 1048476

Instance Type: g6e.4xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20448996466)
