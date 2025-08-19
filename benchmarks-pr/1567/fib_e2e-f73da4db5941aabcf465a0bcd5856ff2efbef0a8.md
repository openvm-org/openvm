| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  254.97 |  199.82 |
| fib_e2e |  20.93 |  3.22 |
| leaf |  24.70 |  4.20 |
| internal.0 |  28.47 |  11.53 |
| internal.1 |  8.13 |  8.13 |
| root |  38.70 |  38.70 |
| halo2_outer |  89.78 |  89.78 |
| halo2_wrapper |  44.20 |  44.20 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,990 |  20,930 |  3,222 |  2,297 |
| `main_cells_used     ` |  58,704,834.14 |  410,933,839 |  59,842,060 |  51,906,075 |
| `total_cells_used    ` |  144,147,948.14 |  1,009,035,637 |  146,796,066 |  128,298,001 |
| `execute_metered_time_ms` |  61 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  195.54 | -          |  195.54 |  195.54 |
| `execute_preflight_insns` |  1,714,315.71 |  12,000,210 |  1,748,000 |  1,512,210 |
| `execute_preflight_time_ms` |  64.43 |  451 |  67 |  52 |
| `execute_preflight_insn_mi/s` |  37.27 | -          |  37.33 |  37.21 |
| `trace_gen_time_ms   ` |  225.14 |  1,576 |  229 |  206 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  2,485.57 |  17,399 |  2,679 |  1,829 |
| `main_trace_commit_time_ms` |  468.86 |  3,282 |  532 |  338 |
| `generate_perm_trace_time_ms` |  185 |  1,295 |  214 |  135 |
| `perm_trace_commit_time_ms` |  562.71 |  3,939 |  597 |  381 |
| `quotient_poly_compute_time_ms` |  237 |  1,659 |  256 |  180 |
| `quotient_poly_commit_time_ms` |  217 |  1,519 |  269 |  180 |
| `pcs_opening_time_ms ` |  810.29 |  5,672 |  853 |  612 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,529.29 |  24,705 |  4,195 |  3,285 |
| `main_cells_used     ` |  63,012,443.29 |  441,087,103 |  72,248,817 |  60,333,057 |
| `total_cells_used    ` |  147,165,628.14 |  1,030,159,397 |  170,628,411 |  140,165,303 |
| `execute_preflight_insns` |  1,066,060.71 |  7,462,425 |  1,260,083 |  1,007,151 |
| `execute_preflight_time_ms` |  473.57 |  3,315 |  506 |  462 |
| `execute_preflight_insn_mi/s` |  3.42 | -          |  3.88 |  3.24 |
| `trace_gen_time_ms   ` |  156.57 |  1,096 |  184 |  148 |
| `memory_finalize_time_ms` |  8.57 |  60 |  10 |  8 |
| `stark_prove_excluding_trace_time_ms` |  1,856.43 |  12,995 |  2,483 |  1,637 |
| `main_trace_commit_time_ms` |  346.14 |  2,423 |  479 |  303 |
| `generate_perm_trace_time_ms` |  135.57 |  949 |  177 |  115 |
| `perm_trace_commit_time_ms` |  432.86 |  3,030 |  578 |  370 |
| `quotient_poly_compute_time_ms` |  203.71 |  1,426 |  267 |  178 |
| `quotient_poly_commit_time_ms` |  171 |  1,197 |  244 |  150 |
| `pcs_opening_time_ms ` |  560.86 |  3,926 |  732 |  501 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,489.67 |  28,469 |  11,528 |  5,732 |
| `main_cells_used     ` |  162,337,397.67 |  487,012,193 |  207,107,703 |  74,650,857 |
| `total_cells_used    ` |  292,156,451.67 |  876,469,355 |  371,618,977 |  136,140,539 |
| `execute_preflight_insns` |  2,660,584 |  7,981,752 |  3,422,909 |  1,151,426 |
| `execute_preflight_time_ms` |  813.33 |  2,440 |  1,000 |  448 |
| `execute_preflight_insn_mi/s` |  4.03 | -          |  4.09 |  3.95 |
| `trace_gen_time_ms   ` |  390.67 |  1,172 |  496 |  180 |
| `memory_finalize_time_ms` |  10.67 |  32 |  12 |  9 |
| `stark_prove_excluding_trace_time_ms` |  7,234.67 |  21,704 |  8,978 |  4,055 |
| `main_trace_commit_time_ms` |  1,648.33 |  4,945 |  2,105 |  821 |
| `generate_perm_trace_time_ms` |  322 |  966 |  402 |  172 |
| `perm_trace_commit_time_ms` |  1,248 |  3,744 |  1,534 |  678 |
| `quotient_poly_compute_time_ms` |  1,013.33 |  3,040 |  1,276 |  532 |
| `quotient_poly_commit_time_ms` |  1,055.33 |  3,166 |  1,352 |  629 |
| `pcs_opening_time_ms ` |  1,941.67 |  5,825 |  2,306 |  1,215 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,135 |  8,135 |  8,135 |  8,135 |
| `main_cells_used     ` |  116,589,677 |  116,589,677 |  116,589,677 |  116,589,677 |
| `total_cells_used    ` |  207,812,715 |  207,812,715 |  207,812,715 |  207,812,715 |
| `execute_preflight_insns` |  2,330,585 |  2,330,585 |  2,330,585 |  2,330,585 |
| `execute_preflight_time_ms` |  598 |  598 |  598 |  598 |
| `execute_preflight_insn_mi/s` |  5.29 | -          |  5.29 |  5.29 |
| `trace_gen_time_ms   ` |  309 |  309 |  309 |  309 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `stark_prove_excluding_trace_time_ms` |  6,177 |  6,177 |  6,177 |  6,177 |
| `main_trace_commit_time_ms` |  1,351 |  1,351 |  1,351 |  1,351 |
| `generate_perm_trace_time_ms` |  267 |  267 |  267 |  267 |
| `perm_trace_commit_time_ms` |  1,072 |  1,072 |  1,072 |  1,072 |
| `quotient_poly_compute_time_ms` |  815 |  815 |  815 |  815 |
| `quotient_poly_commit_time_ms` |  906 |  906 |  906 |  906 |
| `pcs_opening_time_ms ` |  1,762 |  1,762 |  1,762 |  1,762 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,698 |  38,698 |  38,698 |  38,698 |
| `main_cells_used     ` |  42,136,295 |  42,136,295 |  42,136,295 |  42,136,295 |
| `total_cells_used    ` |  65,007,605 |  65,007,605 |  65,007,605 |  65,007,605 |
| `execute_preflight_insns` |  779,826 |  779,826 |  779,826 |  779,826 |
| `execute_preflight_time_ms` |  171 |  171 |  171 |  171 |
| `execute_preflight_insn_mi/s` |  5.11 | -          |  5.11 |  5.11 |
| `trace_gen_time_ms   ` |  119 |  119 |  119 |  119 |
| `memory_finalize_time_ms` |  12 |  12 |  12 |  12 |
| `stark_prove_excluding_trace_time_ms` |  38,408 |  38,408 |  38,408 |  38,408 |
| `main_trace_commit_time_ms` |  12,328 |  12,328 |  12,328 |  12,328 |
| `generate_perm_trace_time_ms` |  90 |  90 |  90 |  90 |
| `perm_trace_commit_time_ms` |  7,586 |  7,586 |  7,586 |  7,586 |
| `quotient_poly_compute_time_ms` |  722 |  722 |  722 |  722 |
| `quotient_poly_commit_time_ms` |  13,765 |  13,765 |  13,765 |  13,765 |
| `pcs_opening_time_ms ` |  3,893 |  3,893 |  3,893 |  3,893 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  89,782 |  89,782 |  89,782 |  89,782 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,195 |  44,195 |  44,195 |  44,195 |

| agg_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,949.50 |  7,899 |  6,236 |  1,663 |
| `main_cells_used     ` |  46,038,298 |  92,076,596 |  91,157,216 |  919,380 |
| `total_cells_used    ` |  115,856,348 |  231,712,696 |  222,207,130 |  9,505,566 |
| `execute_metered_time_ms` |  0 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.04 | -          |  0.04 |  0.04 |
| `execute_preflight_insns` |  811,161 |  1,622,322 |  1,622,321 |  1 |
| `execute_preflight_time_ms` |  154 |  308 |  306 |  2 |
| `execute_preflight_insn_mi/s` |  9,223,372,036,854,775,807 | -          |  9,223,372,036,854,775,807 |  5.23 |
| `trace_gen_time_ms   ` |  70.50 |  141 |  115 |  26 |
| `memory_finalize_time_ms` |  5 |  10 |  8 |  2 |
| `stark_prove_excluding_trace_time_ms` |  2,345.50 |  4,691 |  4,192 |  499 |
| `main_trace_commit_time_ms` |  457 |  914 |  849 |  65 |
| `generate_perm_trace_time_ms` |  95.50 |  191 |  179 |  12 |
| `perm_trace_commit_time_ms` |  367.50 |  735 |  676 |  59 |
| `quotient_poly_compute_time_ms` |  277 |  554 |  529 |  25 |
| `quotient_poly_commit_time_ms` |  402.50 |  805 |  732 |  73 |
| `pcs_opening_time_ms ` |  741.50 |  1,483 |  1,224 |  259 |

| halo2_keygen |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,735 |  38,735 |  38,735 |  38,735 |
| `main_cells_used     ` |  41,528,188 |  41,528,188 |  41,528,188 |  41,528,188 |
| `total_cells_used    ` |  64,211,222 |  64,211,222 |  64,211,222 |  64,211,222 |
| `execute_preflight_insns` |  772,290 |  772,290 |  772,290 |  772,290 |
| `execute_preflight_time_ms` |  164 |  164 |  164 |  164 |
| `execute_preflight_insn_mi/s` |  5.19 | -          |  5.19 |  5.19 |
| `trace_gen_time_ms   ` |  121 |  121 |  121 |  121 |
| `memory_finalize_time_ms` |  9 |  9 |  9 |  9 |
| `stark_prove_excluding_trace_time_ms` |  38,450 |  38,450 |  38,450 |  38,450 |
| `main_trace_commit_time_ms` |  12,299 |  12,299 |  12,299 |  12,299 |
| `generate_perm_trace_time_ms` |  87 |  87 |  87 |  87 |
| `perm_trace_commit_time_ms` |  7,589 |  7,589 |  7,589 |  7,589 |
| `quotient_poly_compute_time_ms` |  689 |  689 |  689 |  689 |
| `quotient_poly_commit_time_ms` |  13,728 |  13,728 |  13,728 |  13,728 |
| `pcs_opening_time_ms ` |  4,015 |  4,015 |  4,015 |  4,015 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_cells_used | keygen_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | dummy_proof_and_keygen_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 119 | 65,007,605 | 119 | 0 | 89,796 | 44,195 | 22 | 8 | 42,136,295 | 174,329 | 307 | 779,826 | 5.06 | 16,921 | 21,032 | 39,749 | 

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

| group | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | num_children | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | halo2_total_cells | halo2_keygen_time_ms | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 115 | 6,236 | 222,207,130 | 270,872,042 | 115 | 4,192 | 0 |  |  | 529 | 732 | 1,663 | 676 | 1,224 |  | 16 | 8 | 849 | 91,157,216 |  |  | 179 |  | 306 | 1,622,321 | 5.23 | 0 | 1 | 0.04 | 17 | 
| fib_e2e |  |  |  |  |  |  |  |  |  |  |  | 2,297 |  |  |  | 5 |  |  |  |  |  |  | 1 |  |  |  | 61 | 12,000,210 | 195.54 | 35 | 
| halo2_keygen | 121 | 38,735 | 64,211,222 | 80,435,354 | 121 | 38,450 | 0 |  |  | 689 | 13,728 |  | 7,589 | 4,015 |  |  | 9 | 12,299 | 41,528,188 | 5,447,564 | 19,228 | 87 |  | 164 | 772,290 | 5.19 |  |  |  |  | 
| halo2_outer |  | 89,782 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 65,627,358 |  |  |  |  |  |  |  |  |  |  |  | 
| halo2_wrapper |  | 44,195 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  |  |  |  |  |  |  | 5,734 |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| internal.1 |  |  |  |  |  |  |  |  | 8,137 |  |  |  |  |  | 3 |  |  |  |  |  |  |  | 2 |  |  |  |  |  |  |  | 
| leaf |  |  |  |  |  |  |  | 3,759 |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 

| group | air_name | rows | quotient_deg | prep_cols | perm_cols | main_cols | interactions | constraints | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | AccessAdapterAir<16> |  | 2 |  |  |  | 5 | 12 |  | 
| agg_keygen | AccessAdapterAir<2> | 524,288 | 8 |  | 16 | 11 | 5 | 12 | 14,155,776 | 
| agg_keygen | AccessAdapterAir<32> |  | 2 |  |  |  | 5 | 12 |  | 
| agg_keygen | AccessAdapterAir<4> | 262,144 | 8 |  | 16 | 13 | 5 | 12 | 7,602,176 | 
| agg_keygen | AccessAdapterAir<8> | 8,192 | 8 |  | 16 | 17 | 5 | 12 | 270,336 | 
| agg_keygen | BitwiseOperationLookupAir<8> |  | 2 |  |  |  | 2 | 4 |  | 
| agg_keygen | FriReducedOpeningAir | 524,288 | 8 |  | 84 | 27 | 39 | 71 | 58,195,968 | 
| agg_keygen | JalRangeCheckAir | 65,536 | 8 |  | 28 | 12 | 9 | 14 | 2,621,440 | 
| agg_keygen | MemoryMerkleAir<8> |  | 2 |  |  |  | 4 | 39 |  | 
| agg_keygen | NativePoseidon2Air<BabyBearParameters>, 1> | 65,536 | 8 |  | 312 | 398 | 136 | 572 | 46,530,560 | 
| agg_keygen | PersistentBoundaryAir<8> |  | 2 |  |  |  | 3 | 7 |  | 
| agg_keygen | PhantomAir | 32,768 | 4 |  | 12 | 6 | 3 | 5 | 589,824 | 
| agg_keygen | Poseidon2PeripheryAir<BabyBearParameters>, 1> |  | 2 |  |  |  | 1 | 286 |  | 
| agg_keygen | ProgramAir | 131,072 | 1 |  | 8 | 10 | 1 | 4 | 2,359,296 | 
| agg_keygen | RangeTupleCheckerAir<2> |  | 1 |  |  |  | 1 | 4 |  | 
| agg_keygen | Rv32HintStoreAir |  | 2 |  |  |  | 18 | 28 |  | 
| agg_keygen | VariableRangeCheckerAir | 262,144 | 1 | 2 | 8 | 1 | 1 | 4 | 2,359,296 | 
| agg_keygen | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1,048,576 | 8 |  | 36 | 29 | 15 | 27 | 68,157,440 | 
| agg_keygen | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 262,144 | 8 |  | 28 | 23 | 11 | 25 | 13,369,344 | 
| agg_keygen | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 64 | 8 |  | 28 | 27 | 11 | 30 | 3,520 | 
| agg_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 524,288 | 8 |  | 40 | 21 | 15 | 20 | 31,981,568 | 
| agg_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 131,072 | 8 |  | 40 | 27 | 15 | 20 | 8,781,824 | 
| agg_keygen | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 131,072 | 8 |  | 36 | 38 | 15 | 27 | 9,699,328 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | 2 |  |  |  | 20 | 37 |  | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | 2 |  |  |  | 18 | 40 |  | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | 2 |  |  |  | 24 | 91 |  | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | 2 |  |  |  | 11 | 20 |  | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | 2 |  |  |  | 13 | 35 |  | 
| agg_keygen | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | 2 |  |  |  | 10 | 18 |  | 
| agg_keygen | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | 2 |  |  |  | 16 | 20 |  | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | 2 |  |  |  | 18 | 33 |  | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | 2 |  |  |  | 17 | 40 |  | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | 2 |  |  |  | 25 | 84 |  | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | 2 |  |  |  | 24 | 31 |  | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | 2 |  |  |  | 19 | 19 |  | 
| agg_keygen | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | 2 |  |  |  | 12 | 14 |  | 
| agg_keygen | VmConnectorAir | 2 | 8 | 1 | 16 | 5 | 5 | 11 | 42 | 
| agg_keygen | VolatileBoundaryAir | 131,072 | 8 |  | 20 | 12 | 7 | 19 | 4,194,304 | 
| halo2_keygen | AccessAdapterAir<2> | 262,144 |  |  | 8 | 11 |  |  | 4,980,736 | 
| halo2_keygen | AccessAdapterAir<4> | 131,072 |  |  | 8 | 13 |  |  | 2,752,512 | 
| halo2_keygen | AccessAdapterAir<8> | 4,096 |  |  | 8 | 17 |  |  | 102,400 | 
| halo2_keygen | FriReducedOpeningAir | 131,072 |  |  | 24 | 27 |  |  | 6,684,672 | 
| halo2_keygen | JalRangeCheckAir | 32,768 |  |  | 12 | 12 |  |  | 786,432 | 
| halo2_keygen | NativePoseidon2Air<BabyBearParameters>, 1> | 32,768 |  |  | 84 | 398 |  |  | 15,794,176 | 
| halo2_keygen | PhantomAir | 8,192 |  |  | 8 | 6 |  |  | 114,688 | 
| halo2_keygen | ProgramAir | 131,072 |  |  | 8 | 10 |  |  | 2,359,296 | 
| halo2_keygen | VariableRangeCheckerAir | 262,144 |  | 2 | 8 | 1 |  |  | 2,359,296 | 
| halo2_keygen | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 524,288 |  |  | 12 | 29 |  |  | 21,495,808 | 
| halo2_keygen | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 131,072 |  |  | 12 | 23 |  |  | 4,587,520 | 
| halo2_keygen | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 64 |  |  | 12 | 22 |  |  | 2,176 | 
| halo2_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 262,144 |  |  | 16 | 21 |  |  | 9,699,328 | 
| halo2_keygen | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 65,536 |  |  | 16 | 27 |  |  | 2,818,048 | 
| halo2_keygen | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 65,536 |  |  | 12 | 38 |  |  | 3,276,800 | 
| halo2_keygen | VmConnectorAir | 2 |  | 1 | 8 | 5 |  |  | 26 | 
| halo2_keygen | VolatileBoundaryAir | 131,072 |  |  | 8 | 12 |  |  | 2,621,440 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<4> | 1 | 524,288 |  | 12 | 13 | 13,107,200 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | AccessAdapterAir<8> | 1 | 16,384 |  | 12 | 17 | 475,136 | 
| internal.0 | AccessAdapterAir<8> | 2 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 1 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 2 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 1 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 2 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | PhantomAir | 2 | 16,384 |  | 8 | 6 | 229,376 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 21 | 47,185,920 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 1 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 2 | 131,072 |  | 12 | 12 | 3,145,728 | 
| internal.1 | AccessAdapterAir<2> | 3 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 3 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 3 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | FriReducedOpeningAir | 3 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.1 | JalRangeCheckAir | 3 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.1 | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.1 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.1 | VmConnectorAir | 3 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VolatileBoundaryAir | 3 | 262,144 |  | 12 | 12 | 6,291,456 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 1 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 2 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 3 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 4 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 5 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 1 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 2 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 3 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 4 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 5 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 2 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 4 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 5 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 6 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | FriReducedOpeningAir | 1 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 2 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 3 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 4 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 5 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 6 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 2 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 3 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 4 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 5 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 6 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 2 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 3 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 4 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 5 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 6 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 524,288 |  | 36 | 29 | 34,078,720 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 1 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 2 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 3 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 4 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 5 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 6 | 131,072 |  | 20 | 12 | 4,194,304 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 8 | 13 | 2,752,512 | 
| root | AccessAdapterAir<8> | 0 | 4,096 |  | 8 | 17 | 102,400 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| root | JalRangeCheckAir | 0 | 32,768 |  | 12 | 12 | 786,432 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 84 | 398 | 15,794,176 | 
| root | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 524,288 |  | 12 | 29 | 21,495,808 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 12 | 23 | 4,587,520 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 12 | 22 | 2,176 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 16 | 21 | 9,699,328 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 16 | 27 | 2,818,048 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 12 | 38 | 3,276,800 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 5 | 26 | 
| root | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 12 | 2,621,440 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | AccessAdapterAir<16> | 0 | 1 |  | 16 | 25 | 41 | 
| agg_keygen | AccessAdapterAir<2> | 0 | 1 |  | 16 | 11 | 27 | 
| agg_keygen | AccessAdapterAir<32> | 0 | 1 |  | 16 | 41 | 57 | 
| agg_keygen | AccessAdapterAir<4> | 0 | 1 |  | 16 | 13 | 29 | 
| agg_keygen | AccessAdapterAir<8> | 0 | 1 |  | 16 | 17 | 33 | 
| agg_keygen | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| agg_keygen | MemoryMerkleAir<8> | 0 | 64 |  | 16 | 32 | 3,072 | 
| agg_keygen | PersistentBoundaryAir<8> | 0 | 1 |  | 12 | 20 | 32 | 
| agg_keygen | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| agg_keygen | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32 |  | 8 | 300 | 9,856 | 
| agg_keygen | ProgramAir | 0 | 1 |  | 8 | 10 | 18 | 
| agg_keygen | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| agg_keygen | Rv32HintStoreAir | 0 | 1 |  | 44 | 32 | 76 | 
| agg_keygen | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1 |  | 52 | 36 | 88 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 |  | 40 | 37 | 77 | 
| agg_keygen | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1 |  | 52 | 53 | 105 | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 1 |  | 28 | 26 | 54 | 
| agg_keygen | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 1 |  | 32 | 32 | 64 | 
| agg_keygen | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1 |  | 28 | 18 | 46 | 
| agg_keygen | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 1 |  | 36 | 28 | 64 | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1 |  | 52 | 36 | 88 | 
| agg_keygen | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1 |  | 52 | 41 | 93 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 1 |  | 72 | 59 | 131 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 1 |  | 72 | 39 | 111 | 
| agg_keygen | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 1 |  | 52 | 31 | 83 | 
| agg_keygen | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1 |  | 28 | 20 | 48 | 
| agg_keygen | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 2 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 3 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 4 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 5 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 6 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 2 |  | 32 | 32 | 128 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 64 |  | 52 | 41 | 5,952 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 4 |  | 28 | 20 | 192 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 

| group | cell_tracker_span | simple_advice_cells | lookup_advice_cells | fixed_cells |
| --- | --- | --- | --- | --- |
| halo2_keygen | VerifierProgram | 509,456 | 164,237 | 166,961 | 
| halo2_keygen | VerifierProgram;CheckTraceHeightConstraints | 5,316 | 1,125 | 1,942 | 
| halo2_keygen | VerifierProgram;PoseidonCell | 29,400 |  | 8,700 | 
| halo2_keygen | VerifierProgram;stage-c-build-rounds | 18,401 | 2,528 | 6,510 | 
| halo2_keygen | VerifierProgram;stage-c-build-rounds;PoseidonCell | 46,550 |  | 13,775 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs | 1,280,292 | 197,458 | 466,987 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;PoseidonCell | 3,839,150 |  | 1,136,075 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 40,526 | 4,276 | 18,076 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 56,350 |  | 16,675 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;cache-generator-powers | 70,410 | 12,000 | 21,630 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 8,549,550 | 353,940 | 1,581,960 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;pre-compute-rounds-context | 76,224 | 11,116 | 22,232 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 53,280 |  | 6,660 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 9,926,550 |  | 2,940,300 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 8,854,140 | 253,980 | 2,764,710 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 1,088,820 | 184,470 | 307,410 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 109,440 |  | 13,680 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 16,764,840 |  | 4,965,840 | 
| halo2_keygen | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 1,671,570 | 62,940 | 513,270 | 
| halo2_keygen | VerifierProgram;stage-e-verify-constraints | 9,499,973 | 1,889,049 | 2,918,862 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 496 | 11,528 | 371,618,977 | 472,992,226 | 496 | 8,978 | 0 | 1,276 | 1,352 | 1,532 | 2,306 | 12 | 2,105 | 207,107,703 | 402 |  | 1,000 | 3,422,909 | 4.06 | 
| internal.0 | 1 | 496 | 11,209 | 368,709,839 | 472,992,226 | 496 | 8,671 | 0 | 1,232 | 1,185 | 1,534 | 2,304 | 11 | 2,019 | 205,253,633 | 392 |  | 992 | 3,407,417 | 4.09 | 
| internal.0 | 2 | 180 | 5,732 | 136,140,539 | 185,031,138 | 180 | 4,055 | 0 | 532 | 629 | 678 | 1,215 | 9 | 821 | 74,650,857 | 172 |  | 448 | 1,151,426 | 3.95 | 
| internal.1 | 3 | 309 | 8,135 | 207,812,715 | 302,819,810 | 309 | 6,177 | 0 | 815 | 906 | 1,072 | 1,762 | 9 | 1,351 | 116,589,677 | 267 |  | 598 | 2,330,585 | 5.29 | 
| leaf | 0 | 184 | 4,195 | 170,628,411 | 253,173,226 | 184 | 2,483 | 0 | 267 | 244 | 578 | 732 | 8 | 479 | 72,248,817 | 177 |  | 506 | 1,260,083 | 3.88 | 
| leaf | 1 | 148 | 3,378 | 140,165,303 | 169,088,490 | 148 | 1,725 | 0 | 202 | 164 | 416 | 503 | 9 | 307 | 60,333,057 | 124 |  | 463 | 1,007,151 | 3.24 | 
| leaf | 2 | 148 | 3,327 | 140,166,463 | 169,088,490 | 148 | 1,642 | 0 | 187 | 152 | 371 | 503 | 8 | 305 | 60,333,405 | 117 |  | 490 | 1,007,180 | 3.25 | 
| leaf | 3 | 148 | 3,409 | 140,166,223 | 169,088,490 | 148 | 1,765 | 0 | 185 | 151 | 413 | 553 | 8 | 321 | 60,333,333 | 136 |  | 465 | 1,007,174 | 3.25 | 
| leaf | 4 | 148 | 3,285 | 140,166,423 | 169,088,490 | 148 | 1,637 | 0 | 182 | 154 | 370 | 501 | 8 | 308 | 60,333,393 | 117 |  | 462 | 1,007,179 | 3.27 | 
| leaf | 5 | 149 | 3,353 | 140,165,543 | 169,088,490 | 149 | 1,676 | 0 | 178 | 150 | 423 | 503 | 10 | 303 | 60,333,129 | 115 |  | 467 | 1,007,157 | 3.25 | 
| leaf | 6 | 171 | 3,758 | 158,701,031 | 208,084,458 | 171 | 2,067 | 0 | 225 | 182 | 459 | 631 | 9 | 400 | 67,171,969 | 163 |  | 462 | 1,166,501 | 3.79 | 
| root | 0 | 119 | 38,698 | 65,007,605 | 80,435,354 | 119 | 38,408 | 0 | 722 | 13,765 | 7,586 | 3,893 | 12 | 12,328 | 42,136,295 | 90 | 3 | 171 | 779,826 | 5.11 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 65,323,264 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 64,782,596 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 148,914,890 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 11,927,684 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 65,323,264 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 5,963,842 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 64,782,596 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 148,914,890 | 2,013,265,921 | 
| internal.0 | 2 | 0 | 4,882,564 | 2,013,265,921 | 
| internal.0 | 2 | 1 | 26,358,016 | 2,013,265,921 | 
| internal.0 | 2 | 2 | 2,441,282 | 2,013,265,921 | 
| internal.0 | 2 | 3 | 26,091,780 | 2,013,265,921 | 
| internal.0 | 2 | 4 | 131,072 | 2,013,265,921 | 
| internal.0 | 2 | 5 | 60,297,930 | 2,013,265,921 | 
| internal.1 | 3 | 0 | 8,454,276 | 2,013,265,921 | 
| internal.1 | 3 | 1 | 40,132,864 | 2,013,265,921 | 
| internal.1 | 3 | 2 | 4,227,138 | 2,013,265,921 | 
| internal.1 | 3 | 3 | 40,386,820 | 2,013,265,921 | 
| internal.1 | 3 | 4 | 262,144 | 2,013,265,921 | 
| internal.1 | 3 | 5 | 93,856,458 | 2,013,265,921 | 
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 
| leaf | 1 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 1 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 1 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 1 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 2 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 2 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 2 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 2 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 2 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 2 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 3 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 3 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 3 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 3 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 3 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 3 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 4 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 4 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 4 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 4 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 4 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 4 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 5 | 0 | 3,211,396 | 2,013,265,921 | 
| leaf | 5 | 1 | 16,914,688 | 2,013,265,921 | 
| leaf | 5 | 2 | 1,605,698 | 2,013,265,921 | 
| leaf | 5 | 3 | 17,043,716 | 2,013,265,921 | 
| leaf | 5 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 5 | 5 | 39,299,786 | 2,013,265,921 | 
| leaf | 6 | 0 | 4,391,044 | 2,013,265,921 | 
| leaf | 6 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 6 | 2 | 2,195,522 | 2,013,265,921 | 
| leaf | 6 | 3 | 20,586,756 | 2,013,265,921 | 
| leaf | 6 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 6 | 5 | 48,157,386 | 2,013,265,921 | 
| root | 0 | 0 | 2,572,420 | 2,013,265,921 | 
| root | 0 | 1 | 12,005,632 | 2,013,265,921 | 
| root | 0 | 2 | 1,286,210 | 2,013,265,921 | 
| root | 0 | 3 | 12,067,076 | 2,013,265,921 | 
| root | 0 | 4 | 65,536 | 2,013,265,921 | 
| root | 0 | 5 | 28,390,090 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agg_keygen | 0 | 26 | 1,663 | 9,505,566 | 7,747,601 | 26 | 499 |  | 25 | 73 | 59 | 259 | 17 | 2 | 65 | 919,380 | 12 | 2 | 1 | 9,223,372,036,854,775,807 | 
| fib_e2e | 0 | 229 | 3,222 | 146,796,066 | 253,084,876 | 229 | 2,679 | 0 | 238 | 269 | 595 | 852 | 5 | 0 | 532 | 59,842,060 | 188 | 66 | 1,748,000 | 37.24 | 
| fib_e2e | 1 | 228 | 3,086 | 146,788,271 | 253,031,994 | 228 | 2,573 | 112 | 245 | 215 | 592 | 846 | 6 | 0 | 477 | 59,837,129 | 192 | 66 | 1,748,000 | 37.31 | 
| fib_e2e | 2 | 229 | 3,113 | 146,788,335 | 253,031,994 | 229 | 2,609 | 112 | 256 | 216 | 591 | 853 | 5 | 0 | 474 | 59,837,145 | 214 | 66 | 1,748,000 | 37.33 | 
| fib_e2e | 3 | 228 | 3,081 | 146,788,358 | 253,031,994 | 228 | 2,579 | 112 | 248 | 213 | 591 | 840 | 5 | 0 | 495 | 59,837,156 | 188 | 67 | 1,748,000 | 37.28 | 
| fib_e2e | 4 | 228 | 3,063 | 146,788,271 | 253,031,994 | 228 | 2,562 | 112 | 248 | 213 | 597 | 836 | 5 | 0 | 475 | 59,837,129 | 188 | 67 | 1,748,000 | 37.21 | 
| fib_e2e | 5 | 228 | 3,068 | 146,788,335 | 253,031,994 | 228 | 2,568 | 112 | 244 | 213 | 592 | 833 | 6 | 0 | 491 | 59,837,145 | 190 | 67 | 1,748,000 | 37.29 | 
| fib_e2e | 6 | 206 | 2,297 | 128,298,001 | 160,813,290 | 206 | 1,829 | 97 | 180 | 180 | 381 | 612 | 5 | 0 | 338 | 51,906,075 | 135 | 52 | 1,512,210 | 37.21 | 

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
| fib_e2e | 0 | 0 | 6,029,422 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 17,039,880 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 3,014,711 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 17,039,836 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 12,451,904 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 56,502,857 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 6,029,316 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 17,039,424 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 3,014,658 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 17,039,396 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 12,451,840 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 56,501,002 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 3,932,336 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,624 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,168 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,692 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,209,000 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,531,924 | 2,013,265,921 | 

| group | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- |
| agg_keygen | 0 | 5,701,764 | 2,013,265,921 | 
| agg_keygen | 1 | 28,467,456 | 2,013,265,921 | 
| agg_keygen | 2 | 2,850,882 | 2,013,265,921 | 
| agg_keygen | 3 | 28,197,124 | 2,013,265,921 | 
| agg_keygen | 4 | 131,072 | 2,013,265,921 | 
| agg_keygen | 5 | 65,741,514 | 2,013,265,921 | 
| halo2_keygen | 0 | 2,572,420 | 2,013,265,921 | 
| halo2_keygen | 1 | 12,005,632 | 2,013,265,921 | 
| halo2_keygen | 2 | 1,286,210 | 2,013,265,921 | 
| halo2_keygen | 3 | 12,067,076 | 2,013,265,921 | 
| halo2_keygen | 4 | 65,536 | 2,013,265,921 | 
| halo2_keygen | 5 | 28,390,090 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f73da4db5941aabcf465a0bcd5856ff2efbef0a8

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17074296250)
