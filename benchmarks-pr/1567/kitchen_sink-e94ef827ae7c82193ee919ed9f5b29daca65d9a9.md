| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  217.22 |  217.22 |
| kitchen_sink |  14.74 |  14.74 |
| leaf |  22.20 |  22.20 |
| internal.0 |  6.03 |  6.03 |
| root |  38.60 |  38.60 |
| halo2_outer |  91.14 |  91.14 |
| halo2_wrapper |  44.49 |  44.49 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  14,740 |  14,740 |  14,740 |  14,740 |
| `main_cells_used     ` |  904,738,232 |  904,738,232 |  904,738,232 |  904,738,232 |
| `total_cells_used    ` |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |  1,226,975,002 |
| `insns               ` |  153,644 |  307,288 |  153,644 |  153,644 |
| `execute_metered_time_ms` |  10 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  14.68 | -          |  14.68 |  14.68 |
| `execute_preflight_time_ms` |  102 |  102 |  102 |  102 |
| `execute_preflight_insn_mi/s` |  7.34 | -          |  7.34 |  7.34 |
| `trace_gen_time_ms   ` |  206 |  206 |  206 |  206 |
| `memory_finalize_time_ms` |  3 |  3 |  3 |  3 |
| `stark_prove_excluding_trace_time_ms` |  14,186 |  14,186 |  14,186 |  14,186 |
| `main_trace_commit_time_ms` |  4,516 |  4,516 |  4,516 |  4,516 |
| `generate_perm_trace_time_ms` |  463 |  463 |  463 |  463 |
| `perm_trace_commit_time_ms` |  1,364 |  1,364 |  1,364 |  1,364 |
| `quotient_poly_compute_time_ms` |  6,284 |  6,284 |  6,284 |  6,284 |
| `quotient_poly_commit_time_ms` |  285 |  285 |  285 |  285 |
| `pcs_opening_time_ms ` |  1,230 |  1,230 |  1,230 |  1,230 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  22,204 |  22,204 |  22,204 |  22,204 |
| `main_cells_used     ` |  769,362,578 |  769,362,578 |  769,362,578 |  769,362,578 |
| `total_cells_used    ` |  2,050,265,956 |  2,050,265,956 |  2,050,265,956 |  2,050,265,956 |
| `insns               ` |  7,903,936 |  7,903,936 |  7,903,936 |  7,903,936 |
| `execute_preflight_time_ms` |  1,334 |  1,334 |  1,334 |  1,334 |
| `execute_preflight_insn_mi/s` |  6.93 | -          |  6.93 |  6.93 |
| `trace_gen_time_ms   ` |  1,574 |  1,574 |  1,574 |  1,574 |
| `memory_finalize_time_ms` |  20 |  20 |  20 |  20 |
| `stark_prove_excluding_trace_time_ms` |  18,224 |  18,224 |  18,224 |  18,224 |
| `main_trace_commit_time_ms` |  3,754 |  3,754 |  3,754 |  3,754 |
| `generate_perm_trace_time_ms` |  1,624 |  1,624 |  1,624 |  1,624 |
| `perm_trace_commit_time_ms` |  4,809 |  4,809 |  4,809 |  4,809 |
| `quotient_poly_compute_time_ms` |  2,132 |  2,132 |  2,132 |  2,132 |
| `quotient_poly_commit_time_ms` |  1,301 |  1,301 |  1,301 |  1,301 |
| `pcs_opening_time_ms ` |  4,599 |  4,599 |  4,599 |  4,599 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,031 |  6,031 |  6,031 |  6,031 |
| `main_cells_used     ` |  79,527,291 |  79,527,291 |  79,527,291 |  79,527,291 |
| `total_cells_used    ` |  143,464,449 |  143,464,449 |  143,464,449 |  143,464,449 |
| `insns               ` |  1,197,770 |  1,197,770 |  1,197,770 |  1,197,770 |
| `execute_preflight_time_ms` |  482 |  482 |  482 |  482 |
| `execute_preflight_insn_mi/s` |  3.68 | -          |  3.68 |  3.68 |
| `trace_gen_time_ms   ` |  185 |  185 |  185 |  185 |
| `memory_finalize_time_ms` |  10 |  10 |  10 |  10 |
| `stark_prove_excluding_trace_time_ms` |  4,334 |  4,334 |  4,334 |  4,334 |
| `main_trace_commit_time_ms` |  904 |  904 |  904 |  904 |
| `generate_perm_trace_time_ms` |  194 |  194 |  194 |  194 |
| `perm_trace_commit_time_ms` |  686 |  686 |  686 |  686 |
| `quotient_poly_compute_time_ms` |  618 |  618 |  618 |  618 |
| `quotient_poly_commit_time_ms` |  677 |  677 |  677 |  677 |
| `pcs_opening_time_ms ` |  1,249 |  1,249 |  1,249 |  1,249 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,599 |  38,599 |  38,599 |  38,599 |
| `main_cells_used     ` |  41,518,314 |  41,518,314 |  41,518,314 |  41,518,314 |
| `total_cells_used    ` |  64,201,156 |  64,201,156 |  64,201,156 |  64,201,156 |
| `insns               ` |  772,582 |  772,582 |  772,582 |  772,582 |
| `execute_preflight_time_ms` |  161 |  161 |  161 |  161 |
| `execute_preflight_insn_mi/s` |  5.28 | -          |  5.28 |  5.28 |
| `trace_gen_time_ms   ` |  116 |  116 |  116 |  116 |
| `memory_finalize_time_ms` |  8 |  8 |  8 |  8 |
| `stark_prove_excluding_trace_time_ms` |  38,322 |  38,322 |  38,322 |  38,322 |
| `main_trace_commit_time_ms` |  12,309 |  12,309 |  12,309 |  12,309 |
| `generate_perm_trace_time_ms` |  88 |  88 |  88 |  88 |
| `perm_trace_commit_time_ms` |  7,575 |  7,575 |  7,575 |  7,575 |
| `quotient_poly_compute_time_ms` |  720 |  720 |  720 |  720 |
| `quotient_poly_commit_time_ms` |  13,658 |  13,658 |  13,658 |  13,658 |
| `pcs_opening_time_ms ` |  3,947 |  3,947 |  3,947 |  3,947 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  91,139 |  91,139 |  91,139 |  91,139 |
| `main_cells_used     ` |  65,627,358 |  65,627,358 |  65,627,358 |  65,627,358 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,492 |  44,492 |  44,492 |  44,492 |



<details>
<summary>Detailed Metrics</summary>

|  | trace_gen_time_ms | total_cells_used | system_trace_gen_time_ms | single_trace_gen_time_ms | prove_time_ms | prove_for_evm_time_ms | memory_finalize_time_ms | main_cells_used | insns | execute_preflight_time_ms | execute_preflight_insn_mi/s | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 117 | 64,201,156 | 117 | 0 | 91,152 | 44,493 | 9 | 41,518,314 | 772,582 | 298 | 5.42 | 14,793 | 39,629 | 

| group | total_proof_time_ms | single_leaf_agg_time_ms | single_internal_agg_time_ms | prove_segment_time_ms | num_children | memory_to_vec_partition_time_ms | main_cells_used | insns | fri.log_blowup | execute_metered_time_ms | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| halo2_outer | 91,139 |  |  |  |  |  | 65,627,358 |  |  |  |  |  | 
| halo2_wrapper | 44,492 |  |  |  |  |  |  |  |  |  |  |  | 
| internal.0 |  |  | 6,032 |  | 3 |  |  |  | 2 |  |  |  | 
| kitchen_sink |  |  |  | 14,740 |  | 6 |  | 153,644 | 1 | 10 | 14.68 | 37 | 
| leaf |  | 22,211 |  |  | 1 |  |  |  | 1 |  |  |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| leaf | AccessAdapterAir<2> | 0 | 4,194,304 |  | 16 | 11 | 113,246,208 | 
| leaf | AccessAdapterAir<4> | 0 | 2,097,152 |  | 16 | 13 | 60,817,408 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | FriReducedOpeningAir | 0 | 8,388,608 |  | 84 | 27 | 931,135,488 | 
| leaf | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 1,048,576 |  | 312 | 398 | 744,488,960 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 36 | 29 | 272,629,760 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 40 | 21 | 127,926,272 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 40 | 27 | 35,127,296 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 1,048,576 |  | 36 | 38 | 77,594,624 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
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
| kitchen_sink | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| kitchen_sink | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| kitchen_sink | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| kitchen_sink | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | KeccakVmAir | 0 | 262,144 |  | 1,056 | 3,163 | 1,105,985,536 | 
| kitchen_sink | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| kitchen_sink | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| kitchen_sink | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | ProgramAir | 0 | 16,384 |  | 8 | 10 | 294,912 | 
| kitchen_sink | RangeTupleCheckerAir<2> | 0 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | Sha256VmAir | 0 | 524,288 |  | 108 | 470 | 303,038,464 | 
| kitchen_sink | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 32,768 |  | 52 | 36 | 2,883,584 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 8,192 |  | 28 | 26 | 442,368 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| kitchen_sink | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,024 |  | 28 | 18 | 47,104 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 2,048 |  | 192 | 168 | 737,280 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 1,024 |  | 68 | 169 | 242,688 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 256 |  | 192 | 164 | 91,136 | 
| kitchen_sink | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 256 |  | 48 | 124 | 44,032 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 8 |  | 88 | 242 | 2,640 | 
| kitchen_sink | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,048 |  | 36 | 28 | 131,072 | 
| kitchen_sink | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 52 | 41 | 12,189,696 | 
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 4 |  | 836 | 547 | 5,532 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 1,668 | 1,020 | 10,752 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 384 | 294 | 41,920 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2 |  | 860 | 625 | 2,202 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 4 |  | 496 | 393 | 2,404 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 2 |  | 1,340 | 949 | 3,426 | 
| kitchen_sink | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | fri.log_blowup | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 185 | 6,031 | 143,464,449 | 224,975,330 | 185 | 4,334 | 0 | 618 | 677 | 686 | 1,249 | 10 | 904 | 79,527,291 | 1,197,770 | 194 |  | 482 | 3.68 | 
| leaf | 0 | 1,574 | 22,204 | 2,050,265,956 | 2,500,267,498 | 1,574 | 18,224 | 0 | 2,132 | 1,301 | 4,809 | 4,599 | 20 | 3,754 | 769,362,578 | 7,903,936 | 1,624 |  | 1,334 | 6.93 | 
| root | 0 | 116 | 38,599 | 64,201,156 | 80,435,354 | 116 | 38,322 | 0 | 720 | 13,658 | 7,575 | 3,947 | 8 | 12,309 | 41,518,314 | 772,582 | 88 | 3 | 161 | 5.28 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 5,177,476 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 30,814,464 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 2,588,738 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 30,941,444 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 70,177,482 | 2,013,265,921 | 
| leaf | 0 | 0 | 39,125,124 | 2,013,265,921 | 
| leaf | 0 | 1 | 291,111,168 | 2,013,265,921 | 
| leaf | 0 | 2 | 19,562,562 | 2,013,265,921 | 
| leaf | 0 | 3 | 288,096,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 642,351,818 | 2,013,265,921 | 
| root | 0 | 0 | 2,572,420 | 2,013,265,921 | 
| root | 0 | 1 | 12,005,632 | 2,013,265,921 | 
| root | 0 | 2 | 1,286,210 | 2,013,265,921 | 
| root | 0 | 3 | 12,067,076 | 2,013,265,921 | 
| root | 0 | 4 | 65,536 | 2,013,265,921 | 
| root | 0 | 5 | 28,390,090 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 206 | 14,740 | 1,226,975,002 | 1,481,193,346 | 206 | 14,186 | 3 | 6,284 | 285 | 1,364 | 1,230 | 6 | 3 | 4,516 | 904,738,232 | 153,644 | 463 | 102 | 7.34 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,977,944 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 32,428,632 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 988,972 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 32,011,136 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 57,344 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 24,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 49,612,036 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 1,048,576 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,320 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 120,668,384 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/e94ef827ae7c82193ee919ed9f5b29daca65d9a9

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16842909895)
