| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  14.59 |  14.59 | 14.59 |
| ecrecover_program |  4.15 |  4.15 |  4.15 |
| leaf |  10.44 |  10.44 |  10.44 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,140 |  4,140 |  4,140 |  4,140 |
| `main_cells_used     ` |  2,263,998 |  2,263,998 |  2,263,998 |  2,263,998 |
| `total_cells_used    ` |  11,345,240 |  11,345,240 |  11,345,240 |  11,345,240 |
| `execute_metered_time_ms` |  6 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  19.82 | -          |  19.82 |  19.82 |
| `execute_preflight_insns` |  122,859 |  122,859 |  122,859 |  122,859 |
| `execute_preflight_time_ms` |  62 |  62 |  62 |  62 |
| `execute_preflight_insn_mi/s` |  2.25 | -          |  2.25 |  2.25 |
| `trace_gen_time_ms   ` |  209 |  209 |  209 |  209 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` |  3,699 |  3,699 |  3,699 |  3,699 |
| `main_trace_commit_time_ms` |  37 |  37 |  37 |  37 |
| `generate_perm_trace_time_ms` |  51 |  51 |  51 |  51 |
| `perm_trace_commit_time_ms` |  34.24 |  34.24 |  34.24 |  34.24 |
| `quotient_poly_compute_time_ms` |  105.44 |  105.44 |  105.44 |  105.44 |
| `quotient_poly_commit_time_ms` |  9.18 |  9.18 |  9.18 |  9.18 |
| `pcs_opening_time_ms ` |  3,453 |  3,453 |  3,453 |  3,453 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,442 |  10,442 |  10,442 |  10,442 |
| `main_cells_used     ` |  34,468,876 |  34,468,876 |  34,468,876 |  34,468,876 |
| `total_cells_used    ` |  83,392,626 |  83,392,626 |  83,392,626 |  83,392,626 |
| `execute_preflight_insns` |  4,109,552 |  4,109,552 |  4,109,552 |  4,109,552 |
| `execute_preflight_time_ms` |  758 |  758 |  758 |  758 |
| `execute_preflight_insn_mi/s` |  5.77 | -          |  5.77 |  5.77 |
| `trace_gen_time_ms   ` |  313 |  313 |  313 |  313 |
| `memory_finalize_time_ms` |  27 |  27 |  27 |  27 |
| `stark_prove_excluding_trace_time_ms` |  9,368 |  9,368 |  9,368 |  9,368 |
| `main_trace_commit_time_ms` |  728 |  728 |  728 |  728 |
| `generate_perm_trace_time_ms` |  347 |  347 |  347 |  347 |
| `perm_trace_commit_time_ms` |  1,483.90 |  1,483.90 |  1,483.90 |  1,483.90 |
| `quotient_poly_compute_time_ms` |  833.54 |  833.54 |  833.54 |  833.54 |
| `quotient_poly_commit_time_ms` |  185.34 |  185.34 |  185.34 |  185.34 |
| `pcs_opening_time_ms ` |  5,784 |  5,784 |  5,784 |  5,784 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 74 | 4,152 | 10,452 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program |  | 4,140 |  | 1 | 6 | 122,859 | 19.82 | 0 | 
| leaf | 10,448 |  | 1 | 1 |  |  |  |  | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | 15 | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 2,097,152 |  | 16 | 11 | 56,623,104 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 1,048,576 |  | 16 | 13 | 30,408,704 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 524,288 |  | 312 | 398 | 372,244,480 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 8,388,608 |  | 84 | 27 | 931,135,488 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 524,288 |  | 36 | 38 | 38,797,312 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| ecrecover_program | 10 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | 11 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 56 | 166 | 7,104 | 
| ecrecover_program | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 320 | 263 | 9,328 | 
| ecrecover_program | 13 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 192 | 199 | 6,256 | 
| ecrecover_program | 14 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | 15 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | 16 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 16 |  | 192 | 199 | 6,256 | 
| ecrecover_program | 18 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | 19 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 64 |  | 52 | 31 | 5,312 | 
| ecrecover_program | 2 | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | 20 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | 21 | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | 22 | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | 23 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| ecrecover_program | 24 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| ecrecover_program | 25 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| ecrecover_program | 26 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8,192 |  | 32 | 32 | 524,288 | 
| ecrecover_program | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | 28 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 2,048 |  | 52 | 36 | 180,224 | 
| ecrecover_program | 29 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | 3 | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | 30 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 8,192 |  | 52 | 53 | 860,160 | 
| ecrecover_program | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | 33 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | 34 | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | 35 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | 36 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | 6 | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| ecrecover_program | 7 | AccessAdapterAir<16> | 0 | 4,096 |  | 16 | 25 | 167,936 | 
| ecrecover_program | 8 | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | 9 | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,513 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 11 | 
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

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 313 | 10,442 | 83,392,626 | 1,726,057,962 | 313 | 9,368 | 0 | 833.54 | 185.34 | 6 | 1,483.90 | 5,784 | 1,832 | 5,784 | 27 | 728 | 34,468,876 | 347 | 758 | 4,109,552 | 5.77 | 272 | 1,023 | 14 | 5,783 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 28,115,076 | 2,013,265,921 | 
| leaf | 0 | 1 | 222,134,528 | 2,013,265,921 | 
| leaf | 0 | 2 | 14,057,538 | 2,013,265,921 | 
| leaf | 0 | 3 | 222,363,908 | 2,013,265,921 | 
| leaf | 0 | 4 | 1,048,576 | 2,013,265,921 | 
| leaf | 0 | 5 | 488,506,058 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 209 | 4,140 | 11,345,240 | 31,789,298 | 209 | 3,699 | 0 | 105.44 | 9.18 | 11 | 34.24 | 3,453 | 93 | 3,453 | 0 | 37 | 2,263,998 | 51 | 62 | 122,859 | 2.25 | 21 | 115 | 1 | 3,453 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 375,956 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,169,840 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 187,978 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,534,916 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 446,696 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 320 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,694,650 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/df0a436d7edab1830ade118a50492097af1cb528

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/21527836671)
