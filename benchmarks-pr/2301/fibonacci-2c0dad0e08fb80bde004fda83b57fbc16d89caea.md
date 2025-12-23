| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  6.27 |  4.19 | 4.19 |
| fibonacci_program |  2.22 |  1.40 |  1.40 |
| leaf |  4.05 |  2.79 |  2.79 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,106.50 |  2,213 |  1,393 |  820 |
| `main_cells_used     ` |  1,050,201 |  2,100,402 |  1,064,416 |  1,035,986 |
| `total_cells_used    ` |  9,692,939 |  19,385,878 |  9,708,170 |  9,677,708 |
| `execute_metered_time_ms` |  9 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  151.87 | -          |  151.87 |  151.87 |
| `execute_preflight_insns` |  750,104.50 |  1,500,209 |  873,000 |  627,209 |
| `execute_preflight_time_ms` |  29 |  58 |  33 |  25 |
| `execute_preflight_insn_mi/s` |  36.58 | -          |  38.52 |  34.65 |
| `trace_gen_time_ms   ` |  187 |  374 |  192 |  182 |
| `memory_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` |  785.50 |  1,571 |  1,007 |  564 |
| `main_trace_commit_time_ms` |  30 |  60 |  32 |  28 |
| `generate_perm_trace_time_ms` |  26.50 |  53 |  31 |  22 |
| `perm_trace_commit_time_ms` |  40.78 |  81.57 |  43.77 |  37.79 |
| `quotient_poly_compute_time_ms` |  29.82 |  59.64 |  30.98 |  28.66 |
| `quotient_poly_commit_time_ms` |  12.65 |  25.30 |  12.98 |  12.32 |
| `pcs_opening_time_ms ` |  644 |  1,288 |  854 |  434 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  2,025.50 |  4,051 |  2,789 |  1,262 |
| `main_cells_used     ` |  9,910,726 |  19,821,452 |  10,121,356 |  9,700,096 |
| `total_cells_used    ` |  25,895,192 |  51,790,384 |  26,398,858 |  25,391,526 |
| `execute_preflight_insns` |  1,999,758 |  3,999,516 |  2,061,683 |  1,937,833 |
| `execute_preflight_time_ms` |  912.50 |  1,825 |  1,456 |  369 |
| `execute_preflight_insn_mi/s` |  5.80 | -          |  6.01 |  5.59 |
| `trace_gen_time_ms   ` |  98 |  196 |  102 |  94 |
| `memory_finalize_time_ms` |  10 |  20 |  10 |  10 |
| `stark_prove_excluding_trace_time_ms` |  1,013.50 |  2,027 |  1,237 |  790 |
| `main_trace_commit_time_ms` |  130.50 |  261 |  131 |  130 |
| `generate_perm_trace_time_ms` |  46.50 |  93 |  48 |  45 |
| `perm_trace_commit_time_ms` |  182.79 |  365.59 |  183.35 |  182.24 |
| `quotient_poly_compute_time_ms` |  142.26 |  284.52 |  142.47 |  142.05 |
| `quotient_poly_commit_time_ms` |  28.46 |  56.93 |  28.94 |  27.99 |
| `pcs_opening_time_ms ` |  479.50 |  959 |  704 |  255 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 74 | 2,228 | 4,060 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 820 |  | 1 | 9 | 1,500,209 | 151.87 | 0 | 
| leaf | 2,792 |  | 1 | 1 |  |  |  |  | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 11 | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 11 | JalRangeCheckAir | 1 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 15 | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 15 | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 262,144 |  | 20 | 12 | 8,388,608 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | 0 | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 13 | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 4 |  | 28 | 20 | 192 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 1 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 131,072 |  | 40 | 37 | 10,092,544 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 25 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 1 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 1 | 64 |  | 16 | 17 | 2,112 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| fibonacci_program | VmConnectorAir | 2 | 5 | 11 | 
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
| leaf | 0 | 102 | 1,262 | 26,398,858 | 342,564,330 | 102 | 790 | 0 | 142.47 | 27.99 | 5 | 182.24 | 255 | 231 | 255 | 10 | 131 | 10,121,356 | 48 | 369 | 2,061,683 | 6.01 | 33 | 171 | 0 | 255 | 
| leaf | 1 | 94 | 2,789 | 25,391,526 | 342,564,330 | 94 | 1,237 | 0 | 142.05 | 28.94 | 6 | 183.35 | 704 | 229 | 704 | 10 | 130 | 9,700,096 | 45 | 1,456 | 1,937,833 | 5.59 | 33 | 172 | 0 | 704 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 6,619,268 | 2,013,265,921 | 
| leaf | 0 | 1 | 34,615,552 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,309,634 | 2,013,265,921 | 
| leaf | 0 | 3 | 34,873,604 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 80,073,418 | 2,013,265,921 | 
| leaf | 1 | 0 | 6,619,268 | 2,013,265,921 | 
| leaf | 1 | 1 | 34,615,552 | 2,013,265,921 | 
| leaf | 1 | 2 | 3,309,634 | 2,013,265,921 | 
| leaf | 1 | 3 | 34,873,604 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 80,073,418 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 192 | 1,393 | 9,677,708 | 84,395,212 | 192 | 1,007 | 0 | 30.98 | 12.98 | 6 | 43.77 | 854 | 75 | 854 | 0 | 32 | 1,035,986 | 31 | 25 | 873,000 | 34.65 | 14 | 44 | 0 | 854 | 
| fibonacci_program | 1 | 182 | 820 | 9,708,170 | 74,305,770 | 182 | 564 | 1 | 28.66 | 12.32 | 6 | 37.79 | 434 | 60 | 434 | 0 | 28 | 1,064,416 | 22 | 33 | 627,209 | 38.52 | 11 | 41 | 0 | 433 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 1,966,190 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 5,374,472 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 983,095 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 5,374,428 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 3,604,544 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 18,229,833 | 2,013,265,921 | 
| fibonacci_program | 1 | 0 | 1,704,112 | 2,013,265,921 | 
| fibonacci_program | 1 | 1 | 4,588,240 | 2,013,265,921 | 
| fibonacci_program | 1 | 2 | 852,056 | 2,013,265,921 | 
| fibonacci_program | 1 | 3 | 4,588,308 | 2,013,265,921 | 
| fibonacci_program | 1 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 1 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 1 | 6 | 3,211,304 | 2,013,265,921 | 
| fibonacci_program | 1 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 1 | 8 | 15,871,124 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/2c0dad0e08fb80bde004fda83b57fbc16d89caea

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20448996466)
