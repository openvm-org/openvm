| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total |  3.04 |  1.85 | 1.85 |
| fibonacci_program |  1.02 |  0.59 |  0.59 |
| leaf |  2.02 |  1.26 |  1.26 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  506 |  1,012 |  580 |  432 |
| `main_cells_used     ` |  1,050,201 |  2,100,402 |  1,064,416 |  1,035,986 |
| `total_cells_used    ` |  9,692,939 |  19,385,878 |  9,708,170 |  9,677,708 |
| `execute_metered_time_ms` |  10 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  148.55 | -          |  148.55 |  148.55 |
| `execute_preflight_insns` |  750,104.50 |  1,500,209 |  873,000 |  627,209 |
| `execute_preflight_time_ms` |  29 |  58 |  33 |  25 |
| `execute_preflight_insn_mi/s` |  36.79 | -          |  38.71 |  34.87 |
| `trace_gen_time_ms   ` |  187.50 |  375 |  193 |  182 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` |  184 |  368 |  192 |  176 |
| `main_trace_commit_time_ms` |  30 |  60 |  32 |  28 |
| `generate_perm_trace_time_ms` |  25 |  50 |  28 |  22 |
| `perm_trace_commit_time_ms` |  40.74 |  81.48 |  43.71 |  37.77 |
| `quotient_poly_compute_time_ms` |  31.77 |  63.54 |  33.14 |  30.40 |
| `quotient_poly_commit_time_ms` |  12.28 |  24.56 |  12.53 |  12.03 |
| `pcs_opening_time_ms ` |  42.50 |  85 |  44 |  41 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,009.50 |  2,019 |  1,262 |  757 |
| `main_cells_used     ` |  6,475,562 |  12,951,124 |  6,608,072 |  6,343,052 |
| `total_cells_used    ` |  17,568,644 |  35,137,288 |  17,887,054 |  17,250,234 |
| `execute_preflight_insns` |  1,190,380 |  2,380,760 |  1,229,581 |  1,151,179 |
| `execute_preflight_time_ms` |  493 |  986 |  788 |  198 |
| `execute_preflight_insn_mi/s` |  6.88 | -          |  6.95 |  6.81 |
| `trace_gen_time_ms   ` |  58 |  116 |  62 |  54 |
| `memory_finalize_time_ms` |  5.50 |  11 |  6 |  5 |
| `stark_prove_excluding_trace_time_ms` |  457 |  914 |  496 |  418 |
| `main_trace_commit_time_ms` |  82 |  164 |  89 |  75 |
| `generate_perm_trace_time_ms` |  40.50 |  81 |  51 |  30 |
| `perm_trace_commit_time_ms` |  122.50 |  245 |  140.18 |  104.82 |
| `quotient_poly_compute_time_ms` |  92.36 |  184.72 |  100.61 |  84.11 |
| `quotient_poly_commit_time_ms` |  22.46 |  44.92 |  24.36 |  20.56 |
| `pcs_opening_time_ms ` |  94.50 |  189 |  101 |  88 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 72 | 1,028 | 2,024 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 432 |  | 1 | 10 | 1,500,209 | 148.55 | 0 | 
| leaf | 1,263 |  | 1 | 1 |  |  |  |  | 

| group | air_id | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 10 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | 11 | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | 11 | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | 12 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 13 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | 14 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | 15 | PhantomAir | 0 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | 15 | PhantomAir | 1 | 16,384 |  | 12 | 6 | 294,912 | 
| leaf | 16 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 16 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | 3 | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | 3 | VolatileBoundaryAir | 1 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | 4 | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | 4 | AccessAdapterAir<2> | 1 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | 5 | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | 5 | AccessAdapterAir<4> | 1 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | 6 | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 6 | AccessAdapterAir<8> | 1 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | 7 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | 8 | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | 8 | FriReducedOpeningAir | 1 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | 9 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 36 | 38 | 9,699,328 | 

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
| leaf | 0 | 62 | 757 | 17,887,054 | 252,878,314 | 61 | 496 | 0 | 100.61 | 24.36 | 4 | 140.18 | 88 | 192 | 87 | 6 | 89 | 6,608,072 | 51 | 198 | 1,229,581 | 6.95 | 26 | 125 | 0 | 87 | 
| leaf | 1 | 54 | 1,262 | 17,250,234 | 207,789,546 | 54 | 418 | 0 | 84.11 | 20.56 | 4 | 104.82 | 101 | 135 | 100 | 5 | 75 | 6,343,052 | 30 | 788 | 1,151,179 | 6.81 | 22 | 105 | 0 | 100 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,406,852 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,703,426 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,264,010 | 2,013,265,921 | 
| leaf | 1 | 0 | 4,358,276 | 2,013,265,921 | 
| leaf | 1 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 1 | 2 | 2,179,138 | 2,013,265,921 | 
| leaf | 1 | 3 | 20,586,756 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 48,108,234 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 193 | 580 | 9,677,708 | 84,395,212 | 193 | 192 | 0 | 33.14 | 12.53 | 4 | 43.71 | 41 | 72 | 40 | 0 | 32 | 1,035,986 | 28 | 25 | 873,000 | 34.87 | 14 | 46 | 0 | 40 | 
| fibonacci_program | 1 | 182 | 432 | 9,708,170 | 74,305,770 | 182 | 176 | 1 | 30.40 | 12.03 | 4 | 37.77 | 44 | 60 | 44 | 0 | 28 | 1,064,416 | 22 | 33 | 627,209 | 38.71 | 11 | 42 | 0 | 44 | 

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


Commit: https://github.com/openvm-org/openvm/commit/0f1d62e6d596271b6aef81d5c315cc0e2c34d0d7

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20109889605)
