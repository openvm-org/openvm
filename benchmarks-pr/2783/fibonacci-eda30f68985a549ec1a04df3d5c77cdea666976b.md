| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: green'>(-0 [-4.6%])</span> 6.89 | <span style='color: green'>(-0 [-0.4%])</span> 4.12 | 4.12 |
| fibonacci_program | <span style='color: red'>(+0 [+0.3%])</span> 2.40 | <span style='color: red'>(+0 [+7.4%])</span> 1.31 |  1.31 |
| leaf | <span style='color: green'>(-0 [-7.0%])</span> 4.49 | <span style='color: green'>(-0 [-3.7%])</span> 2.80 |  2.80 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4 [+0.3%])</span> 1,196.50 | <span style='color: red'>(+8 [+0.3%])</span> 2,393 | <span style='color: red'>(+91 [+7.5%])</span> 1,306 | <span style='color: green'>(-83 [-7.1%])</span> 1,087 |
| `main_cells_used     ` |  1,050,201 |  2,100,402 |  1,064,416 |  1,035,986 |
| `total_cells_used    ` |  9,692,939 |  19,385,878 |  9,708,170 |  9,677,708 |
| `execute_metered_time_ms` |  9 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-3 [-1.5%])</span> 162.08 | -          | <span style='color: green'>(-3 [-1.5%])</span> 162.08 | <span style='color: green'>(-3 [-1.5%])</span> 162.08 |
| `execute_preflight_insns` |  750,104.50 |  1,500,209 |  873,000 |  627,209 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+1.9%])</span> 27.50 | <span style='color: red'>(+1 [+1.9%])</span> 55 |  33 | <span style='color: red'>(+1 [+4.8%])</span> 22 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-1.2%])</span> 38.06 | -          | <span style='color: green'>(-1 [-2.0%])</span> 38.12 | <span style='color: green'>(-0 [-0.4%])</span> 38 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.8%])</span> 192.50 | <span style='color: red'>(+3 [+0.8%])</span> 385 | <span style='color: red'>(+2 [+1.0%])</span> 196 | <span style='color: red'>(+1 [+0.5%])</span> 189 |
| `memory_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+4 [+0.4%])</span> 875.50 | <span style='color: red'>(+7 [+0.4%])</span> 1,751 | <span style='color: red'>(+11 [+1.2%])</span> 923 | <span style='color: green'>(-4 [-0.5%])</span> 828 |
| `main_trace_commit_time_ms` |  30 |  60 |  32 |  28 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+8 [+35.6%])</span> 30.50 | <span style='color: red'>(+16 [+35.6%])</span> 61 | <span style='color: red'>(+11 [+44.0%])</span> 36 | <span style='color: red'>(+5 [+25.0%])</span> 25 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.8%])</span> 40.53 | <span style='color: green'>(-1 [-0.8%])</span> 81.07 | <span style='color: green'>(-0 [-0.3%])</span> 43.07 | <span style='color: green'>(-0 [-1.3%])</span> 37.99 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+0 [+1.2%])</span> 31.56 | <span style='color: red'>(+1 [+1.2%])</span> 63.11 | <span style='color: green'>(-1 [-2.2%])</span> 33.25 | <span style='color: red'>(+1 [+5.3%])</span> 29.86 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+1 [+5.7%])</span> 12.51 | <span style='color: red'>(+1 [+5.7%])</span> 25.03 | <span style='color: red'>(+0 [+3.7%])</span> 12.84 | <span style='color: red'>(+1 [+7.8%])</span> 12.19 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-6 [-0.8%])</span> 727.50 | <span style='color: green'>(-11 [-0.8%])</span> 1,455 | <span style='color: green'>(-16 [-2.1%])</span> 763 | <span style='color: red'>(+5 [+0.7%])</span> 692 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-170 [-7.0%])</span> 2,246 | <span style='color: green'>(-340 [-7.0%])</span> 4,492 | <span style='color: green'>(-107 [-3.7%])</span> 2,802 | <span style='color: green'>(-233 [-12.1%])</span> 1,690 |
| `main_cells_used     ` |  9,916,841 |  19,833,682 |  10,127,226 |  9,706,456 |
| `total_cells_used    ` |  25,907,255 |  51,814,510 |  26,410,340 |  25,404,170 |
| `execute_preflight_insns` |  2,001,471.50 |  4,002,943 |  2,063,423 |  1,939,520 |
| `execute_preflight_time_ms` | <span style='color: green'>(-4 [-0.4%])</span> 889.50 | <span style='color: green'>(-8 [-0.4%])</span> 1,779 | <span style='color: green'>(-4 [-0.3%])</span> 1,445 | <span style='color: green'>(-4 [-1.2%])</span> 334 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+0 [+1.0%])</span> 6.66 | -          | <span style='color: red'>(+0 [+0.7%])</span> 6.68 | <span style='color: red'>(+0 [+1.2%])</span> 6.63 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+0 [+0.5%])</span> 94.50 | <span style='color: red'>(+1 [+0.5%])</span> 189 | <span style='color: green'>(-1 [-1.0%])</span> 97 | <span style='color: red'>(+2 [+2.2%])</span> 92 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-10.5%])</span> 8.50 | <span style='color: green'>(-2 [-10.5%])</span> 17 | <span style='color: green'>(-1 [-10.0%])</span> 9 | <span style='color: green'>(-1 [-11.1%])</span> 8 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-166 [-11.7%])</span> 1,260.50 | <span style='color: green'>(-333 [-11.7%])</span> 2,521 | <span style='color: green'>(-223 [-15.0%])</span> 1,263 | <span style='color: green'>(-110 [-8.0%])</span> 1,258 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+2 [+1.5%])</span> 133 | <span style='color: red'>(+4 [+1.5%])</span> 266 | <span style='color: red'>(+2 [+1.5%])</span> 134 | <span style='color: red'>(+2 [+1.5%])</span> 132 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+14.7%])</span> 43 | <span style='color: red'>(+11 [+14.7%])</span> 86 | <span style='color: red'>(+5 [+13.2%])</span> 43 | <span style='color: red'>(+6 [+16.2%])</span> 43 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+1 [+0.7%])</span> 184.27 | <span style='color: red'>(+2 [+0.7%])</span> 368.54 | <span style='color: red'>(+1 [+0.5%])</span> 184.93 | <span style='color: red'>(+2 [+0.9%])</span> 183.60 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+1.2%])</span> 142.32 | <span style='color: red'>(+3 [+1.2%])</span> 284.63 | <span style='color: red'>(+1 [+0.8%])</span> 142.39 | <span style='color: red'>(+2 [+1.7%])</span> 142.24 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-0 [-1.6%])</span> 28.46 | <span style='color: green'>(-1 [-1.6%])</span> 56.92 | <span style='color: green'>(-1 [-2.1%])</span> 28.48 | <span style='color: green'>(-0 [-1.1%])</span> 28.44 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-176 [-19.6%])</span> 726 | <span style='color: green'>(-353 [-19.6%])</span> 1,452 | <span style='color: green'>(-233 [-24.3%])</span> 727 | <span style='color: green'>(-120 [-14.2%])</span> 725 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms | agg_layer_time_ms |
| --- | --- | --- |
|  | 70 | 2,408 | 4,500 | 

| group | single_leaf_agg_time_ms | prove_segment_time_ms | num_children | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program |  | 1,087 |  | 1 | 9 | 1,500,209 | 162.08 | 0 | 
| leaf | 2,804 |  | 1 | 1 |  |  |  |  | 

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
| fibonacci_program | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 13 | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4 |  | 36 | 28 | 256 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 32 | 32 | 256 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 262,144 |  | 40 | 37 | 20,185,088 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 25 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | 0 | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | 12 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | 14 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 4 |  | 28 | 20 | 192 | 
| fibonacci_program | 15 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | 16 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| fibonacci_program | 17 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | 18 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| fibonacci_program | 2 | PersistentBoundaryAir<8> | 1 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | 20 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | 22 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 131,072 |  | 40 | 37 | 10,092,544 | 
| fibonacci_program | 23 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| fibonacci_program | 24 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | 26 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | 27 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | 3 | MemoryMerkleAir<8> | 1 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | 6 | AccessAdapterAir<8> | 1 | 64 |  | 16 | 17 | 2,112 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 41 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| fibonacci_program | PhantomAir | 2 | 3 | 6 | 
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
| fibonacci_program | VmConnectorAir | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 574 | 
| leaf | PhantomAir | 2 | 3 | 6 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 12 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 97 | 1,690 | 26,410,340 | 342,564,330 | 97 | 1,258 | 0 | 142.39 | 28.44 | 6 | 183.60 | 725 | 228 | 724 | 8 | 132 | 10,127,226 | 43 | 334 | 2,063,423 | 6.68 | 32 | 172 | 0 | 724 | 
| leaf | 1 | 92 | 2,802 | 25,404,170 | 342,564,330 | 92 | 1,263 | 0 | 142.24 | 28.48 | 6 | 184.93 | 727 | 229 | 726 | 9 | 134 | 9,706,456 | 43 | 1,445 | 1,939,520 | 6.63 | 33 | 172 | 0 | 726 | 

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
| fibonacci_program | 0 | 189 | 1,306 | 9,677,708 | 84,395,212 | 189 | 923 | 0 | 33.25 | 12.84 | 6 | 43.07 | 763 | 80 | 763 | 0 | 32 | 1,035,986 | 36 | 33 | 873,000 | 38 | 14 | 46 | 0 | 763 | 
| fibonacci_program | 1 | 196 | 1,087 | 9,708,170 | 74,305,770 | 196 | 828 | 1 | 29.86 | 12.19 | 5 | 37.99 | 692 | 64 | 692 | 0 | 28 | 1,064,416 | 25 | 22 | 627,209 | 38.12 | 11 | 42 | 0 | 692 | 

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


Commit: https://github.com/openvm-org/openvm/commit/eda30f68985a549ec1a04df3d5c77cdea666976b

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25932210868)
