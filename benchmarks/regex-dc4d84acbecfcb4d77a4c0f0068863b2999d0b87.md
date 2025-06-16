| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.3%])</span> 21.74 | <span style='color: green'>(-0 [-0.9%])</span> 11.62 |
| regex_program | <span style='color: green'>(-0 [-0.7%])</span> 7.76 | <span style='color: green'>(-0 [-1.8%])</span> 4.28 |
| leaf | <span style='color: green'>(-0 [-0.2%])</span> 13.98 | <span style='color: green'>(-0 [-0.3%])</span> 7.03 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-26 [-0.7%])</span> 3,878 | <span style='color: green'>(-52 [-0.7%])</span> 7,756 | <span style='color: green'>(-77 [-1.8%])</span> 4,275 | <span style='color: red'>(+25 [+0.7%])</span> 3,481 |
| `main_cells_used     ` |  83,255,576 |  166,511,152 |  93,500,799 |  73,010,353 |
| `total_cycles        ` |  2,082,613 |  4,165,226 |  2,243,715 |  1,921,511 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-0.4%])</span> 350 | <span style='color: green'>(-3 [-0.4%])</span> 700 | <span style='color: green'>(-2 [-0.5%])</span> 389 | <span style='color: green'>(-1 [-0.3%])</span> 311 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.2%])</span> 956.50 | <span style='color: red'>(+3 [+0.2%])</span> 1,913 | <span style='color: green'>(-15 [-1.3%])</span> 1,124 | <span style='color: red'>(+18 [+2.3%])</span> 789 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-26 [-1.0%])</span> 2,571.50 | <span style='color: green'>(-52 [-1.0%])</span> 5,143 | <span style='color: green'>(-60 [-2.1%])</span> 2,762 | <span style='color: red'>(+8 [+0.3%])</span> 2,381 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-0 [-0.1%])</span> 498.50 | <span style='color: green'>(-1 [-0.1%])</span> 997 |  555 | <span style='color: green'>(-1 [-0.2%])</span> 442 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-6 [-2.6%])</span> 209.50 | <span style='color: green'>(-11 [-2.6%])</span> 419 | <span style='color: green'>(-15 [-6.5%])</span> 215 | <span style='color: red'>(+4 [+2.0%])</span> 204 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.2%])</span> 543.50 | <span style='color: green'>(-2 [-0.2%])</span> 1,087 | <span style='color: green'>(-1 [-0.2%])</span> 572 | <span style='color: green'>(-1 [-0.2%])</span> 515 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+9 [+3.5%])</span> 264.50 | <span style='color: red'>(+18 [+3.5%])</span> 529 | <span style='color: red'>(+3 [+1.1%])</span> 283 | <span style='color: red'>(+15 [+6.5%])</span> 246 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-4 [-1.4%])</span> 242.50 | <span style='color: green'>(-7 [-1.4%])</span> 485 | <span style='color: green'>(-5 [-1.8%])</span> 279 | <span style='color: green'>(-2 [-1.0%])</span> 206 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-26 [-3.1%])</span> 803.50 | <span style='color: green'>(-51 [-3.1%])</span> 1,607 | <span style='color: green'>(-43 [-4.8%])</span> 848 | <span style='color: green'>(-8 [-1.0%])</span> 759 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-11 [-0.2%])</span> 6,992.50 | <span style='color: green'>(-22 [-0.2%])</span> 13,985 | <span style='color: green'>(-24 [-0.3%])</span> 7,034 |  6,951 |
| `main_cells_used     ` |  151,828,241 |  303,656,482 |  154,227,506 |  149,428,976 |
| `total_cycles        ` |  1,975,760 |  3,951,520 |  2,006,538 |  1,944,982 |
| `execute_time_ms     ` | <span style='color: green'>(-6 [-0.7%])</span> 773 | <span style='color: green'>(-11 [-0.7%])</span> 1,546 | <span style='color: green'>(-5 [-0.6%])</span> 781 | <span style='color: green'>(-6 [-0.8%])</span> 765 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+18 [+1.3%])</span> 1,486 | <span style='color: red'>(+37 [+1.3%])</span> 2,972 | <span style='color: red'>(+31 [+2.1%])</span> 1,500 | <span style='color: red'>(+6 [+0.4%])</span> 1,472 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-24 [-0.5%])</span> 4,733.50 | <span style='color: green'>(-48 [-0.5%])</span> 9,467 | <span style='color: green'>(-50 [-1.0%])</span> 4,753 |  4,714 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-6 [-0.6%])</span> 865 | <span style='color: green'>(-11 [-0.6%])</span> 1,730 | <span style='color: green'>(-5 [-0.6%])</span> 867 | <span style='color: green'>(-6 [-0.7%])</span> 863 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-0.9%])</span> 394 | <span style='color: green'>(-7 [-0.9%])</span> 788 | <span style='color: green'>(-6 [-1.4%])</span> 418 | <span style='color: green'>(-1 [-0.3%])</span> 370 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.3%])</span> 1,241.50 | <span style='color: green'>(-8 [-0.3%])</span> 2,483 | <span style='color: green'>(-8 [-0.6%])</span> 1,244 |  1,239 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+7 [+1.3%])</span> 530 | <span style='color: red'>(+14 [+1.3%])</span> 1,060 | <span style='color: red'>(+6 [+1.1%])</span> 531 | <span style='color: red'>(+8 [+1.5%])</span> 529 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-4 [-0.8%])</span> 411 | <span style='color: green'>(-7 [-0.8%])</span> 822 | <span style='color: green'>(-4 [-1.0%])</span> 412 | <span style='color: green'>(-3 [-0.7%])</span> 410 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-15 [-1.2%])</span> 1,287 | <span style='color: green'>(-30 [-1.2%])</span> 2,574 | <span style='color: green'>(-21 [-1.6%])</span> 1,296 | <span style='color: green'>(-9 [-0.7%])</span> 1,278 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | num_children | keygen_time_ms | fri.log_blowup | commit_exe_time_ms |
| --- | --- | --- | --- | --- | --- |
| leaf |  | 1 |  | 1 |  | 
| regex_program | 2 |  | 559 | 1 | 20 | 

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
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
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
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 20 | 12 | 16,777,216 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 20 | 12 | 16,777,216 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 16 | 11 | 1,728 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 16 | 13 | 928 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 1,056 | 3,163 | 4,219 | 
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
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 52 | 36 | 176 | 
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
| leaf | 0 | 1,500 | 7,034 | 2,006,538 | 538,660,330 | 4,753 | 531 | 410 | 1,244 | 1,278 | 867 | 154,227,506 | 418 | 781 | 
| leaf | 1 | 1,472 | 6,951 | 1,944,982 | 538,660,330 | 4,714 | 529 | 412 | 1,239 | 1,296 | 863 | 149,428,976 | 370 | 765 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 9,371,780 | 2,013,265,921 | 
| leaf | 0 | 1 | 64,930,048 | 2,013,265,921 | 
| leaf | 0 | 2 | 4,685,890 | 2,013,265,921 | 
| leaf | 0 | 3 | 65,044,740 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 144,818,890 | 2,013,265,921 | 
| leaf | 1 | 0 | 9,371,780 | 2,013,265,921 | 
| leaf | 1 | 1 | 64,930,048 | 2,013,265,921 | 
| leaf | 1 | 2 | 4,685,890 | 2,013,265,921 | 
| leaf | 1 | 3 | 65,044,740 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 144,818,890 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,124 | 4,275 | 2,243,715 | 275,652,919 | 2,762 | 283 | 279 | 572 | 848 | 555 | 93,500,799 | 215 | 389 | 
| regex_program | 1 | 789 | 3,481 | 1,921,511 | 242,975,404 | 2,381 | 246 | 206 | 515 | 759 | 442 | 73,010,353 | 204 | 311 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 5,868,296 | 2,013,265,921 | 
| regex_program | 0 | 1 | 16,687,450 | 2,013,265,921 | 
| regex_program | 0 | 2 | 2,934,148 | 2,013,265,921 | 
| regex_program | 0 | 3 | 19,705,182 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,938 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 53,849,550 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,406,794 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,182,956 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,703,397 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,193,430 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,508,864 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,197,617 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/dc4d84acbecfcb4d77a4c0f0068863b2999d0b87

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15692713800)
