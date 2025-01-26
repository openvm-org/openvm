| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.7%])</span> 12.44 | <span style='color: red'>(+0 [+0.7%])</span> 12.44 |
| fibonacci_program | <span style='color: red'>(+0 [+0.7%])</span> 5.31 | <span style='color: red'>(+0 [+0.7%])</span> 5.31 |
| leaf | <span style='color: red'>(+0 [+0.7%])</span> 7.13 | <span style='color: red'>(+0 [+0.7%])</span> 7.13 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+35 [+0.7%])</span> 5,307 | <span style='color: red'>(+35 [+0.7%])</span> 5,307 | <span style='color: red'>(+35 [+0.7%])</span> 5,307 | <span style='color: red'>(+35 [+0.7%])</span> 5,307 |
| `main_cells_used     ` |  51,485,080 |  51,485,080 |  51,485,080 |  51,485,080 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.3%])</span> 311 | <span style='color: red'>(+1 [+0.3%])</span> 311 | <span style='color: red'>(+1 [+0.3%])</span> 311 | <span style='color: red'>(+1 [+0.3%])</span> 311 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.1%])</span> 808 | <span style='color: green'>(-1 [-0.1%])</span> 808 | <span style='color: green'>(-1 [-0.1%])</span> 808 | <span style='color: green'>(-1 [-0.1%])</span> 808 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+35 [+0.8%])</span> 4,188 | <span style='color: red'>(+35 [+0.8%])</span> 4,188 | <span style='color: red'>(+35 [+0.8%])</span> 4,188 | <span style='color: red'>(+35 [+0.8%])</span> 4,188 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+7 [+0.8%])</span> 870 | <span style='color: red'>(+7 [+0.8%])</span> 870 | <span style='color: red'>(+7 [+0.8%])</span> 870 | <span style='color: red'>(+7 [+0.8%])</span> 870 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+5 [+3.8%])</span> 136 | <span style='color: red'>(+5 [+3.8%])</span> 136 | <span style='color: red'>(+5 [+3.8%])</span> 136 | <span style='color: red'>(+5 [+3.8%])</span> 136 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+17 [+2.1%])</span> 812 | <span style='color: red'>(+17 [+2.1%])</span> 812 | <span style='color: red'>(+17 [+2.1%])</span> 812 | <span style='color: red'>(+17 [+2.1%])</span> 812 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+12 [+2.3%])</span> 523 | <span style='color: red'>(+12 [+2.3%])</span> 523 | <span style='color: red'>(+12 [+2.3%])</span> 523 | <span style='color: red'>(+12 [+2.3%])</span> 523 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+16 [+2.1%])</span> 781 | <span style='color: red'>(+16 [+2.1%])</span> 781 | <span style='color: red'>(+16 [+2.1%])</span> 781 | <span style='color: red'>(+16 [+2.1%])</span> 781 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-23 [-2.1%])</span> 1,062 | <span style='color: green'>(-23 [-2.1%])</span> 1,062 | <span style='color: green'>(-23 [-2.1%])</span> 1,062 | <span style='color: green'>(-23 [-2.1%])</span> 1,062 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+51 [+0.7%])</span> 7,133 | <span style='color: red'>(+51 [+0.7%])</span> 7,133 | <span style='color: red'>(+51 [+0.7%])</span> 7,133 | <span style='color: red'>(+51 [+0.7%])</span> 7,133 |
| `main_cells_used     ` |  72,167,223 |  72,167,223 |  72,167,223 |  72,167,223 |
| `total_cycles        ` |  1,925,096 |  1,925,096 |  1,925,096 |  1,925,096 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+0.7%])</span> 295 | <span style='color: red'>(+2 [+0.7%])</span> 295 | <span style='color: red'>(+2 [+0.7%])</span> 295 | <span style='color: red'>(+2 [+0.7%])</span> 295 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-29 [-2.1%])</span> 1,358 | <span style='color: green'>(-29 [-2.1%])</span> 1,358 | <span style='color: green'>(-29 [-2.1%])</span> 1,358 | <span style='color: green'>(-29 [-2.1%])</span> 1,358 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+78 [+1.4%])</span> 5,480 | <span style='color: red'>(+78 [+1.4%])</span> 5,480 | <span style='color: red'>(+78 [+1.4%])</span> 5,480 | <span style='color: red'>(+78 [+1.4%])</span> 5,480 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+6 [+0.5%])</span> 1,103 | <span style='color: red'>(+6 [+0.5%])</span> 1,103 | <span style='color: red'>(+6 [+0.5%])</span> 1,103 | <span style='color: red'>(+6 [+0.5%])</span> 1,103 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+1.6%])</span> 129 | <span style='color: red'>(+2 [+1.6%])</span> 129 | <span style='color: red'>(+2 [+1.6%])</span> 129 | <span style='color: red'>(+2 [+1.6%])</span> 129 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-13 [-1.3%])</span> 988 | <span style='color: green'>(-13 [-1.3%])</span> 988 | <span style='color: green'>(-13 [-1.3%])</span> 988 | <span style='color: green'>(-13 [-1.3%])</span> 988 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+32 [+4.2%])</span> 798 | <span style='color: red'>(+32 [+4.2%])</span> 798 | <span style='color: red'>(+32 [+4.2%])</span> 798 | <span style='color: red'>(+32 [+4.2%])</span> 798 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.2%])</span> 1,094 | <span style='color: red'>(+2 [+0.2%])</span> 1,094 | <span style='color: red'>(+2 [+0.2%])</span> 1,094 | <span style='color: red'>(+2 [+0.2%])</span> 1,094 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+49 [+3.7%])</span> 1,364 | <span style='color: red'>(+49 [+3.7%])</span> 1,364 | <span style='color: red'>(+49 [+3.7%])</span> 1,364 | <span style='color: red'>(+49 [+3.7%])</span> 1,364 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 402 | 4 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| fibonacci_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| fibonacci_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| fibonacci_program | PhantomAir | 4 | 3 | 4 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 4 | 19 | 21 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 19 | 30 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 17 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 23 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 21 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 27 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 72 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 11 | 12 | 
| fibonacci_program | VmConnectorAir | 4 | 3 | 8 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 11 | 
| leaf | FriReducedOpeningAir | 4 | 31 | 53 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 176 | 555 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 11 | 20 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 17 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 17 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 216 | 399 | 20,152,320 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 25 | 25,690,112 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 34 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 8 | 6 | 28 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 20 | 32 | 208 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 20 | 28 | 768 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 40 | 1,088 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 21 | 296 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,358 | 7,133 | 1,925,096 | 160,482,264 | 5,480 | 798 | 1,094 | 988 | 1,364 | 1,103 | 72,167,223 | 129 | 295 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 808 | 5,307 | 1,500,095 | 122,458,476 | 4,188 | 523 | 781 | 812 | 1,062 | 870 | 51,485,080 | 136 | 311 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ba15b51a816c1b51fc8d4b27c6562efdbd49d16d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12978466023)
