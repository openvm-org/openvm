| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-7.3%])</span> 11.03 | <span style='color: green'>(-1 [-7.3%])</span> 11.03 |
| fibonacci_program | <span style='color: red'>(+0 [+0.4%])</span> 4.100 | <span style='color: red'>(+0 [+0.4%])</span> 4.100 |
| leaf | <span style='color: green'>(-1 [-12.8%])</span> 6.03 | <span style='color: green'>(-1 [-12.8%])</span> 6.03 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+21 [+0.4%])</span> 4,998 | <span style='color: red'>(+21 [+0.4%])</span> 4,998 | <span style='color: red'>(+21 [+0.4%])</span> 4,998 | <span style='color: red'>(+21 [+0.4%])</span> 4,998 |
| `main_cells_used     ` |  51,485,080 |  51,485,080 |  51,485,080 |  51,485,080 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-0.6%])</span> 309 | <span style='color: green'>(-2 [-0.6%])</span> 309 | <span style='color: green'>(-2 [-0.6%])</span> 309 | <span style='color: green'>(-2 [-0.6%])</span> 309 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1 [-0.1%])</span> 671 | <span style='color: green'>(-1 [-0.1%])</span> 671 | <span style='color: green'>(-1 [-0.1%])</span> 671 | <span style='color: green'>(-1 [-0.1%])</span> 671 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+24 [+0.6%])</span> 4,018 | <span style='color: red'>(+24 [+0.6%])</span> 4,018 | <span style='color: red'>(+24 [+0.6%])</span> 4,018 | <span style='color: red'>(+24 [+0.6%])</span> 4,018 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+7 [+0.9%])</span> 801 | <span style='color: red'>(+7 [+0.9%])</span> 801 | <span style='color: red'>(+7 [+0.9%])</span> 801 | <span style='color: red'>(+7 [+0.9%])</span> 801 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+4.3%])</span> 146 | <span style='color: red'>(+6 [+4.3%])</span> 146 | <span style='color: red'>(+6 [+4.3%])</span> 146 | <span style='color: red'>(+6 [+4.3%])</span> 146 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+10 [+1.4%])</span> 750 | <span style='color: red'>(+10 [+1.4%])</span> 750 | <span style='color: red'>(+10 [+1.4%])</span> 750 | <span style='color: red'>(+10 [+1.4%])</span> 750 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+13 [+2.5%])</span> 529 | <span style='color: red'>(+13 [+2.5%])</span> 529 | <span style='color: red'>(+13 [+2.5%])</span> 529 | <span style='color: red'>(+13 [+2.5%])</span> 529 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+12 [+1.7%])</span> 736 | <span style='color: red'>(+12 [+1.7%])</span> 736 | <span style='color: red'>(+12 [+1.7%])</span> 736 | <span style='color: red'>(+12 [+1.7%])</span> 736 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-24 [-2.2%])</span> 1,053 | <span style='color: green'>(-24 [-2.2%])</span> 1,053 | <span style='color: green'>(-24 [-2.2%])</span> 1,053 | <span style='color: green'>(-24 [-2.2%])</span> 1,053 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-885 [-12.8%])</span> 6,031 | <span style='color: green'>(-885 [-12.8%])</span> 6,031 | <span style='color: green'>(-885 [-12.8%])</span> 6,031 | <span style='color: green'>(-885 [-12.8%])</span> 6,031 |
| `main_cells_used     ` | <span style='color: green'>(-18588282 [-26.8%])</span> 50,832,298 | <span style='color: green'>(-18588282 [-26.8%])</span> 50,832,298 | <span style='color: green'>(-18588282 [-26.8%])</span> 50,832,298 | <span style='color: green'>(-18588282 [-26.8%])</span> 50,832,298 |
| `total_cycles        ` | <span style='color: green'>(-676522 [-35.1%])</span> 1,248,599 | <span style='color: green'>(-676522 [-35.1%])</span> 1,248,599 | <span style='color: green'>(-676522 [-35.1%])</span> 1,248,599 | <span style='color: green'>(-676522 [-35.1%])</span> 1,248,599 |
| `execute_time_ms     ` | <span style='color: red'>(+23 [+7.7%])</span> 321 | <span style='color: red'>(+23 [+7.7%])</span> 321 | <span style='color: red'>(+23 [+7.7%])</span> 321 | <span style='color: red'>(+23 [+7.7%])</span> 321 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-421 [-34.2%])</span> 809 | <span style='color: green'>(-421 [-34.2%])</span> 809 | <span style='color: green'>(-421 [-34.2%])</span> 809 | <span style='color: green'>(-421 [-34.2%])</span> 809 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-487 [-9.0%])</span> 4,901 | <span style='color: green'>(-487 [-9.0%])</span> 4,901 | <span style='color: green'>(-487 [-9.0%])</span> 4,901 | <span style='color: green'>(-487 [-9.0%])</span> 4,901 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-100 [-9.3%])</span> 970 | <span style='color: green'>(-100 [-9.3%])</span> 970 | <span style='color: green'>(-100 [-9.3%])</span> 970 | <span style='color: green'>(-100 [-9.3%])</span> 970 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-12 [-9.2%])</span> 118 | <span style='color: green'>(-12 [-9.2%])</span> 118 | <span style='color: green'>(-12 [-9.2%])</span> 118 | <span style='color: green'>(-12 [-9.2%])</span> 118 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-136 [-13.2%])</span> 898 | <span style='color: green'>(-136 [-13.2%])</span> 898 | <span style='color: green'>(-136 [-13.2%])</span> 898 | <span style='color: green'>(-136 [-13.2%])</span> 898 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-117 [-14.9%])</span> 668 | <span style='color: green'>(-117 [-14.9%])</span> 668 | <span style='color: green'>(-117 [-14.9%])</span> 668 | <span style='color: green'>(-117 [-14.9%])</span> 668 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-105 [-9.7%])</span> 979 | <span style='color: green'>(-105 [-9.7%])</span> 979 | <span style='color: green'>(-105 [-9.7%])</span> 979 | <span style='color: green'>(-105 [-9.7%])</span> 979 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-17 [-1.3%])</span> 1,264 | <span style='color: green'>(-17 [-1.3%])</span> 1,264 | <span style='color: green'>(-17 [-1.3%])</span> 1,264 | <span style='color: green'>(-17 [-1.3%])</span> 1,264 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 392 | 7 | 

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
| leaf | FriReducedOpeningAir | 4 | 31 | 52 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 136 | 530 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 11 | 22 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | PhantomAir | 0 | 16,384 |  | 8 | 6 | 229,376 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 

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
| leaf | 0 | 809 | 6,031 | 1,248,599 | 139,379,672 | 4,901 | 668 | 979 | 898 | 1,264 | 970 | 50,832,298 | 118 | 321 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 671 | 4,998 | 1,500,095 | 122,458,476 | 4,018 | 529 | 736 | 750 | 1,053 | 801 | 51,485,080 | 146 | 309 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b42be8bb6fd2acc0830eb8833ae0b2c07c8eec69

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13097254898)
