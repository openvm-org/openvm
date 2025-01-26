| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.5%])</span> 19.38 | <span style='color: red'>(+0 [+0.5%])</span> 19.38 |
| fibonacci_program | <span style='color: green'>(-0 [-0.5%])</span> 6.07 | <span style='color: green'>(-0 [-0.5%])</span> 6.07 |
| leaf | <span style='color: red'>(+0 [+0.9%])</span> 13.30 | <span style='color: red'>(+0 [+0.9%])</span> 13.30 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-28 [-0.5%])</span> 6,073 | <span style='color: green'>(-28 [-0.5%])</span> 6,073 | <span style='color: green'>(-28 [-0.5%])</span> 6,073 | <span style='color: green'>(-28 [-0.5%])</span> 6,073 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+3 [+1.0%])</span> 309 | <span style='color: red'>(+3 [+1.0%])</span> 309 | <span style='color: red'>(+3 [+1.0%])</span> 309 | <span style='color: red'>(+3 [+1.0%])</span> 309 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-3 [-0.4%])</span> 816 | <span style='color: green'>(-3 [-0.4%])</span> 816 | <span style='color: green'>(-3 [-0.4%])</span> 816 | <span style='color: green'>(-3 [-0.4%])</span> 816 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-28 [-0.6%])</span> 4,948 | <span style='color: green'>(-28 [-0.6%])</span> 4,948 | <span style='color: green'>(-28 [-0.6%])</span> 4,948 | <span style='color: green'>(-28 [-0.6%])</span> 4,948 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-2 [-0.2%])</span> 808 | <span style='color: green'>(-2 [-0.2%])</span> 808 | <span style='color: green'>(-2 [-0.2%])</span> 808 | <span style='color: green'>(-2 [-0.2%])</span> 808 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-5 [-2.7%])</span> 177 | <span style='color: green'>(-5 [-2.7%])</span> 177 | <span style='color: green'>(-5 [-2.7%])</span> 177 | <span style='color: green'>(-5 [-2.7%])</span> 177 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-10 [-0.6%])</span> 1,604 | <span style='color: green'>(-10 [-0.6%])</span> 1,604 | <span style='color: green'>(-10 [-0.6%])</span> 1,604 | <span style='color: green'>(-10 [-0.6%])</span> 1,604 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-0.4%])</span> 782 | <span style='color: green'>(-3 [-0.4%])</span> 782 | <span style='color: green'>(-3 [-0.4%])</span> 782 | <span style='color: green'>(-3 [-0.4%])</span> 782 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-6 [-1.2%])</span> 502 | <span style='color: green'>(-6 [-1.2%])</span> 502 | <span style='color: green'>(-6 [-1.2%])</span> 502 | <span style='color: green'>(-6 [-1.2%])</span> 502 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2 [-0.2%])</span> 1,072 | <span style='color: green'>(-2 [-0.2%])</span> 1,072 | <span style='color: green'>(-2 [-0.2%])</span> 1,072 | <span style='color: green'>(-2 [-0.2%])</span> 1,072 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+115 [+0.9%])</span> 13,302 | <span style='color: red'>(+115 [+0.9%])</span> 13,302 | <span style='color: red'>(+115 [+0.9%])</span> 13,302 | <span style='color: red'>(+115 [+0.9%])</span> 13,302 |
| `main_cells_used     ` |  110,740,008 |  110,740,008 |  110,740,008 |  110,740,008 |
| `total_cycles        ` |  3,086,490 |  3,086,490 |  3,086,490 |  3,086,490 |
| `execute_time_ms     ` | <span style='color: red'>(+50 [+9.0%])</span> 605 | <span style='color: red'>(+50 [+9.0%])</span> 605 | <span style='color: red'>(+50 [+9.0%])</span> 605 | <span style='color: red'>(+50 [+9.0%])</span> 605 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-15 [-0.8%])</span> 1,772 | <span style='color: green'>(-15 [-0.8%])</span> 1,772 | <span style='color: green'>(-15 [-0.8%])</span> 1,772 | <span style='color: green'>(-15 [-0.8%])</span> 1,772 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+80 [+0.7%])</span> 10,925 | <span style='color: red'>(+80 [+0.7%])</span> 10,925 | <span style='color: red'>(+80 [+0.7%])</span> 10,925 | <span style='color: red'>(+80 [+0.7%])</span> 10,925 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.2%])</span> 1,980 | <span style='color: green'>(-4 [-0.2%])</span> 1,980 | <span style='color: green'>(-4 [-0.2%])</span> 1,980 | <span style='color: green'>(-4 [-0.2%])</span> 1,980 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.4%])</span> 239 | <span style='color: green'>(-1 [-0.4%])</span> 239 | <span style='color: green'>(-1 [-0.4%])</span> 239 | <span style='color: green'>(-1 [-0.4%])</span> 239 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+34 [+1.7%])</span> 2,007 | <span style='color: red'>(+34 [+1.7%])</span> 2,007 | <span style='color: red'>(+34 [+1.7%])</span> 2,007 | <span style='color: red'>(+34 [+1.7%])</span> 2,007 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+18 [+0.7%])</span> 2,492 | <span style='color: red'>(+18 [+0.7%])</span> 2,492 | <span style='color: red'>(+18 [+0.7%])</span> 2,492 | <span style='color: red'>(+18 [+0.7%])</span> 2,492 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+16 [+0.9%])</span> 1,864 | <span style='color: red'>(+16 [+0.9%])</span> 1,864 | <span style='color: red'>(+16 [+0.9%])</span> 1,864 | <span style='color: red'>(+16 [+0.9%])</span> 1,864 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+16 [+0.7%])</span> 2,340 | <span style='color: red'>(+16 [+0.7%])</span> 2,340 | <span style='color: red'>(+16 [+0.7%])</span> 2,340 | <span style='color: red'>(+16 [+0.7%])</span> 2,340 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 344 | 5 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 31 | 302 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 23 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 7 | 6 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,772 | 13,302 | 3,086,490 | 301,730,264 | 10,925 | 2,492 | 1,864 | 2,007 | 2,340 | 1,980 | 110,740,008 | 239 | 605 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 816 | 6,073 | 1,500,137 | 197,453,854 | 4,948 | 782 | 502 | 1,604 | 1,072 | 808 | 51,505,102 | 177 | 309 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4d21ab35d4ac419a4bf5ef2e5b938bc30dcdafca

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12824482434)