| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.3%])</span> 13.63 | <span style='color: red'>(+0 [+0.3%])</span> 13.63 |
| fibonacci_program | <span style='color: green'>(-0 [-0.4%])</span> 6.01 | <span style='color: green'>(-0 [-0.4%])</span> 6.01 |
| leaf | <span style='color: red'>(+0 [+0.9%])</span> 7.62 | <span style='color: red'>(+0 [+0.9%])</span> 7.62 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-25 [-0.4%])</span> 6,011 | <span style='color: green'>(-25 [-0.4%])</span> 6,011 | <span style='color: green'>(-25 [-0.4%])</span> 6,011 | <span style='color: green'>(-25 [-0.4%])</span> 6,011 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+3 [+1.0%])</span> 312 | <span style='color: red'>(+3 [+1.0%])</span> 312 | <span style='color: red'>(+3 [+1.0%])</span> 312 | <span style='color: red'>(+3 [+1.0%])</span> 312 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2 [-0.2%])</span> 799 | <span style='color: green'>(-2 [-0.2%])</span> 799 | <span style='color: green'>(-2 [-0.2%])</span> 799 | <span style='color: green'>(-2 [-0.2%])</span> 799 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-26 [-0.5%])</span> 4,900 | <span style='color: green'>(-26 [-0.5%])</span> 4,900 | <span style='color: green'>(-26 [-0.5%])</span> 4,900 | <span style='color: green'>(-26 [-0.5%])</span> 4,900 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+5 [+0.6%])</span> 798 | <span style='color: red'>(+5 [+0.6%])</span> 798 | <span style='color: red'>(+5 [+0.6%])</span> 798 | <span style='color: red'>(+5 [+0.6%])</span> 798 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.6%])</span> 178 | <span style='color: green'>(-1 [-0.6%])</span> 178 | <span style='color: green'>(-1 [-0.6%])</span> 178 | <span style='color: green'>(-1 [-0.6%])</span> 178 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-17 [-1.1%])</span> 1,584 | <span style='color: green'>(-17 [-1.1%])</span> 1,584 | <span style='color: green'>(-17 [-1.1%])</span> 1,584 | <span style='color: green'>(-17 [-1.1%])</span> 1,584 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-0.4%])</span> 775 | <span style='color: green'>(-3 [-0.4%])</span> 775 | <span style='color: green'>(-3 [-0.4%])</span> 775 | <span style='color: green'>(-3 [-0.4%])</span> 775 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-5 [-1.0%])</span> 495 | <span style='color: green'>(-5 [-1.0%])</span> 495 | <span style='color: green'>(-5 [-1.0%])</span> 495 | <span style='color: green'>(-5 [-1.0%])</span> 495 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-4 [-0.4%])</span> 1,068 | <span style='color: green'>(-4 [-0.4%])</span> 1,068 | <span style='color: green'>(-4 [-0.4%])</span> 1,068 | <span style='color: green'>(-4 [-0.4%])</span> 1,068 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 |
| `main_cells_used     ` | <span style='color: green'>(-5225391 [-6.9%])</span> 70,479,870 | <span style='color: green'>(-5225391 [-6.9%])</span> 70,479,870 | <span style='color: green'>(-5225391 [-6.9%])</span> 70,479,870 | <span style='color: green'>(-5225391 [-6.9%])</span> 70,479,870 |
| `total_cycles        ` | <span style='color: green'>(-16002 [-0.9%])</span> 1,823,257 | <span style='color: green'>(-16002 [-0.9%])</span> 1,823,257 | <span style='color: green'>(-16002 [-0.9%])</span> 1,823,257 | <span style='color: green'>(-16002 [-0.9%])</span> 1,823,257 |
| `execute_time_ms     ` | <span style='color: green'>(-39 [-10.2%])</span> 345 | <span style='color: green'>(-39 [-10.2%])</span> 345 | <span style='color: green'>(-39 [-10.2%])</span> 345 | <span style='color: green'>(-39 [-10.2%])</span> 345 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-62 [-5.0%])</span> 1,183 | <span style='color: green'>(-62 [-5.0%])</span> 1,183 | <span style='color: green'>(-62 [-5.0%])</span> 1,183 | <span style='color: green'>(-62 [-5.0%])</span> 1,183 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+169 [+2.9%])</span> 6,089 | <span style='color: red'>(+169 [+2.9%])</span> 6,089 | <span style='color: red'>(+169 [+2.9%])</span> 6,089 | <span style='color: red'>(+169 [+2.9%])</span> 6,089 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+22 [+2.0%])</span> 1,099 | <span style='color: red'>(+22 [+2.0%])</span> 1,099 | <span style='color: red'>(+22 [+2.0%])</span> 1,099 | <span style='color: red'>(+22 [+2.0%])</span> 1,099 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-15 [-1.3%])</span> 1,163 | <span style='color: green'>(-15 [-1.3%])</span> 1,163 | <span style='color: green'>(-15 [-1.3%])</span> 1,163 | <span style='color: green'>(-15 [-1.3%])</span> 1,163 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+152 [+12.1%])</span> 1,413 | <span style='color: red'>(+152 [+12.1%])</span> 1,413 | <span style='color: red'>(+152 [+12.1%])</span> 1,413 | <span style='color: red'>(+152 [+12.1%])</span> 1,413 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+6 [+0.6%])</span> 1,007 | <span style='color: red'>(+6 [+0.6%])</span> 1,007 | <span style='color: red'>(+6 [+0.6%])</span> 1,007 | <span style='color: red'>(+6 [+0.6%])</span> 1,007 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+5 [+0.4%])</span> 1,267 | <span style='color: red'>(+5 [+0.4%])</span> 1,267 | <span style='color: red'>(+5 [+0.4%])</span> 1,267 | <span style='color: red'>(+5 [+0.4%])</span> 1,267 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 338 | 5 | 

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
| leaf | FriReducedOpeningAir | 4 | 31 | 55 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 176 | 590 | 
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
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 48 | 26 | 9,699,328 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
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
| leaf | 0 | 1,183 | 7,617 | 1,823,257 | 181,586,904 | 6,089 | 1,413 | 1,007 | 1,163 | 1,267 | 1,099 | 70,479,870 | 136 | 345 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 799 | 6,011 | 1,500,137 | 197,453,854 | 4,900 | 775 | 495 | 1,584 | 1,068 | 798 | 51,505,102 | 178 | 312 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/259a25f832fed1aa7237dcaccddc48cd92a035ad

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12912604047)
