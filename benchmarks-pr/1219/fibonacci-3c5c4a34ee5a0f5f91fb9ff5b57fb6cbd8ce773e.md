| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-6 [-29.3%])</span> 13.78 | <span style='color: green'>(-6 [-29.3%])</span> 13.78 |
| fibonacci_program | <span style='color: red'>(+0 [+0.8%])</span> 6.16 | <span style='color: red'>(+0 [+0.8%])</span> 6.16 |
| leaf | <span style='color: green'>(-6 [-43.0%])</span> 7.63 | <span style='color: green'>(-6 [-43.0%])</span> 7.63 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+48 [+0.8%])</span> 6,157 | <span style='color: red'>(+48 [+0.8%])</span> 6,157 | <span style='color: red'>(+48 [+0.8%])</span> 6,157 | <span style='color: red'>(+48 [+0.8%])</span> 6,157 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+0.6%])</span> 311 | <span style='color: red'>(+2 [+0.6%])</span> 311 | <span style='color: red'>(+2 [+0.6%])</span> 311 | <span style='color: red'>(+2 [+0.6%])</span> 311 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.2%])</span> 819 | <span style='color: red'>(+2 [+0.2%])</span> 819 | <span style='color: red'>(+2 [+0.2%])</span> 819 | <span style='color: red'>(+2 [+0.2%])</span> 819 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+44 [+0.9%])</span> 5,027 | <span style='color: red'>(+44 [+0.9%])</span> 5,027 | <span style='color: red'>(+44 [+0.9%])</span> 5,027 | <span style='color: red'>(+44 [+0.9%])</span> 5,027 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+12 [+1.5%])</span> 823 | <span style='color: red'>(+12 [+1.5%])</span> 823 | <span style='color: red'>(+12 [+1.5%])</span> 823 | <span style='color: red'>(+12 [+1.5%])</span> 823 |
| `generate_perm_trace_time_ms` |  179 |  179 |  179 |  179 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-26 [-1.6%])</span> 1,620 | <span style='color: green'>(-26 [-1.6%])</span> 1,620 | <span style='color: green'>(-26 [-1.6%])</span> 1,620 | <span style='color: green'>(-26 [-1.6%])</span> 1,620 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.1%])</span> 790 | <span style='color: red'>(+1 [+0.1%])</span> 790 | <span style='color: red'>(+1 [+0.1%])</span> 790 | <span style='color: red'>(+1 [+0.1%])</span> 790 |
| `quotient_poly_commit_time_ms` |  505 |  505 |  505 |  505 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+57 [+5.4%])</span> 1,107 | <span style='color: red'>(+57 [+5.4%])</span> 1,107 | <span style='color: red'>(+57 [+5.4%])</span> 1,107 | <span style='color: red'>(+57 [+5.4%])</span> 1,107 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-5759 [-43.0%])</span> 7,626 | <span style='color: green'>(-5759 [-43.0%])</span> 7,626 | <span style='color: green'>(-5759 [-43.0%])</span> 7,626 | <span style='color: green'>(-5759 [-43.0%])</span> 7,626 |
| `main_cells_used     ` | <span style='color: green'>(-31460200 [-28.4%])</span> 79,277,308 | <span style='color: green'>(-31460200 [-28.4%])</span> 79,277,308 | <span style='color: green'>(-31460200 [-28.4%])</span> 79,277,308 | <span style='color: green'>(-31460200 [-28.4%])</span> 79,277,308 |
| `total_cycles        ` | <span style='color: green'>(-1078719 [-34.9%])</span> 2,008,383 | <span style='color: green'>(-1078719 [-34.9%])</span> 2,008,383 | <span style='color: green'>(-1078719 [-34.9%])</span> 2,008,383 | <span style='color: green'>(-1078719 [-34.9%])</span> 2,008,383 |
| `execute_time_ms     ` | <span style='color: green'>(-199 [-32.6%])</span> 411 | <span style='color: green'>(-199 [-32.6%])</span> 411 | <span style='color: green'>(-199 [-32.6%])</span> 411 | <span style='color: green'>(-199 [-32.6%])</span> 411 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-508 [-27.8%])</span> 1,317 | <span style='color: green'>(-508 [-27.8%])</span> 1,317 | <span style='color: green'>(-508 [-27.8%])</span> 1,317 | <span style='color: green'>(-508 [-27.8%])</span> 1,317 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-5052 [-46.1%])</span> 5,898 | <span style='color: green'>(-5052 [-46.1%])</span> 5,898 | <span style='color: green'>(-5052 [-46.1%])</span> 5,898 | <span style='color: green'>(-5052 [-46.1%])</span> 5,898 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-877 [-44.4%])</span> 1,099 | <span style='color: green'>(-877 [-44.4%])</span> 1,099 | <span style='color: green'>(-877 [-44.4%])</span> 1,099 | <span style='color: green'>(-877 [-44.4%])</span> 1,099 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-105 [-43.2%])</span> 138 | <span style='color: green'>(-105 [-43.2%])</span> 138 | <span style='color: green'>(-105 [-43.2%])</span> 138 | <span style='color: green'>(-105 [-43.2%])</span> 138 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-838 [-41.6%])</span> 1,177 | <span style='color: green'>(-838 [-41.6%])</span> 1,177 | <span style='color: green'>(-838 [-41.6%])</span> 1,177 | <span style='color: green'>(-838 [-41.6%])</span> 1,177 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1262 [-50.7%])</span> 1,229 | <span style='color: green'>(-1262 [-50.7%])</span> 1,229 | <span style='color: green'>(-1262 [-50.7%])</span> 1,229 | <span style='color: green'>(-1262 [-50.7%])</span> 1,229 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-860 [-46.4%])</span> 992 | <span style='color: green'>(-860 [-46.4%])</span> 992 | <span style='color: green'>(-860 [-46.4%])</span> 992 | <span style='color: green'>(-860 [-46.4%])</span> 992 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-1109 [-46.8%])</span> 1,261 | <span style='color: green'>(-1109 [-46.8%])</span> 1,261 | <span style='color: green'>(-1109 [-46.8%])</span> 1,261 | <span style='color: green'>(-1109 [-46.8%])</span> 1,261 |



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
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
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
| leaf | 0 | 1,317 | 7,626 | 2,008,383 | 188,271,576 | 5,898 | 1,229 | 992 | 1,177 | 1,261 | 1,099 | 79,277,308 | 138 | 411 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 819 | 6,157 | 1,500,137 | 197,453,854 | 5,027 | 790 | 505 | 1,620 | 1,107 | 823 | 51,505,102 | 179 | 311 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/3c5c4a34ee5a0f5f91fb9ff5b57fb6cbd8ce773e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12881616966)
