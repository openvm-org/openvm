| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+0.8%])</span> 13.69 | <span style='color: red'>(+0 [+0.8%])</span> 13.69 |
| fibonacci_program | <span style='color: red'>(+0 [+0.7%])</span> 6.08 | <span style='color: red'>(+0 [+0.7%])</span> 6.08 |
| leaf | <span style='color: red'>(+0 [+0.9%])</span> 7.62 | <span style='color: red'>(+0 [+0.9%])</span> 7.62 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+41 [+0.7%])</span> 6,077 | <span style='color: red'>(+41 [+0.7%])</span> 6,077 | <span style='color: red'>(+41 [+0.7%])</span> 6,077 | <span style='color: red'>(+41 [+0.7%])</span> 6,077 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.3%])</span> 310 | <span style='color: red'>(+1 [+0.3%])</span> 310 | <span style='color: red'>(+1 [+0.3%])</span> 310 | <span style='color: red'>(+1 [+0.3%])</span> 310 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+6 [+0.7%])</span> 807 | <span style='color: red'>(+6 [+0.7%])</span> 807 | <span style='color: red'>(+6 [+0.7%])</span> 807 | <span style='color: red'>(+6 [+0.7%])</span> 807 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+34 [+0.7%])</span> 4,960 | <span style='color: red'>(+34 [+0.7%])</span> 4,960 | <span style='color: red'>(+34 [+0.7%])</span> 4,960 | <span style='color: red'>(+34 [+0.7%])</span> 4,960 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+11 [+1.4%])</span> 804 | <span style='color: red'>(+11 [+1.4%])</span> 804 | <span style='color: red'>(+11 [+1.4%])</span> 804 | <span style='color: red'>(+11 [+1.4%])</span> 804 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 | <span style='color: green'>(-2 [-1.1%])</span> 177 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+8 [+0.5%])</span> 1,609 | <span style='color: red'>(+8 [+0.5%])</span> 1,609 | <span style='color: red'>(+8 [+0.5%])</span> 1,609 | <span style='color: red'>(+8 [+0.5%])</span> 1,609 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+13 [+1.7%])</span> 791 | <span style='color: red'>(+13 [+1.7%])</span> 791 | <span style='color: red'>(+13 [+1.7%])</span> 791 | <span style='color: red'>(+13 [+1.7%])</span> 791 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+12 [+2.4%])</span> 512 | <span style='color: red'>(+12 [+2.4%])</span> 512 | <span style='color: red'>(+12 [+2.4%])</span> 512 | <span style='color: red'>(+12 [+2.4%])</span> 512 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-8 [-0.7%])</span> 1,064 | <span style='color: green'>(-8 [-0.7%])</span> 1,064 | <span style='color: green'>(-8 [-0.7%])</span> 1,064 | <span style='color: green'>(-8 [-0.7%])</span> 1,064 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 | <span style='color: red'>(+68 [+0.9%])</span> 7,617 |
| `main_cells_used     ` | <span style='color: green'>(-5231341 [-6.9%])</span> 70,473,920 | <span style='color: green'>(-5231341 [-6.9%])</span> 70,473,920 | <span style='color: green'>(-5231341 [-6.9%])</span> 70,473,920 | <span style='color: green'>(-5231341 [-6.9%])</span> 70,473,920 |
| `total_cycles        ` | <span style='color: green'>(-16597 [-0.9%])</span> 1,822,662 | <span style='color: green'>(-16597 [-0.9%])</span> 1,822,662 | <span style='color: green'>(-16597 [-0.9%])</span> 1,822,662 | <span style='color: green'>(-16597 [-0.9%])</span> 1,822,662 |
| `execute_time_ms     ` | <span style='color: green'>(-34 [-8.9%])</span> 350 | <span style='color: green'>(-34 [-8.9%])</span> 350 | <span style='color: green'>(-34 [-8.9%])</span> 350 | <span style='color: green'>(-34 [-8.9%])</span> 350 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-47 [-3.8%])</span> 1,198 | <span style='color: green'>(-47 [-3.8%])</span> 1,198 | <span style='color: green'>(-47 [-3.8%])</span> 1,198 | <span style='color: green'>(-47 [-3.8%])</span> 1,198 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+149 [+2.5%])</span> 6,069 | <span style='color: red'>(+149 [+2.5%])</span> 6,069 | <span style='color: red'>(+149 [+2.5%])</span> 6,069 | <span style='color: red'>(+149 [+2.5%])</span> 6,069 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+45 [+4.2%])</span> 1,122 | <span style='color: red'>(+45 [+4.2%])</span> 1,122 | <span style='color: red'>(+45 [+4.2%])</span> 1,122 | <span style='color: red'>(+45 [+4.2%])</span> 1,122 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 | <span style='color: green'>(-2 [-1.4%])</span> 136 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+20 [+1.7%])</span> 1,198 | <span style='color: red'>(+20 [+1.7%])</span> 1,198 | <span style='color: red'>(+20 [+1.7%])</span> 1,198 | <span style='color: red'>(+20 [+1.7%])</span> 1,198 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+32 [+2.5%])</span> 1,293 | <span style='color: red'>(+32 [+2.5%])</span> 1,293 | <span style='color: red'>(+32 [+2.5%])</span> 1,293 | <span style='color: red'>(+32 [+2.5%])</span> 1,293 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+24 [+2.4%])</span> 1,025 | <span style='color: red'>(+24 [+2.4%])</span> 1,025 | <span style='color: red'>(+24 [+2.4%])</span> 1,025 | <span style='color: red'>(+24 [+2.4%])</span> 1,025 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+30 [+2.4%])</span> 1,292 | <span style='color: red'>(+30 [+2.4%])</span> 1,292 | <span style='color: red'>(+30 [+2.4%])</span> 1,292 | <span style='color: red'>(+30 [+2.4%])</span> 1,292 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 342 | 6 | 

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
| leaf | 0 | 1,198 | 7,617 | 1,822,662 | 181,586,904 | 6,069 | 1,293 | 1,025 | 1,198 | 1,292 | 1,122 | 70,473,920 | 136 | 350 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 807 | 6,077 | 1,500,137 | 197,453,854 | 4,960 | 791 | 512 | 1,609 | 1,064 | 804 | 51,505,102 | 177 | 310 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/ffff287dcfe9be42e651125539a22b3ca82e5c06

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12913459700)
