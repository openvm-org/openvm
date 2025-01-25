| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-2.2%])</span> 12.09 | <span style='color: green'>(-0 [-2.2%])</span> 12.09 |
| fibonacci_program | <span style='color: green'>(-0 [-7.8%])</span> 5.06 | <span style='color: green'>(-0 [-7.8%])</span> 5.06 |
| leaf | <span style='color: red'>(+0 [+2.2%])</span> 7.03 | <span style='color: red'>(+0 [+2.2%])</span> 7.03 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-428 [-7.8%])</span> 5,058 | <span style='color: green'>(-428 [-7.8%])</span> 5,058 | <span style='color: green'>(-428 [-7.8%])</span> 5,058 | <span style='color: green'>(-428 [-7.8%])</span> 5,058 |
| `main_cells_used     ` |  51,487,838 |  51,487,838 |  51,487,838 |  51,487,838 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+11 [+3.5%])</span> 322 | <span style='color: red'>(+11 [+3.5%])</span> 322 | <span style='color: red'>(+11 [+3.5%])</span> 322 | <span style='color: red'>(+11 [+3.5%])</span> 322 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-17 [-2.1%])</span> 808 | <span style='color: green'>(-17 [-2.1%])</span> 808 | <span style='color: green'>(-17 [-2.1%])</span> 808 | <span style='color: green'>(-17 [-2.1%])</span> 808 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-422 [-9.7%])</span> 3,928 | <span style='color: green'>(-422 [-9.7%])</span> 3,928 | <span style='color: green'>(-422 [-9.7%])</span> 3,928 | <span style='color: green'>(-422 [-9.7%])</span> 3,928 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+9 [+1.1%])</span> 796 | <span style='color: red'>(+9 [+1.1%])</span> 796 | <span style='color: red'>(+9 [+1.1%])</span> 796 | <span style='color: red'>(+9 [+1.1%])</span> 796 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-32 [-18.2%])</span> 144 | <span style='color: green'>(-32 [-18.2%])</span> 144 | <span style='color: green'>(-32 [-18.2%])</span> 144 | <span style='color: green'>(-32 [-18.2%])</span> 144 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-365 [-23.4%])</span> 1,197 | <span style='color: green'>(-365 [-23.4%])</span> 1,197 | <span style='color: green'>(-365 [-23.4%])</span> 1,197 | <span style='color: green'>(-365 [-23.4%])</span> 1,197 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-1.7%])</span> 292 | <span style='color: green'>(-5 [-1.7%])</span> 292 | <span style='color: green'>(-5 [-1.7%])</span> 292 | <span style='color: green'>(-5 [-1.7%])</span> 292 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-16 [-3.0%])</span> 514 | <span style='color: green'>(-16 [-3.0%])</span> 514 | <span style='color: green'>(-16 [-3.0%])</span> 514 | <span style='color: green'>(-16 [-3.0%])</span> 514 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-11 [-1.1%])</span> 983 | <span style='color: green'>(-11 [-1.1%])</span> 983 | <span style='color: green'>(-11 [-1.1%])</span> 983 | <span style='color: green'>(-11 [-1.1%])</span> 983 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+150 [+2.2%])</span> 7,027 | <span style='color: red'>(+150 [+2.2%])</span> 7,027 | <span style='color: red'>(+150 [+2.2%])</span> 7,027 | <span style='color: red'>(+150 [+2.2%])</span> 7,027 |
| `main_cells_used     ` | <span style='color: green'>(-1004674 [-1.4%])</span> 69,687,067 | <span style='color: green'>(-1004674 [-1.4%])</span> 69,687,067 | <span style='color: green'>(-1004674 [-1.4%])</span> 69,687,067 | <span style='color: green'>(-1004674 [-1.4%])</span> 69,687,067 |
| `total_cycles        ` | <span style='color: green'>(-17622 [-1.0%])</span> 1,815,015 | <span style='color: green'>(-17622 [-1.0%])</span> 1,815,015 | <span style='color: green'>(-17622 [-1.0%])</span> 1,815,015 | <span style='color: green'>(-17622 [-1.0%])</span> 1,815,015 |
| `execute_time_ms     ` | <span style='color: red'>(+2 [+0.5%])</span> 372 | <span style='color: red'>(+2 [+0.5%])</span> 372 | <span style='color: red'>(+2 [+0.5%])</span> 372 | <span style='color: red'>(+2 [+0.5%])</span> 372 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+15 [+1.2%])</span> 1,241 | <span style='color: red'>(+15 [+1.2%])</span> 1,241 | <span style='color: red'>(+15 [+1.2%])</span> 1,241 | <span style='color: red'>(+15 [+1.2%])</span> 1,241 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+133 [+2.5%])</span> 5,414 | <span style='color: red'>(+133 [+2.5%])</span> 5,414 | <span style='color: red'>(+133 [+2.5%])</span> 5,414 | <span style='color: red'>(+133 [+2.5%])</span> 5,414 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+84 [+8.1%])</span> 1,122 | <span style='color: red'>(+84 [+8.1%])</span> 1,122 | <span style='color: red'>(+84 [+8.1%])</span> 1,122 | <span style='color: red'>(+84 [+8.1%])</span> 1,122 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-4 [-3.0%])</span> 130 | <span style='color: green'>(-4 [-3.0%])</span> 130 | <span style='color: green'>(-4 [-3.0%])</span> 130 | <span style='color: green'>(-4 [-3.0%])</span> 130 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-120 [-10.6%])</span> 1,012 | <span style='color: green'>(-120 [-10.6%])</span> 1,012 | <span style='color: green'>(-120 [-10.6%])</span> 1,012 | <span style='color: green'>(-120 [-10.6%])</span> 1,012 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+68 [+9.6%])</span> 773 | <span style='color: red'>(+68 [+9.6%])</span> 773 | <span style='color: red'>(+68 [+9.6%])</span> 773 | <span style='color: red'>(+68 [+9.6%])</span> 773 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+80 [+8.1%])</span> 1,071 | <span style='color: red'>(+80 [+8.1%])</span> 1,071 | <span style='color: red'>(+80 [+8.1%])</span> 1,071 | <span style='color: red'>(+80 [+8.1%])</span> 1,071 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+24 [+1.9%])</span> 1,302 | <span style='color: red'>(+24 [+1.9%])</span> 1,302 | <span style='color: red'>(+24 [+1.9%])</span> 1,302 | <span style='color: red'>(+24 [+1.9%])</span> 1,302 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 368 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 36 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 80 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 
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
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 40 | 2,944 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,241 | 7,027 | 1,815,015 | 160,482,264 | 5,414 | 773 | 1,071 | 1,012 | 1,302 | 1,122 | 69,687,067 | 130 | 372 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 808 | 5,058 | 1,500,137 | 160,738,014 | 3,928 | 292 | 514 | 1,197 | 983 | 796 | 51,487,838 | 144 | 322 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/cadd0d1f57c845f80378b02baed0424e3c2f394d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12968961704)
