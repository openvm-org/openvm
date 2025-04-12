| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-0.9%])</span> 6.46 | <span style='color: green'>(-0 [-0.9%])</span> 6.46 |
| fibonacci_program | <span style='color: green'>(-0 [-1.0%])</span> 2.69 | <span style='color: green'>(-0 [-1.0%])</span> 2.69 |
| leaf | <span style='color: green'>(-0 [-0.8%])</span> 3.78 | <span style='color: green'>(-0 [-0.8%])</span> 3.78 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-27 [-1.0%])</span> 2,688 | <span style='color: green'>(-27 [-1.0%])</span> 2,688 | <span style='color: green'>(-27 [-1.0%])</span> 2,688 | <span style='color: green'>(-27 [-1.0%])</span> 2,688 |
| `main_cells_used     ` |  50,589,503 |  50,589,503 |  50,589,503 |  50,589,503 |
| `total_cycles        ` |  1,500,277 |  1,500,277 |  1,500,277 |  1,500,277 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 | <span style='color: red'>(+1 [+0.4%])</span> 233 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-14 [-2.5%])</span> 554 | <span style='color: green'>(-14 [-2.5%])</span> 554 | <span style='color: green'>(-14 [-2.5%])</span> 554 | <span style='color: green'>(-14 [-2.5%])</span> 554 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-14 [-0.7%])</span> 1,901 | <span style='color: green'>(-14 [-0.7%])</span> 1,901 | <span style='color: green'>(-14 [-0.7%])</span> 1,901 | <span style='color: green'>(-14 [-0.7%])</span> 1,901 |
| `main_trace_commit_time_ms` |  330 |  330 |  330 |  330 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-12 [-9.4%])</span> 115 | <span style='color: green'>(-12 [-9.4%])</span> 115 | <span style='color: green'>(-12 [-9.4%])</span> 115 | <span style='color: green'>(-12 [-9.4%])</span> 115 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-6 [-1.3%])</span> 469 | <span style='color: green'>(-6 [-1.3%])</span> 469 | <span style='color: green'>(-6 [-1.3%])</span> 469 | <span style='color: green'>(-6 [-1.3%])</span> 469 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+0.8%])</span> 259 | <span style='color: red'>(+2 [+0.8%])</span> 259 | <span style='color: red'>(+2 [+0.8%])</span> 259 | <span style='color: red'>(+2 [+0.8%])</span> 259 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.9%])</span> 222 | <span style='color: red'>(+2 [+0.9%])</span> 222 | <span style='color: red'>(+2 [+0.9%])</span> 222 | <span style='color: red'>(+2 [+0.9%])</span> 222 |
| `pcs_opening_time_ms ` |  502 |  502 |  502 |  502 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-29 [-0.8%])</span> 3,776 | <span style='color: green'>(-29 [-0.8%])</span> 3,776 | <span style='color: green'>(-29 [-0.8%])</span> 3,776 | <span style='color: green'>(-29 [-0.8%])</span> 3,776 |
| `main_cells_used     ` |  70,284,620 |  70,284,620 |  70,284,620 |  70,284,620 |
| `total_cycles        ` |  1,263,382 |  1,263,382 |  1,263,382 |  1,263,382 |
| `execute_time_ms     ` | <span style='color: red'>(+8 [+1.6%])</span> 524 | <span style='color: red'>(+8 [+1.6%])</span> 524 | <span style='color: red'>(+8 [+1.6%])</span> 524 | <span style='color: red'>(+8 [+1.6%])</span> 524 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-24 [-3.5%])</span> 655 | <span style='color: green'>(-24 [-3.5%])</span> 655 | <span style='color: green'>(-24 [-3.5%])</span> 655 | <span style='color: green'>(-24 [-3.5%])</span> 655 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-13 [-0.5%])</span> 2,597 | <span style='color: green'>(-13 [-0.5%])</span> 2,597 | <span style='color: green'>(-13 [-0.5%])</span> 2,597 | <span style='color: green'>(-13 [-0.5%])</span> 2,597 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.2%])</span> 431 | <span style='color: green'>(-1 [-0.2%])</span> 431 | <span style='color: green'>(-1 [-0.2%])</span> 431 | <span style='color: green'>(-1 [-0.2%])</span> 431 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+0.6%])</span> 164 | <span style='color: red'>(+1 [+0.6%])</span> 164 | <span style='color: red'>(+1 [+0.6%])</span> 164 | <span style='color: red'>(+1 [+0.6%])</span> 164 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+3 [+0.4%])</span> 732 | <span style='color: red'>(+3 [+0.4%])</span> 732 | <span style='color: red'>(+3 [+0.4%])</span> 732 | <span style='color: red'>(+3 [+0.4%])</span> 732 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-0.5%])</span> 375 | <span style='color: green'>(-2 [-0.5%])</span> 375 | <span style='color: green'>(-2 [-0.5%])</span> 375 | <span style='color: green'>(-2 [-0.5%])</span> 375 |
| `quotient_poly_commit_time_ms` |  279 |  279 |  279 |  279 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-13 [-2.1%])</span> 611 | <span style='color: green'>(-13 [-2.1%])</span> 611 | <span style='color: green'>(-13 [-2.1%])</span> 611 | <span style='color: green'>(-13 [-2.1%])</span> 611 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 260 | 5 | 

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

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 128 |  | 16 | 17 | 4,224 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 16 | 32 | 24,576 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 128 |  | 12 | 20 | 4,096 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32 |  | 36 | 28 | 2,048 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 128 |  | 52 | 41 | 11,904 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 20 | 768 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 655 | 3,776 | 1,263,382 | 253,173,226 | 2,597 | 375 | 279 | 732 | 611 | 431 | 70,284,620 | 164 | 524 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,313,162 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 554 | 2,688 | 1,500,277 | 160,837,996 | 1,901 | 259 | 222 | 469 | 502 | 330 | 50,589,503 | 115 | 233 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,542 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,749,400 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,271 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,749,532 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 1,664 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 640 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,100 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,535,101 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f383aaec3ba302b19a8f27b6c84f2c0564f62e9d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14423536055)
