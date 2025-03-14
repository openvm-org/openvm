| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-0 [-1.1%])</span> 6.53 | <span style='color: green'>(-0 [-1.1%])</span> 6.53 |
| fibonacci_program | <span style='color: green'>(-0 [-2.8%])</span> 2.68 | <span style='color: green'>(-0 [-2.8%])</span> 2.68 |
| leaf | <span style='color: red'>(+0 [+0.1%])</span> 3.85 | <span style='color: red'>(+0 [+0.1%])</span> 3.85 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-78 [-2.8%])</span> 2,677 | <span style='color: green'>(-78 [-2.8%])</span> 2,677 | <span style='color: green'>(-78 [-2.8%])</span> 2,677 | <span style='color: green'>(-78 [-2.8%])</span> 2,677 |
| `main_cells_used     ` |  51,485,167 |  51,485,167 |  51,485,167 |  51,485,167 |
| `total_cycles        ` |  1,500,096 |  1,500,096 |  1,500,096 |  1,500,096 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.4%])</span> 236 | <span style='color: red'>(+1 [+0.4%])</span> 236 | <span style='color: red'>(+1 [+0.4%])</span> 236 | <span style='color: red'>(+1 [+0.4%])</span> 236 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+11 [+2.0%])</span> 555 | <span style='color: red'>(+11 [+2.0%])</span> 555 | <span style='color: red'>(+11 [+2.0%])</span> 555 | <span style='color: red'>(+11 [+2.0%])</span> 555 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-90 [-4.6%])</span> 1,886 | <span style='color: green'>(-90 [-4.6%])</span> 1,886 | <span style='color: green'>(-90 [-4.6%])</span> 1,886 | <span style='color: green'>(-90 [-4.6%])</span> 1,886 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-32 [-8.9%])</span> 327 | <span style='color: green'>(-32 [-8.9%])</span> 327 | <span style='color: green'>(-32 [-8.9%])</span> 327 | <span style='color: green'>(-32 [-8.9%])</span> 327 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+0.9%])</span> 118 | <span style='color: red'>(+1 [+0.9%])</span> 118 | <span style='color: red'>(+1 [+0.9%])</span> 118 | <span style='color: red'>(+1 [+0.9%])</span> 118 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-40 [-7.9%])</span> 467 | <span style='color: green'>(-40 [-7.9%])</span> 467 | <span style='color: green'>(-40 [-7.9%])</span> 467 | <span style='color: green'>(-40 [-7.9%])</span> 467 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-0.8%])</span> 251 | <span style='color: green'>(-2 [-0.8%])</span> 251 | <span style='color: green'>(-2 [-0.8%])</span> 251 | <span style='color: green'>(-2 [-0.8%])</span> 251 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-15 [-6.5%])</span> 217 | <span style='color: green'>(-15 [-6.5%])</span> 217 | <span style='color: green'>(-15 [-6.5%])</span> 217 | <span style='color: green'>(-15 [-6.5%])</span> 217 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1 [+0.2%])</span> 503 | <span style='color: red'>(+1 [+0.2%])</span> 503 | <span style='color: red'>(+1 [+0.2%])</span> 503 | <span style='color: red'>(+1 [+0.2%])</span> 503 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+5 [+0.1%])</span> 3,853 | <span style='color: red'>(+5 [+0.1%])</span> 3,853 | <span style='color: red'>(+5 [+0.1%])</span> 3,853 | <span style='color: red'>(+5 [+0.1%])</span> 3,853 |
| `main_cells_used     ` |  70,274,573 |  70,274,573 |  70,274,573 |  70,274,573 |
| `total_cycles        ` |  1,264,964 |  1,264,964 |  1,264,964 |  1,264,964 |
| `execute_time_ms     ` | <span style='color: green'>(-17 [-3.3%])</span> 504 | <span style='color: green'>(-17 [-3.3%])</span> 504 | <span style='color: green'>(-17 [-3.3%])</span> 504 | <span style='color: green'>(-17 [-3.3%])</span> 504 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+15 [+2.3%])</span> 663 | <span style='color: red'>(+15 [+2.3%])</span> 663 | <span style='color: red'>(+15 [+2.3%])</span> 663 | <span style='color: red'>(+15 [+2.3%])</span> 663 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+7 [+0.3%])</span> 2,686 | <span style='color: red'>(+7 [+0.3%])</span> 2,686 | <span style='color: red'>(+7 [+0.3%])</span> 2,686 | <span style='color: red'>(+7 [+0.3%])</span> 2,686 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-4 [-0.9%])</span> 464 | <span style='color: green'>(-4 [-0.9%])</span> 464 | <span style='color: green'>(-4 [-0.9%])</span> 464 | <span style='color: green'>(-4 [-0.9%])</span> 464 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-1.3%])</span> 154 | <span style='color: green'>(-2 [-1.3%])</span> 154 | <span style='color: green'>(-2 [-1.3%])</span> 154 | <span style='color: green'>(-2 [-1.3%])</span> 154 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+9 [+1.2%])</span> 779 | <span style='color: red'>(+9 [+1.2%])</span> 779 | <span style='color: red'>(+9 [+1.2%])</span> 779 | <span style='color: red'>(+9 [+1.2%])</span> 779 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-5 [-1.3%])</span> 370 | <span style='color: green'>(-5 [-1.3%])</span> 370 | <span style='color: green'>(-5 [-1.3%])</span> 370 | <span style='color: green'>(-5 [-1.3%])</span> 370 |
| `quotient_poly_commit_time_ms` |  293 |  293 |  293 |  293 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+8 [+1.3%])</span> 620 | <span style='color: red'>(+8 [+1.3%])</span> 620 | <span style='color: red'>(+8 [+1.3%])</span> 620 | <span style='color: red'>(+8 [+1.3%])</span> 620 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 259 | 4 | 

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
| fibonacci_program | VmConnectorAir | 2 | 5 | 10 | 
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
| leaf | VmConnectorAir | 2 | 5 | 10 | 
| leaf | VolatileBoundaryAir | 2 | 4 | 17 | 

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
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 12 | 11 | 3,014,656 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 32 |  | 16 | 17 | 1,056 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 32 |  | 12 | 20 | 1,024 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 52 | 41 | 1,488 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 663 | 3,853 | 1,264,964 | 251,993,578 | 2,686 | 370 | 293 | 779 | 620 | 464 | 70,274,573 | 154 | 504 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 5,439,620 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,719,810 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,484,996 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 61,919,946 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 555 | 2,677 | 1,500,096 | 160,733,916 | 1,886 | 251 | 217 | 467 | 503 | 327 | 51,485,167 | 118 | 236 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 3,932,270 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 10,748,264 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 1,966,135 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 10,748,300 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 800 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 288 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 7,209,044 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 35,526,957 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/7749b49af03a239a8d79bfea64ed8f81ccb9ed0e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13864161202)
