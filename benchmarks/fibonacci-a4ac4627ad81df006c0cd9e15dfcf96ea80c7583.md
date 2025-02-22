| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  8.79 |  8.79 |
| fibonacci_program |  4.94 |  4.94 |
| leaf |  3.84 |  3.84 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  4,942 |  4,942 |  4,942 |  4,942 |
| `main_cells_used     ` |  51,485,080 |  51,485,080 |  51,485,080 |  51,485,080 |
| `total_cycles        ` |  1,500,095 |  1,500,095 |  1,500,095 |  1,500,095 |
| `execute_time_ms     ` |  300 |  300 |  300 |  300 |
| `trace_gen_time_ms   ` |  618 |  618 |  618 |  618 |
| `stark_prove_excluding_trace_time_ms` |  4,024 |  4,024 |  4,024 |  4,024 |
| `main_trace_commit_time_ms` |  790 |  790 |  790 |  790 |
| `generate_perm_trace_time_ms` |  139 |  139 |  139 |  139 |
| `perm_trace_commit_time_ms` |  784 |  784 |  784 |  784 |
| `quotient_poly_compute_time_ms` |  513 |  513 |  513 |  513 |
| `quotient_poly_commit_time_ms` |  730 |  730 |  730 |  730 |
| `pcs_opening_time_ms ` |  1,066 |  1,066 |  1,066 |  1,066 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  3,843 |  3,843 |  3,843 |  3,843 |
| `main_cells_used     ` |  33,540,529 |  33,540,529 |  33,540,529 |  33,540,529 |
| `total_cycles        ` |  615,297 |  615,297 |  615,297 |  615,297 |
| `execute_time_ms     ` |  242 |  242 |  242 |  242 |
| `trace_gen_time_ms   ` |  575 |  575 |  575 |  575 |
| `stark_prove_excluding_trace_time_ms` |  3,026 |  3,026 |  3,026 |  3,026 |
| `main_trace_commit_time_ms` |  620 |  620 |  620 |  620 |
| `generate_perm_trace_time_ms` |  73 |  73 |  73 |  73 |
| `perm_trace_commit_time_ms` |  561 |  561 |  561 |  561 |
| `quotient_poly_compute_time_ms` |  409 |  409 |  409 |  409 |
| `quotient_poly_commit_time_ms` |  619 |  619 |  619 |  619 |
| `pcs_opening_time_ms ` |  740 |  740 |  740 |  740 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 393 | 5 | 

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
| leaf | FriReducedOpeningAir | 4 | 39 | 60 | 
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
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 44 | 27 | 9,306,112 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 524,288 |  | 20 | 29 | 25,690,112 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 65,536 |  | 16 | 23 | 2,555,904 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 16,384 |  | 12 | 9 | 344,064 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 131,072 |  | 24 | 22 | 6,029,312 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 11 | 2,490,368 | 

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
| leaf | 0 | 575 | 3,843 | 615,297 | 86,295,512 | 3,026 | 409 | 619 | 561 | 740 | 620 | 33,540,529 | 73 | 242 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 618 | 4,942 | 1,500,095 | 122,458,476 | 4,024 | 513 | 730 | 784 | 1,066 | 790 | 51,485,080 | 139 | 300 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/a4ac4627ad81df006c0cd9e15dfcf96ea80c7583

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13467303967)
