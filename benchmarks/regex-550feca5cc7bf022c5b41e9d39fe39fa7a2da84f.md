| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  31.15 |  16.46 |
| regex_program |  14.01 |  7.83 |
| leaf |  17.14 |  8.62 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,006.50 |  14,013 |  7,834 |  6,179 |
| `main_cells_used     ` |  83,694,725 |  167,389,450 |  93,699,990 |  73,689,460 |
| `total_cycles        ` |  2,070,082 |  4,140,164 |  2,225,434 |  1,914,730 |
| `execute_time_ms     ` |  388 |  776 |  470 |  306 |
| `trace_gen_time_ms   ` |  1,260 |  2,520 |  1,298 |  1,222 |
| `stark_prove_excluding_trace_time_ms` |  5,358.50 |  10,717 |  6,066 |  4,651 |
| `main_trace_commit_time_ms` |  1,150.50 |  2,301 |  1,327 |  974 |
| `generate_perm_trace_time_ms` |  174.50 |  349 |  194 |  155 |
| `perm_trace_commit_time_ms` |  1,040.50 |  2,081 |  1,181 |  900 |
| `quotient_poly_compute_time_ms` |  744 |  1,488 |  884 |  604 |
| `quotient_poly_commit_time_ms` |  928 |  1,856 |  1,086 |  770 |
| `pcs_opening_time_ms ` |  1,311.50 |  2,623 |  1,385 |  1,238 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  8,568 |  17,136 |  8,623 |  8,513 |
| `main_cells_used     ` |  86,869,524.50 |  173,739,049 |  88,130,050 |  85,608,999 |
| `total_cycles        ` |  1,451,194 |  2,902,388 |  1,475,939 |  1,426,449 |
| `execute_time_ms     ` |  445.50 |  891 |  486 |  405 |
| `trace_gen_time_ms   ` |  1,271.50 |  2,543 |  1,345 |  1,198 |
| `stark_prove_excluding_trace_time_ms` |  6,851 |  13,702 |  6,873 |  6,829 |
| `main_trace_commit_time_ms` |  1,395 |  2,790 |  1,396 |  1,394 |
| `generate_perm_trace_time_ms` |  190 |  380 |  192 |  188 |
| `perm_trace_commit_time_ms` |  1,402.50 |  2,805 |  1,409 |  1,396 |
| `quotient_poly_compute_time_ms` |  1,086.50 |  2,173 |  1,099 |  1,074 |
| `quotient_poly_commit_time_ms` |  1,387 |  2,774 |  1,392 |  1,382 |
| `pcs_opening_time_ms ` |  1,384.50 |  2,769 |  1,397 |  1,372 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 634 | 41 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 11 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 11 | 
| leaf | FriReducedOpeningAir | 4 | 39 | 60 | 
| leaf | JalRangeCheckAir | 4 | 9 | 11 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 136 | 533 | 
| leaf | PhantomAir | 4 | 3 | 4 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 15 | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 11 | 22 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 11 | 23 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 15 | 16 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 5 | 9 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 
| regex_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 4 | 321 | 4,380 | 
| regex_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| regex_program | PhantomAir | 4 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 4 | 18 | 23 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 20 | 31 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 18 | 36 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 24 | 85 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 27 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 34 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 76 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 12 | 11 | 
| regex_program | VmConnectorAir | 4 | 5 | 9 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 12 | 13 | 13,107,200 | 
| leaf | AccessAdapterAir<8> | 0 | 16,384 |  | 12 | 17 | 475,136 | 
| leaf | AccessAdapterAir<8> | 1 | 16,384 |  | 12 | 17 | 475,136 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 44 | 27 | 37,224,448 | 
| leaf | FriReducedOpeningAir | 1 | 524,288 |  | 44 | 27 | 37,224,448 | 
| leaf | JalRangeCheckAir | 0 | 32,768 |  | 16 | 12 | 917,504 | 
| leaf | JalRangeCheckAir | 1 | 32,768 |  | 16 | 12 | 917,504 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 160 | 399 | 36,634,624 | 
| leaf | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | PhantomAir | 1 | 8,192 |  | 8 | 6 | 114,688 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 21 | 23,592,960 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 24 | 21 | 23,592,960 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 27 | 3,342,336 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 65,536 |  | 24 | 27 | 3,342,336 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 20 | 38 | 7,602,176 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 12 | 17 | 59,392 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 532 | 3,163 | 3,695 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 532 | 3,163 | 118,240 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 12 | 32 | 180,224 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 8 | 20 | 3,670,016 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 8 | 20 | 57,344 | 
| regex_program | PhantomAir | 0 | 1 |  | 8 | 6 | 14 | 
| regex_program | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 24 | 32 | 917,504 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 28 | 36 | 33,554,432 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 16 | 26 | 5,505,024 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 20 | 28 | 6,291,456 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 20 | 28 | 3,145,728 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 28 | 36 | 65,536 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 28 | 36 | 128 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 28 | 41 | 72,351,744 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 28 | 41 | 72,351,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 40 | 59 | 12,672 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 40 | 39 | 20,224 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 16 | 20 | 1,179,648 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 16 | 20 | 1,179,648 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 1,198 | 8,513 | 1,475,939 | 225,774,050 | 6,829 | 1,074 | 1,382 | 1,409 | 1,372 | 1,396 | 88,130,050 | 192 | 486 | 
| leaf | 1 | 1,345 | 8,623 | 1,426,449 | 225,774,050 | 6,873 | 1,099 | 1,392 | 1,396 | 1,397 | 1,394 | 85,608,999 | 188 | 405 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,298 | 7,834 | 2,225,434 | 210,931,487 | 6,066 | 884 | 1,086 | 1,181 | 1,385 | 1,327 | 93,699,990 | 194 | 470 | 
| regex_program | 1 | 1,222 | 6,179 | 1,914,730 | 150,470,512 | 4,651 | 604 | 770 | 900 | 1,238 | 974 | 73,689,460 | 155 | 306 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/550feca5cc7bf022c5b41e9d39fe39fa7a2da84f

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13768672425)
