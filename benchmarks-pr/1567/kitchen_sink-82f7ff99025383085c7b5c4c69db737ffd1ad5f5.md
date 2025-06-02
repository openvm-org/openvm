| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  232.14 |  217.97 |
| kitchen_sink |  20.43 |  13.41 |
| leaf |  30.28 |  23.13 |
| internal.0 |  9.10 |  9.10 |
| root |  38.57 |  38.57 |
| halo2_outer |  89.47 |  89.47 |
| halo2_wrapper |  44.29 |  44.29 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,215.50 |  20,431 |  13,409 |  7,022 |
| `main_cells_used     ` |  448,220,398 |  896,440,796 |  632,077,133 |  264,363,663 |
| `total_cycles        ` |  45,387 |  90,774 |  78,577 |  12,197 |
| `execute_time_ms     ` |  38 |  76 |  57 |  19 |
| `trace_gen_time_ms   ` |  360 |  720 |  427 |  293 |
| `stark_prove_excluding_trace_time_ms` |  9,817.50 |  19,635 |  12,925 |  6,710 |
| `main_trace_commit_time_ms` |  3,125 |  6,250 |  4,229 |  2,021 |
| `generate_perm_trace_time_ms` |  286 |  572 |  378 |  194 |
| `perm_trace_commit_time_ms` |  945.50 |  1,891 |  1,259 |  632 |
| `quotient_poly_compute_time_ms` |  4,342.50 |  8,685 |  5,766 |  2,919 |
| `quotient_poly_commit_time_ms` |  205.50 |  411 |  244 |  167 |
| `pcs_opening_time_ms ` |  883 |  1,766 |  1,002 |  764 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,140.50 |  30,281 |  23,131 |  7,150 |
| `main_cells_used     ` |  461,574,865 |  923,149,730 |  738,150,867 |  184,998,863 |
| `total_cycles        ` |  5,207,027.50 |  10,414,055 |  8,057,529 |  2,356,526 |
| `execute_time_ms     ` |  2,156.50 |  4,313 |  3,298 |  1,015 |
| `trace_gen_time_ms   ` |  1,058 |  2,116 |  1,612 |  504 |
| `stark_prove_excluding_trace_time_ms` |  11,926 |  23,852 |  18,221 |  5,631 |
| `main_trace_commit_time_ms` |  2,473.50 |  4,947 |  3,837 |  1,110 |
| `generate_perm_trace_time_ms` |  1,020 |  2,040 |  1,583 |  457 |
| `perm_trace_commit_time_ms` |  3,138.50 |  6,277 |  4,893 |  1,384 |
| `quotient_poly_compute_time_ms` |  1,872 |  3,744 |  2,858 |  886 |
| `quotient_poly_commit_time_ms` |  900.50 |  1,801 |  1,324 |  477 |
| `pcs_opening_time_ms ` |  2,516 |  5,032 |  3,721 |  1,311 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,097 |  9,097 |  9,097 |  9,097 |
| `main_cells_used     ` |  143,914,797 |  143,914,797 |  143,914,797 |  143,914,797 |
| `total_cycles        ` |  2,398,019 |  2,398,019 |  2,398,019 |  2,398,019 |
| `execute_time_ms     ` |  1,109 |  1,109 |  1,109 |  1,109 |
| `trace_gen_time_ms   ` |  351 |  351 |  351 |  351 |
| `stark_prove_excluding_trace_time_ms` |  7,637 |  7,637 |  7,637 |  7,637 |
| `main_trace_commit_time_ms` |  1,746 |  1,746 |  1,746 |  1,746 |
| `generate_perm_trace_time_ms` |  356 |  356 |  356 |  356 |
| `perm_trace_commit_time_ms` |  1,244 |  1,244 |  1,244 |  1,244 |
| `quotient_poly_compute_time_ms` |  1,421 |  1,421 |  1,421 |  1,421 |
| `quotient_poly_commit_time_ms` |  1,154 |  1,154 |  1,154 |  1,154 |
| `pcs_opening_time_ms ` |  1,712 |  1,712 |  1,712 |  1,712 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,567 |  38,567 |  38,567 |  38,567 |
| `main_cells_used     ` |  38,182,455 |  38,182,455 |  38,182,455 |  38,182,455 |
| `total_cycles        ` |  766,860 |  766,860 |  766,860 |  766,860 |
| `execute_time_ms     ` |  266 |  266 |  266 |  266 |
| `trace_gen_time_ms   ` |  128 |  128 |  128 |  128 |
| `stark_prove_excluding_trace_time_ms` |  38,173 |  38,173 |  38,173 |  38,173 |
| `main_trace_commit_time_ms` |  12,317 |  12,317 |  12,317 |  12,317 |
| `generate_perm_trace_time_ms` |  77 |  77 |  77 |  77 |
| `perm_trace_commit_time_ms` |  7,585 |  7,585 |  7,585 |  7,585 |
| `quotient_poly_compute_time_ms` |  827 |  827 |  827 |  827 |
| `quotient_poly_commit_time_ms` |  13,657 |  13,657 |  13,657 |  13,657 |
| `pcs_opening_time_ms ` |  3,693 |  3,693 |  3,693 |  3,693 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  89,474 |  89,474 |  89,474 |  89,474 |
| `main_cells_used     ` |  62,297,054 |  62,297,054 |  62,297,054 |  62,297,054 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,288 |  44,288 |  44,288 |  44,288 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 267 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| halo2_outer | 89,474 |  | 62,297,054 | 
| halo2_wrapper | 44,288 |  |  | 
| kitchen_sink |  | 2 |  | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 12 | 11 | 24,117,248 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | JalRangeCheckAir | 0 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 160 | 398 | 146,276,352 | 
| internal.0 | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| leaf | AccessAdapterAir<2> | 0 | 4,194,304 |  | 16 | 11 | 113,246,208 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 2,097,152 |  | 16 | 13 | 60,817,408 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 8,388,608 |  | 84 | 27 | 931,135,488 | 
| leaf | FriReducedOpeningAir | 1 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | JalRangeCheckAir | 0 | 131,072 |  | 28 | 12 | 5,242,880 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 1,048,576 |  | 312 | 398 | 744,488,960 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | PhantomAir | 1 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | ProgramAir | 1 | 2,097,152 |  | 8 | 10 | 37,748,736 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 36 | 29 | 272,629,760 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 40 | 21 | 127,926,272 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 524,288 |  | 40 | 27 | 35,127,296 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 1,048,576 |  | 36 | 38 | 77,594,624 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 20 | 12 | 33,554,432 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 20 | 12 | 16,777,216 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 8 | 13 | 2,752,512 | 
| root | AccessAdapterAir<8> | 0 | 4,096 |  | 8 | 17 | 102,400 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| root | JalRangeCheckAir | 0 | 32,768 |  | 12 | 12 | 786,432 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 84 | 398 | 15,794,176 | 
| root | PhantomAir | 0 | 8,192 |  | 8 | 6 | 114,688 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 524,288 |  | 12 | 29 | 21,495,808 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 12 | 23 | 4,587,520 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 12 | 22 | 2,176 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 16 | 21 | 9,699,328 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 16 | 27 | 2,818,048 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 12 | 38 | 3,276,800 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 5 | 26 | 
| root | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 12 | 2,621,440 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| kitchen_sink | AccessAdapterAir<16> | 1 | 65,536 |  | 16 | 25 | 2,686,976 | 
| kitchen_sink | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| kitchen_sink | AccessAdapterAir<32> | 1 | 1,024 |  | 16 | 41 | 58,368 | 
| kitchen_sink | AccessAdapterAir<8> | 0 | 262,144 |  | 16 | 17 | 8,650,752 | 
| kitchen_sink | AccessAdapterAir<8> | 1 | 131,072 |  | 16 | 17 | 4,325,376 | 
| kitchen_sink | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| kitchen_sink | KeccakVmAir | 0 | 262,144 |  | 1,056 | 3,163 | 1,105,985,536 | 
| kitchen_sink | KeccakVmAir | 1 | 131,072 |  | 1,056 | 3,163 | 552,992,768 | 
| kitchen_sink | MemoryMerkleAir<8> | 0 | 8,192 |  | 16 | 32 | 393,216 | 
| kitchen_sink | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| kitchen_sink | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| kitchen_sink | PersistentBoundaryAir<8> | 1 | 4,096 |  | 12 | 20 | 131,072 | 
| kitchen_sink | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| kitchen_sink | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| kitchen_sink | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| kitchen_sink | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| kitchen_sink | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| kitchen_sink | RangeTupleCheckerAir<2> | 0 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | RangeTupleCheckerAir<2> | 1 | 2,097,152 | 2 | 8 | 1 | 18,874,368 | 
| kitchen_sink | Sha256VmAir | 0 | 262,144 |  | 108 | 470 | 151,519,232 | 
| kitchen_sink | Sha256VmAir | 1 | 131,072 |  | 108 | 470 | 75,759,616 | 
| kitchen_sink | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 16,384 |  | 52 | 36 | 1,441,792 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 4,096 |  | 52 | 36 | 360,448 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,048 |  | 40 | 37 | 157,696 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 256 |  | 40 | 37 | 19,712 | 
| kitchen_sink | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 32 |  | 52 | 53 | 3,360 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 4,096 |  | 28 | 26 | 221,184 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 1,024 |  | 28 | 26 | 55,296 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| kitchen_sink | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 512 |  | 32 | 32 | 32,768 | 
| kitchen_sink | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 512 |  | 28 | 18 | 23,552 | 
| kitchen_sink | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 64 |  | 28 | 18 | 2,944 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 0 | 1,024 |  | 192 | 168 | 368,640 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, BaseAluCoreAir<32, 8> | 1 | 256 |  | 192 | 168 | 92,160 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 0 | 1,024 |  | 68 | 169 | 242,688 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, LessThanCoreAir<32, 8> | 1 | 128 |  | 68 | 169 | 30,336 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 0 | 256 |  | 192 | 164 | 91,136 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, MultiplicationCoreAir<32, 8> | 1 | 64 |  | 192 | 164 | 22,784 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, ShiftCoreAir<32, 8> | 0 | 512 |  | 164 | 241 | 207,360 | 
| kitchen_sink | VmAirWrapper<Rv32HeapAdapterAir<2, 32, 32>, ShiftCoreAir<32, 8> | 1 | 128 |  | 164 | 241 | 51,840 | 
| kitchen_sink | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 0 | 256 |  | 48 | 124 | 44,032 | 
| kitchen_sink | VmAirWrapper<Rv32HeapBranchAdapterAir<2, 32>, BranchEqualCoreAir<32> | 1 | 64 |  | 48 | 124 | 11,008 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4 |  | 56 | 166 | 888 | 
| kitchen_sink | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 3, 16, 48>, ModularIsEqualCoreAir<48, 4, 8> | 0 | 1 |  | 88 | 242 | 330 | 
| kitchen_sink | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,048 |  | 36 | 28 | 131,072 | 
| kitchen_sink | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 256 |  | 36 | 28 | 16,384 | 
| kitchen_sink | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| kitchen_sink | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 8,192 |  | 52 | 41 | 761,856 | 
| kitchen_sink | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4 |  | 52 | 31 | 332 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,024 |  | 28 | 20 | 49,152 | 
| kitchen_sink | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 128 |  | 28 | 20 | 6,144 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 836 | 547 | 1,383 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<1, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 1 |  | 1,668 | 1,020 | 2,688 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 384 | 294 | 41,920 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 860 | 625 | 1,485 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 3, 3, 16, 16>, FieldExpressionCoreAir> | 0 | 1 |  | 496 | 393 | 889 | 
| kitchen_sink | VmAirWrapper<Rv32VecHeapAdapterAir<2, 6, 6, 16, 16>, FieldExpressionCoreAir> | 0 | 1 |  | 1,340 | 949 | 2,289 | 
| kitchen_sink | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| kitchen_sink | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 351 | 9,097 | 2,398,019 | 432,384,482 | 7,637 | 1,421 | 1,154 | 1,244 | 1,712 | 1,746 | 143,914,797 | 356 | 1,109 | 
| leaf | 0 | 1,612 | 23,131 | 8,057,529 | 2,500,267,498 | 18,221 | 2,858 | 1,324 | 4,893 | 3,721 | 3,837 | 738,150,867 | 1,583 | 3,298 | 
| leaf | 1 | 504 | 7,150 | 2,356,526 | 746,278,378 | 5,631 | 886 | 477 | 1,384 | 1,311 | 1,110 | 184,998,863 | 457 | 1,015 | 
| root | 0 | 128 | 38,567 | 766,860 | 80,435,354 | 38,173 | 827 | 13,657 | 7,585 | 3,693 | 12,317 | 38,182,455 | 77 | 266 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 10,354,820 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 60,317,952 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 5,177,410 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 60,047,620 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 524,288 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 136,815,306 | 2,013,265,921 | 
| leaf | 0 | 0 | 39,125,124 | 2,013,265,921 | 
| leaf | 0 | 1 | 291,111,168 | 2,013,265,921 | 
| leaf | 0 | 2 | 19,562,562 | 2,013,265,921 | 
| leaf | 0 | 3 | 288,096,516 | 2,013,265,921 | 
| leaf | 0 | 4 | 2,097,152 | 2,013,265,921 | 
| leaf | 0 | 5 | 642,351,818 | 2,013,265,921 | 
| leaf | 1 | 0 | 12,517,508 | 2,013,265,921 | 
| leaf | 1 | 1 | 80,658,688 | 2,013,265,921 | 
| leaf | 1 | 2 | 6,258,754 | 2,013,265,921 | 
| leaf | 1 | 3 | 80,773,380 | 2,013,265,921 | 
| leaf | 1 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 1 | 5 | 183,091,914 | 2,013,265,921 | 
| root | 0 | 0 | 2,252,928 | 2,013,265,921 | 
| root | 0 | 1 | 14,557,184 | 2,013,265,921 | 
| root | 0 | 2 | 1,126,464 | 2,013,265,921 | 
| root | 0 | 3 | 15,540,224 | 2,013,265,921 | 
| root | 0 | 4 | 262,144 | 2,013,265,921 | 
| root | 0 | 5 | 34,263,234 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 427 | 13,409 | 78,577 | 1,305,398,457 | 12,925 | 5,766 | 244 | 1,259 | 1,002 | 4,229 | 632,077,133 | 378 | 57 | 
| kitchen_sink | 1 | 293 | 7,022 | 12,197 | 660,912,572 | 6,710 | 2,919 | 167 | 632 | 764 | 2,021 | 264,363,663 | 194 | 19 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| kitchen_sink | 0 | 0 | 1,246,656 | 2,013,265,921 | 
| kitchen_sink | 0 | 1 | 28,013,068 | 2,013,265,921 | 
| kitchen_sink | 0 | 2 | 623,328 | 2,013,265,921 | 
| kitchen_sink | 0 | 3 | 27,783,323 | 2,013,265,921 | 
| kitchen_sink | 0 | 4 | 32,768 | 2,013,265,921 | 
| kitchen_sink | 0 | 5 | 16,384 | 2,013,265,921 | 
| kitchen_sink | 0 | 6 | 42,884,334 | 2,013,265,921 | 
| kitchen_sink | 0 | 7 | 524,288 | 2,013,265,921 | 
| kitchen_sink | 0 | 8 | 8,208 | 2,013,265,921 | 
| kitchen_sink | 0 | 9 | 103,635,013 | 2,013,265,921 | 
| kitchen_sink | 1 | 0 | 554,630 | 2,013,265,921 | 
| kitchen_sink | 1 | 1 | 13,794,432 | 2,013,265,921 | 
| kitchen_sink | 1 | 2 | 277,315 | 2,013,265,921 | 
| kitchen_sink | 1 | 3 | 13,614,340 | 2,013,265,921 | 
| kitchen_sink | 1 | 4 | 16,384 | 2,013,265,921 | 
| kitchen_sink | 1 | 5 | 8,192 | 2,013,265,921 | 
| kitchen_sink | 1 | 6 | 21,399,936 | 2,013,265,921 | 
| kitchen_sink | 1 | 7 | 262,144 | 2,013,265,921 | 
| kitchen_sink | 1 | 8 | 2,048 | 2,013,265,921 | 
| kitchen_sink | 1 | 9 | 52,432,077 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/82f7ff99025383085c7b5c4c69db737ffd1ad5f5

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15400253063)
