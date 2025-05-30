| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  233.18 |  219.04 |
| kitchen_sink |  20.31 |  13.31 |
| leaf |  30.33 |  23.20 |
| internal.0 |  9.08 |  9.08 |
| root |  38.66 |  38.66 |
| halo2_outer |  90.42 |  90.42 |
| halo2_wrapper |  44.38 |  44.38 |


| kitchen_sink |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,156.50 |  20,313 |  13,308 |  7,005 |
| `main_cells_used     ` |  448,220,398 |  896,440,796 |  632,077,133 |  264,363,663 |
| `total_cycles        ` |  45,387 |  90,774 |  78,577 |  12,197 |
| `execute_time_ms     ` |  40.50 |  81 |  59 |  22 |
| `trace_gen_time_ms   ` |  336 |  672 |  427 |  245 |
| `stark_prove_excluding_trace_time_ms` |  9,780 |  19,560 |  12,822 |  6,738 |
| `main_trace_commit_time_ms` |  3,079.50 |  6,159 |  4,136 |  2,023 |
| `generate_perm_trace_time_ms` |  288 |  576 |  383 |  193 |
| `perm_trace_commit_time_ms` |  949.50 |  1,899 |  1,267 |  632 |
| `quotient_poly_compute_time_ms` |  4,341.50 |  8,683 |  5,767 |  2,916 |
| `quotient_poly_commit_time_ms` |  217 |  434 |  234 |  200 |
| `pcs_opening_time_ms ` |  874 |  1,748 |  988 |  760 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,166.50 |  30,333 |  23,200 |  7,133 |
| `main_cells_used     ` |  461,575,549 |  923,151,098 |  738,152,151 |  184,998,947 |
| `total_cycles        ` |  5,207,084.50 |  10,414,169 |  8,057,636 |  2,356,533 |
| `execute_time_ms     ` |  2,151 |  4,302 |  3,284 |  1,018 |
| `trace_gen_time_ms   ` |  1,094 |  2,188 |  1,686 |  502 |
| `stark_prove_excluding_trace_time_ms` |  11,921.50 |  23,843 |  18,230 |  5,613 |
| `main_trace_commit_time_ms` |  2,461.50 |  4,923 |  3,831 |  1,092 |
| `generate_perm_trace_time_ms` |  1,014.50 |  2,029 |  1,571 |  458 |
| `perm_trace_commit_time_ms` |  3,143.50 |  6,287 |  4,905 |  1,382 |
| `quotient_poly_compute_time_ms` |  1,871.50 |  3,743 |  2,854 |  889 |
| `quotient_poly_commit_time_ms` |  899.50 |  1,799 |  1,333 |  466 |
| `pcs_opening_time_ms ` |  2,526 |  5,052 |  3,730 |  1,322 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,083 |  9,083 |  9,083 |  9,083 |
| `main_cells_used     ` |  143,912,181 |  143,912,181 |  143,912,181 |  143,912,181 |
| `total_cycles        ` |  2,397,801 |  2,397,801 |  2,397,801 |  2,397,801 |
| `execute_time_ms     ` |  1,115 |  1,115 |  1,115 |  1,115 |
| `trace_gen_time_ms   ` |  350 |  350 |  350 |  350 |
| `stark_prove_excluding_trace_time_ms` |  7,618 |  7,618 |  7,618 |  7,618 |
| `main_trace_commit_time_ms` |  1,764 |  1,764 |  1,764 |  1,764 |
| `generate_perm_trace_time_ms` |  351 |  351 |  351 |  351 |
| `perm_trace_commit_time_ms` |  1,241 |  1,241 |  1,241 |  1,241 |
| `quotient_poly_compute_time_ms` |  1,417 |  1,417 |  1,417 |  1,417 |
| `quotient_poly_commit_time_ms` |  1,127 |  1,127 |  1,127 |  1,127 |
| `pcs_opening_time_ms ` |  1,713 |  1,713 |  1,713 |  1,713 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  38,659 |  38,659 |  38,659 |  38,659 |
| `main_cells_used     ` |  38,182,863 |  38,182,863 |  38,182,863 |  38,182,863 |
| `total_cycles        ` |  766,894 |  766,894 |  766,894 |  766,894 |
| `execute_time_ms     ` |  270 |  270 |  270 |  270 |
| `trace_gen_time_ms   ` |  130 |  130 |  130 |  130 |
| `stark_prove_excluding_trace_time_ms` |  38,259 |  38,259 |  38,259 |  38,259 |
| `main_trace_commit_time_ms` |  12,328 |  12,328 |  12,328 |  12,328 |
| `generate_perm_trace_time_ms` |  77 |  77 |  77 |  77 |
| `perm_trace_commit_time_ms` |  7,585 |  7,585 |  7,585 |  7,585 |
| `quotient_poly_compute_time_ms` |  827 |  827 |  827 |  827 |
| `quotient_poly_commit_time_ms` |  13,731 |  13,731 |  13,731 |  13,731 |
| `pcs_opening_time_ms ` |  3,698 |  3,698 |  3,698 |  3,698 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,416 |  90,416 |  90,416 |  90,416 |
| `main_cells_used     ` |  62,297,054 |  62,297,054 |  62,297,054 |  62,297,054 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,378 |  44,378 |  44,378 |  44,378 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 270 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| halo2_outer | 90,416 |  | 62,297,054 | 
| halo2_wrapper | 44,378 |  |  | 
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
| internal.0 | 0 | 350 | 9,083 | 2,397,801 | 432,384,482 | 7,618 | 1,417 | 1,127 | 1,241 | 1,713 | 1,764 | 143,912,181 | 351 | 1,115 | 
| leaf | 0 | 1,686 | 23,200 | 8,057,636 | 2,500,267,498 | 18,230 | 2,854 | 1,333 | 4,905 | 3,730 | 3,831 | 738,152,151 | 1,571 | 3,284 | 
| leaf | 1 | 502 | 7,133 | 2,356,533 | 746,278,378 | 5,613 | 889 | 466 | 1,382 | 1,322 | 1,092 | 184,998,947 | 458 | 1,018 | 
| root | 0 | 130 | 38,659 | 766,894 | 80,435,354 | 38,259 | 827 | 13,731 | 7,585 | 3,698 | 12,328 | 38,182,863 | 77 | 270 | 

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
| kitchen_sink | 0 | 427 | 13,308 | 78,577 | 1,305,398,457 | 12,822 | 5,767 | 234 | 1,267 | 988 | 4,136 | 632,077,133 | 383 | 59 | 
| kitchen_sink | 1 | 245 | 7,005 | 12,197 | 660,912,572 | 6,738 | 2,916 | 200 | 632 | 760 | 2,023 | 264,363,663 | 193 | 22 | 

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


Commit: https://github.com/openvm-org/openvm/commit/7d74e0313948b3acc5ef98aae85fa2322dd315a7

Max Segment Length: 4194204

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15355085564)
