| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-5 [-30.8%])</span> 11.94 | <span style='color: green'>(-5 [-30.8%])</span> 11.94 |
| pairing | <span style='color: green'>(-0 [-3.1%])</span> 4.36 | <span style='color: green'>(-0 [-3.1%])</span> 4.36 |
| leaf | <span style='color: green'>(-5 [-40.6%])</span> 7.59 | <span style='color: green'>(-5 [-40.6%])</span> 7.59 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-140 [-3.1%])</span> 4,357 | <span style='color: green'>(-140 [-3.1%])</span> 4,357 | <span style='color: green'>(-140 [-3.1%])</span> 4,357 | <span style='color: green'>(-140 [-3.1%])</span> 4,357 |
| `main_cells_used     ` | <span style='color: red'>(+346586 [+0.4%])</span> 96,178,993 | <span style='color: red'>(+346586 [+0.4%])</span> 96,178,993 | <span style='color: red'>(+346586 [+0.4%])</span> 96,178,993 | <span style='color: red'>(+346586 [+0.4%])</span> 96,178,993 |
| `total_cycles        ` |  1,820,436 |  1,820,436 |  1,820,436 |  1,820,436 |
| `execute_time_ms     ` | <span style='color: red'>(+485 [+74.8%])</span> 1,133 | <span style='color: red'>(+485 [+74.8%])</span> 1,133 | <span style='color: red'>(+485 [+74.8%])</span> 1,133 | <span style='color: red'>(+485 [+74.8%])</span> 1,133 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-570 [-61.4%])</span> 359 | <span style='color: green'>(-570 [-61.4%])</span> 359 | <span style='color: green'>(-570 [-61.4%])</span> 359 | <span style='color: green'>(-570 [-61.4%])</span> 359 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-55 [-1.9%])</span> 2,865 | <span style='color: green'>(-55 [-1.9%])</span> 2,865 | <span style='color: green'>(-55 [-1.9%])</span> 2,865 | <span style='color: green'>(-55 [-1.9%])</span> 2,865 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-9 [-1.5%])</span> 611 | <span style='color: green'>(-9 [-1.5%])</span> 611 | <span style='color: green'>(-9 [-1.5%])</span> 611 | <span style='color: green'>(-9 [-1.5%])</span> 611 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-1 [-0.5%])</span> 191 | <span style='color: green'>(-1 [-0.5%])</span> 191 | <span style='color: green'>(-1 [-0.5%])</span> 191 | <span style='color: green'>(-1 [-0.5%])</span> 191 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-32 [-5.1%])</span> 595 | <span style='color: green'>(-32 [-5.1%])</span> 595 | <span style='color: green'>(-32 [-5.1%])</span> 595 | <span style='color: green'>(-32 [-5.1%])</span> 595 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+1 [+0.2%])</span> 406 | <span style='color: red'>(+1 [+0.2%])</span> 406 | <span style='color: red'>(+1 [+0.2%])</span> 406 | <span style='color: red'>(+1 [+0.2%])</span> 406 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-10 [-3.3%])</span> 291 | <span style='color: green'>(-10 [-3.3%])</span> 291 | <span style='color: green'>(-10 [-3.3%])</span> 291 | <span style='color: green'>(-10 [-3.3%])</span> 291 |
| `pcs_opening_time_ms ` |  760 |  760 |  760 |  760 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-5180 [-40.6%])</span> 7,586 | <span style='color: green'>(-5180 [-40.6%])</span> 7,586 | <span style='color: green'>(-5180 [-40.6%])</span> 7,586 | <span style='color: green'>(-5180 [-40.6%])</span> 7,586 |
| `main_cells_used     ` | <span style='color: green'>(-70391183 [-25.7%])</span> 203,465,831 | <span style='color: green'>(-70391183 [-25.7%])</span> 203,465,831 | <span style='color: green'>(-70391183 [-25.7%])</span> 203,465,831 | <span style='color: green'>(-70391183 [-25.7%])</span> 203,465,831 |
| `total_cycles        ` | <span style='color: green'>(-563926 [-17.3%])</span> 2,703,462 | <span style='color: green'>(-563926 [-17.3%])</span> 2,703,462 | <span style='color: green'>(-563926 [-17.3%])</span> 2,703,462 | <span style='color: green'>(-563926 [-17.3%])</span> 2,703,462 |
| `execute_time_ms     ` | <span style='color: green'>(-138 [-11.3%])</span> 1,086 | <span style='color: green'>(-138 [-11.3%])</span> 1,086 | <span style='color: green'>(-138 [-11.3%])</span> 1,086 | <span style='color: green'>(-138 [-11.3%])</span> 1,086 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-2006 [-79.4%])</span> 522 | <span style='color: green'>(-2006 [-79.4%])</span> 522 | <span style='color: green'>(-2006 [-79.4%])</span> 522 | <span style='color: green'>(-2006 [-79.4%])</span> 522 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-3036 [-33.7%])</span> 5,978 | <span style='color: green'>(-3036 [-33.7%])</span> 5,978 | <span style='color: green'>(-3036 [-33.7%])</span> 5,978 | <span style='color: green'>(-3036 [-33.7%])</span> 5,978 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-501 [-29.4%])</span> 1,202 | <span style='color: green'>(-501 [-29.4%])</span> 1,202 | <span style='color: green'>(-501 [-29.4%])</span> 1,202 | <span style='color: green'>(-501 [-29.4%])</span> 1,202 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-307 [-38.9%])</span> 482 | <span style='color: green'>(-307 [-38.9%])</span> 482 | <span style='color: green'>(-307 [-38.9%])</span> 482 | <span style='color: green'>(-307 [-38.9%])</span> 482 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-877 [-35.6%])</span> 1,587 | <span style='color: green'>(-877 [-35.6%])</span> 1,587 | <span style='color: green'>(-877 [-35.6%])</span> 1,587 | <span style='color: green'>(-877 [-35.6%])</span> 1,587 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-417 [-31.5%])</span> 905 | <span style='color: green'>(-417 [-31.5%])</span> 905 | <span style='color: green'>(-417 [-31.5%])</span> 905 | <span style='color: green'>(-417 [-31.5%])</span> 905 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-256 [-34.0%])</span> 496 | <span style='color: green'>(-256 [-34.0%])</span> 496 | <span style='color: green'>(-256 [-34.0%])</span> 496 | <span style='color: green'>(-256 [-34.0%])</span> 496 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-679 [-34.3%])</span> 1,301 | <span style='color: green'>(-679 [-34.3%])</span> 1,301 | <span style='color: green'>(-679 [-34.3%])</span> 1,301 | <span style='color: green'>(-679 [-34.3%])</span> 1,301 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| pairing | 1 | 1,128 | 10 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
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
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | KeccakVmAir | 2 | 321 | 4,513 | 
| pairing | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| pairing | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| pairing | PhantomAir | 2 | 3 | 5 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| pairing | ProgramAir | 1 | 1 | 4 | 
| pairing | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| pairing | Rv32HintStoreAir | 2 | 18 | 28 | 
| pairing | VariableRangeCheckerAir | 1 | 1 | 4 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| pairing | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| leaf | FriReducedOpeningAir | 0 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | PhantomAir | 0 | 32,768 |  | 12 | 6 | 589,824 | 
| leaf | ProgramAir | 0 | 1,048,576 |  | 8 | 10 | 18,874,368 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 20 | 12 | 16,777,216 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<32> | 0 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 32,768 |  | 16 | 32 | 1,572,864 | 
| pairing | PersistentBoundaryAir<8> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32,768 |  | 8 | 300 | 10,092,544 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 56 | 166 | 7,104 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 65,536 |  | 36 | 28 | 4,194,304 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 836 | 547 | 1,383 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 320 | 263 | 596,992 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 16,384 |  | 860 | 625 | 18,038,784 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 522 | 7,586 | 2,703,462 | 768,708,074 | 5,978 | 905 | 496 | 1,587 | 1,301 | 1,202 | 203,465,831 | 482 | 1,086 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 13,828,228 | 2,013,265,921 | 
| leaf | 0 | 1 | 84,640,000 | 2,013,265,921 | 
| leaf | 0 | 2 | 6,914,114 | 2,013,265,921 | 
| leaf | 0 | 3 | 84,738,308 | 2,013,265,921 | 
| leaf | 0 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 0 | 5 | 191,955,658 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 359 | 4,357 | 1,820,436 | 297,669,276 | 2,865 | 406 | 291 | 595 | 760 | 611 | 96,178,993 | 191 | 1,133 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 5,112,016 | 2,013,265,921 | 
| pairing | 0 | 1 | 17,620,096 | 2,013,265,921 | 
| pairing | 0 | 2 | 2,556,008 | 2,013,265,921 | 
| pairing | 0 | 3 | 24,468,620 | 2,013,265,921 | 
| pairing | 0 | 4 | 131,072 | 2,013,265,921 | 
| pairing | 0 | 5 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 6 | 6,003,913 | 2,013,265,921 | 
| pairing | 0 | 7 | 4,096 | 2,013,265,921 | 
| pairing | 0 | 8 | 56,944,397 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/a7f81dccb422fd45837ab83f7f939c118d0ed7cd

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15356879503)
