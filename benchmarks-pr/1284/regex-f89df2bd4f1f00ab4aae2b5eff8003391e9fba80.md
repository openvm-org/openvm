| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-3 [-9.1%])</span> 32.88 | <span style='color: green'>(-3 [-9.1%])</span> 32.88 |
| regex_program | <span style='color: green'>(-1 [-6.2%])</span> 17.57 | <span style='color: green'>(-1 [-6.2%])</span> 17.57 |
| leaf | <span style='color: green'>(-2 [-12.2%])</span> 15.31 | <span style='color: green'>(-2 [-12.2%])</span> 15.31 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1170 [-6.2%])</span> 17,569 | <span style='color: green'>(-1170 [-6.2%])</span> 17,569 | <span style='color: green'>(-1170 [-6.2%])</span> 17,569 | <span style='color: green'>(-1170 [-6.2%])</span> 17,569 |
| `main_cells_used     ` |  165,010,909 |  165,010,909 |  165,010,909 |  165,010,909 |
| `total_cycles        ` |  4,190,904 |  4,190,904 |  4,190,904 |  4,190,904 |
| `execute_time_ms     ` | <span style='color: red'>(+25 [+2.5%])</span> 1,012 | <span style='color: red'>(+25 [+2.5%])</span> 1,012 | <span style='color: red'>(+25 [+2.5%])</span> 1,012 | <span style='color: red'>(+25 [+2.5%])</span> 1,012 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+34 [+1.1%])</span> 3,097 | <span style='color: red'>(+34 [+1.1%])</span> 3,097 | <span style='color: red'>(+34 [+1.1%])</span> 3,097 | <span style='color: red'>(+34 [+1.1%])</span> 3,097 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1229 [-8.4%])</span> 13,460 | <span style='color: green'>(-1229 [-8.4%])</span> 13,460 | <span style='color: green'>(-1229 [-8.4%])</span> 13,460 | <span style='color: green'>(-1229 [-8.4%])</span> 13,460 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+206 [+9.3%])</span> 2,416 | <span style='color: red'>(+206 [+9.3%])</span> 2,416 | <span style='color: red'>(+206 [+9.3%])</span> 2,416 | <span style='color: red'>(+206 [+9.3%])</span> 2,416 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-8 [-1.6%])</span> 496 | <span style='color: green'>(-8 [-1.6%])</span> 496 | <span style='color: green'>(-8 [-1.6%])</span> 496 | <span style='color: green'>(-8 [-1.6%])</span> 496 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+328 [+6.1%])</span> 5,703 | <span style='color: red'>(+328 [+6.1%])</span> 5,703 | <span style='color: red'>(+328 [+6.1%])</span> 5,703 | <span style='color: red'>(+328 [+6.1%])</span> 5,703 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-98 [-9.4%])</span> 948 | <span style='color: green'>(-98 [-9.4%])</span> 948 | <span style='color: green'>(-98 [-9.4%])</span> 948 | <span style='color: green'>(-98 [-9.4%])</span> 948 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-95 [-7.5%])</span> 1,171 | <span style='color: green'>(-95 [-7.5%])</span> 1,171 | <span style='color: green'>(-95 [-7.5%])</span> 1,171 | <span style='color: green'>(-95 [-7.5%])</span> 1,171 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+40 [+1.5%])</span> 2,719 | <span style='color: red'>(+40 [+1.5%])</span> 2,719 | <span style='color: red'>(+40 [+1.5%])</span> 2,719 | <span style='color: red'>(+40 [+1.5%])</span> 2,719 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-2122 [-12.2%])</span> 15,310 | <span style='color: green'>(-2122 [-12.2%])</span> 15,310 | <span style='color: green'>(-2122 [-12.2%])</span> 15,310 | <span style='color: green'>(-2122 [-12.2%])</span> 15,310 |
| `main_cells_used     ` |  142,194,373 |  142,194,373 |  142,194,373 |  142,194,373 |
| `total_cycles        ` |  3,027,784 |  3,027,784 |  3,027,784 |  3,027,784 |
| `execute_time_ms     ` | <span style='color: green'>(-34 [-4.5%])</span> 727 | <span style='color: green'>(-34 [-4.5%])</span> 727 | <span style='color: green'>(-34 [-4.5%])</span> 727 | <span style='color: green'>(-34 [-4.5%])</span> 727 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-37 [-1.4%])</span> 2,651 | <span style='color: green'>(-37 [-1.4%])</span> 2,651 | <span style='color: green'>(-37 [-1.4%])</span> 2,651 | <span style='color: green'>(-37 [-1.4%])</span> 2,651 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-2051 [-14.7%])</span> 11,932 | <span style='color: green'>(-2051 [-14.7%])</span> 11,932 | <span style='color: green'>(-2051 [-14.7%])</span> 11,932 | <span style='color: green'>(-2051 [-14.7%])</span> 11,932 |
| `main_trace_commit_time_ms` |  2,288 |  2,288 |  2,288 |  2,288 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+3 [+0.9%])</span> 329 | <span style='color: red'>(+3 [+0.9%])</span> 329 | <span style='color: red'>(+3 [+0.9%])</span> 329 | <span style='color: red'>(+3 [+0.9%])</span> 329 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+97 [+3.4%])</span> 2,916 | <span style='color: red'>(+97 [+3.4%])</span> 2,916 | <span style='color: red'>(+97 [+3.4%])</span> 2,916 | <span style='color: red'>(+97 [+3.4%])</span> 2,916 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-86 [-4.9%])</span> 1,658 | <span style='color: green'>(-86 [-4.9%])</span> 1,658 | <span style='color: green'>(-86 [-4.9%])</span> 1,658 | <span style='color: green'>(-86 [-4.9%])</span> 1,658 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-122 [-5.4%])</span> 2,139 | <span style='color: green'>(-122 [-5.4%])</span> 2,139 | <span style='color: green'>(-122 [-5.4%])</span> 2,139 | <span style='color: green'>(-122 [-5.4%])</span> 2,139 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+9 [+0.3%])</span> 2,599 | <span style='color: red'>(+9 [+0.3%])</span> 2,599 | <span style='color: red'>(+9 [+0.3%])</span> 2,599 | <span style='color: red'>(+9 [+0.3%])</span> 2,599 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 670 | 46 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 31 | 53 | 
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
| regex_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,571 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| regex_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 36 | 26 | 65,011,712 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 356 | 399 | 49,479,680 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 40 | 7,864,320 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 0 | 64 |  | 24 | 11 | 2,240 | 
| regex_program | AccessAdapterAir<4> | 0 | 32 |  | 24 | 13 | 1,184 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 24 | 17 | 5,373,952 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 32 |  | 1,288 | 3,164 | 142,464 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PhantomAir | 0 | 512 |  | 12 | 6 | 9,216 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 80 | 36 | 243,269,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 262,144 |  | 52 | 53 | 27,525,120 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 524,288 |  | 48 | 26 | 38,797,312 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 262,144 |  | 56 | 32 | 23,068,672 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| regex_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 16,384 |  | 36 | 26 | 1,015,808 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 76 | 35 | 113,664 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 72 | 40 | 234,881,024 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 104 | 57 | 20,608 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 100 | 39 | 35,584 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 65,536 |  | 80 | 31 | 7,274,496 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 65,536 |  | 28 | 21 | 3,211,264 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 2,651 | 15,310 | 3,027,784 | 421,678,040 | 11,932 | 1,658 | 2,139 | 2,916 | 2,599 | 2,288 | 142,194,373 | 329 | 727 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,097 | 17,569 | 4,190,904 | 632,452,480 | 13,460 | 948 | 1,171 | 5,703 | 2,719 | 2,416 | 165,010,909 | 496 | 1,012 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/f89df2bd4f1f00ab4aae2b5eff8003391e9fba80

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12961315001)
