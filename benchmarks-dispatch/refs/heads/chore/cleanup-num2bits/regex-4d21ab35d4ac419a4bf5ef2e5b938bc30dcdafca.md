| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  49.28 |  49.28 |
| regex_program | <span style='color: red'>(+0 [+0.3%])</span> 18.95 | <span style='color: red'>(+0 [+0.3%])</span> 18.95 |
| leaf | <span style='color: green'>(-0 [-0.3%])</span> 30.33 | <span style='color: green'>(-0 [-0.3%])</span> 30.33 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+49 [+0.3%])</span> 18,946 | <span style='color: red'>(+49 [+0.3%])</span> 18,946 | <span style='color: red'>(+49 [+0.3%])</span> 18,946 | <span style='color: red'>(+49 [+0.3%])</span> 18,946 |
| `main_cells_used     ` |  165,028,173 |  165,028,173 |  165,028,173 |  165,028,173 |
| `total_cycles        ` |  4,190,904 |  4,190,904 |  4,190,904 |  4,190,904 |
| `execute_time_ms     ` | <span style='color: green'>(-14 [-1.2%])</span> 1,127 | <span style='color: green'>(-14 [-1.2%])</span> 1,127 | <span style='color: green'>(-14 [-1.2%])</span> 1,127 | <span style='color: green'>(-14 [-1.2%])</span> 1,127 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+35 [+1.1%])</span> 3,360 | <span style='color: red'>(+35 [+1.1%])</span> 3,360 | <span style='color: red'>(+35 [+1.1%])</span> 3,360 | <span style='color: red'>(+35 [+1.1%])</span> 3,360 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+28 [+0.2%])</span> 14,459 | <span style='color: red'>(+28 [+0.2%])</span> 14,459 | <span style='color: red'>(+28 [+0.2%])</span> 14,459 | <span style='color: red'>(+28 [+0.2%])</span> 14,459 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-30 [-1.3%])</span> 2,355 | <span style='color: green'>(-30 [-1.3%])</span> 2,355 | <span style='color: green'>(-30 [-1.3%])</span> 2,355 | <span style='color: green'>(-30 [-1.3%])</span> 2,355 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+6 [+1.2%])</span> 499 | <span style='color: red'>(+6 [+1.2%])</span> 499 | <span style='color: red'>(+6 [+1.2%])</span> 499 | <span style='color: red'>(+6 [+1.2%])</span> 499 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-56 [-1.1%])</span> 5,091 | <span style='color: green'>(-56 [-1.1%])</span> 5,091 | <span style='color: green'>(-56 [-1.1%])</span> 5,091 | <span style='color: green'>(-56 [-1.1%])</span> 5,091 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+107 [+4.2%])</span> 2,627 | <span style='color: red'>(+107 [+4.2%])</span> 2,627 | <span style='color: red'>(+107 [+4.2%])</span> 2,627 | <span style='color: red'>(+107 [+4.2%])</span> 2,627 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+21 [+1.7%])</span> 1,228 | <span style='color: red'>(+21 [+1.7%])</span> 1,228 | <span style='color: red'>(+21 [+1.7%])</span> 1,228 | <span style='color: red'>(+21 [+1.7%])</span> 1,228 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-22 [-0.8%])</span> 2,655 | <span style='color: green'>(-22 [-0.8%])</span> 2,655 | <span style='color: green'>(-22 [-0.8%])</span> 2,655 | <span style='color: green'>(-22 [-0.8%])</span> 2,655 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-92 [-0.3%])</span> 30,333 | <span style='color: green'>(-92 [-0.3%])</span> 30,333 | <span style='color: green'>(-92 [-0.3%])</span> 30,333 | <span style='color: green'>(-92 [-0.3%])</span> 30,333 |
| `main_cells_used     ` |  244,203,631 |  244,203,631 |  244,203,631 |  244,203,631 |
| `total_cycles        ` |  5,938,263 |  5,938,263 |  5,938,263 |  5,938,263 |
| `execute_time_ms     ` | <span style='color: red'>(+124 [+8.0%])</span> 1,680 | <span style='color: red'>(+124 [+8.0%])</span> 1,680 | <span style='color: red'>(+124 [+8.0%])</span> 1,680 | <span style='color: red'>(+124 [+8.0%])</span> 1,680 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-176 [-4.2%])</span> 4,059 | <span style='color: green'>(-176 [-4.2%])</span> 4,059 | <span style='color: green'>(-176 [-4.2%])</span> 4,059 | <span style='color: green'>(-176 [-4.2%])</span> 4,059 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-40 [-0.2%])</span> 24,594 | <span style='color: green'>(-40 [-0.2%])</span> 24,594 | <span style='color: green'>(-40 [-0.2%])</span> 24,594 | <span style='color: green'>(-40 [-0.2%])</span> 24,594 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+22 [+0.5%])</span> 4,476 | <span style='color: red'>(+22 [+0.5%])</span> 4,476 | <span style='color: red'>(+22 [+0.5%])</span> 4,476 | <span style='color: red'>(+22 [+0.5%])</span> 4,476 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+2 [+0.4%])</span> 536 | <span style='color: red'>(+2 [+0.4%])</span> 536 | <span style='color: red'>(+2 [+0.4%])</span> 536 | <span style='color: red'>(+2 [+0.4%])</span> 536 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-127 [-2.5%])</span> 5,049 | <span style='color: green'>(-127 [-2.5%])</span> 5,049 | <span style='color: green'>(-127 [-2.5%])</span> 5,049 | <span style='color: green'>(-127 [-2.5%])</span> 5,049 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+20 [+0.4%])</span> 5,462 | <span style='color: red'>(+20 [+0.4%])</span> 5,462 | <span style='color: red'>(+20 [+0.4%])</span> 5,462 | <span style='color: red'>(+20 [+0.4%])</span> 5,462 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-24 [-0.6%])</span> 3,956 | <span style='color: green'>(-24 [-0.6%])</span> 3,956 | <span style='color: green'>(-24 [-0.6%])</span> 3,956 | <span style='color: green'>(-24 [-0.6%])</span> 3,956 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+67 [+1.3%])</span> 5,113 | <span style='color: red'>(+67 [+1.3%])</span> 5,113 | <span style='color: red'>(+67 [+1.3%])</span> 5,113 | <span style='color: red'>(+67 [+1.3%])</span> 5,113 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 615 | 42 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 31 | 302 | 
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
| leaf | AccessAdapterAir<2> | 0 | 2,097,152 |  | 16 | 11 | 56,623,104 | 
| leaf | AccessAdapterAir<4> | 0 | 1,048,576 |  | 16 | 13 | 30,408,704 | 
| leaf | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 36 | 348 | 25,165,824 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
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
| leaf | 0 | 4,059 | 30,333 | 5,938,263 | 750,717,400 | 24,594 | 5,462 | 3,956 | 5,049 | 5,113 | 4,476 | 244,203,631 | 536 | 1,680 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,360 | 18,946 | 4,190,904 | 632,452,480 | 14,459 | 2,627 | 1,228 | 5,091 | 2,655 | 2,355 | 165,028,173 | 499 | 1,127 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/4d21ab35d4ac419a4bf5ef2e5b938bc30dcdafca

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12824482434)