| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-12 [-24.8%])</span> 37.02 | <span style='color: green'>(-12 [-24.8%])</span> 37.02 |
| regex_program | <span style='color: green'>(-0 [-0.4%])</span> 18.85 | <span style='color: green'>(-0 [-0.4%])</span> 18.85 |
| leaf | <span style='color: green'>(-12 [-40.1%])</span> 18.18 | <span style='color: green'>(-12 [-40.1%])</span> 18.18 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-77 [-0.4%])</span> 18,846 | <span style='color: green'>(-77 [-0.4%])</span> 18,846 | <span style='color: green'>(-77 [-0.4%])</span> 18,846 | <span style='color: green'>(-77 [-0.4%])</span> 18,846 |
| `main_cells_used     ` |  165,028,173 |  165,028,173 |  165,028,173 |  165,028,173 |
| `total_cycles        ` |  4,190,904 |  4,190,904 |  4,190,904 |  4,190,904 |
| `execute_time_ms     ` | <span style='color: green'>(-4 [-0.4%])</span> 1,136 | <span style='color: green'>(-4 [-0.4%])</span> 1,136 | <span style='color: green'>(-4 [-0.4%])</span> 1,136 | <span style='color: green'>(-4 [-0.4%])</span> 1,136 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-32 [-1.0%])</span> 3,332 | <span style='color: green'>(-32 [-1.0%])</span> 3,332 | <span style='color: green'>(-32 [-1.0%])</span> 3,332 | <span style='color: green'>(-32 [-1.0%])</span> 3,332 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-41 [-0.3%])</span> 14,378 | <span style='color: green'>(-41 [-0.3%])</span> 14,378 | <span style='color: green'>(-41 [-0.3%])</span> 14,378 | <span style='color: green'>(-41 [-0.3%])</span> 14,378 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+4 [+0.2%])</span> 2,371 | <span style='color: red'>(+4 [+0.2%])</span> 2,371 | <span style='color: red'>(+4 [+0.2%])</span> 2,371 | <span style='color: red'>(+4 [+0.2%])</span> 2,371 |
| `generate_perm_trace_time_ms` |  494 |  494 |  494 |  494 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+65 [+1.3%])</span> 5,154 | <span style='color: red'>(+65 [+1.3%])</span> 5,154 | <span style='color: red'>(+65 [+1.3%])</span> 5,154 | <span style='color: red'>(+65 [+1.3%])</span> 5,154 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-75 [-2.9%])</span> 2,521 | <span style='color: green'>(-75 [-2.9%])</span> 2,521 | <span style='color: green'>(-75 [-2.9%])</span> 2,521 | <span style='color: green'>(-75 [-2.9%])</span> 2,521 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+0.2%])</span> 1,201 | <span style='color: red'>(+2 [+0.2%])</span> 1,201 | <span style='color: red'>(+2 [+0.2%])</span> 1,201 | <span style='color: red'>(+2 [+0.2%])</span> 1,201 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-37 [-1.4%])</span> 2,633 | <span style='color: green'>(-37 [-1.4%])</span> 2,633 | <span style='color: green'>(-37 [-1.4%])</span> 2,633 | <span style='color: green'>(-37 [-1.4%])</span> 2,633 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-12152 [-40.1%])</span> 18,178 | <span style='color: green'>(-12152 [-40.1%])</span> 18,178 | <span style='color: green'>(-12152 [-40.1%])</span> 18,178 | <span style='color: green'>(-12152 [-40.1%])</span> 18,178 |
| `main_cells_used     ` | <span style='color: green'>(-69919056 [-28.6%])</span> 174,238,455 | <span style='color: green'>(-69919056 [-28.6%])</span> 174,238,455 | <span style='color: green'>(-69919056 [-28.6%])</span> 174,238,455 | <span style='color: green'>(-69919056 [-28.6%])</span> 174,238,455 |
| `total_cycles        ` | <span style='color: green'>(-2450652 [-41.3%])</span> 3,483,861 | <span style='color: green'>(-2450652 [-41.3%])</span> 3,483,861 | <span style='color: green'>(-2450652 [-41.3%])</span> 3,483,861 | <span style='color: green'>(-2450652 [-41.3%])</span> 3,483,861 |
| `execute_time_ms     ` | <span style='color: green'>(-781 [-46.9%])</span> 886 | <span style='color: green'>(-781 [-46.9%])</span> 886 | <span style='color: green'>(-781 [-46.9%])</span> 886 | <span style='color: green'>(-781 [-46.9%])</span> 886 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1175 [-28.7%])</span> 2,914 | <span style='color: green'>(-1175 [-28.7%])</span> 2,914 | <span style='color: green'>(-1175 [-28.7%])</span> 2,914 | <span style='color: green'>(-1175 [-28.7%])</span> 2,914 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-10196 [-41.5%])</span> 14,378 | <span style='color: green'>(-10196 [-41.5%])</span> 14,378 | <span style='color: green'>(-10196 [-41.5%])</span> 14,378 | <span style='color: green'>(-10196 [-41.5%])</span> 14,378 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1920 [-43.1%])</span> 2,531 | <span style='color: green'>(-1920 [-43.1%])</span> 2,531 | <span style='color: green'>(-1920 [-43.1%])</span> 2,531 | <span style='color: green'>(-1920 [-43.1%])</span> 2,531 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-180 [-33.9%])</span> 351 | <span style='color: green'>(-180 [-33.9%])</span> 351 | <span style='color: green'>(-180 [-33.9%])</span> 351 | <span style='color: green'>(-180 [-33.9%])</span> 351 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1884 [-37.4%])</span> 3,156 | <span style='color: green'>(-1884 [-37.4%])</span> 3,156 | <span style='color: green'>(-1884 [-37.4%])</span> 3,156 | <span style='color: green'>(-1884 [-37.4%])</span> 3,156 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-1910 [-34.8%])</span> 3,575 | <span style='color: green'>(-1910 [-34.8%])</span> 3,575 | <span style='color: green'>(-1910 [-34.8%])</span> 3,575 | <span style='color: green'>(-1910 [-34.8%])</span> 3,575 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-1832 [-45.9%])</span> 2,158 | <span style='color: green'>(-1832 [-45.9%])</span> 2,158 | <span style='color: green'>(-1832 [-45.9%])</span> 2,158 | <span style='color: green'>(-1832 [-45.9%])</span> 2,158 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-2471 [-48.7%])</span> 2,603 | <span style='color: green'>(-2471 [-48.7%])</span> 2,603 | <span style='color: green'>(-2471 [-48.7%])</span> 2,603 | <span style='color: green'>(-2471 [-48.7%])</span> 2,603 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 1 | 624 | 40 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 4 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 4 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 4 | 35 | 59 | 
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
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 76 | 64 | 146,800,640 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 356 | 399 | 49,479,680 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
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
| leaf | 0 | 2,914 | 18,178 | 3,483,861 | 504,908,760 | 14,378 | 3,575 | 2,158 | 3,156 | 2,603 | 2,531 | 174,238,455 | 351 | 886 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 3,332 | 18,846 | 4,190,904 | 632,452,480 | 14,378 | 2,521 | 1,201 | 5,154 | 2,633 | 2,371 | 165,028,173 | 494 | 1,136 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/3c5c4a34ee5a0f5f91fb9ff5b57fb6cbd8ce773e

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12881616966)
