| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: green'>(-1 [-2.4%])</span> 21.25 | <span style='color: green'>(-1 [-2.4%])</span> 21.25 |
| fibonacci_program | <span style='color: red'>(+0 [+0.2%])</span> 6.20 | <span style='color: red'>(+0 [+0.2%])</span> 6.20 |
| leaf | <span style='color: green'>(-1 [-3.4%])</span> 15.05 | <span style='color: green'>(-1 [-3.4%])</span> 15.05 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+10 [+0.2%])</span> 6,199 | <span style='color: red'>(+10 [+0.2%])</span> 6,199 | <span style='color: red'>(+10 [+0.2%])</span> 6,199 | <span style='color: red'>(+10 [+0.2%])</span> 6,199 |
| `main_cells_used     ` |  51,505,102 |  51,505,102 |  51,505,102 |  51,505,102 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: green'>(-2 [-0.5%])</span> 431 | <span style='color: green'>(-2 [-0.5%])</span> 431 | <span style='color: green'>(-2 [-0.5%])</span> 431 | <span style='color: green'>(-2 [-0.5%])</span> 431 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+7 [+0.8%])</span> 874 | <span style='color: red'>(+7 [+0.8%])</span> 874 | <span style='color: red'>(+7 [+0.8%])</span> 874 | <span style='color: red'>(+7 [+0.8%])</span> 874 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+5 [+0.1%])</span> 4,894 | <span style='color: red'>(+5 [+0.1%])</span> 4,894 | <span style='color: red'>(+5 [+0.1%])</span> 4,894 | <span style='color: red'>(+5 [+0.1%])</span> 4,894 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-8 [-1.1%])</span> 753 | <span style='color: green'>(-8 [-1.1%])</span> 753 | <span style='color: green'>(-8 [-1.1%])</span> 753 | <span style='color: green'>(-8 [-1.1%])</span> 753 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+4 [+2.2%])</span> 182 | <span style='color: red'>(+4 [+2.2%])</span> 182 | <span style='color: red'>(+4 [+2.2%])</span> 182 | <span style='color: red'>(+4 [+2.2%])</span> 182 |
| `perm_trace_commit_time_ms` |  1,643 |  1,643 |  1,643 |  1,643 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+2 [+0.3%])</span> 794 | <span style='color: red'>(+2 [+0.3%])</span> 794 | <span style='color: red'>(+2 [+0.3%])</span> 794 | <span style='color: red'>(+2 [+0.3%])</span> 794 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+27 [+5.8%])</span> 494 | <span style='color: red'>(+27 [+5.8%])</span> 494 | <span style='color: red'>(+27 [+5.8%])</span> 494 | <span style='color: red'>(+27 [+5.8%])</span> 494 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-20 [-1.9%])</span> 1,026 | <span style='color: green'>(-20 [-1.9%])</span> 1,026 | <span style='color: green'>(-20 [-1.9%])</span> 1,026 | <span style='color: green'>(-20 [-1.9%])</span> 1,026 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-525 [-3.4%])</span> 15,051 | <span style='color: green'>(-525 [-3.4%])</span> 15,051 | <span style='color: green'>(-525 [-3.4%])</span> 15,051 | <span style='color: green'>(-525 [-3.4%])</span> 15,051 |
| `main_cells_used     ` | <span style='color: red'>(+8538965 [+6.6%])</span> 137,419,712 | <span style='color: red'>(+8538965 [+6.6%])</span> 137,419,712 | <span style='color: red'>(+8538965 [+6.6%])</span> 137,419,712 | <span style='color: red'>(+8538965 [+6.6%])</span> 137,419,712 |
| `total_cycles        ` | <span style='color: red'>(+647352 [+20.4%])</span> 3,820,759 | <span style='color: red'>(+647352 [+20.4%])</span> 3,820,759 | <span style='color: red'>(+647352 [+20.4%])</span> 3,820,759 | <span style='color: red'>(+647352 [+20.4%])</span> 3,820,759 |
| `execute_time_ms     ` | <span style='color: red'>(+112 [+9.4%])</span> 1,303 | <span style='color: red'>(+112 [+9.4%])</span> 1,303 | <span style='color: red'>(+112 [+9.4%])</span> 1,303 | <span style='color: red'>(+112 [+9.4%])</span> 1,303 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+160 [+6.9%])</span> 2,490 | <span style='color: red'>(+160 [+6.9%])</span> 2,490 | <span style='color: red'>(+160 [+6.9%])</span> 2,490 | <span style='color: red'>(+160 [+6.9%])</span> 2,490 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-797 [-6.6%])</span> 11,258 | <span style='color: green'>(-797 [-6.6%])</span> 11,258 | <span style='color: green'>(-797 [-6.6%])</span> 11,258 | <span style='color: green'>(-797 [-6.6%])</span> 11,258 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-155 [-6.7%])</span> 2,170 | <span style='color: green'>(-155 [-6.7%])</span> 2,170 | <span style='color: green'>(-155 [-6.7%])</span> 2,170 | <span style='color: green'>(-155 [-6.7%])</span> 2,170 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-67 [-20.1%])</span> 266 | <span style='color: green'>(-67 [-20.1%])</span> 266 | <span style='color: green'>(-67 [-20.1%])</span> 266 | <span style='color: green'>(-67 [-20.1%])</span> 266 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-82 [-4.1%])</span> 1,895 | <span style='color: green'>(-82 [-4.1%])</span> 1,895 | <span style='color: green'>(-82 [-4.1%])</span> 1,895 | <span style='color: green'>(-82 [-4.1%])</span> 1,895 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-400 [-13.2%])</span> 2,624 | <span style='color: green'>(-400 [-13.2%])</span> 2,624 | <span style='color: green'>(-400 [-13.2%])</span> 2,624 | <span style='color: green'>(-400 [-13.2%])</span> 2,624 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-36 [-1.8%])</span> 1,959 | <span style='color: green'>(-36 [-1.8%])</span> 1,959 | <span style='color: green'>(-36 [-1.8%])</span> 1,959 | <span style='color: green'>(-36 [-1.8%])</span> 1,959 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-56 [-2.3%])</span> 2,342 | <span style='color: green'>(-56 [-2.3%])</span> 2,342 | <span style='color: green'>(-56 [-2.3%])</span> 2,342 | <span style='color: green'>(-56 [-2.3%])</span> 2,342 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 343 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 
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
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 15 | 24 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 15 | 23 | 
| leaf | VmConnectorAir | 4 | 3 | 8 | 
| leaf | VolatileBoundaryAir | 4 | 4 | 16 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 20 | 31 | 106,954,752 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 2,490 | 15,051 | 3,820,759 | 340,134,360 | 11,258 | 2,624 | 1,959 | 1,895 | 2,342 | 2,170 | 137,419,712 | 266 | 1,303 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 874 | 6,199 | 1,500,137 | 197,453,854 | 4,894 | 794 | 494 | 1,643 | 1,026 | 753 | 51,505,102 | 182 | 431 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/b242cd7127d2ccdcede0738539aeeaa731a1d8bb

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12645839014)