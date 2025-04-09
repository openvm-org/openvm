| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+6 [+76.9%])</span> 14.41 | <span style='color: red'>(+3 [+75.5%])</span> 7.93 |
| regex_program | <span style='color: red'>(+6 [+76.9%])</span> 14.41 | <span style='color: red'>(+3 [+75.5%])</span> 7.93 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+3132 [+76.9%])</span> 7,204 | <span style='color: red'>(+6264 [+76.9%])</span> 14,408 | <span style='color: red'>(+3409 [+75.5%])</span> 7,927 | <span style='color: red'>(+2855 [+78.7%])</span> 6,481 |
| `main_cells_used     ` |  83,255,576 |  166,511,152 |  93,500,799 |  73,010,353 |
| `total_cycles        ` |  2,082,613 |  4,165,226 |  2,243,715 |  1,921,511 |
| `execute_time_ms     ` | <span style='color: red'>(+0 [+0.1%])</span> 340 | <span style='color: red'>(+1 [+0.1%])</span> 680 | <span style='color: green'>(-3 [-0.8%])</span> 376 | <span style='color: red'>(+4 [+1.3%])</span> 304 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-45 [-4.5%])</span> 945.50 | <span style='color: green'>(-90 [-4.5%])</span> 1,891 | <span style='color: green'>(-21 [-1.8%])</span> 1,144 | <span style='color: green'>(-69 [-8.5%])</span> 747 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+3176 [+115.8%])</span> 5,918.50 | <span style='color: red'>(+6353 [+115.8%])</span> 11,837 | <span style='color: red'>(+3433 [+115.4%])</span> 6,407 | <span style='color: red'>(+2920 [+116.3%])</span> 5,430 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+47 [+10.0%])</span> 519 | <span style='color: red'>(+94 [+10.0%])</span> 1,038 | <span style='color: red'>(+49 [+9.2%])</span> 579 | <span style='color: red'>(+45 [+10.9%])</span> 459 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+142 [+75.2%])</span> 332 | <span style='color: red'>(+285 [+75.2%])</span> 664 | <span style='color: red'>(+144 [+75.4%])</span> 335 | <span style='color: red'>(+141 [+75.0%])</span> 329 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-382 [-55.6%])</span> 306 | <span style='color: green'>(-765 [-55.6%])</span> 612 | <span style='color: green'>(-414 [-55.9%])</span> 326 | <span style='color: green'>(-351 [-55.1%])</span> 286 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+771 [+194.2%])</span> 1,168 | <span style='color: red'>(+1542 [+194.2%])</span> 2,336 | <span style='color: red'>(+818 [+188.0%])</span> 1,253 | <span style='color: red'>(+724 [+201.7%])</span> 1,083 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+40 [+14.2%])</span> 317.50 | <span style='color: red'>(+79 [+14.2%])</span> 635 | <span style='color: red'>(+45 [+14.4%])</span> 357 | <span style='color: red'>(+34 [+13.9%])</span> 278 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1218 [+172.0%])</span> 1,926 | <span style='color: red'>(+2436 [+172.0%])</span> 3,852 | <span style='color: red'>(+1333 [+176.3%])</span> 2,089 | <span style='color: red'>(+1103 [+167.1%])</span> 1,763 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 516 | 20 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 10 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 10 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 10 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 10 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 10 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,251 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 37 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| regex_program | PhantomAir | 2 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 2 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 2 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 18 | 19 | 
| regex_program | VariableRangeCheckerAir | 2 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 26 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 32 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 80 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 15 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 29 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 13 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 13 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 22 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 29 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 68 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 15 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 8 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 9 | 
| regex_program | VmConnectorAir | 2 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 12 | 17 | 59,392 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 12 | 2 | 917,504 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 12 | 3,163 | 3,175 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 12 | 3,163 | 101,600 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 12 | 32 | 180,224 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 12 | 300 | 5,111,808 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 12 | 300 | 638,976 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 12 | 10 | 2,883,584 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 12 | 1 | 6,815,744 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 12 | 1 | 6,815,744 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 12 | 32 | 720,896 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 12 | 36 | 50,331,648 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 12 | 36 | 50,331,648 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 12 | 37 | 1,605,632 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 12 | 37 | 802,816 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 12 | 53 | 8,519,680 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 12 | 53 | 8,519,680 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 12 | 26 | 9,961,472 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 12 | 26 | 4,980,736 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 12 | 18 | 1,966,080 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 12 | 18 | 1,966,080 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 12 | 28 | 5,242,880 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 12 | 28 | 2,621,440 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 12 | 36 | 49,152 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 12 | 36 | 96 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 12 | 41 | 55,574,528 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 12 | 41 | 55,574,528 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 12 | 59 | 9,088 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 12 | 39 | 13,056 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 12 | 31 | 1,409,024 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 12 | 31 | 1,409,024 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 12 | 20 | 1,048,576 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_gen_layers_ms | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,144 | 7,927 | 2,243,715 | 175,121,179 | 351 | 6,407 | 1,253 | 357 | 326 | 2,089 | 579 | 93,500,799 | 1,237 | 166 | 329 | 376 | 
| regex_program | 1 | 747 | 6,481 | 1,921,511 | 148,094,548 | 324 | 5,430 | 1,083 | 278 | 286 | 1,763 | 459 | 73,010,353 | 1,056 | 132 | 335 | 304 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 5,868,296 | 2,013,265,921 | 
| regex_program | 0 | 1 | 16,687,450 | 2,013,265,921 | 
| regex_program | 0 | 2 | 2,934,148 | 2,013,265,921 | 
| regex_program | 0 | 3 | 19,705,182 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,938 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 53,849,550 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,406,794 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,182,956 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,703,397 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,193,430 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,508,864 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,197,617 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/dcac4d9ffbafcf16b326637d955aac17736c9337

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14360224821)
