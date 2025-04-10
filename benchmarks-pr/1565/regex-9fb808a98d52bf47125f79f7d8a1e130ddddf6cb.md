| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+6 [+77.7%])</span> 14.47 | <span style='color: red'>(+3 [+73.1%])</span> 7.82 |
| regex_program | <span style='color: red'>(+6 [+77.7%])</span> 14.47 | <span style='color: red'>(+3 [+73.1%])</span> 7.82 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+3163 [+77.7%])</span> 7,235 | <span style='color: red'>(+6326 [+77.7%])</span> 14,470 | <span style='color: red'>(+3304 [+73.1%])</span> 7,822 | <span style='color: red'>(+3022 [+83.3%])</span> 6,648 |
| `main_cells_used     ` |  83,255,576 |  166,511,152 |  93,500,799 |  73,010,353 |
| `total_cycles        ` |  2,082,613 |  4,165,226 |  2,243,715 |  1,921,511 |
| `execute_time_ms     ` | <span style='color: green'>(-8 [-2.2%])</span> 332 | <span style='color: green'>(-15 [-2.2%])</span> 664 | <span style='color: green'>(-10 [-2.6%])</span> 369 | <span style='color: green'>(-5 [-1.7%])</span> 295 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-36 [-3.6%])</span> 954.50 | <span style='color: green'>(-72 [-3.6%])</span> 1,909 | <span style='color: green'>(-50 [-4.3%])</span> 1,115 | <span style='color: green'>(-22 [-2.7%])</span> 794 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+3206 [+116.9%])</span> 5,948.50 | <span style='color: red'>(+6413 [+116.9%])</span> 11,897 | <span style='color: red'>(+3364 [+113.1%])</span> 6,338 | <span style='color: red'>(+3049 [+121.5%])</span> 5,559 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+48 [+10.3%])</span> 520.50 | <span style='color: red'>(+97 [+10.3%])</span> 1,041 | <span style='color: red'>(+53 [+10.0%])</span> 583 | <span style='color: red'>(+44 [+10.6%])</span> 458 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-382 [-55.4%])</span> 307 | <span style='color: green'>(-763 [-55.4%])</span> 614 | <span style='color: green'>(-413 [-55.8%])</span> 327 | <span style='color: green'>(-350 [-54.9%])</span> 287 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+774 [+195.0%])</span> 1,171 | <span style='color: red'>(+1548 [+195.0%])</span> 2,342 | <span style='color: red'>(+825 [+189.7%])</span> 1,260 | <span style='color: red'>(+723 [+201.4%])</span> 1,082 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+40 [+14.6%])</span> 318.50 | <span style='color: red'>(+81 [+14.6%])</span> 637 | <span style='color: red'>(+47 [+15.1%])</span> 359 | <span style='color: red'>(+34 [+13.9%])</span> 278 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1212 [+171.3%])</span> 1,920.50 | <span style='color: red'>(+2425 [+171.3%])</span> 3,841 | <span style='color: red'>(+1315 [+173.9%])</span> 2,071 | <span style='color: red'>(+1110 [+168.2%])</span> 1,770 |
| `sumcheck_prove_batch_ms` |  954.50 |  1,909 |  966 |  943 |
| `gkr_prove_batch_ms  ` |  1,169 |  2,338 |  1,172 |  1,166 |
| `gkr_gen_layers_ms   ` |  147 |  294 |  159 |  135 |



<details>
<summary>Detailed Metrics</summary>

|  | generate_perm_trace_time_ms |
| --- |
|  | 149 | 

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 520 | 20 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_gen_layers_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,115 | 7,822 | 2,243,715 | 175,121,179 | 943 | 6,338 | 1,260 | 359 | 327 | 2,071 | 583 | 93,500,799 | 1,166 | 159 | 369 | 
| regex_program | 1 | 794 | 6,648 | 1,921,511 | 148,094,548 | 966 | 5,559 | 1,082 | 278 | 287 | 1,770 | 458 | 73,010,353 | 1,172 | 135 | 295 | 

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


Commit: https://github.com/openvm-org/openvm/commit/9fb808a98d52bf47125f79f7d8a1e130ddddf6cb

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14382779015)
