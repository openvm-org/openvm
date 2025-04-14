| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+7 [+80.7%])</span> 14.63 | <span style='color: red'>(+4 [+78.0%])</span> 8 |
| regex_program | <span style='color: red'>(+7 [+80.7%])</span> 14.63 | <span style='color: red'>(+4 [+78.0%])</span> 8 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+3268 [+80.7%])</span> 7,317 | <span style='color: red'>(+6537 [+80.7%])</span> 14,634 | <span style='color: red'>(+3508 [+78.0%])</span> 8,004 | <span style='color: red'>(+3029 [+84.1%])</span> 6,630 |
| `main_cells_used     ` |  83,255,576 |  166,511,152 |  93,500,799 |  73,010,353 |
| `total_cycles        ` |  2,082,613 |  4,165,226 |  2,243,715 |  1,921,511 |
| `execute_time_ms     ` | <span style='color: red'>(+1 [+0.3%])</span> 341 | <span style='color: red'>(+2 [+0.3%])</span> 682 | <span style='color: red'>(+1 [+0.3%])</span> 379 | <span style='color: red'>(+1 [+0.3%])</span> 303 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+0.2%])</span> 971 | <span style='color: red'>(+3 [+0.2%])</span> 1,942 | <span style='color: red'>(+23 [+2.0%])</span> 1,175 | <span style='color: green'>(-20 [-2.5%])</span> 767 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+3266 [+119.2%])</span> 6,005 | <span style='color: red'>(+6532 [+119.2%])</span> 12,010 | <span style='color: red'>(+3484 [+117.5%])</span> 6,450 | <span style='color: red'>(+3048 [+121.3%])</span> 5,560 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+2 [+0.4%])</span> 475.50 | <span style='color: red'>(+4 [+0.4%])</span> 951 | <span style='color: green'>(-2 [-0.4%])</span> 533 | <span style='color: red'>(+6 [+1.5%])</span> 418 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-32 [-16.9%])</span> 157 | <span style='color: green'>(-64 [-16.9%])</span> 314 | <span style='color: green'>(-28 [-14.7%])</span> 162 | <span style='color: green'>(-36 [-19.1%])</span> 152 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-404 [-58.3%])</span> 288.50 | <span style='color: green'>(-807 [-58.3%])</span> 577 | <span style='color: green'>(-431 [-58.2%])</span> 310 | <span style='color: green'>(-376 [-58.5%])</span> 267 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+815 [+205.3%])</span> 1,212 | <span style='color: red'>(+1630 [+205.3%])</span> 2,424 | <span style='color: red'>(+866 [+201.9%])</span> 1,295 | <span style='color: red'>(+764 [+209.3%])</span> 1,129 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+26 [+8.9%])</span> 311.50 | <span style='color: red'>(+51 [+8.9%])</span> 623 | <span style='color: red'>(+28 [+8.5%])</span> 356 | <span style='color: red'>(+23 [+9.4%])</span> 267 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1223 [+176.9%])</span> 1,914.50 | <span style='color: red'>(+2446 [+176.9%])</span> 3,829 | <span style='color: red'>(+1312 [+179.2%])</span> 2,044 | <span style='color: red'>(+1134 [+174.2%])</span> 1,785 |
| `sumcheck_prove_batch_ms` |  1,011 |  2,022 |  1,061 |  961 |
| `gkr_prove_batch_ms  ` |  1,230 |  2,460 |  1,299 |  1,161 |
| `gkr_gen_layers_ms   ` |  150 |  300 |  167 |  133 |
| `gkr_generate_aux    ` |  328 |  656 |  334 |  322 |
| `gkr_build_instances_ms` |  110 |  220 |  119 |  101 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 522 | 19 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_generate_aux | gkr_gen_layers_ms | gkr_build_instances_ms | generate_perm_trace_time_ms | execute_time_ms | build_gkr_input_layer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,175 | 8,004 | 2,243,715 | 175,121,179 | 1,061 | 6,450 | 1,295 | 356 | 310 | 2,044 | 533 | 93,500,799 | 1,299 | 334 | 167 | 119 | 162 | 379 | 148 | 
| regex_program | 1 | 767 | 6,630 | 1,921,511 | 148,094,548 | 961 | 5,560 | 1,129 | 267 | 267 | 1,785 | 418 | 73,010,353 | 1,161 | 322 | 133 | 101 | 152 | 303 | 99 | 

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


Commit: https://github.com/openvm-org/openvm/commit/a6845bad6ffa35551eb66bf0b4b159bd77cdd105

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14449987733)
