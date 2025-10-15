| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+3.7%])</span> 1.57 | <span style='color: red'>(+0 [+2.2%])</span> 0.92 |
| pairing | <span style='color: red'>(+0 [+3.9%])</span> 1.48 | <span style='color: red'>(+0 [+2.3%])</span> 0.83 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+28 [+3.9%])</span> 738 | <span style='color: red'>(+55 [+3.9%])</span> 1,476 | <span style='color: red'>(+19 [+2.3%])</span> 832 | <span style='color: red'>(+36 [+5.9%])</span> 644 |
| `main_cells_used     ` |  12,734,105 |  25,468,210 |  16,059,916 |  9,408,294 |
| `total_cells_used    ` |  27,501,699 |  55,003,398 |  32,689,110 |  22,314,288 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+1.1%])</span> 90 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-0 [-1.1%])</span> 19.21 | -          | <span style='color: green'>(-0 [-1.1%])</span> 19.21 | <span style='color: green'>(-0 [-1.1%])</span> 19.21 |
| `execute_preflight_insns` |  872,871 |  1,745,742 |  1,157,000 |  588,742 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+0.5%])</span> 95.50 | <span style='color: red'>(+1 [+0.5%])</span> 191 |  130 | <span style='color: red'>(+1 [+1.7%])</span> 61 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-0.4%])</span> 13.55 | -          | <span style='color: green'>(-0 [-0.5%])</span> 18.01 | <span style='color: green'>(-0 [-0.4%])</span> 9.09 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+2 [+1.1%])</span> 178 | <span style='color: red'>(+4 [+1.1%])</span> 356 | <span style='color: red'>(+1 [+0.5%])</span> 194 | <span style='color: red'>(+3 [+1.9%])</span> 162 |
| `memory_finalize_time_ms` | <span style='color: red'>(+0 [+inf%])</span> 0.50 | <span style='color: red'>(+1 [+inf%])</span> 1 | <span style='color: red'>(+1 [+inf%])</span> 1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+26 [+7.4%])</span> 371 | <span style='color: red'>(+51 [+7.4%])</span> 742 | <span style='color: red'>(+17 [+4.5%])</span> 395 | <span style='color: red'>(+34 [+10.9%])</span> 347 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+0 [+0.8%])</span> 64.50 | <span style='color: red'>(+1 [+0.8%])</span> 129 | <span style='color: red'>(+1 [+1.4%])</span> 75 |  54 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+25.4%])</span> 44.50 | <span style='color: red'>(+18 [+25.4%])</span> 89 | <span style='color: red'>(+12 [+31.6%])</span> 50 | <span style='color: red'>(+6 [+18.2%])</span> 39 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.7%])</span> 74.97 | <span style='color: green'>(-3 [-1.7%])</span> 149.94 | <span style='color: green'>(-2 [-2.8%])</span> 85.29 | <span style='color: green'>(-0 [-0.2%])</span> 64.65 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+0 [+0.4%])</span> 78.66 | <span style='color: red'>(+1 [+0.4%])</span> 157.32 | <span style='color: green'>(-2 [-2.6%])</span> 84.95 | <span style='color: red'>(+3 [+4.3%])</span> 72.37 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+0 [+2.9%])</span> 16.10 | <span style='color: red'>(+1 [+2.9%])</span> 32.20 | <span style='color: green'>(-0 [-0.6%])</span> 17.53 | <span style='color: red'>(+1 [+7.4%])</span> 14.67 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+16 [+23.4%])</span> 87 | <span style='color: red'>(+33 [+23.4%])</span> 174 | <span style='color: red'>(+24 [+32.9%])</span> 97 | <span style='color: red'>(+9 [+13.2%])</span> 77 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 843 | 1,718 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 644 | 40 | 1 | 90 | 1,745,742 | 19.21 | 145 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
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

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | AccessAdapterAir<16> | 1 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | AccessAdapterAir<32> | 0 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | AccessAdapterAir<32> | 1 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 8,192 |  | 8 | 300 | 2,523,136 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 262,144 |  | 52 | 36 | 23,068,672 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 512 |  | 52 | 53 | 53,760 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 16 |  | 56 | 166 | 3,552 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16,384 |  | 36 | 28 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 64 |  | 72 | 39 | 7,104 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 256 |  | 52 | 31 | 21,248 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 8,192 |  | 28 | 20 | 393,216 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 256 |  | 320 | 263 | 149,248 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 604 | 497 | 4,509,696 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 162 | 832 | 32,689,110 | 172,558,444 | 162 | 395 | 0 | 84.95 | 17.53 | 6 | 85.29 | 77 | 139 | 77 | 1 | 75 | 16,059,916 | 50 | 130 | 1,157,000 | 9.09 | 26 | 103 | 1 | 77 | 
| pairing | 1 | 194 | 644 | 22,314,288 | 119,958,502 | 194 | 347 | 1 | 72.37 | 14.67 | 6 | 64.65 | 97 | 107 | 97 | 0 | 54 | 9,408,294 | 39 | 61 | 588,742 | 18.01 | 21 | 87 | 1 | 97 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 2,833,302 | 2,013,265,921 | 
| pairing | 0 | 1 | 10,207,312 | 2,013,265,921 | 
| pairing | 0 | 2 | 1,416,651 | 2,013,265,921 | 
| pairing | 0 | 3 | 13,908,628 | 2,013,265,921 | 
| pairing | 0 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 0 | 6 | 3,151,888 | 2,013,265,921 | 
| pairing | 0 | 7 | 3,072 | 2,013,265,921 | 
| pairing | 0 | 8 | 32,585,813 | 2,013,265,921 | 
| pairing | 1 | 0 | 1,939,628 | 2,013,265,921 | 
| pairing | 1 | 1 | 6,975,568 | 2,013,265,921 | 
| pairing | 1 | 2 | 969,814 | 2,013,265,921 | 
| pairing | 1 | 3 | 9,239,640 | 2,013,265,921 | 
| pairing | 1 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 1 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 1 | 6 | 1,573,480 | 2,013,265,921 | 
| pairing | 1 | 7 | 1,536 | 2,013,265,921 | 
| pairing | 1 | 8 | 21,756,434 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/0f7c204d7b749ce600fef91ad54809f11a60dada

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/18542915278)
