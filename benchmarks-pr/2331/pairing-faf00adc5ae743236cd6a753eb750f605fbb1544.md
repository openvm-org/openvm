| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time (32 provers) (s) |
|:---|---:|---:|---:|
| Total | <span style='color: red'>(+1 [+51.0%])</span> 2.32 | <span style='color: red'>(+0 [+44.3%])</span> 1.35 | 1.35 |
| pairing | <span style='color: red'>(+1 [+51.0%])</span> 2.32 | <span style='color: red'>(+0 [+44.3%])</span> 1.35 |  1.35 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+392 [+53.2%])</span> 1,129 | <span style='color: red'>(+784 [+53.2%])</span> 2,258 | <span style='color: red'>(+414 [+47.5%])</span> 1,285 | <span style='color: red'>(+370 [+61.4%])</span> 973 |
| `main_cells_used     ` |  12,704,151 |  25,408,302 |  15,909,408 |  9,498,894 |
| `total_cells_used    ` |  27,472,173 |  54,944,346 |  32,480,114 |  22,464,232 |
| `execute_metered_time_ms` |  63 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+0.2%])</span> 27.49 | -          | <span style='color: red'>(+0 [+0.2%])</span> 27.49 | <span style='color: red'>(+0 [+0.2%])</span> 27.49 |
| `execute_preflight_insns` |  872,871 |  1,745,742 |  1,149,000 |  596,742 |
| `execute_preflight_time_ms` | <span style='color: red'>(+0 [+0.6%])</span> 84 | <span style='color: red'>(+1 [+0.6%])</span> 168 | <span style='color: red'>(+1 [+0.9%])</span> 107 |  61 |
| `execute_preflight_insn_mi/s` | <span style='color: green'>(-0 [-1.1%])</span> 14.58 | -          | <span style='color: green'>(-0 [-0.8%])</span> 18.08 | <span style='color: green'>(-0 [-1.5%])</span> 11.08 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-4 [-2.2%])</span> 182 | <span style='color: green'>(-8 [-2.2%])</span> 364 | <span style='color: green'>(-3 [-1.6%])</span> 189 | <span style='color: green'>(-5 [-2.8%])</span> 175 |
| `memory_finalize_time_ms` |  0.50 |  1 |  1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+396 [+109.8%])</span> 756.50 | <span style='color: red'>(+792 [+109.8%])</span> 1,513 | <span style='color: red'>(+419 [+101.5%])</span> 832 | <span style='color: red'>(+373 [+121.1%])</span> 681 |
| `main_trace_commit_time_ms` |  64.50 |  129 |  75 |  54 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-6.7%])</span> 42 | <span style='color: green'>(-6 [-6.7%])</span> 84 | <span style='color: green'>(-9 [-17.0%])</span> 44 | <span style='color: red'>(+3 [+8.1%])</span> 40 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-1 [-1.0%])</span> 76.18 | <span style='color: green'>(-2 [-1.0%])</span> 152.35 | <span style='color: green'>(-2 [-2.3%])</span> 87.44 | <span style='color: red'>(+0 [+0.7%])</span> 64.91 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-2.1%])</span> 82.40 | <span style='color: green'>(-4 [-2.1%])</span> 164.79 | <span style='color: green'>(-1 [-1.6%])</span> 90.91 | <span style='color: green'>(-2 [-2.7%])</span> 73.88 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-0 [-2.5%])</span> 16.47 | <span style='color: green'>(-1 [-2.5%])</span> 32.94 | <span style='color: green'>(-1 [-3.6%])</span> 17.93 | <span style='color: green'>(-0 [-1.3%])</span> 15.01 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+403 [+597.0%])</span> 470.50 | <span style='color: red'>(+806 [+597.0%])</span> 941 | <span style='color: red'>(+434 [+549.4%])</span> 513 | <span style='color: red'>(+372 [+664.3%])</span> 428 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 743 | 2,327 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| pairing | 973 | 1 | 63 | 1,745,742 | 27.49 | 0 | 

| group | air_id | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 0 | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | 1 | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 1 | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | 11 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 604 | 497 | 4,509,696 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 348 | 369 | 5,873,664 | 
| pairing | 12 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 348 | 369 | 2,936,832 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| pairing | 16 | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 16 |  | 56 | 166 | 3,552 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | 17 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 256 |  | 320 | 263 | 149,248 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 192 | 199 | 25,024 | 
| pairing | 18 | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 4 |  | 192 | 199 | 1,564 | 
| pairing | 2 | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 2 | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | 20 | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 64 |  | 72 | 39 | 7,104 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | 21 | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 256 |  | 52 | 31 | 21,248 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 22 | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | 23 | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | 24 | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 8,192 |  | 28 | 20 | 393,216 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | 25 | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 16,384 |  | 36 | 28 | 1,048,576 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | 26 | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | 27 | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 65,536 |  | 32 | 32 | 4,194,304 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | 28 | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 65,536 |  | 28 | 26 | 3,538,944 | 
| pairing | 3 | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 3 | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 30 | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | 31 | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 512 |  | 52 | 53 | 53,760 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | 32 | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,384 |  | 40 | 37 | 1,261,568 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | 33 | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 262,144 |  | 52 | 36 | 23,068,672 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 34 | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | 35 | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | 36 | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 8,192 |  | 8 | 300 | 2,523,136 | 
| pairing | 37 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 37 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | 6 | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | 6 | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | 7 | AccessAdapterAir<16> | 0 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 7 | AccessAdapterAir<16> | 1 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | 8 | AccessAdapterAir<32> | 0 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | 8 | AccessAdapterAir<32> | 1 | 65,536 |  | 16 | 41 | 3,735,552 | 

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

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 175 | 1,285 | 32,480,114 | 172,558,444 | 175 | 832 | 0 | 90.91 | 17.93 | 9 | 87.44 | 513 | 134 | 512 | 1 | 75 | 15,909,408 | 44 | 107 | 1,149,000 | 11.08 | 26 | 109 | 1 | 512 | 
| pairing | 1 | 189 | 973 | 22,464,232 | 119,958,502 | 189 | 681 | 1 | 73.88 | 15.01 | 8 | 64.91 | 428 | 108 | 427 | 0 | 54 | 9,498,894 | 40 | 61 | 596,742 | 18.08 | 20 | 89 | 1 | 427 | 

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


Commit: https://github.com/openvm-org/openvm/commit/faf00adc5ae743236cd6a753eb750f605fbb1544

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/20473415060)
