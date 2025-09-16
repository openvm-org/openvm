| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+0 [+18.5%])</span> 1.88 | <span style='color: green'>(-0 [-30.1%])</span> 1.11 |
| pairing | <span style='color: red'>(+0 [+19.8%])</span> 1.78 | <span style='color: green'>(-0 [-31.8%])</span> 1.02 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-598 [-40.1%])</span> 892.50 | <span style='color: red'>(+295 [+19.8%])</span> 1,785 | <span style='color: green'>(-474 [-31.8%])</span> 1,016 | <span style='color: green'>(-721 [-48.4%])</span> 769 |
| `main_cells_used     ` | <span style='color: green'>(-10798741 [-45.5%])</span> 12,923,881 | <span style='color: red'>(+2125140 [+9.0%])</span> 25,847,762 | <span style='color: green'>(-8708176 [-36.7%])</span> 15,014,446 | <span style='color: green'>(-12889306 [-54.3%])</span> 10,833,316 |
| `total_cells_used    ` | <span style='color: green'>(-16727849 [-37.6%])</span> 27,763,511 | <span style='color: red'>(+11035662 [+24.8%])</span> 55,527,022 | <span style='color: green'>(-13449416 [-30.2%])</span> 31,041,944 | <span style='color: green'>(-20006282 [-45.0%])</span> 24,485,078 |
| `execute_metered_time_ms` | <span style='color: green'>(-2 [-2.2%])</span> 90 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+1 [+2.5%])</span> 20.83 | -          | <span style='color: red'>(+1 [+2.5%])</span> 20.83 | <span style='color: red'>(+1 [+2.5%])</span> 20.83 |
| `execute_preflight_insns` | <span style='color: green'>(-941470 [-50.0%])</span> 941,469.50 |  1,882,939 | <span style='color: green'>(-739939 [-39.3%])</span> 1,143,000 | <span style='color: green'>(-1143000 [-60.7%])</span> 739,939 |
| `execute_preflight_time_ms` | <span style='color: green'>(-69 [-40.6%])</span> 101 | <span style='color: red'>(+32 [+18.8%])</span> 202 | <span style='color: green'>(-40 [-23.5%])</span> 130 | <span style='color: green'>(-98 [-57.6%])</span> 72 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+2 [+20.7%])</span> 13.71 | -          | <span style='color: red'>(+7 [+62.1%])</span> 18.40 | <span style='color: green'>(-2 [-20.6%])</span> 9.01 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+10 [+5.8%])</span> 190.50 | <span style='color: red'>(+201 [+111.7%])</span> 381 | <span style='color: red'>(+27 [+15.0%])</span> 207 | <span style='color: green'>(-6 [-3.3%])</span> 174 |
| `memory_finalize_time_ms` | <span style='color: green'>(-1 [-100.0%])</span> 0 | <span style='color: green'>(-1 [-100.0%])</span> 0 | <span style='color: green'>(-1 [-100.0%])</span> 0 | <span style='color: green'>(-1 [-100.0%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-488 [-49.1%])</span> 506 | <span style='color: red'>(+18 [+1.8%])</span> 1,012 | <span style='color: green'>(-429 [-43.2%])</span> 565 | <span style='color: green'>(-547 [-55.0%])</span> 447 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-80 [-53.7%])</span> 69.50 | <span style='color: green'>(-11 [-7.3%])</span> 139 | <span style='color: green'>(-74 [-49.3%])</span> 76 | <span style='color: green'>(-87 [-58.0%])</span> 63 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-3 [-4.5%])</span> 63 | <span style='color: red'>(+60 [+90.9%])</span> 126 | <span style='color: red'>(+16 [+24.2%])</span> 82 | <span style='color: green'>(-22 [-33.3%])</span> 44 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-89 [-52.0%])</span> 82.48 | <span style='color: green'>(-7 [-3.9%])</span> 164.95 | <span style='color: green'>(-80 [-46.4%])</span> 92.07 | <span style='color: green'>(-99 [-57.6%])</span> 72.88 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-50 [-34.3%])</span> 95.66 | <span style='color: red'>(+46 [+31.3%])</span> 191.31 | <span style='color: green'>(-44 [-30.0%])</span> 101.99 | <span style='color: green'>(-56 [-38.7%])</span> 89.32 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-16 [-45.7%])</span> 18.63 | <span style='color: red'>(+3 [+8.6%])</span> 37.27 | <span style='color: green'>(-15 [-42.5%])</span> 19.73 | <span style='color: green'>(-17 [-48.9%])</span> 17.54 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-247 [-58.8%])</span> 173 | <span style='color: green'>(-74 [-17.6%])</span> 346 | <span style='color: green'>(-231 [-55.0%])</span> 189 | <span style='color: green'>(-263 [-62.6%])</span> 157 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 58 | 837 | 2,031 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 769 | 40 | 1 | 90 | 1,882,939 | 20.83 | 149 | 

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
| pairing | AccessAdapterAir<8> | 0 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | AccessAdapterAir<8> | 1 | 262,144 |  | 16 | 17 | 8,650,752 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 16,384 |  | 8 | 300 | 5,046,272 | 
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
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 256 |  | 52 | 31 | 21,248 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 256 |  | 52 | 31 | 21,248 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 4,096 |  | 604 | 497 | 4,509,696 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 174 | 1,016 | 31,041,944 | 160,949,612 | 174 | 565 | 0 | 101.99 | 19.73 | 6 | 92.07 | 189 | 177 | 189 | 0 | 76 | 15,014,446 | 82 | 130 | 1,143,000 | 9.01 | 24 | 122 | 3 | 189 | 
| pairing | 1 | 207 | 769 | 24,485,078 | 124,079,782 | 207 | 447 | 1 | 89.32 | 17.54 | 6 | 72.88 | 157 | 119 | 157 | 0 | 63 | 10,833,316 | 44 | 72 | 739,939 | 18.40 | 19 | 107 | 2 | 157 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 2,824,598 | 2,013,265,921 | 
| pairing | 0 | 1 | 9,345,616 | 2,013,265,921 | 
| pairing | 0 | 2 | 1,412,299 | 2,013,265,921 | 
| pairing | 0 | 3 | 12,776,596 | 2,013,265,921 | 
| pairing | 0 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 0 | 6 | 3,143,696 | 2,013,265,921 | 
| pairing | 0 | 7 | 2,048 | 2,013,265,921 | 
| pairing | 0 | 8 | 30,569,813 | 2,013,265,921 | 
| pairing | 1 | 0 | 1,989,420 | 2,013,265,921 | 
| pairing | 1 | 1 | 7,060,944 | 2,013,265,921 | 
| pairing | 1 | 2 | 994,710 | 2,013,265,921 | 
| pairing | 1 | 3 | 9,423,576 | 2,013,265,921 | 
| pairing | 1 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 1 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 1 | 6 | 1,631,400 | 2,013,265,921 | 
| pairing | 1 | 7 | 2,048 | 2,013,265,921 | 
| pairing | 1 | 8 | 22,167,058 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/5e1f1e862831b4aa3b31c3ce1333aee2b026cf9f

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/17778093811)
