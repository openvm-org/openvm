| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+3 [+65.9%])</span> 8.09 | <span style='color: red'>(+3 [+65.9%])</span> 8.09 |
| pairing | <span style='color: red'>(+3 [+65.9%])</span> 8.09 | <span style='color: red'>(+3 [+65.9%])</span> 8.09 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+3214 [+65.9%])</span> 8,094 | <span style='color: red'>(+3214 [+65.9%])</span> 8,094 | <span style='color: red'>(+3214 [+65.9%])</span> 8,094 | <span style='color: red'>(+3214 [+65.9%])</span> 8,094 |
| `main_cells_used     ` |  95,832,407 |  95,832,407 |  95,832,407 |  95,832,407 |
| `total_cycles        ` |  1,820,436 |  1,820,436 |  1,820,436 |  1,820,436 |
| `execute_time_ms     ` | <span style='color: green'>(-4 [-0.6%])</span> 650 | <span style='color: green'>(-4 [-0.6%])</span> 650 | <span style='color: green'>(-4 [-0.6%])</span> 650 | <span style='color: green'>(-4 [-0.6%])</span> 650 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+21 [+2.3%])</span> 951 | <span style='color: red'>(+21 [+2.3%])</span> 951 | <span style='color: red'>(+21 [+2.3%])</span> 951 | <span style='color: red'>(+21 [+2.3%])</span> 951 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+3197 [+97.0%])</span> 6,493 | <span style='color: red'>(+3197 [+97.0%])</span> 6,493 | <span style='color: red'>(+3197 [+97.0%])</span> 6,493 | <span style='color: red'>(+3197 [+97.0%])</span> 6,493 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-49 [-7.5%])</span> 602 | <span style='color: green'>(-49 [-7.5%])</span> 602 | <span style='color: green'>(-49 [-7.5%])</span> 602 | <span style='color: green'>(-49 [-7.5%])</span> 602 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-532 [-63.6%])</span> 305 | <span style='color: green'>(-532 [-63.6%])</span> 305 | <span style='color: green'>(-532 [-63.6%])</span> 305 | <span style='color: green'>(-532 [-63.6%])</span> 305 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+854 [+181.7%])</span> 1,324 | <span style='color: red'>(+854 [+181.7%])</span> 1,324 | <span style='color: red'>(+854 [+181.7%])</span> 1,324 | <span style='color: red'>(+854 [+181.7%])</span> 1,324 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-3 [-0.8%])</span> 350 | <span style='color: green'>(-3 [-0.8%])</span> 350 | <span style='color: green'>(-3 [-0.8%])</span> 350 | <span style='color: green'>(-3 [-0.8%])</span> 350 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+1340 [+173.8%])</span> 2,111 | <span style='color: red'>(+1340 [+173.8%])</span> 2,111 | <span style='color: red'>(+1340 [+173.8%])</span> 2,111 | <span style='color: red'>(+1340 [+173.8%])</span> 2,111 |
| `sumcheck_prove_batch_ms` |  978 |  978 |  978 |  978 |
| `gkr_prove_batch_ms  ` |  1,219 |  1,219 |  1,219 |  1,219 |
| `gkr_gen_layers_ms   ` |  176 |  176 |  176 |  176 |



<details>
<summary>Detailed Metrics</summary>

|  | generate_perm_trace_time_ms |
| --- |
|  | 107 | 

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| pairing | 1 | 910 | 10 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 2 | 5 | 10 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 10 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 10 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 10 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 10 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | KeccakVmAir | 2 | 321 | 4,251 | 
| pairing | MemoryMerkleAir<8> | 2 | 4 | 37 | 
| pairing | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| pairing | PhantomAir | 2 | 3 | 4 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| pairing | ProgramAir | 2 | 1 | 4 | 
| pairing | RangeTupleCheckerAir<2> | 2 | 1 | 4 | 
| pairing | Rv32HintStoreAir | 2 | 18 | 19 | 
| pairing | VariableRangeCheckerAir | 2 | 1 | 4 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 26 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 32 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 80 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 15 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 29 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 13 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 213 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 13 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 22 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 29 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 68 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 15 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 8 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 9 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 273 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 112 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 244 | 
| pairing | VmConnectorAir | 2 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 0 | 262,144 |  | 12 | 25 | 9,699,328 | 
| pairing | AccessAdapterAir<32> | 0 | 131,072 |  | 12 | 41 | 6,946,816 | 
| pairing | AccessAdapterAir<4> | 0 | 64 |  | 12 | 13 | 1,600 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 12 | 17 | 15,204,352 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| pairing | KeccakVmAir | 0 | 1 |  | 12 | 3,163 | 3,175 | 
| pairing | MemoryMerkleAir<8> | 0 | 32,768 |  | 12 | 32 | 1,441,792 | 
| pairing | PersistentBoundaryAir<8> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32,768 |  | 12 | 300 | 10,223,616 | 
| pairing | ProgramAir | 0 | 32,768 |  | 12 | 10 | 720,896 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 12 | 1 | 6,815,744 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 12 | 32 | 11,264 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 12 | 36 | 50,331,648 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 12 | 37 | 3,211,264 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 12 | 53 | 133,120 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 12 | 26 | 4,980,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 12 | 18 | 122,880 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 12 | 166 | 5,696 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 65,536 |  | 12 | 28 | 2,621,440 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 12 | 41 | 55,574,528 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 12 | 39 | 13,056 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 12 | 31 | 22,016 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1 |  | 12 | 547 | 559 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 12 | 263 | 281,600 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 16,384 |  | 12 | 625 | 8,339,456 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_gen_layers_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 951 | 8,094 | 1,820,436 | 192,032,287 | 978 | 6,493 | 1,324 | 350 | 305 | 2,111 | 602 | 95,832,407 | 1,219 | 176 | 650 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 5,112,018 | 2,013,265,921 | 
| pairing | 0 | 1 | 17,620,378 | 2,013,265,921 | 
| pairing | 0 | 2 | 2,556,009 | 2,013,265,921 | 
| pairing | 0 | 3 | 24,468,838 | 2,013,265,921 | 
| pairing | 0 | 4 | 131,072 | 2,013,265,921 | 
| pairing | 0 | 5 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 6 | 6,004,051 | 2,013,265,921 | 
| pairing | 0 | 7 | 4,096 | 2,013,265,921 | 
| pairing | 0 | 8 | 56,945,038 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/9fb808a98d52bf47125f79f7d8a1e130ddddf6cb

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14382779015)
