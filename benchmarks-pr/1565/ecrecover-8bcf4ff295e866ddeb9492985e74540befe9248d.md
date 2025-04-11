| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+1 [+101.0%])</span> 2.88 | <span style='color: red'>(+1 [+101.0%])</span> 2.88 |
| ecrecover_program | <span style='color: red'>(+1 [+101.0%])</span> 2.88 | <span style='color: red'>(+1 [+101.0%])</span> 2.88 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+1445 [+101.0%])</span> 2,875 | <span style='color: red'>(+1445 [+101.0%])</span> 2,875 | <span style='color: red'>(+1445 [+101.0%])</span> 2,875 | <span style='color: red'>(+1445 [+101.0%])</span> 2,875 |
| `main_cells_used     ` |  14,470,186 |  14,470,186 |  14,470,186 |  14,470,186 |
| `total_cycles        ` |  289,447 |  289,447 |  289,447 |  289,447 |
| `execute_time_ms     ` | <span style='color: green'>(-1 [-0.7%])</span> 146 | <span style='color: green'>(-1 [-0.7%])</span> 146 | <span style='color: green'>(-1 [-0.7%])</span> 146 | <span style='color: green'>(-1 [-0.7%])</span> 146 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-7 [-3.8%])</span> 177 | <span style='color: green'>(-7 [-3.8%])</span> 177 | <span style='color: green'>(-7 [-3.8%])</span> 177 | <span style='color: green'>(-7 [-3.8%])</span> 177 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+1453 [+132.2%])</span> 2,552 | <span style='color: red'>(+1453 [+132.2%])</span> 2,552 | <span style='color: red'>(+1453 [+132.2%])</span> 2,552 | <span style='color: red'>(+1453 [+132.2%])</span> 2,552 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+16 [+9.6%])</span> 183 | <span style='color: red'>(+16 [+9.6%])</span> 183 | <span style='color: red'>(+16 [+9.6%])</span> 183 | <span style='color: red'>(+16 [+9.6%])</span> 183 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+18 [+52.9%])</span> 52 | <span style='color: red'>(+18 [+52.9%])</span> 52 | <span style='color: red'>(+18 [+52.9%])</span> 52 | <span style='color: red'>(+18 [+52.9%])</span> 52 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-45 [-23.9%])</span> 143 | <span style='color: green'>(-45 [-23.9%])</span> 143 | <span style='color: green'>(-45 [-23.9%])</span> 143 | <span style='color: green'>(-45 [-23.9%])</span> 143 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+280 [+222.2%])</span> 406 | <span style='color: red'>(+280 [+222.2%])</span> 406 | <span style='color: red'>(+280 [+222.2%])</span> 406 | <span style='color: red'>(+280 [+222.2%])</span> 406 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+33 [+21.6%])</span> 186 | <span style='color: red'>(+33 [+21.6%])</span> 186 | <span style='color: red'>(+33 [+21.6%])</span> 186 | <span style='color: red'>(+33 [+21.6%])</span> 186 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+565 [+136.5%])</span> 979 | <span style='color: red'>(+565 [+136.5%])</span> 979 | <span style='color: red'>(+565 [+136.5%])</span> 979 | <span style='color: red'>(+565 [+136.5%])</span> 979 |
| `sumcheck_prove_batch_ms` |  403 |  403 |  403 |  403 |
| `gkr_prove_batch_ms  ` |  503 |  503 |  503 |  503 |
| `gkr_gen_layers_ms   ` |  58 |  58 |  58 |  58 |
| `gkr_generate_aux    ` |  96 |  96 |  96 |  96 |
| `gkr_build_instances_ms` |  20 |  20 |  20 |  20 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| ecrecover_program | 1 | 843 | 8 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 10 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 10 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 10 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 10 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 10 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,251 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 37 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| ecrecover_program | PhantomAir | 2 | 3 | 4 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 2 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 2 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 19 | 
| ecrecover_program | VariableRangeCheckerAir | 2 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 26 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 32 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 80 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 29 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 13 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 213 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 13 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 22 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 29 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 68 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 15 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 8 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 9 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 273 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 112 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 244 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 16,384 |  | 12 | 25 | 606,208 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 8,192 |  | 12 | 41 | 434,176 | 
| ecrecover_program | AccessAdapterAir<4> | 0 | 64 |  | 12 | 13 | 1,600 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 32,768 |  | 12 | 17 | 950,272 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 12 | 2 | 917,504 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 12 | 3,163 | 406,400 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 12 | 32 | 180,224 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 12 | 300 | 1,277,952 | 
| ecrecover_program | ProgramAir | 0 | 16,384 |  | 12 | 10 | 360,448 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 12 | 1 | 6,815,744 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 12 | 32 | 11,264 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 12 | 1 | 3,407,872 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 12 | 36 | 6,291,456 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 12 | 37 | 200,704 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 12 | 53 | 1,064,960 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 12 | 26 | 622,592 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 32,768 |  | 12 | 32 | 1,441,792 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 12 | 18 | 245,760 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 12 | 166 | 729,088 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8,192 |  | 12 | 28 | 327,680 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 4,096 |  | 12 | 36 | 196,608 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 131,072 |  | 12 | 41 | 6,946,816 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 12 | 39 | 408 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 4,096 |  | 12 | 31 | 176,128 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 12 | 547 | 1,144,832 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 12 | 263 | 8,800 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 12 | 625 | 652,288 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | sumcheck_prove_batch_ms | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | gkr_prove_batch_ms | gkr_generate_aux | gkr_gen_layers_ms | gkr_build_instances_ms | generate_perm_trace_time_ms | execute_time_ms | build_gkr_input_layer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 177 | 2,875 | 289,447 | 35,692,877 | 403 | 2,552 | 406 | 186 | 143 | 979 | 183 | 14,470,186 | 503 | 96 | 58 | 20 | 52 | 146 | 19 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 736,214 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 2,273,180 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 368,107 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 3,796,961 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 882,858 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 16,448 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 9,036,328 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/8bcf4ff295e866ddeb9492985e74540befe9248d

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14406709434)
