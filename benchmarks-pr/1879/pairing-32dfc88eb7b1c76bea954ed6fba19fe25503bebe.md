| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+15 [+384.5%])</span> 19.38 | <span style='color: green'>(-2 [-62.0%])</span> 1.52 |
| pairing | <span style='color: red'>(+15 [+404.4%])</span> 19.18 | <span style='color: green'>(-2 [-65.1%])</span> 1.33 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-2524 [-66.4%])</span> 1,278.80 | <span style='color: red'>(+15379 [+404.4%])</span> 19,182 | <span style='color: green'>(-2477 [-65.1%])</span> 1,326 | <span style='color: green'>(-2746 [-72.2%])</span> 1,057 |
| `main_cells_used     ` | <span style='color: green'>(-95323657 [-93.0%])</span> 7,207,196.07 | <span style='color: red'>(+5577088 [+5.4%])</span> 108,107,941 | <span style='color: green'>(-94732897 [-92.4%])</span> 7,797,956 | <span style='color: green'>(-101332402 [-98.8%])</span> 1,198,451 |
| `total_cycles        ` | <span style='color: green'>(-1738766 [-93.3%])</span> 124,197.60 |  1,862,964 | <span style='color: green'>(-1730964 [-92.9%])</span> 132,000 | <span style='color: green'>(-1848000 [-99.2%])</span> 14,964 |
| `execute_metered_time_ms` | <span style='color: green'>(-3 [-1.5%])</span> 193 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+0 [+1.7%])</span> 9.62 | -          | <span style='color: red'>(+0 [+1.7%])</span> 9.62 | <span style='color: red'>(+0 [+1.7%])</span> 9.62 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-316 [-93.5%])</span> 22.13 | <span style='color: green'>(-6 [-1.8%])</span> 332 | <span style='color: green'>(-201 [-59.5%])</span> 137 | <span style='color: green'>(-337 [-99.7%])</span> 1 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+3 [+47.5%])</span> 8.13 | -          | <span style='color: red'>(+3 [+59.9%])</span> 8.81 | <span style='color: green'>(-5 [-82.5%])</span> 0.96 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-114 [-19.2%])</span> 480.73 | <span style='color: red'>(+6616 [+1111.9%])</span> 7,211 | <span style='color: green'>(-97 [-16.3%])</span> 498 | <span style='color: green'>(-269 [-45.2%])</span> 326 |
| `memory_finalize_time_ms` | <span style='color: green'>(-65 [-47.8%])</span> 71 | <span style='color: red'>(+929 [+683.1%])</span> 1,065 | <span style='color: green'>(-54 [-39.7%])</span> 82 | <span style='color: green'>(-70 [-51.5%])</span> 66 |
| `boundary_finalize_time_ms` | <span style='color: green'>(-2 [-100.0%])</span> 0 | <span style='color: green'>(-2 [-100.0%])</span> 0 | <span style='color: green'>(-2 [-100.0%])</span> 0 | <span style='color: green'>(-2 [-100.0%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-57 [-46.3%])</span> 66 | <span style='color: red'>(+867 [+704.9%])</span> 990 | <span style='color: green'>(-48 [-39.0%])</span> 75 | <span style='color: green'>(-60 [-48.8%])</span> 63 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-2094 [-73.0%])</span> 775.93 | <span style='color: red'>(+8769 [+305.5%])</span> 11,639 | <span style='color: green'>(-2026 [-70.6%])</span> 844 | <span style='color: green'>(-2287 [-79.7%])</span> 583 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-496 [-83.7%])</span> 96.40 | <span style='color: red'>(+854 [+144.3%])</span> 1,446 | <span style='color: green'>(-477 [-80.6%])</span> 115 | <span style='color: green'>(-527 [-89.0%])</span> 65 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-214 [-89.7%])</span> 24.67 | <span style='color: red'>(+131 [+54.8%])</span> 370 | <span style='color: green'>(-210 [-87.9%])</span> 29 | <span style='color: green'>(-225 [-94.1%])</span> 14 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-448 [-80.3%])</span> 110.07 | <span style='color: red'>(+1093 [+195.9%])</span> 1,651 | <span style='color: green'>(-416 [-74.6%])</span> 142 | <span style='color: green'>(-489 [-87.6%])</span> 69 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-268 [-81.5%])</span> 60.93 | <span style='color: red'>(+585 [+177.8%])</span> 914 | <span style='color: green'>(-262 [-79.6%])</span> 67 | <span style='color: green'>(-290 [-88.1%])</span> 39 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-182 [-65.2%])</span> 97.13 | <span style='color: red'>(+1178 [+422.2%])</span> 1,457 | <span style='color: green'>(-159 [-57.0%])</span> 120 | <span style='color: green'>(-205 [-73.5%])</span> 74 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-487 [-56.4%])</span> 376.53 | <span style='color: red'>(+4784 [+553.7%])</span> 5,648 | <span style='color: green'>(-472 [-54.6%])</span> 392 | <span style='color: green'>(-550 [-63.7%])</span> 314 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 1,079 | 10 | 73,788 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 15 | 24 | 1,862,965 | 1 | 4,529 | 193 | 9.62 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<2> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<32> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<4> | 2 | 5 | 12 | 
| pairing | AccessAdapterAir<8> | 2 | 5 | 12 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| pairing | KeccakVmAir | 2 | 321 | 4,513 | 
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
| pairing | AccessAdapterAir<16> | 0 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 1 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 10 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 11 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 12 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 13 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 14 | 2,048 |  | 16 | 25 | 83,968 | 
| pairing | AccessAdapterAir<16> | 2 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 3 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 4 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 5 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 6 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 7 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 8 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<16> | 9 | 16,384 |  | 16 | 25 | 671,744 | 
| pairing | AccessAdapterAir<32> | 0 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 1 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 10 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 11 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 12 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 13 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 14 | 1,024 |  | 16 | 41 | 58,368 | 
| pairing | AccessAdapterAir<32> | 2 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 3 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 4 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 5 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 6 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 7 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 8 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<32> | 9 | 8,192 |  | 16 | 41 | 466,944 | 
| pairing | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 1 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 10 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 11 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 12 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 13 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 14 | 8,192 |  | 16 | 17 | 270,336 | 
| pairing | AccessAdapterAir<8> | 2 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 3 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 4 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 5 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 6 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 7 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 8 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | AccessAdapterAir<8> | 9 | 32,768 |  | 16 | 17 | 1,081,344 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 13 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 14 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 8,192 |  | 16 | 32 | 393,216 | 
| pairing | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 10 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 11 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 12 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 13 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 14 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 2 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 3 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 4 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 5 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 6 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 7 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 8 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | MemoryMerkleAir<8> | 9 | 4,096 |  | 16 | 32 | 196,608 | 
| pairing | PersistentBoundaryAir<8> | 0 | 8,192 |  | 12 | 20 | 262,144 | 
| pairing | PersistentBoundaryAir<8> | 1 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 10 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 11 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 12 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 13 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 14 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 2 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 3 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 4 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 5 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 6 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 7 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 8 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PersistentBoundaryAir<8> | 9 | 4,096 |  | 12 | 20 | 131,072 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 10 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 11 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 12 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 13 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 14 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 7 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 8 | 1 |  | 12 | 6 | 18 | 
| pairing | PhantomAir | 9 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 13 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 14 | 1,024 |  | 8 | 300 | 315,392 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 4,096 |  | 8 | 300 | 1,261,568 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 10 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 11 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 12 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 13 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 14 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 2 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 3 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 4 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 5 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 6 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 7 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 8 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 9 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 10 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 11 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 12 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 13 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 14 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 7 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 8 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 9 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 13 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 14 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 13 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 14 | 8,192 |  | 52 | 36 | 720,896 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 65,536 |  | 52 | 36 | 5,767,168 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 13 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 14 | 512 |  | 40 | 37 | 39,424 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 4,096 |  | 40 | 37 | 315,392 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1,024 |  | 52 | 53 | 107,520 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 11 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 12 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 13 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 5 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 6 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 7 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 8 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 9 | 64 |  | 52 | 53 | 6,720 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 14 | 2,048 |  | 28 | 26 | 110,592 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 16,384 |  | 28 | 26 | 884,736 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 10 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 11 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 12 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | 1,024 |  | 32 | 32 | 65,536 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 5 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 7 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 8 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 9 | 16,384 |  | 32 | 32 | 1,048,576 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 13 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 14 | 64 |  | 28 | 18 | 2,944 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 512 |  | 28 | 18 | 23,552 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 14 | 16 |  | 56 | 166 | 3,552 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 10 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 11 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 12 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 13 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 14 | 512 |  | 36 | 28 | 32,768 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 5 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 7 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 8 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 9 | 4,096 |  | 36 | 28 | 262,144 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 10 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 12 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 13 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 14 | 8,192 |  | 52 | 41 | 761,856 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 5 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 7 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 8 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 9 | 65,536 |  | 52 | 41 | 6,094,848 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 10 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 11 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 12 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 13 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 3 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 5 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 6 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 7 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 8 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 9 | 16 |  | 72 | 39 | 1,776 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 10 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 11 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 12 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 13 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 64 |  | 52 | 31 | 5,312 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 5 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 6 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 7 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 8 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 9 | 32 |  | 52 | 31 | 2,656 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 10 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 11 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 13 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 14 | 256 |  | 28 | 20 | 12,288 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 5 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 7 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 8 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 9 | 2,048 |  | 28 | 20 | 98,304 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 64 |  | 320 | 263 | 25,024 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 10 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 11 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 12 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 13 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 3 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 4 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 5 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 6 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 7 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 8 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 9 | 64 |  | 320 | 263 | 37,312 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 604 | 497 | 563,712 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 10 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 11 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 12 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 13 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 14 | 128 |  | 604 | 497 | 140,928 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 3 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 5 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 6 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 7 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 8 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 9 | 1,024 |  | 604 | 497 | 1,127,424 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 14 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 326 | 1,307 | 132,000 | 28,085,156 | 844 | 67 | 120 | 1,344 | 112 | 392 | 75 | 24 | 82 | 115 | 7,660,237 | 132,000 | 26 | 137 | 0.96 | 0 | 
| pairing | 1 | 487 | 1,311 | 132,000 | 28,170,124 | 809 | 61 | 96 | 1,242 | 142 | 376 | 64 | 23 | 70 | 98 | 7,636,449 | 132,000 | 27 | 15 | 8.66 | 0 | 
| pairing | 10 | 493 | 1,326 | 132,000 | 28,170,124 | 818 | 60 | 100 | 1,251 | 139 | 375 | 67 | 23 | 71 | 97 | 7,690,042 | 132,000 | 29 | 15 | 8.57 | 0 | 
| pairing | 11 | 492 | 1,275 | 132,000 | 28,170,124 | 768 | 61 | 98 | 1,201 | 103 | 375 | 65 | 23 | 70 | 96 | 7,596,225 | 132,000 | 25 | 15 | 8.75 | 0 | 
| pairing | 12 | 490 | 1,277 | 132,000 | 28,170,124 | 772 | 64 | 96 | 1,205 | 104 | 378 | 66 | 23 | 71 | 97 | 7,663,570 | 132,000 | 24 | 15 | 8.57 | 0 | 
| pairing | 13 | 498 | 1,308 | 132,000 | 28,178,408 | 795 | 63 | 96 | 1,266 | 108 | 392 | 68 | 23 | 72 | 98 | 7,797,956 | 132,000 | 29 | 15 | 8.68 | 0 | 
| pairing | 14 | 473 | 1,057 | 14,964 | 11,151,708 | 583 | 39 | 74 | 951 | 69 | 314 | 63 | 23 | 66 | 65 | 1,198,451 | 14,965 | 14 | 1 | 8.61 | 0 | 
| pairing | 2 | 491 | 1,287 | 132,000 | 28,170,124 | 781 | 65 | 98 | 1,214 | 109 | 378 | 65 | 23 | 71 | 97 | 7,551,011 | 132,000 | 25 | 15 | 8.65 | 0 | 
| pairing | 3 | 491 | 1,289 | 132,000 | 28,170,124 | 783 | 62 | 98 | 1,216 | 104 | 383 | 65 | 23 | 71 | 98 | 7,696,391 | 132,000 | 26 | 15 | 8.61 | 0 | 
| pairing | 4 | 497 | 1,288 | 132,000 | 28,172,780 | 776 | 62 | 95 | 1,209 | 105 | 383 | 66 | 23 | 71 | 96 | 7,679,098 | 132,000 | 23 | 15 | 8.39 | 0 | 
| pairing | 5 | 492 | 1,277 | 132,000 | 28,170,124 | 771 | 61 | 96 | 1,204 | 105 | 378 | 65 | 23 | 69 | 98 | 7,538,789 | 132,000 | 24 | 14 | 8.81 | 0 | 
| pairing | 6 | 497 | 1,285 | 132,000 | 28,170,124 | 773 | 62 | 96 | 1,206 | 104 | 380 | 65 | 23 | 70 | 99 | 7,587,057 | 132,000 | 25 | 15 | 8.69 | 0 | 
| pairing | 7 | 493 | 1,280 | 132,000 | 28,170,124 | 772 | 65 | 98 | 1,205 | 103 | 374 | 65 | 23 | 70 | 97 | 7,611,587 | 132,000 | 25 | 15 | 8.62 | 0 | 
| pairing | 8 | 498 | 1,293 | 132,000 | 28,170,124 | 780 | 62 | 97 | 1,213 | 104 | 386 | 65 | 24 | 70 | 98 | 7,570,329 | 132,000 | 23 | 15 | 8.69 | 0 | 
| pairing | 9 | 493 | 1,322 | 132,000 | 28,170,124 | 814 | 60 | 99 | 1,247 | 140 | 384 | 66 | 24 | 71 | 97 | 7,630,749 | 132,000 | 25 | 15 | 8.64 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 355,110 | 2,013,265,921 | 
| pairing | 0 | 1 | 1,172,416 | 2,013,265,921 | 
| pairing | 0 | 2 | 177,555 | 2,013,265,921 | 
| pairing | 0 | 3 | 1,466,404 | 2,013,265,921 | 
| pairing | 0 | 4 | 32,768 | 2,013,265,921 | 
| pairing | 0 | 5 | 16,384 | 2,013,265,921 | 
| pairing | 0 | 6 | 397,272 | 2,013,265,921 | 
| pairing | 0 | 7 | 192 | 2,013,265,921 | 
| pairing | 0 | 8 | 4,572,469 | 2,013,265,921 | 
| pairing | 1 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 1 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 1 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 1 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 1 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 1 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 1 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 1 | 7 | 256 | 2,013,265,921 | 
| pairing | 1 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 10 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 10 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 10 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 10 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 10 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 10 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 10 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 10 | 7 | 256 | 2,013,265,921 | 
| pairing | 10 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 11 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 11 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 11 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 11 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 11 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 11 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 11 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 11 | 7 | 256 | 2,013,265,921 | 
| pairing | 11 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 12 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 12 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 12 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 12 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 12 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 12 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 12 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 12 | 7 | 256 | 2,013,265,921 | 
| pairing | 12 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 13 | 0 | 352,750 | 2,013,265,921 | 
| pairing | 13 | 1 | 1,169,232 | 2,013,265,921 | 
| pairing | 13 | 2 | 176,375 | 2,013,265,921 | 
| pairing | 13 | 3 | 1,595,160 | 2,013,265,921 | 
| pairing | 13 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 13 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 13 | 6 | 392,344 | 2,013,265,921 | 
| pairing | 13 | 7 | 256 | 2,013,265,921 | 
| pairing | 13 | 8 | 4,665,061 | 2,013,265,921 | 
| pairing | 14 | 0 | 42,022 | 2,013,265,921 | 
| pairing | 14 | 1 | 155,808 | 2,013,265,921 | 
| pairing | 14 | 2 | 21,011 | 2,013,265,921 | 
| pairing | 14 | 3 | 202,276 | 2,013,265,921 | 
| pairing | 14 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 14 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 14 | 6 | 46,944 | 2,013,265,921 | 
| pairing | 14 | 7 |  | 2,013,265,921 | 
| pairing | 14 | 8 | 1,431,645 | 2,013,265,921 | 
| pairing | 2 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 2 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 2 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 2 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 2 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 2 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 2 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 2 | 7 | 256 | 2,013,265,921 | 
| pairing | 2 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 3 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 3 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 3 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 3 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 3 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 3 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 3 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 3 | 7 | 256 | 2,013,265,921 | 
| pairing | 3 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 4 | 0 | 352,678 | 2,013,265,921 | 
| pairing | 4 | 1 | 1,168,992 | 2,013,265,921 | 
| pairing | 4 | 2 | 176,339 | 2,013,265,921 | 
| pairing | 4 | 3 | 1,594,340 | 2,013,265,921 | 
| pairing | 4 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 4 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 4 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 4 | 7 | 384 | 2,013,265,921 | 
| pairing | 4 | 8 | 4,663,757 | 2,013,265,921 | 
| pairing | 5 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 5 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 5 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 5 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 5 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 5 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 5 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 5 | 7 | 256 | 2,013,265,921 | 
| pairing | 5 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 6 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 6 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 6 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 6 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 6 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 6 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 6 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 6 | 7 | 256 | 2,013,265,921 | 
| pairing | 6 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 7 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 7 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 7 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 7 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 7 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 7 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 7 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 7 | 7 | 256 | 2,013,265,921 | 
| pairing | 7 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 8 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 8 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 8 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 8 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 8 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 8 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 8 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 8 | 7 | 256 | 2,013,265,921 | 
| pairing | 8 | 8 | 4,663,149 | 2,013,265,921 | 
| pairing | 9 | 0 | 352,614 | 2,013,265,921 | 
| pairing | 9 | 1 | 1,168,800 | 2,013,265,921 | 
| pairing | 9 | 2 | 176,307 | 2,013,265,921 | 
| pairing | 9 | 3 | 1,594,148 | 2,013,265,921 | 
| pairing | 9 | 4 | 16,384 | 2,013,265,921 | 
| pairing | 9 | 5 | 8,192 | 2,013,265,921 | 
| pairing | 9 | 6 | 392,080 | 2,013,265,921 | 
| pairing | 9 | 7 | 256 | 2,013,265,921 | 
| pairing | 9 | 8 | 4,663,149 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/32dfc88eb7b1c76bea954ed6fba19fe25503bebe

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16333776877)
