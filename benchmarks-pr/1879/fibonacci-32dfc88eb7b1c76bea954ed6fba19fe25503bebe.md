| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+13 [+539.5%])</span> 14.94 | <span style='color: green'>(-1 [-48.8%])</span> 1.20 |
| fibonacci_program | <span style='color: red'>(+13 [+542.5%])</span> 14.93 | <span style='color: green'>(-1 [-49.1%])</span> 1.18 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-1175 [-50.6%])</span> 1,148.54 | <span style='color: red'>(+12607 [+542.5%])</span> 14,931 | <span style='color: green'>(-1141 [-49.1%])</span> 1,183 | <span style='color: green'>(-1284 [-55.2%])</span> 1,040 |
| `main_cells_used     ` | <span style='color: green'>(-46669354 [-92.3%])</span> 3,919,877.46 | <span style='color: red'>(+369176 [+0.7%])</span> 50,958,407 | <span style='color: green'>(-46621970 [-92.2%])</span> 3,967,261 | <span style='color: green'>(-46853969 [-92.6%])</span> 3,735,262 |
| `total_cycles        ` | <span style='color: green'>(-1384871 [-92.3%])</span> 115,405.92 |  1,500,277 | <span style='color: green'>(-1384277 [-92.3%])</span> 116,000 | <span style='color: green'>(-1392000 [-92.8%])</span> 108,277 |
| `execute_metered_time_ms` | <span style='color: red'>(+1 [+7.7%])</span> 14 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: green'>(-5 [-4.5%])</span> 104.22 | -          | <span style='color: green'>(-5 [-4.5%])</span> 104.22 | <span style='color: green'>(-5 [-4.5%])</span> 104.22 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-88 [-92.6%])</span> 7 | <span style='color: green'>(-4 [-4.2%])</span> 91 | <span style='color: green'>(-88 [-92.6%])</span> 7 | <span style='color: green'>(-88 [-92.6%])</span> 7 |
| `execute_e3_insn_mi/s` | <span style='color: green'>(-0 [-3.1%])</span> 15.23 | -          | <span style='color: green'>(-0 [-2.1%])</span> 15.38 | <span style='color: green'>(-1 [-5.6%])</span> 14.83 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+33 [+7.4%])</span> 482.38 | <span style='color: red'>(+5822 [+1296.7%])</span> 6,271 | <span style='color: red'>(+62 [+13.8%])</span> 511 | <span style='color: green'>(-139 [-31.0%])</span> 310 |
| `memory_finalize_time_ms` | <span style='color: green'>(-7 [-10.1%])</span> 58.46 | <span style='color: red'>(+695 [+1069.2%])</span> 760 |  65 | <span style='color: green'>(-8 [-12.3%])</span> 57 |
| `boundary_finalize_time_ms` | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-5 [-8.6%])</span> 56.69 | <span style='color: red'>(+675 [+1088.7%])</span> 737 | <span style='color: red'>(+1 [+1.6%])</span> 63 | <span style='color: green'>(-7 [-11.3%])</span> 55 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1121 [-63.0%])</span> 659.15 | <span style='color: red'>(+6789 [+381.4%])</span> 8,569 | <span style='color: green'>(-1057 [-59.4%])</span> 723 | <span style='color: green'>(-1181 [-66.3%])</span> 599 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-233 [-72.9%])</span> 86.69 | <span style='color: red'>(+807 [+252.2%])</span> 1,127 | <span style='color: green'>(-209 [-65.3%])</span> 111 | <span style='color: green'>(-246 [-76.9%])</span> 74 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-116 [-82.2%])</span> 25.08 | <span style='color: red'>(+185 [+131.2%])</span> 326 | <span style='color: green'>(-110 [-78.0%])</span> 31 | <span style='color: green'>(-121 [-85.8%])</span> 20 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-248 [-72.2%])</span> 95.54 | <span style='color: red'>(+898 [+261.0%])</span> 1,242 | <span style='color: green'>(-226 [-65.7%])</span> 118 | <span style='color: green'>(-264 [-76.7%])</span> 80 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-122 [-73.1%])</span> 44.85 | <span style='color: red'>(+416 [+249.1%])</span> 583 | <span style='color: green'>(-119 [-71.3%])</span> 48 | <span style='color: green'>(-127 [-76.0%])</span> 40 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-102 [-55.5%])</span> 81.92 | <span style='color: red'>(+881 [+478.8%])</span> 1,065 | <span style='color: green'>(-78 [-42.4%])</span> 106 | <span style='color: green'>(-109 [-59.2%])</span> 75 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-296 [-48.1%])</span> 319.46 | <span style='color: red'>(+3538 [+575.3%])</span> 4,153 | <span style='color: green'>(-283 [-46.0%])</span> 332 | <span style='color: green'>(-310 [-50.4%])</span> 305 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 246 | 5 | 58,654 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 13 | 24 | 1,500,278 | 1 | 4,450 | 14 | 104.22 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| fibonacci_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 10 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 11 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 12 | 64 |  | 16 | 17 | 2,112 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 6 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 7 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 8 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | AccessAdapterAir<8> | 9 | 16 |  | 16 | 17 | 528 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | MemoryMerkleAir<8> | 1 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 10 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 11 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 12 | 256 |  | 16 | 32 | 12,288 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 3 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 4 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 5 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 6 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 7 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 8 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | MemoryMerkleAir<8> | 9 | 128 |  | 16 | 32 | 6,144 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 10 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 11 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 12 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 6 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 7 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 8 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PersistentBoundaryAir<8> | 9 | 16 |  | 12 | 20 | 512 | 
| fibonacci_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 10 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 11 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 12 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 7 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 8 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | PhantomAir | 9 | 1 |  | 12 | 6 | 18 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 512 |  | 8 | 300 | 157,696 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 128 |  | 8 | 300 | 39,424 | 
| fibonacci_program | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 10 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 11 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 12 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 2 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 3 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 4 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 5 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 6 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 7 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 8 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | ProgramAir | 9 | 8,192 |  | 8 | 10 | 147,456 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 10 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 11 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 12 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 7 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 8 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 9 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 65,536 |  | 52 | 36 | 5,767,168 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 131,072 |  | 52 | 36 | 11,534,336 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 32,768 |  | 40 | 37 | 2,523,136 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 16,384 |  | 28 | 26 | 884,736 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 12 | 2 |  | 32 | 32 | 128 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 8,192 |  | 28 | 18 | 376,832 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 12 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 12 | 64 |  | 52 | 41 | 5,952 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 8 |  | 28 | 20 | 384 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| fibonacci_program | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 310 | 1,040 | 116,000 | 23,300,300 | 723 | 44 | 106 | 894 | 99 | 332 | 63 | 25 | 65 | 111 | 3,967,261 | 116,000 | 26 | 7 | 15.06 | 0 | 
| fibonacci_program | 1 | 493 | 1,166 | 116,000 | 23,246,412 | 666 | 48 | 83 | 799 | 95 | 318 | 57 | 24 | 57 | 94 | 3,932,356 | 116,000 | 23 | 7 | 15.34 | 0 | 
| fibonacci_program | 10 | 507 | 1,166 | 116,000 | 23,246,412 | 652 | 42 | 78 | 786 | 93 | 321 | 56 | 24 | 58 | 85 | 3,932,356 | 116,000 | 23 | 7 | 15.20 | 0 | 
| fibonacci_program | 11 | 511 | 1,182 | 116,000 | 23,246,412 | 664 | 48 | 86 | 797 | 94 | 320 | 58 | 26 | 59 | 84 | 3,932,348 | 116,000 | 27 | 7 | 15.38 | 0 | 
| fibonacci_program | 12 | 498 | 1,104 | 108,277 | 17,614,268 | 599 | 40 | 75 | 761 | 80 | 305 | 58 | 23 | 60 | 74 | 3,735,262 | 108,278 | 20 | 7 | 15.01 | 0 | 
| fibonacci_program | 2 | 501 | 1,159 | 116,000 | 23,246,412 | 651 | 44 | 80 | 785 | 95 | 320 | 56 | 24 | 58 | 85 | 3,932,348 | 116,000 | 24 | 7 | 15.30 | 0 | 
| fibonacci_program | 3 | 483 | 1,154 | 116,000 | 23,246,412 | 664 | 48 | 84 | 797 | 95 | 320 | 56 | 23 | 58 | 86 | 3,932,356 | 116,000 | 24 | 7 | 14.83 | 0 | 
| fibonacci_program | 4 | 489 | 1,153 | 116,000 | 23,246,412 | 657 | 48 | 78 | 791 | 94 | 318 | 55 | 23 | 57 | 84 | 3,932,356 | 116,000 | 30 | 7 | 15.38 | 0 | 
| fibonacci_program | 5 | 494 | 1,155 | 116,000 | 23,246,412 | 654 | 44 | 78 | 788 | 95 | 322 | 56 | 24 | 57 | 85 | 3,932,348 | 116,000 | 25 | 7 | 15.12 | 0 | 
| fibonacci_program | 6 | 496 | 1,161 | 116,000 | 23,246,412 | 658 | 45 | 81 | 792 | 96 | 322 | 56 | 23 | 58 | 85 | 3,932,356 | 116,000 | 25 | 7 | 15.33 | 0 | 
| fibonacci_program | 7 | 500 | 1,158 | 116,000 | 23,246,412 | 651 | 42 | 78 | 784 | 94 | 321 | 55 | 23 | 58 | 85 | 3,932,356 | 116,000 | 24 | 7 | 15.30 | 0 | 
| fibonacci_program | 8 | 496 | 1,150 | 116,000 | 23,246,412 | 647 | 42 | 80 | 780 | 94 | 317 | 55 | 23 | 57 | 85 | 3,932,348 | 116,000 | 24 | 7 | 15.35 | 0 | 
| fibonacci_program | 9 | 493 | 1,183 | 116,000 | 23,246,412 | 683 | 48 | 78 | 816 | 118 | 317 | 56 | 24 | 58 | 84 | 3,932,356 | 116,000 | 31 | 7 | 15.36 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 0 | 376,974 | 2,013,265,921 | 
| fibonacci_program | 0 | 1 | 1,065,544 | 2,013,265,921 | 
| fibonacci_program | 0 | 2 | 188,487 | 2,013,265,921 | 
| fibonacci_program | 0 | 3 | 1,065,548 | 2,013,265,921 | 
| fibonacci_program | 0 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 0 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 0 | 6 | 778,324 | 2,013,265,921 | 
| fibonacci_program | 0 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 0 | 8 | 4,401,981 | 2,013,265,921 | 
| fibonacci_program | 1 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 1 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 1 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 1 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 1 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 1 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 1 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 1 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 1 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 10 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 10 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 10 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 10 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 10 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 10 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 10 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 10 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 10 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 11 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 11 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 11 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 11 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 11 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 11 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 11 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 11 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 11 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 12 | 0 | 245,946 | 2,013,265,921 | 
| fibonacci_program | 12 | 1 | 672,472 | 2,013,265,921 | 
| fibonacci_program | 12 | 2 | 122,973 | 2,013,265,921 | 
| fibonacci_program | 12 | 3 | 672,540 | 2,013,265,921 | 
| fibonacci_program | 12 | 4 | 832 | 2,013,265,921 | 
| fibonacci_program | 12 | 5 | 320 | 2,013,265,921 | 
| fibonacci_program | 12 | 6 | 450,620 | 2,013,265,921 | 
| fibonacci_program | 12 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 12 | 8 | 3,091,911 | 2,013,265,921 | 
| fibonacci_program | 2 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 2 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 2 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 2 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 2 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 2 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 2 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 2 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 2 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 3 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 3 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 3 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 3 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 3 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 3 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 3 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 3 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 3 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 4 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 4 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 4 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 4 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 4 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 4 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 4 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 4 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 4 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 5 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 5 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 5 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 5 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 5 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 5 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 5 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 5 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 5 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 6 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 6 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 6 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 6 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 6 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 6 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 6 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 6 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 6 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 7 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 7 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 7 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 7 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 7 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 7 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 7 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 7 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 7 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 8 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 8 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 8 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 8 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 8 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 8 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 8 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 8 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 8 | 8 | 4,399,885 | 2,013,265,921 | 
| fibonacci_program | 9 | 0 | 376,838 | 2,013,265,921 | 
| fibonacci_program | 9 | 1 | 1,065,024 | 2,013,265,921 | 
| fibonacci_program | 9 | 2 | 188,419 | 2,013,265,921 | 
| fibonacci_program | 9 | 3 | 1,064,996 | 2,013,265,921 | 
| fibonacci_program | 9 | 4 | 400 | 2,013,265,921 | 
| fibonacci_program | 9 | 5 | 144 | 2,013,265,921 | 
| fibonacci_program | 9 | 6 | 778,240 | 2,013,265,921 | 
| fibonacci_program | 9 | 7 |  | 2,013,265,921 | 
| fibonacci_program | 9 | 8 | 4,399,885 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/32dfc88eb7b1c76bea954ed6fba19fe25503bebe

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16333776877)
