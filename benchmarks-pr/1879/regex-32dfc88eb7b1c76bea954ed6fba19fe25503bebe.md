| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+34 [+479.9%])</span> 40.67 | <span style='color: green'>(-2 [-58.4%])</span> 1.58 |
| regex_program | <span style='color: red'>(+34 [+484.1%])</span> 40.62 | <span style='color: green'>(-2 [-59.2%])</span> 1.52 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-2246 [-64.6%])</span> 1,230.79 | <span style='color: red'>(+33662 [+484.1%])</span> 40,616 | <span style='color: green'>(-2212 [-59.2%])</span> 1,523 | <span style='color: green'>(-2115 [-65.7%])</span> 1,104 |
| `main_cells_used     ` | <span style='color: green'>(-77926923 [-93.6%])</span> 5,297,870.48 | <span style='color: red'>(+8380140 [+5.0%])</span> 174,829,726 | <span style='color: green'>(-84222772 [-90.1%])</span> 9,216,448 | <span style='color: green'>(-71976116 [-98.6%])</span> 1,034,250 |
| `total_cycles        ` | <span style='color: green'>(-1956491 [-93.9%])</span> 126,225.21 |  4,165,432 | <span style='color: green'>(-2112700 [-94.2%])</span> 131,000 | <span style='color: green'>(-1907300 [-99.2%])</span> 14,432 |
| `execute_metered_time_ms` | <span style='color: green'>(-4 [-6.7%])</span> 56 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+5 [+7.5%])</span> 74.23 | -          | <span style='color: red'>(+5 [+7.5%])</span> 74.23 | <span style='color: red'>(+5 [+7.5%])</span> 74.23 |
| `execute_e3_time_ms  ` | <span style='color: green'>(-156 [-94.4%])</span> 9.24 | <span style='color: green'>(-25 [-7.6%])</span> 305 | <span style='color: green'>(-168 [-93.9%])</span> 11 | <span style='color: green'>(-150 [-99.3%])</span> 1 |
| `execute_e3_insn_mi/s` | <span style='color: red'>(+0 [+2.5%])</span> 12.89 | -          | <span style='color: red'>(+0 [+3.5%])</span> 13.09 | <span style='color: green'>(-2 [-13.5%])</span> 10.82 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-235 [-32.7%])</span> 482.24 | <span style='color: red'>(+14480 [+1009.8%])</span> 15,914 | <span style='color: green'>(-131 [-17.4%])</span> 624 | <span style='color: green'>(-320 [-47.1%])</span> 359 |
| `memory_finalize_time_ms` | <span style='color: green'>(-94 [-55.6%])</span> 75 | <span style='color: red'>(+2137 [+632.2%])</span> 2,475 | <span style='color: green'>(-55 [-20.8%])</span> 209 | <span style='color: green'>(-8 [-10.8%])</span> 66 |
| `boundary_finalize_time_ms` | <span style='color: green'>(-2 [-93.9%])</span> 0.15 |  5 | <span style='color: green'>(-2 [-40.0%])</span> 3 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `merkle_finalize_time_ms` | <span style='color: green'>(-83 [-54.0%])</span> 71.03 | <span style='color: red'>(+2035 [+658.6%])</span> 2,344 | <span style='color: green'>(-49 [-20.4%])</span> 191 | <span style='color: green'>(-6 [-8.7%])</span> 63 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-1856 [-71.5%])</span> 739.30 | <span style='color: red'>(+19207 [+370.1%])</span> 24,397 | <span style='color: green'>(-1897 [-67.7%])</span> 904 | <span style='color: green'>(-1751 [-73.3%])</span> 638 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-413 [-81.3%])</span> 94.94 | <span style='color: red'>(+2117 [+208.4%])</span> 3,133 | <span style='color: green'>(-425 [-74.4%])</span> 146 | <span style='color: green'>(-371 [-83.4%])</span> 74 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-190 [-88.8%])</span> 24.06 | <span style='color: red'>(+365 [+85.1%])</span> 794 | <span style='color: green'>(-191 [-83.4%])</span> 38 | <span style='color: green'>(-185 [-92.5%])</span> 15 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-451 [-80.9%])</span> 106.12 | <span style='color: red'>(+2388 [+214.4%])</span> 3,502 | <span style='color: green'>(-445 [-75.6%])</span> 144 | <span style='color: green'>(-450 [-85.7%])</span> 75 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-214 [-80.0%])</span> 53.58 | <span style='color: red'>(+1232 [+229.9%])</span> 1,768 | <span style='color: green'>(-224 [-75.7%])</span> 72 | <span style='color: green'>(-199 [-82.9%])</span> 41 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-149 [-61.2%])</span> 94.67 | <span style='color: red'>(+2636 [+540.2%])</span> 3,124 | <span style='color: green'>(-153 [-54.6%])</span> 127 | <span style='color: green'>(-120 [-57.7%])</span> 88 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-433 [-54.6%])</span> 359.73 | <span style='color: red'>(+10286 [+649.0%])</span> 11,871 | <span style='color: green'>(-420 [-50.8%])</span> 407 | <span style='color: green'>(-425 [-56.1%])</span> 333 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 537 | 19 | 151,754 | 

| group | num_segments | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 33 | 24 | 4,165,433 | 1 | 4,834 | 56 | 74.23 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| regex_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 2 | 321 | 4,513 | 
| regex_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| regex_program | PhantomAir | 2 | 3 | 5 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| regex_program | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| regex_program | AccessAdapterAir<8> | 1 | 65,536 |  | 16 | 17 | 2,162,688 | 
| regex_program | AccessAdapterAir<8> | 10 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 11 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 12 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 13 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 14 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 15 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 16 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 17 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 18 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 19 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 2 | 32,768 |  | 16 | 17 | 1,081,344 | 
| regex_program | AccessAdapterAir<8> | 20 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 21 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 22 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 23 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 24 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 25 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 26 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 27 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 28 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 29 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 3 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 30 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 31 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 32 | 1,024 |  | 16 | 17 | 33,792 | 
| regex_program | AccessAdapterAir<8> | 4 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 5 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 6 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 7 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 8 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | AccessAdapterAir<8> | 9 | 512 |  | 16 | 17 | 16,896 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 13 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 14 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 15 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 16 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 17 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 18 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 19 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 20 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 21 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 22 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 23 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 24 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 25 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 26 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 27 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 28 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 29 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 30 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 31 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 32 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 32 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 32,768 |  | 16 | 32 | 1,572,864 | 
| regex_program | MemoryMerkleAir<8> | 1 | 65,536 |  | 16 | 32 | 3,145,728 | 
| regex_program | MemoryMerkleAir<8> | 10 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 11 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 12 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 13 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 14 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 15 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 16 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 17 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 18 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 19 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 2 | 65,536 |  | 16 | 32 | 3,145,728 | 
| regex_program | MemoryMerkleAir<8> | 20 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 21 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 22 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 23 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 24 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 25 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 26 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 27 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 28 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 29 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 3 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 30 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 31 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | MemoryMerkleAir<8> | 32 | 2,048 |  | 16 | 32 | 98,304 | 
| regex_program | MemoryMerkleAir<8> | 4 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 5 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 6 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 7 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 8 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | MemoryMerkleAir<8> | 9 | 1,024 |  | 16 | 32 | 49,152 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 65,536 |  | 12 | 20 | 2,097,152 | 
| regex_program | PersistentBoundaryAir<8> | 10 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 11 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 12 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 13 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 14 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 15 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 16 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 17 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 18 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 19 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 2 | 32,768 |  | 12 | 20 | 1,048,576 | 
| regex_program | PersistentBoundaryAir<8> | 20 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 21 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 22 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 23 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 24 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 25 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 26 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 27 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 28 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 29 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 3 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 30 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 31 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 32 | 1,024 |  | 12 | 20 | 32,768 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 5 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 6 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 7 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 8 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PersistentBoundaryAir<8> | 9 | 512 |  | 12 | 20 | 16,384 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 10 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 11 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 12 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 13 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 14 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 15 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 16 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 17 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 18 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 19 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 20 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 21 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 22 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 23 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 24 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 25 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 26 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 27 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 28 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 29 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 30 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 31 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 32 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 7 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 8 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 9 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 4,096 |  | 8 | 300 | 1,261,568 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 13 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 14 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 15 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 16 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 17 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 18 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 19 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 4,096 |  | 8 | 300 | 1,261,568 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 20 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 21 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 22 | 512 |  | 8 | 300 | 157,696 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 23 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 24 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 25 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 26 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 27 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 28 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 29 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 30 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 31 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 32 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 1,024 |  | 8 | 300 | 315,392 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 10 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 11 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 12 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 13 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 14 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 15 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 16 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 17 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 18 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 19 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 20 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 21 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 22 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 23 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 24 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 25 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 26 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 27 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 28 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 29 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 30 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 31 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 32 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 7 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 8 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 9 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 10 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 11 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 12 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 13 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 14 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 15 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 16 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 17 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 18 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 19 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 20 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 21 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 22 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 23 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 24 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 25 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 26 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 27 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 28 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 29 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 30 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 31 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 32 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 7 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 8 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 9 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 10 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 11 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 12 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 13 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 14 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 15 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 16 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 17 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 18 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 19 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 20 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 21 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 22 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 23 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 24 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 25 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 26 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 27 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 28 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 29 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 30 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 31 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 32 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 32,768 |  | 52 | 36 | 2,883,584 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 13 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 14 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 15 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 16 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 17 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 18 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 19 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 21 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 22 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 23 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 24 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 25 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 26 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 27 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 28 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 29 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 30 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 31 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 32 | 8,192 |  | 52 | 36 | 720,896 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 65,536 |  | 52 | 36 | 5,767,168 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1,024 |  | 40 | 37 | 78,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 256 |  | 40 | 37 | 19,712 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 13 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 14 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 15 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 16 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 17 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 19 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 1,024 |  | 40 | 37 | 78,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 20 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 21 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 22 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 23 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 24 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 25 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 26 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 27 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 28 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 29 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 30 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 31 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 32 | 128 |  | 40 | 37 | 9,856 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 2,048 |  | 40 | 37 | 157,696 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1,024 |  | 52 | 53 | 107,520 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 11 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 12 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 13 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 14 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 15 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 16 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 17 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 18 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 19 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 20 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 21 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 22 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 23 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 25 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 26 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 27 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 28 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 29 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 30 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 31 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 32 | 1,024 |  | 52 | 53 | 107,520 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 5 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 6 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 7 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 8 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 9 | 8,192 |  | 52 | 53 | 860,160 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 16,384 |  | 28 | 26 | 884,736 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 14 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 15 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 16 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 17 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 18 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 19 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 32,768 |  | 28 | 26 | 1,769,472 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 20 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 21 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 22 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 23 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 24 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 25 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 26 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 27 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 28 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 29 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 30 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 31 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 32 | 2,048 |  | 28 | 26 | 110,592 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 8,192 |  | 28 | 26 | 442,368 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 16,384 |  | 32 | 32 | 1,048,576 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 2,048 |  | 32 | 32 | 131,072 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 10 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 11 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 12 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 15 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 16 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 17 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 18 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 19 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 4,096 |  | 32 | 32 | 262,144 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 20 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 21 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 22 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 23 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 24 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 25 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 26 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 27 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 28 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 29 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 30 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 31 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 32 | 1,024 |  | 32 | 32 | 65,536 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 5 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 7 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 8 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 9 | 8,192 |  | 32 | 32 | 524,288 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 2,048 |  | 28 | 18 | 94,208 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 13 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 14 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 16 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 17 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 18 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 19 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 2,048 |  | 28 | 18 | 94,208 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 20 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 21 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 22 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 23 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 24 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 25 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 26 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 27 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 28 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 29 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 30 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 31 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 32 | 512 |  | 28 | 18 | 23,552 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 4,096 |  | 28 | 18 | 188,416 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 2,048 |  | 36 | 28 | 131,072 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 10 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 11 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 12 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 13 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 14 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 15 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 17 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 18 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 19 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 4,096 |  | 36 | 28 | 262,144 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 20 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 21 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 22 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 23 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 24 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 25 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 26 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 27 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 28 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 29 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 30 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 31 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 32 | 1,024 |  | 36 | 28 | 65,536 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 5 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 7 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 8 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 9 | 8,192 |  | 36 | 28 | 524,288 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 4 |  | 52 | 36 | 352 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 32 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 131,072 |  | 52 | 41 | 12,189,696 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 10 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 12 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 13 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 14 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 15 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 16 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 18 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 19 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 20 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 21 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 22 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 23 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 24 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 25 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 26 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 27 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 28 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 29 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 30 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 31 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 32 | 8,192 |  | 52 | 41 | 761,856 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 5 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 7 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 8 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 9 | 65,536 |  | 52 | 41 | 6,094,848 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 1 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 64 |  | 72 | 39 | 7,104 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 16 |  | 72 | 39 | 1,776 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 128 |  | 52 | 31 | 10,624 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 1,024 |  | 52 | 31 | 84,992 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 10 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 11 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 12 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 13 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 14 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 15 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 16 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 17 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 18 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 19 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 1,024 |  | 52 | 31 | 84,992 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 20 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 21 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 22 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 23 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 24 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 25 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 26 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 27 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 28 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 29 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 3 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 30 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 31 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 32 | 256 |  | 52 | 31 | 21,248 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 5 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 6 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 7 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 8 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 9 | 2,048 |  | 52 | 31 | 169,984 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 1,024 |  | 28 | 20 | 49,152 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 10 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 11 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 13 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 14 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 15 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 16 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 18 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 19 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 1,024 |  | 28 | 20 | 49,152 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 20 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 21 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 22 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 23 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 24 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 25 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 26 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 27 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 28 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 29 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 30 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 31 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 32 | 256 |  | 28 | 20 | 12,288 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 5 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 7 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 8 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 9 | 2,048 |  | 28 | 20 | 98,304 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 14 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 15 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 16 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 17 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 18 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 19 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 20 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 21 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 22 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 23 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 24 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 25 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 26 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 27 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 28 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 29 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 30 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 31 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 32 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 359 | 1,272 | 120,000 | 33,660,284 | 904 | 68 | 127 | 1,113 | 121 | 407 | 108 | 23 | 118 | 146 | 9,216,448 | 120,000 | 27 | 9 | 12.69 | 1 | 
| regex_program | 1 | 624 | 1,523 | 130,000 | 36,133,660 | 888 | 72 | 111 | 1,104 | 144 | 398 | 191 | 22 | 209 | 120 | 9,077,462 | 130,000 | 38 | 11 | 10.82 | 3 | 
| regex_program | 10 | 480 | 1,241 | 130,000 | 25,317,948 | 751 | 50 | 90 | 932 | 136 | 354 | 65 | 23 | 68 | 91 | 5,094,049 | 130,000 | 24 | 10 | 12.95 | 0 | 
| regex_program | 11 | 479 | 1,210 | 130,000 | 25,317,948 | 721 | 54 | 90 | 902 | 102 | 354 | 64 | 22 | 67 | 93 | 5,084,084 | 130,000 | 23 | 10 | 12.94 | 0 | 
| regex_program | 12 | 478 | 1,218 | 130,000 | 25,317,948 | 731 | 54 | 91 | 912 | 100 | 363 | 65 | 22 | 68 | 94 | 5,081,151 | 130,000 | 23 | 9 | 13.01 | 0 | 
| regex_program | 13 | 480 | 1,214 | 130,000 | 25,317,948 | 725 | 52 | 91 | 906 | 100 | 359 | 64 | 22 | 67 | 92 | 5,088,987 | 130,000 | 23 | 9 | 12.100 | 0 | 
| regex_program | 14 | 479 | 1,208 | 130,000 | 25,317,948 | 719 | 53 | 93 | 900 | 101 | 352 | 64 | 22 | 68 | 93 | 5,089,941 | 130,000 | 21 | 10 | 12.96 | 0 | 
| regex_program | 15 | 480 | 1,218 | 130,000 | 25,317,948 | 729 | 50 | 93 | 910 | 106 | 358 | 65 | 22 | 68 | 92 | 5,073,426 | 130,000 | 24 | 9 | 12.99 | 0 | 
| regex_program | 16 | 479 | 1,214 | 130,000 | 25,317,948 | 725 | 53 | 94 | 906 | 100 | 353 | 65 | 23 | 67 | 94 | 5,093,081 | 130,000 | 25 | 10 | 12.97 | 0 | 
| regex_program | 17 | 482 | 1,223 | 130,000 | 25,317,948 | 731 | 53 | 95 | 912 | 101 | 358 | 65 | 22 | 68 | 93 | 5,094,524 | 130,000 | 25 | 10 | 12.96 | 0 | 
| regex_program | 18 | 481 | 1,216 | 130,000 | 25,317,948 | 725 | 53 | 92 | 906 | 102 | 357 | 64 | 22 | 68 | 92 | 5,086,478 | 130,000 | 23 | 10 | 12.94 | 0 | 
| regex_program | 19 | 479 | 1,206 | 130,000 | 25,317,948 | 718 | 53 | 92 | 899 | 100 | 354 | 64 | 23 | 67 | 92 | 5,085,582 | 130,000 | 23 | 9 | 12.100 | 0 | 
| regex_program | 2 | 536 | 1,383 | 131,000 | 31,954,732 | 838 | 64 | 106 | 1,030 | 139 | 388 | 112 | 23 | 124 | 111 | 7,920,075 | 131,000 | 25 | 9 | 13.09 | 1 | 
| regex_program | 20 | 484 | 1,213 | 130,000 | 25,317,948 | 719 | 54 | 90 | 900 | 102 | 354 | 63 | 22 | 66 | 92 | 5,094,738 | 130,000 | 23 | 10 | 12.95 | 0 | 
| regex_program | 21 | 479 | 1,207 | 130,000 | 25,317,948 | 719 | 52 | 91 | 900 | 100 | 356 | 64 | 22 | 66 | 92 | 5,093,965 | 130,000 | 24 | 9 | 13.01 | 0 | 
| regex_program | 22 | 482 | 1,227 | 130,000 | 25,160,252 | 736 | 52 | 95 | 917 | 104 | 360 | 64 | 23 | 67 | 93 | 5,069,188 | 130,000 | 23 | 9 | 12.99 | 0 | 
| regex_program | 23 | 480 | 1,220 | 130,000 | 25,317,948 | 731 | 55 | 97 | 912 | 100 | 357 | 65 | 22 | 68 | 93 | 5,089,611 | 130,000 | 23 | 9 | 12.99 | 0 | 
| regex_program | 24 | 480 | 1,215 | 130,000 | 25,317,948 | 726 | 53 | 97 | 907 | 101 | 353 | 64 | 22 | 67 | 92 | 5,082,197 | 130,000 | 24 | 9 | 12.99 | 0 | 
| regex_program | 25 | 485 | 1,219 | 130,000 | 25,317,948 | 725 | 54 | 90 | 906 | 100 | 357 | 64 | 22 | 67 | 92 | 5,083,496 | 130,000 | 26 | 9 | 13.04 | 0 | 
| regex_program | 26 | 483 | 1,210 | 130,000 | 25,317,948 | 718 | 51 | 91 | 899 | 101 | 355 | 64 | 22 | 67 | 92 | 5,092,501 | 130,000 | 23 | 9 | 13 | 0 | 
| regex_program | 27 | 477 | 1,215 | 130,000 | 25,317,948 | 729 | 53 | 91 | 910 | 105 | 358 | 64 | 22 | 67 | 93 | 5,087,719 | 130,000 | 24 | 9 | 12.100 | 0 | 
| regex_program | 28 | 480 | 1,257 | 130,000 | 25,317,948 | 767 | 50 | 92 | 948 | 141 | 361 | 63 | 22 | 66 | 93 | 5,074,633 | 130,000 | 23 | 10 | 12.97 | 0 | 
| regex_program | 29 | 480 | 1,223 | 130,000 | 25,317,948 | 733 | 53 | 94 | 914 | 100 | 358 | 64 | 22 | 67 | 93 | 5,092,451 | 130,000 | 26 | 10 | 12.97 | 0 | 
| regex_program | 3 | 479 | 1,214 | 130,000 | 25,317,948 | 725 | 53 | 94 | 906 | 102 | 354 | 66 | 24 | 69 | 92 | 5,080,241 | 130,000 | 24 | 10 | 12.96 | 0 | 
| regex_program | 30 | 485 | 1,214 | 130,000 | 25,317,948 | 719 | 51 | 90 | 900 | 101 | 354 | 64 | 23 | 67 | 92 | 5,086,538 | 130,000 | 25 | 10 | 12.97 | 0 | 
| regex_program | 31 | 479 | 1,225 | 130,000 | 25,400,380 | 736 | 52 | 100 | 916 | 102 | 361 | 65 | 23 | 67 | 93 | 5,181,132 | 130,000 | 23 | 10 | 12.97 | 0 | 
| regex_program | 32 | 465 | 1,104 | 14,432 | 12,924,956 | 638 | 41 | 88 | 1,232 | 75 | 333 | 64 | 22 | 68 | 74 | 1,034,250 | 14,433 | 15 | 1 | 12.27 | 0 | 
| regex_program | 4 | 479 | 1,227 | 130,000 | 25,317,948 | 738 | 55 | 96 | 918 | 103 | 360 | 64 | 23 | 68 | 93 | 5,081,349 | 130,000 | 26 | 10 | 12.96 | 0 | 
| regex_program | 5 | 480 | 1,221 | 130,000 | 25,317,948 | 732 | 51 | 93 | 913 | 107 | 361 | 66 | 24 | 69 | 92 | 5,083,168 | 130,000 | 23 | 9 | 13.01 | 0 | 
| regex_program | 6 | 483 | 1,224 | 130,000 | 25,317,948 | 732 | 52 | 98 | 913 | 102 | 357 | 66 | 22 | 69 | 92 | 5,086,268 | 130,000 | 24 | 9 | 12.98 | 0 | 
| regex_program | 7 | 481 | 1,211 | 130,000 | 25,317,948 | 721 | 52 | 91 | 902 | 102 | 354 | 65 | 22 | 68 | 92 | 5,082,435 | 130,000 | 24 | 9 | 12.98 | 0 | 
| regex_program | 8 | 480 | 1,212 | 130,000 | 25,317,948 | 722 | 52 | 90 | 903 | 101 | 357 | 65 | 22 | 68 | 91 | 5,084,624 | 130,000 | 23 | 10 | 12.97 | 0 | 
| regex_program | 9 | 477 | 1,212 | 130,000 | 25,317,948 | 726 | 53 | 91 | 907 | 101 | 356 | 64 | 22 | 67 | 94 | 5,083,934 | 130,000 | 24 | 9 | 12.100 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 387,462 | 2,013,265,921 | 
| regex_program | 0 | 1 | 1,129,600 | 2,013,265,921 | 
| regex_program | 0 | 2 | 193,731 | 2,013,265,921 | 
| regex_program | 0 | 3 | 1,252,484 | 2,013,265,921 | 
| regex_program | 0 | 4 | 114,688 | 2,013,265,921 | 
| regex_program | 0 | 5 | 49,152 | 2,013,265,921 | 
| regex_program | 0 | 6 | 443,456 | 2,013,265,921 | 
| regex_program | 0 | 7 | 1,024 | 2,013,265,921 | 
| regex_program | 0 | 8 | 4,636,557 | 2,013,265,921 | 
| regex_program | 1 | 0 | 394,510 | 2,013,265,921 | 
| regex_program | 1 | 1 | 1,392,408 | 2,013,265,921 | 
| regex_program | 1 | 2 | 197,255 | 2,013,265,921 | 
| regex_program | 1 | 3 | 1,572,648 | 2,013,265,921 | 
| regex_program | 1 | 4 | 262,144 | 2,013,265,921 | 
| regex_program | 1 | 5 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 6 | 215,296 | 2,013,265,921 | 
| regex_program | 1 | 7 | 7,168 | 2,013,265,921 | 
| regex_program | 1 | 8 | 5,225,173 | 2,013,265,921 | 
| regex_program | 10 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 10 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 10 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 10 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 10 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 10 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 10 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 10 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 10 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 11 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 11 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 11 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 11 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 11 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 11 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 11 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 11 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 11 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 12 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 12 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 12 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 12 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 12 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 12 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 12 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 12 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 12 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 13 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 13 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 13 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 13 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 13 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 13 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 13 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 13 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 13 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 14 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 14 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 14 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 14 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 14 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 14 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 14 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 14 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 14 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 15 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 15 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 15 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 15 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 15 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 15 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 15 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 15 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 15 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 16 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 16 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 16 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 16 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 16 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 16 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 16 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 16 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 16 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 17 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 17 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 17 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 17 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 17 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 17 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 17 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 17 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 17 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 18 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 18 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 18 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 18 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 18 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 18 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 18 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 18 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 18 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 19 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 19 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 19 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 19 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 19 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 19 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 19 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 19 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 19 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 2 | 0 | 370,726 | 2,013,265,921 | 
| regex_program | 2 | 1 | 1,149,024 | 2,013,265,921 | 
| regex_program | 2 | 2 | 185,363 | 2,013,265,921 | 
| regex_program | 2 | 3 | 1,271,908 | 2,013,265,921 | 
| regex_program | 2 | 4 | 229,376 | 2,013,265,921 | 
| regex_program | 2 | 5 | 98,304 | 2,013,265,921 | 
| regex_program | 2 | 6 | 387,088 | 2,013,265,921 | 
| regex_program | 2 | 7 | 4,224 | 2,013,265,921 | 
| regex_program | 2 | 8 | 4,748,685 | 2,013,265,921 | 
| regex_program | 20 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 20 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 20 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 20 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 20 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 20 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 20 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 20 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 20 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 21 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 21 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 21 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 21 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 21 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 21 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 21 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 21 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 21 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 22 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 22 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 22 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 22 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 22 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 22 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 22 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 22 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 22 | 8 | 4,147,725 | 2,013,265,921 | 
| regex_program | 23 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 23 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 23 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 23 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 23 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 23 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 23 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 23 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 23 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 24 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 24 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 24 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 24 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 24 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 24 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 24 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 24 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 24 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 25 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 25 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 25 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 25 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 25 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 25 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 25 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 25 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 25 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 26 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 26 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 26 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 26 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 26 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 26 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 26 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 26 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 26 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 27 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 27 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 27 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 27 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 27 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 27 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 27 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 27 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 27 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 28 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 28 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 28 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 28 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 28 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 28 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 28 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 28 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 28 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 29 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 29 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 29 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 29 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 29 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 29 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 29 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 29 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 29 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 3 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 3 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 3 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 3 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 3 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 3 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 3 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 3 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 3 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 30 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 30 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 30 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 30 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 30 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 30 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 30 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 30 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 30 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 31 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 31 | 1 | 974,848 | 2,013,265,921 | 
| regex_program | 31 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 31 | 3 | 1,177,604 | 2,013,265,921 | 
| regex_program | 31 | 4 | 7,168 | 2,013,265,921 | 
| regex_program | 31 | 5 | 3,072 | 2,013,265,921 | 
| regex_program | 31 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 31 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 31 | 8 | 4,156,429 | 2,013,265,921 | 
| regex_program | 32 | 0 | 45,446 | 2,013,265,921 | 
| regex_program | 32 | 1 | 131,840 | 2,013,265,921 | 
| regex_program | 32 | 2 | 22,723 | 2,013,265,921 | 
| regex_program | 32 | 3 | 155,492 | 2,013,265,921 | 
| regex_program | 32 | 4 | 7,168 | 2,013,265,921 | 
| regex_program | 32 | 5 | 3,072 | 2,013,265,921 | 
| regex_program | 32 | 6 | 55,744 | 2,013,265,921 | 
| regex_program | 32 | 7 | 1,024 | 2,013,265,921 | 
| regex_program | 32 | 8 | 1,473,133 | 2,013,265,921 | 
| regex_program | 4 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 4 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 4 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 4 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 4 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 4 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 4 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 4 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 4 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 5 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 5 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 5 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 5 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 5 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 5 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 5 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 5 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 5 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 6 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 6 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 6 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 6 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 6 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 6 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 6 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 6 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 6 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 7 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 7 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 7 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 7 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 7 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 7 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 7 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 7 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 7 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 8 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 8 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 8 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 8 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 8 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 8 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 8 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 8 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 8 | 8 | 4,148,237 | 2,013,265,921 | 
| regex_program | 9 | 0 | 348,166 | 2,013,265,921 | 
| regex_program | 9 | 1 | 972,800 | 2,013,265,921 | 
| regex_program | 9 | 2 | 174,083 | 2,013,265,921 | 
| regex_program | 9 | 3 | 1,176,580 | 2,013,265,921 | 
| regex_program | 9 | 4 | 3,584 | 2,013,265,921 | 
| regex_program | 9 | 5 | 1,536 | 2,013,265,921 | 
| regex_program | 9 | 6 | 413,696 | 2,013,265,921 | 
| regex_program | 9 | 7 | 8,192 | 2,013,265,921 | 
| regex_program | 9 | 8 | 4,148,237 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/32dfc88eb7b1c76bea954ed6fba19fe25503bebe

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16333776877)
