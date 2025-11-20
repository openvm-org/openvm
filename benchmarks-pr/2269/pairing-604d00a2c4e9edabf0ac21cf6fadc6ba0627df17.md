| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+25 [+1643.9%])</span> 26.75 | <span style='color: red'>(+0 [+43.5%])</span> 1.31 |
| pairing | <span style='color: red'>(+25 [+1720.3%])</span> 26.29 | <span style='color: red'>(+0 [+2.7%])</span> 0.85 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: green'>(-12 [-1.6%])</span> 710.41 | <span style='color: red'>(+24841 [+1720.3%])</span> 26,285 | <span style='color: red'>(+22 [+2.7%])</span> 846 | <span style='color: red'>(+62 [+10.0%])</span> 682 |
| `main_cells_used     ` | <span style='color: green'>(-37181 [-0.3%])</span> 12,696,924.38 | <span style='color: red'>(+444317992 [+1744.6%])</span> 469,786,202 | <span style='color: red'>(+476394 [+3.0%])</span> 16,536,310 | <span style='color: red'>(+2590956 [+27.5%])</span> 11,999,250 |
| `total_cells_used    ` | <span style='color: red'>(+1625297 [+5.9%])</span> 29,126,996.22 | <span style='color: red'>(+1022695462 [+1859.3%])</span> 1,077,698,860 | <span style='color: red'>(+503674 [+1.5%])</span> 33,192,784 | <span style='color: red'>(+5842124 [+26.2%])</span> 28,156,412 |
| `execute_metered_time_ms` | <span style='color: red'>(+376 [+417.8%])</span> 466 | -          | -          | -          |
| `execute_metered_insn_mi/s` | <span style='color: red'>(+63 [+330.1%])</span> 82.61 | -          | <span style='color: red'>(+63 [+330.1%])</span> 82.61 | <span style='color: red'>(+63 [+330.1%])</span> 82.61 |
| `execute_preflight_insns` | <span style='color: red'>(+168946 [+19.4%])</span> 1,041,816.62 | <span style='color: red'>(+36801473 [+2108.1%])</span> 38,547,215 |  1,157,000 | <span style='color: red'>(+408473 [+69.4%])</span> 997,215 |
| `execute_preflight_time_ms` | <span style='color: green'>(-4 [-4.6%])</span> 90.19 | <span style='color: red'>(+3148 [+1665.6%])</span> 3,337 | <span style='color: green'>(-18 [-13.8%])</span> 112 | <span style='color: red'>(+24 [+40.7%])</span> 83 |
| `execute_preflight_insn_mi/s` | <span style='color: red'>(+5 [+40.5%])</span> 18.87 | -          | <span style='color: red'>(+2 [+9.2%])</span> 19.38 | <span style='color: red'>(+2 [+21.0%])</span> 11.05 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+14 [+7.7%])</span> 195.95 | <span style='color: red'>(+6886 [+1891.8%])</span> 7,250 | <span style='color: red'>(+2 [+1.0%])</span> 201 | <span style='color: red'>(+12 [+7.3%])</span> 177 |
| `memory_finalize_time_ms` | <span style='color: green'>(-0 [-94.6%])</span> 0.03 |  1 |  1 | <span style='color: green'>(+0 [NaN%])</span> 0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+28 [+7.9%])</span> 379.68 | <span style='color: red'>(+13344 [+1895.5%])</span> 14,048 | <span style='color: red'>(+64 [+16.7%])</span> 448 | <span style='color: red'>(+31 [+9.7%])</span> 351 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+9 [+14.5%])</span> 74.41 | <span style='color: red'>(+2623 [+2017.7%])</span> 2,753 | <span style='color: red'>(+1 [+1.3%])</span> 76 | <span style='color: red'>(+18 [+32.7%])</span> 73 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-2 [-4.7%])</span> 41.92 | <span style='color: red'>(+1463 [+1662.5%])</span> 1,551 | <span style='color: red'>(+34 [+70.8%])</span> 82 | <span style='color: green'>(-11 [-27.5%])</span> 29 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+10 [+13.2%])</span> 86.66 | <span style='color: red'>(+3053 [+1994.9%])</span> 3,206.48 | <span style='color: red'>(+1 [+0.8%])</span> 90.31 | <span style='color: red'>(+20 [+31.2%])</span> 83.21 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+4 [+5.7%])</span> 82.76 | <span style='color: red'>(+2906 [+1854.8%])</span> 3,062.25 |  88.08 | <span style='color: red'>(+11 [+15.9%])</span> 79.35 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+2 [+12.4%])</span> 18.31 | <span style='color: red'>(+645 [+1979.6%])</span> 677.65 | <span style='color: red'>(+1 [+6.7%])</span> 19.53 | <span style='color: red'>(+3 [+20.7%])</span> 17.23 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+5 [+8.0%])</span> 71.27 | <span style='color: red'>(+2505 [+1897.7%])</span> 2,637 | <span style='color: red'>(+33 [+49.3%])</span> 100 | <span style='color: green'>(-7 [-10.8%])</span> 58 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | app_prove_time_ms |
| --- | --- |
|  | 847 | 26,777 | 

| group | prove_segment_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| pairing | 714 | 1 | 466 | 38,547,215 | 82.61 | 0 | 

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
| pairing | AccessAdapterAir<16> | 10 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 11 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 12 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 13 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 14 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 15 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 16 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 17 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 18 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 19 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 2 | 131,072 |  | 16 | 25 | 5,373,952 | 
| pairing | AccessAdapterAir<16> | 20 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 21 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 22 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 23 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 24 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 25 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 26 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 27 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 28 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 29 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 3 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 30 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 31 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 32 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 33 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 34 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 35 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 36 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 4 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 5 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 6 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 7 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 8 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<16> | 9 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<32> | 0 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | AccessAdapterAir<32> | 1 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | AccessAdapterAir<32> | 10 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 11 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 12 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 13 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 14 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 15 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 16 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 17 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 18 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 19 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 2 | 65,536 |  | 16 | 41 | 3,735,552 | 
| pairing | AccessAdapterAir<32> | 20 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 21 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 22 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 23 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 24 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 25 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 26 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 27 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 28 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 29 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 3 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 30 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 31 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 32 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 33 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 34 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 35 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 36 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 4 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 5 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 6 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 7 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 8 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<32> | 9 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 1 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 10 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 11 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 12 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 13 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 14 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 15 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 16 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 17 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 18 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 19 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 2 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 20 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 21 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 22 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 23 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 24 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 25 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 26 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 27 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 28 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 29 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 3 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 30 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 31 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 32 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 33 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 34 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 35 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 36 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 4 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 5 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 6 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 7 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 8 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | AccessAdapterAir<8> | 9 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 10 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 11 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 12 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 13 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 14 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 15 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 16 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 17 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 18 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 19 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 20 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 21 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 22 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 23 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 24 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 25 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 26 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 27 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 28 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 29 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 30 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 31 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 32 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 33 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 34 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 35 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 36 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 7 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 8 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | BitwiseOperationLookupAir<8> | 9 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | MemoryMerkleAir<8> | 1 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | MemoryMerkleAir<8> | 10 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 11 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 12 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 13 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 14 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 15 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 16 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 17 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 18 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 19 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 2 | 16,384 |  | 16 | 32 | 786,432 | 
| pairing | MemoryMerkleAir<8> | 20 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 21 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 22 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 23 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 24 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 25 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 26 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 27 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 28 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 29 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 3 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 30 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 31 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 32 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 33 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 34 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 35 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 36 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 4 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 5 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 6 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 7 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 8 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | MemoryMerkleAir<8> | 9 | 2,048 |  | 16 | 32 | 98,304 | 
| pairing | PersistentBoundaryAir<8> | 0 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PersistentBoundaryAir<8> | 1 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PersistentBoundaryAir<8> | 10 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 11 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 12 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 13 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 14 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 15 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 16 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 17 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 18 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 19 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 2 | 16,384 |  | 12 | 20 | 524,288 | 
| pairing | PersistentBoundaryAir<8> | 20 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 21 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 22 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 23 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 24 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 25 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 26 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 27 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 28 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 29 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 3 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 30 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 31 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 32 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 33 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 34 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 35 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 36 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 4 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 5 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 6 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 7 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 8 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PersistentBoundaryAir<8> | 9 | 2,048 |  | 12 | 20 | 65,536 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 10 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 11 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 12 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 13 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 14 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 15 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 16 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 17 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 18 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 19 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 16,384 |  | 8 | 300 | 5,046,272 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 20 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 21 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 22 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 23 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 24 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 25 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 26 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 27 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 28 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 29 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 30 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 31 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 32 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 33 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 34 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 35 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 36 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 7 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 8 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 9 | 2,048 |  | 8 | 300 | 630,784 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 1 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 10 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 11 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 12 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 13 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 14 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 15 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 16 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 17 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 18 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 19 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 2 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 20 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 21 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 22 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 23 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 24 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 25 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 26 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 27 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 28 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 29 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 3 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 30 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 31 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 32 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 33 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 34 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 35 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | ProgramAir | 36 | 32,768 |  | 8 | 10 | 589,824 | 
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
| pairing | RangeTupleCheckerAir<2> | 15 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 16 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 17 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 18 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 19 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 20 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 21 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 22 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 23 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 24 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 25 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 26 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 27 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 28 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 29 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 30 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 31 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 32 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 33 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 34 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 35 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | RangeTupleCheckerAir<2> | 36 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
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
| pairing | VariableRangeCheckerAir | 15 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 16 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 17 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 18 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 19 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 20 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 21 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 22 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 23 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 24 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 25 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 26 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 27 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 28 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 29 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 30 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 31 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 32 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 33 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 34 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 35 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 36 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 7 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 8 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VariableRangeCheckerAir | 9 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 10 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 11 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 12 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 13 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 14 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 15 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 16 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 17 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 18 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 19 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 20 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 21 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 22 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 23 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 24 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 25 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 26 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 27 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 28 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 29 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 30 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 31 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 32 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 33 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 34 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 35 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 36 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 7 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 8 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 9 | 524,288 |  | 52 | 36 | 46,137,344 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 10 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 11 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 12 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 13 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 14 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 15 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 16 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 17 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 18 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 19 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 20 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 21 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 22 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 23 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 24 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 25 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 26 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 27 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 28 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 29 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 30 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 31 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 32 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 33 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 34 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 35 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 36 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 7 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 8 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 9 | 32,768 |  | 40 | 37 | 2,523,136 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 1,024 |  | 52 | 53 | 107,520 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 10 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 11 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 12 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 13 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 14 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 15 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 16 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 17 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 18 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 19 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 1,024 |  | 52 | 53 | 107,520 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 20 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 21 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 22 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 23 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 24 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 25 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 26 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 27 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 28 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 29 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 3 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 30 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 31 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 32 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 33 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 34 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 35 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 36 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 5 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 6 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 7 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 8 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 9 | 128 |  | 52 | 53 | 13,440 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 10 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 11 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 12 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 13 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 14 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 15 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 16 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 17 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 18 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 19 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 20 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 21 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 22 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 23 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 24 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 25 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 26 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 27 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 28 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 29 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 30 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 31 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 32 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 33 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 34 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 35 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 36 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 7 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 8 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 9 | 131,072 |  | 28 | 26 | 7,077,888 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 10 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 11 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 12 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 13 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 14 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 15 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 16 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 17 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 18 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 19 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 20 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 21 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 22 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 23 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 24 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 25 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 26 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 27 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 28 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 29 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 3 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 30 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 31 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 32 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 33 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 34 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 35 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 36 | 65,536 |  | 32 | 32 | 4,194,304 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 5 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 7 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 8 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 9 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 10 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 11 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 12 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 13 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 14 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 15 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 16 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 17 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 18 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 19 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 4,096 |  | 28 | 18 | 188,416 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 20 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 21 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 22 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 23 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 24 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 25 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 26 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 27 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 28 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 29 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 30 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 31 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 32 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 33 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 34 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 35 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 36 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 7 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 8 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 9 | 1,024 |  | 28 | 18 | 47,104 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 8 |  | 56 | 166 | 1,776 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 1 | 8 |  | 56 | 166 | 1,776 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 36 | 16 |  | 56 | 166 | 3,552 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 10 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 11 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 12 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 13 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 14 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 15 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 16 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 17 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 18 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 19 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 20 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 21 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 22 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 23 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 24 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 25 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 26 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 27 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 28 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 29 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 3 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 30 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 31 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 32 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 33 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 34 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 35 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 36 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 5 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 7 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 8 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 9 | 32,768 |  | 36 | 28 | 2,097,152 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 10 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 11 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 12 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 13 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 14 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 15 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 16 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 17 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 18 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 19 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 20 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 21 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 22 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 23 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 24 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 25 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 26 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 27 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 28 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 29 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 3 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 30 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 31 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 32 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 33 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 34 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 35 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 36 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 5 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 7 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 8 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 9 | 524,288 |  | 52 | 41 | 48,758,784 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 1 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 128 |  | 72 | 39 | 14,208 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 256 |  | 52 | 31 | 21,248 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 10 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 11 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 12 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 13 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 14 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 15 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 16 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 17 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 18 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 19 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 20 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 21 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 22 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 23 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 24 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 25 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 26 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 27 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 28 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 29 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 3 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 30 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 31 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 32 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 33 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 34 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 35 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 36 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 5 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 7 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 8 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 9 | 16,384 |  | 28 | 20 | 786,432 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 1 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 512 |  | 320 | 263 | 298,496 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 1 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 10 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 11 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 12 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 13 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 14 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 15 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 16 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 17 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 18 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 19 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 20 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 21 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 22 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 23 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 24 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 25 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 26 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 27 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 28 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 29 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 3 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 30 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 31 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 32 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 33 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 34 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 35 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 36 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 4 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 5 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 6 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 7 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 8 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 9 | 8,192 |  | 604 | 497 | 9,019,392 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 10 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 11 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 12 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 13 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 14 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 15 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 16 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 17 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 18 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 19 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 20 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 21 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 22 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 23 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 24 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 25 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 26 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 27 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 28 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 29 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 30 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 31 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 32 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 33 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 34 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 35 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 36 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 7 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 8 | 2 | 1 | 16 | 5 | 42 | 
| pairing | VmConnectorAir | 9 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | query phase_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | partially_prove_time_ms | open_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s | evaluate matrix_time_ms | eval_and_commit_quotient_time_ms | build fri inputs_time_ms | OpeningProverGpu::open_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 186 | 846 | 32,754,680 | 172,558,444 | 186 | 409 | 0 | 88.08 | 19.21 | 6 | 90.31 | 90 | 134 | 90 | 0 | 76 | 16,115,454 | 41 | 108 | 1,157,000 | 11.05 | 26 | 108 | 1 | 90 | 
| pairing | 1 | 177 | 718 | 33,192,784 | 172,412,682 | 177 | 387 | 2 | 88.05 | 18.30 | 6 | 89.25 | 73 | 130 | 72 | 1 | 76 | 16,536,310 | 37 | 112 | 1,144,000 | 18.27 | 25 | 107 | 1 | 72 | 
| pairing | 10 | 196 | 722 | 28,823,690 | 175,361,194 | 196 | 395 | 1 | 84.19 | 19.26 | 5 | 86.29 | 72 | 142 | 72 | 0 | 75 | 12,401,136 | 54 | 89 | 1,033,000 | 19.14 | 22 | 104 | 1 | 72 | 
| pairing | 11 | 199 | 724 | 28,835,138 | 175,361,194 | 199 | 395 | 1 | 80.26 | 17.38 | 5 | 85.81 | 91 | 130 | 90 | 0 | 75 | 12,412,736 | 41 | 89 | 1,033,000 | 19.19 | 22 | 98 | 1 | 90 | 
| pairing | 12 | 195 | 686 | 28,856,254 | 175,361,194 | 195 | 359 | 1 | 81.60 | 17.53 | 5 | 86.87 | 60 | 124 | 59 | 0 | 74 | 12,433,156 | 35 | 89 | 1,034,000 | 19.15 | 22 | 99 | 1 | 59 | 
| pairing | 13 | 197 | 776 | 28,821,598 | 175,361,194 | 197 | 448 | 1 | 83.17 | 18.73 | 5 | 86.19 | 100 | 170 | 100 | 0 | 74 | 12,405,396 | 82 | 89 | 1,034,000 | 19.11 | 22 | 102 | 1 | 100 | 
| pairing | 14 | 197 | 686 | 28,812,926 | 175,361,194 | 197 | 359 | 1 | 80.28 | 17.34 | 5 | 87.28 | 61 | 125 | 61 | 0 | 73 | 12,393,620 | 35 | 89 | 1,034,000 | 19.16 | 21 | 98 | 1 | 61 | 
| pairing | 15 | 196 | 715 | 28,802,734 | 175,361,194 | 196 | 388 | 1 | 84.17 | 19.39 | 5 | 86.15 | 63 | 145 | 62 | 0 | 75 | 12,388,252 | 56 | 89 | 1,033,000 | 19.11 | 22 | 104 | 1 | 62 | 
| pairing | 16 | 200 | 682 | 28,788,132 | 175,361,194 | 200 | 351 | 1 | 80.01 | 17.23 | 5 | 86.19 | 59 | 118 | 59 | 0 | 74 | 12,374,426 | 30 | 89 | 1,033,000 | 19.01 | 22 | 98 | 1 | 59 | 
| pairing | 17 | 198 | 729 | 28,819,714 | 175,361,194 | 198 | 401 | 1 | 83.15 | 18.59 | 5 | 86.33 | 74 | 149 | 74 | 0 | 74 | 12,399,392 | 61 | 89 | 1,033,000 | 19.20 | 22 | 102 | 1 | 74 | 
| pairing | 18 | 198 | 699 | 28,807,694 | 175,361,194 | 198 | 370 | 1 | 82.46 | 18.15 | 5 | 87.61 | 74 | 119 | 74 | 0 | 74 | 12,386,756 | 30 | 89 | 1,035,000 | 18.97 | 22 | 101 | 1 | 73 | 
| pairing | 19 | 196 | 716 | 28,784,258 | 175,361,194 | 196 | 389 | 1 | 83.38 | 18.55 | 5 | 87.27 | 91 | 119 | 91 | 0 | 75 | 12,367,376 | 30 | 89 | 1,034,000 | 19.13 | 22 | 102 | 1 | 91 | 
| pairing | 2 | 191 | 739 | 32,803,490 | 172,384,966 | 191 | 408 | 2 | 86.61 | 17.54 | 6 | 88.65 | 83 | 143 | 83 | 0 | 75 | 16,133,456 | 52 | 98 | 1,146,000 | 18.71 | 24 | 104 | 1 | 83 | 
| pairing | 20 | 198 | 686 | 28,798,686 | 175,361,194 | 198 | 358 | 1 | 79.35 | 18.71 | 5 | 83.21 | 66 | 116 | 66 | 0 | 75 | 12,384,788 | 31 | 89 | 1,033,000 | 19.26 | 22 | 98 | 1 | 66 | 
| pairing | 21 | 197 | 710 | 28,819,870 | 175,361,194 | 197 | 383 | 1 | 84.41 | 19.44 | 5 | 85.77 | 74 | 129 | 74 | 0 | 73 | 12,402,004 | 41 | 89 | 1,034,000 | 19.05 | 22 | 104 | 1 | 74 | 
| pairing | 22 | 195 | 688 | 28,830,718 | 175,361,194 | 195 | 361 | 1 | 84.35 | 19.53 | 5 | 86.11 | 61 | 120 | 61 | 0 | 75 | 12,415,988 | 31 | 89 | 1,033,000 | 19.10 | 22 | 104 | 1 | 61 | 
| pairing | 23 | 197 | 702 | 28,829,020 | 175,361,194 | 197 | 374 | 1 | 85.03 | 19.50 | 5 | 86.11 | 62 | 132 | 62 | 0 | 73 | 12,408,794 | 44 | 89 | 1,034,000 | 19.02 | 22 | 105 | 1 | 62 | 
| pairing | 24 | 196 | 712 | 28,812,670 | 175,361,194 | 196 | 385 | 1 | 84.01 | 19.47 | 5 | 85.67 | 59 | 147 | 59 | 0 | 74 | 12,394,932 | 59 | 89 | 1,034,000 | 19.16 | 22 | 104 | 1 | 59 | 
| pairing | 25 | 197 | 686 | 28,819,518 | 175,361,194 | 197 | 358 | 1 | 82.26 | 18.24 | 5 | 87.36 | 59 | 122 | 59 | 0 | 75 | 12,402,140 | 32 | 89 | 1,033,000 | 19.08 | 22 | 101 | 1 | 59 | 
| pairing | 26 | 198 | 697 | 28,790,010 | 175,361,194 | 198 | 368 | 1 | 83.56 | 18.38 | 5 | 86.92 | 59 | 131 | 58 | 0 | 74 | 12,375,888 | 42 | 89 | 1,033,000 | 19.18 | 22 | 102 | 1 | 58 | 
| pairing | 27 | 196 | 706 | 28,811,886 | 175,361,194 | 196 | 379 | 1 | 81.18 | 17.84 | 5 | 85.77 | 58 | 146 | 58 | 0 | 74 | 12,392,548 | 58 | 89 | 1,033,000 | 19.13 | 22 | 99 | 1 | 58 | 
| pairing | 28 | 198 | 716 | 28,833,758 | 175,361,194 | 198 | 387 | 1 | 82.48 | 18.22 | 5 | 87.49 | 81 | 128 | 81 | 0 | 75 | 12,410,972 | 38 | 88 | 1,034,000 | 19.38 | 22 | 101 | 1 | 81 | 
| pairing | 29 | 197 | 703 | 28,821,534 | 175,361,194 | 197 | 376 | 1 | 84.85 | 19.33 | 5 | 86.34 | 66 | 130 | 66 | 0 | 74 | 12,399,036 | 42 | 88 | 1,034,000 | 19.28 | 22 | 105 | 1 | 66 | 
| pairing | 3 | 201 | 695 | 28,812,546 | 175,361,194 | 201 | 369 | 1 | 84.10 | 19.19 | 6 | 85.89 | 61 | 129 | 61 | 0 | 74 | 12,398,288 | 41 | 83 | 1,034,000 | 19.01 | 22 | 104 | 1 | 61 | 
| pairing | 30 | 195 | 692 | 28,807,580 | 175,361,194 | 195 | 365 | 1 | 81.75 | 18.19 | 5 | 86.75 | 69 | 119 | 69 | 0 | 74 | 12,392,890 | 30 | 90 | 1,033,000 | 18.78 | 22 | 100 | 1 | 69 | 
| pairing | 31 | 198 | 696 | 28,818,706 | 175,361,194 | 198 | 366 | 1 | 82.75 | 18.56 | 5 | 86.42 | 63 | 126 | 63 | 0 | 74 | 12,404,288 | 37 | 89 | 1,032,000 | 19.19 | 22 | 102 | 1 | 63 | 
| pairing | 32 | 198 | 694 | 28,854,182 | 175,361,194 | 198 | 366 | 1 | 81.15 | 17.48 | 5 | 86.34 | 62 | 128 | 62 | 0 | 75 | 12,430,836 | 40 | 89 | 1,034,000 | 19.29 | 22 | 99 | 1 | 62 | 
| pairing | 33 | 196 | 704 | 28,785,294 | 175,361,194 | 196 | 376 | 1 | 81.27 | 17.52 | 5 | 87.35 | 70 | 130 | 70 | 0 | 75 | 12,371,324 | 40 | 90 | 1,034,000 | 19.11 | 22 | 99 | 1 | 70 | 
| pairing | 34 | 198 | 700 | 28,742,976 | 175,361,194 | 198 | 371 | 1 | 80.49 | 17.39 | 6 | 87.43 | 69 | 127 | 69 | 0 | 75 | 12,328,734 | 37 | 90 | 1,033,000 | 18.98 | 22 | 98 | 1 | 69 | 
| pairing | 35 | 196 | 720 | 28,783,922 | 175,361,194 | 196 | 392 | 1 | 81.82 | 17.81 | 5 | 87.34 | 93 | 122 | 93 | 0 | 75 | 12,364,920 | 33 | 90 | 1,033,000 | 19.17 | 22 | 100 | 1 | 93 | 
| pairing | 36 | 196 | 714 | 28,156,412 | 171,170,442 | 196 | 391 | 1 | 83.79 | 19.44 | 5 | 85.11 | 95 | 116 | 94 | 0 | 74 | 11,999,250 | 29 | 86 | 997,215 | 19.22 | 22 | 104 | 1 | 94 | 
| pairing | 4 | 195 | 691 | 28,806,542 | 175,361,194 | 195 | 364 | 1 | 81.65 | 17.51 | 5 | 86.63 | 69 | 121 | 68 | 0 | 73 | 12,388,212 | 33 | 89 | 1,033,000 | 19.13 | 21 | 100 | 1 | 68 | 
| pairing | 5 | 196 | 690 | 28,805,264 | 175,361,194 | 196 | 363 | 1 | 81.28 | 17.98 | 5 | 86.81 | 61 | 125 | 61 | 0 | 75 | 12,391,054 | 36 | 89 | 1,033,000 | 19.16 | 22 | 100 | 1 | 61 | 
| pairing | 6 | 198 | 700 | 28,783,482 | 175,361,194 | 198 | 372 | 1 | 80.43 | 17.61 | 6 | 85.62 | 76 | 122 | 76 | 0 | 74 | 12,367,040 | 34 | 89 | 1,033,000 | 19.14 | 22 | 98 | 1 | 75 | 
| pairing | 7 | 196 | 709 | 28,800,862 | 175,361,194 | 196 | 382 | 1 | 81.49 | 17.43 | 6 | 86.66 | 70 | 137 | 70 | 0 | 74 | 12,381,420 | 49 | 89 | 1,033,000 | 18.94 | 22 | 99 | 1 | 70 | 
| pairing | 8 | 196 | 703 | 28,836,446 | 175,361,194 | 196 | 377 | 1 | 81.40 | 17.48 | 5 | 86.54 | 74 | 127 | 73 | 0 | 75 | 12,419,468 | 38 | 89 | 1,033,000 | 19.20 | 22 | 99 | 1 | 73 | 
| pairing | 9 | 196 | 733 | 28,833,884 | 175,361,194 | 196 | 406 | 1 | 81.98 | 18.18 | 5 | 86.63 | 69 | 161 | 69 | 0 | 74 | 12,413,922 | 72 | 89 | 1,034,000 | 19.21 | 22 | 100 | 1 | 69 | 

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
| pairing | 1 | 0 | 2,830,644 | 2,013,265,921 | 
| pairing | 1 | 1 | 10,199,056 | 2,013,265,921 | 
| pairing | 1 | 2 | 1,415,322 | 2,013,265,921 | 
| pairing | 1 | 3 | 13,892,132 | 2,013,265,921 | 
| pairing | 1 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 1 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 1 | 6 | 3,146,928 | 2,013,265,921 | 
| pairing | 1 | 7 | 3,072 | 2,013,265,921 | 
| pairing | 1 | 8 | 32,552,114 | 2,013,265,921 | 
| pairing | 10 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 10 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 10 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 10 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 10 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 10 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 10 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 10 | 7 |  | 2,013,265,921 | 
| pairing | 10 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 11 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 11 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 11 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 11 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 11 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 11 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 11 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 11 | 7 |  | 2,013,265,921 | 
| pairing | 11 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 12 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 12 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 12 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 12 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 12 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 12 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 12 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 12 | 7 |  | 2,013,265,921 | 
| pairing | 12 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 13 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 13 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 13 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 13 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 13 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 13 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 13 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 13 | 7 |  | 2,013,265,921 | 
| pairing | 13 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 14 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 14 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 14 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 14 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 14 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 14 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 14 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 14 | 7 |  | 2,013,265,921 | 
| pairing | 14 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 15 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 15 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 15 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 15 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 15 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 15 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 15 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 15 | 7 |  | 2,013,265,921 | 
| pairing | 15 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 16 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 16 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 16 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 16 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 16 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 16 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 16 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 16 | 7 |  | 2,013,265,921 | 
| pairing | 16 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 17 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 17 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 17 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 17 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 17 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 17 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 17 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 17 | 7 |  | 2,013,265,921 | 
| pairing | 17 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 18 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 18 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 18 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 18 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 18 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 18 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 18 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 18 | 7 |  | 2,013,265,921 | 
| pairing | 18 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 19 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 19 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 19 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 19 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 19 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 19 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 19 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 19 | 7 |  | 2,013,265,921 | 
| pairing | 19 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 2 | 0 | 2,830,092 | 2,013,265,921 | 
| pairing | 2 | 1 | 10,197,296 | 2,013,265,921 | 
| pairing | 2 | 2 | 1,415,046 | 2,013,265,921 | 
| pairing | 2 | 3 | 13,889,592 | 2,013,265,921 | 
| pairing | 2 | 4 | 65,536 | 2,013,265,921 | 
| pairing | 2 | 5 | 32,768 | 2,013,265,921 | 
| pairing | 2 | 6 | 3,146,888 | 2,013,265,921 | 
| pairing | 2 | 7 | 2,048 | 2,013,265,921 | 
| pairing | 2 | 8 | 32,545,922 | 2,013,265,921 | 
| pairing | 20 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 20 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 20 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 20 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 20 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 20 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 20 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 20 | 7 |  | 2,013,265,921 | 
| pairing | 20 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 21 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 21 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 21 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 21 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 21 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 21 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 21 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 21 | 7 |  | 2,013,265,921 | 
| pairing | 21 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 22 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 22 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 22 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 22 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 22 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 22 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 22 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 22 | 7 |  | 2,013,265,921 | 
| pairing | 22 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 23 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 23 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 23 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 23 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 23 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 23 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 23 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 23 | 7 |  | 2,013,265,921 | 
| pairing | 23 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 24 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 24 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 24 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 24 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 24 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 24 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 24 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 24 | 7 |  | 2,013,265,921 | 
| pairing | 24 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 25 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 25 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 25 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 25 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 25 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 25 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 25 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 25 | 7 |  | 2,013,265,921 | 
| pairing | 25 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 26 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 26 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 26 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 26 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 26 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 26 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 26 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 26 | 7 |  | 2,013,265,921 | 
| pairing | 26 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 27 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 27 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 27 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 27 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 27 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 27 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 27 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 27 | 7 |  | 2,013,265,921 | 
| pairing | 27 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 28 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 28 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 28 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 28 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 28 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 28 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 28 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 28 | 7 |  | 2,013,265,921 | 
| pairing | 28 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 29 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 29 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 29 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 29 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 29 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 29 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 29 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 29 | 7 |  | 2,013,265,921 | 
| pairing | 29 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 3 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 3 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 3 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 3 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 3 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 3 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 3 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 3 | 7 |  | 2,013,265,921 | 
| pairing | 3 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 30 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 30 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 30 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 30 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 30 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 30 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 30 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 30 | 7 |  | 2,013,265,921 | 
| pairing | 30 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 31 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 31 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 31 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 31 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 31 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 31 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 31 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 31 | 7 |  | 2,013,265,921 | 
| pairing | 31 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 32 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 32 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 32 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 32 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 32 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 32 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 32 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 32 | 7 |  | 2,013,265,921 | 
| pairing | 32 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 33 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 33 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 33 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 33 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 33 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 33 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 33 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 33 | 7 |  | 2,013,265,921 | 
| pairing | 33 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 34 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 34 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 34 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 34 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 34 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 34 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 34 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 34 | 7 |  | 2,013,265,921 | 
| pairing | 34 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 35 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 35 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 35 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 35 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 35 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 35 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 35 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 35 | 7 |  | 2,013,265,921 | 
| pairing | 35 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 36 | 0 | 2,689,316 | 2,013,265,921 | 
| pairing | 36 | 1 | 10,490,784 | 2,013,265,921 | 
| pairing | 36 | 2 | 1,344,658 | 2,013,265,921 | 
| pairing | 36 | 3 | 13,930,020 | 2,013,265,921 | 
| pairing | 36 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 36 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 36 | 6 | 3,001,888 | 2,013,265,921 | 
| pairing | 36 | 7 |  | 2,013,265,921 | 
| pairing | 36 | 8 | 32,421,274 | 2,013,265,921 | 
| pairing | 4 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 4 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 4 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 4 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 4 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 4 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 4 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 4 | 7 |  | 2,013,265,921 | 
| pairing | 4 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 5 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 5 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 5 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 5 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 5 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 5 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 5 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 5 | 7 |  | 2,013,265,921 | 
| pairing | 5 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 6 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 6 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 6 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 6 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 6 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 6 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 6 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 6 | 7 |  | 2,013,265,921 | 
| pairing | 6 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 7 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 7 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 7 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 7 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 7 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 7 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 7 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 7 | 7 |  | 2,013,265,921 | 
| pairing | 7 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 8 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 8 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 8 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 8 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 8 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 8 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 8 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 8 | 7 |  | 2,013,265,921 | 
| pairing | 8 | 8 | 33,272,842 | 2,013,265,921 | 
| pairing | 9 | 0 | 2,820,356 | 2,013,265,921 | 
| pairing | 9 | 1 | 10,752,768 | 2,013,265,921 | 
| pairing | 9 | 2 | 1,410,178 | 2,013,265,921 | 
| pairing | 9 | 3 | 14,192,004 | 2,013,265,921 | 
| pairing | 9 | 4 | 8,192 | 2,013,265,921 | 
| pairing | 9 | 5 | 4,096 | 2,013,265,921 | 
| pairing | 9 | 6 | 3,132,928 | 2,013,265,921 | 
| pairing | 9 | 7 |  | 2,013,265,921 | 
| pairing | 9 | 8 | 33,272,842 | 2,013,265,921 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/604d00a2c4e9edabf0ac21cf6fadc6ba0627df17

Max Segment Length: 1048476

Instance Type: g6.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/19535910683)
