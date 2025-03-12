| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+6 [+81.3%])</span> 13.97 | <span style='color: red'>(+3 [+79.0%])</span> 7.88 |
| regex_program | <span style='color: red'>(+6 [+81.3%])</span> 13.97 | <span style='color: red'>(+3 [+79.0%])</span> 7.88 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+3133 [+81.3%])</span> 6,987.50 | <span style='color: red'>(+6266 [+81.3%])</span> 13,975 | <span style='color: red'>(+3481 [+79.0%])</span> 7,885 | <span style='color: red'>(+2785 [+84.3%])</span> 6,090 |
| `main_cells_used     ` |  83,694,725 |  167,389,450 |  93,699,990 |  73,689,460 |
| `total_cycles        ` |  2,070,082 |  4,140,164 |  2,225,434 |  1,914,730 |
| `execute_time_ms     ` | <span style='color: red'>(+53 [+15.9%])</span> 387 | <span style='color: red'>(+106 [+15.9%])</span> 774 | <span style='color: red'>(+97 [+26.1%])</span> 468 | <span style='color: red'>(+9 [+3.0%])</span> 306 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+320 [+33.5%])</span> 1,276 | <span style='color: red'>(+641 [+33.5%])</span> 2,552 | <span style='color: red'>(+210 [+19.0%])</span> 1,314 | <span style='color: red'>(+431 [+53.4%])</span> 1,238 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+2760 [+107.6%])</span> 5,324.50 | <span style='color: red'>(+5519 [+107.6%])</span> 10,649 | <span style='color: red'>(+3174 [+108.4%])</span> 6,103 | <span style='color: red'>(+2345 [+106.5%])</span> 4,546 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+702 [+157.4%])</span> 1,148 | <span style='color: red'>(+1404 [+157.4%])</span> 2,296 | <span style='color: red'>(+798 [+152.3%])</span> 1,322 | <span style='color: red'>(+606 [+164.7%])</span> 974 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+4 [+2.1%])</span> 170.50 | <span style='color: red'>(+7 [+2.1%])</span> 341 | <span style='color: red'>(+2 [+1.1%])</span> 190 | <span style='color: red'>(+5 [+3.4%])</span> 151 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+382 [+58.9%])</span> 1,030.50 | <span style='color: red'>(+764 [+58.9%])</span> 2,061 | <span style='color: red'>(+431 [+57.2%])</span> 1,184 | <span style='color: red'>(+333 [+61.2%])</span> 877 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+388 [+106.0%])</span> 754 | <span style='color: red'>(+776 [+106.0%])</span> 1,508 | <span style='color: red'>(+463 [+110.5%])</span> 882 | <span style='color: red'>(+313 [+100.0%])</span> 626 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+662 [+243.6%])</span> 933 | <span style='color: red'>(+1323 [+243.6%])</span> 1,866 | <span style='color: red'>(+766 [+243.2%])</span> 1,081 | <span style='color: red'>(+557 [+244.3%])</span> 785 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+621 [+94.5%])</span> 1,278 | <span style='color: red'>(+1242 [+94.5%])</span> 2,556 | <span style='color: red'>(+716 [+99.4%])</span> 1,436 | <span style='color: red'>(+526 [+88.6%])</span> 1,120 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| regex_program | 2 | 641 | 43 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<16> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<2> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<32> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<4> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<64> | 4 | 5 | 11 | 
| regex_program | AccessAdapterAir<8> | 4 | 5 | 11 | 
| regex_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| regex_program | KeccakVmAir | 4 | 321 | 4,380 | 
| regex_program | MemoryMerkleAir<8> | 4 | 4 | 38 | 
| regex_program | PersistentBoundaryAir<8> | 4 | 3 | 5 | 
| regex_program | PhantomAir | 4 | 3 | 4 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| regex_program | ProgramAir | 1 | 1 | 4 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| regex_program | Rv32HintStoreAir | 4 | 18 | 23 | 
| regex_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 20 | 31 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 18 | 36 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 4 | 24 | 85 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 11 | 17 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 4 | 13 | 32 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 10 | 15 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 4 | 16 | 16 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 4 | 18 | 27 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 4 | 17 | 34 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 4 | 25 | 76 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 4 | 24 | 23 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 4 | 19 | 13 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 4 | 12 | 11 | 
| regex_program | VmConnectorAir | 4 | 5 | 9 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<2> | 1 | 64 |  | 12 | 11 | 1,472 | 
| regex_program | AccessAdapterAir<4> | 1 | 32 |  | 12 | 13 | 800 | 
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 12 | 17 | 3,801,088 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 12 | 17 | 59,392 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 0 | 1 |  | 532 | 3,163 | 3,695 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 532 | 3,163 | 118,240 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 12 | 32 | 5,767,168 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 12 | 32 | 180,224 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 8 | 20 | 3,670,016 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 8 | 20 | 57,344 | 
| regex_program | PhantomAir | 0 | 1 |  | 8 | 6 | 14 | 
| regex_program | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 24 | 32 | 917,504 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 524,288 |  | 28 | 36 | 33,554,432 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 24 | 37 | 1,998,848 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 28 | 53 | 10,616,832 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 16 | 26 | 5,505,024 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 20 | 32 | 6,815,744 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 16 | 18 | 2,228,224 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 20 | 28 | 6,291,456 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 20 | 28 | 3,145,728 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 28 | 36 | 65,536 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 2 |  | 28 | 36 | 128 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 28 | 41 | 72,351,744 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 28 | 41 | 72,351,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 40 | 59 | 12,672 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 40 | 39 | 20,224 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 28 | 31 | 1,933,312 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 16 | 20 | 1,179,648 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 16 | 20 | 1,179,648 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 1,314 | 7,885 | 2,225,434 | 210,931,487 | 6,103 | 882 | 1,081 | 1,184 | 1,436 | 1,322 | 93,699,990 | 190 | 468 | 
| regex_program | 1 | 1,238 | 6,090 | 1,914,730 | 150,470,512 | 4,546 | 626 | 785 | 877 | 1,120 | 974 | 73,689,460 | 151 | 306 | 

</details>


Commit: https://github.com/openvm-org/openvm/commit/0943cae21d53c35fef04cc6c167fcc34d0ac8bcf

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13821340664)
