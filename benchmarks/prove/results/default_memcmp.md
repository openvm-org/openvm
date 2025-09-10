| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  17.50 |  17.50 |
| random_slice_comparison |  17.49 |  17.49 |


| random_slice_comparison |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  17,490 |  17,490 |  17,490 |  17,490 |
| `main_cells_used     ` |  144,705,218 |  144,705,218 |  144,705,218 |  144,705,218 |
| `total_cells_used    ` |  338,537,276 |  338,537,276 |  338,537,276 |  338,537,276 |
| `execute_metered_time_ms` |  13 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  309.25 | -          |  309.25 |  309.25 |
| `execute_preflight_insns` |  4,040,671 |  4,040,671 |  4,040,671 |  4,040,671 |
| `execute_preflight_time_ms` |  86 |  86 |  86 |  86 |
| `execute_preflight_insn_mi/s` |  48.13 | -          |  48.13 |  48.13 |
| `trace_gen_time_ms   ` |  839 |  839 |  839 |  839 |
| `memory_finalize_time_ms` |  1 |  1 |  1 |  1 |
| `stark_prove_excluding_trace_time_ms` |  16,526 |  16,526 |  16,526 |  16,526 |
| `main_trace_commit_time_ms` |  3,293 |  3,293 |  3,293 |  3,293 |
| `generate_perm_trace_time_ms` |  2,442 |  2,442 |  2,442 |  2,442 |
| `perm_trace_commit_time_ms` |  4,017 |  4,017 |  4,017 |  4,017 |
| `quotient_poly_compute_time_ms` |  1,855 |  1,855 |  1,855 |  1,855 |
| `quotient_poly_commit_time_ms` |  1,226 |  1,226 |  1,226 |  1,226 |
| `pcs_opening_time_ms ` |  3,652 |  3,652 |  3,652 |  3,652 |



<details>
<summary>Detailed Metrics</summary>

|  | memory_to_vec_partition_time_ms | keygen_time_ms | app proof_time_ms |
| --- | --- | --- |
|  | 36 | 294 | 17,666 | 

| group | prove_segment_time_ms | memory_to_vec_partition_time_ms | fri.log_blowup | execute_metered_time_ms | execute_metered_insns | execute_metered_insn_mi/s | compute_user_public_values_proof_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_slice_comparison | 17,490 | 29 | 1 | 13 | 4,040,671 | 309.25 | 132 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| random_slice_comparison | AccessAdapterAir<16> | 2 | 5 | 12 | 
| random_slice_comparison | AccessAdapterAir<2> | 2 | 5 | 12 | 
| random_slice_comparison | AccessAdapterAir<32> | 2 | 5 | 12 | 
| random_slice_comparison | AccessAdapterAir<4> | 2 | 5 | 12 | 
| random_slice_comparison | AccessAdapterAir<8> | 2 | 5 | 12 | 
| random_slice_comparison | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| random_slice_comparison | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| random_slice_comparison | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| random_slice_comparison | PhantomAir | 2 | 3 | 5 | 
| random_slice_comparison | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| random_slice_comparison | ProgramAir | 1 | 1 | 4 | 
| random_slice_comparison | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| random_slice_comparison | Rv32HintStoreAir | 2 | 18 | 28 | 
| random_slice_comparison | VariableRangeCheckerAir | 1 | 1 | 4 | 
| random_slice_comparison | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| random_slice_comparison | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| random_slice_comparison | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| random_slice_comparison | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| random_slice_comparison | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| random_slice_comparison | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| random_slice_comparison | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| random_slice_comparison | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| random_slice_comparison | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| random_slice_comparison | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| random_slice_comparison | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| random_slice_comparison | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| random_slice_comparison | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| random_slice_comparison | VmConnectorAir | 2 | 5 | 11 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_slice_comparison | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| random_slice_comparison | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| random_slice_comparison | MemoryMerkleAir<8> | 0 | 65,536 |  | 16 | 32 | 3,145,728 | 
| random_slice_comparison | PersistentBoundaryAir<8> | 0 | 65,536 |  | 12 | 20 | 2,097,152 | 
| random_slice_comparison | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| random_slice_comparison | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| random_slice_comparison | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| random_slice_comparison | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| random_slice_comparison | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 2,097,152 |  | 52 | 36 | 184,549,376 | 
| random_slice_comparison | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 1 |  | 40 | 37 | 77 | 
| random_slice_comparison | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 2,097,152 |  | 28 | 26 | 113,246,208 | 
| random_slice_comparison | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8,192 |  | 32 | 32 | 524,288 | 
| random_slice_comparison | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 64 |  | 28 | 18 | 2,944 | 
| random_slice_comparison | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 64 |  | 36 | 28 | 4,096 | 
| random_slice_comparison | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 2,097,152 |  | 52 | 41 | 195,035,136 | 
| random_slice_comparison | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 2 |  | 52 | 31 | 166 | 
| random_slice_comparison | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32 |  | 28 | 20 | 1,536 | 
| random_slice_comparison | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cells_used | total_cells | system_trace_gen_time_ms | stark_prove_excluding_trace_time_ms | single_trace_gen_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_preflight_time_ms | execute_preflight_insns | execute_preflight_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_slice_comparison | 0 | 839 | 17,490 | 338,537,276 | 508,728,989 | 839 | 16,526 | 228 | 1,855 | 1,226 | 4,017 | 3,652 | 28 | 1 | 3,293 | 144,705,218 | 2,442 | 86 | 4,040,671 | 48.13 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| random_slice_comparison | 0 | 0 | 12,599,626 | 2,013,265,921 | 
| random_slice_comparison | 0 | 1 | 33,849,810 | 2,013,265,921 | 
| random_slice_comparison | 0 | 2 | 6,299,813 | 2,013,265,921 | 
| random_slice_comparison | 0 | 3 | 37,913,302 | 2,013,265,921 | 
| random_slice_comparison | 0 | 4 | 262,144 | 2,013,265,921 | 
| random_slice_comparison | 0 | 5 | 131,072 | 2,013,265,921 | 
| random_slice_comparison | 0 | 6 | 10,502,563 | 2,013,265,921 | 
| random_slice_comparison | 0 | 7 | 8 | 2,013,265,921 | 
| random_slice_comparison | 0 | 8 | 102,484,290 | 2,013,265,921 | 

</details>

