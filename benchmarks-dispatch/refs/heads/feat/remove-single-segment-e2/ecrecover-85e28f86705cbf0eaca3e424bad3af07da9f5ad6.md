| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  27.72 |  27.72 |
| ecrecover_program |  1.87 |  1.87 |
| leaf |  25.47 |  25.47 |


| ecrecover_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  1,872 |  1,872 |  1,872 |  1,872 |
| `main_cells_used     ` | <span style='color: red'>(+104407 [+1.3%])</span> 8,193,677 | <span style='color: red'>(+104407 [+1.3%])</span> 8,193,677 | <span style='color: red'>(+104407 [+1.3%])</span> 8,193,677 | <span style='color: red'>(+104407 [+1.3%])</span> 8,193,677 |
| `total_cycles        ` | <span style='color: red'>(+1224 [+0.9%])</span> 137,465 | <span style='color: red'>(+1224 [+0.9%])</span> 137,465 | <span style='color: red'>(+1224 [+0.9%])</span> 137,465 | <span style='color: red'>(+1224 [+0.9%])</span> 137,465 |
| `execute_metered_time_ms` |  372 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.40 | -          |  0.40 |  0.40 |
| `execute_e3_time_ms  ` |  559 |  559 |  559 |  559 |
| `execute_e3_insn_mi/s` |  0.25 | -          |  0.25 |  0.25 |
| `trace_gen_time_ms   ` | <span style='color: red'>(+233 [+206.2%])</span> 346 | <span style='color: red'>(+233 [+206.2%])</span> 346 | <span style='color: red'>(+233 [+206.2%])</span> 346 | <span style='color: red'>(+233 [+206.2%])</span> 346 |
| `memory_finalize_time_ms` |  77 |  77 |  77 |  77 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `merkle_finalize_time_ms` |  70 |  70 |  70 |  70 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-6 [-0.6%])</span> 967 | <span style='color: green'>(-6 [-0.6%])</span> 967 | <span style='color: green'>(-6 [-0.6%])</span> 967 | <span style='color: green'>(-6 [-0.6%])</span> 967 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-1 [-0.8%])</span> 132 | <span style='color: green'>(-1 [-0.8%])</span> 132 | <span style='color: green'>(-1 [-0.8%])</span> 132 | <span style='color: green'>(-1 [-0.8%])</span> 132 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+3.3%])</span> 31 | <span style='color: red'>(+1 [+3.3%])</span> 31 | <span style='color: red'>(+1 [+3.3%])</span> 31 | <span style='color: red'>(+1 [+3.3%])</span> 31 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-14 [-8.2%])</span> 157 | <span style='color: green'>(-14 [-8.2%])</span> 157 | <span style='color: green'>(-14 [-8.2%])</span> 157 | <span style='color: green'>(-14 [-8.2%])</span> 157 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-2 [-2.5%])</span> 78 | <span style='color: green'>(-2 [-2.5%])</span> 78 | <span style='color: green'>(-2 [-2.5%])</span> 78 | <span style='color: green'>(-2 [-2.5%])</span> 78 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+4 [+3.2%])</span> 129 | <span style='color: red'>(+4 [+3.2%])</span> 129 | <span style='color: red'>(+4 [+3.2%])</span> 129 | <span style='color: red'>(+4 [+3.2%])</span> 129 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+3 [+0.7%])</span> 421 | <span style='color: red'>(+3 [+0.7%])</span> 421 | <span style='color: red'>(+3 [+0.7%])</span> 421 | <span style='color: red'>(+3 [+0.7%])</span> 421 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  25,473 |  25,473 |  25,473 |  25,473 |
| `main_cells_used     ` | <span style='color: green'>(-2160716 [-0.9%])</span> 242,933,756 | <span style='color: green'>(-2160716 [-0.9%])</span> 242,933,756 | <span style='color: green'>(-2160716 [-0.9%])</span> 242,933,756 | <span style='color: green'>(-2160716 [-0.9%])</span> 242,933,756 |
| `total_cycles        ` |  3,015,310 |  3,015,310 |  3,015,310 |  3,015,310 |
| `execute_metered_time_ms` |  5,918 |  5,918 |  5,918 |  5,918 |
| `execute_metered_insn_mi/s` |  0.51 | -          |  0.51 |  0.51 |
| `execute_e3_time_ms  ` |  10,891 |  10,891 |  10,891 |  10,891 |
| `execute_e3_insn_mi/s` |  0.28 | -          |  0.28 |  0.28 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1799 [-75.5%])</span> 585 | <span style='color: green'>(-1799 [-75.5%])</span> 585 | <span style='color: green'>(-1799 [-75.5%])</span> 585 | <span style='color: green'>(-1799 [-75.5%])</span> 585 |
| `memory_finalize_time_ms` |  19 |  19 |  19 |  19 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-186 [-2.3%])</span> 8,079 | <span style='color: green'>(-186 [-2.3%])</span> 8,079 | <span style='color: green'>(-186 [-2.3%])</span> 8,079 | <span style='color: green'>(-186 [-2.3%])</span> 8,079 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-14 [-0.9%])</span> 1,501 | <span style='color: green'>(-14 [-0.9%])</span> 1,501 | <span style='color: green'>(-14 [-0.9%])</span> 1,501 | <span style='color: green'>(-14 [-0.9%])</span> 1,501 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+9 [+1.1%])</span> 800 | <span style='color: red'>(+9 [+1.1%])</span> 800 | <span style='color: red'>(+9 [+1.1%])</span> 800 | <span style='color: red'>(+9 [+1.1%])</span> 800 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-53 [-2.5%])</span> 2,062 | <span style='color: green'>(-53 [-2.5%])</span> 2,062 | <span style='color: green'>(-53 [-2.5%])</span> 2,062 | <span style='color: green'>(-53 [-2.5%])</span> 2,062 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-96 [-9.8%])</span> 886 | <span style='color: green'>(-96 [-9.8%])</span> 886 | <span style='color: green'>(-96 [-9.8%])</span> 886 | <span style='color: green'>(-96 [-9.8%])</span> 886 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-11 [-1.7%])</span> 632 | <span style='color: green'>(-11 [-1.7%])</span> 632 | <span style='color: green'>(-11 [-1.7%])</span> 632 | <span style='color: green'>(-11 [-1.7%])</span> 632 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-23 [-1.0%])</span> 2,191 | <span style='color: green'>(-23 [-1.0%])</span> 2,191 | <span style='color: green'>(-23 [-1.0%])</span> 2,191 | <span style='color: green'>(-23 [-1.0%])</span> 2,191 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 51 | 11 | 7,178 | 25,613 | 

| group | single_leaf_agg_time_ms | num_segments | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program |  | 1 |  | 24 | 137,466 | 1 | 6,372 | 372 | 0.40 | 
| leaf | 25,611 |  | 1 |  |  | 1 |  |  |  | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<2> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<32> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<4> | 2 | 5 | 12 | 
| ecrecover_program | AccessAdapterAir<8> | 2 | 5 | 12 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| ecrecover_program | KeccakVmAir | 2 | 321 | 4,513 | 
| ecrecover_program | MemoryMerkleAir<8> | 2 | 4 | 39 | 
| ecrecover_program | PersistentBoundaryAir<8> | 2 | 3 | 7 | 
| ecrecover_program | PhantomAir | 2 | 3 | 5 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| ecrecover_program | ProgramAir | 1 | 1 | 4 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| ecrecover_program | Rv32HintStoreAir | 2 | 18 | 28 | 
| ecrecover_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 20 | 37 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 18 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 24 | 91 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 35 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 18 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 2 | 25 | 225 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 40 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 84 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 19 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 12 | 14 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 415 | 480 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 2 | 158 | 190 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 2 | 428 | 457 | 
| ecrecover_program | VmConnectorAir | 2 | 5 | 11 | 
| leaf | AccessAdapterAir<2> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<4> | 2 | 5 | 12 | 
| leaf | AccessAdapterAir<8> | 2 | 5 | 12 | 
| leaf | FriReducedOpeningAir | 2 | 39 | 71 | 
| leaf | JalRangeCheckAir | 2 | 9 | 14 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 136 | 572 | 
| leaf | PhantomAir | 2 | 3 | 5 | 
| leaf | ProgramAir | 1 | 1 | 4 | 
| leaf | VariableRangeCheckerAir | 1 | 1 | 4 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 15 | 27 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 11 | 25 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 11 | 30 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 15 | 20 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 15 | 27 | 
| leaf | VmConnectorAir | 2 | 5 | 11 | 
| leaf | VolatileBoundaryAir | 2 | 7 | 19 | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 61,716,600 | 
| leaf | JalRangeCheckAir |  | 0 | JAL | 12 | 
| leaf | JalRangeCheckAir | Alloc | 0 | RANGE_CHECK | 331,620 | 
| leaf | JalRangeCheckAir | IfEqI | 0 | JAL | 47,508 | 
| leaf | JalRangeCheckAir | IfNe | 0 | JAL | 24 | 
| leaf | JalRangeCheckAir | ZipFor | 0 | JAL | 227,088 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,746 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 4,493,022 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchExt | 0 | VERIFY_BATCH | 9,074,400 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchFelt | 0 | VERIFY_BATCH | 64,396,400 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 165,600 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 253,200 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 22,800 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 5,472 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 72,636 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 1,686 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 21,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 29,928 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 128,180 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEI | 0 | ADD | 6,271,772 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddF | 0 | ADD | 925,680 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddFI | 0 | ADD | 2,765,353 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddV | 0 | ADD | 629,474 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddVI | 0 | ADD | 7,098,591 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,036,402 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | MUL | 283,214 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | CastFV | 0 | ADD | 26,477 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 11,136 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivF | 0 | DIV | 58,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 6,554 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmE | 0 | ADD | 130,152 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmF | 0 | ADD | 911,122 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmV | 0 | ADD | 1,338,089 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | ADD | 858,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | MUL | 858,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | ADD | 427,489 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | MUL | 24,360 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | ADD | 220,922 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | MUL | 203,087 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEF | 0 | MUL | 242,672 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 322,364 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEI | 0 | ADD | 1,119,632 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,061,226 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulFI | 0 | MUL | 820,874 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulV | 0 | MUL | 52,461 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulVI | 0 | MUL | 457,301 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | NegE | 0 | MUL | 3,596 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | ADD | 855,500 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | MUL | 855,500 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | ADD | 22,156 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | MUL | 21,692 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | ADD | 92,887 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | MUL | 51,272 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,843,356 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | SUB | 614,452 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 341,852 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEI | 0 | ADD | 22,272 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubFI | 0 | SUB | 819,888 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubV | 0 | SUB | 601,373 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVI | 0 | SUB | 6,641 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 55,100 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 25,491 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 8,447,120 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 12,328 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 92 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 671,416 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 35,420 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 20,309 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertNonZero | 0 | NativeBranchEqualOpcode(BEQ) | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 2,079,085 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 309,488 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 244,973 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 4,692 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 4,649,841 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | Publish | 0 | PUBLISH | 972 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadF | 0 | LOADW | 3,446,058 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadV | 0 | LOADW | 4,541,229 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreF | 0 | STOREW | 1,964,151 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreHintWord | 0 | HINT_STOREW | 2,816,037 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreV | 0 | STOREW | 389,193 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | LoadE | 0 | LOADW | 2,664,576 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | StoreE | 0 | STOREW | 1,110,051 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 3,337,084 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 876,584 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 3,648 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 3,575,306 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 366,776 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 745,218 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| ecrecover_program | KeccakVmAir |  | KECCAK256 | 0 | 379,560 | 
| ecrecover_program | PhantomAir |  | PHANTOM | 0 | 66 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 6,656 | 
| ecrecover_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 352 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | ADD | 0 | 1,132,056 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | AND | 0 | 264,744 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | OR | 0 | 201,312 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | SUB | 0 | 20,844 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | XOR | 0 | 35,820 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | SLTU | 0 | 106,930 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SLL | 0 | 296,588 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SRL | 0 | 271,943 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BEQ | 0 | 188,942 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BNE | 0 | 174,330 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BGEU | 0 | 3,936 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLT | 0 | 640 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLTU | 0 | 116,096 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | JAL | 0 | 38,700 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | LUI | 0 | 7,866 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | IS_EQ | 0 | 536,678 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | SETUP_ISEQ | 0 | 332 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | JALR | 0 | 71,400 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | LOADB | 0 | 196,884 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADBU | 0 | 211,314 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADW | 0 | 627,423 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREB | 0 | 356,946 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREW | 0 | 563,217 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | MULHU | 0 | 195 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | MUL | 0 | 961 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | AUIPC | 0 | 25,380 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> |  | EcDouble | 0 | 695,237 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> |  | ModularAddSub | 0 | 4,975 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 8,416 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> |  | EcAddNe | 0 | 453,750 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 32,768 |  | 16 | 17 | 1,081,344 | 
| leaf | FriReducedOpeningAir | 0 | 4,194,304 |  | 84 | 27 | 465,567,744 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 262,144 |  | 312 | 398 | 186,122,240 | 
| leaf | PhantomAir | 0 | 131,072 |  | 12 | 6 | 2,359,296 | 
| leaf | ProgramAir | 0 | 524,288 |  | 8 | 10 | 9,437,184 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 36 | 29 | 136,314,880 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 40 | 21 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 40 | 27 | 17,563,648 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 20 | 12 | 16,777,216 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | AccessAdapterAir<16> | 0 | 4,096 |  | 16 | 25 | 167,936 | 
| ecrecover_program | AccessAdapterAir<32> | 0 | 2,048 |  | 16 | 41 | 116,736 | 
| ecrecover_program | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| ecrecover_program | KeccakVmAir | 0 | 128 |  | 1,056 | 3,163 | 540,032 | 
| ecrecover_program | MemoryMerkleAir<8> | 0 | 4,096 |  | 16 | 32 | 196,608 | 
| ecrecover_program | PersistentBoundaryAir<8> | 0 | 4,096 |  | 12 | 20 | 131,072 | 
| ecrecover_program | PhantomAir | 0 | 16 |  | 12 | 6 | 288 | 
| ecrecover_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 4,096 |  | 8 | 300 | 1,261,568 | 
| ecrecover_program | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 65,536 |  | 52 | 36 | 5,767,168 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 4,096 |  | 40 | 37 | 315,392 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 16,384 |  | 52 | 53 | 1,720,320 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 16,384 |  | 28 | 26 | 884,736 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4,096 |  | 32 | 32 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 4,096 |  | 28 | 18 | 188,416 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 4,096 |  | 56 | 166 | 909,312 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 4,096 |  | 36 | 28 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 8,192 |  | 52 | 36 | 720,896 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 65,536 |  | 52 | 41 | 6,094,848 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 8 |  | 72 | 39 | 888 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32 |  | 52 | 31 | 2,656 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 2,048 |  | 28 | 20 | 98,304 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 2,048 |  | 836 | 547 | 2,832,384 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 32 |  | 320 | 263 | 18,656 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 860 | 625 | 1,520,640 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | AccessAdapter<2> | 0 | 1,048,576 | 
| leaf | AccessAdapter<4> | 0 | 524,288 | 
| leaf | AccessAdapter<8> | 0 | 32,768 | 
| leaf | Boundary | 0 | 523,478 | 
| leaf | FriReducedOpeningAir | 0 | 2,285,800 | 
| leaf | JalRangeCheckAir | 0 | 50,521 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 195,916 | 
| leaf | PhantomAir | 0 | 97,223 | 
| leaf | ProgramChip | 0 | 439,367 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,482,744 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 349,029 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 36 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 626,508 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 139,801 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 234,332 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| ecrecover_program | AccessAdapter<16> | 0 | 4,096 | 
| ecrecover_program | AccessAdapter<32> | 0 | 2,048 | 
| ecrecover_program | AccessAdapter<8> | 0 | 8,192 | 
| ecrecover_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| ecrecover_program | Boundary | 0 | 3,382 | 
| ecrecover_program | KeccakVmAir | 0 | 120 | 
| ecrecover_program | Merkle | 0 | 3,678 | 
| ecrecover_program | PhantomAir | 0 | 11 | 
| ecrecover_program | Poseidon2PeripheryAir<F, 1> | 0 | 2,274 | 
| ecrecover_program | ProgramChip | 0 | 19,863 | 
| ecrecover_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| ecrecover_program | Rv32HintStoreAir | 0 | 219 | 
| ecrecover_program | VariableRangeCheckerAir | 0 | 262,144 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 45,966 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 2,890 | 
| ecrecover_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 10,727 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 13,972 | 
| ecrecover_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 3,771 | 
| ecrecover_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 2,587 | 
| ecrecover_program | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 3,204 | 
| ecrecover_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 2,550 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 5,469 | 
| ecrecover_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 42,900 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 5 | 
| ecrecover_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 31 | 
| ecrecover_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 1,270 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<1, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 1,271 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 21 | 
| ecrecover_program | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 726 | 
| ecrecover_program | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 87,818 | 
| leaf | AddEFFI | 0 | ADD | 1,032 | 
| leaf | AddEFI | 0 | ADD | 4,420 | 
| leaf | AddEI | 0 | ADD | 216,268 | 
| leaf | AddF | 0 | ADD | 31,920 | 
| leaf | AddFI | 0 | ADD | 95,357 | 
| leaf | AddV | 0 | ADD | 21,706 | 
| leaf | AddVI | 0 | ADD | 244,779 | 
| leaf | Alloc | 0 | ADD | 35,738 | 
| leaf | Alloc | 0 | MUL | 9,766 | 
| leaf | Alloc | 0 | RANGE_CHECK | 27,635 | 
| leaf | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 536 | 
| leaf | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 4 | 
| leaf | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 29,192 | 
| leaf | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 1,540 | 
| leaf | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 883 | 
| leaf | AssertNonZero | 0 | NativeBranchEqualOpcode(BEQ) | 1 | 
| leaf | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 1,600 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 200 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 1,600 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 27,600 | 
| leaf | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 42,200 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 3,800 | 
| leaf | CT-verify-query | 0 | PHANTOM | 200 | 
| leaf | CastFV | 0 | ADD | 913 | 
| leaf | DivE | 0 | BBE4DIV | 23,068 | 
| leaf | DivEIN | 0 | ADD | 384 | 
| leaf | DivEIN | 0 | BBE4DIV | 96 | 
| leaf | DivF | 0 | DIV | 2,000 | 
| leaf | DivFIN | 0 | DIV | 226 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 21,100 | 
| leaf | HintBitsF | 0 | PHANTOM | 912 | 
| leaf | HintFelt | 0 | PHANTOM | 12,106 | 
| leaf | HintInputVec | 0 | PHANTOM | 281 | 
| leaf | HintLoad | 0 | PHANTOM | 3,500 | 
| leaf | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 90,395 | 
| leaf | IfEqI | 0 | JAL | 3,959 | 
| leaf | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 13,456 | 
| leaf | IfNe | 0 | JAL | 2 | 
| leaf | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 10,651 | 
| leaf | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 204 | 
| leaf | ImmE | 0 | ADD | 4,488 | 
| leaf | ImmF | 0 | ADD | 31,418 | 
| leaf | ImmV | 0 | ADD | 46,141 | 
| leaf | LoadE | 0 | ADD | 29,600 | 
| leaf | LoadE | 0 | LOADW | 98,688 | 
| leaf | LoadE | 0 | MUL | 29,600 | 
| leaf | LoadF | 0 | ADD | 14,741 | 
| leaf | LoadF | 0 | LOADW | 164,098 | 
| leaf | LoadF | 0 | MUL | 840 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 7,618 | 
| leaf | LoadV | 0 | LOADW | 216,249 | 
| leaf | LoadV | 0 | MUL | 7,003 | 
| leaf | MulE | 0 | BBE4MUL | 94,087 | 
| leaf | MulEF | 0 | MUL | 8,368 | 
| leaf | MulEFI | 0 | MUL | 11,116 | 
| leaf | MulEI | 0 | ADD | 38,608 | 
| leaf | MulEI | 0 | BBE4MUL | 9,652 | 
| leaf | MulF | 0 | MUL | 36,594 | 
| leaf | MulFI | 0 | MUL | 28,306 | 
| leaf | MulV | 0 | MUL | 1,809 | 
| leaf | MulVI | 0 | MUL | 15,769 | 
| leaf | NegE | 0 | MUL | 124 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 11,289 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 29,500 | 
| leaf | StoreE | 0 | MUL | 29,500 | 
| leaf | StoreE | 0 | STOREW | 41,113 | 
| leaf | StoreF | 0 | ADD | 764 | 
| leaf | StoreF | 0 | MUL | 748 | 
| leaf | StoreF | 0 | STOREW | 93,531 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 134,097 | 
| leaf | StoreV | 0 | ADD | 3,203 | 
| leaf | StoreV | 0 | MUL | 1,768 | 
| leaf | StoreV | 0 | STOREW | 18,533 | 
| leaf | SubE | 0 | FE4SUB | 19,611 | 
| leaf | SubEF | 0 | ADD | 63,564 | 
| leaf | SubEF | 0 | SUB | 21,188 | 
| leaf | SubEFI | 0 | ADD | 11,788 | 
| leaf | SubEI | 0 | ADD | 768 | 
| leaf | SubFI | 0 | SUB | 28,272 | 
| leaf | SubV | 0 | SUB | 20,737 | 
| leaf | SubVI | 0 | SUB | 229 | 
| leaf | SubVIN | 0 | SUB | 1,900 | 
| leaf | UnsafeCastVF | 0 | ADD | 879 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 1,900 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 800 | 
| leaf | ZipFor | 0 | ADD | 291,280 | 
| leaf | ZipFor | 0 | JAL | 18,924 | 
| leaf | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 202,167 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| ecrecover_program |  | ADD | 0 | 31,446 | 
| ecrecover_program |  | AND | 0 | 7,354 | 
| ecrecover_program |  | AUIPC | 0 | 1,270 | 
| ecrecover_program |  | BEQ | 0 | 7,267 | 
| ecrecover_program |  | BGEU | 0 | 123 | 
| ecrecover_program |  | BLT | 0 | 20 | 
| ecrecover_program |  | BLTU | 0 | 3,628 | 
| ecrecover_program |  | BNE | 0 | 6,705 | 
| ecrecover_program |  | EcAddNe | 0 | 726 | 
| ecrecover_program |  | EcDouble | 0 | 1,271 | 
| ecrecover_program |  | HINT_BUFFER | 0 | 11 | 
| ecrecover_program |  | HINT_STOREW | 0 | 11 | 
| ecrecover_program |  | IS_EQ | 0 | 3,233 | 
| ecrecover_program |  | JAL | 0 | 2,150 | 
| ecrecover_program |  | JALR | 0 | 2,550 | 
| ecrecover_program |  | KECCAK256 | 0 | 5 | 
| ecrecover_program |  | LOADB | 0 | 5,469 | 
| ecrecover_program |  | LOADBU | 0 | 5,154 | 
| ecrecover_program |  | LOADW | 0 | 15,303 | 
| ecrecover_program |  | LUI | 0 | 437 | 
| ecrecover_program |  | MUL | 0 | 31 | 
| ecrecover_program |  | MULHU | 0 | 5 | 
| ecrecover_program |  | ModularAddSub | 0 | 25 | 
| ecrecover_program |  | ModularMulDiv | 0 | 32 | 
| ecrecover_program |  | OR | 0 | 5,592 | 
| ecrecover_program |  | PHANTOM | 0 | 11 | 
| ecrecover_program |  | SETUP_ISEQ | 0 | 2 | 
| ecrecover_program |  | SLL | 0 | 5,596 | 
| ecrecover_program |  | SLTU | 0 | 2,890 | 
| ecrecover_program |  | SRL | 0 | 5,131 | 
| ecrecover_program |  | STOREB | 0 | 8,706 | 
| ecrecover_program |  | STOREW | 0 | 13,737 | 
| ecrecover_program |  | SUB | 0 | 579 | 
| ecrecover_program |  | XOR | 0 | 995 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 585 | 25,473 | 3,015,310 | 993,824,234 | 8,079 | 886 | 632 | 2,062 | 2,191 | 19 | 1,501 | 242,933,756 | 3,015,311 | 800 | 5,918 | 0.51 | 10,891 | 0.28 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 18,219,140 | 2,013,265,921 | 
| leaf | 0 | 1 | 122,388,736 | 2,013,265,921 | 
| leaf | 0 | 2 | 9,109,570 | 2,013,265,921 | 
| leaf | 0 | 3 | 122,487,044 | 2,013,265,921 | 
| leaf | 0 | 4 | 524,288 | 2,013,265,921 | 
| leaf | 0 | 5 | 273,515,210 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 346 | 1,872 | 137,465 | 32,925,330 | 967 | 78 | 129 | 2,092 | 157 | 421 | 70 | 25 | 77 | 132 | 8,193,677 | 137,466 | 31 | 559 | 0.25 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| ecrecover_program | 0 | 0 | 396,372 | 2,013,265,921 | 
| ecrecover_program | 0 | 1 | 1,239,280 | 2,013,265,921 | 
| ecrecover_program | 0 | 2 | 198,186 | 2,013,265,921 | 
| ecrecover_program | 0 | 3 | 2,663,748 | 2,013,265,921 | 
| ecrecover_program | 0 | 4 | 16,384 | 2,013,265,921 | 
| ecrecover_program | 0 | 5 | 8,192 | 2,013,265,921 | 
| ecrecover_program | 0 | 6 | 471,272 | 2,013,265,921 | 
| ecrecover_program | 0 | 7 | 192 | 2,013,265,921 | 
| ecrecover_program | 0 | 8 | 5,947,994 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-ecrecover_program.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ecrecover-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/ecrecover-leaf.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/85e28f86705cbf0eaca3e424bad3af07da9f5ad6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16180787988)
