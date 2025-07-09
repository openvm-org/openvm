| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  29.100 |  29.100 |
| pairing |  10.50 |  10.50 |
| leaf |  15.76 |  15.76 |


| pairing |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,500 |  10,500 |  10,500 |  10,500 |
| `main_cells_used     ` | <span style='color: red'>(+5318230 [+5.5%])</span> 102,596,013 | <span style='color: red'>(+5318230 [+5.5%])</span> 102,596,013 | <span style='color: red'>(+5318230 [+5.5%])</span> 102,596,013 | <span style='color: red'>(+5318230 [+5.5%])</span> 102,596,013 |
| `total_cycles        ` |  1,862,964 |  1,862,964 |  1,862,964 |  1,862,964 |
| `execute_metered_time_ms` |  3,741 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.50 | -          |  0.50 |  0.50 |
| `execute_e3_time_ms  ` |  6,681 |  6,681 |  6,681 |  6,681 |
| `execute_e3_insn_mi/s` |  0.28 | -          |  0.28 |  0.28 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-318 [-33.8%])</span> 622 | <span style='color: green'>(-318 [-33.8%])</span> 622 | <span style='color: green'>(-318 [-33.8%])</span> 622 | <span style='color: green'>(-318 [-33.8%])</span> 622 |
| `memory_finalize_time_ms` |  133 |  133 |  133 |  133 |
| `boundary_finalize_time_ms` |  2 |  2 |  2 |  2 |
| `merkle_finalize_time_ms` |  120 |  120 |  120 |  120 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: red'>(+335 [+11.7%])</span> 3,197 | <span style='color: red'>(+335 [+11.7%])</span> 3,197 | <span style='color: red'>(+335 [+11.7%])</span> 3,197 | <span style='color: red'>(+335 [+11.7%])</span> 3,197 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+73 [+12.5%])</span> 656 | <span style='color: red'>(+73 [+12.5%])</span> 656 | <span style='color: red'>(+73 [+12.5%])</span> 656 | <span style='color: red'>(+73 [+12.5%])</span> 656 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+55 [+24.6%])</span> 279 | <span style='color: red'>(+55 [+24.6%])</span> 279 | <span style='color: red'>(+55 [+24.6%])</span> 279 | <span style='color: red'>(+55 [+24.6%])</span> 279 |
| `perm_trace_commit_time_ms` | <span style='color: red'>(+108 [+19.7%])</span> 656 | <span style='color: red'>(+108 [+19.7%])</span> 656 | <span style='color: red'>(+108 [+19.7%])</span> 656 | <span style='color: red'>(+108 [+19.7%])</span> 656 |
| `quotient_poly_compute_time_ms` |  342 |  342 |  342 |  342 |
| `quotient_poly_commit_time_ms` | <span style='color: red'>(+18 [+6.5%])</span> 293 | <span style='color: red'>(+18 [+6.5%])</span> 293 | <span style='color: red'>(+18 [+6.5%])</span> 293 | <span style='color: red'>(+18 [+6.5%])</span> 293 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+84 [+9.6%])</span> 962 | <span style='color: red'>(+84 [+9.6%])</span> 962 | <span style='color: red'>(+84 [+9.6%])</span> 962 | <span style='color: red'>(+84 [+9.6%])</span> 962 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,758 |  15,758 |  15,758 |  15,758 |
| `main_cells_used     ` | <span style='color: green'>(-70285431 [-34.2%])</span> 135,239,963 | <span style='color: green'>(-70285431 [-34.2%])</span> 135,239,963 | <span style='color: green'>(-70285431 [-34.2%])</span> 135,239,963 | <span style='color: green'>(-70285431 [-34.2%])</span> 135,239,963 |
| `total_cycles        ` | <span style='color: green'>(-493426 [-19.2%])</span> 2,081,108 | <span style='color: green'>(-493426 [-19.2%])</span> 2,081,108 | <span style='color: green'>(-493426 [-19.2%])</span> 2,081,108 | <span style='color: green'>(-493426 [-19.2%])</span> 2,081,108 |
| `execute_metered_time_ms` |  4,191 |  4,191 |  4,191 |  4,191 |
| `execute_metered_insn_mi/s` |  0.50 | -          |  0.50 |  0.50 |
| `execute_e3_time_ms  ` |  7,548 |  7,548 |  7,548 |  7,548 |
| `execute_e3_insn_mi/s` |  0.28 | -          |  0.28 |  0.28 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1568 [-81.3%])</span> 361 | <span style='color: green'>(-1568 [-81.3%])</span> 361 | <span style='color: green'>(-1568 [-81.3%])</span> 361 | <span style='color: green'>(-1568 [-81.3%])</span> 361 |
| `memory_finalize_time_ms` |  12 |  12 |  12 |  12 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-2115 [-36.6%])</span> 3,658 | <span style='color: green'>(-2115 [-36.6%])</span> 3,658 | <span style='color: green'>(-2115 [-36.6%])</span> 3,658 | <span style='color: green'>(-2115 [-36.6%])</span> 3,658 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-413 [-37.3%])</span> 695 | <span style='color: green'>(-413 [-37.3%])</span> 695 | <span style='color: green'>(-413 [-37.3%])</span> 695 | <span style='color: green'>(-413 [-37.3%])</span> 695 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-237 [-42.7%])</span> 318 | <span style='color: green'>(-237 [-42.7%])</span> 318 | <span style='color: green'>(-237 [-42.7%])</span> 318 | <span style='color: green'>(-237 [-42.7%])</span> 318 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-548 [-37.9%])</span> 897 | <span style='color: green'>(-548 [-37.9%])</span> 897 | <span style='color: green'>(-548 [-37.9%])</span> 897 | <span style='color: green'>(-548 [-37.9%])</span> 897 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-253 [-36.8%])</span> 434 | <span style='color: green'>(-253 [-36.8%])</span> 434 | <span style='color: green'>(-253 [-36.8%])</span> 434 | <span style='color: green'>(-253 [-36.8%])</span> 434 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-126 [-27.6%])</span> 330 | <span style='color: green'>(-126 [-27.6%])</span> 330 | <span style='color: green'>(-126 [-27.6%])</span> 330 | <span style='color: green'>(-126 [-27.6%])</span> 330 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-539 [-35.5%])</span> 978 | <span style='color: green'>(-539 [-35.5%])</span> 978 | <span style='color: green'>(-539 [-35.5%])</span> 978 | <span style='color: green'>(-539 [-35.5%])</span> 978 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 48 | 11 | 18,586 | 15,925 | 

| group | single_leaf_agg_time_ms | num_segments | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 15,924 |  | 1 |  |  | 1 |  |  |  | 
| pairing |  | 1 |  | 23 | 1,862,965 | 1 | 14,391 | 3,741 | 0.50 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
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

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 26,060,400 | 
| leaf | JalRangeCheckAir |  | 0 | JAL | 12 | 
| leaf | JalRangeCheckAir | Alloc | 0 | RANGE_CHECK | 336,384 | 
| leaf | JalRangeCheckAir | IfEqI | 0 | JAL | 50,220 | 
| leaf | JalRangeCheckAir | IfNe | 0 | JAL | 24 | 
| leaf | JalRangeCheckAir | ZipFor | 0 | JAL | 201,600 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 10,746 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 1,875,774 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchExt | 0 | VERIFY_BATCH | 9,950,000 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchFelt | 0 | VERIFY_BATCH | 31,840,000 | 
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
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 141,600 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 217,200 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 4,752 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 75,354 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 1,446 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 21,600 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 29,232 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 68,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,778,316 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddF | 0 | ADD | 803,880 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddFI | 0 | ADD | 1,241,809 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddV | 0 | ADD | 570,053 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddVI | 0 | ADD | 3,587,851 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,050,960 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | MUL | 287,448 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | CastFV | 0 | ADD | 22,997 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 9,396 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivF | 0 | DIV | 60,900 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 5,539 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmE | 0 | ADD | 113,564 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmF | 0 | ADD | 792,512 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmV | 0 | ADD | 1,194,481 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | ADD | 745,300 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | MUL | 745,300 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | ADD | 366,009 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | MUL | 20,880 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | ADD | 189,428 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | MUL | 173,797 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEF | 0 | MUL | 250,792 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 420,036 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEI | 0 | ADD | 552,276 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulF | 0 | MUL | 963,496 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulFI | 0 | MUL | 712,849 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulV | 0 | MUL | 43,094 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulVI | 0 | MUL | 396,256 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | NegE | 0 | MUL | 3,016 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | ADD | 742,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | MUL | 742,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | ADD | 19,836 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | MUL | 19,372 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | ADD | 81,287 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | MUL | 43,877 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,581,486 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | SUB | 527,162 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 47,212 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEI | 0 | ADD | 18,792 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubFI | 0 | SUB | 712,008 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubV | 0 | SUB | 540,966 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVI | 0 | SUB | 6,612 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 58,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 21,721 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 6,009,989 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 11,868 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 92 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 583,096 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 33,695 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 17,434 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertNonZero | 0 | NativeBranchEqualOpcode(BEQ) | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 869,239 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 300,840 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 209,461 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 4,002 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 2,947,795 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | Publish | 0 | PUBLISH | 972 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadF | 0 | LOADW | 2,135,406 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadV | 0 | LOADW | 4,030,425 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreF | 0 | STOREW | 859,845 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreHintWord | 0 | HINT_STOREW | 1,643,901 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreV | 0 | STOREW | 387,093 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | LoadE | 0 | LOADW | 2,083,995 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | StoreE | 0 | STOREW | 1,011,960 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,744,846 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 766,004 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 3,078 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 2,419,308 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 180,918 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 334,248 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| pairing | PhantomAir |  | PHANTOM | 0 | 6 | 
| pairing | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 6,144 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | ADD | 0 | 17,142,228 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | AND | 0 | 4,373,064 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | OR | 0 | 723,132 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | SUB | 0 | 68,544 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | SLTU | 0 | 1,448,957 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SLL | 0 | 79,977 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SRL | 0 | 4,240 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BEQ | 0 | 1,504,984 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BNE | 0 | 2,074,072 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BGEU | 0 | 71,648 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLT | 0 | 6,016 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLTU | 0 | 3,805,728 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | JAL | 0 | 18,054 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | LUI | 0 | 76,878 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | IS_EQ | 0 | 2,822 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> |  | SETUP_ISEQ | 0 | 166 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | JALR | 0 | 1,181,796 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADBU | 0 | 62,320 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADW | 0 | 17,768,990 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREB | 0 | 115,292 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREW | 0 | 17,263,788 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | MULHU | 0 | 6,084 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | MUL | 0 | 12,772 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | AUIPC | 0 | 422,100 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> |  | ModularAddSub | 0 | 7,363 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> |  | ModularMulDiv | 0 | 189,097 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> |  | Fp2AddSub | 0 | 2,387,061 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> |  | Fp2MulDiv | 0 | 4,161,878 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 1,048,576 |  | 84 | 27 | 116,391,936 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | PhantomAir | 0 | 131,072 |  | 12 | 6 | 2,359,296 | 
| leaf | ProgramAir | 0 | 1,048,576 |  | 8 | 10 | 18,874,368 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 20 | 12 | 8,388,608 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | AccessAdapterAir<16> | 0 | 262,144 |  | 16 | 25 | 10,747,904 | 
| pairing | AccessAdapterAir<32> | 0 | 131,072 |  | 16 | 41 | 7,471,104 | 
| pairing | AccessAdapterAir<8> | 0 | 524,288 |  | 16 | 17 | 17,301,504 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| pairing | MemoryMerkleAir<8> | 0 | 32,768 |  | 16 | 32 | 1,572,864 | 
| pairing | PersistentBoundaryAir<8> | 0 | 32,768 |  | 12 | 20 | 1,048,576 | 
| pairing | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| pairing | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 32,768 |  | 8 | 300 | 10,092,544 | 
| pairing | ProgramAir | 0 | 32,768 |  | 8 | 10 | 589,824 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| pairing | Rv32HintStoreAir | 0 | 256 |  | 44 | 32 | 19,456 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 65,536 |  | 40 | 37 | 5,046,272 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2,048 |  | 52 | 53 | 215,040 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 8,192 |  | 28 | 18 | 376,832 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 32 |  | 56 | 166 | 7,104 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 65,536 |  | 36 | 28 | 4,194,304 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 512 |  | 52 | 31 | 42,496 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 1,024 |  | 320 | 263 | 596,992 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 16,384 |  | 604 | 497 | 18,038,784 | 
| pairing | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | AccessAdapter<2> | 0 | 524,288 | 
| leaf | AccessAdapter<4> | 0 | 262,144 | 
| leaf | AccessAdapter<8> | 0 | 16,384 | 
| leaf | Boundary | 0 | 248,471 | 
| leaf | FriReducedOpeningAir | 0 | 965,200 | 
| leaf | JalRangeCheckAir | 0 | 49,020 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 109,740 | 
| leaf | PhantomAir | 0 | 87,816 | 
| leaf | ProgramChip | 0 | 532,456 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,012,867 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 216,415 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 36 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 431,270 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 114,665 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 143,379 | 
| leaf | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| pairing | AccessAdapter<16> | 0 | 262,144 | 
| pairing | AccessAdapter<32> | 0 | 131,072 | 
| pairing | AccessAdapter<8> | 0 | 524,288 | 
| pairing | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| pairing | Boundary | 0 | 21,534 | 
| pairing | Merkle | 0 | 23,102 | 
| pairing | PhantomAir | 0 | 1 | 
| pairing | Poseidon2PeripheryAir<F, 1> | 0 | 18,668 | 
| pairing | ProgramChip | 0 | 22,493 | 
| pairing | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| pairing | Rv32HintStoreAir | 0 | 192 | 
| pairing | VariableRangeCheckerAir | 0 | 262,144 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 619,638 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 39,161 | 
| pairing | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 1,589 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 137,656 | 
| pairing | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 121,356 | 
| pairing | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 5,274 | 
| pairing | VmAirWrapper<Rv32IsEqualModAdapterAir<2, 1, 32, 32>, ModularIsEqualCoreAir<32, 4, 8> | 0 | 18 | 
| pairing | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 42,207 | 
| pairing | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 858,790 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 156 | 
| pairing | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 412 | 
| pairing | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 21,106 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 1, 1, 32, 32>, FieldExpressionCoreAir> | 0 | 719 | 
| pairing | VmAirWrapper<Rv32VecHeapAdapterAir<2, 2, 2, 32, 32>, FieldExpressionCoreAir> | 0 | 8,374 | 
| pairing | VmConnectorAir | 0 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 45,917 | 
| leaf | AddEFFI | 0 | ADD | 1,008 | 
| leaf | AddEFI | 0 | ADD | 2,360 | 
| leaf | AddEI | 0 | ADD | 95,804 | 
| leaf | AddF | 0 | ADD | 27,720 | 
| leaf | AddFI | 0 | ADD | 42,821 | 
| leaf | AddV | 0 | ADD | 19,657 | 
| leaf | AddVI | 0 | ADD | 123,719 | 
| leaf | Alloc | 0 | ADD | 36,240 | 
| leaf | Alloc | 0 | MUL | 9,912 | 
| leaf | Alloc | 0 | RANGE_CHECK | 28,032 | 
| leaf | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 516 | 
| leaf | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 4 | 
| leaf | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 25,352 | 
| leaf | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 1,465 | 
| leaf | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 758 | 
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
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 23,600 | 
| leaf | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 36,200 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 4,000 | 
| leaf | CT-verify-query | 0 | PHANTOM | 200 | 
| leaf | CastFV | 0 | ADD | 793 | 
| leaf | DivE | 0 | BBE4DIV | 20,158 | 
| leaf | DivEIN | 0 | ADD | 324 | 
| leaf | DivEIN | 0 | BBE4DIV | 81 | 
| leaf | DivF | 0 | DIV | 2,100 | 
| leaf | DivFIN | 0 | DIV | 191 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 18,100 | 
| leaf | HintBitsF | 0 | PHANTOM | 792 | 
| leaf | HintFelt | 0 | PHANTOM | 12,559 | 
| leaf | HintInputVec | 0 | PHANTOM | 241 | 
| leaf | HintLoad | 0 | PHANTOM | 3,600 | 
| leaf | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 37,793 | 
| leaf | IfEqI | 0 | JAL | 4,185 | 
| leaf | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 13,080 | 
| leaf | IfNe | 0 | JAL | 2 | 
| leaf | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 9,107 | 
| leaf | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 174 | 
| leaf | ImmE | 0 | ADD | 3,916 | 
| leaf | ImmF | 0 | ADD | 27,328 | 
| leaf | ImmV | 0 | ADD | 41,189 | 
| leaf | LoadE | 0 | ADD | 25,700 | 
| leaf | LoadE | 0 | LOADW | 77,185 | 
| leaf | LoadE | 0 | MUL | 25,700 | 
| leaf | LoadF | 0 | ADD | 12,621 | 
| leaf | LoadF | 0 | LOADW | 101,686 | 
| leaf | LoadF | 0 | MUL | 720 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 6,532 | 
| leaf | LoadV | 0 | LOADW | 191,925 | 
| leaf | LoadV | 0 | MUL | 5,993 | 
| leaf | MulE | 0 | BBE4MUL | 63,666 | 
| leaf | MulEF | 0 | MUL | 8,648 | 
| leaf | MulEFI | 0 | MUL | 14,484 | 
| leaf | MulEI | 0 | ADD | 19,044 | 
| leaf | MulEI | 0 | BBE4MUL | 4,761 | 
| leaf | MulF | 0 | MUL | 33,224 | 
| leaf | MulFI | 0 | MUL | 24,581 | 
| leaf | MulV | 0 | MUL | 1,486 | 
| leaf | MulVI | 0 | MUL | 13,664 | 
| leaf | NegE | 0 | MUL | 104 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 4,713 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 25,600 | 
| leaf | StoreE | 0 | MUL | 25,600 | 
| leaf | StoreE | 0 | STOREW | 37,480 | 
| leaf | StoreF | 0 | ADD | 684 | 
| leaf | StoreF | 0 | MUL | 668 | 
| leaf | StoreF | 0 | STOREW | 40,945 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 78,281 | 
| leaf | StoreV | 0 | ADD | 2,803 | 
| leaf | StoreV | 0 | MUL | 1,513 | 
| leaf | StoreV | 0 | STOREW | 18,433 | 
| leaf | SubE | 0 | FE4SUB | 8,796 | 
| leaf | SubEF | 0 | ADD | 54,534 | 
| leaf | SubEF | 0 | SUB | 18,178 | 
| leaf | SubEFI | 0 | ADD | 1,628 | 
| leaf | SubEI | 0 | ADD | 648 | 
| leaf | SubFI | 0 | SUB | 24,552 | 
| leaf | SubV | 0 | SUB | 18,654 | 
| leaf | SubVI | 0 | SUB | 228 | 
| leaf | SubVIN | 0 | SUB | 2,000 | 
| leaf | UnsafeCastVF | 0 | ADD | 749 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 800 | 
| leaf | ZipFor | 0 | ADD | 207,241 | 
| leaf | ZipFor | 0 | JAL | 16,800 | 
| leaf | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 128,165 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| pairing |  | ADD | 0 | 476,173 | 
| pairing |  | AND | 0 | 121,474 | 
| pairing |  | AUIPC | 0 | 21,106 | 
| pairing |  | BEQ | 0 | 57,884 | 
| pairing |  | BGEU | 0 | 2,239 | 
| pairing |  | BLT | 0 | 188 | 
| pairing |  | BLTU | 0 | 118,929 | 
| pairing |  | BNE | 0 | 79,772 | 
| pairing |  | Fp2AddSub | 0 | 6,469 | 
| pairing |  | Fp2MulDiv | 0 | 8,374 | 
| pairing |  | HINT_BUFFER | 0 | 1 | 
| pairing |  | IS_EQ | 0 | 17 | 
| pairing |  | JAL | 0 | 1,003 | 
| pairing |  | JALR | 0 | 42,207 | 
| pairing |  | LOADBU | 0 | 1,520 | 
| pairing |  | LOADW | 0 | 433,390 | 
| pairing |  | LUI | 0 | 4,271 | 
| pairing |  | MUL | 0 | 412 | 
| pairing |  | MULHU | 0 | 156 | 
| pairing |  | ModularAddSub | 0 | 37 | 
| pairing |  | ModularMulDiv | 0 | 719 | 
| pairing |  | OR | 0 | 20,087 | 
| pairing |  | PHANTOM | 0 | 1 | 
| pairing |  | SETUP_ISEQ | 0 | 1 | 
| pairing |  | SLL | 0 | 1,509 | 
| pairing |  | SLTU | 0 | 39,161 | 
| pairing |  | SRL | 0 | 80 | 
| pairing |  | STOREB | 0 | 2,812 | 
| pairing |  | STOREW | 0 | 421,068 | 
| pairing |  | SUB | 0 | 1,904 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 361 | 15,758 | 2,081,108 | 429,805,034 | 3,658 | 434 | 330 | 897 | 978 | 12 | 695 | 135,239,963 | 2,081,109 | 318 | 4,191 | 0.50 | 7,548 | 0.28 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 7,471,236 | 2,013,265,921 | 
| leaf | 0 | 1 | 45,531,392 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,735,618 | 2,013,265,921 | 
| leaf | 0 | 3 | 44,859,652 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 103,170,762 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pairing | 0 | 622 | 10,500 | 1,862,964 | 304,931,516 | 3,197 | 342 | 293 | 3,701 | 656 | 962 | 120 | 24 | 133 | 656 | 102,596,013 | 1,862,965 | 279 | 6,681 | 0.28 | 2 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| pairing | 0 | 0 | 5,382,342 | 2,013,265,921 | 
| pairing | 0 | 1 | 18,152,512 | 2,013,265,921 | 
| pairing | 0 | 2 | 2,691,171 | 2,013,265,921 | 
| pairing | 0 | 3 | 25,000,068 | 2,013,265,921 | 
| pairing | 0 | 4 | 131,072 | 2,013,265,921 | 
| pairing | 0 | 5 | 65,536 | 2,013,265,921 | 
| pairing | 0 | 6 | 6,016,192 | 2,013,265,921 | 
| pairing | 0 | 7 | 4,096 | 2,013,265,921 | 
| pairing | 0 | 8 | 58,426,029 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/pairing-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/pairing-pairing.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/85e28f86705cbf0eaca3e424bad3af07da9f5ad6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16180787988)
