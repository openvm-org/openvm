| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  55.66 |  34.36 |
| regex_program |  20.62 |  11.03 |
| leaf |  27.67 |  15.95 |


| regex_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,310 |  20,620 |  11,034 |  9,586 |
| `main_cells_used     ` | <span style='color: red'>(+667391 [+0.8%])</span> 83,922,967 | <span style='color: red'>(+1334782 [+0.8%])</span> 167,845,934 | <span style='color: green'>(-269546 [-0.3%])</span> 93,231,253 | <span style='color: red'>(+1604328 [+2.2%])</span> 74,614,681 |
| `total_cycles        ` | <span style='color: red'>(+20664 [+1.0%])</span> 2,103,276.50 | <span style='color: red'>(+41327 [+1.0%])</span> 4,206,553 | <span style='color: green'>(-3215 [-0.1%])</span> 2,240,500 | <span style='color: red'>(+44542 [+2.3%])</span> 1,966,053 |
| `execute_metered_time_ms` |  7,372 | -          | -          | -          |
| `execute_metered_insn_mi/s` |  0.57 | -          |  0.57 |  0.57 |
| `execute_e3_time_ms  ` |  6,978.50 |  13,957 |  7,465 |  6,492 |
| `execute_e3_insn_mi/s` |  0.30 | -          |  0.30 |  0.30 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-228 [-23.5%])</span> 742 | <span style='color: green'>(-456 [-23.5%])</span> 1,484 | <span style='color: green'>(-407 [-34.9%])</span> 759 | <span style='color: green'>(-49 [-6.3%])</span> 725 |
| `memory_finalize_time_ms` |  163 |  326 |  254 |  72 |
| `boundary_finalize_time_ms` |  2.50 |  5 |  5 |  0 |
| `merkle_finalize_time_ms` |  149 |  298 |  230 |  68 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-50 [-1.9%])</span> 2,589.50 | <span style='color: green'>(-99 [-1.9%])</span> 5,179 | <span style='color: green'>(-64 [-2.2%])</span> 2,810 | <span style='color: green'>(-35 [-1.5%])</span> 2,369 |
| `main_trace_commit_time_ms` | <span style='color: red'>(+8 [+1.5%])</span> 507.50 | <span style='color: red'>(+15 [+1.5%])</span> 1,015 | <span style='color: red'>(+17 [+3.1%])</span> 573 | <span style='color: green'>(-2 [-0.5%])</span> 442 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-26 [-11.4%])</span> 203 | <span style='color: green'>(-52 [-11.4%])</span> 406 | <span style='color: green'>(-25 [-10.1%])</span> 223 | <span style='color: green'>(-27 [-12.9%])</span> 183 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-13 [-2.3%])</span> 564.50 | <span style='color: green'>(-26 [-2.3%])</span> 1,129 | <span style='color: green'>(-22 [-3.5%])</span> 613 | <span style='color: green'>(-4 [-0.8%])</span> 516 |
| `quotient_poly_compute_time_ms` | <span style='color: red'>(+5 [+1.9%])</span> 271.50 | <span style='color: red'>(+10 [+1.9%])</span> 543 | <span style='color: red'>(+3 [+1.0%])</span> 292 | <span style='color: red'>(+7 [+2.9%])</span> 251 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-6 [-2.7%])</span> 238.50 | <span style='color: green'>(-13 [-2.7%])</span> 477 | <span style='color: green'>(-11 [-3.9%])</span> 272 | <span style='color: green'>(-2 [-1.0%])</span> 205 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-18 [-2.2%])</span> 795 | <span style='color: green'>(-36 [-2.2%])</span> 1,590 | <span style='color: green'>(-24 [-2.8%])</span> 832 | <span style='color: green'>(-12 [-1.6%])</span> 758 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  13,836.50 |  27,673 |  15,951 |  11,722 |
| `main_cells_used     ` | <span style='color: green'>(-36491543 [-24.0%])</span> 115,336,518 | <span style='color: green'>(-72983086 [-24.0%])</span> 230,673,036 | <span style='color: green'>(-10666868 [-6.9%])</span> 143,561,106 | <span style='color: green'>(-62316218 [-41.7%])</span> 87,111,930 |
| `total_cycles        ` | <span style='color: green'>(-242669 [-12.3%])</span> 1,733,076 | <span style='color: green'>(-485338 [-12.3%])</span> 3,466,152 | <span style='color: green'>(-82390 [-4.1%])</span> 1,924,187 | <span style='color: green'>(-402948 [-20.7%])</span> 1,541,965 |
| `execute_metered_time_ms` |  3,449.50 |  6,899 |  3,835 |  3,064 |
| `execute_metered_insn_mi/s` |  0.50 | -          |  0.50 |  0.50 |
| `execute_e3_time_ms  ` |  6,257.50 |  12,515 |  6,959 |  5,556 |
| `execute_e3_insn_mi/s` |  0.28 | -          |  0.28 |  0.28 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-1179 [-79.3%])</span> 307.50 | <span style='color: green'>(-2358 [-79.3%])</span> 615 | <span style='color: green'>(-1132 [-74.8%])</span> 382 | <span style='color: green'>(-1226 [-84.0%])</span> 233 |
| `memory_finalize_time_ms` |  11.50 |  23 |  14 |  9 |
| `boundary_finalize_time_ms` |  0 |  0 |  0 |  0 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-920 [-19.4%])</span> 3,822 | <span style='color: green'>(-1840 [-19.4%])</span> 7,644 |  4,775 | <span style='color: green'>(-1838 [-39.0%])</span> 2,869 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-144 [-16.6%])</span> 722 | <span style='color: green'>(-288 [-16.6%])</span> 1,444 | <span style='color: green'>(-7 [-0.8%])</span> 861 | <span style='color: green'>(-281 [-32.5%])</span> 583 |
| `generate_perm_trace_time_ms` | <span style='color: green'>(-62 [-15.7%])</span> 329.50 | <span style='color: green'>(-123 [-15.7%])</span> 659 | <span style='color: red'>(+5 [+1.2%])</span> 420 | <span style='color: green'>(-128 [-34.9%])</span> 239 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-292 [-23.2%])</span> 964 | <span style='color: green'>(-583 [-23.2%])</span> 1,928 | <span style='color: green'>(-21 [-1.7%])</span> 1,249 | <span style='color: green'>(-562 [-45.3%])</span> 679 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-94 [-17.9%])</span> 429.50 | <span style='color: green'>(-187 [-17.9%])</span> 859 | <span style='color: red'>(+11 [+2.1%])</span> 536 | <span style='color: green'>(-198 [-38.0%])</span> 323 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-84 [-20.2%])</span> 330.50 | <span style='color: green'>(-167 [-20.2%])</span> 661 | <span style='color: green'>(-9 [-2.2%])</span> 407 | <span style='color: green'>(-158 [-38.3%])</span> 254 |
| `pcs_opening_time_ms ` | <span style='color: green'>(-248 [-19.2%])</span> 1,040 | <span style='color: green'>(-495 [-19.2%])</span> 2,080 | <span style='color: red'>(+5 [+0.4%])</span> 1,295 | <span style='color: green'>(-500 [-38.9%])</span> 785 |



<details>
<summary>Detailed Metrics</summary>

|  | keygen_time_ms | commit_exe_time_ms | app proof_time_ms | agg_layer_time_ms |
| --- | --- | --- | --- |
|  | 49 | 19 | 35,804 | 27,789 | 

| group | single_leaf_agg_time_ms | num_segments | num_children | memory_to_vec_partition_time_ms | insns | fri.log_blowup | execute_segment_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 16,016 |  | 1 |  |  | 1 |  |  |  | 
| regex_program |  | 2 |  | 23 | 4,206,554 | 1 | 13,322 | 7,372 | 0.57 | 

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

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 10,168,200 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 31,125,600 | 
| leaf | JalRangeCheckAir |  | 0 | JAL | 12 | 
| leaf | JalRangeCheckAir |  | 1 | JAL | 12 | 
| leaf | JalRangeCheckAir | Alloc | 0 | RANGE_CHECK | 326,328 | 
| leaf | JalRangeCheckAir | Alloc | 1 | RANGE_CHECK | 326,220 | 
| leaf | JalRangeCheckAir | IfEqI | 0 | JAL | 49,380 | 
| leaf | JalRangeCheckAir | IfEqI | 1 | JAL | 48,912 | 
| leaf | JalRangeCheckAir | IfNe | 0 | JAL | 24 | 
| leaf | JalRangeCheckAir | IfNe | 1 | JAL | 24 | 
| leaf | JalRangeCheckAir | ZipFor | 0 | JAL | 174,888 | 
| leaf | JalRangeCheckAir | ZipFor | 1 | JAL | 164,220 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 10,746 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 714,808 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 2,263,824 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchExt | 0 | VERIFY_BATCH | 9,950,000 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchExt | 1 | VERIFY_BATCH | 9,950,000 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchFelt | 0 | VERIFY_BATCH | 16,755,800 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | VerifyBatchFelt | 1 | VERIFY_BATCH | 35,859,800 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 1 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 117,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 108,000 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 181,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 166,800 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 1 | PHANTOM | 1,200 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | HintBitsF | 1 | PHANTOM | 3,744 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 73,674 | 
| leaf | PhantomAir | HintFelt | 1 | PHANTOM | 74,892 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 1,206 | 
| leaf | PhantomAir | HintInputVec | 1 | PHANTOM | 1,110 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 1 | PHANTOM | 21,600 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> |  | 1 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 28,072 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 27,608 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 18,908 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 21,460 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEI | 0 | ADD | 1,332,492 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddEI | 1 | ADD | 3,436,152 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddF | 0 | ADD | 682,080 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddF | 1 | ADD | 633,360 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddFI | 0 | ADD | 561,121 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddFI | 1 | ADD | 1,462,673 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddV | 0 | ADD | 504,803 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddV | 1 | ADD | 478,703 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,963,155 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | AddVI | 1 | ADD | 3,956,035 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,017,378 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 0 | MUL | 279,937 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 1 | ADD | 1,022,540 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | Alloc | 1 | MUL | 277,095 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | CastFV | 0 | ADD | 19,517 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | CastFV | 1 | ADD | 18,125 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 7,656 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 6,960 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivF | 0 | DIV | 60,900 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivF | 1 | DIV | 60,900 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 4,524 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 4,118 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmE | 0 | ADD | 97,324 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmE | 1 | ADD | 92,916 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmF | 0 | ADD | 673,902 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmF | 1 | ADD | 626,458 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmV | 0 | ADD | 1,038,229 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ImmV | 1 | ADD | 977,126 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | ADD | 629,300 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 0 | MUL | 629,300 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 1 | ADD | 582,900 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadE | 1 | MUL | 582,900 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | ADD | 304,529 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 0 | MUL | 17,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 1 | ADD | 279,937 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadF | 1 | MUL | 16,008 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | ADD | 157,238 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 0 | MUL | 144,507 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 1 | ADD | 144,594 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | LoadV | 1 | MUL | 132,791 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEF | 0 | MUL | 247,312 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEF | 1 | MUL | 245,920 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 17,864 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 46,980 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEI | 0 | ADD | 214,600 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulEI | 1 | ADD | 413,424 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulF | 0 | MUL | 854,166 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulF | 1 | MUL | 810,434 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulFI | 0 | MUL | 604,824 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulFI | 1 | MUL | 561,614 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulV | 0 | MUL | 33,031 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulV | 1 | MUL | 28,420 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulVI | 0 | MUL | 335,211 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | MulVI | 1 | MUL | 310,793 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | NegE | 0 | MUL | 2,668 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | NegE | 1 | MUL | 2,436 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | ADD | 626,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 0 | MUL | 626,400 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 1 | ADD | 580,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreE | 1 | MUL | 580,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | ADD | 11,252 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 0 | MUL | 10,788 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 1 | ADD | 16,588 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreF | 1 | MUL | 16,124 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 29 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | ADD | 72,587 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 0 | MUL | 36,482 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 1 | ADD | 69,107 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | StoreV | 1 | MUL | 33,524 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,319,616 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 0 | SUB | 439,872 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 1 | ADD | 1,214,868 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEF | 1 | SUB | 404,956 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 13,572 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 272,716 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEI | 0 | ADD | 15,312 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubEI | 1 | ADD | 13,920 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubFI | 0 | SUB | 604,128 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubFI | 1 | SUB | 560,976 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubV | 0 | SUB | 474,063 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubV | 1 | SUB | 446,716 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVI | 0 | SUB | 6,467 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVI | 1 | SUB | 6,409 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 58,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 58,000 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 17,951 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 16,443 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 4,578,027 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 5,467,805 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 11,408 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqE | 1 | NativeBranchEqualOpcode(BNE) | 11,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 92 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqEI | 1 | NativeBranchEqualOpcode(BNE) | 92 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 494,592 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqF | 1 | NativeBranchEqualOpcode(BNE) | 459,448 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 31,970 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqV | 1 | NativeBranchEqualOpcode(BNE) | 31,280 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 14,536 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertEqVI | 1 | NativeBranchEqualOpcode(BNE) | 13,409 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertNonZero | 0 | NativeBranchEqualOpcode(BEQ) | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | AssertNonZero | 1 | NativeBranchEqualOpcode(BEQ) | 23 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 332,557 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEq | 1 | NativeBranchEqualOpcode(BNE) | 1,048,593 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 282,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfEqI | 1 | NativeBranchEqualOpcode(BNE) | 275,264 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 173,581 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNe | 1 | NativeBranchEqualOpcode(BEQ) | 159,321 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 3,312 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | IfNeI | 1 | NativeBranchEqualOpcode(BEQ) | 3,036 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 2,056,821 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | ZipFor | 1 | NativeBranchEqualOpcode(BNE) | 2,860,395 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | Publish | 0 | PUBLISH | 972 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | Publish | 1 | PUBLISH | 972 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadF | 0 | LOADW | 1,423,506 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadF | 1 | LOADW | 2,007,810 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadV | 0 | LOADW | 3,495,954 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | LoadV | 1 | LOADW | 3,282,363 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreF | 0 | STOREW | 354,543 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreF | 1 | STOREW | 1,021,251 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreHintWord | 0 | HINT_STOREW | 1,069,152 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreHintWord | 1 | HINT_STOREW | 1,695,645 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreV | 0 | STOREW | 373,758 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | StoreV | 1 | STOREW | 369,348 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | LoadE | 0 | LOADW | 1,682,019 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | LoadE | 1 | LOADW | 1,794,555 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | StoreE | 0 | STOREW | 903,015 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | StoreE | 1 | STOREW | 859,437 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 955,966 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 1,800,402 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 651,624 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 605,872 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 2,508 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 2,280 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,771,598 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 2,286,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 70,300 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 135,432 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 264,518 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 603,630 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| regex_program | KeccakVmAir |  | KECCAK256 | 1 | 75,912 | 
| regex_program | PhantomAir |  | PHANTOM | 0 | 6 | 
| regex_program | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 408,512 | 
| regex_program | Rv32HintStoreAir |  | HINT_STOREW | 0 | 32 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | ADD | 0 | 19,877,472 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | ADD | 1 | 17,458,884 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | AND | 0 | 1,128,456 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | AND | 1 | 783,252 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | OR | 0 | 593,316 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | OR | 1 | 254,988 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | SUB | 0 | 778,032 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | SUB | 1 | 754,920 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | XOR | 0 | 179,280 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> |  | XOR | 1 | 164,988 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | SLT | 0 | 185 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | SLTU | 0 | 630,961 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> |  | SLTU | 1 | 606,837 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SLL | 0 | 5,846,907 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SLL | 1 | 5,712,287 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SRA | 1 | 53 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SRL | 0 | 269,611 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> |  | SRL | 1 | 53 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BEQ | 0 | 2,495,558 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BEQ | 1 | 1,895,504 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BNE | 0 | 1,734,902 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> |  | BNE | 1 | 1,227,668 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BGE | 0 | 9,408 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BGEU | 0 | 1,951,744 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BGEU | 1 | 1,951,136 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLT | 0 | 91,392 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLT | 1 | 73,024 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLTU | 0 | 1,264,864 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> |  | BLTU | 1 | 1,003,712 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | JAL | 0 | 518,094 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | JAL | 1 | 506,358 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | LUI | 0 | 418,626 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> |  | LUI | 1 | 379,656 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | JALR | 0 | 1,869,728 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> |  | JALR | 1 | 1,778,588 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | LOADB | 0 | 24,876 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | LOADB | 1 | 720 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> |  | LOADH | 0 | 288 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADBU | 0 | 641,650 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADBU | 1 | 492,328 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADHU | 0 | 3,813 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADHU | 1 | 82 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADW | 0 | 24,436,820 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | LOADW | 1 | 22,949,463 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREB | 0 | 522,135 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREB | 1 | 1,230 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREH | 0 | 413,034 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREW | 0 | 16,971,540 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> |  | STOREW | 1 | 14,691,284 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> |  | DIVU | 0 | 6,726 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> |  | MULHU | 0 | 9,477 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | MUL | 0 | 808,604 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> |  | MUL | 1 | 806,093 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | AUIPC | 0 | 412,280 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> |  | AUIPC | 1 | 377,800 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | AccessAdapterAir<8> | 1 | 16,384 |  | 16 | 17 | 540,672 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | FriReducedOpeningAir | 1 | 2,097,152 |  | 84 | 27 | 232,783,872 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 131,072 |  | 312 | 398 | 93,061,120 | 
| leaf | PhantomAir | 0 | 131,072 |  | 12 | 6 | 2,359,296 | 
| leaf | PhantomAir | 1 | 131,072 |  | 12 | 6 | 2,359,296 | 
| leaf | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 28 | 23 | 13,369,344 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 36 | 38 | 19,398,656 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 20 | 12 | 16,777,216 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| regex_program | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| regex_program | KeccakVmAir | 1 | 32 |  | 1,056 | 3,163 | 135,008 | 
| regex_program | MemoryMerkleAir<8> | 0 | 131,072 |  | 16 | 32 | 6,291,456 | 
| regex_program | MemoryMerkleAir<8> | 1 | 4,096 |  | 16 | 32 | 196,608 | 
| regex_program | PersistentBoundaryAir<8> | 0 | 131,072 |  | 12 | 20 | 4,194,304 | 
| regex_program | PersistentBoundaryAir<8> | 1 | 2,048 |  | 12 | 20 | 65,536 | 
| regex_program | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| regex_program | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 16,384 |  | 8 | 300 | 5,046,272 | 
| regex_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 2,048 |  | 8 | 300 | 630,784 | 
| regex_program | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| regex_program | Rv32HintStoreAir | 0 | 16,384 |  | 44 | 32 | 1,245,184 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 32,768 |  | 40 | 37 | 2,523,136 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 32,768 |  | 40 | 37 | 2,523,136 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 131,072 |  | 52 | 53 | 13,762,560 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 131,072 |  | 28 | 26 | 7,077,888 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 131,072 |  | 32 | 32 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 65,536 |  | 28 | 18 | 3,014,656 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 131,072 |  | 36 | 28 | 8,388,608 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 65,536 |  | 36 | 28 | 4,194,304 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 1,024 |  | 52 | 36 | 90,112 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 32 |  | 52 | 36 | 2,816 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 1,048,576 |  | 52 | 41 | 97,517,568 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 128 |  | 72 | 59 | 16,768 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 256 |  | 72 | 39 | 28,416 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 32,768 |  | 52 | 31 | 2,719,744 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 32,768 |  | 28 | 20 | 1,572,864 | 
| regex_program | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| regex_program | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| leaf | AccessAdapter<2> | 0 | 524,288 | 
| leaf | AccessAdapter<2> | 1 | 524,288 | 
| leaf | AccessAdapter<4> | 0 | 262,144 | 
| leaf | AccessAdapter<4> | 1 | 262,144 | 
| leaf | AccessAdapter<8> | 0 | 4,096 | 
| leaf | AccessAdapter<8> | 1 | 16,384 | 
| leaf | Boundary | 0 | 120,818 | 
| leaf | Boundary | 1 | 464,190 | 
| leaf | FriReducedOpeningAir | 0 | 376,600 | 
| leaf | FriReducedOpeningAir | 1 | 1,152,800 | 
| leaf | JalRangeCheckAir | 0 | 45,886 | 
| leaf | JalRangeCheckAir | 1 | 44,949 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 68,896 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 120,815 | 
| leaf | PhantomAir | 0 | 77,376 | 
| leaf | PhantomAir | 1 | 73,515 | 
| leaf | ProgramChip | 0 | 229,382 | 
| leaf | ProgramChip | 1 | 229,382 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 737,689 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 931,712 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 147,884 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 211,395 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 36 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 36 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 319,853 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 398,877 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 95,742 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 98,296 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 97,803 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 142,992 | 
| leaf | VmConnectorAir | 0 | 2 | 
| leaf | VmConnectorAir | 1 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| regex_program | AccessAdapter<8> | 0 | 65,536 | 
| regex_program | AccessAdapter<8> | 1 | 1,024 | 
| regex_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| regex_program | BitwiseOperationLookupAir<8> | 1 | 65,536 | 
| regex_program | Boundary | 0 | 69,142 | 
| regex_program | Boundary | 1 | 1,426 | 
| regex_program | KeccakVmAir | 1 | 24 | 
| regex_program | Merkle | 0 | 70,336 | 
| regex_program | Merkle | 1 | 2,156 | 
| regex_program | PhantomAir | 0 | 1 | 
| regex_program | Poseidon2PeripheryAir<F, 1> | 0 | 13,884 | 
| regex_program | Poseidon2PeripheryAir<F, 1> | 1 | 1,843 | 
| regex_program | ProgramChip | 0 | 90,415 | 
| regex_program | ProgramChip | 1 | 90,415 | 
| regex_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| regex_program | RangeTupleCheckerAir<2> | 1 | 524,288 | 
| regex_program | Rv32HintStoreAir | 0 | 12,767 | 
| regex_program | VariableRangeCheckerAir | 0 | 262,144 | 
| regex_program | VariableRangeCheckerAir | 1 | 262,144 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 626,571 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 539,362 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 17,058 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 16,401 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 115,406 | 
| regex_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 1 | 107,781 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 162,710 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 120,122 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 103,669 | 
| regex_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 1 | 94,621 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 52,040 | 
| regex_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 49,223 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 66,776 | 
| regex_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 1 | 63,521 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 0 | 699 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 1 | 20 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 1,048,512 | 
| regex_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 1 | 930,108 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 0 | 114 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 0 | 243 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 0 | 26,084 | 
| regex_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 1 | 26,003 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 20,615 | 
| regex_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 1 | 18,890 | 
| regex_program | VmConnectorAir | 0 | 2 | 
| regex_program | VmConnectorAir | 1 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf |  | 1 | ADD | 2 | 
| leaf |  | 1 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 25,157 | 
| leaf | AddE | 1 | FE4ADD | 47,379 | 
| leaf | AddEFFI | 0 | ADD | 968 | 
| leaf | AddEFFI | 1 | ADD | 952 | 
| leaf | AddEFI | 0 | ADD | 652 | 
| leaf | AddEFI | 1 | ADD | 740 | 
| leaf | AddEI | 0 | ADD | 45,948 | 
| leaf | AddEI | 1 | ADD | 118,488 | 
| leaf | AddF | 0 | ADD | 23,520 | 
| leaf | AddF | 1 | ADD | 21,840 | 
| leaf | AddFI | 0 | ADD | 19,349 | 
| leaf | AddFI | 1 | ADD | 50,437 | 
| leaf | AddV | 0 | ADD | 17,407 | 
| leaf | AddV | 1 | ADD | 16,507 | 
| leaf | AddVI | 0 | ADD | 67,695 | 
| leaf | AddVI | 1 | ADD | 136,415 | 
| leaf | Alloc | 0 | ADD | 35,082 | 
| leaf | Alloc | 0 | MUL | 9,653 | 
| leaf | Alloc | 0 | RANGE_CHECK | 27,194 | 
| leaf | Alloc | 1 | ADD | 35,260 | 
| leaf | Alloc | 1 | MUL | 9,555 | 
| leaf | Alloc | 1 | RANGE_CHECK | 27,185 | 
| leaf | AssertEqE | 0 | NativeBranchEqualOpcode(BNE) | 496 | 
| leaf | AssertEqE | 1 | NativeBranchEqualOpcode(BNE) | 488 | 
| leaf | AssertEqEI | 0 | NativeBranchEqualOpcode(BNE) | 4 | 
| leaf | AssertEqEI | 1 | NativeBranchEqualOpcode(BNE) | 4 | 
| leaf | AssertEqF | 0 | NativeBranchEqualOpcode(BNE) | 21,504 | 
| leaf | AssertEqF | 1 | NativeBranchEqualOpcode(BNE) | 19,976 | 
| leaf | AssertEqV | 0 | NativeBranchEqualOpcode(BNE) | 1,390 | 
| leaf | AssertEqV | 1 | NativeBranchEqualOpcode(BNE) | 1,360 | 
| leaf | AssertEqVI | 0 | NativeBranchEqualOpcode(BNE) | 632 | 
| leaf | AssertEqVI | 1 | NativeBranchEqualOpcode(BNE) | 583 | 
| leaf | AssertNonZero | 0 | NativeBranchEqualOpcode(BEQ) | 1 | 
| leaf | AssertNonZero | 1 | NativeBranchEqualOpcode(BEQ) | 1 | 
| leaf | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 2 | 
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 1 | PHANTOM | 1,600 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 1 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 1 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 1 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 1 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 1 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 1 | PHANTOM | 200 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 1 | PHANTOM | 1,600 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 19,600 | 
| leaf | CT-exp-reverse-bits-len | 1 | PHANTOM | 18,000 | 
| leaf | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 1 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 30,200 | 
| leaf | CT-single-reduced-opening-eval | 1 | PHANTOM | 27,800 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 1 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 1 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 1 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 1 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 1 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 1 | PHANTOM | 4,000 | 
| leaf | CT-verify-query | 0 | PHANTOM | 200 | 
| leaf | CT-verify-query | 1 | PHANTOM | 200 | 
| leaf | CastFV | 0 | ADD | 673 | 
| leaf | CastFV | 1 | ADD | 625 | 
| leaf | DivE | 0 | BBE4DIV | 17,148 | 
| leaf | DivE | 1 | BBE4DIV | 15,944 | 
| leaf | DivEIN | 0 | ADD | 264 | 
| leaf | DivEIN | 0 | BBE4DIV | 66 | 
| leaf | DivEIN | 1 | ADD | 240 | 
| leaf | DivEIN | 1 | BBE4DIV | 60 | 
| leaf | DivF | 0 | DIV | 2,100 | 
| leaf | DivF | 1 | DIV | 2,100 | 
| leaf | DivFIN | 0 | DIV | 156 | 
| leaf | DivFIN | 1 | DIV | 142 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 15,100 | 
| leaf | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 13,900 | 
| leaf | HintBitsF | 0 | PHANTOM | 672 | 
| leaf | HintBitsF | 1 | PHANTOM | 624 | 
| leaf | HintFelt | 0 | PHANTOM | 12,279 | 
| leaf | HintFelt | 1 | PHANTOM | 12,482 | 
| leaf | HintInputVec | 0 | PHANTOM | 201 | 
| leaf | HintInputVec | 1 | PHANTOM | 185 | 
| leaf | HintLoad | 0 | PHANTOM | 3,600 | 
| leaf | HintLoad | 1 | PHANTOM | 3,600 | 
| leaf | IfEq | 0 | NativeBranchEqualOpcode(BNE) | 14,459 | 
| leaf | IfEq | 1 | NativeBranchEqualOpcode(BNE) | 45,591 | 
| leaf | IfEqI | 0 | JAL | 4,115 | 
| leaf | IfEqI | 0 | NativeBranchEqualOpcode(BNE) | 12,280 | 
| leaf | IfEqI | 1 | JAL | 4,076 | 
| leaf | IfEqI | 1 | NativeBranchEqualOpcode(BNE) | 11,968 | 
| leaf | IfNe | 0 | JAL | 2 | 
| leaf | IfNe | 0 | NativeBranchEqualOpcode(BEQ) | 7,547 | 
| leaf | IfNe | 1 | JAL | 2 | 
| leaf | IfNe | 1 | NativeBranchEqualOpcode(BEQ) | 6,927 | 
| leaf | IfNeI | 0 | NativeBranchEqualOpcode(BEQ) | 144 | 
| leaf | IfNeI | 1 | NativeBranchEqualOpcode(BEQ) | 132 | 
| leaf | ImmE | 0 | ADD | 3,356 | 
| leaf | ImmE | 1 | ADD | 3,204 | 
| leaf | ImmF | 0 | ADD | 23,238 | 
| leaf | ImmF | 1 | ADD | 21,602 | 
| leaf | ImmV | 0 | ADD | 35,801 | 
| leaf | ImmV | 1 | ADD | 33,694 | 
| leaf | LoadE | 0 | ADD | 21,700 | 
| leaf | LoadE | 0 | LOADW | 62,297 | 
| leaf | LoadE | 0 | MUL | 21,700 | 
| leaf | LoadE | 1 | ADD | 20,100 | 
| leaf | LoadE | 1 | LOADW | 66,465 | 
| leaf | LoadE | 1 | MUL | 20,100 | 
| leaf | LoadF | 0 | ADD | 10,501 | 
| leaf | LoadF | 0 | LOADW | 67,786 | 
| leaf | LoadF | 0 | MUL | 600 | 
| leaf | LoadF | 1 | ADD | 9,653 | 
| leaf | LoadF | 1 | LOADW | 95,610 | 
| leaf | LoadF | 1 | MUL | 552 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadHeapPtr | 1 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 5,422 | 
| leaf | LoadV | 0 | LOADW | 166,474 | 
| leaf | LoadV | 0 | MUL | 4,983 | 
| leaf | LoadV | 1 | ADD | 4,986 | 
| leaf | LoadV | 1 | LOADW | 156,303 | 
| leaf | LoadV | 1 | MUL | 4,579 | 
| leaf | MulE | 0 | BBE4MUL | 46,621 | 
| leaf | MulE | 1 | BBE4MUL | 60,160 | 
| leaf | MulEF | 0 | MUL | 8,528 | 
| leaf | MulEF | 1 | MUL | 8,480 | 
| leaf | MulEFI | 0 | MUL | 616 | 
| leaf | MulEFI | 1 | MUL | 1,620 | 
| leaf | MulEI | 0 | ADD | 7,400 | 
| leaf | MulEI | 0 | BBE4MUL | 1,850 | 
| leaf | MulEI | 1 | ADD | 14,256 | 
| leaf | MulEI | 1 | BBE4MUL | 3,564 | 
| leaf | MulF | 0 | MUL | 29,454 | 
| leaf | MulF | 1 | MUL | 27,946 | 
| leaf | MulFI | 0 | MUL | 20,856 | 
| leaf | MulFI | 1 | MUL | 19,366 | 
| leaf | MulV | 0 | MUL | 1,139 | 
| leaf | MulV | 1 | MUL | 980 | 
| leaf | MulVI | 0 | MUL | 11,559 | 
| leaf | MulVI | 1 | MUL | 10,717 | 
| leaf | NegE | 0 | MUL | 92 | 
| leaf | NegE | 1 | MUL | 84 | 
| leaf | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 1,796 | 
| leaf | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 5,688 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | Publish | 1 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 21,600 | 
| leaf | StoreE | 0 | MUL | 21,600 | 
| leaf | StoreE | 0 | STOREW | 33,445 | 
| leaf | StoreE | 1 | ADD | 20,000 | 
| leaf | StoreE | 1 | MUL | 20,000 | 
| leaf | StoreE | 1 | STOREW | 31,831 | 
| leaf | StoreF | 0 | ADD | 388 | 
| leaf | StoreF | 0 | MUL | 372 | 
| leaf | StoreF | 0 | STOREW | 16,883 | 
| leaf | StoreF | 1 | ADD | 572 | 
| leaf | StoreF | 1 | MUL | 556 | 
| leaf | StoreF | 1 | STOREW | 48,631 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHeapPtr | 1 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 50,912 | 
| leaf | StoreHintWord | 1 | HINT_STOREW | 80,745 | 
| leaf | StoreV | 0 | ADD | 2,503 | 
| leaf | StoreV | 0 | MUL | 1,258 | 
| leaf | StoreV | 0 | STOREW | 17,798 | 
| leaf | StoreV | 1 | ADD | 2,383 | 
| leaf | StoreV | 1 | MUL | 1,156 | 
| leaf | StoreV | 1 | STOREW | 17,588 | 
| leaf | SubE | 0 | FE4SUB | 6,961 | 
| leaf | SubE | 1 | FE4SUB | 15,885 | 
| leaf | SubEF | 0 | ADD | 45,504 | 
| leaf | SubEF | 0 | SUB | 15,168 | 
| leaf | SubEF | 1 | ADD | 41,892 | 
| leaf | SubEF | 1 | SUB | 13,964 | 
| leaf | SubEFI | 0 | ADD | 468 | 
| leaf | SubEFI | 1 | ADD | 9,404 | 
| leaf | SubEI | 0 | ADD | 528 | 
| leaf | SubEI | 1 | ADD | 480 | 
| leaf | SubFI | 0 | SUB | 20,832 | 
| leaf | SubFI | 1 | SUB | 19,344 | 
| leaf | SubV | 0 | SUB | 16,347 | 
| leaf | SubV | 1 | SUB | 15,404 | 
| leaf | SubVI | 0 | SUB | 223 | 
| leaf | SubVI | 1 | SUB | 221 | 
| leaf | SubVIN | 0 | SUB | 2,000 | 
| leaf | SubVIN | 1 | SUB | 2,000 | 
| leaf | UnsafeCastVF | 0 | ADD | 619 | 
| leaf | UnsafeCastVF | 1 | ADD | 567 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 1 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 1 | VERIFY_BATCH | 800 | 
| leaf | ZipFor | 0 | ADD | 157,863 | 
| leaf | ZipFor | 0 | JAL | 14,574 | 
| leaf | ZipFor | 0 | NativeBranchEqualOpcode(BNE) | 89,427 | 
| leaf | ZipFor | 1 | ADD | 188,545 | 
| leaf | ZipFor | 1 | JAL | 13,685 | 
| leaf | ZipFor | 1 | NativeBranchEqualOpcode(BNE) | 124,365 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| regex_program |  | ADD | 0 | 552,152 | 
| regex_program |  | ADD | 1 | 484,969 | 
| regex_program |  | AND | 0 | 31,346 | 
| regex_program |  | AND | 1 | 21,757 | 
| regex_program |  | AUIPC | 0 | 20,615 | 
| regex_program |  | AUIPC | 1 | 18,890 | 
| regex_program |  | BEQ | 0 | 95,983 | 
| regex_program |  | BEQ | 1 | 72,904 | 
| regex_program |  | BGE | 0 | 294 | 
| regex_program |  | BGEU | 0 | 60,992 | 
| regex_program |  | BGEU | 1 | 60,973 | 
| regex_program |  | BLT | 0 | 2,856 | 
| regex_program |  | BLT | 1 | 2,282 | 
| regex_program |  | BLTU | 0 | 39,527 | 
| regex_program |  | BLTU | 1 | 31,366 | 
| regex_program |  | BNE | 0 | 66,727 | 
| regex_program |  | BNE | 1 | 47,218 | 
| regex_program |  | DIVU | 0 | 114 | 
| regex_program |  | HINT_BUFFER | 0 | 1 | 
| regex_program |  | HINT_STOREW | 0 | 1 | 
| regex_program |  | JAL | 0 | 28,783 | 
| regex_program |  | JAL | 1 | 28,131 | 
| regex_program |  | JALR | 0 | 66,776 | 
| regex_program |  | JALR | 1 | 63,521 | 
| regex_program |  | KECCAK256 | 1 | 1 | 
| regex_program |  | LOADB | 0 | 691 | 
| regex_program |  | LOADB | 1 | 20 | 
| regex_program |  | LOADBU | 0 | 15,650 | 
| regex_program |  | LOADBU | 1 | 12,008 | 
| regex_program |  | LOADH | 0 | 8 | 
| regex_program |  | LOADHU | 0 | 93 | 
| regex_program |  | LOADHU | 1 | 2 | 
| regex_program |  | LOADW | 0 | 596,020 | 
| regex_program |  | LOADW | 1 | 559,743 | 
| regex_program |  | LUI | 0 | 23,257 | 
| regex_program |  | LUI | 1 | 21,092 | 
| regex_program |  | MUL | 0 | 26,084 | 
| regex_program |  | MUL | 1 | 26,003 | 
| regex_program |  | MULHU | 0 | 243 | 
| regex_program |  | OR | 0 | 16,481 | 
| regex_program |  | OR | 1 | 7,083 | 
| regex_program |  | PHANTOM | 0 | 1 | 
| regex_program |  | SLL | 0 | 110,319 | 
| regex_program |  | SLL | 1 | 107,779 | 
| regex_program |  | SLT | 0 | 5 | 
| regex_program |  | SLTU | 0 | 17,053 | 
| regex_program |  | SLTU | 1 | 16,401 | 
| regex_program |  | SRA | 1 | 1 | 
| regex_program |  | SRL | 0 | 5,087 | 
| regex_program |  | SRL | 1 | 1 | 
| regex_program |  | STOREB | 0 | 12,735 | 
| regex_program |  | STOREB | 1 | 30 | 
| regex_program |  | STOREH | 0 | 10,074 | 
| regex_program |  | STOREW | 0 | 413,940 | 
| regex_program |  | STOREW | 1 | 358,325 | 
| regex_program |  | SUB | 0 | 21,612 | 
| regex_program |  | SUB | 1 | 20,970 | 
| regex_program |  | XOR | 0 | 4,980 | 
| regex_program |  | XOR | 1 | 4,583 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_metered_time_ms | execute_metered_insn_mi/s | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 0 | 233 | 11,722 | 1,541,965 | 321,396,202 | 2,869 | 323 | 254 | 679 | 785 | 9 | 583 | 87,111,930 | 1,541,966 | 239 | 3,064 | 0.50 | 5,556 | 0.28 | 0 | 
| leaf | 1 | 382 | 15,951 | 1,924,187 | 540,429,802 | 4,775 | 536 | 407 | 1,249 | 1,295 | 14 | 861 | 143,561,106 | 1,924,188 | 420 | 3,835 | 0.50 | 6,959 | 0.28 | 0 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| leaf | 0 | 0 | 6,160,516 | 2,013,265,921 | 
| leaf | 0 | 1 | 32,649,472 | 2,013,265,921 | 
| leaf | 0 | 2 | 3,080,258 | 2,013,265,921 | 
| leaf | 0 | 3 | 32,383,236 | 2,013,265,921 | 
| leaf | 0 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 0 | 5 | 75,059,914 | 2,013,265,921 | 
| leaf | 1 | 0 | 9,568,388 | 2,013,265,921 | 
| leaf | 1 | 1 | 64,930,048 | 2,013,265,921 | 
| leaf | 1 | 2 | 4,784,194 | 2,013,265,921 | 
| leaf | 1 | 3 | 65,044,740 | 2,013,265,921 | 
| leaf | 1 | 4 | 262,144 | 2,013,265,921 | 
| leaf | 1 | 5 | 145,113,802 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | prove_segment_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | merkle_finalize_time_ms | memory_to_vec_partition_time_ms | memory_finalize_time_ms | main_trace_commit_time_ms | main_cells_used | insns | generate_perm_trace_time_ms | execute_e3_time_ms | execute_e3_insn_mi/s | boundary_finalize_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex_program | 0 | 759 | 11,034 | 2,240,500 | 275,648,700 | 2,810 | 292 | 272 | 3,035 | 613 | 832 | 230 | 23 | 254 | 573 | 93,231,253 | 2,240,500 | 223 | 7,465 | 0.30 | 5 | 
| regex_program | 1 | 725 | 9,586 | 1,966,053 | 244,236,956 | 2,369 | 251 | 205 | 2,964 | 516 | 758 | 68 | 22 | 72 | 442 | 74,614,681 | 1,966,054 | 183 | 6,492 | 0.30 | 0 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| regex_program | 0 | 0 | 5,868,294 | 2,013,265,921 | 
| regex_program | 0 | 1 | 16,687,360 | 2,013,265,921 | 
| regex_program | 0 | 2 | 2,934,147 | 2,013,265,921 | 
| regex_program | 0 | 3 | 19,705,092 | 2,013,265,921 | 
| regex_program | 0 | 4 | 524,288 | 2,013,265,921 | 
| regex_program | 0 | 5 | 262,144 | 2,013,265,921 | 
| regex_program | 0 | 6 | 6,668,800 | 2,013,265,921 | 
| regex_program | 0 | 7 | 134,144 | 2,013,265,921 | 
| regex_program | 0 | 8 | 53,849,229 | 2,013,265,921 | 
| regex_program | 1 | 0 | 5,439,622 | 2,013,265,921 | 
| regex_program | 1 | 1 | 15,281,152 | 2,013,265,921 | 
| regex_program | 1 | 2 | 2,719,811 | 2,013,265,921 | 
| regex_program | 1 | 3 | 18,291,812 | 2,013,265,921 | 
| regex_program | 1 | 4 | 14,336 | 2,013,265,921 | 
| regex_program | 1 | 5 | 6,144 | 2,013,265,921 | 
| regex_program | 1 | 6 | 6,558,016 | 2,013,265,921 | 
| regex_program | 1 | 7 | 131,072 | 2,013,265,921 | 
| regex_program | 1 | 8 | 49,492,589 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/regex-85e28f86705cbf0eaca3e424bad3af07da9f5ad6/regex-regex_program.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/85e28f86705cbf0eaca3e424bad3af07da9f5ad6

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/16180787988)
