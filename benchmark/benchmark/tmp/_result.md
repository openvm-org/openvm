## Benchmark for VM Verifier for Fibonacci Air
| Total Cells | Total Prove (ms) | Main Trace Gen (ms) | Perm Trace Gen (ms) | Calc Quotient Values (ms) | Rest of Prove (ms) |
|-----------------------------|-----------------------|--------------------------|--------------------------|-----------------|----------------|
| 133_513_236 | 36600.00 | 1560.00 | 631.00 | 8330.00 | 26079.00 |

### AIR metrics
| Name | Rows | Cells | Prep Cols | Main Cols | Perm Cols |
|------|------|-------|-----------|-----------|-----------|
| CoreAir              | 1_048_576  | 93_323_264  | 0     | [69] | [20] |
| ProgramAir           | 65_536     | 589_824     | 9     | [1] | [8] |
| FieldArithmeticAir   | 524_288    | 24_641_536  | 0     | [31] | [16] |
| FieldExtensionArithmeticAir | 16_384     | 1_769_472   | 0     | [68] | [40] |
| Poseidon2VmAir       | 8_192      | 4_931_584   | 0     | [502] | [100] |
| MemoryAuditAir       | 262_144    | 7_077_888   | 0     | [19] | [8] |
| VariableRangeCheckerAir | 131_072    | 1_179_648   | 2     | [1] | [8] |
| VmConnectorAir       | 2          | 20          | 1     | [2] | [8] |
<details>
<summary>

### Custom VM metrics

</summary>

| Name | Value |
|------|------:|

#### Opcode metrics
| Name | Frequency | Trace Cells Contributed |
|------|------:|-----:|
| FADD                 | `        276_515` | `      8_638_047` |
| BNE                  | `        144_081` | `      9_941_589` |
| STOREW               | `        100_825` | `      8_022_578` |
| LOADW                | `         84_566` | `      5_909_249` |
| SHINTW               | `         72_202` | `      6_353_776` |
| LOADW2               | `         59_899` | `      4_136_375` |
| STOREW2              | `         37_090` | `      3_113_364` |
| FMUL                 | `         36_787` | `      1_200_855` |
| JAL                  | `         23_160` | `      1_598_059` |
| FSUB                 | `         12_383` | `        446_174` |
| HINT_INPUT           | `          9_923` | `        684_687` |
| BEQ                  | `          7_771` | `        536_199` |
| COMP_POS2            | `          6_592` | `      3_309_184` |
| CT_END               | `          5_569` | `        384_261` |
| CT_START             | `          5_569` | `        384_261` |
| BBE4MUL              | `          5_427` | `        371_544` |
| FE4ADD               | `          2_514` | `        171_864` |
| FE4SUB               | `          2_474` | `        168_384` |
| BBE4DIV              | `          1_651` | `        112_344` |
| PERM_POS2            | `          1_047` | `        525_594` |
| HINT_BITS            | `            104` | `          7_176` |
| FDIV                 | `              3` | `             93` |
| TERMINATE            | `              1` | `             69` |

### DSL counts
How many opcodes each DSL instruction generates:
| Name | Count |
|------|------:|
| For                  | `        236_216` |
| StoreHintWord        | `        131_257` |
| AddVI                | `         77_003` |
| Alloc                | `         74_752` |
| LoadV                | `         49_694` |
| StoreE               | `         47_836` |
| LoadE                | `         32_584` |
| LoadF                | `         27_723` |
| IfEqI                | `         26_072` |
| StoreV               | `         24_156` |
| ImmV                 | `         18_024` |
| StoreF               | `         17_682` |
| AddEI                | `         11_220` |
| ImmF                 | `         10_539` |
| HintInputVec         | `          9_923` |
| AssertEqF            | `          8_344` |
| MulVI                | `          7_626` |
| IfNe                 | `          6_751` |
| SubEF                | `          6_612` |
| Poseidon2CompressBabyBear | `          6_592` |
| AddV                 | `          5_784` |
| CycleTrackerEnd      | `          5_569` |
| CycleTrackerStart    | `          5_569` |
| SubV                 | `          5_562` |
| MulE                 | `          5_378` |
| SubVI                | `          4_344` |
| MulF                 | `          4_330` |
| AddFI                | `          4_262` |
| AssertEqV            | `          4_052` |
| MulEF                | `          3_304` |
| MulV                 | `          3_224` |
| AddE                 | `          2_514` |
| SubE                 | `          2_474` |
| ImmE                 | `          2_068` |
| DivE                 | `          1_650` |
| IfEq                 | `          1_167` |
| Poseidon2PermuteBabyBear | `          1_047` |
| IfNeI                | `          1_031` |
| SubVIN               | `            824` |
| AddEFFI              | `            588` |
| AssertEqE            | `            416` |
| MulEI                | `            245` |
| HintBitsF            | `            104` |
| AssertEqVI           | `             16` |
| SubEI                | `              8` |
| DivEIN               | `              5` |
| AssertEqEI           | `              4` |
| DivFIN               | `              3` |
| Halt                 | `              1` |
| MulFI                | `              1` |
</details>
