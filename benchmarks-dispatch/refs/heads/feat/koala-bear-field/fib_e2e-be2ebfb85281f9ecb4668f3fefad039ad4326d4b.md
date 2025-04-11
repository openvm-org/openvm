| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  370.38 |  229.98 |
| fib_e2e |  54.06 |  7.92 |
| leaf |  48.53 |  7.93 |
| internal.0 |  60.51 |  17.14 |
| internal.1 |  20.64 |  10.35 |
| internal.2 |  10.26 |  10.26 |
| root |  41.51 |  41.51 |
| halo2_outer |  90.89 |  90.89 |
| halo2_wrapper |  43.97 |  43.97 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  7,723.43 |  54,064 |  7,922 |  7,024 |
| `main_cells_used     ` |  57,756,362.14 |  404,294,535 |  58,898,534 |  51,097,540 |
| `total_cycles        ` |  1,714,325.29 |  12,000,277 |  1,747,603 |  1,515,164 |
| `execute_time_ms     ` |  5,334.43 |  37,341 |  5,458 |  4,712 |
| `trace_gen_time_ms   ` |  604.71 |  4,233 |  626 |  540 |
| `stark_prove_excluding_trace_time_ms` |  1,784.29 |  12,490 |  1,864 |  1,763 |
| `main_trace_commit_time_ms` |  288.29 |  2,018 |  318 |  281 |
| `generate_perm_trace_time_ms` |  118.29 |  828 |  122 |  115 |
| `perm_trace_commit_time_ms` |  439.14 |  3,074 |  443 |  437 |
| `quotient_poly_compute_time_ms` |  251.57 |  1,761 |  256 |  245 |
| `quotient_poly_commit_time_ms` |  186.14 |  1,303 |  216 |  179 |
| `pcs_opening_time_ms ` |  496.57 |  3,476 |  510 |  487 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  6,933.57 |  48,535 |  7,935 |  6,597 |
| `main_cells_used     ` |  61,974,599.57 |  433,822,197 |  70,452,126 |  59,082,337 |
| `total_cycles        ` |  1,143,729.29 |  8,006,105 |  1,310,905 |  1,085,388 |
| `execute_time_ms     ` |  4,145.29 |  29,017 |  4,713 |  3,933 |
| `trace_gen_time_ms   ` |  590.71 |  4,135 |  662 |  553 |
| `stark_prove_excluding_trace_time_ms` |  2,197.57 |  15,383 |  2,560 |  2,111 |
| `main_trace_commit_time_ms` |  365.14 |  2,556 |  424 |  352 |
| `generate_perm_trace_time_ms` |  132.86 |  930 |  163 |  126 |
| `perm_trace_commit_time_ms` |  558 |  3,906 |  691 |  531 |
| `quotient_poly_compute_time_ms` |  316.14 |  2,213 |  377 |  301 |
| `quotient_poly_commit_time_ms` |  243.86 |  1,707 |  281 |  236 |
| `pcs_opening_time_ms ` |  575.57 |  4,029 |  628 |  553 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  15,126.75 |  60,507 |  17,144 |  9,146 |
| `main_cells_used     ` |  119,823,550 |  479,294,200 |  136,809,223 |  69,107,531 |
| `total_cycles        ` |  2,118,763.75 |  8,475,055 |  2,421,386 |  1,210,998 |
| `execute_time_ms     ` |  7,588 |  30,352 |  8,680 |  4,343 |
| `trace_gen_time_ms   ` |  1,097 |  4,388 |  1,261 |  662 |
| `stark_prove_excluding_trace_time_ms` |  6,441.75 |  25,767 |  7,228 |  4,141 |
| `main_trace_commit_time_ms` |  1,257.50 |  5,030 |  1,426 |  761 |
| `generate_perm_trace_time_ms` |  265.50 |  1,062 |  301 |  160 |
| `perm_trace_commit_time_ms` |  1,153 |  4,612 |  1,313 |  704 |
| `quotient_poly_compute_time_ms` |  1,167 |  4,668 |  1,315 |  737 |
| `quotient_poly_commit_time_ms` |  1,123.50 |  4,494 |  1,246 |  778 |
| `pcs_opening_time_ms ` |  1,470.25 |  5,881 |  1,643 |  995 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,320.50 |  20,641 |  10,345 |  10,296 |
| `main_cells_used     ` |  74,977,613 |  149,955,226 |  75,273,875 |  74,681,351 |
| `total_cycles        ` |  1,575,568.50 |  3,151,137 |  1,579,468 |  1,571,669 |
| `execute_time_ms     ` |  5,451.50 |  10,903 |  5,481 |  5,422 |
| `trace_gen_time_ms   ` |  730 |  1,460 |  738 |  722 |
| `stark_prove_excluding_trace_time_ms` |  4,139 |  8,278 |  4,185 |  4,093 |
| `main_trace_commit_time_ms` |  764 |  1,528 |  772 |  756 |
| `generate_perm_trace_time_ms` |  153.50 |  307 |  154 |  153 |
| `perm_trace_commit_time_ms` |  694 |  1,388 |  709 |  679 |
| `quotient_poly_compute_time_ms` |  715.50 |  1,431 |  723 |  708 |
| `quotient_poly_commit_time_ms` |  797 |  1,594 |  810 |  784 |
| `pcs_opening_time_ms ` |  1,010 |  2,020 |  1,012 |  1,008 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,258 |  10,258 |  10,258 |  10,258 |
| `main_cells_used     ` |  73,837,376 |  73,837,376 |  73,837,376 |  73,837,376 |
| `total_cycles        ` |  1,564,464 |  1,564,464 |  1,564,464 |  1,564,464 |
| `execute_time_ms     ` |  5,414 |  5,414 |  5,414 |  5,414 |
| `trace_gen_time_ms   ` |  746 |  746 |  746 |  746 |
| `stark_prove_excluding_trace_time_ms` |  4,098 |  4,098 |  4,098 |  4,098 |
| `main_trace_commit_time_ms` |  758 |  758 |  758 |  758 |
| `generate_perm_trace_time_ms` |  150 |  150 |  150 |  150 |
| `perm_trace_commit_time_ms` |  679 |  679 |  679 |  679 |
| `quotient_poly_compute_time_ms` |  710 |  710 |  710 |  710 |
| `quotient_poly_commit_time_ms` |  798 |  798 |  798 |  798 |
| `pcs_opening_time_ms ` |  997 |  997 |  997 |  997 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  41,505 |  41,505 |  41,505 |  41,505 |
| `main_cells_used     ` |  37,566,481 |  37,566,481 |  37,566,481 |  37,566,481 |
| `total_cycles        ` |  782,946 |  782,946 |  782,946 |  782,946 |
| `execute_time_ms     ` |  2,722 |  2,722 |  2,722 |  2,722 |
| `trace_gen_time_ms   ` |  381 |  381 |  381 |  381 |
| `stark_prove_excluding_trace_time_ms` |  38,402 |  38,402 |  38,402 |  38,402 |
| `main_trace_commit_time_ms` |  12,274 |  12,274 |  12,274 |  12,274 |
| `generate_perm_trace_time_ms` |  77 |  77 |  77 |  77 |
| `perm_trace_commit_time_ms` |  7,650 |  7,650 |  7,650 |  7,650 |
| `quotient_poly_compute_time_ms` |  863 |  863 |  863 |  863 |
| `quotient_poly_commit_time_ms` |  13,813 |  13,813 |  13,813 |  13,813 |
| `pcs_opening_time_ms ` |  3,706 |  3,706 |  3,706 |  3,706 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  90,895 |  90,895 |  90,895 |  90,895 |
| `main_cells_used     ` |  62,046,937 |  62,046,937 |  62,046,937 |  62,046,937 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  43,974 |  43,974 |  43,974 |  43,974 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 1,502 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 90,895 |  | 62,046,937 | 
| halo2_wrapper | 43,974 |  |  | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 1 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 2 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 61,712 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 61,712 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 61,712 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 30,856 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 41,760 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 41,760 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 41,760 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 20,880 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,610,928 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 2,610,928 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 2,610,928 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 1,305,464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 822,150 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 1 | ADD | 822,150 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 2 | ADD | 822,150 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 3 | ADD | 411,075 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 992,583 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 992,583 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 992,583 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 496,654 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 785,813 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 1 | ADD | 785,813 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 2 | ADD | 785,813 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 3 | ADD | 392,921 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 3,542,843 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 3,542,959 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 3,542,959 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 1,772,074 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,870,848 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 532,179 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 1,870,848 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 532,179 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 1,870,848 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 532,179 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 935,830 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 266,162 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 23,954 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 23,954 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 23,954 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 11,977 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 10,904 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 10,904 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 10,904 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 5,452 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 121,800 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 1 | DIV | 121,800 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 2 | DIV | 121,800 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 3 | DIV | 60,900 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 6,438 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 6,438 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 6,438 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 3,219 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 189,776 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 189,776 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 189,776 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 94,888 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 819,540 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 819,540 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 819,540 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 412,206 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 1,605,237 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 1,605,237 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 1,605,237 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 803,503 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 1,148,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 574,200 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 574,200 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 430,650 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 24,592 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 430,650 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 24,592 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 430,650 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 24,592 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 215,325 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 12,296 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 799,588 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 668,914 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 799,588 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 668,914 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 799,588 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 668,914 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 399,794 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 334,457 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 485,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 485,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 485,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 242,904 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 35,264 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 35,264 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 35,264 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 17,632 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 409,016 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 409,016 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 409,016 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 204,508 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,224,554 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 1 | MUL | 1,224,554 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 2 | MUL | 1,224,554 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 3 | MUL | 612,277 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 729,176 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 729,176 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 729,176 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 364,588 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 0 | MUL | 38,309 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 1 | MUL | 38,164 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 2 | MUL | 38,164 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 3 | MUL | 19,140 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 475,629 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 475,629 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 475,629 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 237,829 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 3,480 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 1 | MUL | 3,480 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 2 | MUL | 3,480 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 3 | MUL | 1,740 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 916,400 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 458,200 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 458,200 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 16,240 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 15,312 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 16,240 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 15,312 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 16,240 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 15,312 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 8,120 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 7,656 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 120,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 51,794 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 120,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 51,794 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 120,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 51,794 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 60,204 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 25,897 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,851,012 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 617,004 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 1,851,012 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 617,004 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 1,851,012 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 617,004 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 925,506 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 308,502 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 23,664 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 23,664 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 23,664 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 11,832 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 21,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 21,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 21,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 10,904 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 0 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 1 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 2 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 3 | SUB | 232 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 728,190 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 728,190 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 728,190 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 364,095 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 735,179 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 1 | SUB | 735,034 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 2 | SUB | 735,034 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 3 | SUB | 367,575 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 128,586 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 128,586 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 128,586 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 64,293 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 116,000 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 116,000 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 116,000 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 58,000 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 19,604 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 19,604 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 19,604 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 9,802 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 6,437,072 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 6,436,202 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 6,436,202 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 3,218,478 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 21,528 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 1 | BNE | 21,528 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 2 | BNE | 21,528 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 3 | BNE | 10,764 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 368 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 1 | BNE | 368 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 2 | BNE | 368 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 3 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 596,735 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 596,735 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 596,735 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 298,080 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 1 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 2 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 3 | BNE | 69 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 49,588 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 49,588 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 49,588 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 24,794 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 18,814 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 18,814 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 18,814 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 9,407 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 573,712 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 573,712 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 573,712 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 286,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 507,426 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 507,426 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 507,426 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 253,713 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 176,916 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 176,916 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 176,916 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 88,458 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 5,060 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 5,060 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 5,060 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 2,530 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 3,264,252 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 3,263,562 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 3,263,562 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 1,632,080 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 1,196 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 2,040,738 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 2,040,738 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 2,040,738 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 1,020,453 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 5,028,891 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 5,028,891 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 5,028,891 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 2,514,456 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 620,298 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 620,298 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 620,298 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 311,913 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 1,668,933 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 1,668,933 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 1,668,933 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 834,561 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 672,210 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 672,210 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 672,210 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 336,693 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 2,508,408 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 2,508,408 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 2,508,408 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 1,254,204 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 1,489,104 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 1,489,104 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 1,489,104 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 744,552 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,544,548 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 1,544,548 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 1,544,548 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 772,274 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 960,184 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 960,184 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 960,184 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 480,092 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 3,572 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 3,572 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 3,572 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 1,786 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 2,658,594 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 2,657,644 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 2,657,644 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 1,329,202 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 133,988 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 133,988 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 133,988 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 66,994 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 527,060 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 527,060 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 527,060 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 263,530 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 17,215,200 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 17,215,200 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 17,215,200 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 8,607,600 | 
| internal.0 | JalRangeCheck |  | 0 | JAL | 12 | 
| internal.0 | JalRangeCheck |  | 1 | JAL | 12 | 
| internal.0 | JalRangeCheck |  | 2 | JAL | 12 | 
| internal.0 | JalRangeCheck |  | 3 | JAL | 12 | 
| internal.0 | JalRangeCheck | Alloc | 0 | RANGE_CHECK | 607,284 | 
| internal.0 | JalRangeCheck | Alloc | 1 | RANGE_CHECK | 607,284 | 
| internal.0 | JalRangeCheck | Alloc | 2 | RANGE_CHECK | 607,284 | 
| internal.0 | JalRangeCheck | Alloc | 3 | RANGE_CHECK | 303,756 | 
| internal.0 | JalRangeCheck | IfEqI | 0 | JAL | 98,928 | 
| internal.0 | JalRangeCheck | IfEqI | 1 | JAL | 100,752 | 
| internal.0 | JalRangeCheck | IfEqI | 2 | JAL | 100,272 | 
| internal.0 | JalRangeCheck | IfEqI | 3 | JAL | 49,728 | 
| internal.0 | JalRangeCheck | IfNe | 0 | JAL | 72 | 
| internal.0 | JalRangeCheck | IfNe | 1 | JAL | 72 | 
| internal.0 | JalRangeCheck | IfNe | 2 | JAL | 72 | 
| internal.0 | JalRangeCheck | IfNe | 3 | JAL | 36 | 
| internal.0 | JalRangeCheck | ZipFor | 0 | JAL | 266,424 | 
| internal.0 | JalRangeCheck | ZipFor | 1 | JAL | 266,424 | 
| internal.0 | JalRangeCheck | ZipFor | 2 | JAL | 266,424 | 
| internal.0 | JalRangeCheck | ZipFor | 3 | JAL | 133,224 | 
| internal.0 | PhantomAir | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-CheckTraceHeightConstraints | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-CheckTraceHeightConstraints | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 1 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 2 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 3 | PHANTOM | 7,200 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 14,424 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 1 | PHANTOM | 14,424 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 2 | PHANTOM | 14,424 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 3 | PHANTOM | 7,212 | 
| internal.0 | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-HintOpeningValues | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-HintOpeningValues | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-HintOpeningValues | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-InitializePcsConst | 1 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-InitializePcsConst | 2 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-InitializePcsConst | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-ReadProofsFromInput | 1 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-ReadProofsFromInput | 2 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-ReadProofsFromInput | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-VerifyProofs | 1 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-VerifyProofs | 2 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-VerifyProofs | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 2 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 3 | PHANTOM | 1,200 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 7,200 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 165,600 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 165,600 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 165,600 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 82,800 | 
| internal.0 | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-rounds-context | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-rounds-context | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-rounds-context | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 254,400 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 254,400 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 254,400 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 127,200 | 
| internal.0 | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-c-build-rounds | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-c-build-rounds | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-c-build-rounds | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verifier-verify | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verifier-verify | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verifier-verify | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verify-pcs | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verify-pcs | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-d-verify-pcs | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-e-verify-constraints | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-e-verify-constraints | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-stage-e-verify-constraints | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-verify-batch | 0 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-verify-batch | 1 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-verify-batch | 2 | PHANTOM | 14,400 | 
| internal.0 | PhantomAir | CT-verify-batch | 3 | PHANTOM | 7,200 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 48,000 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 48,000 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 2 | PHANTOM | 48,000 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 3 | PHANTOM | 24,000 | 
| internal.0 | PhantomAir | CT-verify-query | 0 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-verify-query | 1 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-verify-query | 2 | PHANTOM | 2,400 | 
| internal.0 | PhantomAir | CT-verify-query | 3 | PHANTOM | 1,200 | 
| internal.0 | PhantomAir | HintBitsF | 0 | PHANTOM | 4,860 | 
| internal.0 | PhantomAir | HintBitsF | 1 | PHANTOM | 4,860 | 
| internal.0 | PhantomAir | HintBitsF | 2 | PHANTOM | 4,860 | 
| internal.0 | PhantomAir | HintBitsF | 3 | PHANTOM | 2,430 | 
| internal.0 | PhantomAir | HintFelt | 0 | PHANTOM | 141,630 | 
| internal.0 | PhantomAir | HintFelt | 1 | PHANTOM | 141,630 | 
| internal.0 | PhantomAir | HintFelt | 2 | PHANTOM | 141,630 | 
| internal.0 | PhantomAir | HintFelt | 3 | PHANTOM | 70,842 | 
| internal.0 | PhantomAir | HintInputVec | 0 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintInputVec | 1 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintInputVec | 2 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintInputVec | 3 | PHANTOM | 852 | 
| internal.0 | PhantomAir | HintLoad | 0 | PHANTOM | 38,400 | 
| internal.0 | PhantomAir | HintLoad | 1 | PHANTOM | 38,400 | 
| internal.0 | PhantomAir | HintLoad | 2 | PHANTOM | 38,400 | 
| internal.0 | PhantomAir | HintLoad | 3 | PHANTOM | 19,200 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 1,233,004 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 1,233,004 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 1,233,004 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 616,502 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 19,900,000 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 1 | VERIFY_BATCH | 19,900,000 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 2 | VERIFY_BATCH | 19,900,000 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 3 | VERIFY_BATCH | 9,950,000 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 26,307,800 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 26,188,400 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 26,188,400 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 13,094,200 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 4 | ADD | 29 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 5 | ADD | 29 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 87,696 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 87,232 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 41,760 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 41,760 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 1,918,176 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 1,903,328 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 4 | ADD | 765,310 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 5 | ADD | 765,310 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 735,875 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 733,207 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 4 | ADD | 444,541 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 5 | ADD | 442,076 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 2,477,557 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 2,465,928 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 918,836 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 269,845 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 904,162 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 264,944 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 22,330 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 22,330 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 45,704 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 45,704 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 4 | DIV | 56,028 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 5 | DIV | 54,810 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 23,838 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 23,838 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 134,328 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 134,328 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 757,596 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 757,596 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 864,635 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 860,952 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 635,796 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 635,796 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 632,142 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 632,142 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 275,210 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 31,552 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 275,210 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 31,552 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 494,508 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 426,938 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 488,418 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 422,066 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 296,032 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 291,160 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 35,264 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 35,264 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 296,264 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 295,800 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 4 | MUL | 919,822 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 5 | MUL | 914,950 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 678,832 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 678,832 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 4 | MUL | 39,034 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 5 | MUL | 38,715 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 291,537 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 291,537 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 4 | MUL | 3,480 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 5 | MUL | 3,480 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 533,484 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 533,484 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 532,266 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 532,266 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 23,200 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 22,272 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 23,200 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 22,272 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 99,296 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 65,714 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 100,514 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 65,714 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 1,000,500 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 333,500 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 1,000,500 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 333,500 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 22,736 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 22,736 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 91,408 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 91,408 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 4 | SUB | 464 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 5 | SUB | 464 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 677,846 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 677,846 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 4 | SUB | 406,812 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 5 | SUB | 404,057 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 60,610 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 59,276 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 51,156 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 49,938 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 21,344 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 21,344 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 4,308,008 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 4,287,186 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 736 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 736 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 555,887 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 555,887 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 4 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 5 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 29,624 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 29,624 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 21,574 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 21,574 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 475,548 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 475,272 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 342,378 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 338,514 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 174,432 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 174,340 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 5,060 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 5,060 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 2,300,322 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 2,284,797 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 1,196 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 1,531,194 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 1,530,060 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 3,065,475 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 3,056,613 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 490,350 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 489,300 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 1,202,565 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 1,197,105 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 361,578 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 358,008 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 1,411,074 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 1,404,270 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 777,870 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 773,307 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 960,412 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 957,106 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 503,728 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 502,132 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 14,972 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 14,972 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 2,013,240 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 1,996,406 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 97,052 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 96,900 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 266,380 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 261,592 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 6,168,960 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 6,168,960 | 
| internal.1 | JalRangeCheck |  | 4 | JAL | 12 | 
| internal.1 | JalRangeCheck |  | 5 | JAL | 12 | 
| internal.1 | JalRangeCheck | Alloc | 4 | RANGE_CHECK | 301,764 | 
| internal.1 | JalRangeCheck | Alloc | 5 | RANGE_CHECK | 296,700 | 
| internal.1 | JalRangeCheck | IfEqI | 4 | JAL | 45,840 | 
| internal.1 | JalRangeCheck | IfEqI | 5 | JAL | 44,364 | 
| internal.1 | JalRangeCheck | IfNe | 4 | JAL | 72 | 
| internal.1 | JalRangeCheck | IfNe | 5 | JAL | 72 | 
| internal.1 | JalRangeCheck | ZipFor | 4 | JAL | 163,728 | 
| internal.1 | JalRangeCheck | ZipFor | 5 | JAL | 163,212 | 
| internal.1 | PhantomAir | CT-CheckTraceHeightConstraints | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-CheckTraceHeightConstraints | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-HintOpenedValues | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-HintOpenedValues | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-HintOpeningProof | 4 | PHANTOM | 6,072 | 
| internal.1 | PhantomAir | CT-HintOpeningProof | 5 | PHANTOM | 6,072 | 
| internal.1 | PhantomAir | CT-HintOpeningValues | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-HintOpeningValues | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-cache-generator-powers | 4 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | CT-cache-generator-powers | 5 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 99,792 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 99,792 | 
| internal.1 | PhantomAir | CT-pre-compute-rounds-context | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-pre-compute-rounds-context | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-single-reduced-opening-eval | 4 | PHANTOM | 137,088 | 
| internal.1 | PhantomAir | CT-single-reduced-opening-eval | 5 | PHANTOM | 137,088 | 
| internal.1 | PhantomAir | CT-stage-c-build-rounds | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-c-build-rounds | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-d-verifier-verify | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-d-verifier-verify | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-d-verify-pcs | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-d-verify-pcs | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-e-verify-constraints | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-stage-e-verify-constraints | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-verify-batch | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-verify-batch | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-verify-batch-ext | 4 | PHANTOM | 21,168 | 
| internal.1 | PhantomAir | CT-verify-batch-ext | 5 | PHANTOM | 20,664 | 
| internal.1 | PhantomAir | CT-verify-query | 4 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | CT-verify-query | 5 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | HintBitsF | 4 | PHANTOM | 4,524 | 
| internal.1 | PhantomAir | HintBitsF | 5 | PHANTOM | 4,524 | 
| internal.1 | PhantomAir | HintFelt | 4 | PHANTOM | 65,262 | 
| internal.1 | PhantomAir | HintFelt | 5 | PHANTOM | 63,954 | 
| internal.1 | PhantomAir | HintInputVec | 4 | PHANTOM | 2,064 | 
| internal.1 | PhantomAir | HintInputVec | 5 | PHANTOM | 2,064 | 
| internal.1 | PhantomAir | HintLoad | 4 | PHANTOM | 16,632 | 
| internal.1 | PhantomAir | HintLoad | 5 | PHANTOM | 16,380 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 1,020,472 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 1,020,074 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 4 | VERIFY_BATCH | 9,829,008 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 5 | VERIFY_BATCH | 9,427,824 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 10,597,944 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 10,514,364 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 6 | ADD | 29 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 86,768 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 41,760 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 1,888,480 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 6 | ADD | 765,310 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 730,539 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 6 | ADD | 439,611 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 2,454,386 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 889,488 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 260,043 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 22,330 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 45,704 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 6 | DIV | 53,592 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 23,838 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 134,328 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 757,596 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 857,269 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 628,488 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 628,488 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 275,210 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 31,552 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 58 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 482,328 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 417,194 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 286,288 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 35,264 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 295,336 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 6 | MUL | 910,078 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 678,832 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 6 | MUL | 38,541 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 291,537 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 6 | MUL | 3,480 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 531,048 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 531,048 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 23,200 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 22,272 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 58 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 101,732 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 65,714 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 1,000,500 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 333,500 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 22,736 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 91,408 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 6 | SUB | 464 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 677,846 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 6 | SUB | 401,447 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 57,942 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 48,720 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 21,344 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 4,270,134 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 10,856 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 736 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 555,887 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 6 | BNE | 161 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 29,624 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 21,574 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 474,996 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 334,650 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 174,248 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 5,060 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 2,272,262 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 1,196 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 1,528,926 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 3,047,751 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 488,250 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 1,191,645 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 354,438 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 1,397,466 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 768,744 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 953,800 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 500,536 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 14,972 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 1,984,322 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 96,748 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 256,804 | 
| internal.2 | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 6,168,960 | 
| internal.2 | JalRangeCheck |  | 6 | JAL | 12 | 
| internal.2 | JalRangeCheck | Alloc | 6 | RANGE_CHECK | 291,636 | 
| internal.2 | JalRangeCheck | IfEqI | 6 | JAL | 45,240 | 
| internal.2 | JalRangeCheck | IfNe | 6 | JAL | 72 | 
| internal.2 | JalRangeCheck | ZipFor | 6 | JAL | 162,696 | 
| internal.2 | PhantomAir | CT-CheckTraceHeightConstraints | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-HintOpenedValues | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-HintOpeningProof | 6 | PHANTOM | 6,072 | 
| internal.2 | PhantomAir | CT-HintOpeningValues | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-InitializePcsConst | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-ReadProofsFromInput | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-VerifyProofs | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-cache-generator-powers | 6 | PHANTOM | 1,008 | 
| internal.2 | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 99,792 | 
| internal.2 | PhantomAir | CT-pre-compute-rounds-context | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 137,088 | 
| internal.2 | PhantomAir | CT-stage-c-build-rounds | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verifier-verify | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verify-pcs | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-e-verify-constraints | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-verify-batch | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 20,160 | 
| internal.2 | PhantomAir | CT-verify-query | 6 | PHANTOM | 1,008 | 
| internal.2 | PhantomAir | HintBitsF | 6 | PHANTOM | 4,524 | 
| internal.2 | PhantomAir | HintFelt | 6 | PHANTOM | 62,646 | 
| internal.2 | PhantomAir | HintInputVec | 6 | PHANTOM | 2,064 | 
| internal.2 | PhantomAir | HintLoad | 6 | PHANTOM | 16,128 | 
| internal.2 | VerifyBatchAir | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 1,019,676 | 
| internal.2 | VerifyBatchAir | VerifyBatchExt | 6 | VERIFY_BATCH | 9,026,640 | 
| internal.2 | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 10,430,784 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 26,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 25,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 25,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 25,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 25,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 25,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 26,680 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 15,776 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 13,572 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 13,572 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 13,572 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 13,572 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 13,572 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 15,196 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 1,169,744 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 1,058,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 1,058,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 1,058,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 1,058,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 1,058,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 1,143,296 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 560,280 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 1 | ADD | 438,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 2 | ADD | 438,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 3 | ADD | 438,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 4 | ADD | 438,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 5 | ADD | 438,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 6 | ADD | 535,920 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 431,781 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 342,461 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 342,461 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 342,461 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 342,461 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 342,461 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 412,525 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 1 | ADD | 371,432 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 2 | ADD | 371,432 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 3 | ADD | 371,432 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 4 | ADD | 371,432 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 5 | ADD | 371,432 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 6 | ADD | 423,632 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,697,080 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 1,406,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 1,406,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 1,406,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 1,406,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 1,406,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 1,634,817 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 997,020 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 272,513 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 976,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 265,118 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 976,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 265,118 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 976,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 265,118 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 976,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 265,118 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 976,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 265,118 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 1,006,242 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 271,150 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 16,037 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 12,557 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 12,557 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 12,557 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 12,557 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 12,557 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 15,341 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 5,916 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 5,568 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 1 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 2 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 3 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 4 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 5 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 6 | DIV | 60,900 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,509 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 2,494 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 2,494 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 2,494 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 2,494 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 2,494 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 3,306 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 95,932 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 84,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 84,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 84,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 84,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 84,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 94,076 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 555,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 436,682 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 531,570 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 884,964 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 731,989 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 731,989 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 731,989 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 731,989 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 731,989 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 854,456 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 626,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 626,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 603,200 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 603,200 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 243,049 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 13,920 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 181,569 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 10,440 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 181,569 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 10,440 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 181,569 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 10,440 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 181,569 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 10,440 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 181,569 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 10,440 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 230,753 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 13,224 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 415,570 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 347,217 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 383,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 317,927 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 383,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 317,927 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 383,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 317,927 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 383,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 317,927 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 383,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 317,927 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 409,248 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 341,359 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 243,832 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 240,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 240,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 240,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 240,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 240,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 243,136 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 10,788 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 5,104 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 5,104 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 5,104 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 5,104 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 5,104 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 9,628 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 138,620 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 87,696 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 87,696 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 87,696 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 87,696 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 87,696 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 127,136 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 744,836 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 1 | MUL | 635,506 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 2 | MUL | 635,506 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 3 | MUL | 635,506 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 4 | MUL | 635,506 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 5 | MUL | 635,506 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 6 | MUL | 722,970 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 496,799 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 388,774 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 388,774 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 388,774 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 388,774 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 388,774 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 475,194 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 0 | MUL | 21,025 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 1 | MUL | 14,413 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 2 | MUL | 14,413 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 3 | MUL | 14,413 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 4 | MUL | 14,413 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 5 | MUL | 14,413 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 6 | MUL | 19,633 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 271,266 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 210,221 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 210,221 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 210,221 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 210,221 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 210,221 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 259,057 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 2,088 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 1 | MUL | 1,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 2 | MUL | 1,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 3 | MUL | 1,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 4 | MUL | 1,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 5 | MUL | 1,508 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 6 | MUL | 1,972 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 510,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 394,400 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 487,200 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 487,200 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 8,932 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 8,468 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 6,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 6,148 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 6,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 6,148 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 6,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 6,148 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 6,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 6,148 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 6,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 6,148 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 14,732 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 14,268 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 63,916 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 29,087 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 55,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 21,692 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 55,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 21,692 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 55,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 21,692 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 55,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 21,692 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 55,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 21,692 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 62,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 27,608 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,057,746 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 352,582 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 795,876 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 265,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 795,876 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 265,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 795,876 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 265,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 795,876 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 265,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 795,876 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 265,292 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 1,005,372 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 335,124 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 7,656 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 4,176 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 7,308 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 11,832 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 8,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 8,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 8,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 8,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 8,352 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 11,136 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 496,248 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 388,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 388,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 388,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 388,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 388,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 474,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 405,217 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 1 | SUB | 341,765 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 2 | SUB | 341,765 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 3 | SUB | 341,765 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 4 | SUB | 341,765 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 5 | SUB | 341,765 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 6 | SUB | 392,457 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 64,322 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 64,177 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 64,177 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 64,177 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 64,177 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 64,177 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 64,293 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 58,000 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 14,181 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 10,411 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 10,411 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 10,411 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 10,411 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 10,411 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 13,427 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 3,511,987 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 2,826,195 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 2,826,195 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 2,826,195 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 2,826,195 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 2,826,195 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 3,373,773 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 10,948 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 1 | BNE | 10,488 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 2 | BNE | 10,488 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 3 | BNE | 10,488 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 10,488 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 10,488 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 10,856 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 1 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 2 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 3 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 184 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 406,272 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 317,952 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 317,952 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 317,952 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 317,952 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 317,952 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 388,792 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 30,245 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 28,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 28,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 28,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 28,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 28,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 29,900 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 11,638 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 8,763 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 8,763 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 8,763 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 8,763 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 8,763 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 11,086 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 232,944 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 165,117 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 165,117 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 165,117 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 165,117 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 165,117 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 218,109 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 264,454 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 246,514 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 246,514 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 246,514 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 246,514 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 246,514 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 260,866 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 137,908 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 102,258 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 102,258 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 102,258 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 102,258 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 102,258 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 130,778 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,622 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 1,932 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 1,932 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 1,932 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 1,932 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 1,932 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 2,484 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 1,698,527 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 1,399,343 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 1,399,343 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 1,399,343 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 1,399,343 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 1,399,343 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 1,637,853 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 972 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 972 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 1,124,466 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 854,406 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 854,406 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 854,406 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 854,406 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 854,406 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 1,083,054 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 2,919,861 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 2,385,831 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 2,385,831 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 2,385,831 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 2,385,831 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 2,385,831 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 2,813,076 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 261,933 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 198,303 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 198,303 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 198,303 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 198,303 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 198,303 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 261,639 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 898,212 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 756,252 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 756,252 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 756,252 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 756,252 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 756,252 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 873,369 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 362,103 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 351,078 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 351,078 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 351,078 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 351,078 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 351,078 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 359,898 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,351,755 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 1,087,776 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 1,087,776 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 1,087,776 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 1,087,776 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 1,087,776 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 1,298,592 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 793,557 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 684,612 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 684,612 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 684,612 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 684,612 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 684,612 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 771,768 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 773,186 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 612,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 612,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 612,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 612,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 612,978 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 740,164 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 537,244 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 422,864 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 422,864 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 422,864 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 422,864 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 422,864 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 514,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 1,938 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 1,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 1,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 1,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 1,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 1,368 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 1,824 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,386,126 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 1,120,582 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 1,120,582 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 1,120,582 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 1,120,582 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 1,120,582 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 1,332,926 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 45,410 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 28,728 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 28,728 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 28,728 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 28,728 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 28,728 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 41,648 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 253,498 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 247,228 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 247,228 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 247,228 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 247,228 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 247,228 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 252,130 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,101,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 4,968,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 4,968,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 4,968,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 4,968,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 4,968,000 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 6,636,600 | 
| leaf | JalRangeCheck |  | 0 | JAL | 12 | 
| leaf | JalRangeCheck |  | 1 | JAL | 12 | 
| leaf | JalRangeCheck |  | 2 | JAL | 12 | 
| leaf | JalRangeCheck |  | 3 | JAL | 12 | 
| leaf | JalRangeCheck |  | 4 | JAL | 12 | 
| leaf | JalRangeCheck |  | 5 | JAL | 12 | 
| leaf | JalRangeCheck |  | 6 | JAL | 12 | 
| leaf | JalRangeCheck | Alloc | 0 | RANGE_CHECK | 319,044 | 
| leaf | JalRangeCheck | Alloc | 1 | RANGE_CHECK | 311,784 | 
| leaf | JalRangeCheck | Alloc | 2 | RANGE_CHECK | 311,784 | 
| leaf | JalRangeCheck | Alloc | 3 | RANGE_CHECK | 311,784 | 
| leaf | JalRangeCheck | Alloc | 4 | RANGE_CHECK | 311,784 | 
| leaf | JalRangeCheck | Alloc | 5 | RANGE_CHECK | 311,784 | 
| leaf | JalRangeCheck | Alloc | 6 | RANGE_CHECK | 320,388 | 
| leaf | JalRangeCheck | IfEqI | 0 | JAL | 50,688 | 
| leaf | JalRangeCheck | IfEqI | 1 | JAL | 49,824 | 
| leaf | JalRangeCheck | IfEqI | 2 | JAL | 50,616 | 
| leaf | JalRangeCheck | IfEqI | 3 | JAL | 49,536 | 
| leaf | JalRangeCheck | IfEqI | 4 | JAL | 51,204 | 
| leaf | JalRangeCheck | IfEqI | 5 | JAL | 49,716 | 
| leaf | JalRangeCheck | IfEqI | 6 | JAL | 50,364 | 
| leaf | JalRangeCheck | IfNe | 0 | JAL | 36 | 
| leaf | JalRangeCheck | IfNe | 1 | JAL | 24 | 
| leaf | JalRangeCheck | IfNe | 2 | JAL | 24 | 
| leaf | JalRangeCheck | IfNe | 3 | JAL | 24 | 
| leaf | JalRangeCheck | IfNe | 4 | JAL | 24 | 
| leaf | JalRangeCheck | IfNe | 5 | JAL | 24 | 
| leaf | JalRangeCheck | IfNe | 6 | JAL | 24 | 
| leaf | JalRangeCheck | ZipFor | 0 | JAL | 148,200 | 
| leaf | JalRangeCheck | ZipFor | 1 | JAL | 121,500 | 
| leaf | JalRangeCheck | ZipFor | 2 | JAL | 121,500 | 
| leaf | JalRangeCheck | ZipFor | 3 | JAL | 121,500 | 
| leaf | JalRangeCheck | ZipFor | 4 | JAL | 121,500 | 
| leaf | JalRangeCheck | ZipFor | 5 | JAL | 121,500 | 
| leaf | JalRangeCheck | ZipFor | 6 | JAL | 142,872 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-CheckTraceHeightConstraints | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 2 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 3 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 4 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 5 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpenedValues | 6 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 1 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 2 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 3 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 4 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 5 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningProof | 6 | PHANTOM | 9,612 | 
| leaf | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpeningValues | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-InitializePcsConst | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ReadProofsFromInput | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-VerifyProofs | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 2 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 3 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 4 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 5 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-cache-generator-powers | 6 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 93,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 69,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 69,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 69,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 69,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 69,600 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 88,800 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-rounds-context | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 145,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 109,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 109,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 109,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 4 | PHANTOM | 109,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 5 | PHANTOM | 109,200 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 138,000 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-c-build-rounds | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verifier-verify | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-d-verify-pcs | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-stage-e-verify-constraints | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 1 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 2 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 3 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 4 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 5 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch | 6 | PHANTOM | 9,600 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 2 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 3 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 4 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 5 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 24,000 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 1 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 2 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 3 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 4 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 5 | PHANTOM | 1,200 | 
| leaf | PhantomAir | CT-verify-query | 6 | PHANTOM | 1,200 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 3,312 | 
| leaf | PhantomAir | HintBitsF | 1 | PHANTOM | 2,592 | 
| leaf | PhantomAir | HintBitsF | 2 | PHANTOM | 2,592 | 
| leaf | PhantomAir | HintBitsF | 3 | PHANTOM | 2,592 | 
| leaf | PhantomAir | HintBitsF | 4 | PHANTOM | 2,592 | 
| leaf | PhantomAir | HintBitsF | 5 | PHANTOM | 2,592 | 
| leaf | PhantomAir | HintBitsF | 6 | PHANTOM | 3,168 | 
| leaf | PhantomAir | HintFelt | 0 | PHANTOM | 73,344 | 
| leaf | PhantomAir | HintFelt | 1 | PHANTOM | 73,014 | 
| leaf | PhantomAir | HintFelt | 2 | PHANTOM | 73,014 | 
| leaf | PhantomAir | HintFelt | 3 | PHANTOM | 73,014 | 
| leaf | PhantomAir | HintFelt | 4 | PHANTOM | 73,014 | 
| leaf | PhantomAir | HintFelt | 5 | PHANTOM | 73,014 | 
| leaf | PhantomAir | HintFelt | 6 | PHANTOM | 74,628 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 966 | 
| leaf | PhantomAir | HintInputVec | 1 | PHANTOM | 726 | 
| leaf | PhantomAir | HintInputVec | 2 | PHANTOM | 726 | 
| leaf | PhantomAir | HintInputVec | 3 | PHANTOM | 726 | 
| leaf | PhantomAir | HintInputVec | 4 | PHANTOM | 726 | 
| leaf | PhantomAir | HintInputVec | 5 | PHANTOM | 726 | 
| leaf | PhantomAir | HintInputVec | 6 | PHANTOM | 918 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 1 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 2 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 3 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 4 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 5 | PHANTOM | 21,600 | 
| leaf | PhantomAir | HintLoad | 6 | PHANTOM | 21,600 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 10,746 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 499,888 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 353,026 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 353,026 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 353,026 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 353,026 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 353,026 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 467,650 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 1 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 2 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 3 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 4 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 5 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 6 | VERIFY_BATCH | 9,950,000 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 14,208,600 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 11,661,400 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 11,661,400 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 11,661,400 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 11,661,400 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 11,661,400 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 13,452,400 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 43,384 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 20,880 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 944,240 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 382,655 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 365,632 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 219,849 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,227,976 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 445,498 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 130,471 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 11,165 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,852 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 26,796 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,919 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 67,164 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 381,495 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 429,867 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 314,244 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 314,244 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 137,605 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 15,776 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 241,164 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 208,597 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 143,144 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 17,632 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 147,668 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 455,039 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 339,416 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulV | 0 | MUL | 19,256 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 145,783 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 1,740 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 265,524 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 265,524 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 11,600 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 11,136 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 50,866 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 32,857 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 500,250 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 166,750 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 11,368 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 45,704 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 0 | SUB | 232 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 338,923 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 200,709 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 28,971 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 24,360 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 10,672 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 2,135,676 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,428 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 368 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 277,840 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 115 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 14,812 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 10,810 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 237,498 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 167,325 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 87,124 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,530 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 1,136,614 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,056 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 767,235 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 1,523,907 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 249,585 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 596,442 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 177,807 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 698,733 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 384,372 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 476,900 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 250,268 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,486 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 991,686 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 48,374 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 128,402 | 
| root | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 3,084,480 | 
| root | JalRangeCheck |  | 0 | JAL | 12 | 
| root | JalRangeCheck | Alloc | 0 | RANGE_CHECK | 146,160 | 
| root | JalRangeCheck | IfEqI | 0 | JAL | 22,008 | 
| root | JalRangeCheck | IfNe | 0 | JAL | 36 | 
| root | JalRangeCheck | ZipFor | 0 | JAL | 81,372 | 
| root | PhantomAir | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-ExtractPublicValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 3,036 | 
| root | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 504 | 
| root | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 49,896 | 
| root | PhantomAir | CT-pre-compute-rounds-context | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 68,544 | 
| root | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-verify-batch | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| root | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| root | PhantomAir | HintBitsF | 0 | PHANTOM | 2,262 | 
| root | PhantomAir | HintFelt | 0 | PHANTOM | 31,302 | 
| root | PhantomAir | HintInputVec | 0 | PHANTOM | 1,038 | 
| root | PhantomAir | HintLoad | 0 | PHANTOM | 8,064 | 
| root | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 4,776 | 
| root | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 509,838 | 
| root | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,513,320 | 
| root | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 5,198,676 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 37,746,756 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 1 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 2 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 3 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 4 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 5 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 6 | 32,726,124 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 108 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 6 | 36 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 108 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 72 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 12,931,389 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 1 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 2 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 3 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 4 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 5 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 6 | 11,211,185 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 3,029,026 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 1 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 2 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 3 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 4 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 5 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 6 | 2,626,182 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 1 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 2 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 3 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 4 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 5 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 6 | 2,626,104 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 64 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLT | 6 | 64 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 128 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 2,096,982 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 1 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 2 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 3 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 4 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 5 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 6 | 1,818,054 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 270 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 6 | 108 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 308 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 6 | 392 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 0 | 41 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADBU | 6 | 287 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 533 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 6 | 697 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREB | 6 | 410 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 697 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 6 | 943 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 140 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 6 | 100 | 
| fib_e2e | PhantomAir |  | PHANTOM | 0 | 6 | 
| fib_e2e | Rv32HintStoreAir |  | HINT_BUFFER | 0 | 64 | 
| fib_e2e | Rv32HintStoreAir |  | HINT_STOREW | 0 | 32 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 3 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 3 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<8> | 0 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 1 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 2 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.0 | AccessAdapterAir<8> | 3 | 4,096 |  | 12 | 17 | 118,784 | 
| internal.0 | FriReducedOpeningAir | 0 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 1 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 2 | 1,048,576 |  | 44 | 27 | 74,448,896 | 
| internal.0 | FriReducedOpeningAir | 3 | 524,288 |  | 44 | 27 | 37,224,448 | 
| internal.0 | JalRangeCheckAir | 0 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 1 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 2 | 131,072 |  | 16 | 12 | 3,670,016 | 
| internal.0 | JalRangeCheckAir | 3 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 0 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 1 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 2 | 131,072 |  | 160 | 398 | 73,138,176 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 3 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.0 | PhantomAir | 0 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 1 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 2 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 3 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 131,072 |  | 16 | 23 | 5,111,808 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 24 | 21 | 11,796,480 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 262,144 |  | 24 | 27 | 13,369,344 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 262,144 |  | 20 | 38 | 15,204,352 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 12 | 5 | 34 | 
| internal.0 | VolatileBoundaryAir | 0 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 1 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 2 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.0 | VolatileBoundaryAir | 3 | 131,072 |  | 12 | 12 | 3,145,728 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 4 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | AccessAdapterAir<8> | 5 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.1 | JalRangeCheckAir | 4 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.1 | JalRangeCheckAir | 5 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 4 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 5 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.1 | PhantomAir | 4 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | PhantomAir | 5 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.1 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 12 | 5 | 34 | 
| internal.1 | VolatileBoundaryAir | 4 | 131,072 |  | 12 | 12 | 3,145,728 | 
| internal.1 | VolatileBoundaryAir | 5 | 262,144 |  | 12 | 12 | 6,291,456 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.2 | AccessAdapterAir<8> | 6 | 8,192 |  | 12 | 17 | 237,568 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 44 | 27 | 18,612,224 | 
| internal.2 | JalRangeCheckAir | 6 | 65,536 |  | 16 | 12 | 1,835,008 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 6 | 65,536 |  | 160 | 398 | 36,569,088 | 
| internal.2 | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.2 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 524,288 |  | 24 | 21 | 23,592,960 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 24 | 27 | 6,684,672 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 12 | 5 | 34 | 
| internal.2 | VolatileBoundaryAir | 6 | 131,072 |  | 12 | 12 | 3,145,728 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 1 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 2 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 3 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 4 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 5 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 1 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 2 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 3 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 4 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 5 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | AccessAdapterAir<8> | 1 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 2 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 3 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 4 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 5 | 2,048 |  | 16 | 17 | 67,584 | 
| leaf | AccessAdapterAir<8> | 6 | 4,096 |  | 16 | 17 | 135,168 | 
| leaf | FriReducedOpeningAir | 0 | 524,288 |  | 84 | 27 | 58,195,968 | 
| leaf | FriReducedOpeningAir | 1 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 2 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 3 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 4 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 5 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | FriReducedOpeningAir | 6 | 262,144 |  | 84 | 27 | 29,097,984 | 
| leaf | JalRangeCheckAir | 0 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 1 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 2 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 3 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 4 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 5 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | JalRangeCheckAir | 6 | 65,536 |  | 28 | 12 | 2,621,440 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 0 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 1 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 2 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 3 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 4 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 5 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 6 | 65,536 |  | 312 | 398 | 46,530,560 | 
| leaf | PhantomAir | 0 | 131,072 |  | 12 | 6 | 2,359,296 | 
| leaf | PhantomAir | 1 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | PhantomAir | 2 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | PhantomAir | 3 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | PhantomAir | 4 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | PhantomAir | 5 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | PhantomAir | 6 | 65,536 |  | 12 | 6 | 1,179,648 | 
| leaf | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 36 | 29 | 68,157,440 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 131,072 |  | 28 | 23 | 6,684,672 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 28 | 27 | 3,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 40 | 21 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 262,144 |  | 40 | 21 | 15,990,784 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 40 | 27 | 8,781,824 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 65,536 |  | 36 | 38 | 4,849,664 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 36 | 38 | 9,699,328 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 
| leaf | VolatileBoundaryAir | 0 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 1 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 2 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 3 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 4 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 5 | 131,072 |  | 20 | 12 | 4,194,304 | 
| leaf | VolatileBoundaryAir | 6 | 131,072 |  | 20 | 12 | 4,194,304 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 8 | 13 | 2,752,512 | 
| root | AccessAdapterAir<8> | 0 | 4,096 |  | 8 | 17 | 102,400 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 24 | 27 | 6,684,672 | 
| root | JalRangeCheckAir | 0 | 32,768 |  | 12 | 12 | 786,432 | 
| root | NativePoseidon2Air<BabyBearParameters>, 7, 1, 13> | 0 | 32,768 |  | 84 | 398 | 15,794,176 | 
| root | PhantomAir | 0 | 32,768 |  | 8 | 6 | 458,752 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 524,288 |  | 12 | 29 | 21,495,808 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 131,072 |  | 12 | 23 | 4,587,520 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 12 | 22 | 2,176 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 262,144 |  | 16 | 21 | 9,699,328 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 16 | 27 | 2,818,048 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 12 | 38 | 3,276,800 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 5 | 26 | 
| root | VolatileBoundaryAir | 0 | 131,072 |  | 8 | 12 | 2,621,440 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 16 | 17 | 528 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 64 |  | 16 | 17 | 2,112 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 16 | 32 | 6,144 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 16 | 32 | 12,288 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PhantomAir | 0 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 7, 1, 13> | 6 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | ProgramAir | 0 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 1 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 2 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 3 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 4 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 5 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | ProgramAir | 6 | 8,192 |  | 8 | 10 | 147,456 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | Rv32HintStoreAir | 0 | 4 |  | 44 | 32 | 304 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 52 | 36 | 92,274,688 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 28 | 26 | 14,155,776 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 32 | 32 | 512 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 6 | 2 |  | 32 | 32 | 128 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 28 | 18 | 6,029,312 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 52 | 41 | 2,976 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 64 |  | 52 | 41 | 5,952 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 6 | 8 |  | 28 | 20 | 384 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 16 | 5 | 42 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 16 | 5 | 42 | 

| group | cell_tracker_span | simple_advice_cells | lookup_advice_cells |
| --- | --- | --- | --- |
| halo2_outer | VerifierProgram | 482,264 | 155,399 | 
| halo2_outer | VerifierProgram;CheckTraceHeightConstraints | 4,789 | 972 | 
| halo2_outer | VerifierProgram;PoseidonCell | 29,400 |  | 
| halo2_outer | VerifierProgram;stage-c-build-rounds | 18,392 | 2,528 | 
| halo2_outer | VerifierProgram;stage-c-build-rounds;PoseidonCell | 46,550 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs | 1,280,628 | 197,514 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;PoseidonCell | 3,839,150 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 42,907 | 5,094 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 68,600 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;cache-generator-powers | 65,716 | 11,200 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 7,983,472 | 331,520 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;pre-compute-rounds-context | 76,224 | 11,116 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 49,728 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 9,058,896 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 8,195,124 | 237,048 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 936,572 | 160,468 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 102,144 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 15,647,184 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 1,539,860 | 51,912 | 
| halo2_outer | VerifierProgram;stage-e-verify-constraints | 9,520,848 | 1,893,717 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 1,185,364 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 1 | 1,185,328 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 2 | 1,185,328 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 3 | 592,847 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 226,721 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 226,691 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 226,691 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 113,346 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 52 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 477,670 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 477,670 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 477,670 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 238,956 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 148,056 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 148,056 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 148,056 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 74,028 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 153,367 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 153,342 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 153,342 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 76,681 | 
| internal.0 | AccessAdapter<2> | 0 | 507,788 | 
| internal.0 | AccessAdapter<2> | 1 | 507,788 | 
| internal.0 | AccessAdapter<2> | 2 | 507,788 | 
| internal.0 | AccessAdapter<2> | 3 | 266,812 | 
| internal.0 | AccessAdapter<4> | 0 | 249,758 | 
| internal.0 | AccessAdapter<4> | 1 | 249,758 | 
| internal.0 | AccessAdapter<4> | 2 | 249,758 | 
| internal.0 | AccessAdapter<4> | 3 | 131,338 | 
| internal.0 | AccessAdapter<8> | 0 | 6,304 | 
| internal.0 | AccessAdapter<8> | 1 | 6,304 | 
| internal.0 | AccessAdapter<8> | 2 | 6,304 | 
| internal.0 | AccessAdapter<8> | 3 | 3,152 | 
| internal.0 | Boundary | 0 | 172,977 | 
| internal.0 | Boundary | 1 | 172,977 | 
| internal.0 | Boundary | 2 | 172,977 | 
| internal.0 | Boundary | 3 | 130,540 | 
| internal.0 | FriReducedOpeningAir | 0 | 637,600 | 
| internal.0 | FriReducedOpeningAir | 1 | 637,600 | 
| internal.0 | FriReducedOpeningAir | 2 | 637,600 | 
| internal.0 | FriReducedOpeningAir | 3 | 318,800 | 
| internal.0 | JalRangeCheck | 0 | 81,060 | 
| internal.0 | JalRangeCheck | 1 | 81,212 | 
| internal.0 | JalRangeCheck | 2 | 81,172 | 
| internal.0 | JalRangeCheck | 3 | 40,563 | 
| internal.0 | PhantomAir | 0 | 119,537 | 
| internal.0 | PhantomAir | 1 | 119,537 | 
| internal.0 | PhantomAir | 2 | 119,537 | 
| internal.0 | PhantomAir | 3 | 59,776 | 
| internal.0 | ProgramChip | 0 | 124,502 | 
| internal.0 | ProgramChip | 1 | 124,502 | 
| internal.0 | ProgramChip | 2 | 124,502 | 
| internal.0 | ProgramChip | 3 | 124,502 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 
| internal.0 | VerifyBatchAir | 0 | 119,198 | 
| internal.0 | VerifyBatchAir | 1 | 118,898 | 
| internal.0 | VerifyBatchAir | 2 | 118,898 | 
| internal.0 | VerifyBatchAir | 3 | 59,449 | 
| internal.0 | VmConnectorAir | 0 | 2 | 
| internal.0 | VmConnectorAir | 1 | 2 | 
| internal.0 | VmConnectorAir | 2 | 2 | 
| internal.0 | VmConnectorAir | 3 | 2 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 4 | 788,832 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 5 | 784,946 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 170,287 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 169,428 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 52 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 52 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 316,722 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 315,766 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 81,072 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 80,651 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 101,468 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 100,766 | 
| internal.1 | AccessAdapter<2> | 4 | 329,564 | 
| internal.1 | AccessAdapter<2> | 5 | 327,838 | 
| internal.1 | AccessAdapter<4> | 4 | 162,822 | 
| internal.1 | AccessAdapter<4> | 5 | 162,004 | 
| internal.1 | AccessAdapter<8> | 4 | 5,240 | 
| internal.1 | AccessAdapter<8> | 5 | 5,236 | 
| internal.1 | Boundary | 4 | 121,027 | 
| internal.1 | Boundary | 5 | 131,195 | 
| internal.1 | FriReducedOpeningAir | 4 | 228,480 | 
| internal.1 | FriReducedOpeningAir | 5 | 228,480 | 
| internal.1 | JalRangeCheck | 4 | 42,618 | 
| internal.1 | JalRangeCheck | 5 | 42,030 | 
| internal.1 | PhantomAir | 4 | 62,161 | 
| internal.1 | PhantomAir | 5 | 61,817 | 
| internal.1 | ProgramChip | 4 | 124,502 | 
| internal.1 | ProgramChip | 5 | 124,502 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 
| internal.1 | VerifyBatchAir | 4 | 53,888 | 
| internal.1 | VerifyBatchAir | 5 | 52,669 | 
| internal.1 | VmConnectorAir | 4 | 2 | 
| internal.1 | VmConnectorAir | 5 | 2 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 6 | 781,203 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 168,699 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 52 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 314,810 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 80,230 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 100,189 | 
| internal.2 | AccessAdapter<2> | 6 | 325,904 | 
| internal.2 | AccessAdapter<4> | 6 | 161,076 | 
| internal.2 | AccessAdapter<8> | 6 | 5,232 | 
| internal.2 | Boundary | 6 | 119,532 | 
| internal.2 | FriReducedOpeningAir | 6 | 228,480 | 
| internal.2 | JalRangeCheck | 6 | 41,638 | 
| internal.2 | PhantomAir | 6 | 61,473 | 
| internal.2 | ProgramChip | 6 | 124,502 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 
| internal.2 | VerifyBatchAir | 6 | 51,450 | 
| internal.2 | VmConnectorAir | 6 | 2 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 639,422 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 1 | 530,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 2 | 530,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 3 | 530,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 4 | 530,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 5 | 530,043 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 6 | 618,043 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 121,555 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 99,178 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 99,178 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 99,178 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 99,178 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 99,178 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 116,997 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 36 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 265,075 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 216,470 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 216,470 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 216,470 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 216,470 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 216,470 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 256,716 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 79,456 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 65,644 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 65,644 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 65,644 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 65,644 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 65,644 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 76,680 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 78,879 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 64,046 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 64,046 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 64,046 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 64,046 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 64,046 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 75,870 | 
| leaf | AccessAdapter<2> | 0 | 249,762 | 
| leaf | AccessAdapter<2> | 1 | 210,024 | 
| leaf | AccessAdapter<2> | 2 | 210,024 | 
| leaf | AccessAdapter<2> | 3 | 210,024 | 
| leaf | AccessAdapter<2> | 4 | 210,024 | 
| leaf | AccessAdapter<2> | 5 | 210,024 | 
| leaf | AccessAdapter<2> | 6 | 242,096 | 
| leaf | AccessAdapter<4> | 0 | 122,804 | 
| leaf | AccessAdapter<4> | 1 | 102,956 | 
| leaf | AccessAdapter<4> | 2 | 102,956 | 
| leaf | AccessAdapter<4> | 3 | 102,956 | 
| leaf | AccessAdapter<4> | 4 | 102,956 | 
| leaf | AccessAdapter<4> | 5 | 102,956 | 
| leaf | AccessAdapter<4> | 6 | 118,976 | 
| leaf | AccessAdapter<8> | 0 | 2,570 | 
| leaf | AccessAdapter<8> | 1 | 1,832 | 
| leaf | AccessAdapter<8> | 2 | 1,832 | 
| leaf | AccessAdapter<8> | 3 | 1,832 | 
| leaf | AccessAdapter<8> | 4 | 1,832 | 
| leaf | AccessAdapter<8> | 5 | 1,832 | 
| leaf | AccessAdapter<8> | 6 | 2,570 | 
| leaf | Boundary | 0 | 111,242 | 
| leaf | Boundary | 1 | 103,352 | 
| leaf | Boundary | 2 | 103,352 | 
| leaf | Boundary | 3 | 103,352 | 
| leaf | Boundary | 4 | 103,352 | 
| leaf | Boundary | 5 | 103,352 | 
| leaf | Boundary | 6 | 109,918 | 
| leaf | FriReducedOpeningAir | 0 | 263,000 | 
| leaf | FriReducedOpeningAir | 1 | 184,000 | 
| leaf | FriReducedOpeningAir | 2 | 184,000 | 
| leaf | FriReducedOpeningAir | 3 | 184,000 | 
| leaf | FriReducedOpeningAir | 4 | 184,000 | 
| leaf | FriReducedOpeningAir | 5 | 184,000 | 
| leaf | FriReducedOpeningAir | 6 | 245,800 | 
| leaf | JalRangeCheck | 0 | 43,165 | 
| leaf | JalRangeCheck | 1 | 40,262 | 
| leaf | JalRangeCheck | 2 | 40,328 | 
| leaf | JalRangeCheck | 3 | 40,238 | 
| leaf | JalRangeCheck | 4 | 40,377 | 
| leaf | JalRangeCheck | 5 | 40,253 | 
| leaf | JalRangeCheck | 6 | 42,805 | 
| leaf | PhantomAir | 0 | 67,161 | 
| leaf | PhantomAir | 1 | 56,946 | 
| leaf | PhantomAir | 2 | 56,946 | 
| leaf | PhantomAir | 3 | 56,946 | 
| leaf | PhantomAir | 4 | 56,946 | 
| leaf | PhantomAir | 5 | 56,946 | 
| leaf | PhantomAir | 6 | 65,343 | 
| leaf | ProgramChip | 0 | 72,919 | 
| leaf | ProgramChip | 1 | 72,919 | 
| leaf | ProgramChip | 2 | 72,919 | 
| leaf | ProgramChip | 3 | 72,919 | 
| leaf | ProgramChip | 4 | 72,919 | 
| leaf | ProgramChip | 5 | 72,919 | 
| leaf | ProgramChip | 6 | 72,919 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 61,956 | 
| leaf | VerifyBatchAir | 1 | 55,187 | 
| leaf | VerifyBatchAir | 2 | 55,187 | 
| leaf | VerifyBatchAir | 3 | 55,187 | 
| leaf | VerifyBatchAir | 4 | 55,187 | 
| leaf | VerifyBatchAir | 5 | 55,187 | 
| leaf | VerifyBatchAir | 6 | 60,002 | 
| leaf | VmConnectorAir | 0 | 2 | 
| leaf | VmConnectorAir | 1 | 2 | 
| leaf | VmConnectorAir | 2 | 2 | 
| leaf | VmConnectorAir | 3 | 2 | 
| leaf | VmConnectorAir | 4 | 2 | 
| leaf | VmConnectorAir | 5 | 2 | 
| leaf | VmConnectorAir | 6 | 2 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 390,841 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 84,369 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 48 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 157,856 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 40,115 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 50,082 | 
| root | AccessAdapter<2> | 0 | 174,664 | 
| root | AccessAdapter<4> | 0 | 86,394 | 
| root | AccessAdapter<8> | 0 | 2,682 | 
| root | Boundary | 0 | 96,185 | 
| root | FriReducedOpeningAir | 0 | 114,240 | 
| root | JalRangeCheck | 0 | 20,799 | 
| root | PhantomAir | 0 | 30,739 | 
| root | ProgramChip | 0 | 124,882 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 
| root | VerifyBatchAir | 0 | 25,695 | 
| root | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,048,531 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 1 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 2 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 3 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 4 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 5 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 6 | 909,060 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 349,497 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 1 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 2 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 3 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 4 | 349,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 5 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 6 | 303,005 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 233,001 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 1 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 2 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 3 | 233,001 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 4 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 5 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 6 | 202,012 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 6 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 6 | 2 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 116,514 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 1 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 2 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 3 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 4 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 5 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 6 | 101,009 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 11 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 6 | 14 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 31 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 6 | 57 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 8 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 6 | 5 | 
| fib_e2e | AccessAdapter<8> | 0 | 50 | 
| fib_e2e | AccessAdapter<8> | 1 | 14 | 
| fib_e2e | AccessAdapter<8> | 2 | 14 | 
| fib_e2e | AccessAdapter<8> | 3 | 14 | 
| fib_e2e | AccessAdapter<8> | 4 | 14 | 
| fib_e2e | AccessAdapter<8> | 5 | 14 | 
| fib_e2e | AccessAdapter<8> | 6 | 60 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 0 | 179 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 1 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 2 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 3 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 4 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 5 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 7, 1, 13> | 6 | 256 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 
| fib_e2e | Boundary | 0 | 50 | 
| fib_e2e | Boundary | 1 | 14 | 
| fib_e2e | Boundary | 2 | 14 | 
| fib_e2e | Boundary | 3 | 14 | 
| fib_e2e | Boundary | 4 | 14 | 
| fib_e2e | Boundary | 5 | 14 | 
| fib_e2e | Boundary | 6 | 60 | 
| fib_e2e | Merkle | 0 | 224 | 
| fib_e2e | Merkle | 1 | 72 | 
| fib_e2e | Merkle | 2 | 72 | 
| fib_e2e | Merkle | 3 | 72 | 
| fib_e2e | Merkle | 4 | 72 | 
| fib_e2e | Merkle | 5 | 72 | 
| fib_e2e | Merkle | 6 | 244 | 
| fib_e2e | PhantomAir | 0 | 1 | 
| fib_e2e | ProgramChip | 0 | 6,942 | 
| fib_e2e | ProgramChip | 1 | 6,942 | 
| fib_e2e | ProgramChip | 2 | 6,942 | 
| fib_e2e | ProgramChip | 3 | 6,942 | 
| fib_e2e | ProgramChip | 4 | 6,942 | 
| fib_e2e | ProgramChip | 5 | 6,942 | 
| fib_e2e | ProgramChip | 6 | 6,942 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 
| fib_e2e | Rv32HintStoreAir | 0 | 3 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 

| group | dsl_ir | idx | opcode | frequency |
| --- | --- | --- | --- | --- |
| internal.0 |  | 0 | ADD | 2 | 
| internal.0 |  | 0 | JAL | 1 | 
| internal.0 |  | 1 | ADD | 2 | 
| internal.0 |  | 1 | JAL | 1 | 
| internal.0 |  | 2 | ADD | 2 | 
| internal.0 |  | 2 | JAL | 1 | 
| internal.0 |  | 3 | ADD | 2 | 
| internal.0 |  | 3 | JAL | 1 | 
| internal.0 | AddE | 0 | FE4ADD | 40,646 | 
| internal.0 | AddE | 1 | FE4ADD | 40,646 | 
| internal.0 | AddE | 2 | FE4ADD | 40,646 | 
| internal.0 | AddE | 3 | FE4ADD | 20,323 | 
| internal.0 | AddEFFI | 0 | ADD | 2,128 | 
| internal.0 | AddEFFI | 1 | ADD | 2,128 | 
| internal.0 | AddEFFI | 2 | ADD | 2,128 | 
| internal.0 | AddEFFI | 3 | ADD | 1,064 | 
| internal.0 | AddEFI | 0 | ADD | 1,440 | 
| internal.0 | AddEFI | 1 | ADD | 1,440 | 
| internal.0 | AddEFI | 2 | ADD | 1,440 | 
| internal.0 | AddEFI | 3 | ADD | 720 | 
| internal.0 | AddEI | 0 | ADD | 90,032 | 
| internal.0 | AddEI | 1 | ADD | 90,032 | 
| internal.0 | AddEI | 2 | ADD | 90,032 | 
| internal.0 | AddEI | 3 | ADD | 45,016 | 
| internal.0 | AddF | 0 | ADD | 28,350 | 
| internal.0 | AddF | 1 | ADD | 28,350 | 
| internal.0 | AddF | 2 | ADD | 28,350 | 
| internal.0 | AddF | 3 | ADD | 14,175 | 
| internal.0 | AddFI | 0 | ADD | 34,227 | 
| internal.0 | AddFI | 1 | ADD | 34,227 | 
| internal.0 | AddFI | 2 | ADD | 34,227 | 
| internal.0 | AddFI | 3 | ADD | 17,126 | 
| internal.0 | AddV | 0 | ADD | 27,097 | 
| internal.0 | AddV | 1 | ADD | 27,097 | 
| internal.0 | AddV | 2 | ADD | 27,097 | 
| internal.0 | AddV | 3 | ADD | 13,549 | 
| internal.0 | AddVI | 0 | ADD | 122,167 | 
| internal.0 | AddVI | 1 | ADD | 122,171 | 
| internal.0 | AddVI | 2 | ADD | 122,171 | 
| internal.0 | AddVI | 3 | ADD | 61,106 | 
| internal.0 | Alloc | 0 | ADD | 64,512 | 
| internal.0 | Alloc | 0 | MUL | 18,351 | 
| internal.0 | Alloc | 0 | RANGE_CHECK | 50,607 | 
| internal.0 | Alloc | 1 | ADD | 64,512 | 
| internal.0 | Alloc | 1 | MUL | 18,351 | 
| internal.0 | Alloc | 1 | RANGE_CHECK | 50,607 | 
| internal.0 | Alloc | 2 | ADD | 64,512 | 
| internal.0 | Alloc | 2 | MUL | 18,351 | 
| internal.0 | Alloc | 2 | RANGE_CHECK | 50,607 | 
| internal.0 | Alloc | 3 | ADD | 32,270 | 
| internal.0 | Alloc | 3 | MUL | 9,178 | 
| internal.0 | Alloc | 3 | RANGE_CHECK | 25,313 | 
| internal.0 | AssertEqE | 0 | BNE | 936 | 
| internal.0 | AssertEqE | 1 | BNE | 936 | 
| internal.0 | AssertEqE | 2 | BNE | 936 | 
| internal.0 | AssertEqE | 3 | BNE | 468 | 
| internal.0 | AssertEqEI | 0 | BNE | 16 | 
| internal.0 | AssertEqEI | 1 | BNE | 16 | 
| internal.0 | AssertEqEI | 2 | BNE | 16 | 
| internal.0 | AssertEqEI | 3 | BNE | 8 | 
| internal.0 | AssertEqF | 0 | BNE | 25,945 | 
| internal.0 | AssertEqF | 1 | BNE | 25,945 | 
| internal.0 | AssertEqF | 2 | BNE | 25,945 | 
| internal.0 | AssertEqF | 3 | BNE | 12,960 | 
| internal.0 | AssertEqFI | 0 | BNE | 7 | 
| internal.0 | AssertEqFI | 1 | BNE | 7 | 
| internal.0 | AssertEqFI | 2 | BNE | 7 | 
| internal.0 | AssertEqFI | 3 | BNE | 3 | 
| internal.0 | AssertEqV | 0 | BNE | 2,156 | 
| internal.0 | AssertEqV | 1 | BNE | 2,156 | 
| internal.0 | AssertEqV | 2 | BNE | 2,156 | 
| internal.0 | AssertEqV | 3 | BNE | 1,078 | 
| internal.0 | AssertEqVI | 0 | BNE | 818 | 
| internal.0 | AssertEqVI | 1 | BNE | 818 | 
| internal.0 | AssertEqVI | 2 | BNE | 818 | 
| internal.0 | AssertEqVI | 3 | BNE | 409 | 
| internal.0 | AssertNonZero | 0 | BEQ | 1 | 
| internal.0 | AssertNonZero | 1 | BEQ | 1 | 
| internal.0 | AssertNonZero | 2 | BEQ | 1 | 
| internal.0 | AssertNonZero | 3 | BEQ | 1 | 
| internal.0 | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 4 | 
| internal.0 | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 4 | 
| internal.0 | CT-CheckTraceHeightConstraints | 2 | PHANTOM | 4 | 
| internal.0 | CT-CheckTraceHeightConstraints | 3 | PHANTOM | 2 | 
| internal.0 | CT-HintOpenedValues | 0 | PHANTOM | 2,400 | 
| internal.0 | CT-HintOpenedValues | 1 | PHANTOM | 2,400 | 
| internal.0 | CT-HintOpenedValues | 2 | PHANTOM | 2,400 | 
| internal.0 | CT-HintOpenedValues | 3 | PHANTOM | 1,200 | 
| internal.0 | CT-HintOpeningProof | 0 | PHANTOM | 2,404 | 
| internal.0 | CT-HintOpeningProof | 1 | PHANTOM | 2,404 | 
| internal.0 | CT-HintOpeningProof | 2 | PHANTOM | 2,404 | 
| internal.0 | CT-HintOpeningProof | 3 | PHANTOM | 1,202 | 
| internal.0 | CT-HintOpeningValues | 0 | PHANTOM | 4 | 
| internal.0 | CT-HintOpeningValues | 1 | PHANTOM | 4 | 
| internal.0 | CT-HintOpeningValues | 2 | PHANTOM | 4 | 
| internal.0 | CT-HintOpeningValues | 3 | PHANTOM | 2 | 
| internal.0 | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| internal.0 | CT-InitializePcsConst | 1 | PHANTOM | 2 | 
| internal.0 | CT-InitializePcsConst | 2 | PHANTOM | 2 | 
| internal.0 | CT-InitializePcsConst | 3 | PHANTOM | 2 | 
| internal.0 | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| internal.0 | CT-ReadProofsFromInput | 1 | PHANTOM | 2 | 
| internal.0 | CT-ReadProofsFromInput | 2 | PHANTOM | 2 | 
| internal.0 | CT-ReadProofsFromInput | 3 | PHANTOM | 2 | 
| internal.0 | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| internal.0 | CT-VerifyProofs | 1 | PHANTOM | 2 | 
| internal.0 | CT-VerifyProofs | 2 | PHANTOM | 2 | 
| internal.0 | CT-VerifyProofs | 3 | PHANTOM | 2 | 
| internal.0 | CT-cache-generator-powers | 0 | PHANTOM | 400 | 
| internal.0 | CT-cache-generator-powers | 1 | PHANTOM | 400 | 
| internal.0 | CT-cache-generator-powers | 2 | PHANTOM | 400 | 
| internal.0 | CT-cache-generator-powers | 3 | PHANTOM | 200 | 
| internal.0 | CT-compute-reduced-opening | 0 | PHANTOM | 2,400 | 
| internal.0 | CT-compute-reduced-opening | 1 | PHANTOM | 2,400 | 
| internal.0 | CT-compute-reduced-opening | 2 | PHANTOM | 2,400 | 
| internal.0 | CT-compute-reduced-opening | 3 | PHANTOM | 1,200 | 
| internal.0 | CT-exp-reverse-bits-len | 0 | PHANTOM | 27,600 | 
| internal.0 | CT-exp-reverse-bits-len | 1 | PHANTOM | 27,600 | 
| internal.0 | CT-exp-reverse-bits-len | 2 | PHANTOM | 27,600 | 
| internal.0 | CT-exp-reverse-bits-len | 3 | PHANTOM | 13,800 | 
| internal.0 | CT-pre-compute-rounds-context | 0 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-rounds-context | 1 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-rounds-context | 2 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-rounds-context | 3 | PHANTOM | 2 | 
| internal.0 | CT-single-reduced-opening-eval | 0 | PHANTOM | 42,400 | 
| internal.0 | CT-single-reduced-opening-eval | 1 | PHANTOM | 42,400 | 
| internal.0 | CT-single-reduced-opening-eval | 2 | PHANTOM | 42,400 | 
| internal.0 | CT-single-reduced-opening-eval | 3 | PHANTOM | 21,200 | 
| internal.0 | CT-stage-c-build-rounds | 0 | PHANTOM | 4 | 
| internal.0 | CT-stage-c-build-rounds | 1 | PHANTOM | 4 | 
| internal.0 | CT-stage-c-build-rounds | 2 | PHANTOM | 4 | 
| internal.0 | CT-stage-c-build-rounds | 3 | PHANTOM | 2 | 
| internal.0 | CT-stage-d-verifier-verify | 0 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verifier-verify | 1 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verifier-verify | 2 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verifier-verify | 3 | PHANTOM | 2 | 
| internal.0 | CT-stage-d-verify-pcs | 0 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verify-pcs | 1 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verify-pcs | 2 | PHANTOM | 4 | 
| internal.0 | CT-stage-d-verify-pcs | 3 | PHANTOM | 2 | 
| internal.0 | CT-stage-e-verify-constraints | 0 | PHANTOM | 4 | 
| internal.0 | CT-stage-e-verify-constraints | 1 | PHANTOM | 4 | 
| internal.0 | CT-stage-e-verify-constraints | 2 | PHANTOM | 4 | 
| internal.0 | CT-stage-e-verify-constraints | 3 | PHANTOM | 2 | 
| internal.0 | CT-verify-batch | 0 | PHANTOM | 2,400 | 
| internal.0 | CT-verify-batch | 1 | PHANTOM | 2,400 | 
| internal.0 | CT-verify-batch | 2 | PHANTOM | 2,400 | 
| internal.0 | CT-verify-batch | 3 | PHANTOM | 1,200 | 
| internal.0 | CT-verify-batch-ext | 0 | PHANTOM | 8,000 | 
| internal.0 | CT-verify-batch-ext | 1 | PHANTOM | 8,000 | 
| internal.0 | CT-verify-batch-ext | 2 | PHANTOM | 8,000 | 
| internal.0 | CT-verify-batch-ext | 3 | PHANTOM | 4,000 | 
| internal.0 | CT-verify-query | 0 | PHANTOM | 400 | 
| internal.0 | CT-verify-query | 1 | PHANTOM | 400 | 
| internal.0 | CT-verify-query | 2 | PHANTOM | 400 | 
| internal.0 | CT-verify-query | 3 | PHANTOM | 200 | 
| internal.0 | CastFV | 0 | ADD | 826 | 
| internal.0 | CastFV | 1 | ADD | 826 | 
| internal.0 | CastFV | 2 | ADD | 826 | 
| internal.0 | CastFV | 3 | ADD | 413 | 
| internal.0 | DivE | 0 | BBE4DIV | 25,268 | 
| internal.0 | DivE | 1 | BBE4DIV | 25,268 | 
| internal.0 | DivE | 2 | BBE4DIV | 25,268 | 
| internal.0 | DivE | 3 | BBE4DIV | 12,634 | 
| internal.0 | DivEIN | 0 | ADD | 376 | 
| internal.0 | DivEIN | 0 | BBE4DIV | 94 | 
| internal.0 | DivEIN | 1 | ADD | 376 | 
| internal.0 | DivEIN | 1 | BBE4DIV | 94 | 
| internal.0 | DivEIN | 2 | ADD | 376 | 
| internal.0 | DivEIN | 2 | BBE4DIV | 94 | 
| internal.0 | DivEIN | 3 | ADD | 188 | 
| internal.0 | DivEIN | 3 | BBE4DIV | 47 | 
| internal.0 | DivF | 0 | DIV | 4,200 | 
| internal.0 | DivF | 1 | DIV | 4,200 | 
| internal.0 | DivF | 2 | DIV | 4,200 | 
| internal.0 | DivF | 3 | DIV | 2,100 | 
| internal.0 | DivFIN | 0 | DIV | 222 | 
| internal.0 | DivFIN | 1 | DIV | 222 | 
| internal.0 | DivFIN | 2 | DIV | 222 | 
| internal.0 | DivFIN | 3 | DIV | 111 | 
| internal.0 | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 21,200 | 
| internal.0 | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 21,200 | 
| internal.0 | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 21,200 | 
| internal.0 | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 10,600 | 
| internal.0 | HintBitsF | 0 | PHANTOM | 810 | 
| internal.0 | HintBitsF | 1 | PHANTOM | 810 | 
| internal.0 | HintBitsF | 2 | PHANTOM | 810 | 
| internal.0 | HintBitsF | 3 | PHANTOM | 405 | 
| internal.0 | HintFelt | 0 | PHANTOM | 23,605 | 
| internal.0 | HintFelt | 1 | PHANTOM | 23,605 | 
| internal.0 | HintFelt | 2 | PHANTOM | 23,605 | 
| internal.0 | HintFelt | 3 | PHANTOM | 11,807 | 
| internal.0 | HintInputVec | 0 | PHANTOM | 284 | 
| internal.0 | HintInputVec | 1 | PHANTOM | 284 | 
| internal.0 | HintInputVec | 2 | PHANTOM | 284 | 
| internal.0 | HintInputVec | 3 | PHANTOM | 142 | 
| internal.0 | HintLoad | 0 | PHANTOM | 6,400 | 
| internal.0 | HintLoad | 1 | PHANTOM | 6,400 | 
| internal.0 | HintLoad | 2 | PHANTOM | 6,400 | 
| internal.0 | HintLoad | 3 | PHANTOM | 3,200 | 
| internal.0 | IfEq | 0 | BNE | 24,944 | 
| internal.0 | IfEq | 1 | BNE | 24,944 | 
| internal.0 | IfEq | 2 | BNE | 24,944 | 
| internal.0 | IfEq | 3 | BNE | 12,472 | 
| internal.0 | IfEqI | 0 | BNE | 22,062 | 
| internal.0 | IfEqI | 0 | JAL | 8,244 | 
| internal.0 | IfEqI | 1 | BNE | 22,062 | 
| internal.0 | IfEqI | 1 | JAL | 8,396 | 
| internal.0 | IfEqI | 2 | BNE | 22,062 | 
| internal.0 | IfEqI | 2 | JAL | 8,356 | 
| internal.0 | IfEqI | 3 | BNE | 11,031 | 
| internal.0 | IfEqI | 3 | JAL | 4,144 | 
| internal.0 | IfNe | 0 | BEQ | 7,692 | 
| internal.0 | IfNe | 0 | JAL | 6 | 
| internal.0 | IfNe | 1 | BEQ | 7,692 | 
| internal.0 | IfNe | 1 | JAL | 6 | 
| internal.0 | IfNe | 2 | BEQ | 7,692 | 
| internal.0 | IfNe | 2 | JAL | 6 | 
| internal.0 | IfNe | 3 | BEQ | 3,846 | 
| internal.0 | IfNe | 3 | JAL | 3 | 
| internal.0 | IfNeI | 0 | BEQ | 220 | 
| internal.0 | IfNeI | 1 | BEQ | 220 | 
| internal.0 | IfNeI | 2 | BEQ | 220 | 
| internal.0 | IfNeI | 3 | BEQ | 110 | 
| internal.0 | ImmE | 0 | ADD | 6,544 | 
| internal.0 | ImmE | 1 | ADD | 6,544 | 
| internal.0 | ImmE | 2 | ADD | 6,544 | 
| internal.0 | ImmE | 3 | ADD | 3,272 | 
| internal.0 | ImmF | 0 | ADD | 28,260 | 
| internal.0 | ImmF | 1 | ADD | 28,260 | 
| internal.0 | ImmF | 2 | ADD | 28,260 | 
| internal.0 | ImmF | 3 | ADD | 14,214 | 
| internal.0 | ImmV | 0 | ADD | 55,353 | 
| internal.0 | ImmV | 1 | ADD | 55,353 | 
| internal.0 | ImmV | 2 | ADD | 55,353 | 
| internal.0 | ImmV | 3 | ADD | 27,707 | 
| internal.0 | LoadE | 0 | ADD | 39,600 | 
| internal.0 | LoadE | 0 | LOADW | 92,904 | 
| internal.0 | LoadE | 0 | MUL | 39,600 | 
| internal.0 | LoadE | 1 | ADD | 39,600 | 
| internal.0 | LoadE | 1 | LOADW | 92,904 | 
| internal.0 | LoadE | 1 | MUL | 39,600 | 
| internal.0 | LoadE | 2 | ADD | 39,600 | 
| internal.0 | LoadE | 2 | LOADW | 92,904 | 
| internal.0 | LoadE | 2 | MUL | 39,600 | 
| internal.0 | LoadE | 3 | ADD | 19,800 | 
| internal.0 | LoadE | 3 | LOADW | 46,452 | 
| internal.0 | LoadE | 3 | MUL | 19,800 | 
| internal.0 | LoadF | 0 | ADD | 14,850 | 
| internal.0 | LoadF | 0 | LOADW | 97,178 | 
| internal.0 | LoadF | 0 | MUL | 848 | 
| internal.0 | LoadF | 1 | ADD | 14,850 | 
| internal.0 | LoadF | 1 | LOADW | 97,178 | 
| internal.0 | LoadF | 1 | MUL | 848 | 
| internal.0 | LoadF | 2 | ADD | 14,850 | 
| internal.0 | LoadF | 2 | LOADW | 97,178 | 
| internal.0 | LoadF | 2 | MUL | 848 | 
| internal.0 | LoadF | 3 | ADD | 7,425 | 
| internal.0 | LoadF | 3 | LOADW | 48,593 | 
| internal.0 | LoadF | 3 | MUL | 424 | 
| internal.0 | LoadHeapPtr | 0 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 1 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 2 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 3 | ADD | 1 | 
| internal.0 | LoadV | 0 | ADD | 27,572 | 
| internal.0 | LoadV | 0 | LOADW | 239,471 | 
| internal.0 | LoadV | 0 | MUL | 23,066 | 
| internal.0 | LoadV | 1 | ADD | 27,572 | 
| internal.0 | LoadV | 1 | LOADW | 239,471 | 
| internal.0 | LoadV | 1 | MUL | 23,066 | 
| internal.0 | LoadV | 2 | ADD | 27,572 | 
| internal.0 | LoadV | 2 | LOADW | 239,471 | 
| internal.0 | LoadV | 2 | MUL | 23,066 | 
| internal.0 | LoadV | 3 | ADD | 13,786 | 
| internal.0 | LoadV | 3 | LOADW | 119,736 | 
| internal.0 | LoadV | 3 | MUL | 11,533 | 
| internal.0 | MulE | 0 | BBE4MUL | 69,963 | 
| internal.0 | MulE | 1 | BBE4MUL | 69,938 | 
| internal.0 | MulE | 2 | BBE4MUL | 69,938 | 
| internal.0 | MulE | 3 | BBE4MUL | 34,979 | 
| internal.0 | MulEF | 0 | MUL | 16,752 | 
| internal.0 | MulEF | 1 | MUL | 16,752 | 
| internal.0 | MulEF | 2 | MUL | 16,752 | 
| internal.0 | MulEF | 3 | MUL | 8,376 | 
| internal.0 | MulEFI | 0 | MUL | 1,216 | 
| internal.0 | MulEFI | 1 | MUL | 1,216 | 
| internal.0 | MulEFI | 2 | MUL | 1,216 | 
| internal.0 | MulEFI | 3 | MUL | 608 | 
| internal.0 | MulEI | 0 | ADD | 14,104 | 
| internal.0 | MulEI | 0 | BBE4MUL | 3,526 | 
| internal.0 | MulEI | 1 | ADD | 14,104 | 
| internal.0 | MulEI | 1 | BBE4MUL | 3,526 | 
| internal.0 | MulEI | 2 | ADD | 14,104 | 
| internal.0 | MulEI | 2 | BBE4MUL | 3,526 | 
| internal.0 | MulEI | 3 | ADD | 7,052 | 
| internal.0 | MulEI | 3 | BBE4MUL | 1,763 | 
| internal.0 | MulF | 0 | MUL | 42,226 | 
| internal.0 | MulF | 1 | MUL | 42,226 | 
| internal.0 | MulF | 2 | MUL | 42,226 | 
| internal.0 | MulF | 3 | MUL | 21,113 | 
| internal.0 | MulFI | 0 | MUL | 25,144 | 
| internal.0 | MulFI | 1 | MUL | 25,144 | 
| internal.0 | MulFI | 2 | MUL | 25,144 | 
| internal.0 | MulFI | 3 | MUL | 12,572 | 
| internal.0 | MulV | 0 | MUL | 1,321 | 
| internal.0 | MulV | 1 | MUL | 1,316 | 
| internal.0 | MulV | 2 | MUL | 1,316 | 
| internal.0 | MulV | 3 | MUL | 660 | 
| internal.0 | MulVI | 0 | MUL | 16,401 | 
| internal.0 | MulVI | 1 | MUL | 16,401 | 
| internal.0 | MulVI | 2 | MUL | 16,401 | 
| internal.0 | MulVI | 3 | MUL | 8,201 | 
| internal.0 | NegE | 0 | MUL | 120 | 
| internal.0 | NegE | 1 | MUL | 120 | 
| internal.0 | NegE | 2 | MUL | 120 | 
| internal.0 | NegE | 3 | MUL | 60 | 
| internal.0 | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 3,098 | 
| internal.0 | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 3,098 | 
| internal.0 | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 3,098 | 
| internal.0 | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 1,549 | 
| internal.0 | Publish | 0 | PUBLISH | 52 | 
| internal.0 | Publish | 1 | PUBLISH | 52 | 
| internal.0 | Publish | 2 | PUBLISH | 52 | 
| internal.0 | Publish | 3 | PUBLISH | 52 | 
| internal.0 | StoreE | 0 | ADD | 31,600 | 
| internal.0 | StoreE | 0 | MUL | 31,600 | 
| internal.0 | StoreE | 0 | STOREW | 55,152 | 
| internal.0 | StoreE | 1 | ADD | 31,600 | 
| internal.0 | StoreE | 1 | MUL | 31,600 | 
| internal.0 | StoreE | 1 | STOREW | 55,152 | 
| internal.0 | StoreE | 2 | ADD | 31,600 | 
| internal.0 | StoreE | 2 | MUL | 31,600 | 
| internal.0 | StoreE | 2 | STOREW | 55,152 | 
| internal.0 | StoreE | 3 | ADD | 15,800 | 
| internal.0 | StoreE | 3 | MUL | 15,800 | 
| internal.0 | StoreE | 3 | STOREW | 27,576 | 
| internal.0 | StoreF | 0 | ADD | 560 | 
| internal.0 | StoreF | 0 | MUL | 528 | 
| internal.0 | StoreF | 0 | STOREW | 29,538 | 
| internal.0 | StoreF | 1 | ADD | 560 | 
| internal.0 | StoreF | 1 | MUL | 528 | 
| internal.0 | StoreF | 1 | STOREW | 29,538 | 
| internal.0 | StoreF | 2 | ADD | 560 | 
| internal.0 | StoreF | 2 | MUL | 528 | 
| internal.0 | StoreF | 2 | STOREW | 29,538 | 
| internal.0 | StoreF | 3 | ADD | 280 | 
| internal.0 | StoreF | 3 | MUL | 264 | 
| internal.0 | StoreF | 3 | STOREW | 14,853 | 
| internal.0 | StoreHeapPtr | 0 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 1 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 2 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 3 | ADD | 1 | 
| internal.0 | StoreHintWord | 0 | HINT_STOREW | 79,473 | 
| internal.0 | StoreHintWord | 1 | HINT_STOREW | 79,473 | 
| internal.0 | StoreHintWord | 2 | HINT_STOREW | 79,473 | 
| internal.0 | StoreHintWord | 3 | HINT_STOREW | 39,741 | 
| internal.0 | StoreV | 0 | ADD | 4,152 | 
| internal.0 | StoreV | 0 | MUL | 1,786 | 
| internal.0 | StoreV | 0 | STOREW | 32,010 | 
| internal.0 | StoreV | 1 | ADD | 4,152 | 
| internal.0 | StoreV | 1 | MUL | 1,786 | 
| internal.0 | StoreV | 1 | STOREW | 32,010 | 
| internal.0 | StoreV | 2 | ADD | 4,152 | 
| internal.0 | StoreV | 2 | MUL | 1,786 | 
| internal.0 | StoreV | 2 | STOREW | 32,010 | 
| internal.0 | StoreV | 3 | ADD | 2,076 | 
| internal.0 | StoreV | 3 | MUL | 893 | 
| internal.0 | StoreV | 3 | STOREW | 16,033 | 
| internal.0 | SubE | 0 | FE4SUB | 13,870 | 
| internal.0 | SubE | 1 | FE4SUB | 13,870 | 
| internal.0 | SubE | 2 | FE4SUB | 13,870 | 
| internal.0 | SubE | 3 | FE4SUB | 6,935 | 
| internal.0 | SubEF | 0 | ADD | 63,828 | 
| internal.0 | SubEF | 0 | SUB | 21,276 | 
| internal.0 | SubEF | 1 | ADD | 63,828 | 
| internal.0 | SubEF | 1 | SUB | 21,276 | 
| internal.0 | SubEF | 2 | ADD | 63,828 | 
| internal.0 | SubEF | 2 | SUB | 21,276 | 
| internal.0 | SubEF | 3 | ADD | 31,914 | 
| internal.0 | SubEF | 3 | SUB | 10,638 | 
| internal.0 | SubEFI | 0 | ADD | 816 | 
| internal.0 | SubEFI | 1 | ADD | 816 | 
| internal.0 | SubEFI | 2 | ADD | 816 | 
| internal.0 | SubEFI | 3 | ADD | 408 | 
| internal.0 | SubEI | 0 | ADD | 752 | 
| internal.0 | SubEI | 1 | ADD | 752 | 
| internal.0 | SubEI | 2 | ADD | 752 | 
| internal.0 | SubEI | 3 | ADD | 376 | 
| internal.0 | SubF | 0 | SUB | 16 | 
| internal.0 | SubF | 1 | SUB | 16 | 
| internal.0 | SubF | 2 | SUB | 16 | 
| internal.0 | SubF | 3 | SUB | 8 | 
| internal.0 | SubFI | 0 | SUB | 25,110 | 
| internal.0 | SubFI | 1 | SUB | 25,110 | 
| internal.0 | SubFI | 2 | SUB | 25,110 | 
| internal.0 | SubFI | 3 | SUB | 12,555 | 
| internal.0 | SubV | 0 | SUB | 25,351 | 
| internal.0 | SubV | 1 | SUB | 25,346 | 
| internal.0 | SubV | 2 | SUB | 25,346 | 
| internal.0 | SubV | 3 | SUB | 12,675 | 
| internal.0 | SubVI | 0 | SUB | 4,434 | 
| internal.0 | SubVI | 1 | SUB | 4,434 | 
| internal.0 | SubVI | 2 | SUB | 4,434 | 
| internal.0 | SubVI | 3 | SUB | 2,217 | 
| internal.0 | SubVIN | 0 | SUB | 4,000 | 
| internal.0 | SubVIN | 1 | SUB | 4,000 | 
| internal.0 | SubVIN | 2 | SUB | 4,000 | 
| internal.0 | SubVIN | 3 | SUB | 2,000 | 
| internal.0 | UnsafeCastVF | 0 | ADD | 676 | 
| internal.0 | UnsafeCastVF | 1 | ADD | 676 | 
| internal.0 | UnsafeCastVF | 2 | ADD | 676 | 
| internal.0 | UnsafeCastVF | 3 | ADD | 338 | 
| internal.0 | VerifyBatchExt | 0 | VERIFY_BATCH | 4,000 | 
| internal.0 | VerifyBatchExt | 1 | VERIFY_BATCH | 4,000 | 
| internal.0 | VerifyBatchExt | 2 | VERIFY_BATCH | 4,000 | 
| internal.0 | VerifyBatchExt | 3 | VERIFY_BATCH | 2,000 | 
| internal.0 | VerifyBatchFelt | 0 | VERIFY_BATCH | 1,200 | 
| internal.0 | VerifyBatchFelt | 1 | VERIFY_BATCH | 1,200 | 
| internal.0 | VerifyBatchFelt | 2 | VERIFY_BATCH | 1,200 | 
| internal.0 | VerifyBatchFelt | 3 | VERIFY_BATCH | 600 | 
| internal.0 | ZipFor | 0 | ADD | 221,968 | 
| internal.0 | ZipFor | 0 | BNE | 141,924 | 
| internal.0 | ZipFor | 0 | JAL | 22,202 | 
| internal.0 | ZipFor | 1 | ADD | 221,938 | 
| internal.0 | ZipFor | 1 | BNE | 141,894 | 
| internal.0 | ZipFor | 1 | JAL | 22,202 | 
| internal.0 | ZipFor | 2 | ADD | 221,938 | 
| internal.0 | ZipFor | 2 | BNE | 141,894 | 
| internal.0 | ZipFor | 2 | JAL | 22,202 | 
| internal.0 | ZipFor | 3 | ADD | 110,982 | 
| internal.0 | ZipFor | 3 | BNE | 70,960 | 
| internal.0 | ZipFor | 3 | JAL | 11,102 | 
| internal.1 |  | 4 | ADD | 2 | 
| internal.1 |  | 4 | JAL | 1 | 
| internal.1 |  | 5 | ADD | 2 | 
| internal.1 |  | 5 | JAL | 1 | 
| internal.1 | AddE | 4 | FE4ADD | 25,274 | 
| internal.1 | AddE | 5 | FE4ADD | 25,187 | 
| internal.1 | AddEFFI | 4 | ADD | 3,024 | 
| internal.1 | AddEFFI | 5 | ADD | 3,008 | 
| internal.1 | AddEFI | 4 | ADD | 1,440 | 
| internal.1 | AddEFI | 5 | ADD | 1,440 | 
| internal.1 | AddEI | 4 | ADD | 66,144 | 
| internal.1 | AddEI | 5 | ADD | 65,632 | 
| internal.1 | AddF | 4 | ADD | 26,390 | 
| internal.1 | AddF | 5 | ADD | 26,390 | 
| internal.1 | AddFI | 4 | ADD | 25,375 | 
| internal.1 | AddFI | 5 | ADD | 25,283 | 
| internal.1 | AddV | 4 | ADD | 15,329 | 
| internal.1 | AddV | 5 | ADD | 15,244 | 
| internal.1 | AddVI | 4 | ADD | 85,433 | 
| internal.1 | AddVI | 5 | ADD | 85,032 | 
| internal.1 | Alloc | 4 | ADD | 31,684 | 
| internal.1 | Alloc | 4 | MUL | 9,305 | 
| internal.1 | Alloc | 4 | RANGE_CHECK | 25,147 | 
| internal.1 | Alloc | 5 | ADD | 31,178 | 
| internal.1 | Alloc | 5 | MUL | 9,136 | 
| internal.1 | Alloc | 5 | RANGE_CHECK | 24,725 | 
| internal.1 | AssertEqE | 4 | BNE | 472 | 
| internal.1 | AssertEqE | 5 | BNE | 472 | 
| internal.1 | AssertEqEI | 4 | BNE | 32 | 
| internal.1 | AssertEqEI | 5 | BNE | 32 | 
| internal.1 | AssertEqF | 4 | BNE | 24,169 | 
| internal.1 | AssertEqF | 5 | BNE | 24,169 | 
| internal.1 | AssertEqFI | 4 | BNE | 7 | 
| internal.1 | AssertEqFI | 5 | BNE | 7 | 
| internal.1 | AssertEqV | 4 | BNE | 1,288 | 
| internal.1 | AssertEqV | 5 | BNE | 1,288 | 
| internal.1 | AssertEqVI | 4 | BNE | 938 | 
| internal.1 | AssertEqVI | 5 | BNE | 938 | 
| internal.1 | AssertNonZero | 4 | BEQ | 1 | 
| internal.1 | AssertNonZero | 5 | BEQ | 1 | 
| internal.1 | CT-CheckTraceHeightConstraints | 4 | PHANTOM | 4 | 
| internal.1 | CT-CheckTraceHeightConstraints | 5 | PHANTOM | 4 | 
| internal.1 | CT-HintOpenedValues | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-HintOpenedValues | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-HintOpeningProof | 4 | PHANTOM | 1,012 | 
| internal.1 | CT-HintOpeningProof | 5 | PHANTOM | 1,012 | 
| internal.1 | CT-HintOpeningValues | 4 | PHANTOM | 4 | 
| internal.1 | CT-HintOpeningValues | 5 | PHANTOM | 4 | 
| internal.1 | CT-InitializePcsConst | 4 | PHANTOM | 2 | 
| internal.1 | CT-InitializePcsConst | 5 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 4 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 5 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 4 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 5 | PHANTOM | 2 | 
| internal.1 | CT-cache-generator-powers | 4 | PHANTOM | 168 | 
| internal.1 | CT-cache-generator-powers | 5 | PHANTOM | 168 | 
| internal.1 | CT-compute-reduced-opening | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-exp-reverse-bits-len | 4 | PHANTOM | 16,632 | 
| internal.1 | CT-exp-reverse-bits-len | 5 | PHANTOM | 16,632 | 
| internal.1 | CT-pre-compute-rounds-context | 4 | PHANTOM | 4 | 
| internal.1 | CT-pre-compute-rounds-context | 5 | PHANTOM | 4 | 
| internal.1 | CT-single-reduced-opening-eval | 4 | PHANTOM | 22,848 | 
| internal.1 | CT-single-reduced-opening-eval | 5 | PHANTOM | 22,848 | 
| internal.1 | CT-stage-c-build-rounds | 4 | PHANTOM | 4 | 
| internal.1 | CT-stage-c-build-rounds | 5 | PHANTOM | 4 | 
| internal.1 | CT-stage-d-verifier-verify | 4 | PHANTOM | 4 | 
| internal.1 | CT-stage-d-verifier-verify | 5 | PHANTOM | 4 | 
| internal.1 | CT-stage-d-verify-pcs | 4 | PHANTOM | 4 | 
| internal.1 | CT-stage-d-verify-pcs | 5 | PHANTOM | 4 | 
| internal.1 | CT-stage-e-verify-constraints | 4 | PHANTOM | 4 | 
| internal.1 | CT-stage-e-verify-constraints | 5 | PHANTOM | 4 | 
| internal.1 | CT-verify-batch | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-verify-batch | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-verify-batch-ext | 4 | PHANTOM | 3,528 | 
| internal.1 | CT-verify-batch-ext | 5 | PHANTOM | 3,444 | 
| internal.1 | CT-verify-query | 4 | PHANTOM | 168 | 
| internal.1 | CT-verify-query | 5 | PHANTOM | 168 | 
| internal.1 | CastFV | 4 | ADD | 770 | 
| internal.1 | CastFV | 5 | ADD | 770 | 
| internal.1 | DivE | 4 | BBE4DIV | 13,256 | 
| internal.1 | DivE | 5 | BBE4DIV | 13,214 | 
| internal.1 | DivEIN | 4 | ADD | 1,576 | 
| internal.1 | DivEIN | 4 | BBE4DIV | 394 | 
| internal.1 | DivEIN | 5 | ADD | 1,576 | 
| internal.1 | DivEIN | 5 | BBE4DIV | 394 | 
| internal.1 | DivF | 4 | DIV | 1,932 | 
| internal.1 | DivF | 5 | DIV | 1,890 | 
| internal.1 | DivFIN | 4 | DIV | 822 | 
| internal.1 | DivFIN | 5 | DIV | 822 | 
| internal.1 | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 11,424 | 
| internal.1 | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 11,424 | 
| internal.1 | HintBitsF | 4 | PHANTOM | 754 | 
| internal.1 | HintBitsF | 5 | PHANTOM | 754 | 
| internal.1 | HintFelt | 4 | PHANTOM | 10,877 | 
| internal.1 | HintFelt | 5 | PHANTOM | 10,659 | 
| internal.1 | HintInputVec | 4 | PHANTOM | 344 | 
| internal.1 | HintInputVec | 5 | PHANTOM | 344 | 
| internal.1 | HintLoad | 4 | PHANTOM | 2,772 | 
| internal.1 | HintLoad | 5 | PHANTOM | 2,730 | 
| internal.1 | IfEq | 4 | BNE | 20,676 | 
| internal.1 | IfEq | 5 | BNE | 20,664 | 
| internal.1 | IfEqI | 4 | BNE | 14,886 | 
| internal.1 | IfEqI | 4 | JAL | 3,820 | 
| internal.1 | IfEqI | 5 | BNE | 14,718 | 
| internal.1 | IfEqI | 5 | JAL | 3,697 | 
| internal.1 | IfNe | 4 | BEQ | 7,584 | 
| internal.1 | IfNe | 4 | JAL | 6 | 
| internal.1 | IfNe | 5 | BEQ | 7,580 | 
| internal.1 | IfNe | 5 | JAL | 6 | 
| internal.1 | IfNeI | 4 | BEQ | 220 | 
| internal.1 | IfNeI | 5 | BEQ | 220 | 
| internal.1 | ImmE | 4 | ADD | 4,632 | 
| internal.1 | ImmE | 5 | ADD | 4,632 | 
| internal.1 | ImmF | 4 | ADD | 26,124 | 
| internal.1 | ImmF | 5 | ADD | 26,124 | 
| internal.1 | ImmV | 4 | ADD | 29,815 | 
| internal.1 | ImmV | 5 | ADD | 29,688 | 
| internal.1 | LoadE | 4 | ADD | 21,924 | 
| internal.1 | LoadE | 4 | LOADW | 52,262 | 
| internal.1 | LoadE | 4 | MUL | 21,924 | 
| internal.1 | LoadE | 5 | ADD | 21,798 | 
| internal.1 | LoadE | 5 | LOADW | 52,010 | 
| internal.1 | LoadE | 5 | MUL | 21,798 | 
| internal.1 | LoadF | 4 | ADD | 9,490 | 
| internal.1 | LoadF | 4 | LOADW | 72,914 | 
| internal.1 | LoadF | 4 | MUL | 1,088 | 
| internal.1 | LoadF | 5 | ADD | 9,490 | 
| internal.1 | LoadF | 5 | LOADW | 72,860 | 
| internal.1 | LoadF | 5 | MUL | 1,088 | 
| internal.1 | LoadHeapPtr | 4 | ADD | 2 | 
| internal.1 | LoadHeapPtr | 5 | ADD | 2 | 
| internal.1 | LoadV | 4 | ADD | 17,052 | 
| internal.1 | LoadV | 4 | LOADW | 145,975 | 
| internal.1 | LoadV | 4 | MUL | 14,722 | 
| internal.1 | LoadV | 5 | ADD | 16,842 | 
| internal.1 | LoadV | 5 | LOADW | 145,553 | 
| internal.1 | LoadV | 5 | MUL | 14,554 | 
| internal.1 | MulE | 4 | BBE4MUL | 52,980 | 
| internal.1 | MulE | 5 | BBE4MUL | 52,537 | 
| internal.1 | MulEF | 4 | MUL | 10,208 | 
| internal.1 | MulEF | 5 | MUL | 10,040 | 
| internal.1 | MulEFI | 4 | MUL | 1,216 | 
| internal.1 | MulEFI | 5 | MUL | 1,216 | 
| internal.1 | MulEI | 4 | ADD | 10,216 | 
| internal.1 | MulEI | 4 | BBE4MUL | 2,554 | 
| internal.1 | MulEI | 5 | ADD | 10,200 | 
| internal.1 | MulEI | 5 | BBE4MUL | 2,550 | 
| internal.1 | MulF | 4 | MUL | 31,718 | 
| internal.1 | MulF | 5 | MUL | 31,550 | 
| internal.1 | MulFI | 4 | MUL | 23,408 | 
| internal.1 | MulFI | 5 | MUL | 23,408 | 
| internal.1 | MulV | 4 | MUL | 1,346 | 
| internal.1 | MulV | 5 | MUL | 1,335 | 
| internal.1 | MulVI | 4 | MUL | 10,053 | 
| internal.1 | MulVI | 5 | MUL | 10,053 | 
| internal.1 | NegE | 4 | MUL | 120 | 
| internal.1 | NegE | 5 | MUL | 120 | 
| internal.1 | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 2,564 | 
| internal.1 | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 2,563 | 
| internal.1 | Publish | 4 | PUBLISH | 52 | 
| internal.1 | Publish | 5 | PUBLISH | 52 | 
| internal.1 | StoreE | 4 | ADD | 18,396 | 
| internal.1 | StoreE | 4 | MUL | 18,396 | 
| internal.1 | StoreE | 4 | STOREW | 28,810 | 
| internal.1 | StoreE | 5 | ADD | 18,354 | 
| internal.1 | StoreE | 5 | MUL | 18,354 | 
| internal.1 | StoreE | 5 | STOREW | 28,641 | 
| internal.1 | StoreF | 4 | ADD | 800 | 
| internal.1 | StoreF | 4 | MUL | 768 | 
| internal.1 | StoreF | 4 | STOREW | 23,350 | 
| internal.1 | StoreF | 5 | ADD | 800 | 
| internal.1 | StoreF | 5 | MUL | 768 | 
| internal.1 | StoreF | 5 | STOREW | 23,300 | 
| internal.1 | StoreHeapPtr | 4 | ADD | 2 | 
| internal.1 | StoreHeapPtr | 5 | ADD | 2 | 
| internal.1 | StoreHintWord | 4 | HINT_STOREW | 57,265 | 
| internal.1 | StoreHintWord | 5 | HINT_STOREW | 57,005 | 
| internal.1 | StoreV | 4 | ADD | 3,424 | 
| internal.1 | StoreV | 4 | MUL | 2,266 | 
| internal.1 | StoreV | 4 | STOREW | 17,218 | 
| internal.1 | StoreV | 5 | ADD | 3,466 | 
| internal.1 | StoreV | 5 | MUL | 2,266 | 
| internal.1 | StoreV | 5 | STOREW | 17,048 | 
| internal.1 | SubE | 4 | FE4SUB | 7,010 | 
| internal.1 | SubE | 5 | FE4SUB | 6,884 | 
| internal.1 | SubEF | 4 | ADD | 34,500 | 
| internal.1 | SubEF | 4 | SUB | 11,500 | 
| internal.1 | SubEF | 5 | ADD | 34,500 | 
| internal.1 | SubEF | 5 | SUB | 11,500 | 
| internal.1 | SubEFI | 4 | ADD | 784 | 
| internal.1 | SubEFI | 5 | ADD | 784 | 
| internal.1 | SubEI | 4 | ADD | 3,152 | 
| internal.1 | SubEI | 5 | ADD | 3,152 | 
| internal.1 | SubF | 4 | SUB | 16 | 
| internal.1 | SubF | 5 | SUB | 16 | 
| internal.1 | SubFI | 4 | SUB | 23,374 | 
| internal.1 | SubFI | 5 | SUB | 23,374 | 
| internal.1 | SubV | 4 | SUB | 14,028 | 
| internal.1 | SubV | 5 | SUB | 13,933 | 
| internal.1 | SubVI | 4 | SUB | 2,090 | 
| internal.1 | SubVI | 5 | SUB | 2,044 | 
| internal.1 | SubVIN | 4 | SUB | 1,764 | 
| internal.1 | SubVIN | 5 | SUB | 1,722 | 
| internal.1 | UnsafeCastVF | 4 | ADD | 736 | 
| internal.1 | UnsafeCastVF | 5 | ADD | 736 | 
| internal.1 | VerifyBatchExt | 4 | VERIFY_BATCH | 1,764 | 
| internal.1 | VerifyBatchExt | 5 | VERIFY_BATCH | 1,722 | 
| internal.1 | VerifyBatchFelt | 4 | VERIFY_BATCH | 504 | 
| internal.1 | VerifyBatchFelt | 5 | VERIFY_BATCH | 504 | 
| internal.1 | ZipFor | 4 | ADD | 148,552 | 
| internal.1 | ZipFor | 4 | BNE | 100,014 | 
| internal.1 | ZipFor | 4 | JAL | 13,644 | 
| internal.1 | ZipFor | 5 | ADD | 147,834 | 
| internal.1 | ZipFor | 5 | BNE | 99,339 | 
| internal.1 | ZipFor | 5 | JAL | 13,601 | 
| internal.2 |  | 6 | ADD | 2 | 
| internal.2 |  | 6 | JAL | 1 | 
| internal.2 | AddE | 6 | FE4ADD | 25,100 | 
| internal.2 | AddEFFI | 6 | ADD | 2,992 | 
| internal.2 | AddEFI | 6 | ADD | 1,440 | 
| internal.2 | AddEI | 6 | ADD | 65,120 | 
| internal.2 | AddF | 6 | ADD | 26,390 | 
| internal.2 | AddFI | 6 | ADD | 25,191 | 
| internal.2 | AddV | 6 | ADD | 15,159 | 
| internal.2 | AddVI | 6 | ADD | 84,634 | 
| internal.2 | Alloc | 6 | ADD | 30,672 | 
| internal.2 | Alloc | 6 | MUL | 8,967 | 
| internal.2 | Alloc | 6 | RANGE_CHECK | 24,303 | 
| internal.2 | AssertEqE | 6 | BNE | 472 | 
| internal.2 | AssertEqEI | 6 | BNE | 32 | 
| internal.2 | AssertEqF | 6 | BNE | 24,169 | 
| internal.2 | AssertEqFI | 6 | BNE | 7 | 
| internal.2 | AssertEqV | 6 | BNE | 1,288 | 
| internal.2 | AssertEqVI | 6 | BNE | 938 | 
| internal.2 | AssertNonZero | 6 | BEQ | 1 | 
| internal.2 | CT-CheckTraceHeightConstraints | 6 | PHANTOM | 4 | 
| internal.2 | CT-HintOpenedValues | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-HintOpeningProof | 6 | PHANTOM | 1,012 | 
| internal.2 | CT-HintOpeningValues | 6 | PHANTOM | 4 | 
| internal.2 | CT-InitializePcsConst | 6 | PHANTOM | 2 | 
| internal.2 | CT-ReadProofsFromInput | 6 | PHANTOM | 2 | 
| internal.2 | CT-VerifyProofs | 6 | PHANTOM | 2 | 
| internal.2 | CT-cache-generator-powers | 6 | PHANTOM | 168 | 
| internal.2 | CT-compute-reduced-opening | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-exp-reverse-bits-len | 6 | PHANTOM | 16,632 | 
| internal.2 | CT-pre-compute-rounds-context | 6 | PHANTOM | 4 | 
| internal.2 | CT-single-reduced-opening-eval | 6 | PHANTOM | 22,848 | 
| internal.2 | CT-stage-c-build-rounds | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verifier-verify | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verify-pcs | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-e-verify-constraints | 6 | PHANTOM | 4 | 
| internal.2 | CT-verify-batch | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-verify-batch-ext | 6 | PHANTOM | 3,360 | 
| internal.2 | CT-verify-query | 6 | PHANTOM | 168 | 
| internal.2 | CastFV | 6 | ADD | 770 | 
| internal.2 | DivE | 6 | BBE4DIV | 13,172 | 
| internal.2 | DivEIN | 6 | ADD | 1,576 | 
| internal.2 | DivEIN | 6 | BBE4DIV | 394 | 
| internal.2 | DivF | 6 | DIV | 1,848 | 
| internal.2 | DivFIN | 6 | DIV | 822 | 
| internal.2 | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 11,424 | 
| internal.2 | HintBitsF | 6 | PHANTOM | 754 | 
| internal.2 | HintFelt | 6 | PHANTOM | 10,441 | 
| internal.2 | HintInputVec | 6 | PHANTOM | 344 | 
| internal.2 | HintLoad | 6 | PHANTOM | 2,688 | 
| internal.2 | IfEq | 6 | BNE | 20,652 | 
| internal.2 | IfEqI | 6 | BNE | 14,550 | 
| internal.2 | IfEqI | 6 | JAL | 3,770 | 
| internal.2 | IfNe | 6 | BEQ | 7,576 | 
| internal.2 | IfNe | 6 | JAL | 6 | 
| internal.2 | IfNeI | 6 | BEQ | 220 | 
| internal.2 | ImmE | 6 | ADD | 4,632 | 
| internal.2 | ImmF | 6 | ADD | 26,124 | 
| internal.2 | ImmV | 6 | ADD | 29,561 | 
| internal.2 | LoadE | 6 | ADD | 21,672 | 
| internal.2 | LoadE | 6 | LOADW | 51,758 | 
| internal.2 | LoadE | 6 | MUL | 21,672 | 
| internal.2 | LoadF | 6 | ADD | 9,490 | 
| internal.2 | LoadF | 6 | LOADW | 72,806 | 
| internal.2 | LoadF | 6 | MUL | 1,088 | 
| internal.2 | LoadHeapPtr | 6 | ADD | 2 | 
| internal.2 | LoadV | 6 | ADD | 16,632 | 
| internal.2 | LoadV | 6 | LOADW | 145,131 | 
| internal.2 | LoadV | 6 | MUL | 14,386 | 
| internal.2 | MulE | 6 | BBE4MUL | 52,219 | 
| internal.2 | MulEF | 6 | MUL | 9,872 | 
| internal.2 | MulEFI | 6 | MUL | 1,216 | 
| internal.2 | MulEI | 6 | ADD | 10,184 | 
| internal.2 | MulEI | 6 | BBE4MUL | 2,546 | 
| internal.2 | MulF | 6 | MUL | 31,382 | 
| internal.2 | MulFI | 6 | MUL | 23,408 | 
| internal.2 | MulV | 6 | MUL | 1,329 | 
| internal.2 | MulVI | 6 | MUL | 10,053 | 
| internal.2 | NegE | 6 | MUL | 120 | 
| internal.2 | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 2,562 | 
| internal.2 | Publish | 6 | PUBLISH | 52 | 
| internal.2 | StoreE | 6 | ADD | 18,312 | 
| internal.2 | StoreE | 6 | MUL | 18,312 | 
| internal.2 | StoreE | 6 | STOREW | 28,472 | 
| internal.2 | StoreF | 6 | ADD | 800 | 
| internal.2 | StoreF | 6 | MUL | 768 | 
| internal.2 | StoreF | 6 | STOREW | 23,250 | 
| internal.2 | StoreHeapPtr | 6 | ADD | 2 | 
| internal.2 | StoreHintWord | 6 | HINT_STOREW | 56,745 | 
| internal.2 | StoreV | 6 | ADD | 3,508 | 
| internal.2 | StoreV | 6 | MUL | 2,266 | 
| internal.2 | StoreV | 6 | STOREW | 16,878 | 
| internal.2 | SubE | 6 | FE4SUB | 6,758 | 
| internal.2 | SubEF | 6 | ADD | 34,500 | 
| internal.2 | SubEF | 6 | SUB | 11,500 | 
| internal.2 | SubEFI | 6 | ADD | 784 | 
| internal.2 | SubEI | 6 | ADD | 3,152 | 
| internal.2 | SubF | 6 | SUB | 16 | 
| internal.2 | SubFI | 6 | SUB | 23,374 | 
| internal.2 | SubV | 6 | SUB | 13,843 | 
| internal.2 | SubVI | 6 | SUB | 1,998 | 
| internal.2 | SubVIN | 6 | SUB | 1,680 | 
| internal.2 | UnsafeCastVF | 6 | ADD | 736 | 
| internal.2 | VerifyBatchExt | 6 | VERIFY_BATCH | 1,680 | 
| internal.2 | VerifyBatchFelt | 6 | VERIFY_BATCH | 504 | 
| internal.2 | ZipFor | 6 | ADD | 147,246 | 
| internal.2 | ZipFor | 6 | BNE | 98,794 | 
| internal.2 | ZipFor | 6 | JAL | 13,558 | 
| leaf |  | 0 | ADD | 2 | 
| leaf |  | 0 | JAL | 1 | 
| leaf |  | 1 | ADD | 2 | 
| leaf |  | 1 | JAL | 1 | 
| leaf |  | 2 | ADD | 2 | 
| leaf |  | 2 | JAL | 1 | 
| leaf |  | 3 | ADD | 2 | 
| leaf |  | 3 | JAL | 1 | 
| leaf |  | 4 | ADD | 2 | 
| leaf |  | 4 | JAL | 1 | 
| leaf |  | 5 | ADD | 2 | 
| leaf |  | 5 | JAL | 1 | 
| leaf |  | 6 | ADD | 2 | 
| leaf |  | 6 | JAL | 1 | 
| leaf | AddE | 0 | FE4ADD | 20,347 | 
| leaf | AddE | 1 | FE4ADD | 16,131 | 
| leaf | AddE | 2 | FE4ADD | 16,131 | 
| leaf | AddE | 3 | FE4ADD | 16,131 | 
| leaf | AddE | 4 | FE4ADD | 16,131 | 
| leaf | AddE | 5 | FE4ADD | 16,131 | 
| leaf | AddE | 6 | FE4ADD | 19,478 | 
| leaf | AddEFFI | 0 | ADD | 928 | 
| leaf | AddEFFI | 1 | ADD | 888 | 
| leaf | AddEFFI | 2 | ADD | 888 | 
| leaf | AddEFFI | 3 | ADD | 888 | 
| leaf | AddEFFI | 4 | ADD | 888 | 
| leaf | AddEFFI | 5 | ADD | 888 | 
| leaf | AddEFFI | 6 | ADD | 920 | 
| leaf | AddEFI | 0 | ADD | 544 | 
| leaf | AddEFI | 1 | ADD | 468 | 
| leaf | AddEFI | 2 | ADD | 468 | 
| leaf | AddEFI | 3 | ADD | 468 | 
| leaf | AddEFI | 4 | ADD | 468 | 
| leaf | AddEFI | 5 | ADD | 468 | 
| leaf | AddEFI | 6 | ADD | 524 | 
| leaf | AddEI | 0 | ADD | 40,336 | 
| leaf | AddEI | 1 | ADD | 36,492 | 
| leaf | AddEI | 2 | ADD | 36,492 | 
| leaf | AddEI | 3 | ADD | 36,492 | 
| leaf | AddEI | 4 | ADD | 36,492 | 
| leaf | AddEI | 5 | ADD | 36,492 | 
| leaf | AddEI | 6 | ADD | 39,424 | 
| leaf | AddF | 0 | ADD | 19,320 | 
| leaf | AddF | 1 | ADD | 15,120 | 
| leaf | AddF | 2 | ADD | 15,120 | 
| leaf | AddF | 3 | ADD | 15,120 | 
| leaf | AddF | 4 | ADD | 15,120 | 
| leaf | AddF | 5 | ADD | 15,120 | 
| leaf | AddF | 6 | ADD | 18,480 | 
| leaf | AddFI | 0 | ADD | 14,889 | 
| leaf | AddFI | 1 | ADD | 11,809 | 
| leaf | AddFI | 2 | ADD | 11,809 | 
| leaf | AddFI | 3 | ADD | 11,809 | 
| leaf | AddFI | 4 | ADD | 11,809 | 
| leaf | AddFI | 5 | ADD | 11,809 | 
| leaf | AddFI | 6 | ADD | 14,225 | 
| leaf | AddV | 0 | ADD | 15,058 | 
| leaf | AddV | 1 | ADD | 12,808 | 
| leaf | AddV | 2 | ADD | 12,808 | 
| leaf | AddV | 3 | ADD | 12,808 | 
| leaf | AddV | 4 | ADD | 12,808 | 
| leaf | AddV | 5 | ADD | 12,808 | 
| leaf | AddV | 6 | ADD | 14,608 | 
| leaf | AddVI | 0 | ADD | 58,520 | 
| leaf | AddVI | 1 | ADD | 48,508 | 
| leaf | AddVI | 2 | ADD | 48,508 | 
| leaf | AddVI | 3 | ADD | 48,508 | 
| leaf | AddVI | 4 | ADD | 48,508 | 
| leaf | AddVI | 5 | ADD | 48,508 | 
| leaf | AddVI | 6 | ADD | 56,373 | 
| leaf | Alloc | 0 | ADD | 34,380 | 
| leaf | Alloc | 0 | MUL | 9,397 | 
| leaf | Alloc | 0 | RANGE_CHECK | 26,587 | 
| leaf | Alloc | 1 | ADD | 33,680 | 
| leaf | Alloc | 1 | MUL | 9,142 | 
| leaf | Alloc | 1 | RANGE_CHECK | 25,982 | 
| leaf | Alloc | 2 | ADD | 33,680 | 
| leaf | Alloc | 2 | MUL | 9,142 | 
| leaf | Alloc | 2 | RANGE_CHECK | 25,982 | 
| leaf | Alloc | 3 | ADD | 33,680 | 
| leaf | Alloc | 3 | MUL | 9,142 | 
| leaf | Alloc | 3 | RANGE_CHECK | 25,982 | 
| leaf | Alloc | 4 | ADD | 33,680 | 
| leaf | Alloc | 4 | MUL | 9,142 | 
| leaf | Alloc | 4 | RANGE_CHECK | 25,982 | 
| leaf | Alloc | 5 | ADD | 33,680 | 
| leaf | Alloc | 5 | MUL | 9,142 | 
| leaf | Alloc | 5 | RANGE_CHECK | 25,982 | 
| leaf | Alloc | 6 | ADD | 34,698 | 
| leaf | Alloc | 6 | MUL | 9,350 | 
| leaf | Alloc | 6 | RANGE_CHECK | 26,699 | 
| leaf | AssertEqE | 0 | BNE | 476 | 
| leaf | AssertEqE | 1 | BNE | 456 | 
| leaf | AssertEqE | 2 | BNE | 456 | 
| leaf | AssertEqE | 3 | BNE | 456 | 
| leaf | AssertEqE | 4 | BNE | 456 | 
| leaf | AssertEqE | 5 | BNE | 456 | 
| leaf | AssertEqE | 6 | BNE | 472 | 
| leaf | AssertEqEI | 0 | BNE | 8 | 
| leaf | AssertEqEI | 1 | BNE | 8 | 
| leaf | AssertEqEI | 2 | BNE | 8 | 
| leaf | AssertEqEI | 3 | BNE | 8 | 
| leaf | AssertEqEI | 4 | BNE | 8 | 
| leaf | AssertEqEI | 5 | BNE | 8 | 
| leaf | AssertEqEI | 6 | BNE | 8 | 
| leaf | AssertEqF | 0 | BNE | 17,664 | 
| leaf | AssertEqF | 1 | BNE | 13,824 | 
| leaf | AssertEqF | 2 | BNE | 13,824 | 
| leaf | AssertEqF | 3 | BNE | 13,824 | 
| leaf | AssertEqF | 4 | BNE | 13,824 | 
| leaf | AssertEqF | 5 | BNE | 13,824 | 
| leaf | AssertEqF | 6 | BNE | 16,904 | 
| leaf | AssertEqV | 0 | BNE | 1,315 | 
| leaf | AssertEqV | 1 | BNE | 1,240 | 
| leaf | AssertEqV | 2 | BNE | 1,240 | 
| leaf | AssertEqV | 3 | BNE | 1,240 | 
| leaf | AssertEqV | 4 | BNE | 1,240 | 
| leaf | AssertEqV | 5 | BNE | 1,240 | 
| leaf | AssertEqV | 6 | BNE | 1,300 | 
| leaf | AssertEqVI | 0 | BNE | 506 | 
| leaf | AssertEqVI | 1 | BNE | 381 | 
| leaf | AssertEqVI | 2 | BNE | 381 | 
| leaf | AssertEqVI | 3 | BNE | 381 | 
| leaf | AssertEqVI | 4 | BNE | 381 | 
| leaf | AssertEqVI | 5 | BNE | 381 | 
| leaf | AssertEqVI | 6 | BNE | 482 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | AssertNonZero | 1 | BEQ | 1 | 
| leaf | AssertNonZero | 2 | BEQ | 1 | 
| leaf | AssertNonZero | 3 | BEQ | 1 | 
| leaf | AssertNonZero | 4 | BEQ | 1 | 
| leaf | AssertNonZero | 5 | BEQ | 1 | 
| leaf | AssertNonZero | 6 | BEQ | 1 | 
| leaf | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 1 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 2 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 3 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 4 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 5 | PHANTOM | 2 | 
| leaf | CT-CheckTraceHeightConstraints | 6 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 2 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 3 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 4 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 5 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 6 | PHANTOM | 2 | 
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 1 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 2 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 3 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 4 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 5 | PHANTOM | 1,600 | 
| leaf | CT-HintOpenedValues | 6 | PHANTOM | 1,600 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 1 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 2 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 3 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 4 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 5 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningProof | 6 | PHANTOM | 1,602 | 
| leaf | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 1 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 2 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 3 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 4 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 5 | PHANTOM | 2 | 
| leaf | CT-HintOpeningValues | 6 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 1 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 2 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 3 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 4 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 5 | PHANTOM | 2 | 
| leaf | CT-InitializePcsConst | 6 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 1 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 2 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 3 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 4 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 5 | PHANTOM | 2 | 
| leaf | CT-ReadProofsFromInput | 6 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 1 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 2 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 3 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 4 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 5 | PHANTOM | 2 | 
| leaf | CT-VerifyProofs | 6 | PHANTOM | 2 | 
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 1 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 2 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 3 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 4 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 5 | PHANTOM | 200 | 
| leaf | CT-cache-generator-powers | 6 | PHANTOM | 200 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 1 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 2 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 3 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 4 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 5 | PHANTOM | 1,600 | 
| leaf | CT-compute-reduced-opening | 6 | PHANTOM | 1,600 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 15,600 | 
| leaf | CT-exp-reverse-bits-len | 1 | PHANTOM | 11,600 | 
| leaf | CT-exp-reverse-bits-len | 2 | PHANTOM | 11,600 | 
| leaf | CT-exp-reverse-bits-len | 3 | PHANTOM | 11,600 | 
| leaf | CT-exp-reverse-bits-len | 4 | PHANTOM | 11,600 | 
| leaf | CT-exp-reverse-bits-len | 5 | PHANTOM | 11,600 | 
| leaf | CT-exp-reverse-bits-len | 6 | PHANTOM | 14,800 | 
| leaf | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 1 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 2 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 3 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 4 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 5 | PHANTOM | 2 | 
| leaf | CT-pre-compute-rounds-context | 6 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 24,200 | 
| leaf | CT-single-reduced-opening-eval | 1 | PHANTOM | 18,200 | 
| leaf | CT-single-reduced-opening-eval | 2 | PHANTOM | 18,200 | 
| leaf | CT-single-reduced-opening-eval | 3 | PHANTOM | 18,200 | 
| leaf | CT-single-reduced-opening-eval | 4 | PHANTOM | 18,200 | 
| leaf | CT-single-reduced-opening-eval | 5 | PHANTOM | 18,200 | 
| leaf | CT-single-reduced-opening-eval | 6 | PHANTOM | 23,000 | 
| leaf | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 1 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 2 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 3 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 4 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 5 | PHANTOM | 2 | 
| leaf | CT-stage-c-build-rounds | 6 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 1 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 2 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 3 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 4 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 5 | PHANTOM | 2 | 
| leaf | CT-stage-d-verifier-verify | 6 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 1 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 2 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 3 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 4 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 5 | PHANTOM | 2 | 
| leaf | CT-stage-d-verify-pcs | 6 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 1 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 2 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 3 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 4 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 5 | PHANTOM | 2 | 
| leaf | CT-stage-e-verify-constraints | 6 | PHANTOM | 2 | 
| leaf | CT-verify-batch | 0 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 1 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 2 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 3 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 4 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 5 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch | 6 | PHANTOM | 1,600 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 1 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 2 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 3 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 4 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 5 | PHANTOM | 4,000 | 
| leaf | CT-verify-batch-ext | 6 | PHANTOM | 4,000 | 
| leaf | CT-verify-query | 0 | PHANTOM | 200 | 
| leaf | CT-verify-query | 1 | PHANTOM | 200 | 
| leaf | CT-verify-query | 2 | PHANTOM | 200 | 
| leaf | CT-verify-query | 3 | PHANTOM | 200 | 
| leaf | CT-verify-query | 4 | PHANTOM | 200 | 
| leaf | CT-verify-query | 5 | PHANTOM | 200 | 
| leaf | CT-verify-query | 6 | PHANTOM | 200 | 
| leaf | CastFV | 0 | ADD | 553 | 
| leaf | CastFV | 1 | ADD | 433 | 
| leaf | CastFV | 2 | ADD | 433 | 
| leaf | CastFV | 3 | ADD | 433 | 
| leaf | CastFV | 4 | ADD | 433 | 
| leaf | CastFV | 5 | ADD | 433 | 
| leaf | CastFV | 6 | ADD | 529 | 
| leaf | DivE | 0 | BBE4DIV | 14,138 | 
| leaf | DivE | 1 | BBE4DIV | 11,128 | 
| leaf | DivE | 2 | BBE4DIV | 11,128 | 
| leaf | DivE | 3 | BBE4DIV | 11,128 | 
| leaf | DivE | 4 | BBE4DIV | 11,128 | 
| leaf | DivE | 5 | BBE4DIV | 11,128 | 
| leaf | DivE | 6 | BBE4DIV | 13,536 | 
| leaf | DivEIN | 0 | ADD | 204 | 
| leaf | DivEIN | 0 | BBE4DIV | 51 | 
| leaf | DivEIN | 1 | ADD | 144 | 
| leaf | DivEIN | 1 | BBE4DIV | 36 | 
| leaf | DivEIN | 2 | ADD | 144 | 
| leaf | DivEIN | 2 | BBE4DIV | 36 | 
| leaf | DivEIN | 3 | ADD | 144 | 
| leaf | DivEIN | 3 | BBE4DIV | 36 | 
| leaf | DivEIN | 4 | ADD | 144 | 
| leaf | DivEIN | 4 | BBE4DIV | 36 | 
| leaf | DivEIN | 5 | ADD | 144 | 
| leaf | DivEIN | 5 | BBE4DIV | 36 | 
| leaf | DivEIN | 6 | ADD | 192 | 
| leaf | DivEIN | 6 | BBE4DIV | 48 | 
| leaf | DivF | 0 | DIV | 2,100 | 
| leaf | DivF | 1 | DIV | 2,100 | 
| leaf | DivF | 2 | DIV | 2,100 | 
| leaf | DivF | 3 | DIV | 2,100 | 
| leaf | DivF | 4 | DIV | 2,100 | 
| leaf | DivF | 5 | DIV | 2,100 | 
| leaf | DivF | 6 | DIV | 2,100 | 
| leaf | DivFIN | 0 | DIV | 121 | 
| leaf | DivFIN | 1 | DIV | 86 | 
| leaf | DivFIN | 2 | DIV | 86 | 
| leaf | DivFIN | 3 | DIV | 86 | 
| leaf | DivFIN | 4 | DIV | 86 | 
| leaf | DivFIN | 5 | DIV | 86 | 
| leaf | DivFIN | 6 | DIV | 114 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 12,100 | 
| leaf | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 9,100 | 
| leaf | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 9,100 | 
| leaf | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 9,100 | 
| leaf | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 9,100 | 
| leaf | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 9,100 | 
| leaf | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 11,500 | 
| leaf | HintBitsF | 0 | PHANTOM | 552 | 
| leaf | HintBitsF | 1 | PHANTOM | 432 | 
| leaf | HintBitsF | 2 | PHANTOM | 432 | 
| leaf | HintBitsF | 3 | PHANTOM | 432 | 
| leaf | HintBitsF | 4 | PHANTOM | 432 | 
| leaf | HintBitsF | 5 | PHANTOM | 432 | 
| leaf | HintBitsF | 6 | PHANTOM | 528 | 
| leaf | HintFelt | 0 | PHANTOM | 12,224 | 
| leaf | HintFelt | 1 | PHANTOM | 12,169 | 
| leaf | HintFelt | 2 | PHANTOM | 12,169 | 
| leaf | HintFelt | 3 | PHANTOM | 12,169 | 
| leaf | HintFelt | 4 | PHANTOM | 12,169 | 
| leaf | HintFelt | 5 | PHANTOM | 12,169 | 
| leaf | HintFelt | 6 | PHANTOM | 12,438 | 
| leaf | HintInputVec | 0 | PHANTOM | 161 | 
| leaf | HintInputVec | 1 | PHANTOM | 121 | 
| leaf | HintInputVec | 2 | PHANTOM | 121 | 
| leaf | HintInputVec | 3 | PHANTOM | 121 | 
| leaf | HintInputVec | 4 | PHANTOM | 121 | 
| leaf | HintInputVec | 5 | PHANTOM | 121 | 
| leaf | HintInputVec | 6 | PHANTOM | 153 | 
| leaf | HintLoad | 0 | PHANTOM | 3,600 | 
| leaf | HintLoad | 1 | PHANTOM | 3,600 | 
| leaf | HintLoad | 2 | PHANTOM | 3,600 | 
| leaf | HintLoad | 3 | PHANTOM | 3,600 | 
| leaf | HintLoad | 4 | PHANTOM | 3,600 | 
| leaf | HintLoad | 5 | PHANTOM | 3,600 | 
| leaf | HintLoad | 6 | PHANTOM | 3,600 | 
| leaf | IfEq | 0 | BNE | 10,128 | 
| leaf | IfEq | 1 | BNE | 7,179 | 
| leaf | IfEq | 2 | BNE | 7,179 | 
| leaf | IfEq | 3 | BNE | 7,179 | 
| leaf | IfEq | 4 | BNE | 7,179 | 
| leaf | IfEq | 5 | BNE | 7,179 | 
| leaf | IfEq | 6 | BNE | 9,483 | 
| leaf | IfEqI | 0 | BNE | 11,498 | 
| leaf | IfEqI | 0 | JAL | 4,224 | 
| leaf | IfEqI | 1 | BNE | 10,718 | 
| leaf | IfEqI | 1 | JAL | 4,152 | 
| leaf | IfEqI | 2 | BNE | 10,718 | 
| leaf | IfEqI | 2 | JAL | 4,218 | 
| leaf | IfEqI | 3 | BNE | 10,718 | 
| leaf | IfEqI | 3 | JAL | 4,128 | 
| leaf | IfEqI | 4 | BNE | 10,718 | 
| leaf | IfEqI | 4 | JAL | 4,267 | 
| leaf | IfEqI | 5 | BNE | 10,718 | 
| leaf | IfEqI | 5 | JAL | 4,143 | 
| leaf | IfEqI | 6 | BNE | 11,342 | 
| leaf | IfEqI | 6 | JAL | 4,197 | 
| leaf | IfNe | 0 | BEQ | 5,996 | 
| leaf | IfNe | 0 | JAL | 3 | 
| leaf | IfNe | 1 | BEQ | 4,446 | 
| leaf | IfNe | 1 | JAL | 2 | 
| leaf | IfNe | 2 | BEQ | 4,446 | 
| leaf | IfNe | 2 | JAL | 2 | 
| leaf | IfNe | 3 | BEQ | 4,446 | 
| leaf | IfNe | 3 | JAL | 2 | 
| leaf | IfNe | 4 | BEQ | 4,446 | 
| leaf | IfNe | 4 | JAL | 2 | 
| leaf | IfNe | 5 | BEQ | 4,446 | 
| leaf | IfNe | 5 | JAL | 2 | 
| leaf | IfNe | 6 | BEQ | 5,686 | 
| leaf | IfNe | 6 | JAL | 2 | 
| leaf | IfNeI | 0 | BEQ | 114 | 
| leaf | IfNeI | 1 | BEQ | 84 | 
| leaf | IfNeI | 2 | BEQ | 84 | 
| leaf | IfNeI | 3 | BEQ | 84 | 
| leaf | IfNeI | 4 | BEQ | 84 | 
| leaf | IfNeI | 5 | BEQ | 84 | 
| leaf | IfNeI | 6 | BEQ | 108 | 
| leaf | ImmE | 0 | ADD | 3,308 | 
| leaf | ImmE | 1 | ADD | 2,912 | 
| leaf | ImmE | 2 | ADD | 2,912 | 
| leaf | ImmE | 3 | ADD | 2,912 | 
| leaf | ImmE | 4 | ADD | 2,912 | 
| leaf | ImmE | 5 | ADD | 2,912 | 
| leaf | ImmE | 6 | ADD | 3,244 | 
| leaf | ImmF | 0 | ADD | 19,148 | 
| leaf | ImmF | 1 | ADD | 15,058 | 
| leaf | ImmF | 2 | ADD | 15,058 | 
| leaf | ImmF | 3 | ADD | 15,058 | 
| leaf | ImmF | 4 | ADD | 15,058 | 
| leaf | ImmF | 5 | ADD | 15,058 | 
| leaf | ImmF | 6 | ADD | 18,330 | 
| leaf | ImmV | 0 | ADD | 30,516 | 
| leaf | ImmV | 1 | ADD | 25,241 | 
| leaf | ImmV | 2 | ADD | 25,241 | 
| leaf | ImmV | 3 | ADD | 25,241 | 
| leaf | ImmV | 4 | ADD | 25,241 | 
| leaf | ImmV | 5 | ADD | 25,241 | 
| leaf | ImmV | 6 | ADD | 29,464 | 
| leaf | LoadE | 0 | ADD | 21,600 | 
| leaf | LoadE | 0 | LOADW | 50,065 | 
| leaf | LoadE | 0 | MUL | 21,600 | 
| leaf | LoadE | 1 | ADD | 17,600 | 
| leaf | LoadE | 1 | LOADW | 40,288 | 
| leaf | LoadE | 1 | MUL | 17,600 | 
| leaf | LoadE | 2 | ADD | 17,600 | 
| leaf | LoadE | 2 | LOADW | 40,288 | 
| leaf | LoadE | 2 | MUL | 17,600 | 
| leaf | LoadE | 3 | ADD | 17,600 | 
| leaf | LoadE | 3 | LOADW | 40,288 | 
| leaf | LoadE | 3 | MUL | 17,600 | 
| leaf | LoadE | 4 | ADD | 17,600 | 
| leaf | LoadE | 4 | LOADW | 40,288 | 
| leaf | LoadE | 4 | MUL | 17,600 | 
| leaf | LoadE | 5 | ADD | 17,600 | 
| leaf | LoadE | 5 | LOADW | 40,288 | 
| leaf | LoadE | 5 | MUL | 17,600 | 
| leaf | LoadE | 6 | ADD | 20,800 | 
| leaf | LoadE | 6 | LOADW | 48,096 | 
| leaf | LoadE | 6 | MUL | 20,800 | 
| leaf | LoadF | 0 | ADD | 8,381 | 
| leaf | LoadF | 0 | LOADW | 53,546 | 
| leaf | LoadF | 0 | MUL | 480 | 
| leaf | LoadF | 1 | ADD | 6,261 | 
| leaf | LoadF | 1 | LOADW | 40,686 | 
| leaf | LoadF | 1 | MUL | 360 | 
| leaf | LoadF | 2 | ADD | 6,261 | 
| leaf | LoadF | 2 | LOADW | 40,686 | 
| leaf | LoadF | 2 | MUL | 360 | 
| leaf | LoadF | 3 | ADD | 6,261 | 
| leaf | LoadF | 3 | LOADW | 40,686 | 
| leaf | LoadF | 3 | MUL | 360 | 
| leaf | LoadF | 4 | ADD | 6,261 | 
| leaf | LoadF | 4 | LOADW | 40,686 | 
| leaf | LoadF | 4 | MUL | 360 | 
| leaf | LoadF | 5 | ADD | 6,261 | 
| leaf | LoadF | 5 | LOADW | 40,686 | 
| leaf | LoadF | 5 | MUL | 360 | 
| leaf | LoadF | 6 | ADD | 7,957 | 
| leaf | LoadF | 6 | LOADW | 51,574 | 
| leaf | LoadF | 6 | MUL | 456 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadHeapPtr | 1 | ADD | 1 | 
| leaf | LoadHeapPtr | 2 | ADD | 1 | 
| leaf | LoadHeapPtr | 3 | ADD | 1 | 
| leaf | LoadHeapPtr | 4 | ADD | 1 | 
| leaf | LoadHeapPtr | 5 | ADD | 1 | 
| leaf | LoadHeapPtr | 6 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 14,330 | 
| leaf | LoadV | 0 | LOADW | 139,041 | 
| leaf | LoadV | 0 | MUL | 11,973 | 
| leaf | LoadV | 1 | ADD | 13,240 | 
| leaf | LoadV | 1 | LOADW | 113,611 | 
| leaf | LoadV | 1 | MUL | 10,963 | 
| leaf | LoadV | 2 | ADD | 13,240 | 
| leaf | LoadV | 2 | LOADW | 113,611 | 
| leaf | LoadV | 2 | MUL | 10,963 | 
| leaf | LoadV | 3 | ADD | 13,240 | 
| leaf | LoadV | 3 | LOADW | 113,611 | 
| leaf | LoadV | 3 | MUL | 10,963 | 
| leaf | LoadV | 4 | ADD | 13,240 | 
| leaf | LoadV | 4 | LOADW | 113,611 | 
| leaf | LoadV | 4 | MUL | 10,963 | 
| leaf | LoadV | 5 | ADD | 13,240 | 
| leaf | LoadV | 5 | LOADW | 113,611 | 
| leaf | LoadV | 5 | MUL | 10,963 | 
| leaf | LoadV | 6 | ADD | 14,112 | 
| leaf | LoadV | 6 | LOADW | 133,956 | 
| leaf | LoadV | 6 | MUL | 11,771 | 
| leaf | MulE | 0 | BBE4MUL | 36,477 | 
| leaf | MulE | 1 | BBE4MUL | 29,489 | 
| leaf | MulE | 2 | BBE4MUL | 29,489 | 
| leaf | MulE | 3 | BBE4MUL | 29,489 | 
| leaf | MulE | 4 | BBE4MUL | 29,489 | 
| leaf | MulE | 5 | BBE4MUL | 29,489 | 
| leaf | MulE | 6 | BBE4MUL | 35,077 | 
| leaf | MulEF | 0 | MUL | 8,408 | 
| leaf | MulEF | 1 | MUL | 8,288 | 
| leaf | MulEF | 2 | MUL | 8,288 | 
| leaf | MulEF | 3 | MUL | 8,288 | 
| leaf | MulEF | 4 | MUL | 8,288 | 
| leaf | MulEF | 5 | MUL | 8,288 | 
| leaf | MulEF | 6 | MUL | 8,384 | 
| leaf | MulEFI | 0 | MUL | 372 | 
| leaf | MulEFI | 1 | MUL | 176 | 
| leaf | MulEFI | 2 | MUL | 176 | 
| leaf | MulEFI | 3 | MUL | 176 | 
| leaf | MulEFI | 4 | MUL | 176 | 
| leaf | MulEFI | 5 | MUL | 176 | 
| leaf | MulEFI | 6 | MUL | 332 | 
| leaf | MulEI | 0 | ADD | 4,780 | 
| leaf | MulEI | 0 | BBE4MUL | 1,195 | 
| leaf | MulEI | 1 | ADD | 3,024 | 
| leaf | MulEI | 1 | BBE4MUL | 756 | 
| leaf | MulEI | 2 | ADD | 3,024 | 
| leaf | MulEI | 2 | BBE4MUL | 756 | 
| leaf | MulEI | 3 | ADD | 3,024 | 
| leaf | MulEI | 3 | BBE4MUL | 756 | 
| leaf | MulEI | 4 | ADD | 3,024 | 
| leaf | MulEI | 4 | BBE4MUL | 756 | 
| leaf | MulEI | 5 | ADD | 3,024 | 
| leaf | MulEI | 5 | BBE4MUL | 756 | 
| leaf | MulEI | 6 | ADD | 4,384 | 
| leaf | MulEI | 6 | BBE4MUL | 1,096 | 
| leaf | MulF | 0 | MUL | 25,684 | 
| leaf | MulF | 1 | MUL | 21,914 | 
| leaf | MulF | 2 | MUL | 21,914 | 
| leaf | MulF | 3 | MUL | 21,914 | 
| leaf | MulF | 4 | MUL | 21,914 | 
| leaf | MulF | 5 | MUL | 21,914 | 
| leaf | MulF | 6 | MUL | 24,930 | 
| leaf | MulFI | 0 | MUL | 17,131 | 
| leaf | MulFI | 1 | MUL | 13,406 | 
| leaf | MulFI | 2 | MUL | 13,406 | 
| leaf | MulFI | 3 | MUL | 13,406 | 
| leaf | MulFI | 4 | MUL | 13,406 | 
| leaf | MulFI | 5 | MUL | 13,406 | 
| leaf | MulFI | 6 | MUL | 16,386 | 
| leaf | MulV | 0 | MUL | 725 | 
| leaf | MulV | 1 | MUL | 497 | 
| leaf | MulV | 2 | MUL | 497 | 
| leaf | MulV | 3 | MUL | 497 | 
| leaf | MulV | 4 | MUL | 497 | 
| leaf | MulV | 5 | MUL | 497 | 
| leaf | MulV | 6 | MUL | 677 | 
| leaf | MulVI | 0 | MUL | 9,354 | 
| leaf | MulVI | 1 | MUL | 7,249 | 
| leaf | MulVI | 2 | MUL | 7,249 | 
| leaf | MulVI | 3 | MUL | 7,249 | 
| leaf | MulVI | 4 | MUL | 7,249 | 
| leaf | MulVI | 5 | MUL | 7,249 | 
| leaf | MulVI | 6 | MUL | 8,933 | 
| leaf | NegE | 0 | MUL | 72 | 
| leaf | NegE | 1 | MUL | 52 | 
| leaf | NegE | 2 | MUL | 52 | 
| leaf | NegE | 3 | MUL | 52 | 
| leaf | NegE | 4 | MUL | 52 | 
| leaf | NegE | 5 | MUL | 52 | 
| leaf | NegE | 6 | MUL | 68 | 
| leaf | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 1,256 | 
| leaf | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 887 | 
| leaf | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 887 | 
| leaf | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 887 | 
| leaf | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 887 | 
| leaf | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 887 | 
| leaf | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 1,175 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | Publish | 1 | PUBLISH | 36 | 
| leaf | Publish | 2 | PUBLISH | 36 | 
| leaf | Publish | 3 | PUBLISH | 36 | 
| leaf | Publish | 4 | PUBLISH | 36 | 
| leaf | Publish | 5 | PUBLISH | 36 | 
| leaf | Publish | 6 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 17,600 | 
| leaf | StoreE | 0 | MUL | 17,600 | 
| leaf | StoreE | 0 | STOREW | 29,391 | 
| leaf | StoreE | 1 | ADD | 13,600 | 
| leaf | StoreE | 1 | MUL | 13,600 | 
| leaf | StoreE | 1 | STOREW | 25,356 | 
| leaf | StoreE | 2 | ADD | 13,600 | 
| leaf | StoreE | 2 | MUL | 13,600 | 
| leaf | StoreE | 2 | STOREW | 25,356 | 
| leaf | StoreE | 3 | ADD | 13,600 | 
| leaf | StoreE | 3 | MUL | 13,600 | 
| leaf | StoreE | 3 | STOREW | 25,356 | 
| leaf | StoreE | 4 | ADD | 13,600 | 
| leaf | StoreE | 4 | MUL | 13,600 | 
| leaf | StoreE | 4 | STOREW | 25,356 | 
| leaf | StoreE | 5 | ADD | 13,600 | 
| leaf | StoreE | 5 | MUL | 13,600 | 
| leaf | StoreE | 5 | STOREW | 25,356 | 
| leaf | StoreE | 6 | ADD | 16,800 | 
| leaf | StoreE | 6 | MUL | 16,800 | 
| leaf | StoreE | 6 | STOREW | 28,584 | 
| leaf | StoreF | 0 | ADD | 308 | 
| leaf | StoreF | 0 | MUL | 292 | 
| leaf | StoreF | 0 | STOREW | 12,473 | 
| leaf | StoreF | 1 | ADD | 228 | 
| leaf | StoreF | 1 | MUL | 212 | 
| leaf | StoreF | 1 | STOREW | 9,443 | 
| leaf | StoreF | 2 | ADD | 228 | 
| leaf | StoreF | 2 | MUL | 212 | 
| leaf | StoreF | 2 | STOREW | 9,443 | 
| leaf | StoreF | 3 | ADD | 228 | 
| leaf | StoreF | 3 | MUL | 212 | 
| leaf | StoreF | 3 | STOREW | 9,443 | 
| leaf | StoreF | 4 | ADD | 228 | 
| leaf | StoreF | 4 | MUL | 212 | 
| leaf | StoreF | 4 | STOREW | 9,443 | 
| leaf | StoreF | 5 | ADD | 228 | 
| leaf | StoreF | 5 | MUL | 212 | 
| leaf | StoreF | 5 | STOREW | 9,443 | 
| leaf | StoreF | 6 | ADD | 508 | 
| leaf | StoreF | 6 | MUL | 492 | 
| leaf | StoreF | 6 | STOREW | 12,459 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHeapPtr | 1 | ADD | 1 | 
| leaf | StoreHeapPtr | 2 | ADD | 1 | 
| leaf | StoreHeapPtr | 3 | ADD | 1 | 
| leaf | StoreHeapPtr | 4 | ADD | 1 | 
| leaf | StoreHeapPtr | 5 | ADD | 1 | 
| leaf | StoreHeapPtr | 6 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 42,772 | 
| leaf | StoreHintWord | 1 | HINT_STOREW | 36,012 | 
| leaf | StoreHintWord | 2 | HINT_STOREW | 36,012 | 
| leaf | StoreHintWord | 3 | HINT_STOREW | 36,012 | 
| leaf | StoreHintWord | 4 | HINT_STOREW | 36,012 | 
| leaf | StoreHintWord | 5 | HINT_STOREW | 36,012 | 
| leaf | StoreHintWord | 6 | HINT_STOREW | 41,589 | 
| leaf | StoreV | 0 | ADD | 2,204 | 
| leaf | StoreV | 0 | MUL | 1,003 | 
| leaf | StoreV | 0 | STOREW | 17,243 | 
| leaf | StoreV | 1 | ADD | 1,904 | 
| leaf | StoreV | 1 | MUL | 748 | 
| leaf | StoreV | 1 | STOREW | 16,718 | 
| leaf | StoreV | 2 | ADD | 1,904 | 
| leaf | StoreV | 2 | MUL | 748 | 
| leaf | StoreV | 2 | STOREW | 16,718 | 
| leaf | StoreV | 3 | ADD | 1,904 | 
| leaf | StoreV | 3 | MUL | 748 | 
| leaf | StoreV | 3 | STOREW | 16,718 | 
| leaf | StoreV | 4 | ADD | 1,904 | 
| leaf | StoreV | 4 | MUL | 748 | 
| leaf | StoreV | 4 | STOREW | 16,718 | 
| leaf | StoreV | 5 | ADD | 1,904 | 
| leaf | StoreV | 5 | MUL | 748 | 
| leaf | StoreV | 5 | STOREW | 16,718 | 
| leaf | StoreV | 6 | ADD | 2,144 | 
| leaf | StoreV | 6 | MUL | 952 | 
| leaf | StoreV | 6 | STOREW | 17,138 | 
| leaf | SubE | 0 | FE4SUB | 6,671 | 
| leaf | SubE | 1 | FE4SUB | 6,506 | 
| leaf | SubE | 2 | FE4SUB | 6,506 | 
| leaf | SubE | 3 | FE4SUB | 6,506 | 
| leaf | SubE | 4 | FE4SUB | 6,506 | 
| leaf | SubE | 5 | FE4SUB | 6,506 | 
| leaf | SubE | 6 | FE4SUB | 6,635 | 
| leaf | SubEF | 0 | ADD | 36,474 | 
| leaf | SubEF | 0 | SUB | 12,158 | 
| leaf | SubEF | 1 | ADD | 27,444 | 
| leaf | SubEF | 1 | SUB | 9,148 | 
| leaf | SubEF | 2 | ADD | 27,444 | 
| leaf | SubEF | 2 | SUB | 9,148 | 
| leaf | SubEF | 3 | ADD | 27,444 | 
| leaf | SubEF | 3 | SUB | 9,148 | 
| leaf | SubEF | 4 | ADD | 27,444 | 
| leaf | SubEF | 4 | SUB | 9,148 | 
| leaf | SubEF | 5 | ADD | 27,444 | 
| leaf | SubEF | 5 | SUB | 9,148 | 
| leaf | SubEF | 6 | ADD | 34,668 | 
| leaf | SubEF | 6 | SUB | 11,556 | 
| leaf | SubEFI | 0 | ADD | 264 | 
| leaf | SubEFI | 1 | ADD | 144 | 
| leaf | SubEFI | 2 | ADD | 144 | 
| leaf | SubEFI | 3 | ADD | 144 | 
| leaf | SubEFI | 4 | ADD | 144 | 
| leaf | SubEFI | 5 | ADD | 144 | 
| leaf | SubEFI | 6 | ADD | 252 | 
| leaf | SubEI | 0 | ADD | 408 | 
| leaf | SubEI | 1 | ADD | 288 | 
| leaf | SubEI | 2 | ADD | 288 | 
| leaf | SubEI | 3 | ADD | 288 | 
| leaf | SubEI | 4 | ADD | 288 | 
| leaf | SubEI | 5 | ADD | 288 | 
| leaf | SubEI | 6 | ADD | 384 | 
| leaf | SubFI | 0 | SUB | 17,112 | 
| leaf | SubFI | 1 | SUB | 13,392 | 
| leaf | SubFI | 2 | SUB | 13,392 | 
| leaf | SubFI | 3 | SUB | 13,392 | 
| leaf | SubFI | 4 | SUB | 13,392 | 
| leaf | SubFI | 5 | SUB | 13,392 | 
| leaf | SubFI | 6 | SUB | 16,368 | 
| leaf | SubV | 0 | SUB | 13,973 | 
| leaf | SubV | 1 | SUB | 11,785 | 
| leaf | SubV | 2 | SUB | 11,785 | 
| leaf | SubV | 3 | SUB | 11,785 | 
| leaf | SubV | 4 | SUB | 11,785 | 
| leaf | SubV | 5 | SUB | 11,785 | 
| leaf | SubV | 6 | SUB | 13,533 | 
| leaf | SubVI | 0 | SUB | 2,218 | 
| leaf | SubVI | 1 | SUB | 2,213 | 
| leaf | SubVI | 2 | SUB | 2,213 | 
| leaf | SubVI | 3 | SUB | 2,213 | 
| leaf | SubVI | 4 | SUB | 2,213 | 
| leaf | SubVI | 5 | SUB | 2,213 | 
| leaf | SubVI | 6 | SUB | 2,217 | 
| leaf | SubVIN | 0 | SUB | 2,000 | 
| leaf | SubVIN | 1 | SUB | 2,000 | 
| leaf | SubVIN | 2 | SUB | 2,000 | 
| leaf | SubVIN | 3 | SUB | 2,000 | 
| leaf | SubVIN | 4 | SUB | 2,000 | 
| leaf | SubVIN | 5 | SUB | 2,000 | 
| leaf | SubVIN | 6 | SUB | 2,000 | 
| leaf | UnsafeCastVF | 0 | ADD | 489 | 
| leaf | UnsafeCastVF | 1 | ADD | 359 | 
| leaf | UnsafeCastVF | 2 | ADD | 359 | 
| leaf | UnsafeCastVF | 3 | ADD | 359 | 
| leaf | UnsafeCastVF | 4 | ADD | 359 | 
| leaf | UnsafeCastVF | 5 | ADD | 359 | 
| leaf | UnsafeCastVF | 6 | ADD | 463 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 1 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 2 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 3 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 4 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 5 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchExt | 6 | VERIFY_BATCH | 2,000 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 1 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 2 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 3 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 4 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 5 | VERIFY_BATCH | 800 | 
| leaf | VerifyBatchFelt | 6 | VERIFY_BATCH | 800 | 
| leaf | ZipFor | 0 | ADD | 121,103 | 
| leaf | ZipFor | 0 | BNE | 73,849 | 
| leaf | ZipFor | 0 | JAL | 12,350 | 
| leaf | ZipFor | 1 | ADD | 97,455 | 
| leaf | ZipFor | 1 | BNE | 60,841 | 
| leaf | ZipFor | 1 | JAL | 10,125 | 
| leaf | ZipFor | 2 | ADD | 97,455 | 
| leaf | ZipFor | 2 | BNE | 60,841 | 
| leaf | ZipFor | 2 | JAL | 10,125 | 
| leaf | ZipFor | 3 | ADD | 97,455 | 
| leaf | ZipFor | 3 | BNE | 60,841 | 
| leaf | ZipFor | 3 | JAL | 10,125 | 
| leaf | ZipFor | 4 | ADD | 97,455 | 
| leaf | ZipFor | 4 | BNE | 60,841 | 
| leaf | ZipFor | 4 | JAL | 10,125 | 
| leaf | ZipFor | 5 | ADD | 97,455 | 
| leaf | ZipFor | 5 | BNE | 60,841 | 
| leaf | ZipFor | 5 | JAL | 10,125 | 
| leaf | ZipFor | 6 | ADD | 116,337 | 
| leaf | ZipFor | 6 | BNE | 71,211 | 
| leaf | ZipFor | 6 | JAL | 11,906 | 
| root |  | 0 | ADD | 2 | 
| root |  | 0 | JAL | 1 | 
| root | AddE | 0 | FE4ADD | 12,550 | 
| root | AddEFFI | 0 | ADD | 1,496 | 
| root | AddEFI | 0 | ADD | 720 | 
| root | AddEI | 0 | ADD | 32,560 | 
| root | AddF | 0 | ADD | 13,195 | 
| root | AddFI | 0 | ADD | 12,608 | 
| root | AddV | 0 | ADD | 7,581 | 
| root | AddVI | 0 | ADD | 42,344 | 
| root | Alloc | 0 | ADD | 15,362 | 
| root | Alloc | 0 | MUL | 4,499 | 
| root | Alloc | 0 | RANGE_CHECK | 12,180 | 
| root | AssertEqE | 0 | BNE | 236 | 
| root | AssertEqEI | 0 | BNE | 16 | 
| root | AssertEqF | 0 | BNE | 12,080 | 
| root | AssertEqFI | 0 | BNE | 5 | 
| root | AssertEqV | 0 | BNE | 644 | 
| root | AssertEqVI | 0 | BNE | 470 | 
| root | AssertNonZero | 0 | BEQ | 1 | 
| root | CT-CheckTraceHeightConstraints | 0 | PHANTOM | 2 | 
| root | CT-ExtractPublicValues | 0 | PHANTOM | 2 | 
| root | CT-HintOpenedValues | 0 | PHANTOM | 504 | 
| root | CT-HintOpeningProof | 0 | PHANTOM | 506 | 
| root | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| root | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| root | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| root | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| root | CT-cache-generator-powers | 0 | PHANTOM | 84 | 
| root | CT-compute-reduced-opening | 0 | PHANTOM | 504 | 
| root | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,316 | 
| root | CT-pre-compute-rounds-context | 0 | PHANTOM | 2 | 
| root | CT-single-reduced-opening-eval | 0 | PHANTOM | 11,424 | 
| root | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| root | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| root | CT-verify-batch | 0 | PHANTOM | 504 | 
| root | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| root | CT-verify-query | 0 | PHANTOM | 84 | 
| root | CastFV | 0 | ADD | 385 | 
| root | DivE | 0 | BBE4DIV | 6,586 | 
| root | DivEIN | 0 | ADD | 788 | 
| root | DivEIN | 0 | BBE4DIV | 197 | 
| root | DivF | 0 | DIV | 924 | 
| root | DivFIN | 0 | DIV | 411 | 
| root | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,712 | 
| root | HintBitsF | 0 | PHANTOM | 377 | 
| root | HintFelt | 0 | PHANTOM | 5,217 | 
| root | HintInputVec | 0 | PHANTOM | 173 | 
| root | HintLoad | 0 | PHANTOM | 1,344 | 
| root | IfEq | 0 | BNE | 10,326 | 
| root | IfEqI | 0 | BNE | 7,275 | 
| root | IfEqI | 0 | JAL | 1,834 | 
| root | IfNe | 0 | BEQ | 3,788 | 
| root | IfNe | 0 | JAL | 3 | 
| root | IfNeI | 0 | BEQ | 110 | 
| root | ImmE | 0 | ADD | 2,316 | 
| root | ImmF | 0 | ADD | 13,155 | 
| root | ImmV | 0 | ADD | 14,823 | 
| root | LoadE | 0 | ADD | 10,836 | 
| root | LoadE | 0 | LOADW | 25,879 | 
| root | LoadE | 0 | MUL | 10,836 | 
| root | LoadF | 0 | ADD | 4,745 | 
| root | LoadF | 0 | LOADW | 36,535 | 
| root | LoadF | 0 | MUL | 544 | 
| root | LoadHeapPtr | 0 | ADD | 1 | 
| root | LoadV | 0 | ADD | 8,316 | 
| root | LoadV | 0 | LOADW | 72,567 | 
| root | LoadV | 0 | MUL | 7,193 | 
| root | MulE | 0 | BBE4MUL | 26,097 | 
| root | MulEF | 0 | MUL | 4,936 | 
| root | MulEFI | 0 | MUL | 608 | 
| root | MulEI | 0 | ADD | 5,092 | 
| root | MulEI | 0 | BBE4MUL | 1,273 | 
| root | MulF | 0 | MUL | 15,691 | 
| root | MulFI | 0 | MUL | 11,704 | 
| root | MulV | 0 | MUL | 664 | 
| root | MulVI | 0 | MUL | 5,027 | 
| root | NegE | 0 | MUL | 60 | 
| root | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 12 | 
| root | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 1,281 | 
| root | Publish | 0 | PUBLISH | 48 | 
| root | StoreE | 0 | ADD | 9,156 | 
| root | StoreE | 0 | MUL | 9,156 | 
| root | StoreE | 0 | STOREW | 14,236 | 
| root | StoreF | 0 | ADD | 400 | 
| root | StoreF | 0 | MUL | 384 | 
| root | StoreF | 0 | STOREW | 11,885 | 
| root | StoreHeapPtr | 0 | ADD | 1 | 
| root | StoreHintWord | 0 | HINT_STOREW | 28,402 | 
| root | StoreV | 0 | ADD | 1,754 | 
| root | StoreV | 0 | MUL | 1,133 | 
| root | StoreV | 0 | STOREW | 8,467 | 
| root | SubE | 0 | FE4SUB | 3,379 | 
| root | SubEF | 0 | ADD | 17,250 | 
| root | SubEF | 0 | SUB | 5,750 | 
| root | SubEFI | 0 | ADD | 392 | 
| root | SubEI | 0 | ADD | 1,576 | 
| root | SubF | 0 | SUB | 8 | 
| root | SubFI | 0 | SUB | 11,687 | 
| root | SubV | 0 | SUB | 6,921 | 
| root | SubVI | 0 | SUB | 999 | 
| root | SubVIN | 0 | SUB | 840 | 
| root | UnsafeCastVF | 0 | ADD | 368 | 
| root | VerifyBatchExt | 0 | VERIFY_BATCH | 840 | 
| root | VerifyBatchFelt | 0 | VERIFY_BATCH | 252 | 
| root | ZipFor | 0 | ADD | 73,644 | 
| root | ZipFor | 0 | BNE | 49,418 | 
| root | ZipFor | 0 | JAL | 6,781 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fib_e2e |  | ADD | 0 | 1,048,521 | 
| fib_e2e |  | ADD | 1 | 1,048,502 | 
| fib_e2e |  | ADD | 2 | 1,048,502 | 
| fib_e2e |  | ADD | 3 | 1,048,501 | 
| fib_e2e |  | ADD | 4 | 1,048,501 | 
| fib_e2e |  | ADD | 5 | 1,048,502 | 
| fib_e2e |  | ADD | 6 | 909,059 | 
| fib_e2e |  | AND | 0 | 3 | 
| fib_e2e |  | AND | 6 | 1 | 
| fib_e2e |  | AUIPC | 0 | 8 | 
| fib_e2e |  | AUIPC | 6 | 5 | 
| fib_e2e |  | BEQ | 0 | 116,501 | 
| fib_e2e |  | BEQ | 1 | 116,500 | 
| fib_e2e |  | BEQ | 2 | 116,500 | 
| fib_e2e |  | BEQ | 3 | 116,501 | 
| fib_e2e |  | BEQ | 4 | 116,500 | 
| fib_e2e |  | BEQ | 5 | 116,500 | 
| fib_e2e |  | BEQ | 6 | 101,007 | 
| fib_e2e |  | BGEU | 0 | 2 | 
| fib_e2e |  | BLT | 6 | 2 | 
| fib_e2e |  | BLTU | 0 | 4 | 
| fib_e2e |  | BNE | 0 | 116,500 | 
| fib_e2e |  | BNE | 1 | 116,500 | 
| fib_e2e |  | BNE | 2 | 116,500 | 
| fib_e2e |  | BNE | 3 | 116,500 | 
| fib_e2e |  | BNE | 4 | 116,500 | 
| fib_e2e |  | BNE | 5 | 116,500 | 
| fib_e2e |  | BNE | 6 | 101,005 | 
| fib_e2e |  | HINT_BUFFER | 0 | 2 | 
| fib_e2e |  | HINT_STOREW | 0 | 1 | 
| fib_e2e |  | JAL | 0 | 116,499 | 
| fib_e2e |  | JAL | 1 | 116,500 | 
| fib_e2e |  | JAL | 2 | 116,500 | 
| fib_e2e |  | JAL | 3 | 116,500 | 
| fib_e2e |  | JAL | 4 | 116,500 | 
| fib_e2e |  | JAL | 5 | 116,500 | 
| fib_e2e |  | JAL | 6 | 101,003 | 
| fib_e2e |  | JALR | 0 | 11 | 
| fib_e2e |  | JALR | 6 | 14 | 
| fib_e2e |  | LOADBU | 0 | 1 | 
| fib_e2e |  | LOADBU | 6 | 7 | 
| fib_e2e |  | LOADW | 0 | 13 | 
| fib_e2e |  | LOADW | 6 | 17 | 
| fib_e2e |  | LUI | 0 | 15 | 
| fib_e2e |  | LUI | 6 | 6 | 
| fib_e2e |  | OR | 0 | 3 | 
| fib_e2e |  | PHANTOM | 0 | 1 | 
| fib_e2e |  | SLTU | 0 | 349,497 | 
| fib_e2e |  | SLTU | 1 | 349,500 | 
| fib_e2e |  | SLTU | 2 | 349,500 | 
| fib_e2e |  | SLTU | 3 | 349,500 | 
| fib_e2e |  | SLTU | 4 | 349,501 | 
| fib_e2e |  | SLTU | 5 | 349,500 | 
| fib_e2e |  | SLTU | 6 | 303,005 | 
| fib_e2e |  | STOREB | 6 | 10 | 
| fib_e2e |  | STOREW | 0 | 17 | 
| fib_e2e |  | STOREW | 6 | 23 | 
| fib_e2e |  | SUB | 0 | 2 | 
| fib_e2e |  | XOR | 0 | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 1,261 | 17,095 | 2,421,325 | 348,105,186 | 7,182 | 1,302 | 1,246 | 1,291 | 1,616 | 1,422 | 136,809,223 | 300 | 8,652 | 
| internal.0 | 1 | 1,239 | 17,144 | 2,421,386 | 348,105,186 | 7,228 | 1,314 | 1,236 | 1,304 | 1,643 | 1,426 | 136,688,963 | 301 | 8,677 | 
| internal.0 | 2 | 1,226 | 17,122 | 2,421,346 | 348,105,186 | 7,216 | 1,315 | 1,234 | 1,313 | 1,627 | 1,421 | 136,688,483 | 301 | 8,680 | 
| internal.0 | 3 | 662 | 9,146 | 1,210,998 | 185,719,266 | 4,141 | 737 | 778 | 704 | 995 | 761 | 69,107,531 | 160 | 4,343 | 
| internal.1 | 4 | 722 | 10,296 | 1,579,468 | 184,134,114 | 4,093 | 708 | 784 | 679 | 1,008 | 756 | 75,273,875 | 153 | 5,481 | 
| internal.1 | 5 | 738 | 10,345 | 1,571,669 | 187,279,842 | 4,185 | 723 | 810 | 709 | 1,012 | 772 | 74,681,351 | 154 | 5,422 | 
| internal.2 | 6 | 746 | 10,258 | 1,564,464 | 184,134,114 | 4,098 | 710 | 798 | 679 | 997 | 758 | 73,837,376 | 150 | 5,414 | 
| leaf | 0 | 662 | 7,935 | 1,310,905 | 254,942,698 | 2,560 | 377 | 281 | 691 | 618 | 424 | 70,452,126 | 163 | 4,713 | 
| leaf | 1 | 573 | 6,658 | 1,085,412 | 203,757,034 | 2,123 | 306 | 237 | 533 | 558 | 354 | 59,082,625 | 128 | 3,962 | 
| leaf | 2 | 560 | 6,614 | 1,085,478 | 203,757,034 | 2,114 | 307 | 237 | 533 | 553 | 352 | 59,083,417 | 127 | 3,940 | 
| leaf | 3 | 571 | 6,639 | 1,085,388 | 203,757,034 | 2,117 | 303 | 236 | 532 | 555 | 354 | 59,082,337 | 129 | 3,951 | 
| leaf | 4 | 563 | 6,635 | 1,085,527 | 203,757,034 | 2,118 | 301 | 239 | 533 | 561 | 353 | 59,084,005 | 126 | 3,954 | 
| leaf | 5 | 553 | 6,597 | 1,085,403 | 203,757,034 | 2,111 | 305 | 236 | 531 | 556 | 352 | 59,082,517 | 126 | 3,933 | 
| leaf | 6 | 653 | 7,457 | 1,267,992 | 208,674,282 | 2,240 | 314 | 241 | 553 | 628 | 367 | 67,955,170 | 131 | 4,564 | 
| root | 0 | 381 | 41,505 | 782,946 | 80,779,418 | 38,402 | 863 | 13,813 | 7,650 | 3,706 | 12,274 | 37,566,481 | 77 | 2,722 | 

| group | idx | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| internal.0 | 0 | 0 | 9,961,604 | 2,013,265,921 | 
| internal.0 | 0 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 0 | 2 | 4,980,802 | 2,013,265,921 | 
| internal.0 | 0 | 3 | 50,610,436 | 2,013,265,921 | 
| internal.0 | 0 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 0 | 5 | 116,564,682 | 2,013,265,921 | 
| internal.0 | 1 | 0 | 9,961,604 | 2,013,265,921 | 
| internal.0 | 1 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 1 | 2 | 4,980,802 | 2,013,265,921 | 
| internal.0 | 1 | 3 | 50,610,436 | 2,013,265,921 | 
| internal.0 | 1 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 1 | 5 | 116,564,682 | 2,013,265,921 | 
| internal.0 | 2 | 0 | 9,961,604 | 2,013,265,921 | 
| internal.0 | 2 | 1 | 50,356,480 | 2,013,265,921 | 
| internal.0 | 2 | 2 | 4,980,802 | 2,013,265,921 | 
| internal.0 | 2 | 3 | 50,610,436 | 2,013,265,921 | 
| internal.0 | 2 | 4 | 262,144 | 2,013,265,921 | 
| internal.0 | 2 | 5 | 116,564,682 | 2,013,265,921 | 
| internal.0 | 3 | 0 | 4,980,868 | 2,013,265,921 | 
| internal.0 | 3 | 1 | 26,358,016 | 2,013,265,921 | 
| internal.0 | 3 | 2 | 2,490,434 | 2,013,265,921 | 
| internal.0 | 3 | 3 | 26,091,780 | 2,013,265,921 | 
| internal.0 | 3 | 4 | 131,072 | 2,013,265,921 | 
| internal.0 | 3 | 5 | 60,445,386 | 2,013,265,921 | 
| internal.1 | 4 | 0 | 5,243,012 | 2,013,265,921 | 
| internal.1 | 4 | 1 | 23,748,864 | 2,013,265,921 | 
| internal.1 | 4 | 2 | 2,621,506 | 2,013,265,921 | 
| internal.1 | 4 | 3 | 23,478,532 | 2,013,265,921 | 
| internal.1 | 4 | 4 | 131,072 | 2,013,265,921 | 
| internal.1 | 4 | 5 | 55,616,202 | 2,013,265,921 | 
| internal.1 | 5 | 0 | 5,243,012 | 2,013,265,921 | 
| internal.1 | 5 | 1 | 24,011,008 | 2,013,265,921 | 
| internal.1 | 5 | 2 | 2,621,506 | 2,013,265,921 | 
| internal.1 | 5 | 3 | 24,133,892 | 2,013,265,921 | 
| internal.1 | 5 | 4 | 131,072 | 2,013,265,921 | 
| internal.1 | 5 | 5 | 56,533,706 | 2,013,265,921 | 
| internal.2 | 6 | 0 | 5,243,012 | 2,013,265,921 | 
| internal.2 | 6 | 1 | 23,748,864 | 2,013,265,921 | 
| internal.2 | 6 | 2 | 2,621,506 | 2,013,265,921 | 
| internal.2 | 6 | 3 | 23,478,532 | 2,013,265,921 | 
| internal.2 | 6 | 4 | 131,072 | 2,013,265,921 | 
| internal.2 | 6 | 5 | 55,616,202 | 2,013,265,921 | 
| leaf | 0 | 0 | 5,636,228 | 2,013,265,921 | 
| leaf | 0 | 1 | 26,751,232 | 2,013,265,921 | 
| leaf | 0 | 2 | 2,818,114 | 2,013,265,921 | 
| leaf | 0 | 3 | 26,878,212 | 2,013,265,921 | 
| leaf | 0 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 0 | 5 | 62,608,074 | 2,013,265,921 | 
| leaf | 1 | 0 | 4,325,508 | 2,013,265,921 | 
| leaf | 1 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 1 | 2 | 2,162,754 | 2,013,265,921 | 
| leaf | 1 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 1 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 1 | 5 | 47,262,410 | 2,013,265,921 | 
| leaf | 2 | 0 | 4,325,508 | 2,013,265,921 | 
| leaf | 2 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 2 | 2 | 2,162,754 | 2,013,265,921 | 
| leaf | 2 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 2 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 2 | 5 | 47,262,410 | 2,013,265,921 | 
| leaf | 3 | 0 | 4,325,508 | 2,013,265,921 | 
| leaf | 3 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 3 | 2 | 2,162,754 | 2,013,265,921 | 
| leaf | 3 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 3 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 3 | 5 | 47,262,410 | 2,013,265,921 | 
| leaf | 4 | 0 | 4,325,508 | 2,013,265,921 | 
| leaf | 4 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 4 | 2 | 2,162,754 | 2,013,265,921 | 
| leaf | 4 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 4 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 4 | 5 | 47,262,410 | 2,013,265,921 | 
| leaf | 5 | 0 | 4,325,508 | 2,013,265,921 | 
| leaf | 5 | 1 | 20,060,416 | 2,013,265,921 | 
| leaf | 5 | 2 | 2,162,754 | 2,013,265,921 | 
| leaf | 5 | 3 | 20,189,444 | 2,013,265,921 | 
| leaf | 5 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 5 | 5 | 47,262,410 | 2,013,265,921 | 
| leaf | 6 | 0 | 4,456,580 | 2,013,265,921 | 
| leaf | 6 | 1 | 20,459,776 | 2,013,265,921 | 
| leaf | 6 | 2 | 2,228,290 | 2,013,265,921 | 
| leaf | 6 | 3 | 20,586,756 | 2,013,265,921 | 
| leaf | 6 | 4 | 131,072 | 2,013,265,921 | 
| leaf | 6 | 5 | 48,255,690 | 2,013,265,921 | 
| root | 0 | 0 | 2,302,080 | 2,013,265,921 | 
| root | 0 | 1 | 14,704,640 | 2,013,265,921 | 
| root | 0 | 2 | 1,151,040 | 2,013,265,921 | 
| root | 0 | 3 | 15,687,680 | 2,013,265,921 | 
| root | 0 | 4 | 262,144 | 2,013,265,921 | 
| root | 0 | 5 | 34,631,874 | 2,013,265,921 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 621 | 7,922 | 1,747,603 | 160,811,212 | 1,864 | 256 | 216 | 443 | 510 | 318 | 58,898,534 | 116 | 5,437 | 
| fib_e2e | 1 | 613 | 7,800 | 1,747,502 | 160,757,324 | 1,777 | 245 | 183 | 437 | 498 | 286 | 58,859,694 | 122 | 5,410 | 
| fib_e2e | 2 | 625 | 7,831 | 1,747,502 | 160,757,324 | 1,772 | 253 | 182 | 439 | 493 | 284 | 58,859,694 | 118 | 5,434 | 
| fib_e2e | 3 | 617 | 7,833 | 1,747,502 | 160,757,324 | 1,778 | 253 | 179 | 438 | 499 | 281 | 58,859,684 | 122 | 5,438 | 
| fib_e2e | 4 | 591 | 7,806 | 1,747,502 | 160,757,324 | 1,763 | 254 | 179 | 440 | 487 | 283 | 58,859,695 | 117 | 5,452 | 
| fib_e2e | 5 | 626 | 7,848 | 1,747,502 | 160,757,324 | 1,764 | 250 | 182 | 439 | 489 | 282 | 58,859,694 | 118 | 5,458 | 
| fib_e2e | 6 | 540 | 7,024 | 1,515,164 | 160,813,500 | 1,772 | 250 | 182 | 438 | 500 | 284 | 51,097,540 | 115 | 4,712 | 

| group | segment | trace_height_constraint | weighted_sum | threshold |
| --- | --- | --- | --- | --- |
| fib_e2e | 0 | 0 | 3,932,302 | 2,013,265,921 | 
| fib_e2e | 0 | 1 | 10,748,488 | 2,013,265,921 | 
| fib_e2e | 0 | 2 | 1,966,151 | 2,013,265,921 | 
| fib_e2e | 0 | 3 | 10,748,492 | 2,013,265,921 | 
| fib_e2e | 0 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 0 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 0 | 6 | 7,209,044 | 2,013,265,921 | 
| fib_e2e | 0 | 7 |  | 2,013,265,921 | 
| fib_e2e | 0 | 8 | 35,531,581 | 2,013,265,921 | 
| fib_e2e | 1 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 1 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 1 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 1 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 1 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 1 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 1 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 1 | 7 |  | 2,013,265,921 | 
| fib_e2e | 1 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 2 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 2 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 2 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 2 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 2 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 2 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 2 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 2 | 7 |  | 2,013,265,921 | 
| fib_e2e | 2 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 3 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 3 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 3 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 3 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 3 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 3 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 3 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 3 | 7 |  | 2,013,265,921 | 
| fib_e2e | 3 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 4 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 4 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 4 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 4 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 4 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 4 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 4 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 4 | 7 |  | 2,013,265,921 | 
| fib_e2e | 4 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 5 | 0 | 3,932,166 | 2,013,265,921 | 
| fib_e2e | 5 | 1 | 10,747,968 | 2,013,265,921 | 
| fib_e2e | 5 | 2 | 1,966,083 | 2,013,265,921 | 
| fib_e2e | 5 | 3 | 10,747,940 | 2,013,265,921 | 
| fib_e2e | 5 | 4 | 400 | 2,013,265,921 | 
| fib_e2e | 5 | 5 | 144 | 2,013,265,921 | 
| fib_e2e | 5 | 6 | 7,208,960 | 2,013,265,921 | 
| fib_e2e | 5 | 7 |  | 2,013,265,921 | 
| fib_e2e | 5 | 8 | 35,529,485 | 2,013,265,921 | 
| fib_e2e | 6 | 0 | 3,932,346 | 2,013,265,921 | 
| fib_e2e | 6 | 1 | 10,748,632 | 2,013,265,921 | 
| fib_e2e | 6 | 2 | 1,966,173 | 2,013,265,921 | 
| fib_e2e | 6 | 3 | 10,748,700 | 2,013,265,921 | 
| fib_e2e | 6 | 4 | 832 | 2,013,265,921 | 
| fib_e2e | 6 | 5 | 320 | 2,013,265,921 | 
| fib_e2e | 6 | 6 | 7,209,020 | 2,013,265,921 | 
| fib_e2e | 6 | 7 |  | 2,013,265,921 | 
| fib_e2e | 6 | 8 | 35,531,975 | 2,013,265,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-fib_e2e.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-halo2_outer.cell_tracker_span.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-halo2_outer.cell_tracker_span.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-halo2_outer.cell_tracker_span.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-halo2_outer.cell_tracker_span.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.0.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.1.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-internal.2.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/be2ebfb85281f9ecb4668f3fefad039ad4326d4b/fib_e2e-be2ebfb85281f9ecb4668f3fefad039ad4326d4b-root.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/be2ebfb85281f9ecb4668f3fefad039ad4326d4b

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: jemalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/14394391873)
