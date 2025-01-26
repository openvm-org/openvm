| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  547.69 |  318.48 |
| fib_e2e |  76.09 |  11.40 |
| leaf |  84.91 |  13.02 |
| internal.0 |  92.38 |  26.35 |
| internal.1 |  53.51 |  26.90 |
| internal.2 |  27.21 |  27.21 |
| root |  45.31 |  45.31 |
| halo2_outer |  127.55 |  127.55 |
| halo2_wrapper |  40.76 |  40.76 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,869.43 |  76,086 |  11,404 |  9,979 |
| `main_cells_used     ` |  58,671,151.43 |  410,698,060 |  59,808,409 |  51,983,912 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  5,731.14 |  40,118 |  6,189 |  5,028 |
| `trace_gen_time_ms   ` |  878.71 |  6,151 |  977 |  758 |
| `stark_prove_excluding_trace_time_ms` |  4,259.57 |  29,817 |  4,594 |  4,135 |
| `main_trace_commit_time_ms` |  617.57 |  4,323 |  850 |  536 |
| `generate_perm_trace_time_ms` |  136.71 |  957 |  187 |  116 |
| `perm_trace_commit_time_ms` |  1,756.14 |  12,293 |  1,953 |  1,571 |
| `quotient_poly_compute_time_ms` |  302.71 |  2,119 |  325 |  292 |
| `quotient_poly_commit_time_ms` |  459.29 |  3,215 |  547 |  437 |
| `pcs_opening_time_ms ` |  983.71 |  6,886 |  1,012 |  958 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  12,129.29 |  84,905 |  13,018 |  11,852 |
| `main_cells_used     ` |  62,598,608.14 |  438,190,257 |  70,644,665 |  60,645,385 |
| `total_cycles        ` |  1,653,628 |  11,575,396 |  1,847,991 |  1,605,968 |
| `execute_time_ms     ` |  5,538.29 |  38,768 |  6,097 |  5,337 |
| `trace_gen_time_ms   ` |  1,244.57 |  8,712 |  1,405 |  1,190 |
| `stark_prove_excluding_trace_time_ms` |  5,346.43 |  37,425 |  5,516 |  5,217 |
| `main_trace_commit_time_ms` |  1,129.71 |  7,908 |  1,159 |  1,089 |
| `generate_perm_trace_time_ms` |  132.29 |  926 |  143 |  126 |
| `perm_trace_commit_time_ms` |  1,152.86 |  8,070 |  1,189 |  1,105 |
| `quotient_poly_compute_time_ms` |  681.29 |  4,769 |  752 |  651 |
| `quotient_poly_commit_time_ms` |  992.86 |  6,950 |  1,040 |  963 |
| `pcs_opening_time_ms ` |  1,252.57 |  8,768 |  1,316 |  1,213 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  23,093.75 |  92,375 |  26,346 |  13,906 |
| `main_cells_used     ` |  124,063,377.25 |  496,253,509 |  141,440,292 |  71,938,447 |
| `total_cycles        ` |  3,253,979.50 |  13,015,918 |  3,718,878 |  1,860,167 |
| `execute_time_ms     ` |  11,046.50 |  44,186 |  12,660 |  6,399 |
| `trace_gen_time_ms   ` |  2,478 |  9,912 |  2,899 |  1,570 |
| `stark_prove_excluding_trace_time_ms` |  9,569.25 |  38,277 |  10,832 |  5,937 |
| `main_trace_commit_time_ms` |  1,965.75 |  7,863 |  2,201 |  1,371 |
| `generate_perm_trace_time_ms` |  245.75 |  983 |  281 |  153 |
| `perm_trace_commit_time_ms` |  2,134.50 |  8,538 |  2,439 |  1,294 |
| `quotient_poly_compute_time_ms` |  1,251.50 |  5,006 |  1,427 |  747 |
| `quotient_poly_commit_time_ms` |  1,724.75 |  6,899 |  1,964 |  1,044 |
| `pcs_opening_time_ms ` |  2,241.50 |  8,966 |  2,575 |  1,324 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  26,752.50 |  53,505 |  26,898 |  26,607 |
| `main_cells_used     ` |  145,189,181.50 |  290,378,363 |  146,058,076 |  144,320,287 |
| `total_cycles        ` |  3,815,902.50 |  7,631,805 |  3,846,585 |  3,785,220 |
| `execute_time_ms     ` |  13,163 |  26,326 |  13,228 |  13,098 |
| `trace_gen_time_ms   ` |  2,825 |  5,650 |  3,014 |  2,636 |
| `stark_prove_excluding_trace_time_ms` |  10,764.50 |  21,529 |  10,786 |  10,743 |
| `main_trace_commit_time_ms` |  2,161.50 |  4,323 |  2,167 |  2,156 |
| `generate_perm_trace_time_ms` |  280.50 |  561 |  283 |  278 |
| `perm_trace_commit_time_ms` |  2,431 |  4,862 |  2,451 |  2,411 |
| `quotient_poly_compute_time_ms` |  1,438.50 |  2,877 |  1,445 |  1,432 |
| `quotient_poly_commit_time_ms` |  1,939.50 |  3,879 |  1,956 |  1,923 |
| `pcs_opening_time_ms ` |  2,508.50 |  5,017 |  2,526 |  2,491 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  27,206 |  27,206 |  27,206 |  27,206 |
| `main_cells_used     ` |  146,057,156 |  146,057,156 |  146,057,156 |  146,057,156 |
| `total_cycles        ` |  3,846,493 |  3,846,493 |  3,846,493 |  3,846,493 |
| `execute_time_ms     ` |  13,355 |  13,355 |  13,355 |  13,355 |
| `trace_gen_time_ms   ` |  2,982 |  2,982 |  2,982 |  2,982 |
| `stark_prove_excluding_trace_time_ms` |  10,869 |  10,869 |  10,869 |  10,869 |
| `main_trace_commit_time_ms` |  2,212 |  2,212 |  2,212 |  2,212 |
| `generate_perm_trace_time_ms` |  279 |  279 |  279 |  279 |
| `perm_trace_commit_time_ms` |  2,448 |  2,448 |  2,448 |  2,448 |
| `quotient_poly_compute_time_ms` |  1,453 |  1,453 |  1,453 |  1,453 |
| `quotient_poly_commit_time_ms` |  1,914 |  1,914 |  1,914 |  1,914 |
| `pcs_opening_time_ms ` |  2,557 |  2,557 |  2,557 |  2,557 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  45,307 |  45,307 |  45,307 |  45,307 |
| `main_cells_used     ` |  74,290,893 |  74,290,893 |  74,290,893 |  74,290,893 |
| `total_cycles        ` |  1,923,977 |  1,923,977 |  1,923,977 |  1,923,977 |
| `execute_time_ms     ` |  6,896 |  6,896 |  6,896 |  6,896 |
| `trace_gen_time_ms   ` |  1,476 |  1,476 |  1,476 |  1,476 |
| `stark_prove_excluding_trace_time_ms` |  36,935 |  36,935 |  36,935 |  36,935 |
| `main_trace_commit_time_ms` |  11,326 |  11,326 |  11,326 |  11,326 |
| `generate_perm_trace_time_ms` |  146 |  146 |  146 |  146 |
| `perm_trace_commit_time_ms` |  12,607 |  12,607 |  12,607 |  12,607 |
| `quotient_poly_compute_time_ms` |  767 |  767 |  767 |  767 |
| `quotient_poly_commit_time_ms` |  8,220 |  8,220 |  8,220 |  8,220 |
| `pcs_opening_time_ms ` |  3,862 |  3,862 |  3,862 |  3,862 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  127,545 |  127,545 |  127,545 |  127,545 |
| `main_cells_used     ` |  78,361,729 |  78,361,729 |  78,361,729 |  78,361,729 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  40,756 |  40,756 |  40,756 |  40,756 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 3,551 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 127,545 |  | 78,361,729 | 
| halo2_wrapper | 40,756 |  |  | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 1 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 2 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 3 | BNE | 5,428 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 1 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 2 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 3 | BNE | 92 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 206,655 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 206,655 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 206,655 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 103,040 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 1 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 2 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 3 | BNE | 69 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 62,514 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 62,514 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 62,514 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 31,257 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 10,902 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 10,902 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 10,902 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 5,451 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 3,220 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 955,374 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 955,374 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 955,374 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 477,687 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 3,289 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 2,139 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 13,357,204 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 13,354,720 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 13,354,720 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 6,680,074 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 131,510 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 130,400 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 130,270 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 67,140 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 30 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 598,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 598,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 598,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 299,390 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 1 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 2 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 83,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 83,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 83,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 41,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 66,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 66,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 66,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 33,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,235,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 2,235,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 2,235,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 1,117,920 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 294,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 1 | ADD | 294,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 2 | ADD | 294,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 3 | ADD | 147,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 717,390 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 717,390 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 717,390 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 359,070 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 2,382,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 1 | ADD | 2,382,240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 2 | ADD | 2,382,240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 3 | ADD | 1,191,510 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 5,291,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 5,291,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 5,291,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 2,646,720 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 5,865,780 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,624,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 5,865,780 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 1,624,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 5,865,780 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 1,624,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 2,933,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 812,370 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 8,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 8,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 8,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 4,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 22,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 332,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 1 | DIV | 332,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 2 | DIV | 332,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 3 | DIV | 166,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 11,730 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 290,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 290,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 290,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 145,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 340,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 340,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 340,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 172,530 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 313,470 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 313,470 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 313,470 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 158,490 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 640,080 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 320,040 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 320,040 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 809,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 809,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 809,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 404,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 269,430 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 2,728,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 2,437,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 2,728,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 2,437,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 2,728,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 2,437,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 1,364,490 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 1,218,780 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 291,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 291,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 291,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 145,680 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 49,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 49,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 49,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 24,600 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 410,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 410,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 410,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 205,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,608,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 1 | MUL | 1,608,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 2 | MUL | 1,608,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 3 | MUL | 804,420 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 261,420 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 261,420 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 261,420 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 130,710 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 656,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 656,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 656,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 328,170 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 20,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 1 | MUL | 20,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 2 | MUL | 20,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 3 | MUL | 10,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 539,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 269,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 269,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 44,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 44,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 44,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 22,260 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 11,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 829,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 555,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 829,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 555,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 829,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 555,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 414,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 277,830 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 509,940 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 169,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 23,040 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 23,040 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 23,040 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 11,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 0 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 1 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 2 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 3 | SUB | 240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 260,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 260,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 260,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 130,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 376,500 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 1 | SUB | 376,500 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 2 | SUB | 376,500 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 3 | SUB | 188,250 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 59,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 59,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 59,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 29,940 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 50,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 50,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 50,400 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 25,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 6,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 6,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 6,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 3,420 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 18,914,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 18,911,340 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 18,911,340 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 9,459,210 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 1,384,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 1,384,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 1,384,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 692,375 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 6,674,950 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 6,674,950 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 6,674,950 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 3,337,600 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 319,650 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 319,650 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 319,650 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 161,925 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 11,182,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 11,182,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 11,182,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 5,591,625 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 2,833,250 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 2,833,250 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 2,833,250 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 1,418,025 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,770,380 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 1,770,380 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 1,770,380 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 885,190 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 979,336 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 979,336 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 979,336 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 489,668 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,169,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 1,169,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 1,169,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 584,600 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 520,160 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 520,160 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 520,160 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 260,080 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 7,480 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 2,191,920 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 2,187,920 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 2,187,920 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 1,097,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 136,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 136,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 136,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 68,440 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 283,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 283,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 283,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 141,600 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 3,468,192 | 
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
| internal.0 | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 48,888 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 135,072 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 135,072 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 135,072 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 67,536 | 
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
| internal.0 | PhantomAir | CT-verify-batch | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-verify-batch | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-verify-batch | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-verify-batch | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 20,160 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 20,160 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 2 | PHANTOM | 20,160 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 3 | PHANTOM | 10,080 | 
| internal.0 | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 1 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 2 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 3 | PHANTOM | 504 | 
| internal.0 | PhantomAir | HintBitsF | 0 | PHANTOM | 1,680 | 
| internal.0 | PhantomAir | HintBitsF | 1 | PHANTOM | 1,680 | 
| internal.0 | PhantomAir | HintBitsF | 2 | PHANTOM | 1,680 | 
| internal.0 | PhantomAir | HintBitsF | 3 | PHANTOM | 840 | 
| internal.0 | PhantomAir | HintInputVec | 0 | PHANTOM | 261,714 | 
| internal.0 | PhantomAir | HintInputVec | 1 | PHANTOM | 261,714 | 
| internal.0 | PhantomAir | HintInputVec | 2 | PHANTOM | 261,714 | 
| internal.0 | PhantomAir | HintInputVec | 3 | PHANTOM | 130,884 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 20,349 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 1 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 2 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 3 | VERIFY_BATCH | 4,524,660 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 11,546,262 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 11,563,020 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 11,563,020 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 5,747,994 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 207,023 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 207,023 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 4 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 5 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 64,446 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 63,480 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 10,902 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 10,902 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 6,624 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 6,532 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 983,526 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 969,818 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 6,762 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 6,670 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 4,278 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 4,278 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 13,971,396 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 13,676,628 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 10 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 10 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 138,540 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 134,390 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 60 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 60 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 624,730 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 612,120 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 4 | ADD | 30 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 5 | ADD | 30 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 88,320 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 87,840 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 66,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 66,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 2,270,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 2,255,040 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 4 | ADD | 294,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 5 | ADD | 294,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 749,550 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 734,190 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 4 | ADD | 2,518,140 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 5 | ADD | 2,451,810 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 5,546,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 5,423,970 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 6,178,500 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 1,707,540 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 6,027,180 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 1,667,190 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 8,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 8,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 44,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 44,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 4 | DIV | 347,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 5 | DIV | 340,200 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 23,460 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 23,460 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 294,240 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 294,240 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 340,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 340,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 322,590 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 318,390 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 647,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 647,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 643,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 643,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 809,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 538,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 809,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 538,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 2,741,580 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 2,447,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 2,735,280 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 2,442,600 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 301,440 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 296,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 49,200 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 49,200 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 411,600 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 411,120 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 4 | MUL | 1,669,320 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 5 | MUL | 1,639,080 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 261,420 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 261,420 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 661,350 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 658,830 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 4 | MUL | 20,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 5 | MUL | 20,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 541,800 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 541,800 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 540,540 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 540,540 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 45,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 22,560 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 45,720 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 22,560 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 826,800 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 555,660 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 828,060 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 555,660 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 1,019,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 339,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 1,019,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 339,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 23,040 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 23,040 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 89,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 89,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 4 | SUB | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 5 | SUB | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 260,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 260,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 4 | SUB | 381,540 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 5 | SUB | 379,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 62,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 61,260 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 52,920 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 51,660 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 6,840 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 6,840 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 19,730,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 19,338,810 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 1,389,650 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 1,388,300 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 6,778,000 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 6,727,525 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 333,450 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 326,950 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 11,694,350 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 11,448,425 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 2,950,950 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 2,894,200 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 1,787,516 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 1,778,948 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 990,828 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 985,082 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 1,177,440 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 1,173,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 523,520 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 521,840 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 14,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 14,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 2,238,400 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 2,219,480 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 137,200 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 137,040 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 293,280 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 288,240 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-cache-generator-powers | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-cache-generator-powers | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 97,776 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 97,776 | 
| internal.1 | PhantomAir | CT-pre-compute-alpha-pows | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-pre-compute-alpha-pows | 5 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-single-reduced-opening-eval | 4 | PHANTOM | 135,072 | 
| internal.1 | PhantomAir | CT-single-reduced-opening-eval | 5 | PHANTOM | 135,072 | 
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
| internal.1 | PhantomAir | HintBitsF | 4 | PHANTOM | 1,680 | 
| internal.1 | PhantomAir | HintBitsF | 5 | PHANTOM | 1,680 | 
| internal.1 | PhantomAir | HintInputVec | 4 | PHANTOM | 276,342 | 
| internal.1 | PhantomAir | HintInputVec | 5 | PHANTOM | 269,280 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 43,092 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 42,693 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 4 | VERIFY_BATCH | 9,853,704 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 5 | VERIFY_BATCH | 9,451,512 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 11,630,052 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 11,579,778 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 10,856 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 184 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 207,023 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 6 | BNE | 161 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 64,446 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 10,902 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 6,624 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 983,526 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 6,762 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 4,278 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 13,971,396 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 10 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 137,620 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 60 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 624,730 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 1,196 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 6 | ADD | 30 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 88,320 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 66,000 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 2,270,400 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 6 | ADD | 294,000 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 749,550 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 6 | ADD | 2,518,140 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 5,546,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 6,178,500 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 1,707,540 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 8,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 44,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 6 | DIV | 347,760 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 23,460 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 294,240 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 340,020 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 322,590 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 647,640 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 647,640 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 809,760 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 538,860 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 60 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 2,741,580 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 2,447,640 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 301,440 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 49,200 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 411,600 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 6 | MUL | 1,669,320 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 261,420 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 661,350 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 6 | MUL | 20,400 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 541,800 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 541,800 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 45,960 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 22,560 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 60 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 826,800 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 555,660 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 1,019,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 339,960 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 23,040 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 89,760 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 6 | SUB | 480 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 260,400 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 6 | SUB | 381,540 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 62,640 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 52,920 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 6,840 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 19,730,880 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 1,389,650 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 6,778,000 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 333,450 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 11,694,350 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 2,950,950 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 1,787,516 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 990,828 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 1,177,440 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 523,520 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 14,960 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 2,238,400 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 137,200 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 293,280 | 
| internal.2 | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 6,936,384 | 
| internal.2 | PhantomAir | CT-InitializePcsConst | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-ReadProofsFromInput | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-VerifyProofs | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-cache-generator-powers | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 97,776 | 
| internal.2 | PhantomAir | CT-pre-compute-alpha-pows | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 135,072 | 
| internal.2 | PhantomAir | CT-stage-c-build-rounds | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verifier-verify | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verify-pcs | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-e-verify-constraints | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-verify-batch | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 21,168 | 
| internal.2 | PhantomAir | CT-verify-query | 6 | PHANTOM | 1,008 | 
| internal.2 | PhantomAir | HintBitsF | 6 | PHANTOM | 1,680 | 
| internal.2 | PhantomAir | HintInputVec | 6 | PHANTOM | 276,342 | 
| internal.2 | VerifyBatchAir | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 43,092 | 
| internal.2 | VerifyBatchAir | VerifyBatchExt | 6 | VERIFY_BATCH | 9,853,704 | 
| internal.2 | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 11,630,052 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,704 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 1 | BNE | 5,152 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 2 | BNE | 5,152 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 3 | BNE | 5,152 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 5,152 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 5,152 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 5,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 1 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 2 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 3 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 92 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 92,000 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 74,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 74,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 74,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 74,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 74,336 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 80,408 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 33,373 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 31,855 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 31,855 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 31,855 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 31,855 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 31,855 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 32,361 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,543 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 4,025 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 4,025 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 4,025 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 4,025 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 4,025 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 4,554 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,266 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 540,109 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 498,157 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 498,157 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 498,157 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 498,157 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 498,157 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 512,141 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,300 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 1,840 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,822,375 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 6,052,841 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 6,052,841 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 6,052,841 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 6,052,841 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 6,052,841 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 6,338,478 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 82,480 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 84,540 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 91,270 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 91,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 92,450 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 86,450 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 90,320 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 305,470 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 283,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 283,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 283,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 283,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 283,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 292,850 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 21,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 19,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 19,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 19,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 19,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 19,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 20,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 18,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 14,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 14,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 14,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 14,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 14,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 15,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 843,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 672,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 672,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 672,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 672,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 672,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 730,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 131,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 1 | ADD | 106,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 2 | ADD | 106,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 3 | ADD | 106,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 4 | ADD | 106,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 5 | ADD | 106,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 6 | ADD | 114,450 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 471,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 469,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 469,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 469,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 469,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 469,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 470,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,212,390 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 1 | ADD | 1,144,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 2 | ADD | 1,144,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 3 | ADD | 1,144,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 4 | ADD | 1,144,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 5 | ADD | 1,144,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 6 | ADD | 1,173,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 2,835,870 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 2,679,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 2,679,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 2,679,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 2,679,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 2,679,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 2,738,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,075,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 854,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 2,933,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 817,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 2,933,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 817,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 2,933,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 817,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 2,933,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 817,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 2,933,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 817,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 3,007,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 836,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 3,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 3,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 6,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 5,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 1 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 2 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 3 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 4 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 5 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 6 | DIV | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 3,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 106,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 65,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 65,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 65,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 65,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 65,280 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 78,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 162,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 138,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 138,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 138,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 138,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 138,090 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 146,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 160,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 154,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 154,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 154,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 154,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 154,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 156,450 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 282,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 282,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 221,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 241,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 241,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 355,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 236,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 261,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 261,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 261,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 261,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 261,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 293,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 194,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,220,970 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,095,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 916,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 822,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 916,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 822,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 916,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 822,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 916,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 822,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 916,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 822,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 1,018,170 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 913,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 113,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 109,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 109,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 109,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 109,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 109,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 110,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 15,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 5,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 5,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 5,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 5,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 5,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 9,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 183,120 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 108,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 108,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 108,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 108,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 108,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 136,560 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,008,750 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 1 | MUL | 984,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 2 | MUL | 984,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 3 | MUL | 984,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 4 | MUL | 984,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 5 | MUL | 984,630 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 6 | MUL | 992,670 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 116,850 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 94,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 94,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 94,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 94,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 94,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 101,850 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 298,140 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 235,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 235,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 235,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 235,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 235,860 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 256,620 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 5,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 1 | MUL | 2,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 2 | MUL | 2,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 3 | MUL | 2,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 4 | MUL | 2,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 5 | MUL | 2,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 6 | MUL | 3,720 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 231,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 171,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 191,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 191,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 20,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 9,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 16,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 16,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 16,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 16,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 16,770 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 24,510 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 13,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 355,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 238,020 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 254,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 168,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 254,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 168,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 254,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 168,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 254,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 168,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 254,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 168,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 288,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 191,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 485,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 161,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 348,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 116,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 348,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 116,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 348,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 116,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 348,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 116,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 348,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 116,100 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 394,020 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 131,340 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 10,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 4,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 4,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 4,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 4,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 4,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 6,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 12,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 8,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 8,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 8,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 8,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 8,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 10,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 116,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 93,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 93,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 93,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 93,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 93,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 101,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 174,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 1 | SUB | 144,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 2 | SUB | 144,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 3 | SUB | 144,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 4 | SUB | 144,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 5 | SUB | 144,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 6 | SUB | 154,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 30,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 29,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 29,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 29,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 29,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 29,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 29,880 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 25,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 2,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 2,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 2,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 2,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 2,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 2,460 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 9,647,010 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 8,506,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 8,506,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 8,506,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 8,506,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 8,506,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 8,924,640 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 608,500 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 459,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 459,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 459,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 459,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 459,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 525,700 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 3,210,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 2,621,650 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 2,621,650 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 2,621,650 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 2,621,650 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 2,621,650 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 2,823,500 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 205,100 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 201,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 201,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 201,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 201,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 201,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 219,200 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,783,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 5,078,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 5,078,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 5,078,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 5,078,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 5,078,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 5,344,750 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,416,400 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 1,276,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 1,276,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 1,276,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 1,276,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 1,276,600 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 1,323,200 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 807,398 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 616,828 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 616,828 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 616,828 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 616,828 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 616,828 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 681,394 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 446,624 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 376,652 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 376,652 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 376,652 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 376,652 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 376,652 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 399,976 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 497,480 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 402,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 248,560 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 187,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 187,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 187,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 187,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 187,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 207,920 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 2,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 1,440 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 1,440 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 1,440 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 1,440 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 1,440 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 1,680 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 774,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 578,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 578,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 578,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 578,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 578,600 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 641,160 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 61,040 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 36,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 36,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 36,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 36,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 36,000 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 45,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 132,240 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 121,840 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 121,840 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 121,840 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 121,840 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 121,840 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 124,760 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 3,343,704 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 2,164,344 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 2,164,344 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 2,164,344 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 2,164,344 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 2,164,344 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 2,592,408 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 6 | PHANTOM | 12 | 
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
| leaf | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 2 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 3 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 4 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 5 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-cache-generator-powers | 6 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 41,328 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 33,264 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 64,008 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 45,864 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 45,864 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 45,864 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 4 | PHANTOM | 45,864 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 5 | PHANTOM | 45,864 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 51,912 | 
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
| leaf | PhantomAir | CT-verify-batch | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 1 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 2 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 3 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 4 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 5 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch | 6 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 2 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 3 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 4 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 5 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 1 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 2 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 3 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 4 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 5 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 6 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 750 | 
| leaf | PhantomAir | HintBitsF | 1 | PHANTOM | 606 | 
| leaf | PhantomAir | HintBitsF | 2 | PHANTOM | 606 | 
| leaf | PhantomAir | HintBitsF | 3 | PHANTOM | 606 | 
| leaf | PhantomAir | HintBitsF | 4 | PHANTOM | 606 | 
| leaf | PhantomAir | HintBitsF | 5 | PHANTOM | 606 | 
| leaf | PhantomAir | HintBitsF | 6 | PHANTOM | 654 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 136,674 | 
| leaf | PhantomAir | HintInputVec | 1 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 2 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 3 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 4 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 5 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 6 | PHANTOM | 133,536 | 
| leaf | VerifyBatchAir | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 10,773 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 20,349 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 19,152 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 19,152 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 19,152 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 19,152 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 19,152 | 
| leaf | VerifyBatchAir | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 19,551 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 1 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 2 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 3 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 4 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 5 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchExt | 6 | VERIFY_BATCH | 4,524,660 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 6,451,830 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 5,178,222 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 5,178,222 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 5,178,222 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 5,178,222 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 5,178,222 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 5,630,688 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,428 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 103,408 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 115 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 32,223 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,474 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,312 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 491,763 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,381 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,139 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,986,319 | 
| root | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 68,250 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 30 | 
| root | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 312,350 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,104 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 44,160 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 33,000 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 1,135,200 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 147,000 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 375,150 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,259,010 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 2,774,040 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,089,820 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 854,130 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 4,440 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,440 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivF | 0 | DIV | 173,880 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,730 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 147,120 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 172,800 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 163,410 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 323,820 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 323,820 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 404,880 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 269,430 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,370,790 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,223,820 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 150,720 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 24,600 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 205,800 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 834,660 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 130,710 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 330,690 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 10,200 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 270,900 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 270,900 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 22,980 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 11,280 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 413,400 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 277,830 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 509,940 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 169,980 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 11,520 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 44,880 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 0 | SUB | 240 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 130,200 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 190,770 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 31,320 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 26,460 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,420 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 9,866,250 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 698,125 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 3,388,950 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 173,225 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,847,825 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,476,875 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 893,758 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 495,414 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 588,720 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 261,760 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,480 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,119,200 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 68,600 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 146,640 | 
| root | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 3,468,192 | 
| root | PhantomAir | CT-ExtractPublicValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 48,888 | 
| root | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 67,536 | 
| root | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-verify-batch | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,584 | 
| root | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| root | PhantomAir | HintBitsF | 0 | PHANTOM | 840 | 
| root | PhantomAir | HintInputVec | 0 | PHANTOM | 138,156 | 
| root | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 4,788 | 
| root | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 21,546 | 
| root | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,926,852 | 
| root | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 5,815,026 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 37,747,008 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 1 | 37,746,000 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 2 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 3 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 4 | 37,746,036 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 5 | 37,746,000 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 6 | 32,724,396 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 72 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 36 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 144 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 12,931,389 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 1 | 12,931,537 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 2 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 3 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 4 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 5 | 12,931,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 6 | 11,211,111 | 
| fib_e2e | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 106 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 3,029,052 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 1 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 2 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 3 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 4 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 5 | 3,029,026 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 6 | 2,626,026 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 3,029,078 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 1 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 2 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 3 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 4 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 5 | 3,029,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 6 | 2,626,026 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 96 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 64 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 2,096,982 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 1 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 2 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 3 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 4 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 5 | 2,097,000 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 6 | 1,818,018 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 162 | 
| fib_e2e | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 78 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 336 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 6 | 28 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 400 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 6 | 120 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 520 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 6 | 80 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 168 | 
| fib_e2e | PhantomAir |  | PHANTOM | 0 | 12 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<2> | 3 | 262,144 |  | 16 | 11 | 7,077,888 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<4> | 3 | 131,072 |  | 16 | 13 | 3,801,088 | 
| internal.0 | AccessAdapterAir<8> | 0 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 1 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 2 | 512 |  | 16 | 17 | 16,896 | 
| internal.0 | AccessAdapterAir<8> | 3 | 256 |  | 16 | 17 | 8,448 | 
| internal.0 | FriReducedOpeningAir | 0 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 1 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 2 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.0 | FriReducedOpeningAir | 3 | 262,144 |  | 36 | 26 | 16,252,928 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 356 | 399 | 24,739,840 | 
| internal.0 | PhantomAir | 0 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 1 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 2 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.0 | PhantomAir | 3 | 65,536 |  | 8 | 6 | 917,504 | 
| internal.0 | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 1 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 2 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | ProgramAir | 3 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 36 | 25 | 31,981,568 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 36 | 34 | 4,587,520 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 40 | 3,932,160 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VolatileBoundaryAir | 0 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 1 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 2 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.0 | VolatileBoundaryAir | 3 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.1 | AccessAdapterAir<8> | 4 | 512 |  | 16 | 17 | 16,896 | 
| internal.1 | AccessAdapterAir<8> | 5 | 512 |  | 16 | 17 | 16,896 | 
| internal.1 | FriReducedOpeningAir | 4 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.1 | FriReducedOpeningAir | 5 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.1 | PhantomAir | 4 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.1 | PhantomAir | 5 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.1 | ProgramAir | 4 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | ProgramAir | 5 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.1 | VolatileBoundaryAir | 5 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.2 | AccessAdapterAir<8> | 6 | 512 |  | 16 | 17 | 16,896 | 
| internal.2 | FriReducedOpeningAir | 6 | 524,288 |  | 36 | 26 | 32,505,856 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 356 | 399 | 49,479,680 | 
| internal.2 | PhantomAir | 6 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.2 | ProgramAir | 6 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<2> | 1 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 2 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 3 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 4 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 5 | 131,072 |  | 16 | 11 | 3,538,944 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 16 | 11 | 7,077,888 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<4> | 1 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 2 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 3 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 4 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 5 | 65,536 |  | 16 | 13 | 1,900,544 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 16 | 13 | 3,801,088 | 
| leaf | AccessAdapterAir<8> | 0 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 1 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 2 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 3 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 4 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 5 | 256 |  | 16 | 17 | 8,448 | 
| leaf | AccessAdapterAir<8> | 6 | 512 |  | 16 | 17 | 16,896 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 36 | 26 | 8,126,464 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 356 | 399 | 24,739,840 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 1 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 2 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 3 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 4 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 5 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 6 | 65,536 |  | 8 | 6 | 917,504 | 
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
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 524,288 |  | 36 | 25 | 31,981,568 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 65,536 |  | 20 | 40 | 3,932,160 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 2 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 3 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 4 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 5 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | VolatileBoundaryAir | 6 | 524,288 |  | 8 | 11 | 9,961,472 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 16 | 11 | 7,077,888 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 16 | 13 | 3,801,088 | 
| root | AccessAdapterAir<8> | 0 | 256 |  | 16 | 17 | 8,448 | 
| root | FriReducedOpeningAir | 0 | 262,144 |  | 36 | 26 | 16,252,928 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 356 | 399 | 24,739,840 | 
| root | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| root | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 28 | 23 | 26,738,688 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 10 | 1,441,792 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 30 | 52,428,800 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 36 | 25 | 31,981,568 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 40 | 3,932,160 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| root | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 24 | 17 | 656 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 32 |  | 24 | 17 | 1,312 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 20 | 32 | 13,312 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 20 | 32 | 6,656 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 20 | 32 | 13,312 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 12 | 20 | 512 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 32 |  | 12 | 20 | 1,024 | 
| fib_e2e | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fib_e2e | PhantomAir | 1 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 2 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 3 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 4 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 5 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | PhantomAir | 6 | 1 |  | 12 | 6 | 18 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 128 |  | 8 | 300 | 39,424 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 6 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 1 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 2 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 3 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 4 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 5 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | ProgramAir | 6 | 4,096 |  | 8 | 10 | 73,728 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fib_e2e | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 1 |  | 36 | 28 | 64 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 8 |  | 72 | 40 | 896 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 12 | 4 | 32 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 12 | 4 | 32 | 

| group | cell_tracker_span | simple_advice_cells | lookup_advice_cells |
| --- | --- | --- | --- |
| halo2_outer | VerifierProgram | 874,615 | 244,301 | 
| halo2_outer | VerifierProgram;PoseidonCell | 20,120 |  | 
| halo2_outer | VerifierProgram;stage-c-build-rounds | 334,819 | 697 | 
| halo2_outer | VerifierProgram;stage-c-build-rounds;PoseidonCell | 47,785 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs | 161 | 40 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 561,251 | 2,686 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 67,905 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;cache-generator-powers | 585,396 | 99,036 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 13,740,258 | 439,068 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;pre-compute-alpha-pows | 76,596 | 11,168 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 119,028 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 14,110,236 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 13,862,898 | 396,648 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 1,500,954 | 247,758 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 251,160 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 24,401,160 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 2,469,684 | 76,860 | 
| halo2_outer | VerifierProgram;stage-e-verify-constraints | 3,287,639 | 531,801 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 635,703 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 635,595 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 635,595 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 317,903 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 0 | 73,025 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 1 | 72,914 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 2 | 72,901 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 3 | 36,657 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 52 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,822,681 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 1 | 1,822,565 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 2 | 1,822,565 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 3 | 911,631 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 895,808 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 895,808 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 895,808 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 448,062 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 80,874 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 80,874 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 80,874 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 40,437 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 107,908 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 107,808 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 107,808 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 54,004 | 
| internal.0 | AccessAdapter<2> | 0 | 373,884 | 
| internal.0 | AccessAdapter<2> | 1 | 373,884 | 
| internal.0 | AccessAdapter<2> | 2 | 373,884 | 
| internal.0 | AccessAdapter<2> | 3 | 210,428 | 
| internal.0 | AccessAdapter<4> | 0 | 177,118 | 
| internal.0 | AccessAdapter<4> | 1 | 177,118 | 
| internal.0 | AccessAdapter<4> | 2 | 177,118 | 
| internal.0 | AccessAdapter<4> | 3 | 100,302 | 
| internal.0 | AccessAdapter<8> | 0 | 312 | 
| internal.0 | AccessAdapter<8> | 1 | 312 | 
| internal.0 | AccessAdapter<8> | 2 | 312 | 
| internal.0 | AccessAdapter<8> | 3 | 156 | 
| internal.0 | Boundary | 0 | 636,325 | 
| internal.0 | Boundary | 1 | 636,325 | 
| internal.0 | Boundary | 2 | 636,325 | 
| internal.0 | Boundary | 3 | 373,747 | 
| internal.0 | FriReducedOpeningAir | 0 | 266,784 | 
| internal.0 | FriReducedOpeningAir | 1 | 266,784 | 
| internal.0 | FriReducedOpeningAir | 2 | 266,784 | 
| internal.0 | FriReducedOpeningAir | 3 | 133,392 | 
| internal.0 | PhantomAir | 0 | 89,285 | 
| internal.0 | PhantomAir | 1 | 89,285 | 
| internal.0 | PhantomAir | 2 | 89,285 | 
| internal.0 | PhantomAir | 3 | 44,650 | 
| internal.0 | ProgramChip | 0 | 149,664 | 
| internal.0 | ProgramChip | 1 | 149,664 | 
| internal.0 | ProgramChip | 2 | 149,664 | 
| internal.0 | ProgramChip | 3 | 149,664 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 
| internal.0 | VerifyBatchAir | 0 | 51,720 | 
| internal.0 | VerifyBatchAir | 1 | 51,762 | 
| internal.0 | VerifyBatchAir | 2 | 51,762 | 
| internal.0 | VerifyBatchAir | 3 | 25,797 | 
| internal.0 | VmConnectorAir | 0 | 2 | 
| internal.0 | VmConnectorAir | 1 | 2 | 
| internal.0 | VmConnectorAir | 2 | 2 | 
| internal.0 | VmConnectorAir | 3 | 2 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 663,747 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 650,285 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 4 | 76,334 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 5 | 74,658 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 52 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 52 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 4 | 1,883,737 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 5 | 1,854,407 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 925,856 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 911,416 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 81,716 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 81,295 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 109,620 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 108,888 | 
| internal.1 | AccessAdapter<2> | 4 | 378,072 | 
| internal.1 | AccessAdapter<2> | 5 | 384,960 | 
| internal.1 | AccessAdapter<4> | 4 | 179,128 | 
| internal.1 | AccessAdapter<4> | 5 | 183,222 | 
| internal.1 | AccessAdapter<8> | 4 | 328 | 
| internal.1 | AccessAdapter<8> | 5 | 324 | 
| internal.1 | Boundary | 4 | 662,389 | 
| internal.1 | Boundary | 5 | 680,989 | 
| internal.1 | FriReducedOpeningAir | 4 | 266,784 | 
| internal.1 | FriReducedOpeningAir | 5 | 266,784 | 
| internal.1 | PhantomAir | 4 | 91,891 | 
| internal.1 | PhantomAir | 5 | 90,630 | 
| internal.1 | ProgramChip | 4 | 149,664 | 
| internal.1 | ProgramChip | 5 | 149,664 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 
| internal.1 | VerifyBatchAir | 4 | 53,952 | 
| internal.1 | VerifyBatchAir | 5 | 52,817 | 
| internal.1 | VmConnectorAir | 4 | 2 | 
| internal.1 | VmConnectorAir | 5 | 2 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 663,747 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | 6 | 76,242 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 52 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 6 | 1,883,737 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 925,856 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 81,716 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 109,620 | 
| internal.2 | AccessAdapter<2> | 6 | 378,072 | 
| internal.2 | AccessAdapter<4> | 6 | 179,128 | 
| internal.2 | AccessAdapter<8> | 6 | 328 | 
| internal.2 | Boundary | 6 | 662,389 | 
| internal.2 | FriReducedOpeningAir | 6 | 266,784 | 
| internal.2 | PhantomAir | 6 | 91,891 | 
| internal.2 | ProgramChip | 6 | 149,664 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 
| internal.2 | VerifyBatchAir | 6 | 53,952 | 
| internal.2 | VmConnectorAir | 6 | 2 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 326,438 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 290,201 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 290,201 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 290,201 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 290,201 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 290,201 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 303,555 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 38,797 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 1 | 36,772 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 2 | 37,445 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 3 | 37,433 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 4 | 37,563 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 5 | 36,963 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 6 | 38,320 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 903,197 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 1 | 789,053 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 2 | 789,053 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 3 | 789,053 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 4 | 789,053 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 5 | 789,053 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 6 | 830,551 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 448,930 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 385,546 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 385,546 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 385,546 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 385,546 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 385,546 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 409,454 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 36,883 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 29,220 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 29,220 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 29,220 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 29,220 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 29,220 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 31,805 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 42,891 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 32,022 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 32,022 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 32,022 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 32,022 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 32,022 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 35,591 | 
| leaf | AccessAdapter<2> | 0 | 161,922 | 
| leaf | AccessAdapter<2> | 1 | 128,276 | 
| leaf | AccessAdapter<2> | 2 | 128,276 | 
| leaf | AccessAdapter<2> | 3 | 128,276 | 
| leaf | AccessAdapter<2> | 4 | 128,276 | 
| leaf | AccessAdapter<2> | 5 | 128,276 | 
| leaf | AccessAdapter<2> | 6 | 140,350 | 
| leaf | AccessAdapter<4> | 0 | 76,678 | 
| leaf | AccessAdapter<4> | 1 | 60,864 | 
| leaf | AccessAdapter<4> | 2 | 60,864 | 
| leaf | AccessAdapter<4> | 3 | 60,864 | 
| leaf | AccessAdapter<4> | 4 | 60,864 | 
| leaf | AccessAdapter<4> | 5 | 60,864 | 
| leaf | AccessAdapter<4> | 6 | 66,564 | 
| leaf | AccessAdapter<8> | 0 | 160 | 
| leaf | AccessAdapter<8> | 1 | 154 | 
| leaf | AccessAdapter<8> | 2 | 154 | 
| leaf | AccessAdapter<8> | 3 | 154 | 
| leaf | AccessAdapter<8> | 4 | 154 | 
| leaf | AccessAdapter<8> | 5 | 154 | 
| leaf | AccessAdapter<8> | 6 | 318 | 
| leaf | Boundary | 0 | 339,499 | 
| leaf | Boundary | 1 | 305,473 | 
| leaf | Boundary | 2 | 305,473 | 
| leaf | Boundary | 3 | 305,473 | 
| leaf | Boundary | 4 | 305,473 | 
| leaf | Boundary | 5 | 305,473 | 
| leaf | Boundary | 6 | 318,117 | 
| leaf | FriReducedOpeningAir | 0 | 128,604 | 
| leaf | FriReducedOpeningAir | 1 | 83,244 | 
| leaf | FriReducedOpeningAir | 2 | 83,244 | 
| leaf | FriReducedOpeningAir | 3 | 83,244 | 
| leaf | FriReducedOpeningAir | 4 | 83,244 | 
| leaf | FriReducedOpeningAir | 5 | 83,244 | 
| leaf | FriReducedOpeningAir | 6 | 99,708 | 
| leaf | PhantomAir | 0 | 44,258 | 
| leaf | PhantomAir | 1 | 38,072 | 
| leaf | PhantomAir | 2 | 38,072 | 
| leaf | PhantomAir | 3 | 38,072 | 
| leaf | PhantomAir | 4 | 38,072 | 
| leaf | PhantomAir | 5 | 38,072 | 
| leaf | PhantomAir | 6 | 40,359 | 
| leaf | ProgramChip | 0 | 75,563 | 
| leaf | ProgramChip | 1 | 75,563 | 
| leaf | ProgramChip | 2 | 75,563 | 
| leaf | ProgramChip | 3 | 75,563 | 
| leaf | ProgramChip | 4 | 75,563 | 
| leaf | ProgramChip | 5 | 75,563 | 
| leaf | ProgramChip | 6 | 75,563 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 27,561 | 
| leaf | VerifyBatchAir | 1 | 24,366 | 
| leaf | VerifyBatchAir | 2 | 24,366 | 
| leaf | VerifyBatchAir | 3 | 24,366 | 
| leaf | VerifyBatchAir | 4 | 24,366 | 
| leaf | VerifyBatchAir | 5 | 24,366 | 
| leaf | VerifyBatchAir | 6 | 25,528 | 
| leaf | VmConnectorAir | 0 | 2 | 
| leaf | VmConnectorAir | 1 | 2 | 
| leaf | VmConnectorAir | 2 | 2 | 
| leaf | VmConnectorAir | 3 | 2 | 
| leaf | VmConnectorAir | 4 | 2 | 
| leaf | VmConnectorAir | 5 | 2 | 
| leaf | VmConnectorAir | 6 | 2 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 331,899 | 
| root | <JalNativeAdapterAir,JalCoreAir> | 0 | 38,064 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 48 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 942,122 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 463,400 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 40,858 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 54,810 | 
| root | AccessAdapter<2> | 0 | 212,724 | 
| root | AccessAdapter<4> | 0 | 101,408 | 
| root | AccessAdapter<8> | 0 | 230 | 
| root | Boundary | 0 | 387,171 | 
| root | FriReducedOpeningAir | 0 | 133,392 | 
| root | PhantomAir | 0 | 45,948 | 
| root | ProgramChip | 0 | 149,937 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 
| root | VerifyBatchAir | 0 | 26,988 | 
| root | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,048,537 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 1 | 1,048,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 2 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 3 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 4 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 5 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 6 | 909,012 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 349,497 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 1 | 349,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 2 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 3 | 349,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 4 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 5 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 6 | 303,003 | 
| fib_e2e | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 2 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 233,005 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 1 | 233,001 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 2 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 3 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 4 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 5 | 233,001 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 6 | 202,002 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 5 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 116,508 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 1 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 2 | 116,501 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 3 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 4 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 5 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 6 | 101,001 | 
| fib_e2e | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 3 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 12 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 6 | 1 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 23 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 6 | 5 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 9 | 
| fib_e2e | AccessAdapter<8> | 0 | 34 | 
| fib_e2e | AccessAdapter<8> | 1 | 16 | 
| fib_e2e | AccessAdapter<8> | 2 | 16 | 
| fib_e2e | AccessAdapter<8> | 3 | 16 | 
| fib_e2e | AccessAdapter<8> | 4 | 16 | 
| fib_e2e | AccessAdapter<8> | 5 | 16 | 
| fib_e2e | AccessAdapter<8> | 6 | 24 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 0 | 150 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 1 | 82 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 2 | 82 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 3 | 83 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 4 | 84 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 5 | 84 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 6 | 169 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 
| fib_e2e | Boundary | 0 | 34 | 
| fib_e2e | Boundary | 1 | 16 | 
| fib_e2e | Boundary | 2 | 16 | 
| fib_e2e | Boundary | 3 | 16 | 
| fib_e2e | Boundary | 4 | 16 | 
| fib_e2e | Boundary | 5 | 16 | 
| fib_e2e | Boundary | 6 | 24 | 
| fib_e2e | Merkle | 0 | 174 | 
| fib_e2e | Merkle | 1 | 72 | 
| fib_e2e | Merkle | 2 | 72 | 
| fib_e2e | Merkle | 3 | 72 | 
| fib_e2e | Merkle | 4 | 72 | 
| fib_e2e | Merkle | 5 | 72 | 
| fib_e2e | Merkle | 6 | 178 | 
| fib_e2e | PhantomAir | 0 | 2 | 
| fib_e2e | ProgramChip | 0 | 3,275 | 
| fib_e2e | ProgramChip | 1 | 3,275 | 
| fib_e2e | ProgramChip | 2 | 3,275 | 
| fib_e2e | ProgramChip | 3 | 3,275 | 
| fib_e2e | ProgramChip | 4 | 3,275 | 
| fib_e2e | ProgramChip | 5 | 3,275 | 
| fib_e2e | ProgramChip | 6 | 3,275 | 
| fib_e2e | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 1 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 2 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 3 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 4 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 5 | 524,288 | 
| fib_e2e | RangeTupleCheckerAir<2> | 6 | 524,288 | 
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
| internal.0 | AddE | 0 | FE4ADD | 29,230 | 
| internal.0 | AddE | 1 | FE4ADD | 29,230 | 
| internal.0 | AddE | 2 | FE4ADD | 29,230 | 
| internal.0 | AddE | 3 | FE4ADD | 14,615 | 
| internal.0 | AddEFFI | 0 | ADD | 2,784 | 
| internal.0 | AddEFFI | 1 | ADD | 2,784 | 
| internal.0 | AddEFFI | 2 | ADD | 2,784 | 
| internal.0 | AddEFFI | 3 | ADD | 1,392 | 
| internal.0 | AddEFI | 0 | ADD | 2,200 | 
| internal.0 | AddEFI | 1 | ADD | 2,200 | 
| internal.0 | AddEFI | 2 | ADD | 2,200 | 
| internal.0 | AddEFI | 3 | ADD | 1,100 | 
| internal.0 | AddEI | 0 | ADD | 74,528 | 
| internal.0 | AddEI | 1 | ADD | 74,528 | 
| internal.0 | AddEI | 2 | ADD | 74,528 | 
| internal.0 | AddEI | 3 | ADD | 37,264 | 
| internal.0 | AddF | 0 | ADD | 9,800 | 
| internal.0 | AddF | 1 | ADD | 9,800 | 
| internal.0 | AddF | 2 | ADD | 9,800 | 
| internal.0 | AddF | 3 | ADD | 4,900 | 
| internal.0 | AddFI | 0 | ADD | 23,913 | 
| internal.0 | AddFI | 1 | ADD | 23,913 | 
| internal.0 | AddFI | 2 | ADD | 23,913 | 
| internal.0 | AddFI | 3 | ADD | 11,969 | 
| internal.0 | AddV | 0 | ADD | 79,416 | 
| internal.0 | AddV | 1 | ADD | 79,408 | 
| internal.0 | AddV | 2 | ADD | 79,408 | 
| internal.0 | AddV | 3 | ADD | 39,717 | 
| internal.0 | AddVI | 0 | ADD | 176,386 | 
| internal.0 | AddVI | 1 | ADD | 176,386 | 
| internal.0 | AddVI | 2 | ADD | 176,386 | 
| internal.0 | AddVI | 3 | ADD | 88,224 | 
| internal.0 | Alloc | 0 | ADD | 195,526 | 
| internal.0 | Alloc | 0 | MUL | 54,144 | 
| internal.0 | Alloc | 1 | ADD | 195,526 | 
| internal.0 | Alloc | 1 | MUL | 54,144 | 
| internal.0 | Alloc | 2 | ADD | 195,526 | 
| internal.0 | Alloc | 2 | MUL | 54,144 | 
| internal.0 | Alloc | 3 | ADD | 97,786 | 
| internal.0 | Alloc | 3 | MUL | 27,079 | 
| internal.0 | AssertEqE | 0 | BNE | 472 | 
| internal.0 | AssertEqE | 1 | BNE | 472 | 
| internal.0 | AssertEqE | 2 | BNE | 472 | 
| internal.0 | AssertEqE | 3 | BNE | 236 | 
| internal.0 | AssertEqEI | 0 | BNE | 8 | 
| internal.0 | AssertEqEI | 1 | BNE | 8 | 
| internal.0 | AssertEqEI | 2 | BNE | 8 | 
| internal.0 | AssertEqEI | 3 | BNE | 4 | 
| internal.0 | AssertEqF | 0 | BNE | 8,985 | 
| internal.0 | AssertEqF | 1 | BNE | 8,985 | 
| internal.0 | AssertEqF | 2 | BNE | 8,985 | 
| internal.0 | AssertEqF | 3 | BNE | 4,480 | 
| internal.0 | AssertEqFI | 0 | BNE | 7 | 
| internal.0 | AssertEqFI | 1 | BNE | 7 | 
| internal.0 | AssertEqFI | 2 | BNE | 7 | 
| internal.0 | AssertEqFI | 3 | BNE | 3 | 
| internal.0 | AssertEqV | 0 | BNE | 2,718 | 
| internal.0 | AssertEqV | 1 | BNE | 2,718 | 
| internal.0 | AssertEqV | 2 | BNE | 2,718 | 
| internal.0 | AssertEqV | 3 | BNE | 1,359 | 
| internal.0 | AssertEqVI | 0 | BNE | 474 | 
| internal.0 | AssertEqVI | 1 | BNE | 474 | 
| internal.0 | AssertEqVI | 2 | BNE | 474 | 
| internal.0 | AssertEqVI | 3 | BNE | 237 | 
| internal.0 | AssertNonZero | 0 | BEQ | 1 | 
| internal.0 | AssertNonZero | 1 | BEQ | 1 | 
| internal.0 | AssertNonZero | 2 | BEQ | 1 | 
| internal.0 | AssertNonZero | 3 | BEQ | 1 | 
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
| internal.0 | CT-cache-generator-powers | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 3 | PHANTOM | 504 | 
| internal.0 | CT-compute-reduced-opening | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 3 | PHANTOM | 504 | 
| internal.0 | CT-exp-reverse-bits-len | 0 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 1 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 2 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 3 | PHANTOM | 8,148 | 
| internal.0 | CT-pre-compute-alpha-pows | 0 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 1 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 2 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 3 | PHANTOM | 2 | 
| internal.0 | CT-single-reduced-opening-eval | 0 | PHANTOM | 22,512 | 
| internal.0 | CT-single-reduced-opening-eval | 1 | PHANTOM | 22,512 | 
| internal.0 | CT-single-reduced-opening-eval | 2 | PHANTOM | 22,512 | 
| internal.0 | CT-single-reduced-opening-eval | 3 | PHANTOM | 11,256 | 
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
| internal.0 | CT-verify-batch | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-verify-batch | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-verify-batch | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-verify-batch | 3 | PHANTOM | 504 | 
| internal.0 | CT-verify-batch-ext | 0 | PHANTOM | 3,360 | 
| internal.0 | CT-verify-batch-ext | 1 | PHANTOM | 3,360 | 
| internal.0 | CT-verify-batch-ext | 2 | PHANTOM | 3,360 | 
| internal.0 | CT-verify-batch-ext | 3 | PHANTOM | 1,680 | 
| internal.0 | CT-verify-query | 0 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 1 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 2 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 3 | PHANTOM | 84 | 
| internal.0 | CastFV | 0 | ADD | 296 | 
| internal.0 | CastFV | 1 | ADD | 296 | 
| internal.0 | CastFV | 2 | ADD | 296 | 
| internal.0 | CastFV | 3 | ADD | 148 | 
| internal.0 | DivE | 0 | BBE4DIV | 13,004 | 
| internal.0 | DivE | 1 | BBE4DIV | 13,004 | 
| internal.0 | DivE | 2 | BBE4DIV | 13,004 | 
| internal.0 | DivE | 3 | BBE4DIV | 6,502 | 
| internal.0 | DivEIN | 0 | ADD | 1,496 | 
| internal.0 | DivEIN | 0 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 1 | ADD | 1,496 | 
| internal.0 | DivEIN | 1 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 2 | ADD | 1,496 | 
| internal.0 | DivEIN | 2 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 3 | ADD | 748 | 
| internal.0 | DivEIN | 3 | BBE4DIV | 187 | 
| internal.0 | DivF | 0 | DIV | 11,088 | 
| internal.0 | DivF | 1 | DIV | 11,088 | 
| internal.0 | DivF | 2 | DIV | 11,088 | 
| internal.0 | DivF | 3 | DIV | 5,544 | 
| internal.0 | DivFIN | 0 | DIV | 782 | 
| internal.0 | DivFIN | 1 | DIV | 782 | 
| internal.0 | DivFIN | 2 | DIV | 782 | 
| internal.0 | DivFIN | 3 | DIV | 391 | 
| internal.0 | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 5,628 | 
| internal.0 | HintBitsF | 0 | PHANTOM | 280 | 
| internal.0 | HintBitsF | 1 | PHANTOM | 280 | 
| internal.0 | HintBitsF | 2 | PHANTOM | 280 | 
| internal.0 | HintBitsF | 3 | PHANTOM | 140 | 
| internal.0 | HintInputVec | 0 | PHANTOM | 43,619 | 
| internal.0 | HintInputVec | 1 | PHANTOM | 43,619 | 
| internal.0 | HintInputVec | 2 | PHANTOM | 43,619 | 
| internal.0 | HintInputVec | 3 | PHANTOM | 21,814 | 
| internal.0 | IfEq | 0 | BNE | 280 | 
| internal.0 | IfEq | 1 | BNE | 280 | 
| internal.0 | IfEq | 2 | BNE | 280 | 
| internal.0 | IfEq | 3 | BNE | 140 | 
| internal.0 | IfEqI | 0 | BNE | 41,538 | 
| internal.0 | IfEqI | 0 | JAL | 13,151 | 
| internal.0 | IfEqI | 1 | BNE | 41,538 | 
| internal.0 | IfEqI | 1 | JAL | 13,040 | 
| internal.0 | IfEqI | 2 | BNE | 41,538 | 
| internal.0 | IfEqI | 2 | JAL | 13,027 | 
| internal.0 | IfEqI | 3 | BNE | 20,769 | 
| internal.0 | IfEqI | 3 | JAL | 6,714 | 
| internal.0 | IfNe | 0 | BEQ | 286 | 
| internal.0 | IfNe | 0 | JAL | 6 | 
| internal.0 | IfNe | 1 | BEQ | 286 | 
| internal.0 | IfNe | 1 | JAL | 6 | 
| internal.0 | IfNe | 2 | BEQ | 286 | 
| internal.0 | IfNe | 2 | JAL | 6 | 
| internal.0 | IfNe | 3 | BEQ | 143 | 
| internal.0 | IfNe | 3 | JAL | 3 | 
| internal.0 | IfNeI | 0 | BEQ | 186 | 
| internal.0 | IfNeI | 1 | BEQ | 186 | 
| internal.0 | IfNeI | 2 | BEQ | 186 | 
| internal.0 | IfNeI | 3 | BEQ | 93 | 
| internal.0 | ImmE | 0 | ADD | 9,680 | 
| internal.0 | ImmE | 1 | ADD | 9,680 | 
| internal.0 | ImmE | 2 | ADD | 9,680 | 
| internal.0 | ImmE | 3 | ADD | 4,840 | 
| internal.0 | ImmF | 0 | ADD | 11,334 | 
| internal.0 | ImmF | 1 | ADD | 11,334 | 
| internal.0 | ImmF | 2 | ADD | 11,334 | 
| internal.0 | ImmF | 3 | ADD | 5,751 | 
| internal.0 | ImmV | 0 | ADD | 10,449 | 
| internal.0 | ImmV | 1 | ADD | 10,449 | 
| internal.0 | ImmV | 2 | ADD | 10,449 | 
| internal.0 | ImmV | 3 | ADD | 5,283 | 
| internal.0 | LoadE | 0 | ADD | 21,336 | 
| internal.0 | LoadE | 0 | LOADW | 52,070 | 
| internal.0 | LoadE | 0 | MUL | 21,336 | 
| internal.0 | LoadE | 1 | ADD | 21,336 | 
| internal.0 | LoadE | 1 | LOADW | 52,070 | 
| internal.0 | LoadE | 1 | MUL | 21,336 | 
| internal.0 | LoadE | 2 | ADD | 21,336 | 
| internal.0 | LoadE | 2 | LOADW | 52,070 | 
| internal.0 | LoadE | 2 | MUL | 21,336 | 
| internal.0 | LoadE | 3 | ADD | 10,668 | 
| internal.0 | LoadE | 3 | LOADW | 26,035 | 
| internal.0 | LoadE | 3 | MUL | 10,668 | 
| internal.0 | LoadF | 0 | ADD | 26,984 | 
| internal.0 | LoadF | 0 | LOADW | 55,382 | 
| internal.0 | LoadF | 0 | MUL | 17,962 | 
| internal.0 | LoadF | 1 | ADD | 26,984 | 
| internal.0 | LoadF | 1 | LOADW | 55,382 | 
| internal.0 | LoadF | 1 | MUL | 17,962 | 
| internal.0 | LoadF | 2 | ADD | 26,984 | 
| internal.0 | LoadF | 2 | LOADW | 55,382 | 
| internal.0 | LoadF | 2 | MUL | 17,962 | 
| internal.0 | LoadF | 3 | ADD | 13,492 | 
| internal.0 | LoadF | 3 | LOADW | 27,695 | 
| internal.0 | LoadF | 3 | MUL | 8,981 | 
| internal.0 | LoadHeapPtr | 0 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 1 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 2 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 3 | ADD | 1 | 
| internal.0 | LoadV | 0 | ADD | 90,966 | 
| internal.0 | LoadV | 0 | LOADW | 266,998 | 
| internal.0 | LoadV | 0 | MUL | 81,252 | 
| internal.0 | LoadV | 1 | ADD | 90,966 | 
| internal.0 | LoadV | 1 | LOADW | 266,998 | 
| internal.0 | LoadV | 1 | MUL | 81,252 | 
| internal.0 | LoadV | 2 | ADD | 90,966 | 
| internal.0 | LoadV | 2 | LOADW | 266,998 | 
| internal.0 | LoadV | 2 | MUL | 81,252 | 
| internal.0 | LoadV | 3 | ADD | 45,483 | 
| internal.0 | LoadV | 3 | LOADW | 133,504 | 
| internal.0 | LoadV | 3 | MUL | 40,626 | 
| internal.0 | MulE | 0 | BBE4MUL | 54,798 | 
| internal.0 | MulE | 1 | BBE4MUL | 54,698 | 
| internal.0 | MulE | 2 | BBE4MUL | 54,698 | 
| internal.0 | MulE | 3 | BBE4MUL | 27,449 | 
| internal.0 | MulEF | 0 | MUL | 9,712 | 
| internal.0 | MulEF | 1 | MUL | 9,712 | 
| internal.0 | MulEF | 2 | MUL | 9,712 | 
| internal.0 | MulEF | 3 | MUL | 4,856 | 
| internal.0 | MulEFI | 0 | MUL | 1,640 | 
| internal.0 | MulEFI | 1 | MUL | 1,640 | 
| internal.0 | MulEFI | 2 | MUL | 1,640 | 
| internal.0 | MulEFI | 3 | MUL | 820 | 
| internal.0 | MulEI | 0 | ADD | 13,688 | 
| internal.0 | MulEI | 0 | BBE4MUL | 3,422 | 
| internal.0 | MulEI | 1 | ADD | 13,688 | 
| internal.0 | MulEI | 1 | BBE4MUL | 3,422 | 
| internal.0 | MulEI | 2 | ADD | 13,688 | 
| internal.0 | MulEI | 2 | BBE4MUL | 3,422 | 
| internal.0 | MulEI | 3 | ADD | 6,844 | 
| internal.0 | MulEI | 3 | BBE4MUL | 1,711 | 
| internal.0 | MulF | 0 | MUL | 53,628 | 
| internal.0 | MulF | 1 | MUL | 53,628 | 
| internal.0 | MulF | 2 | MUL | 53,628 | 
| internal.0 | MulF | 3 | MUL | 26,814 | 
| internal.0 | MulFI | 0 | MUL | 8,714 | 
| internal.0 | MulFI | 1 | MUL | 8,714 | 
| internal.0 | MulFI | 2 | MUL | 8,714 | 
| internal.0 | MulFI | 3 | MUL | 4,357 | 
| internal.0 | MulVI | 0 | MUL | 21,877 | 
| internal.0 | MulVI | 1 | MUL | 21,877 | 
| internal.0 | MulVI | 2 | MUL | 21,877 | 
| internal.0 | MulVI | 3 | MUL | 10,939 | 
| internal.0 | NegE | 0 | MUL | 680 | 
| internal.0 | NegE | 1 | MUL | 680 | 
| internal.0 | NegE | 2 | MUL | 680 | 
| internal.0 | NegE | 3 | MUL | 340 | 
| internal.0 | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 51 | 
| internal.0 | Publish | 0 | PUBLISH | 52 | 
| internal.0 | Publish | 1 | PUBLISH | 52 | 
| internal.0 | Publish | 2 | PUBLISH | 52 | 
| internal.0 | Publish | 3 | PUBLISH | 52 | 
| internal.0 | StoreE | 0 | ADD | 17,976 | 
| internal.0 | StoreE | 0 | MUL | 17,976 | 
| internal.0 | StoreE | 0 | STOREW | 28,804 | 
| internal.0 | StoreE | 1 | ADD | 17,976 | 
| internal.0 | StoreE | 1 | MUL | 17,976 | 
| internal.0 | StoreE | 1 | STOREW | 28,804 | 
| internal.0 | StoreE | 2 | ADD | 17,976 | 
| internal.0 | StoreE | 2 | MUL | 17,976 | 
| internal.0 | StoreE | 2 | STOREW | 28,804 | 
| internal.0 | StoreE | 3 | ADD | 8,988 | 
| internal.0 | StoreE | 3 | MUL | 8,988 | 
| internal.0 | StoreE | 3 | STOREW | 14,402 | 
| internal.0 | StoreF | 0 | ADD | 1,484 | 
| internal.0 | StoreF | 0 | MUL | 752 | 
| internal.0 | StoreF | 0 | STOREW | 12,786 | 
| internal.0 | StoreF | 1 | ADD | 1,484 | 
| internal.0 | StoreF | 1 | MUL | 752 | 
| internal.0 | StoreF | 1 | STOREW | 12,786 | 
| internal.0 | StoreF | 2 | ADD | 1,484 | 
| internal.0 | StoreF | 2 | MUL | 752 | 
| internal.0 | StoreF | 2 | STOREW | 12,786 | 
| internal.0 | StoreF | 3 | ADD | 742 | 
| internal.0 | StoreF | 3 | MUL | 376 | 
| internal.0 | StoreF | 3 | STOREW | 6,477 | 
| internal.0 | StoreHeapPtr | 0 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 1 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 2 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 3 | ADD | 1 | 
| internal.0 | StoreHintWord | 0 | HINT_STOREW | 447,312 | 
| internal.0 | StoreHintWord | 1 | HINT_STOREW | 447,312 | 
| internal.0 | StoreHintWord | 2 | HINT_STOREW | 447,312 | 
| internal.0 | StoreHintWord | 3 | HINT_STOREW | 223,665 | 
| internal.0 | StoreV | 0 | ADD | 27,644 | 
| internal.0 | StoreV | 0 | MUL | 18,522 | 
| internal.0 | StoreV | 0 | STOREW | 113,330 | 
| internal.0 | StoreV | 1 | ADD | 27,644 | 
| internal.0 | StoreV | 1 | MUL | 18,522 | 
| internal.0 | StoreV | 1 | STOREW | 113,330 | 
| internal.0 | StoreV | 2 | ADD | 27,644 | 
| internal.0 | StoreV | 2 | MUL | 18,522 | 
| internal.0 | StoreV | 2 | STOREW | 113,330 | 
| internal.0 | StoreV | 3 | ADD | 13,822 | 
| internal.0 | StoreV | 3 | MUL | 9,261 | 
| internal.0 | StoreV | 3 | STOREW | 56,721 | 
| internal.0 | SubE | 0 | FE4SUB | 7,080 | 
| internal.0 | SubE | 1 | FE4SUB | 7,080 | 
| internal.0 | SubE | 2 | FE4SUB | 7,080 | 
| internal.0 | SubE | 3 | FE4SUB | 3,540 | 
| internal.0 | SubEF | 0 | ADD | 33,996 | 
| internal.0 | SubEF | 0 | SUB | 11,332 | 
| internal.0 | SubEF | 1 | ADD | 33,996 | 
| internal.0 | SubEF | 1 | SUB | 11,332 | 
| internal.0 | SubEF | 2 | ADD | 33,996 | 
| internal.0 | SubEF | 2 | SUB | 11,332 | 
| internal.0 | SubEF | 3 | ADD | 16,998 | 
| internal.0 | SubEF | 3 | SUB | 5,666 | 
| internal.0 | SubEFI | 0 | ADD | 768 | 
| internal.0 | SubEFI | 1 | ADD | 768 | 
| internal.0 | SubEFI | 2 | ADD | 768 | 
| internal.0 | SubEFI | 3 | ADD | 384 | 
| internal.0 | SubEI | 0 | ADD | 2,992 | 
| internal.0 | SubEI | 1 | ADD | 2,992 | 
| internal.0 | SubEI | 2 | ADD | 2,992 | 
| internal.0 | SubEI | 3 | ADD | 1,496 | 
| internal.0 | SubF | 0 | SUB | 16 | 
| internal.0 | SubF | 1 | SUB | 16 | 
| internal.0 | SubF | 2 | SUB | 16 | 
| internal.0 | SubF | 3 | SUB | 8 | 
| internal.0 | SubFI | 0 | SUB | 8,680 | 
| internal.0 | SubFI | 1 | SUB | 8,680 | 
| internal.0 | SubFI | 2 | SUB | 8,680 | 
| internal.0 | SubFI | 3 | SUB | 4,340 | 
| internal.0 | SubV | 0 | SUB | 12,550 | 
| internal.0 | SubV | 1 | SUB | 12,550 | 
| internal.0 | SubV | 2 | SUB | 12,550 | 
| internal.0 | SubV | 3 | SUB | 6,275 | 
| internal.0 | SubVI | 0 | SUB | 1,996 | 
| internal.0 | SubVI | 1 | SUB | 1,996 | 
| internal.0 | SubVI | 2 | SUB | 1,996 | 
| internal.0 | SubVI | 3 | SUB | 998 | 
| internal.0 | SubVIN | 0 | SUB | 1,680 | 
| internal.0 | SubVIN | 1 | SUB | 1,680 | 
| internal.0 | SubVIN | 2 | SUB | 1,680 | 
| internal.0 | SubVIN | 3 | SUB | 840 | 
| internal.0 | UnsafeCastVF | 0 | ADD | 228 | 
| internal.0 | UnsafeCastVF | 1 | ADD | 228 | 
| internal.0 | UnsafeCastVF | 2 | ADD | 228 | 
| internal.0 | UnsafeCastVF | 3 | ADD | 114 | 
| internal.0 | VerifyBatchExt | 0 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 1 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 2 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 3 | VERIFY_BATCH | 840 | 
| internal.0 | VerifyBatchFelt | 0 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 1 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 2 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 3 | VERIFY_BATCH | 252 | 
| internal.0 | ZipFor | 0 | ADD | 630,486 | 
| internal.0 | ZipFor | 0 | BNE | 580,748 | 
| internal.0 | ZipFor | 0 | JAL | 59,867 | 
| internal.0 | ZipFor | 1 | ADD | 630,378 | 
| internal.0 | ZipFor | 1 | BNE | 580,640 | 
| internal.0 | ZipFor | 1 | JAL | 59,867 | 
| internal.0 | ZipFor | 2 | ADD | 630,378 | 
| internal.0 | ZipFor | 2 | BNE | 580,640 | 
| internal.0 | ZipFor | 2 | JAL | 59,867 | 
| internal.0 | ZipFor | 3 | ADD | 315,307 | 
| internal.0 | ZipFor | 3 | BNE | 290,438 | 
| internal.0 | ZipFor | 3 | JAL | 29,939 | 
| internal.1 |  | 4 | ADD | 2 | 
| internal.1 |  | 4 | JAL | 1 | 
| internal.1 |  | 5 | ADD | 2 | 
| internal.1 |  | 5 | JAL | 1 | 
| internal.1 | AddE | 4 | FE4ADD | 29,436 | 
| internal.1 | AddE | 5 | FE4ADD | 29,349 | 
| internal.1 | AddEFFI | 4 | ADD | 2,944 | 
| internal.1 | AddEFFI | 5 | ADD | 2,928 | 
| internal.1 | AddEFI | 4 | ADD | 2,200 | 
| internal.1 | AddEFI | 5 | ADD | 2,200 | 
| internal.1 | AddEI | 4 | ADD | 75,680 | 
| internal.1 | AddEI | 5 | ADD | 75,168 | 
| internal.1 | AddF | 4 | ADD | 9,800 | 
| internal.1 | AddF | 5 | ADD | 9,800 | 
| internal.1 | AddFI | 4 | ADD | 24,985 | 
| internal.1 | AddFI | 5 | ADD | 24,473 | 
| internal.1 | AddV | 4 | ADD | 83,938 | 
| internal.1 | AddV | 5 | ADD | 81,727 | 
| internal.1 | AddVI | 4 | ADD | 184,896 | 
| internal.1 | AddVI | 5 | ADD | 180,799 | 
| internal.1 | Alloc | 4 | ADD | 205,950 | 
| internal.1 | Alloc | 4 | MUL | 56,918 | 
| internal.1 | Alloc | 5 | ADD | 200,906 | 
| internal.1 | Alloc | 5 | MUL | 55,573 | 
| internal.1 | AssertEqE | 4 | BNE | 472 | 
| internal.1 | AssertEqE | 5 | BNE | 472 | 
| internal.1 | AssertEqEI | 4 | BNE | 8 | 
| internal.1 | AssertEqEI | 5 | BNE | 8 | 
| internal.1 | AssertEqF | 4 | BNE | 9,001 | 
| internal.1 | AssertEqF | 5 | BNE | 9,001 | 
| internal.1 | AssertEqFI | 4 | BNE | 7 | 
| internal.1 | AssertEqFI | 5 | BNE | 7 | 
| internal.1 | AssertEqV | 4 | BNE | 2,802 | 
| internal.1 | AssertEqV | 5 | BNE | 2,760 | 
| internal.1 | AssertEqVI | 4 | BNE | 474 | 
| internal.1 | AssertEqVI | 5 | BNE | 474 | 
| internal.1 | AssertNonZero | 4 | BEQ | 1 | 
| internal.1 | AssertNonZero | 5 | BEQ | 1 | 
| internal.1 | CT-InitializePcsConst | 4 | PHANTOM | 2 | 
| internal.1 | CT-InitializePcsConst | 5 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 4 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 5 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 4 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 5 | PHANTOM | 2 | 
| internal.1 | CT-cache-generator-powers | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-cache-generator-powers | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-exp-reverse-bits-len | 4 | PHANTOM | 16,296 | 
| internal.1 | CT-exp-reverse-bits-len | 5 | PHANTOM | 16,296 | 
| internal.1 | CT-pre-compute-alpha-pows | 4 | PHANTOM | 4 | 
| internal.1 | CT-pre-compute-alpha-pows | 5 | PHANTOM | 4 | 
| internal.1 | CT-single-reduced-opening-eval | 4 | PHANTOM | 22,512 | 
| internal.1 | CT-single-reduced-opening-eval | 5 | PHANTOM | 22,512 | 
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
| internal.1 | CastFV | 4 | ADD | 296 | 
| internal.1 | CastFV | 5 | ADD | 296 | 
| internal.1 | DivE | 4 | BBE4DIV | 13,088 | 
| internal.1 | DivE | 5 | BBE4DIV | 13,046 | 
| internal.1 | DivEIN | 4 | ADD | 1,496 | 
| internal.1 | DivEIN | 4 | BBE4DIV | 374 | 
| internal.1 | DivEIN | 5 | ADD | 1,496 | 
| internal.1 | DivEIN | 5 | BBE4DIV | 374 | 
| internal.1 | DivF | 4 | DIV | 11,592 | 
| internal.1 | DivF | 5 | DIV | 11,340 | 
| internal.1 | DivFIN | 4 | DIV | 782 | 
| internal.1 | DivFIN | 5 | DIV | 782 | 
| internal.1 | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 11,256 | 
| internal.1 | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 11,256 | 
| internal.1 | HintBitsF | 4 | PHANTOM | 280 | 
| internal.1 | HintBitsF | 5 | PHANTOM | 280 | 
| internal.1 | HintInputVec | 4 | PHANTOM | 46,057 | 
| internal.1 | HintInputVec | 5 | PHANTOM | 44,880 | 
| internal.1 | IfEq | 4 | BNE | 288 | 
| internal.1 | IfEq | 5 | BNE | 284 | 
| internal.1 | IfEqI | 4 | BNE | 42,762 | 
| internal.1 | IfEqI | 4 | JAL | 13,854 | 
| internal.1 | IfEqI | 5 | BNE | 42,166 | 
| internal.1 | IfEqI | 5 | JAL | 13,439 | 
| internal.1 | IfNe | 4 | BEQ | 294 | 
| internal.1 | IfNe | 4 | JAL | 6 | 
| internal.1 | IfNe | 5 | BEQ | 290 | 
| internal.1 | IfNe | 5 | JAL | 6 | 
| internal.1 | IfNeI | 4 | BEQ | 186 | 
| internal.1 | IfNeI | 5 | BEQ | 186 | 
| internal.1 | ImmE | 4 | ADD | 9,808 | 
| internal.1 | ImmE | 5 | ADD | 9,808 | 
| internal.1 | ImmF | 4 | ADD | 11,334 | 
| internal.1 | ImmF | 5 | ADD | 11,334 | 
| internal.1 | ImmV | 4 | ADD | 10,753 | 
| internal.1 | ImmV | 5 | ADD | 10,613 | 
| internal.1 | LoadE | 4 | ADD | 21,588 | 
| internal.1 | LoadE | 4 | LOADW | 52,574 | 
| internal.1 | LoadE | 4 | MUL | 21,588 | 
| internal.1 | LoadE | 5 | ADD | 21,462 | 
| internal.1 | LoadE | 5 | LOADW | 52,322 | 
| internal.1 | LoadE | 5 | MUL | 21,462 | 
| internal.1 | LoadF | 4 | ADD | 26,992 | 
| internal.1 | LoadF | 4 | LOADW | 55,586 | 
| internal.1 | LoadF | 4 | MUL | 17,962 | 
| internal.1 | LoadF | 5 | ADD | 26,988 | 
| internal.1 | LoadF | 5 | LOADW | 55,532 | 
| internal.1 | LoadF | 5 | MUL | 17,962 | 
| internal.1 | LoadHeapPtr | 4 | ADD | 2 | 
| internal.1 | LoadHeapPtr | 5 | ADD | 2 | 
| internal.1 | LoadV | 4 | ADD | 91,386 | 
| internal.1 | LoadV | 4 | LOADW | 271,120 | 
| internal.1 | LoadV | 4 | MUL | 81,588 | 
| internal.1 | LoadV | 5 | ADD | 91,176 | 
| internal.1 | LoadV | 5 | LOADW | 269,101 | 
| internal.1 | LoadV | 5 | MUL | 81,420 | 
| internal.1 | MulE | 4 | BBE4MUL | 55,960 | 
| internal.1 | MulE | 5 | BBE4MUL | 55,487 | 
| internal.1 | MulEF | 4 | MUL | 10,048 | 
| internal.1 | MulEF | 5 | MUL | 9,880 | 
| internal.1 | MulEFI | 4 | MUL | 1,640 | 
| internal.1 | MulEFI | 5 | MUL | 1,640 | 
| internal.1 | MulEI | 4 | ADD | 13,720 | 
| internal.1 | MulEI | 4 | BBE4MUL | 3,430 | 
| internal.1 | MulEI | 5 | ADD | 13,704 | 
| internal.1 | MulEI | 5 | BBE4MUL | 3,426 | 
| internal.1 | MulF | 4 | MUL | 55,644 | 
| internal.1 | MulF | 5 | MUL | 54,636 | 
| internal.1 | MulFI | 4 | MUL | 8,714 | 
| internal.1 | MulFI | 5 | MUL | 8,714 | 
| internal.1 | MulVI | 4 | MUL | 22,045 | 
| internal.1 | MulVI | 5 | MUL | 21,961 | 
| internal.1 | NegE | 4 | MUL | 680 | 
| internal.1 | NegE | 5 | MUL | 680 | 
| internal.1 | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 108 | 
| internal.1 | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 107 | 
| internal.1 | Publish | 4 | PUBLISH | 52 | 
| internal.1 | Publish | 5 | PUBLISH | 52 | 
| internal.1 | StoreE | 4 | ADD | 18,060 | 
| internal.1 | StoreE | 4 | MUL | 18,060 | 
| internal.1 | StoreE | 4 | STOREW | 29,142 | 
| internal.1 | StoreE | 5 | ADD | 18,018 | 
| internal.1 | StoreE | 5 | MUL | 18,018 | 
| internal.1 | StoreE | 5 | STOREW | 28,973 | 
| internal.1 | StoreF | 4 | ADD | 1,532 | 
| internal.1 | StoreF | 4 | MUL | 752 | 
| internal.1 | StoreF | 4 | STOREW | 13,338 | 
| internal.1 | StoreF | 5 | ADD | 1,524 | 
| internal.1 | StoreF | 5 | MUL | 752 | 
| internal.1 | StoreF | 5 | STOREW | 13,078 | 
| internal.1 | StoreHeapPtr | 4 | ADD | 2 | 
| internal.1 | StoreHeapPtr | 5 | ADD | 2 | 
| internal.1 | StoreHintWord | 4 | HINT_STOREW | 467,774 | 
| internal.1 | StoreHintWord | 5 | HINT_STOREW | 457,937 | 
| internal.1 | StoreV | 4 | ADD | 27,560 | 
| internal.1 | StoreV | 4 | MUL | 18,522 | 
| internal.1 | StoreV | 4 | STOREW | 118,038 | 
| internal.1 | StoreV | 5 | ADD | 27,602 | 
| internal.1 | StoreV | 5 | MUL | 18,522 | 
| internal.1 | StoreV | 5 | STOREW | 115,768 | 
| internal.1 | SubE | 4 | FE4SUB | 7,332 | 
| internal.1 | SubE | 5 | FE4SUB | 7,206 | 
| internal.1 | SubEF | 4 | ADD | 33,996 | 
| internal.1 | SubEF | 4 | SUB | 11,332 | 
| internal.1 | SubEF | 5 | ADD | 33,996 | 
| internal.1 | SubEF | 5 | SUB | 11,332 | 
| internal.1 | SubEFI | 4 | ADD | 768 | 
| internal.1 | SubEFI | 5 | ADD | 768 | 
| internal.1 | SubEI | 4 | ADD | 2,992 | 
| internal.1 | SubEI | 5 | ADD | 2,992 | 
| internal.1 | SubF | 4 | SUB | 16 | 
| internal.1 | SubF | 5 | SUB | 16 | 
| internal.1 | SubFI | 4 | SUB | 8,680 | 
| internal.1 | SubFI | 5 | SUB | 8,680 | 
| internal.1 | SubV | 4 | SUB | 12,718 | 
| internal.1 | SubV | 5 | SUB | 12,634 | 
| internal.1 | SubVI | 4 | SUB | 2,088 | 
| internal.1 | SubVI | 5 | SUB | 2,042 | 
| internal.1 | SubVIN | 4 | SUB | 1,764 | 
| internal.1 | SubVIN | 5 | SUB | 1,722 | 
| internal.1 | UnsafeCastVF | 4 | ADD | 228 | 
| internal.1 | UnsafeCastVF | 5 | ADD | 228 | 
| internal.1 | VerifyBatchExt | 4 | VERIFY_BATCH | 1,764 | 
| internal.1 | VerifyBatchExt | 5 | VERIFY_BATCH | 1,722 | 
| internal.1 | VerifyBatchFelt | 4 | VERIFY_BATCH | 504 | 
| internal.1 | VerifyBatchFelt | 5 | VERIFY_BATCH | 504 | 
| internal.1 | ZipFor | 4 | ADD | 657,696 | 
| internal.1 | ZipFor | 4 | BNE | 607,452 | 
| internal.1 | ZipFor | 4 | JAL | 62,473 | 
| internal.1 | ZipFor | 5 | ADD | 644,627 | 
| internal.1 | ZipFor | 5 | BNE | 594,636 | 
| internal.1 | ZipFor | 5 | JAL | 61,212 | 
| internal.2 |  | 6 | ADD | 2 | 
| internal.2 |  | 6 | JAL | 1 | 
| internal.2 | AddE | 6 | FE4ADD | 29,436 | 
| internal.2 | AddEFFI | 6 | ADD | 2,944 | 
| internal.2 | AddEFI | 6 | ADD | 2,200 | 
| internal.2 | AddEI | 6 | ADD | 75,680 | 
| internal.2 | AddF | 6 | ADD | 9,800 | 
| internal.2 | AddFI | 6 | ADD | 24,985 | 
| internal.2 | AddV | 6 | ADD | 83,938 | 
| internal.2 | AddVI | 6 | ADD | 184,896 | 
| internal.2 | Alloc | 6 | ADD | 205,950 | 
| internal.2 | Alloc | 6 | MUL | 56,918 | 
| internal.2 | AssertEqE | 6 | BNE | 472 | 
| internal.2 | AssertEqEI | 6 | BNE | 8 | 
| internal.2 | AssertEqF | 6 | BNE | 9,001 | 
| internal.2 | AssertEqFI | 6 | BNE | 7 | 
| internal.2 | AssertEqV | 6 | BNE | 2,802 | 
| internal.2 | AssertEqVI | 6 | BNE | 474 | 
| internal.2 | AssertNonZero | 6 | BEQ | 1 | 
| internal.2 | CT-InitializePcsConst | 6 | PHANTOM | 2 | 
| internal.2 | CT-ReadProofsFromInput | 6 | PHANTOM | 2 | 
| internal.2 | CT-VerifyProofs | 6 | PHANTOM | 2 | 
| internal.2 | CT-cache-generator-powers | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-compute-reduced-opening | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-exp-reverse-bits-len | 6 | PHANTOM | 16,296 | 
| internal.2 | CT-pre-compute-alpha-pows | 6 | PHANTOM | 4 | 
| internal.2 | CT-single-reduced-opening-eval | 6 | PHANTOM | 22,512 | 
| internal.2 | CT-stage-c-build-rounds | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verifier-verify | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verify-pcs | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-e-verify-constraints | 6 | PHANTOM | 4 | 
| internal.2 | CT-verify-batch | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-verify-batch-ext | 6 | PHANTOM | 3,528 | 
| internal.2 | CT-verify-query | 6 | PHANTOM | 168 | 
| internal.2 | CastFV | 6 | ADD | 296 | 
| internal.2 | DivE | 6 | BBE4DIV | 13,088 | 
| internal.2 | DivEIN | 6 | ADD | 1,496 | 
| internal.2 | DivEIN | 6 | BBE4DIV | 374 | 
| internal.2 | DivF | 6 | DIV | 11,592 | 
| internal.2 | DivFIN | 6 | DIV | 782 | 
| internal.2 | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 11,256 | 
| internal.2 | HintBitsF | 6 | PHANTOM | 280 | 
| internal.2 | HintInputVec | 6 | PHANTOM | 46,057 | 
| internal.2 | IfEq | 6 | BNE | 288 | 
| internal.2 | IfEqI | 6 | BNE | 42,762 | 
| internal.2 | IfEqI | 6 | JAL | 13,762 | 
| internal.2 | IfNe | 6 | BEQ | 294 | 
| internal.2 | IfNe | 6 | JAL | 6 | 
| internal.2 | IfNeI | 6 | BEQ | 186 | 
| internal.2 | ImmE | 6 | ADD | 9,808 | 
| internal.2 | ImmF | 6 | ADD | 11,334 | 
| internal.2 | ImmV | 6 | ADD | 10,753 | 
| internal.2 | LoadE | 6 | ADD | 21,588 | 
| internal.2 | LoadE | 6 | LOADW | 52,574 | 
| internal.2 | LoadE | 6 | MUL | 21,588 | 
| internal.2 | LoadF | 6 | ADD | 26,992 | 
| internal.2 | LoadF | 6 | LOADW | 55,586 | 
| internal.2 | LoadF | 6 | MUL | 17,962 | 
| internal.2 | LoadHeapPtr | 6 | ADD | 2 | 
| internal.2 | LoadV | 6 | ADD | 91,386 | 
| internal.2 | LoadV | 6 | LOADW | 271,120 | 
| internal.2 | LoadV | 6 | MUL | 81,588 | 
| internal.2 | MulE | 6 | BBE4MUL | 55,960 | 
| internal.2 | MulEF | 6 | MUL | 10,048 | 
| internal.2 | MulEFI | 6 | MUL | 1,640 | 
| internal.2 | MulEI | 6 | ADD | 13,720 | 
| internal.2 | MulEI | 6 | BBE4MUL | 3,430 | 
| internal.2 | MulF | 6 | MUL | 55,644 | 
| internal.2 | MulFI | 6 | MUL | 8,714 | 
| internal.2 | MulVI | 6 | MUL | 22,045 | 
| internal.2 | NegE | 6 | MUL | 680 | 
| internal.2 | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 108 | 
| internal.2 | Publish | 6 | PUBLISH | 52 | 
| internal.2 | StoreE | 6 | ADD | 18,060 | 
| internal.2 | StoreE | 6 | MUL | 18,060 | 
| internal.2 | StoreE | 6 | STOREW | 29,142 | 
| internal.2 | StoreF | 6 | ADD | 1,532 | 
| internal.2 | StoreF | 6 | MUL | 752 | 
| internal.2 | StoreF | 6 | STOREW | 13,338 | 
| internal.2 | StoreHeapPtr | 6 | ADD | 2 | 
| internal.2 | StoreHintWord | 6 | HINT_STOREW | 467,774 | 
| internal.2 | StoreV | 6 | ADD | 27,560 | 
| internal.2 | StoreV | 6 | MUL | 18,522 | 
| internal.2 | StoreV | 6 | STOREW | 118,038 | 
| internal.2 | SubE | 6 | FE4SUB | 7,332 | 
| internal.2 | SubEF | 6 | ADD | 33,996 | 
| internal.2 | SubEF | 6 | SUB | 11,332 | 
| internal.2 | SubEFI | 6 | ADD | 768 | 
| internal.2 | SubEI | 6 | ADD | 2,992 | 
| internal.2 | SubF | 6 | SUB | 16 | 
| internal.2 | SubFI | 6 | SUB | 8,680 | 
| internal.2 | SubV | 6 | SUB | 12,718 | 
| internal.2 | SubVI | 6 | SUB | 2,088 | 
| internal.2 | SubVIN | 6 | SUB | 1,764 | 
| internal.2 | UnsafeCastVF | 6 | ADD | 228 | 
| internal.2 | VerifyBatchExt | 6 | VERIFY_BATCH | 1,764 | 
| internal.2 | VerifyBatchFelt | 6 | VERIFY_BATCH | 504 | 
| internal.2 | ZipFor | 6 | ADD | 657,696 | 
| internal.2 | ZipFor | 6 | BNE | 607,452 | 
| internal.2 | ZipFor | 6 | JAL | 62,473 | 
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
| leaf | AddE | 0 | FE4ADD | 12,437 | 
| leaf | AddE | 1 | FE4ADD | 8,885 | 
| leaf | AddE | 2 | FE4ADD | 8,885 | 
| leaf | AddE | 3 | FE4ADD | 8,885 | 
| leaf | AddE | 4 | FE4ADD | 8,885 | 
| leaf | AddE | 5 | FE4ADD | 8,885 | 
| leaf | AddE | 6 | FE4ADD | 10,065 | 
| leaf | AddEFFI | 0 | ADD | 704 | 
| leaf | AddEFFI | 1 | ADD | 656 | 
| leaf | AddEFFI | 2 | ADD | 656 | 
| leaf | AddEFFI | 3 | ADD | 656 | 
| leaf | AddEFFI | 4 | ADD | 656 | 
| leaf | AddEFFI | 5 | ADD | 656 | 
| leaf | AddEFFI | 6 | ADD | 672 | 
| leaf | AddEFI | 0 | ADD | 616 | 
| leaf | AddEFI | 1 | ADD | 476 | 
| leaf | AddEFI | 2 | ADD | 476 | 
| leaf | AddEFI | 3 | ADD | 476 | 
| leaf | AddEFI | 4 | ADD | 476 | 
| leaf | AddEFI | 5 | ADD | 476 | 
| leaf | AddEFI | 6 | ADD | 532 | 
| leaf | AddEI | 0 | ADD | 28,124 | 
| leaf | AddEI | 1 | ADD | 22,408 | 
| leaf | AddEI | 2 | ADD | 22,408 | 
| leaf | AddEI | 3 | ADD | 22,408 | 
| leaf | AddEI | 4 | ADD | 22,408 | 
| leaf | AddEI | 5 | ADD | 22,408 | 
| leaf | AddEI | 6 | ADD | 24,348 | 
| leaf | AddF | 0 | ADD | 4,375 | 
| leaf | AddF | 1 | ADD | 3,535 | 
| leaf | AddF | 2 | ADD | 3,535 | 
| leaf | AddF | 3 | ADD | 3,535 | 
| leaf | AddF | 4 | ADD | 3,535 | 
| leaf | AddF | 5 | ADD | 3,535 | 
| leaf | AddF | 6 | ADD | 3,815 | 
| leaf | AddFI | 0 | ADD | 15,707 | 
| leaf | AddFI | 1 | ADD | 15,641 | 
| leaf | AddFI | 2 | ADD | 15,641 | 
| leaf | AddFI | 3 | ADD | 15,641 | 
| leaf | AddFI | 4 | ADD | 15,641 | 
| leaf | AddFI | 5 | ADD | 15,641 | 
| leaf | AddFI | 6 | ADD | 15,671 | 
| leaf | AddV | 0 | ADD | 40,413 | 
| leaf | AddV | 1 | ADD | 38,153 | 
| leaf | AddV | 2 | ADD | 38,153 | 
| leaf | AddV | 3 | ADD | 38,153 | 
| leaf | AddV | 4 | ADD | 38,153 | 
| leaf | AddV | 5 | ADD | 38,153 | 
| leaf | AddV | 6 | ADD | 39,128 | 
| leaf | AddVI | 0 | ADD | 94,529 | 
| leaf | AddVI | 1 | ADD | 89,303 | 
| leaf | AddVI | 2 | ADD | 89,303 | 
| leaf | AddVI | 3 | ADD | 89,303 | 
| leaf | AddVI | 4 | ADD | 89,303 | 
| leaf | AddVI | 5 | ADD | 89,303 | 
| leaf | AddVI | 6 | ADD | 91,271 | 
| leaf | Alloc | 0 | ADD | 102,500 | 
| leaf | Alloc | 0 | MUL | 28,471 | 
| leaf | Alloc | 1 | ADD | 97,784 | 
| leaf | Alloc | 1 | MUL | 27,235 | 
| leaf | Alloc | 2 | ADD | 97,784 | 
| leaf | Alloc | 2 | MUL | 27,235 | 
| leaf | Alloc | 3 | ADD | 97,784 | 
| leaf | Alloc | 3 | MUL | 27,235 | 
| leaf | Alloc | 4 | ADD | 97,784 | 
| leaf | Alloc | 4 | MUL | 27,235 | 
| leaf | Alloc | 5 | ADD | 97,784 | 
| leaf | Alloc | 5 | MUL | 27,235 | 
| leaf | Alloc | 6 | ADD | 100,264 | 
| leaf | Alloc | 6 | MUL | 27,876 | 
| leaf | AssertEqE | 0 | BNE | 248 | 
| leaf | AssertEqE | 1 | BNE | 224 | 
| leaf | AssertEqE | 2 | BNE | 224 | 
| leaf | AssertEqE | 3 | BNE | 224 | 
| leaf | AssertEqE | 4 | BNE | 224 | 
| leaf | AssertEqE | 5 | BNE | 224 | 
| leaf | AssertEqE | 6 | BNE | 232 | 
| leaf | AssertEqEI | 0 | BNE | 4 | 
| leaf | AssertEqEI | 1 | BNE | 4 | 
| leaf | AssertEqEI | 2 | BNE | 4 | 
| leaf | AssertEqEI | 3 | BNE | 4 | 
| leaf | AssertEqEI | 4 | BNE | 4 | 
| leaf | AssertEqEI | 5 | BNE | 4 | 
| leaf | AssertEqEI | 6 | BNE | 4 | 
| leaf | AssertEqF | 0 | BNE | 4,000 | 
| leaf | AssertEqF | 1 | BNE | 3,232 | 
| leaf | AssertEqF | 2 | BNE | 3,232 | 
| leaf | AssertEqF | 3 | BNE | 3,232 | 
| leaf | AssertEqF | 4 | BNE | 3,232 | 
| leaf | AssertEqF | 5 | BNE | 3,232 | 
| leaf | AssertEqF | 6 | BNE | 3,496 | 
| leaf | AssertEqV | 0 | BNE | 1,451 | 
| leaf | AssertEqV | 1 | BNE | 1,385 | 
| leaf | AssertEqV | 2 | BNE | 1,385 | 
| leaf | AssertEqV | 3 | BNE | 1,385 | 
| leaf | AssertEqV | 4 | BNE | 1,385 | 
| leaf | AssertEqV | 5 | BNE | 1,385 | 
| leaf | AssertEqV | 6 | BNE | 1,407 | 
| leaf | AssertEqVI | 0 | BNE | 241 | 
| leaf | AssertEqVI | 1 | BNE | 175 | 
| leaf | AssertEqVI | 2 | BNE | 175 | 
| leaf | AssertEqVI | 3 | BNE | 175 | 
| leaf | AssertEqVI | 4 | BNE | 175 | 
| leaf | AssertEqVI | 5 | BNE | 175 | 
| leaf | AssertEqVI | 6 | BNE | 198 | 
| leaf | AssertNonZero | 0 | BEQ | 1 | 
| leaf | AssertNonZero | 1 | BEQ | 1 | 
| leaf | AssertNonZero | 2 | BEQ | 1 | 
| leaf | AssertNonZero | 3 | BEQ | 1 | 
| leaf | AssertNonZero | 4 | BEQ | 1 | 
| leaf | AssertNonZero | 5 | BEQ | 1 | 
| leaf | AssertNonZero | 6 | BEQ | 1 | 
| leaf | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 2 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 3 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 4 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 5 | PHANTOM | 2 | 
| leaf | CT-ExtractPublicValuesCommit | 6 | PHANTOM | 2 | 
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
| leaf | CT-cache-generator-powers | 0 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 1 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 2 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 3 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 4 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 5 | PHANTOM | 672 | 
| leaf | CT-cache-generator-powers | 6 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 0 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 1 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 2 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 3 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 4 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 5 | PHANTOM | 672 | 
| leaf | CT-compute-reduced-opening | 6 | PHANTOM | 672 | 
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 6,888 | 
| leaf | CT-exp-reverse-bits-len | 1 | PHANTOM | 4,872 | 
| leaf | CT-exp-reverse-bits-len | 2 | PHANTOM | 4,872 | 
| leaf | CT-exp-reverse-bits-len | 3 | PHANTOM | 4,872 | 
| leaf | CT-exp-reverse-bits-len | 4 | PHANTOM | 4,872 | 
| leaf | CT-exp-reverse-bits-len | 5 | PHANTOM | 4,872 | 
| leaf | CT-exp-reverse-bits-len | 6 | PHANTOM | 5,544 | 
| leaf | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 1 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 2 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 3 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 4 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 5 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 6 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 10,668 | 
| leaf | CT-single-reduced-opening-eval | 1 | PHANTOM | 7,644 | 
| leaf | CT-single-reduced-opening-eval | 2 | PHANTOM | 7,644 | 
| leaf | CT-single-reduced-opening-eval | 3 | PHANTOM | 7,644 | 
| leaf | CT-single-reduced-opening-eval | 4 | PHANTOM | 7,644 | 
| leaf | CT-single-reduced-opening-eval | 5 | PHANTOM | 7,644 | 
| leaf | CT-single-reduced-opening-eval | 6 | PHANTOM | 8,652 | 
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
| leaf | CT-verify-batch | 0 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 1 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 2 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 3 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 4 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 5 | PHANTOM | 672 | 
| leaf | CT-verify-batch | 6 | PHANTOM | 672 | 
| leaf | CT-verify-batch-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 1 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 2 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 3 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 4 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 5 | PHANTOM | 1,680 | 
| leaf | CT-verify-batch-ext | 6 | PHANTOM | 1,680 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CT-verify-query | 1 | PHANTOM | 84 | 
| leaf | CT-verify-query | 2 | PHANTOM | 84 | 
| leaf | CT-verify-query | 3 | PHANTOM | 84 | 
| leaf | CT-verify-query | 4 | PHANTOM | 84 | 
| leaf | CT-verify-query | 5 | PHANTOM | 84 | 
| leaf | CT-verify-query | 6 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 126 | 
| leaf | CastFV | 1 | ADD | 102 | 
| leaf | CastFV | 2 | ADD | 102 | 
| leaf | CastFV | 3 | ADD | 102 | 
| leaf | CastFV | 4 | ADD | 102 | 
| leaf | CastFV | 5 | ADD | 102 | 
| leaf | CastFV | 6 | ADD | 110 | 
| leaf | DivE | 0 | BBE4DIV | 6,214 | 
| leaf | DivE | 1 | BBE4DIV | 4,690 | 
| leaf | DivE | 2 | BBE4DIV | 4,690 | 
| leaf | DivE | 3 | BBE4DIV | 4,690 | 
| leaf | DivE | 4 | BBE4DIV | 4,690 | 
| leaf | DivE | 5 | BBE4DIV | 4,690 | 
| leaf | DivE | 6 | BBE4DIV | 5,198 | 
| leaf | DivEIN | 0 | ADD | 216 | 
| leaf | DivEIN | 0 | BBE4DIV | 54 | 
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
| leaf | DivEIN | 6 | ADD | 168 | 
| leaf | DivEIN | 6 | BBE4DIV | 42 | 
| leaf | DivF | 0 | DIV | 7,392 | 
| leaf | DivF | 1 | DIV | 7,392 | 
| leaf | DivF | 2 | DIV | 7,392 | 
| leaf | DivF | 3 | DIV | 7,392 | 
| leaf | DivF | 4 | DIV | 7,392 | 
| leaf | DivF | 5 | DIV | 7,392 | 
| leaf | DivF | 6 | DIV | 7,392 | 
| leaf | DivFIN | 0 | DIV | 128 | 
| leaf | DivFIN | 1 | DIV | 86 | 
| leaf | DivFIN | 2 | DIV | 86 | 
| leaf | DivFIN | 3 | DIV | 86 | 
| leaf | DivFIN | 4 | DIV | 86 | 
| leaf | DivFIN | 5 | DIV | 86 | 
| leaf | DivFIN | 6 | DIV | 100 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,334 | 
| leaf | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 4,326 | 
| leaf | HintBitsF | 0 | PHANTOM | 125 | 
| leaf | HintBitsF | 1 | PHANTOM | 101 | 
| leaf | HintBitsF | 2 | PHANTOM | 101 | 
| leaf | HintBitsF | 3 | PHANTOM | 101 | 
| leaf | HintBitsF | 4 | PHANTOM | 101 | 
| leaf | HintBitsF | 5 | PHANTOM | 101 | 
| leaf | HintBitsF | 6 | PHANTOM | 109 | 
| leaf | HintInputVec | 0 | PHANTOM | 22,779 | 
| leaf | HintInputVec | 1 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 2 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 3 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 4 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 5 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 6 | PHANTOM | 22,256 | 
| leaf | IfEq | 0 | BNE | 142 | 
| leaf | IfEq | 1 | BNE | 141 | 
| leaf | IfEq | 2 | BNE | 141 | 
| leaf | IfEq | 3 | BNE | 141 | 
| leaf | IfEq | 4 | BNE | 141 | 
| leaf | IfEq | 5 | BNE | 141 | 
| leaf | IfEq | 6 | BNE | 141 | 
| leaf | IfEqI | 0 | BNE | 23,483 | 
| leaf | IfEqI | 0 | JAL | 8,248 | 
| leaf | IfEqI | 1 | BNE | 21,659 | 
| leaf | IfEqI | 1 | JAL | 8,454 | 
| leaf | IfEqI | 2 | BNE | 21,659 | 
| leaf | IfEqI | 2 | JAL | 9,127 | 
| leaf | IfEqI | 3 | BNE | 21,659 | 
| leaf | IfEqI | 3 | JAL | 9,115 | 
| leaf | IfEqI | 4 | BNE | 21,659 | 
| leaf | IfEqI | 4 | JAL | 9,245 | 
| leaf | IfEqI | 5 | BNE | 21,659 | 
| leaf | IfEqI | 5 | JAL | 8,645 | 
| leaf | IfEqI | 6 | BNE | 22,267 | 
| leaf | IfEqI | 6 | JAL | 9,032 | 
| leaf | IfNe | 0 | BEQ | 143 | 
| leaf | IfNe | 0 | JAL | 1 | 
| leaf | IfNe | 1 | BEQ | 143 | 
| leaf | IfNe | 1 | JAL | 2 | 
| leaf | IfNe | 2 | BEQ | 143 | 
| leaf | IfNe | 2 | JAL | 2 | 
| leaf | IfNe | 3 | BEQ | 143 | 
| leaf | IfNe | 3 | JAL | 2 | 
| leaf | IfNe | 4 | BEQ | 143 | 
| leaf | IfNe | 4 | JAL | 2 | 
| leaf | IfNe | 5 | BEQ | 143 | 
| leaf | IfNe | 5 | JAL | 2 | 
| leaf | IfNe | 6 | BEQ | 143 | 
| leaf | IfNe | 6 | JAL | 2 | 
| leaf | IfNeI | 0 | BEQ | 100 | 
| leaf | IfNeI | 1 | BEQ | 70 | 
| leaf | IfNeI | 2 | BEQ | 70 | 
| leaf | IfNeI | 3 | BEQ | 70 | 
| leaf | IfNeI | 4 | BEQ | 70 | 
| leaf | IfNeI | 5 | BEQ | 70 | 
| leaf | IfNeI | 6 | BEQ | 80 | 
| leaf | ImmE | 0 | ADD | 3,536 | 
| leaf | ImmE | 1 | ADD | 2,176 | 
| leaf | ImmE | 2 | ADD | 2,176 | 
| leaf | ImmE | 3 | ADD | 2,176 | 
| leaf | ImmE | 4 | ADD | 2,176 | 
| leaf | ImmE | 5 | ADD | 2,176 | 
| leaf | ImmE | 6 | ADD | 2,604 | 
| leaf | ImmF | 0 | ADD | 5,431 | 
| leaf | ImmF | 1 | ADD | 4,603 | 
| leaf | ImmF | 2 | ADD | 4,603 | 
| leaf | ImmF | 3 | ADD | 4,603 | 
| leaf | ImmF | 4 | ADD | 4,603 | 
| leaf | ImmF | 5 | ADD | 4,603 | 
| leaf | ImmF | 6 | ADD | 4,879 | 
| leaf | ImmV | 0 | ADD | 5,344 | 
| leaf | ImmV | 1 | ADD | 5,146 | 
| leaf | ImmV | 2 | ADD | 5,146 | 
| leaf | ImmV | 3 | ADD | 5,146 | 
| leaf | ImmV | 4 | ADD | 5,146 | 
| leaf | ImmV | 5 | ADD | 5,146 | 
| leaf | ImmV | 6 | ADD | 5,215 | 
| leaf | LoadE | 0 | ADD | 9,408 | 
| leaf | LoadE | 0 | LOADW | 23,747 | 
| leaf | LoadE | 0 | MUL | 9,408 | 
| leaf | LoadE | 1 | ADD | 7,392 | 
| leaf | LoadE | 1 | LOADW | 18,142 | 
| leaf | LoadE | 1 | MUL | 7,392 | 
| leaf | LoadE | 2 | ADD | 7,392 | 
| leaf | LoadE | 2 | LOADW | 18,142 | 
| leaf | LoadE | 2 | MUL | 7,392 | 
| leaf | LoadE | 3 | ADD | 7,392 | 
| leaf | LoadE | 3 | LOADW | 18,142 | 
| leaf | LoadE | 3 | MUL | 7,392 | 
| leaf | LoadE | 4 | ADD | 7,392 | 
| leaf | LoadE | 4 | LOADW | 18,142 | 
| leaf | LoadE | 4 | MUL | 7,392 | 
| leaf | LoadE | 5 | ADD | 7,392 | 
| leaf | LoadE | 5 | LOADW | 18,142 | 
| leaf | LoadE | 5 | MUL | 7,392 | 
| leaf | LoadE | 6 | ADD | 8,064 | 
| leaf | LoadE | 6 | LOADW | 20,041 | 
| leaf | LoadE | 6 | MUL | 8,064 | 
| leaf | LoadF | 0 | ADD | 11,848 | 
| leaf | LoadF | 0 | LOADW | 24,340 | 
| leaf | LoadF | 0 | MUL | 7,883 | 
| leaf | LoadF | 1 | ADD | 8,728 | 
| leaf | LoadF | 1 | LOADW | 18,388 | 
| leaf | LoadF | 1 | MUL | 5,771 | 
| leaf | LoadF | 2 | ADD | 8,728 | 
| leaf | LoadF | 2 | LOADW | 18,388 | 
| leaf | LoadF | 2 | MUL | 5,771 | 
| leaf | LoadF | 3 | ADD | 8,728 | 
| leaf | LoadF | 3 | LOADW | 18,388 | 
| leaf | LoadF | 3 | MUL | 5,771 | 
| leaf | LoadF | 4 | ADD | 8,728 | 
| leaf | LoadF | 4 | LOADW | 18,388 | 
| leaf | LoadF | 4 | MUL | 5,771 | 
| leaf | LoadF | 5 | ADD | 8,728 | 
| leaf | LoadF | 5 | LOADW | 18,388 | 
| leaf | LoadF | 5 | MUL | 5,771 | 
| leaf | LoadF | 6 | ADD | 9,768 | 
| leaf | LoadF | 6 | LOADW | 21,028 | 
| leaf | LoadF | 6 | MUL | 6,475 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadHeapPtr | 1 | ADD | 1 | 
| leaf | LoadHeapPtr | 2 | ADD | 1 | 
| leaf | LoadHeapPtr | 3 | ADD | 1 | 
| leaf | LoadHeapPtr | 4 | ADD | 1 | 
| leaf | LoadHeapPtr | 5 | ADD | 1 | 
| leaf | LoadHeapPtr | 6 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 40,699 | 
| leaf | LoadV | 0 | LOADW | 128,410 | 
| leaf | LoadV | 0 | MUL | 36,526 | 
| leaf | LoadV | 1 | ADD | 30,559 | 
| leaf | LoadV | 1 | LOADW | 104,866 | 
| leaf | LoadV | 1 | MUL | 27,406 | 
| leaf | LoadV | 2 | ADD | 30,559 | 
| leaf | LoadV | 2 | LOADW | 104,866 | 
| leaf | LoadV | 2 | MUL | 27,406 | 
| leaf | LoadV | 3 | ADD | 30,559 | 
| leaf | LoadV | 3 | LOADW | 104,866 | 
| leaf | LoadV | 3 | MUL | 27,406 | 
| leaf | LoadV | 4 | ADD | 30,559 | 
| leaf | LoadV | 4 | LOADW | 104,866 | 
| leaf | LoadV | 4 | MUL | 27,406 | 
| leaf | LoadV | 5 | ADD | 30,559 | 
| leaf | LoadV | 5 | LOADW | 104,866 | 
| leaf | LoadV | 5 | MUL | 27,406 | 
| leaf | LoadV | 6 | ADD | 33,939 | 
| leaf | LoadV | 6 | LOADW | 112,940 | 
| leaf | LoadV | 6 | MUL | 30,446 | 
| leaf | MulE | 0 | BBE4MUL | 19,354 | 
| leaf | MulE | 1 | BBE4MUL | 14,465 | 
| leaf | MulE | 2 | BBE4MUL | 14,465 | 
| leaf | MulE | 3 | BBE4MUL | 14,465 | 
| leaf | MulE | 4 | BBE4MUL | 14,465 | 
| leaf | MulE | 5 | BBE4MUL | 14,465 | 
| leaf | MulE | 6 | BBE4MUL | 16,029 | 
| leaf | MulEF | 0 | MUL | 3,792 | 
| leaf | MulEF | 1 | MUL | 3,648 | 
| leaf | MulEF | 2 | MUL | 3,648 | 
| leaf | MulEF | 3 | MUL | 3,648 | 
| leaf | MulEF | 4 | MUL | 3,648 | 
| leaf | MulEF | 5 | MUL | 3,648 | 
| leaf | MulEF | 6 | MUL | 3,696 | 
| leaf | MulEFI | 0 | MUL | 500 | 
| leaf | MulEFI | 1 | MUL | 184 | 
| leaf | MulEFI | 2 | MUL | 184 | 
| leaf | MulEFI | 3 | MUL | 184 | 
| leaf | MulEFI | 4 | MUL | 184 | 
| leaf | MulEFI | 5 | MUL | 184 | 
| leaf | MulEFI | 6 | MUL | 312 | 
| leaf | MulEI | 0 | ADD | 6,104 | 
| leaf | MulEI | 0 | BBE4MUL | 1,526 | 
| leaf | MulEI | 1 | ADD | 3,600 | 
| leaf | MulEI | 1 | BBE4MUL | 900 | 
| leaf | MulEI | 2 | ADD | 3,600 | 
| leaf | MulEI | 2 | BBE4MUL | 900 | 
| leaf | MulEI | 3 | ADD | 3,600 | 
| leaf | MulEI | 3 | BBE4MUL | 900 | 
| leaf | MulEI | 4 | ADD | 3,600 | 
| leaf | MulEI | 4 | BBE4MUL | 900 | 
| leaf | MulEI | 5 | ADD | 3,600 | 
| leaf | MulEI | 5 | BBE4MUL | 900 | 
| leaf | MulEI | 6 | ADD | 4,552 | 
| leaf | MulEI | 6 | BBE4MUL | 1,138 | 
| leaf | MulF | 0 | MUL | 33,625 | 
| leaf | MulF | 1 | MUL | 32,821 | 
| leaf | MulF | 2 | MUL | 32,821 | 
| leaf | MulF | 3 | MUL | 32,821 | 
| leaf | MulF | 4 | MUL | 32,821 | 
| leaf | MulF | 5 | MUL | 32,821 | 
| leaf | MulF | 6 | MUL | 33,089 | 
| leaf | MulFI | 0 | MUL | 3,895 | 
| leaf | MulFI | 1 | MUL | 3,145 | 
| leaf | MulFI | 2 | MUL | 3,145 | 
| leaf | MulFI | 3 | MUL | 3,145 | 
| leaf | MulFI | 4 | MUL | 3,145 | 
| leaf | MulFI | 5 | MUL | 3,145 | 
| leaf | MulFI | 6 | MUL | 3,395 | 
| leaf | MulVI | 0 | MUL | 9,938 | 
| leaf | MulVI | 1 | MUL | 7,862 | 
| leaf | MulVI | 2 | MUL | 7,862 | 
| leaf | MulVI | 3 | MUL | 7,862 | 
| leaf | MulVI | 4 | MUL | 7,862 | 
| leaf | MulVI | 5 | MUL | 7,862 | 
| leaf | MulVI | 6 | MUL | 8,554 | 
| leaf | NegE | 0 | MUL | 172 | 
| leaf | NegE | 1 | MUL | 96 | 
| leaf | NegE | 2 | MUL | 96 | 
| leaf | NegE | 3 | MUL | 96 | 
| leaf | NegE | 4 | MUL | 96 | 
| leaf | NegE | 5 | MUL | 96 | 
| leaf | NegE | 6 | MUL | 124 | 
| leaf | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 51 | 
| leaf | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 49 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | Publish | 1 | PUBLISH | 36 | 
| leaf | Publish | 2 | PUBLISH | 36 | 
| leaf | Publish | 3 | PUBLISH | 36 | 
| leaf | Publish | 4 | PUBLISH | 36 | 
| leaf | Publish | 5 | PUBLISH | 36 | 
| leaf | Publish | 6 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 7,728 | 
| leaf | StoreE | 0 | MUL | 7,728 | 
| leaf | StoreE | 0 | STOREW | 13,136 | 
| leaf | StoreE | 1 | ADD | 5,712 | 
| leaf | StoreE | 1 | MUL | 5,712 | 
| leaf | StoreE | 1 | STOREW | 11,078 | 
| leaf | StoreE | 2 | ADD | 5,712 | 
| leaf | StoreE | 2 | MUL | 5,712 | 
| leaf | StoreE | 2 | STOREW | 11,078 | 
| leaf | StoreE | 3 | ADD | 5,712 | 
| leaf | StoreE | 3 | MUL | 5,712 | 
| leaf | StoreE | 3 | STOREW | 11,078 | 
| leaf | StoreE | 4 | ADD | 5,712 | 
| leaf | StoreE | 4 | MUL | 5,712 | 
| leaf | StoreE | 4 | STOREW | 11,078 | 
| leaf | StoreE | 5 | ADD | 5,712 | 
| leaf | StoreE | 5 | MUL | 5,712 | 
| leaf | StoreE | 5 | STOREW | 11,078 | 
| leaf | StoreE | 6 | ADD | 6,384 | 
| leaf | StoreE | 6 | MUL | 6,384 | 
| leaf | StoreE | 6 | STOREW | 11,764 | 
| leaf | StoreF | 0 | ADD | 685 | 
| leaf | StoreF | 0 | MUL | 308 | 
| leaf | StoreF | 0 | STOREW | 8,204 | 
| leaf | StoreF | 1 | ADD | 559 | 
| leaf | StoreF | 1 | MUL | 212 | 
| leaf | StoreF | 1 | STOREW | 8,078 | 
| leaf | StoreF | 2 | ADD | 559 | 
| leaf | StoreF | 2 | MUL | 212 | 
| leaf | StoreF | 2 | STOREW | 8,078 | 
| leaf | StoreF | 3 | ADD | 559 | 
| leaf | StoreF | 3 | MUL | 212 | 
| leaf | StoreF | 3 | STOREW | 8,078 | 
| leaf | StoreF | 4 | ADD | 559 | 
| leaf | StoreF | 4 | MUL | 212 | 
| leaf | StoreF | 4 | STOREW | 8,078 | 
| leaf | StoreF | 5 | ADD | 559 | 
| leaf | StoreF | 5 | MUL | 212 | 
| leaf | StoreF | 5 | STOREW | 8,078 | 
| leaf | StoreF | 6 | ADD | 817 | 
| leaf | StoreF | 6 | MUL | 460 | 
| leaf | StoreF | 6 | STOREW | 8,768 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHeapPtr | 1 | ADD | 1 | 
| leaf | StoreHeapPtr | 2 | ADD | 1 | 
| leaf | StoreHeapPtr | 3 | ADD | 1 | 
| leaf | StoreHeapPtr | 4 | ADD | 1 | 
| leaf | StoreHeapPtr | 5 | ADD | 1 | 
| leaf | StoreHeapPtr | 6 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 231,320 | 
| leaf | StoreHintWord | 1 | HINT_STOREW | 203,150 | 
| leaf | StoreHintWord | 2 | HINT_STOREW | 203,150 | 
| leaf | StoreHintWord | 3 | HINT_STOREW | 203,150 | 
| leaf | StoreHintWord | 4 | HINT_STOREW | 203,150 | 
| leaf | StoreHintWord | 5 | HINT_STOREW | 203,150 | 
| leaf | StoreHintWord | 6 | HINT_STOREW | 213,790 | 
| leaf | StoreV | 0 | ADD | 11,848 | 
| leaf | StoreV | 0 | MUL | 7,934 | 
| leaf | StoreV | 0 | STOREW | 56,656 | 
| leaf | StoreV | 1 | ADD | 8,494 | 
| leaf | StoreV | 1 | MUL | 5,612 | 
| leaf | StoreV | 1 | STOREW | 51,064 | 
| leaf | StoreV | 2 | ADD | 8,494 | 
| leaf | StoreV | 2 | MUL | 5,612 | 
| leaf | StoreV | 2 | STOREW | 51,064 | 
| leaf | StoreV | 3 | ADD | 8,494 | 
| leaf | StoreV | 3 | MUL | 5,612 | 
| leaf | StoreV | 3 | STOREW | 51,064 | 
| leaf | StoreV | 4 | ADD | 8,494 | 
| leaf | StoreV | 4 | MUL | 5,612 | 
| leaf | StoreV | 4 | STOREW | 51,064 | 
| leaf | StoreV | 5 | ADD | 8,494 | 
| leaf | StoreV | 5 | MUL | 5,612 | 
| leaf | StoreV | 5 | STOREW | 51,064 | 
| leaf | StoreV | 6 | ADD | 9,612 | 
| leaf | StoreV | 6 | MUL | 6,386 | 
| leaf | StoreV | 6 | STOREW | 52,928 | 
| leaf | SubE | 0 | FE4SUB | 3,306 | 
| leaf | SubE | 1 | FE4SUB | 3,046 | 
| leaf | SubE | 2 | FE4SUB | 3,046 | 
| leaf | SubE | 3 | FE4SUB | 3,046 | 
| leaf | SubE | 4 | FE4SUB | 3,046 | 
| leaf | SubE | 5 | FE4SUB | 3,046 | 
| leaf | SubE | 6 | FE4SUB | 3,119 | 
| leaf | SubEF | 0 | ADD | 16,182 | 
| leaf | SubEF | 0 | SUB | 5,394 | 
| leaf | SubEF | 1 | ADD | 11,610 | 
| leaf | SubEF | 1 | SUB | 3,870 | 
| leaf | SubEF | 2 | ADD | 11,610 | 
| leaf | SubEF | 2 | SUB | 3,870 | 
| leaf | SubEF | 3 | ADD | 11,610 | 
| leaf | SubEF | 3 | SUB | 3,870 | 
| leaf | SubEF | 4 | ADD | 11,610 | 
| leaf | SubEF | 4 | SUB | 3,870 | 
| leaf | SubEF | 5 | ADD | 11,610 | 
| leaf | SubEF | 5 | SUB | 3,870 | 
| leaf | SubEF | 6 | ADD | 13,134 | 
| leaf | SubEF | 6 | SUB | 4,378 | 
| leaf | SubEFI | 0 | ADD | 344 | 
| leaf | SubEFI | 1 | ADD | 148 | 
| leaf | SubEFI | 2 | ADD | 148 | 
| leaf | SubEFI | 3 | ADD | 148 | 
| leaf | SubEFI | 4 | ADD | 148 | 
| leaf | SubEFI | 5 | ADD | 148 | 
| leaf | SubEFI | 6 | ADD | 208 | 
| leaf | SubEI | 0 | ADD | 432 | 
| leaf | SubEI | 1 | ADD | 288 | 
| leaf | SubEI | 2 | ADD | 288 | 
| leaf | SubEI | 3 | ADD | 288 | 
| leaf | SubEI | 4 | ADD | 288 | 
| leaf | SubEI | 5 | ADD | 288 | 
| leaf | SubEI | 6 | ADD | 336 | 
| leaf | SubFI | 0 | SUB | 3,875 | 
| leaf | SubFI | 1 | SUB | 3,131 | 
| leaf | SubFI | 2 | SUB | 3,131 | 
| leaf | SubFI | 3 | SUB | 3,131 | 
| leaf | SubFI | 4 | SUB | 3,131 | 
| leaf | SubFI | 5 | SUB | 3,131 | 
| leaf | SubFI | 6 | SUB | 3,379 | 
| leaf | SubV | 0 | SUB | 5,816 | 
| leaf | SubV | 1 | SUB | 4,802 | 
| leaf | SubV | 2 | SUB | 4,802 | 
| leaf | SubV | 3 | SUB | 4,802 | 
| leaf | SubV | 4 | SUB | 4,802 | 
| leaf | SubV | 5 | SUB | 4,802 | 
| leaf | SubV | 6 | SUB | 5,140 | 
| leaf | SubVI | 0 | SUB | 1,000 | 
| leaf | SubVI | 1 | SUB | 994 | 
| leaf | SubVI | 2 | SUB | 994 | 
| leaf | SubVI | 3 | SUB | 994 | 
| leaf | SubVI | 4 | SUB | 994 | 
| leaf | SubVI | 5 | SUB | 994 | 
| leaf | SubVI | 6 | SUB | 996 | 
| leaf | SubVIN | 0 | SUB | 840 | 
| leaf | SubVIN | 1 | SUB | 840 | 
| leaf | SubVIN | 2 | SUB | 840 | 
| leaf | SubVIN | 3 | SUB | 840 | 
| leaf | SubVIN | 4 | SUB | 840 | 
| leaf | SubVIN | 5 | SUB | 840 | 
| leaf | SubVIN | 6 | SUB | 840 | 
| leaf | UnsafeCastVF | 0 | ADD | 102 | 
| leaf | UnsafeCastVF | 1 | ADD | 72 | 
| leaf | UnsafeCastVF | 2 | ADD | 72 | 
| leaf | UnsafeCastVF | 3 | ADD | 72 | 
| leaf | UnsafeCastVF | 4 | ADD | 72 | 
| leaf | UnsafeCastVF | 5 | ADD | 72 | 
| leaf | UnsafeCastVF | 6 | ADD | 82 | 
| leaf | VerifyBatchExt | 0 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 1 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 2 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 3 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 4 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 5 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchExt | 6 | VERIFY_BATCH | 840 | 
| leaf | VerifyBatchFelt | 0 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 1 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 2 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 3 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 4 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 5 | VERIFY_BATCH | 336 | 
| leaf | VerifyBatchFelt | 6 | VERIFY_BATCH | 336 | 
| leaf | ZipFor | 0 | ADD | 321,567 | 
| leaf | ZipFor | 0 | BNE | 296,625 | 
| leaf | ZipFor | 0 | JAL | 30,547 | 
| leaf | ZipFor | 1 | ADD | 283,549 | 
| leaf | ZipFor | 1 | BNE | 263,167 | 
| leaf | ZipFor | 1 | JAL | 28,315 | 
| leaf | ZipFor | 2 | ADD | 283,549 | 
| leaf | ZipFor | 2 | BNE | 263,167 | 
| leaf | ZipFor | 2 | JAL | 28,315 | 
| leaf | ZipFor | 3 | ADD | 283,549 | 
| leaf | ZipFor | 3 | BNE | 263,167 | 
| leaf | ZipFor | 3 | JAL | 28,315 | 
| leaf | ZipFor | 4 | ADD | 283,549 | 
| leaf | ZipFor | 4 | BNE | 263,167 | 
| leaf | ZipFor | 4 | JAL | 28,315 | 
| leaf | ZipFor | 5 | ADD | 283,549 | 
| leaf | ZipFor | 5 | BNE | 263,167 | 
| leaf | ZipFor | 5 | JAL | 28,315 | 
| leaf | ZipFor | 6 | ADD | 297,488 | 
| leaf | ZipFor | 6 | BNE | 275,586 | 
| leaf | ZipFor | 6 | JAL | 29,285 | 
| root |  | 0 | ADD | 2 | 
| root |  | 0 | JAL | 1 | 
| root | AddE | 0 | FE4ADD | 14,718 | 
| root | AddEFFI | 0 | ADD | 1,472 | 
| root | AddEFI | 0 | ADD | 1,100 | 
| root | AddEI | 0 | ADD | 37,840 | 
| root | AddF | 0 | ADD | 4,900 | 
| root | AddFI | 0 | ADD | 12,505 | 
| root | AddV | 0 | ADD | 41,967 | 
| root | AddVI | 0 | ADD | 92,468 | 
| root | Alloc | 0 | ADD | 102,994 | 
| root | Alloc | 0 | MUL | 28,471 | 
| root | AssertEqE | 0 | BNE | 236 | 
| root | AssertEqEI | 0 | BNE | 4 | 
| root | AssertEqF | 0 | BNE | 4,496 | 
| root | AssertEqFI | 0 | BNE | 5 | 
| root | AssertEqV | 0 | BNE | 1,401 | 
| root | AssertEqVI | 0 | BNE | 238 | 
| root | AssertNonZero | 0 | BEQ | 1 | 
| root | CT-ExtractPublicValues | 0 | PHANTOM | 2 | 
| root | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| root | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| root | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| root | CT-cache-generator-powers | 0 | PHANTOM | 504 | 
| root | CT-compute-reduced-opening | 0 | PHANTOM | 504 | 
| root | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,148 | 
| root | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| root | CT-single-reduced-opening-eval | 0 | PHANTOM | 11,256 | 
| root | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| root | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| root | CT-verify-batch | 0 | PHANTOM | 504 | 
| root | CT-verify-batch-ext | 0 | PHANTOM | 1,764 | 
| root | CT-verify-query | 0 | PHANTOM | 84 | 
| root | CastFV | 0 | ADD | 148 | 
| root | DivE | 0 | BBE4DIV | 6,544 | 
| root | DivEIN | 0 | ADD | 748 | 
| root | DivEIN | 0 | BBE4DIV | 187 | 
| root | DivF | 0 | DIV | 5,796 | 
| root | DivFIN | 0 | DIV | 391 | 
| root | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,628 | 
| root | HintBitsF | 0 | PHANTOM | 140 | 
| root | HintInputVec | 0 | PHANTOM | 23,026 | 
| root | IfEq | 0 | BNE | 144 | 
| root | IfEqI | 0 | BNE | 21,381 | 
| root | IfEqI | 0 | JAL | 6,825 | 
| root | IfNe | 0 | BEQ | 147 | 
| root | IfNe | 0 | JAL | 3 | 
| root | IfNeI | 0 | BEQ | 93 | 
| root | ImmE | 0 | ADD | 4,904 | 
| root | ImmF | 0 | ADD | 5,760 | 
| root | ImmV | 0 | ADD | 5,447 | 
| root | LoadE | 0 | ADD | 10,794 | 
| root | LoadE | 0 | LOADW | 26,287 | 
| root | LoadE | 0 | MUL | 10,794 | 
| root | LoadF | 0 | ADD | 13,496 | 
| root | LoadF | 0 | LOADW | 27,925 | 
| root | LoadF | 0 | MUL | 8,981 | 
| root | LoadHeapPtr | 0 | ADD | 1 | 
| root | LoadV | 0 | ADD | 45,693 | 
| root | LoadV | 0 | LOADW | 135,558 | 
| root | LoadV | 0 | MUL | 40,794 | 
| root | MulE | 0 | BBE4MUL | 27,980 | 
| root | MulEF | 0 | MUL | 5,024 | 
| root | MulEFI | 0 | MUL | 820 | 
| root | MulEI | 0 | ADD | 6,860 | 
| root | MulEI | 0 | BBE4MUL | 1,715 | 
| root | MulF | 0 | MUL | 27,822 | 
| root | MulFI | 0 | MUL | 4,357 | 
| root | MulVI | 0 | MUL | 11,023 | 
| root | NegE | 0 | MUL | 340 | 
| root | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 12 | 
| root | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 54 | 
| root | Publish | 0 | PUBLISH | 48 | 
| root | StoreE | 0 | ADD | 9,030 | 
| root | StoreE | 0 | MUL | 9,030 | 
| root | StoreE | 0 | STOREW | 14,571 | 
| root | StoreF | 0 | ADD | 766 | 
| root | StoreF | 0 | MUL | 376 | 
| root | StoreF | 0 | STOREW | 6,929 | 
| root | StoreHeapPtr | 0 | ADD | 1 | 
| root | StoreHintWord | 0 | HINT_STOREW | 233,913 | 
| root | StoreV | 0 | ADD | 13,780 | 
| root | StoreV | 0 | MUL | 9,261 | 
| root | StoreV | 0 | STOREW | 59,075 | 
| root | SubE | 0 | FE4SUB | 3,666 | 
| root | SubEF | 0 | ADD | 16,998 | 
| root | SubEF | 0 | SUB | 5,666 | 
| root | SubEFI | 0 | ADD | 384 | 
| root | SubEI | 0 | ADD | 1,496 | 
| root | SubF | 0 | SUB | 8 | 
| root | SubFI | 0 | SUB | 4,340 | 
| root | SubV | 0 | SUB | 6,359 | 
| root | SubVI | 0 | SUB | 1,044 | 
| root | SubVIN | 0 | SUB | 882 | 
| root | UnsafeCastVF | 0 | ADD | 114 | 
| root | VerifyBatchExt | 0 | VERIFY_BATCH | 882 | 
| root | VerifyBatchFelt | 0 | VERIFY_BATCH | 252 | 
| root | ZipFor | 0 | ADD | 328,875 | 
| root | ZipFor | 0 | BNE | 303,753 | 
| root | ZipFor | 0 | JAL | 31,235 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fib_e2e |  | ADD | 0 | 1,048,528 | 
| fib_e2e |  | ADD | 1 | 1,048,500 | 
| fib_e2e |  | ADD | 2 | 1,048,501 | 
| fib_e2e |  | ADD | 3 | 1,048,501 | 
| fib_e2e |  | ADD | 4 | 1,048,502 | 
| fib_e2e |  | ADD | 5 | 1,048,501 | 
| fib_e2e |  | ADD | 6 | 909,012 | 
| fib_e2e |  | AND | 0 | 2 | 
| fib_e2e |  | AUIPC | 0 | 9 | 
| fib_e2e |  | BEQ | 0 | 116,502 | 
| fib_e2e |  | BEQ | 1 | 116,500 | 
| fib_e2e |  | BEQ | 2 | 116,500 | 
| fib_e2e |  | BEQ | 3 | 116,500 | 
| fib_e2e |  | BEQ | 4 | 116,500 | 
| fib_e2e |  | BEQ | 5 | 116,501 | 
| fib_e2e |  | BEQ | 6 | 101,001 | 
| fib_e2e |  | BGEU | 0 | 3 | 
| fib_e2e |  | BLTU | 0 | 2 | 
| fib_e2e |  | BNE | 0 | 116,503 | 
| fib_e2e |  | BNE | 1 | 116,501 | 
| fib_e2e |  | BNE | 2 | 116,500 | 
| fib_e2e |  | BNE | 3 | 116,500 | 
| fib_e2e |  | BNE | 4 | 116,500 | 
| fib_e2e |  | BNE | 5 | 116,500 | 
| fib_e2e |  | BNE | 6 | 101,001 | 
| fib_e2e |  | HINT_STOREW | 0 | 3 | 
| fib_e2e |  | JAL | 0 | 116,499 | 
| fib_e2e |  | JAL | 1 | 116,500 | 
| fib_e2e |  | JAL | 2 | 116,501 | 
| fib_e2e |  | JAL | 3 | 116,500 | 
| fib_e2e |  | JAL | 4 | 116,500 | 
| fib_e2e |  | JAL | 5 | 116,500 | 
| fib_e2e |  | JAL | 6 | 101,001 | 
| fib_e2e |  | JALR | 0 | 12 | 
| fib_e2e |  | JALR | 6 | 1 | 
| fib_e2e |  | LOADW | 0 | 10 | 
| fib_e2e |  | LOADW | 6 | 3 | 
| fib_e2e |  | LUI | 0 | 9 | 
| fib_e2e |  | OR | 0 | 1 | 
| fib_e2e |  | PHANTOM | 0 | 2 | 
| fib_e2e |  | SLL | 0 | 2 | 
| fib_e2e |  | SLTU | 0 | 349,497 | 
| fib_e2e |  | SLTU | 1 | 349,501 | 
| fib_e2e |  | SLTU | 2 | 349,500 | 
| fib_e2e |  | SLTU | 3 | 349,501 | 
| fib_e2e |  | SLTU | 4 | 349,500 | 
| fib_e2e |  | SLTU | 5 | 349,500 | 
| fib_e2e |  | SLTU | 6 | 303,003 | 
| fib_e2e |  | STOREW | 0 | 13 | 
| fib_e2e |  | STOREW | 6 | 2 | 
| fib_e2e |  | SUB | 0 | 4 | 
| fib_e2e |  | XOR | 0 | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 2,574 | 25,779 | 3,718,878 | 374,819,800 | 10,721 | 1,416 | 1,964 | 2,380 | 2,575 | 2,106 | 141,434,608 | 274 | 12,484 | 
| internal.0 | 1 | 2,869 | 26,344 | 3,718,443 | 374,819,800 | 10,832 | 1,427 | 1,944 | 2,439 | 2,534 | 2,201 | 141,440,292 | 281 | 12,643 | 
| internal.0 | 2 | 2,899 | 26,346 | 3,718,430 | 374,819,800 | 10,787 | 1,416 | 1,947 | 2,425 | 2,533 | 2,185 | 141,440,162 | 275 | 12,660 | 
| internal.0 | 3 | 1,570 | 13,906 | 1,860,167 | 190,950,104 | 5,937 | 747 | 1,044 | 1,294 | 1,324 | 1,371 | 71,938,447 | 153 | 6,399 | 
| internal.1 | 4 | 2,636 | 26,607 | 3,846,585 | 374,819,800 | 10,743 | 1,445 | 1,956 | 2,411 | 2,491 | 2,156 | 146,058,076 | 278 | 13,228 | 
| internal.1 | 5 | 3,014 | 26,898 | 3,785,220 | 374,819,800 | 10,786 | 1,432 | 1,923 | 2,451 | 2,526 | 2,167 | 144,320,287 | 283 | 13,098 | 
| internal.2 | 6 | 2,982 | 27,206 | 3,846,493 | 374,819,800 | 10,869 | 1,453 | 1,914 | 2,448 | 2,557 | 2,212 | 146,057,156 | 279 | 13,355 | 
| leaf | 0 | 1,405 | 13,018 | 1,847,991 | 180,464,344 | 5,516 | 752 | 1,040 | 1,179 | 1,316 | 1,089 | 70,644,665 | 135 | 6,097 | 
| leaf | 1 | 1,219 | 11,852 | 1,605,968 | 170,765,016 | 5,296 | 672 | 980 | 1,161 | 1,213 | 1,138 | 60,645,385 | 128 | 5,337 | 
| leaf | 2 | 1,212 | 11,859 | 1,606,641 | 170,765,016 | 5,302 | 662 | 978 | 1,157 | 1,224 | 1,134 | 60,652,115 | 143 | 5,345 | 
| leaf | 3 | 1,216 | 11,886 | 1,606,629 | 170,765,016 | 5,217 | 665 | 963 | 1,105 | 1,228 | 1,120 | 60,651,995 | 129 | 5,453 | 
| leaf | 4 | 1,212 | 11,884 | 1,606,759 | 170,765,016 | 5,302 | 651 | 975 | 1,162 | 1,251 | 1,133 | 60,653,295 | 126 | 5,370 | 
| leaf | 5 | 1,190 | 11,882 | 1,606,159 | 170,765,016 | 5,281 | 676 | 979 | 1,117 | 1,243 | 1,135 | 60,647,295 | 127 | 5,411 | 
| leaf | 6 | 1,258 | 12,524 | 1,695,249 | 178,179,032 | 5,511 | 691 | 1,035 | 1,189 | 1,293 | 1,159 | 64,295,507 | 138 | 5,755 | 
| root | 0 | 1,476 | 45,307 | 1,923,977 | 190,950,104 | 36,935 | 767 | 8,220 | 12,607 | 3,862 | 11,326 | 74,290,893 | 146 | 6,896 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 944 | 11,404 | 1,747,603 | 197,440,542 | 4,594 | 325 | 547 | 1,696 | 984 | 850 | 59,808,409 | 187 | 5,866 | 
| fib_e2e | 1 | 881 | 11,205 | 1,747,502 | 197,384,386 | 4,135 | 314 | 468 | 1,571 | 964 | 695 | 59,780,846 | 120 | 6,189 | 
| fib_e2e | 2 | 977 | 10,809 | 1,747,502 | 197,384,386 | 4,139 | 300 | 445 | 1,758 | 981 | 536 | 59,780,837 | 117 | 5,693 | 
| fib_e2e | 3 | 863 | 10,810 | 1,747,502 | 197,384,386 | 4,167 | 299 | 440 | 1,802 | 958 | 549 | 59,781,156 | 116 | 5,780 | 
| fib_e2e | 4 | 863 | 10,829 | 1,747,502 | 197,384,386 | 4,192 | 297 | 437 | 1,788 | 993 | 549 | 59,781,455 | 125 | 5,774 | 
| fib_e2e | 5 | 865 | 11,050 | 1,747,502 | 197,384,386 | 4,397 | 292 | 438 | 1,953 | 994 | 591 | 59,781,445 | 125 | 5,788 | 
| fib_e2e | 6 | 758 | 9,979 | 1,515,024 | 197,432,594 | 4,193 | 292 | 440 | 1,725 | 1,012 | 553 | 51,983,912 | 167 | 5,028 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-fib_e2e.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-halo2_outer.cell_tracker_span.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-halo2_outer.cell_tracker_span.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-halo2_outer.cell_tracker_span.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-halo2_outer.cell_tracker_span.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.0.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.1.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-internal.2.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57/fib_e2e-352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57-root.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/352ca42a37ead0ffeb0ce2ff559e98bb1d1dcf57

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12970394348)
