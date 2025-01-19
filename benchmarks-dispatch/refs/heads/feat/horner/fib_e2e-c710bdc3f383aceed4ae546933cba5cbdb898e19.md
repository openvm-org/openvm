| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  865.54 |  488.54 |
| fib_e2e |  79.80 |  11.79 |
| leaf |  154.74 |  23.75 |
| internal.0 |  177.22 |  50.71 |
| internal.1 |  103.06 |  51.56 |
| internal.2 |  51.93 |  51.93 |
| root |  81.17 |  81.17 |
| halo2_outer |  175.62 |  175.62 |
| halo2_wrapper |  42.01 |  42.01 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  11,399.43 |  79,796 |  11,789 |  10,654 |
| `main_cells_used     ` |  58,688,415.43 |  410,818,908 |  59,825,673 |  52,001,176 |
| `total_cycles        ` |  1,714,305.29 |  12,000,137 |  1,747,603 |  1,515,024 |
| `execute_time_ms     ` |  5,638.43 |  39,469 |  5,779 |  4,997 |
| `trace_gen_time_ms   ` |  889.86 |  6,229 |  991 |  760 |
| `stark_prove_excluding_trace_time_ms` |  4,871.14 |  34,098 |  5,048 |  4,777 |
| `main_trace_commit_time_ms` |  644.86 |  4,514 |  798 |  585 |
| `generate_perm_trace_time_ms` |  142.57 |  998 |  197 |  117 |
| `perm_trace_commit_time_ms` |  1,813 |  12,691 |  1,931 |  1,664 |
| `quotient_poly_compute_time_ms` |  807.71 |  5,654 |  842 |  725 |
| `quotient_poly_commit_time_ms` |  441.71 |  3,092 |  526 |  368 |
| `pcs_opening_time_ms ` |  1,019 |  7,133 |  1,057 |  987 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  22,105.14 |  154,736 |  23,746 |  21,538 |
| `main_cells_used     ` |  95,146,956.57 |  666,028,696 |  110,760,086 |  91,424,498 |
| `total_cycles        ` |  2,758,948.43 |  19,312,639 |  3,216,795 |  2,649,626 |
| `execute_time_ms     ` |  9,552.71 |  66,869 |  10,873 |  9,058 |
| `trace_gen_time_ms   ` |  1,605.29 |  11,237 |  1,787 |  1,541 |
| `stark_prove_excluding_trace_time_ms` |  10,947.14 |  76,630 |  11,233 |  10,805 |
| `main_trace_commit_time_ms` |  2,183.71 |  15,286 |  2,235 |  2,129 |
| `generate_perm_trace_time_ms` |  235 |  1,645 |  347 |  207 |
| `perm_trace_commit_time_ms` |  2,008.14 |  14,057 |  2,137 |  1,939 |
| `quotient_poly_compute_time_ms` |  2,402.29 |  16,816 |  2,448 |  2,350 |
| `quotient_poly_commit_time_ms` |  1,843.71 |  12,906 |  1,915 |  1,798 |
| `pcs_opening_time_ms ` |  2,271.14 |  15,898 |  2,356 |  2,232 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  44,305 |  177,220 |  50,707 |  26,379 |
| `main_cells_used     ` |  198,675,173.50 |  794,700,694 |  226,780,243 |  114,408,434 |
| `total_cycles        ` |  5,982,402.75 |  23,929,611 |  6,839,770 |  3,416,189 |
| `execute_time_ms     ` |  21,149.25 |  84,597 |  24,333 |  12,304 |
| `trace_gen_time_ms   ` |  3,396 |  13,584 |  3,989 |  1,959 |
| `stark_prove_excluding_trace_time_ms` |  19,759.75 |  79,039 |  22,559 |  12,116 |
| `main_trace_commit_time_ms` |  3,855.25 |  15,421 |  4,627 |  2,138 |
| `generate_perm_trace_time_ms` |  437.50 |  1,750 |  444 |  421 |
| `perm_trace_commit_time_ms` |  3,875.75 |  15,503 |  4,274 |  2,845 |
| `quotient_poly_compute_time_ms` |  4,205.25 |  16,821 |  4,830 |  2,379 |
| `quotient_poly_commit_time_ms` |  3,223.25 |  12,893 |  3,679 |  1,930 |
| `pcs_opening_time_ms ` |  4,160.75 |  16,643 |  4,762 |  2,401 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  51,529.50 |  103,059 |  51,564 |  51,495 |
| `main_cells_used     ` |  232,929,110.50 |  465,858,221 |  234,181,114 |  231,677,107 |
| `total_cycles        ` |  7,009,240 |  14,018,480 |  7,064,557 |  6,953,923 |
| `execute_time_ms     ` |  25,118 |  50,236 |  25,302 |  24,934 |
| `trace_gen_time_ms   ` |  4,031.50 |  8,063 |  4,132 |  3,931 |
| `stark_prove_excluding_trace_time_ms` |  22,380 |  44,760 |  22,630 |  22,130 |
| `main_trace_commit_time_ms` |  4,372 |  8,744 |  4,641 |  4,103 |
| `generate_perm_trace_time_ms` |  440 |  880 |  444 |  436 |
| `perm_trace_commit_time_ms` |  4,274 |  8,548 |  4,281 |  4,267 |
| `quotient_poly_compute_time_ms` |  4,856.50 |  9,713 |  4,872 |  4,841 |
| `quotient_poly_commit_time_ms` |  3,676.50 |  7,353 |  3,691 |  3,662 |
| `pcs_opening_time_ms ` |  4,758 |  9,516 |  4,767 |  4,749 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  51,929 |  51,929 |  51,929 |  51,929 |
| `main_cells_used     ` |  234,176,254 |  234,176,254 |  234,176,254 |  234,176,254 |
| `total_cycles        ` |  7,064,071 |  7,064,071 |  7,064,071 |  7,064,071 |
| `execute_time_ms     ` |  25,539 |  25,539 |  25,539 |  25,539 |
| `trace_gen_time_ms   ` |  3,925 |  3,925 |  3,925 |  3,925 |
| `stark_prove_excluding_trace_time_ms` |  22,465 |  22,465 |  22,465 |  22,465 |
| `main_trace_commit_time_ms` |  4,634 |  4,634 |  4,634 |  4,634 |
| `generate_perm_trace_time_ms` |  439 |  439 |  439 |  439 |
| `perm_trace_commit_time_ms` |  4,089 |  4,089 |  4,089 |  4,089 |
| `quotient_poly_compute_time_ms` |  4,838 |  4,838 |  4,838 |  4,838 |
| `quotient_poly_commit_time_ms` |  3,722 |  3,722 |  3,722 |  3,722 |
| `pcs_opening_time_ms ` |  4,740 |  4,740 |  4,740 |  4,740 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  81,172 |  81,172 |  81,172 |  81,172 |
| `main_cells_used     ` |  118,184,203 |  118,184,203 |  118,184,203 |  118,184,203 |
| `total_cycles        ` |  3,531,442 |  3,531,442 |  3,531,442 |  3,531,442 |
| `execute_time_ms     ` |  12,872 |  12,872 |  12,872 |  12,872 |
| `trace_gen_time_ms   ` |  2,752 |  2,752 |  2,752 |  2,752 |
| `stark_prove_excluding_trace_time_ms` |  65,548 |  65,548 |  65,548 |  65,548 |
| `main_trace_commit_time_ms` |  19,927 |  19,927 |  19,927 |  19,927 |
| `generate_perm_trace_time_ms` |  229 |  229 |  229 |  229 |
| `perm_trace_commit_time_ms` |  20,538 |  20,538 |  20,538 |  20,538 |
| `quotient_poly_compute_time_ms` |  2,746 |  2,746 |  2,746 |  2,746 |
| `quotient_poly_commit_time_ms` |  14,835 |  14,835 |  14,835 |  14,835 |
| `pcs_opening_time_ms ` |  7,270 |  7,270 |  7,270 |  7,270 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  175,620 |  175,620 |  175,620 |  175,620 |
| `main_cells_used     ` |  112,170,012 |  112,170,012 |  112,170,012 |  112,170,012 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  42,011 |  42,011 |  42,011 |  42,011 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 6,740 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 175,620 |  | 112,170,012 | 
| halo2_wrapper | 42,011 |  |  | 

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
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 481,183 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 481,183 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 481,183 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 240,304 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 1 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 2 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 3 | BNE | 69 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 50,922 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 50,922 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 50,922 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 25,461 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 5,428 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 7,778,600 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 1 | BNE | 7,768,158 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 2 | BNE | 7,761,741 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 3 | BNE | 3,891,324 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,538,320 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 3,547,980 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 3,553,776 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 1,767,228 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 4,644,068 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 4,634,408 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 4,628,612 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 2,323,966 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 766,038 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 766,038 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 766,038 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 383,019 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 44,896 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 44,896 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 44,896 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 22,448 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 13,340,529 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 13,350,189 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 13,355,985 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 6,668,551 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 10 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 259,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | For | 1 | JAL | 259,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | For | 2 | JAL | 259,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | For | 3 | JAL | 129,840 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 918,640 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 906,740 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 906,430 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 415,670 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 60 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 30 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 642,960 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 642,960 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 642,960 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 321,530 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 1 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 2 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 84,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 84,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 84,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 42,240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 43,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 43,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 43,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 21,720 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 1,809,600 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 1,809,600 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 1,809,600 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 904,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 1 | ADD | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 2 | ADD | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 3 | ADD | 39,990 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 4,595,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 4,582,710 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 4,575,150 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 2,300,550 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 3,875,820 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 1 | ADD | 3,875,700 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 2 | ADD | 3,875,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 3 | ADD | 1,938,090 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 17,711,340 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 17,697,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 17,688,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 8,860,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 6,730,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,986,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 6,730,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 1,986,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 6,730,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 1,986,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 3,365,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 993,210 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 22,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 23,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 11,730 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 10,146,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 1 | ADD | 10,132,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 2 | ADD | 10,124,010 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 3 | ADD | 5,075,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 181,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 181,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 181,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 90,720 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 2,213,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 2,213,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 2,213,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 1,109,250 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 1,049,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 1,049,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 1,049,310 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 526,410 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 753,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 376,740 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 376,740 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 794,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 794,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 794,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 538,860 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 397,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 269,430 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 3,066,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 2,553,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 3,066,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 2,553,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 3,066,660 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 2,553,480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 1,533,330 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 1,276,740 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 301,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 301,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 301,440 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 150,720 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 27,120 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 27,120 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 27,120 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 13,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 277,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 277,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 277,200 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 138,600 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 8,985,900 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 1 | MUL | 8,960,700 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 2 | MUL | 8,945,580 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 3 | MUL | 4,497,990 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 81,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 81,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 81,000 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 40,500 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 1,847,850 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 1,847,850 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 1,847,850 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 923,940 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 13,680 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 1 | MUL | 13,680 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 2 | MUL | 13,680 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 3 | MUL | 6,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 541,800 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 270,900 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 270,900 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 468,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 468,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 468,360 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 22,560 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 234,180 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 11,280 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 60 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 30 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 1,421,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 1,150,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 1,421,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 1,150,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 1,421,520 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 1,150,380 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 710,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 575,190 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 1,019,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 339,960 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 509,940 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 169,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 15,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 15,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 15,840 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 7,920 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 89,760 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 44,880 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 0 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 1 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 2 | SUB | 480 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 3 | SUB | 240 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 79,980 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 39,990 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 883,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 1 | SUB | 883,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 2 | SUB | 883,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 3 | SUB | 441,510 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 62,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 62,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 62,640 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 31,320 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 52,920 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 52,920 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 52,920 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 26,460 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 1,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 1,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 1,020 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 510 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 20,599,890 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 20,625,090 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 20,640,210 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 10,295,190 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 5,152,100 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 5,152,100 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 5,152,100 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 2,576,150 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 14,065,150 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 14,065,150 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 14,065,150 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 7,032,700 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 4,446,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 4,446,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 4,446,550 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 2,225,375 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 10,791,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 10,791,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 10,791,800 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 5,396,125 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 3,441,750 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 3,441,750 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 3,441,750 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 1,722,275 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW4 | 1,581,204 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW4 | 1,581,204 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW4 | 1,581,204 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW4 | 790,602 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW4 | 1,102,076 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW4 | 1,102,076 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW4 | 1,102,076 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW4 | 551,038 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 998,800 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 998,800 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 998,800 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 499,400 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 523,520 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 523,520 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 523,520 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 261,760 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 14,960 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 7,480 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,144,760 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 1,143,560 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 1,142,560 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 572,480 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 92,400 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 92,400 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 92,400 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 46,200 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 264,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 264,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 264,880 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 132,440 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 11,312,784 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 11,312,784 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 2 | COMP_POS2 | 11,312,784 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 3 | COMP_POS2 | 5,656,392 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 5,604,888 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 5,634,120 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 5,648,736 | 
| internal.0 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 2,780,520 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 6,241,536 | 
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
| internal.0 | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 97,776 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 48,888 | 
| internal.0 | PhantomAir | CT-poseidon2-hash | 0 | PHANTOM | 30,240 | 
| internal.0 | PhantomAir | CT-poseidon2-hash | 1 | PHANTOM | 30,240 | 
| internal.0 | PhantomAir | CT-poseidon2-hash | 2 | PHANTOM | 30,240 | 
| internal.0 | PhantomAir | CT-poseidon2-hash | 3 | PHANTOM | 15,120 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-ext | 0 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-ext | 1 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-ext | 2 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-ext | 3 | PHANTOM | 10,584 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-setup | 0 | PHANTOM | 1,291,248 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-setup | 1 | PHANTOM | 1,291,248 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-setup | 2 | PHANTOM | 1,291,248 | 
| internal.0 | PhantomAir | CT-poseidon2-hash-setup | 3 | PHANTOM | 645,624 | 
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
| internal.0 | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 1 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 2 | PHANTOM | 21,168 | 
| internal.0 | PhantomAir | CT-verify-batch-ext | 3 | PHANTOM | 10,584 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast | 0 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast | 1 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast | 2 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast | 3 | PHANTOM | 25,704 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast-setup | 1 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast-setup | 2 | PHANTOM | 51,408 | 
| internal.0 | PhantomAir | CT-verify-batch-reduce-fast-setup | 3 | PHANTOM | 25,704 | 
| internal.0 | PhantomAir | CT-verify-query | 0 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 1 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 2 | PHANTOM | 1,008 | 
| internal.0 | PhantomAir | CT-verify-query | 3 | PHANTOM | 504 | 
| internal.0 | PhantomAir | HintBitsF | 0 | PHANTOM | 516 | 
| internal.0 | PhantomAir | HintBitsF | 1 | PHANTOM | 516 | 
| internal.0 | PhantomAir | HintBitsF | 2 | PHANTOM | 516 | 
| internal.0 | PhantomAir | HintBitsF | 3 | PHANTOM | 258 | 
| internal.0 | PhantomAir | HintInputVec | 0 | PHANTOM | 275,838 | 
| internal.0 | PhantomAir | HintInputVec | 1 | PHANTOM | 275,838 | 
| internal.0 | PhantomAir | HintInputVec | 2 | PHANTOM | 275,838 | 
| internal.0 | PhantomAir | HintInputVec | 3 | PHANTOM | 137,946 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 497,007 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 489,279 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 4 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 5 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 52,854 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 51,888 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 4 | BNE | 8,029,047 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 5 | BNE | 7,919,038 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 3,530,776 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 3,521,024 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 4,888,604 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 4,780,228 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 816,454 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 791,246 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 46,828 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 45,862 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 13,905,915 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 13,610,066 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 10 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 10 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | For | 4 | JAL | 262,190 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | For | 5 | JAL | 260,930 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 977,840 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 941,740 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 60 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 60 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 669,860 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 656,410 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 4 | ADD | 30 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 5 | ADD | 30 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 89,280 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 88,800 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 43,440 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 43,440 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 1,844,160 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 1,828,800 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 4 | ADD | 79,980 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 5 | ADD | 79,980 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 4,824,030 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 4,728,030 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 4 | ADD | 4,026,420 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 5 | ADD | 3,951,270 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 18,417,720 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 18,084,390 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 7,063,260 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 2,079,300 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 6,896,820 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 2,032,650 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 44,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 44,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 23,460 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 23,460 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 4 | ADD | 10,472,670 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 5 | ADD | 10,329,180 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 185,280 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 185,280 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 2,233,620 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 2,223,540 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 1,081,110 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 1,065,570 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 766,080 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 766,080 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 759,780 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 759,780 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 794,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 538,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 794,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 538,860 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 3,094,380 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 2,568,600 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 3,080,520 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 2,561,040 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 311,520 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 306,480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 27,120 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 27,120 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 278,160 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 277,680 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 4 | MUL | 9,439,500 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 5 | MUL | 9,247,980 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 81,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 81,000 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 1,923,450 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 1,885,650 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 4 | MUL | 13,680 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 5 | MUL | 13,680 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 544,320 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 544,320 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 543,060 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 543,060 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 489,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 22,560 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 479,640 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 22,560 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 60 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 1,424,040 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 1,155,420 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 1,422,780 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 1,152,900 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 1,019,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 339,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 1,019,880 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 339,960 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 15,840 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 15,840 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 89,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 89,760 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 4 | SUB | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 5 | SUB | 480 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 79,980 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 79,980 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 4 | SUB | 893,100 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 5 | SUB | 888,060 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 65,400 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 64,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 55,440 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 54,180 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 1,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 1,020 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 21,390,330 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 20,960,310 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 5,207,600 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 5,181,050 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 14,497,900 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 14,281,525 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 4,481,350 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 4,464,350 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 11,303,350 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 11,047,975 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 3,563,650 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 3,502,700 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW4 | 1,604,052 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW4 | 1,592,628 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW4 | 1,119,280 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW4 | 1,110,678 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 1,007,040 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 1,003,560 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 526,880 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 525,200 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 14,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 14,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 1,189,360 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 1,170,440 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 92,720 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 92,560 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 274,960 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 269,920 | 
| internal.1 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 4 | COMP_POS2 | 12,072,816 | 
| internal.1 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 5 | COMP_POS2 | 11,692,800 | 
| internal.1 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 5,592,360 | 
| internal.1 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 5,577,396 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-InitializePcsConst | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-ReadProofsFromInput | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 4 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-VerifyProofs | 5 | PHANTOM | 12 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 97,776 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 97,776 | 
| internal.1 | PhantomAir | CT-poseidon2-hash | 4 | PHANTOM | 30,240 | 
| internal.1 | PhantomAir | CT-poseidon2-hash | 5 | PHANTOM | 30,240 | 
| internal.1 | PhantomAir | CT-poseidon2-hash-ext | 4 | PHANTOM | 22,176 | 
| internal.1 | PhantomAir | CT-poseidon2-hash-ext | 5 | PHANTOM | 21,672 | 
| internal.1 | PhantomAir | CT-poseidon2-hash-setup | 4 | PHANTOM | 1,291,248 | 
| internal.1 | PhantomAir | CT-poseidon2-hash-setup | 5 | PHANTOM | 1,291,248 | 
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
| internal.1 | PhantomAir | CT-verify-batch-ext | 4 | PHANTOM | 22,176 | 
| internal.1 | PhantomAir | CT-verify-batch-ext | 5 | PHANTOM | 21,672 | 
| internal.1 | PhantomAir | CT-verify-batch-reduce-fast | 4 | PHANTOM | 52,416 | 
| internal.1 | PhantomAir | CT-verify-batch-reduce-fast | 5 | PHANTOM | 51,912 | 
| internal.1 | PhantomAir | CT-verify-batch-reduce-fast-setup | 4 | PHANTOM | 52,416 | 
| internal.1 | PhantomAir | CT-verify-batch-reduce-fast-setup | 5 | PHANTOM | 51,912 | 
| internal.1 | PhantomAir | CT-verify-query | 4 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | CT-verify-query | 5 | PHANTOM | 1,008 | 
| internal.1 | PhantomAir | HintBitsF | 4 | PHANTOM | 516 | 
| internal.1 | PhantomAir | HintBitsF | 5 | PHANTOM | 516 | 
| internal.1 | PhantomAir | HintInputVec | 4 | PHANTOM | 290,466 | 
| internal.1 | PhantomAir | HintInputVec | 5 | PHANTOM | 283,152 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 10,856 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 184 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 497,007 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 6 | BNE | 161 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 52,854 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 10,856 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 6 | BNE | 8,029,047 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 3,530,776 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 4,888,604 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 816,454 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 46,828 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 13,905,915 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 10 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | For | 6 | JAL | 262,190 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 972,980 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 60 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 669,860 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 1,196 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 6 | ADD | 30 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 89,280 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 43,440 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 1,844,160 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 6 | ADD | 79,980 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 4,824,030 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 6 | ADD | 4,026,420 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 18,417,720 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 7,063,260 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 2,079,300 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 480 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 44,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 23,460 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 6 | ADD | 10,472,670 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 185,280 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 2,233,620 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 1,081,110 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 766,080 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 766,080 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 794,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 538,860 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 60 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 3,094,380 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 2,568,600 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 311,520 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 27,120 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 278,160 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 6 | MUL | 9,439,500 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 81,000 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 1,923,450 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 6 | MUL | 13,680 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 544,320 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 544,320 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 489,960 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 22,560 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 60 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 1,424,040 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 1,155,420 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 1,019,880 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 339,960 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 15,840 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 89,760 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 6 | SUB | 480 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 79,980 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 6 | SUB | 893,100 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 65,400 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 55,440 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 1,020 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 21,390,330 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 5,207,600 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 14,497,900 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 4,481,350 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 11,303,350 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 3,563,650 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW4 | 1,604,052 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW4 | 1,119,280 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 1,007,040 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 526,880 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 14,960 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 1,189,360 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 92,720 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 274,960 | 
| internal.2 | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 12,072,816 | 
| internal.2 | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 5,592,360 | 
| internal.2 | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 12,483,072 | 
| internal.2 | PhantomAir | CT-InitializePcsConst | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-ReadProofsFromInput | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-VerifyProofs | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 97,776 | 
| internal.2 | PhantomAir | CT-poseidon2-hash | 6 | PHANTOM | 30,240 | 
| internal.2 | PhantomAir | CT-poseidon2-hash-ext | 6 | PHANTOM | 22,176 | 
| internal.2 | PhantomAir | CT-poseidon2-hash-setup | 6 | PHANTOM | 1,291,248 | 
| internal.2 | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 135,072 | 
| internal.2 | PhantomAir | CT-stage-c-build-rounds | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verifier-verify | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verify-pcs | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-e-verify-constraints | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-verify-batch | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 22,176 | 
| internal.2 | PhantomAir | CT-verify-batch-reduce-fast | 6 | PHANTOM | 52,416 | 
| internal.2 | PhantomAir | CT-verify-batch-reduce-fast-setup | 6 | PHANTOM | 52,416 | 
| internal.2 | PhantomAir | CT-verify-query | 6 | PHANTOM | 1,008 | 
| internal.2 | PhantomAir | HintBitsF | 6 | PHANTOM | 516 | 
| internal.2 | PhantomAir | HintInputVec | 6 | PHANTOM | 290,466 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 248,032 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 248,216 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 24,679 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 23,161 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 23,161 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 23,161 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 23,161 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 23,161 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 23,667 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,520 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 4,002 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 4,002 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 4,002 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 4,002 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 4,002 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 4,531 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 3,132,738 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 1 | BNE | 2,371,921 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 2 | BNE | 2,371,921 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 3 | BNE | 2,371,921 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 4 | BNE | 2,371,921 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 5 | BNE | 2,371,921 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 6 | BNE | 2,630,901 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 2,067,608 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 1,301,547 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 1,301,547 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 1,301,547 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 1,301,547 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 1,301,547 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 1,552,707 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 1,553,926 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 1,386,670 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 1,386,670 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 1,386,670 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 1,386,670 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 1,386,670 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 1,433,406 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 405,145 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 399,349 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 21,643 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 20,953 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 20,953 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 20,953 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 20,953 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 20,953 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 21,183 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 7,067,394 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 6,066,342 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 6,066,342 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 6,066,342 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 6,066,342 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 6,066,342 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 6,415,804 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 120,310 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 1 | JAL | 96,730 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 2 | JAL | 96,730 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 3 | JAL | 96,730 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 4 | JAL | 96,730 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 5 | JAL | 96,730 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | For | 6 | JAL | 103,760 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 285,330 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 245,260 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 265,940 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 247,980 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 254,030 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 249,470 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 259,540 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 10 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 20 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 321,150 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 297,210 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 297,210 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 297,210 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 297,210 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 297,210 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 308,680 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 829,560 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 661,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 661,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 661,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 661,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 661,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 718,440 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 1 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 2 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 3 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 4 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 5 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 6 | ADD | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 1,292,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 1,074,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 1,074,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 1,074,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 1,074,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 1,074,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 1,135,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,915,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 1 | ADD | 1,718,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 2 | ADD | 1,718,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 3 | ADD | 1,718,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 4 | ADD | 1,718,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 5 | ADD | 1,718,940 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 6 | ADD | 1,784,610 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 8,198,820 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 6,986,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 6,986,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 6,986,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 6,986,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 6,986,520 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 7,398,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,381,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,007,610 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 3,226,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 963,690 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 3,226,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 963,690 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 3,226,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 963,690 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 3,226,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 963,690 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 3,226,800 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 963,690 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 3,325,680 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 988,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 6,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 4,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 5,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 3,840 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 2,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 3,000 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 4,086,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 1 | ADD | 3,093,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 2 | ADD | 3,093,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 3 | ADD | 3,093,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 4 | ADD | 3,093,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 5 | ADD | 3,093,810 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 6 | ADD | 3,431,610 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 96,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 58,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 58,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 58,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 58,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 58,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 70,200 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 1,341,750 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 1,159,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 1,159,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 1,159,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 1,159,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 1,159,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 1,180,230 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 571,110 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 543,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 543,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 543,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 543,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 543,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 545,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 332,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 332,640 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 272,160 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 292,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 292,320 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 345,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 236,490 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 251,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 251,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 251,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 251,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 251,760 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 173,130 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 282,960 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 194,250 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,378,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,146,180 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 1,078,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 872,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 1,078,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 872,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 1,078,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 872,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 1,078,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 872,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 1,078,050 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 872,580 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 1,179,450 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 963,780 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 2,499,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 1 | MUL | 2,094,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 2 | MUL | 2,094,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 3 | MUL | 2,094,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 4 | MUL | 2,094,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 5 | MUL | 2,094,930 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 6 | MUL | 2,206,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 40,590 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 40,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 40,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 40,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 40,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 40,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 40,470 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 937,950 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 860,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 860,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 860,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 860,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 860,550 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 881,310 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 222,150 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 9,240 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 218,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 218,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 218,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 218,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 218,370 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 6,360 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 232,350 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 20,040 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 30 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 612,480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 495,060 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 451,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 364,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 451,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 364,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 451,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 364,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 451,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 364,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 451,380 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 364,920 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 505,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 408,300 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 39,990 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 439,080 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 1 | SUB | 363,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 2 | SUB | 363,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 3 | SUB | 363,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 4 | SUB | 363,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 5 | SUB | 363,300 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 6 | SUB | 383,520 | 
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
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 600 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 420 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 420 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 420 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 420 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 420 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 480 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 10,950,210 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 9,193,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 9,193,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 9,193,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 9,193,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 9,193,410 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 9,770,190 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 2,714,350 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 2,030,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 2,030,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 2,030,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 2,030,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 2,030,950 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 2,307,550 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 6,043,900 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 4,944,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 4,944,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 4,944,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 4,944,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 4,944,250 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 5,290,500 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 2,647,400 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 1,989,050 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 1,989,050 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 1,989,050 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 1,989,050 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 1,989,050 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 2,223,500 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,719,450 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 5,033,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 5,033,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 5,033,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 5,033,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 5,033,800 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 5,304,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,630,200 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 1,440,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 1,440,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 1,440,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 1,440,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 1,440,000 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 1,503,400 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW4 | 743,172 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW4 | 588,336 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW4 | 588,336 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW4 | 588,336 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW4 | 588,336 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW4 | 588,336 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW4 | 641,036 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW4 | 499,902 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW4 | 430,746 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW4 | 430,746 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW4 | 430,746 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW4 | 430,746 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW4 | 430,746 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW4 | 453,798 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 497,320 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 355,400 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 402,520 | 
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
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 341,200 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 268,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 268,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 268,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 268,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 268,520 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 290,040 | 
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
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 5,963,328 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 5,875,632 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 2 | COMP_POS2 | 5,875,632 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 3 | COMP_POS2 | 5,875,632 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 4 | COMP_POS2 | 5,875,632 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 5 | COMP_POS2 | 5,875,632 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 5,894,076 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 3,218,652 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 2,238,336 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 2,238,336 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 2,238,336 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 2,238,336 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 2,238,336 | 
| leaf | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 2,589,468 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 7,547,904 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 4,838,400 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 4,838,400 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 4,838,400 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 4,838,400 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 4,838,400 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 5,827,584 | 
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
| leaf | PhantomAir | CT-poseidon2-hash | 0 | PHANTOM | 22,176 | 
| leaf | PhantomAir | CT-poseidon2-hash | 1 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash | 2 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash | 3 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash | 4 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash | 5 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash | 6 | PHANTOM | 19,152 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 0 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 1 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 2 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 3 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 4 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 5 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-ext | 6 | PHANTOM | 10,080 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 0 | PHANTOM | 744,912 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 1 | PHANTOM | 478,800 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 2 | PHANTOM | 478,800 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 3 | PHANTOM | 478,800 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 4 | PHANTOM | 478,800 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 5 | PHANTOM | 478,800 | 
| leaf | PhantomAir | CT-poseidon2-hash-setup | 6 | PHANTOM | 575,568 | 
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
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 0 | PHANTOM | 32,256 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 1 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 2 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 3 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 4 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 5 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast | 6 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 32,256 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 1 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 2 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 3 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 4 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 5 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-batch-reduce-fast-setup | 6 | PHANTOM | 29,232 | 
| leaf | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 1 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 2 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 3 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 4 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 5 | PHANTOM | 504 | 
| leaf | PhantomAir | CT-verify-query | 6 | PHANTOM | 504 | 
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 1 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 2 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 3 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 4 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 5 | PHANTOM | 258 | 
| leaf | PhantomAir | HintBitsF | 6 | PHANTOM | 258 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 136,674 | 
| leaf | PhantomAir | HintInputVec | 1 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 2 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 3 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 4 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 5 | PHANTOM | 129,942 | 
| leaf | PhantomAir | HintInputVec | 6 | PHANTOM | 134,784 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,428 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 248,400 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 115 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 26,427 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,451 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | For | 0 | BNE | 4,014,535 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 1,765,388 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 2,444,302 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 408,227 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 23,414 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,953,567 | 
| root | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 10 | 
| root | <JalNativeAdapterAir,JalCoreAir> | For | 0 | JAL | 131,100 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 472,690 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 30 | 
| root | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 334,910 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,104 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> |  | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 44,640 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 21,720 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 922,080 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddF | 0 | ADD | 39,990 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 2,412,390 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddV | 0 | ADD | 2,013,150 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 9,209,460 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 3,532,200 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 1,040,010 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 240 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,440 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,730 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | For | 0 | ADD | 5,236,350 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 92,640 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 1,119,600 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 542,670 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 383,040 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 383,040 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 397,440 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 269,430 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,547,190 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,284,300 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 155,760 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 13,560 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 139,080 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulF | 0 | MUL | 4,719,750 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 40,500 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 961,740 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | NegE | 0 | MUL | 6,840 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 272,160 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 272,160 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 244,980 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 11,280 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 30 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 712,020 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 577,710 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 509,940 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 169,980 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 7,920 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 44,880 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubF | 0 | SUB | 240 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 39,990 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubV | 0 | SUB | 446,550 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 32,700 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 27,720 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 510 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 10,695,960 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 2,607,100 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 7,248,900 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 2,247,175 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 5,652,325 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,783,225 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW4 | 802,026 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW4 | 559,640 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 503,520 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 263,440 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,480 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 594,680 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 46,360 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 137,480 | 
| root | Arc<BabyBearParameters>, 1> | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 6,040,584 | 
| root | Arc<BabyBearParameters>, 1> | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 2,796,180 | 
| root | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 6,241,536 | 
| root | PhantomAir | CT-ExtractPublicValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 48,888 | 
| root | PhantomAir | CT-poseidon2-hash | 0 | PHANTOM | 15,120 | 
| root | PhantomAir | CT-poseidon2-hash-ext | 0 | PHANTOM | 11,088 | 
| root | PhantomAir | CT-poseidon2-hash-setup | 0 | PHANTOM | 645,624 | 
| root | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 67,536 | 
| root | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-verify-batch | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 11,088 | 
| root | PhantomAir | CT-verify-batch-reduce-fast | 0 | PHANTOM | 26,208 | 
| root | PhantomAir | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 26,208 | 
| root | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| root | PhantomAir | HintBitsF | 0 | PHANTOM | 258 | 
| root | PhantomAir | HintInputVec | 0 | PHANTOM | 145,218 | 

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
| internal.0 | AccessAdapterAir<2> | 0 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 1 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 2 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.0 | AccessAdapterAir<2> | 3 | 524,288 |  | 16 | 11 | 14,155,776 | 
| internal.0 | AccessAdapterAir<4> | 0 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 1 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 2 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.0 | AccessAdapterAir<4> | 3 | 262,144 |  | 16 | 13 | 7,602,176 | 
| internal.0 | AccessAdapterAir<8> | 0 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 1 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 2 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.0 | AccessAdapterAir<8> | 3 | 65,536 |  | 16 | 17 | 2,162,688 | 
| internal.0 | FriReducedOpeningAir | 0 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 1 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 2 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.0 | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 36 | 348 | 12,582,912 | 
| internal.0 | PhantomAir | 0 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.0 | PhantomAir | 1 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.0 | PhantomAir | 2 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.0 | PhantomAir | 3 | 262,144 |  | 8 | 6 | 3,670,016 | 
| internal.0 | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 1 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 2 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | ProgramAir | 3 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 131,072 |  | 12 | 10 | 2,883,584 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| internal.1 | AccessAdapterAir<2> | 4 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.1 | AccessAdapterAir<2> | 5 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.1 | AccessAdapterAir<4> | 4 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.1 | AccessAdapterAir<4> | 5 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.1 | AccessAdapterAir<8> | 4 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.1 | AccessAdapterAir<8> | 5 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.1 | PhantomAir | 4 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.1 | PhantomAir | 5 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.1 | ProgramAir | 4 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | ProgramAir | 5 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.1 | VolatileBoundaryAir | 5 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| internal.2 | AccessAdapterAir<2> | 6 | 1,048,576 |  | 16 | 11 | 28,311,552 | 
| internal.2 | AccessAdapterAir<4> | 6 | 524,288 |  | 16 | 13 | 15,204,352 | 
| internal.2 | AccessAdapterAir<8> | 6 | 131,072 |  | 16 | 17 | 4,325,376 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 76 | 64 | 36,700,160 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 36 | 348 | 25,165,824 | 
| internal.2 | PhantomAir | 6 | 524,288 |  | 8 | 6 | 7,340,032 | 
| internal.2 | ProgramAir | 6 | 131,072 |  | 8 | 10 | 2,359,296 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 2,097,152 |  | 28 | 23 | 106,954,752 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 262,144 |  | 12 | 10 | 5,767,168 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 4,194,304 |  | 20 | 30 | 209,715,200 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 2,097,152 |  | 36 | 25 | 127,926,272 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 36 | 34 | 9,175,040 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 40 | 7,864,320 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 1,048,576 |  | 8 | 11 | 19,922,944 | 
| leaf | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 1 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 2 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 3 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 4 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 5 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<2> | 6 | 524,288 |  | 16 | 11 | 14,155,776 | 
| leaf | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 1 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 2 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 3 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 4 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 5 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<4> | 6 | 262,144 |  | 16 | 13 | 7,602,176 | 
| leaf | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 1 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 2 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 3 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 4 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 5 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | AccessAdapterAir<8> | 6 | 65,536 |  | 16 | 17 | 2,162,688 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 76 | 64 | 18,350,080 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 36 | 348 | 12,582,912 | 
| leaf | PhantomAir | 0 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 1 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 2 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 3 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 4 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 5 | 262,144 |  | 8 | 6 | 3,670,016 | 
| leaf | PhantomAir | 6 | 262,144 |  | 8 | 6 | 3,670,016 | 
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
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 28 | 23 | 26,738,688 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 10 | 1,441,792 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 131,072 |  | 12 | 10 | 2,883,584 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 3 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 36 | 34 | 4,587,520 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 32,768 |  | 36 | 34 | 2,293,760 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 32,768 |  | 20 | 40 | 1,966,080 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 32,768 |  | 20 | 40 | 1,966,080 | 
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
| root | AccessAdapterAir<2> | 0 | 524,288 |  | 16 | 11 | 14,155,776 | 
| root | AccessAdapterAir<4> | 0 | 262,144 |  | 16 | 13 | 7,602,176 | 
| root | AccessAdapterAir<8> | 0 | 65,536 |  | 16 | 17 | 2,162,688 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 76 | 64 | 18,350,080 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 36 | 348 | 12,582,912 | 
| root | PhantomAir | 0 | 262,144 |  | 8 | 6 | 3,670,016 | 
| root | ProgramAir | 0 | 131,072 |  | 8 | 10 | 2,359,296 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 1,048,576 |  | 28 | 23 | 53,477,376 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 131,072 |  | 12 | 10 | 2,883,584 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 30 | 104,857,600 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 36 | 25 | 63,963,136 | 
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
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 1 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 3 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 4 | 256 |  | 8 | 300 | 78,848 | 
| fib_e2e | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 5 | 256 |  | 8 | 300 | 78,848 | 
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
| halo2_outer | VerifierProgram | 745,286 | 200,855 | 
| halo2_outer | VerifierProgram;PoseidonCell | 20,120 |  | 
| halo2_outer | VerifierProgram;stage-c-build-rounds | 334,819 | 697 | 
| halo2_outer | VerifierProgram;stage-c-build-rounds;PoseidonCell | 47,785 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs | 1 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 580,212 | 2,002 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 70,420 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening | 16,296 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;exp-reverse-bits-len | 4,216,086 | 808,080 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 31,980,186 | 5,196,198 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 121,212 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 14,322,420 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 11,524,758 | 322,812 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 3,091,242 | 657,552 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 275,184 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 26,735,184 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 2,508,324 | 52,752 | 
| halo2_outer | VerifierProgram;stage-e-verify-constraints | 6,836,902 | 1,502,626 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 1,333,332 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 1,333,298 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 1,333,271 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 666,667 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 0 | 182,134 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 1 | 180,944 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 2 | 180,913 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 3 | 86,708 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 52 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 3,262,072 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 1 | 3,260,732 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 2 | 3,259,905 | 
| internal.0 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 3 | 1,631,567 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 1,515,894 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 1,515,894 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 1,515,894 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 758,105 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 78,920 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 78,920 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 78,920 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 39,460 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 75,983 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 75,953 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 75,928 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 37,994 | 
| internal.0 | AccessAdapter<2> | 0 | 706,332 | 
| internal.0 | AccessAdapter<2> | 1 | 706,416 | 
| internal.0 | AccessAdapter<2> | 2 | 706,416 | 
| internal.0 | AccessAdapter<2> | 3 | 368,292 | 
| internal.0 | AccessAdapter<4> | 0 | 353,548 | 
| internal.0 | AccessAdapter<4> | 1 | 353,716 | 
| internal.0 | AccessAdapter<4> | 2 | 353,716 | 
| internal.0 | AccessAdapter<4> | 3 | 184,358 | 
| internal.0 | AccessAdapter<8> | 0 | 101,768 | 
| internal.0 | AccessAdapter<8> | 1 | 101,936 | 
| internal.0 | AccessAdapter<8> | 2 | 102,020 | 
| internal.0 | AccessAdapter<8> | 3 | 50,758 | 
| internal.0 | Arc<BabyBearParameters>, 1> | 0 | 48,614 | 
| internal.0 | Arc<BabyBearParameters>, 1> | 1 | 48,698 | 
| internal.0 | Arc<BabyBearParameters>, 1> | 2 | 48,740 | 
| internal.0 | Arc<BabyBearParameters>, 1> | 3 | 24,244 | 
| internal.0 | Boundary | 0 | 631,565 | 
| internal.0 | Boundary | 1 | 631,565 | 
| internal.0 | Boundary | 2 | 631,565 | 
| internal.0 | Boundary | 3 | 371,446 | 
| internal.0 | FriReducedOpeningAir | 0 | 195,048 | 
| internal.0 | FriReducedOpeningAir | 1 | 195,048 | 
| internal.0 | FriReducedOpeningAir | 2 | 195,048 | 
| internal.0 | FriReducedOpeningAir | 3 | 97,524 | 
| internal.0 | PhantomAir | 0 | 331,513 | 
| internal.0 | PhantomAir | 1 | 331,513 | 
| internal.0 | PhantomAir | 2 | 331,513 | 
| internal.0 | PhantomAir | 3 | 165,764 | 
| internal.0 | ProgramChip | 0 | 113,940 | 
| internal.0 | ProgramChip | 1 | 113,940 | 
| internal.0 | ProgramChip | 2 | 113,940 | 
| internal.0 | ProgramChip | 3 | 113,940 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 
| internal.0 | VmConnectorAir | 0 | 2 | 
| internal.0 | VmConnectorAir | 1 | 2 | 
| internal.0 | VmConnectorAir | 2 | 2 | 
| internal.0 | VmConnectorAir | 3 | 2 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 1,382,155 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 1,357,857 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 4 | 190,996 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 5 | 185,915 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 52 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 52 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 4 | 3,374,813 | 
| internal.1 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 5 | 3,320,619 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 1,562,154 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 1,539,104 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 80,098 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 79,509 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 77,648 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 76,916 | 
| internal.1 | AccessAdapter<2> | 4 | 729,464 | 
| internal.1 | AccessAdapter<2> | 5 | 735,888 | 
| internal.1 | AccessAdapter<4> | 4 | 364,988 | 
| internal.1 | AccessAdapter<4> | 5 | 368,200 | 
| internal.1 | AccessAdapter<8> | 4 | 106,232 | 
| internal.1 | AccessAdapter<8> | 5 | 103,878 | 
| internal.1 | Arc<BabyBearParameters>, 1> | 4 | 50,762 | 
| internal.1 | Arc<BabyBearParameters>, 1> | 5 | 49,627 | 
| internal.1 | Boundary | 4 | 658,721 | 
| internal.1 | Boundary | 5 | 721,352 | 
| internal.1 | FriReducedOpeningAir | 4 | 195,048 | 
| internal.1 | FriReducedOpeningAir | 5 | 195,048 | 
| internal.1 | PhantomAir | 4 | 334,623 | 
| internal.1 | PhantomAir | 5 | 333,068 | 
| internal.1 | ProgramChip | 4 | 113,940 | 
| internal.1 | ProgramChip | 5 | 113,940 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 
| internal.1 | VmConnectorAir | 4 | 2 | 
| internal.1 | VmConnectorAir | 5 | 2 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 1,382,155 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | 6 | 190,510 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 52 | 
| internal.2 | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 6 | 3,374,813 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 1,562,154 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 80,098 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 77,648 | 
| internal.2 | AccessAdapter<2> | 6 | 729,464 | 
| internal.2 | AccessAdapter<4> | 6 | 364,988 | 
| internal.2 | AccessAdapter<8> | 6 | 106,232 | 
| internal.2 | Arc<BabyBearParameters>, 1> | 6 | 50,762 | 
| internal.2 | Boundary | 6 | 658,721 | 
| internal.2 | FriReducedOpeningAir | 6 | 195,048 | 
| internal.2 | PhantomAir | 6 | 334,623 | 
| internal.2 | ProgramChip | 6 | 113,940 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 
| internal.2 | VmConnectorAir | 6 | 2 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 631,848 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 514,228 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 514,228 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 514,228 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 514,228 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 514,228 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 553,705 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 72,681 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 1 | 63,923 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 2 | 65,991 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 3 | 64,195 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 4 | 64,800 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 5 | 64,344 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 6 | 67,201 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 36 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,477,893 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 1 | 1,240,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 2 | 1,240,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 3 | 1,240,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 4 | 1,240,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 5 | 1,240,780 | 
| leaf | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 6 | 1,319,889 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 750,212 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 617,522 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 617,522 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 617,522 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 617,522 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 617,522 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 665,158 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 36,561 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 29,973 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 29,973 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 29,973 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 29,973 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 29,973 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 32,201 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 32,063 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 24,270 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 24,270 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 24,270 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 24,270 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 24,270 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 26,811 | 
| leaf | AccessAdapter<2> | 0 | 375,478 | 
| leaf | AccessAdapter<2> | 1 | 320,152 | 
| leaf | AccessAdapter<2> | 2 | 320,152 | 
| leaf | AccessAdapter<2> | 3 | 320,152 | 
| leaf | AccessAdapter<2> | 4 | 320,152 | 
| leaf | AccessAdapter<2> | 5 | 320,152 | 
| leaf | AccessAdapter<2> | 6 | 340,046 | 
| leaf | AccessAdapter<4> | 0 | 187,992 | 
| leaf | AccessAdapter<4> | 1 | 160,246 | 
| leaf | AccessAdapter<4> | 2 | 160,246 | 
| leaf | AccessAdapter<4> | 3 | 160,246 | 
| leaf | AccessAdapter<4> | 4 | 160,246 | 
| leaf | AccessAdapter<4> | 5 | 160,246 | 
| leaf | AccessAdapter<4> | 6 | 170,234 | 
| leaf | AccessAdapter<8> | 0 | 55,124 | 
| leaf | AccessAdapter<8> | 1 | 48,986 | 
| leaf | AccessAdapter<8> | 2 | 48,986 | 
| leaf | AccessAdapter<8> | 3 | 48,986 | 
| leaf | AccessAdapter<8> | 4 | 48,986 | 
| leaf | AccessAdapter<8> | 5 | 48,986 | 
| leaf | AccessAdapter<8> | 6 | 51,322 | 
| leaf | Arc<BabyBearParameters>, 1> | 0 | 26,385 | 
| leaf | Arc<BabyBearParameters>, 1> | 1 | 23,316 | 
| leaf | Arc<BabyBearParameters>, 1> | 2 | 23,316 | 
| leaf | Arc<BabyBearParameters>, 1> | 3 | 23,316 | 
| leaf | Arc<BabyBearParameters>, 1> | 4 | 23,316 | 
| leaf | Arc<BabyBearParameters>, 1> | 5 | 23,316 | 
| leaf | Arc<BabyBearParameters>, 1> | 6 | 24,378 | 
| leaf | Boundary | 0 | 381,824 | 
| leaf | Boundary | 1 | 344,270 | 
| leaf | Boundary | 2 | 344,270 | 
| leaf | Boundary | 3 | 344,270 | 
| leaf | Boundary | 4 | 344,270 | 
| leaf | Boundary | 5 | 344,270 | 
| leaf | Boundary | 6 | 357,369 | 
| leaf | FriReducedOpeningAir | 0 | 117,936 | 
| leaf | FriReducedOpeningAir | 1 | 75,600 | 
| leaf | FriReducedOpeningAir | 2 | 75,600 | 
| leaf | FriReducedOpeningAir | 3 | 75,600 | 
| leaf | FriReducedOpeningAir | 4 | 75,600 | 
| leaf | FriReducedOpeningAir | 5 | 75,600 | 
| leaf | FriReducedOpeningAir | 6 | 91,056 | 
| leaf | PhantomAir | 0 | 183,782 | 
| leaf | PhantomAir | 1 | 131,756 | 
| leaf | PhantomAir | 2 | 131,756 | 
| leaf | PhantomAir | 3 | 131,756 | 
| leaf | PhantomAir | 4 | 131,756 | 
| leaf | PhantomAir | 5 | 131,756 | 
| leaf | PhantomAir | 6 | 150,371 | 
| leaf | ProgramChip | 0 | 76,206 | 
| leaf | ProgramChip | 1 | 76,206 | 
| leaf | ProgramChip | 2 | 76,206 | 
| leaf | ProgramChip | 3 | 76,206 | 
| leaf | ProgramChip | 4 | 76,206 | 
| leaf | ProgramChip | 5 | 76,206 | 
| leaf | ProgramChip | 6 | 76,206 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 
| leaf | VmConnectorAir | 0 | 2 | 
| leaf | VmConnectorAir | 1 | 2 | 
| leaf | VmConnectorAir | 2 | 2 | 
| leaf | VmConnectorAir | 3 | 2 | 
| leaf | VmConnectorAir | 4 | 2 | 
| leaf | VmConnectorAir | 5 | 2 | 
| leaf | VmConnectorAir | 6 | 2 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 691,103 | 
| root | <JalNativeAdapterAir,JalCoreAir> | 0 | 93,874 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 48 | 
| root | <NativeAdapterAir<2, 1>,FieldArithmeticCoreAir> | 0 | 1,687,660 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 781,549 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 40,049 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 38,824 | 
| root | AccessAdapter<2> | 0 | 380,340 | 
| root | AccessAdapter<4> | 0 | 190,298 | 
| root | AccessAdapter<8> | 0 | 53,182 | 
| root | Arc<BabyBearParameters>, 1> | 0 | 25,393 | 
| root | Boundary | 0 | 385,962 | 
| root | FriReducedOpeningAir | 0 | 97,524 | 
| root | PhantomAir | 0 | 167,314 | 
| root | ProgramChip | 0 | 114,213 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 
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
| fib_e2e | Arc<BabyBearParameters>, 1> | 0 | 202 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 1 | 134 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 2 | 134 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 3 | 135 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 4 | 136 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 5 | 136 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 6 | 221 | 
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
| fib_e2e | Merkle | 0 | 226 | 
| fib_e2e | Merkle | 1 | 124 | 
| fib_e2e | Merkle | 2 | 124 | 
| fib_e2e | Merkle | 3 | 124 | 
| fib_e2e | Merkle | 4 | 124 | 
| fib_e2e | Merkle | 5 | 124 | 
| fib_e2e | Merkle | 6 | 230 | 
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
| internal.0 | AddE | 0 | FE4ADD | 24,970 | 
| internal.0 | AddE | 1 | FE4ADD | 24,970 | 
| internal.0 | AddE | 2 | FE4ADD | 24,970 | 
| internal.0 | AddE | 3 | FE4ADD | 12,485 | 
| internal.0 | AddEFFI | 0 | ADD | 2,816 | 
| internal.0 | AddEFFI | 1 | ADD | 2,816 | 
| internal.0 | AddEFFI | 2 | ADD | 2,816 | 
| internal.0 | AddEFFI | 3 | ADD | 1,408 | 
| internal.0 | AddEFI | 0 | ADD | 1,448 | 
| internal.0 | AddEFI | 1 | ADD | 1,448 | 
| internal.0 | AddEFI | 2 | ADD | 1,448 | 
| internal.0 | AddEFI | 3 | ADD | 724 | 
| internal.0 | AddEI | 0 | ADD | 60,320 | 
| internal.0 | AddEI | 1 | ADD | 60,320 | 
| internal.0 | AddEI | 2 | ADD | 60,320 | 
| internal.0 | AddEI | 3 | ADD | 30,160 | 
| internal.0 | AddF | 0 | ADD | 2,666 | 
| internal.0 | AddF | 1 | ADD | 2,666 | 
| internal.0 | AddF | 2 | ADD | 2,666 | 
| internal.0 | AddF | 3 | ADD | 1,333 | 
| internal.0 | AddFI | 0 | ADD | 153,177 | 
| internal.0 | AddFI | 1 | ADD | 152,757 | 
| internal.0 | AddFI | 2 | ADD | 152,505 | 
| internal.0 | AddFI | 3 | ADD | 76,685 | 
| internal.0 | AddV | 0 | ADD | 129,194 | 
| internal.0 | AddV | 1 | ADD | 129,190 | 
| internal.0 | AddV | 2 | ADD | 129,188 | 
| internal.0 | AddV | 3 | ADD | 64,603 | 
| internal.0 | AddVI | 0 | ADD | 590,378 | 
| internal.0 | AddVI | 1 | ADD | 589,916 | 
| internal.0 | AddVI | 2 | ADD | 589,622 | 
| internal.0 | AddVI | 3 | ADD | 295,346 | 
| internal.0 | Alloc | 0 | ADD | 224,346 | 
| internal.0 | Alloc | 0 | MUL | 66,200 | 
| internal.0 | Alloc | 1 | ADD | 224,346 | 
| internal.0 | Alloc | 1 | MUL | 66,200 | 
| internal.0 | Alloc | 2 | ADD | 224,346 | 
| internal.0 | Alloc | 2 | MUL | 66,200 | 
| internal.0 | Alloc | 3 | ADD | 112,196 | 
| internal.0 | Alloc | 3 | MUL | 33,107 | 
| internal.0 | AssertEqE | 0 | BNE | 472 | 
| internal.0 | AssertEqE | 1 | BNE | 472 | 
| internal.0 | AssertEqE | 2 | BNE | 472 | 
| internal.0 | AssertEqE | 3 | BNE | 236 | 
| internal.0 | AssertEqEI | 0 | BNE | 8 | 
| internal.0 | AssertEqEI | 1 | BNE | 8 | 
| internal.0 | AssertEqEI | 2 | BNE | 8 | 
| internal.0 | AssertEqEI | 3 | BNE | 4 | 
| internal.0 | AssertEqF | 0 | BNE | 20,921 | 
| internal.0 | AssertEqF | 1 | BNE | 20,921 | 
| internal.0 | AssertEqF | 2 | BNE | 20,921 | 
| internal.0 | AssertEqF | 3 | BNE | 10,448 | 
| internal.0 | AssertEqFI | 0 | BNE | 7 | 
| internal.0 | AssertEqFI | 1 | BNE | 7 | 
| internal.0 | AssertEqFI | 2 | BNE | 7 | 
| internal.0 | AssertEqFI | 3 | BNE | 3 | 
| internal.0 | AssertEqV | 0 | BNE | 2,214 | 
| internal.0 | AssertEqV | 1 | BNE | 2,214 | 
| internal.0 | AssertEqV | 2 | BNE | 2,214 | 
| internal.0 | AssertEqV | 3 | BNE | 1,107 | 
| internal.0 | AssertEqVI | 0 | BNE | 472 | 
| internal.0 | AssertEqVI | 1 | BNE | 472 | 
| internal.0 | AssertEqVI | 2 | BNE | 472 | 
| internal.0 | AssertEqVI | 3 | BNE | 236 | 
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
| internal.0 | CT-compute-reduced-opening | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 3 | PHANTOM | 504 | 
| internal.0 | CT-exp-reverse-bits-len | 0 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 1 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 2 | PHANTOM | 16,296 | 
| internal.0 | CT-exp-reverse-bits-len | 3 | PHANTOM | 8,148 | 
| internal.0 | CT-poseidon2-hash | 0 | PHANTOM | 5,040 | 
| internal.0 | CT-poseidon2-hash | 1 | PHANTOM | 5,040 | 
| internal.0 | CT-poseidon2-hash | 2 | PHANTOM | 5,040 | 
| internal.0 | CT-poseidon2-hash | 3 | PHANTOM | 2,520 | 
| internal.0 | CT-poseidon2-hash-ext | 0 | PHANTOM | 3,528 | 
| internal.0 | CT-poseidon2-hash-ext | 1 | PHANTOM | 3,528 | 
| internal.0 | CT-poseidon2-hash-ext | 2 | PHANTOM | 3,528 | 
| internal.0 | CT-poseidon2-hash-ext | 3 | PHANTOM | 1,764 | 
| internal.0 | CT-poseidon2-hash-setup | 0 | PHANTOM | 215,208 | 
| internal.0 | CT-poseidon2-hash-setup | 1 | PHANTOM | 215,208 | 
| internal.0 | CT-poseidon2-hash-setup | 2 | PHANTOM | 215,208 | 
| internal.0 | CT-poseidon2-hash-setup | 3 | PHANTOM | 107,604 | 
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
| internal.0 | CT-verify-batch-ext | 0 | PHANTOM | 3,528 | 
| internal.0 | CT-verify-batch-ext | 1 | PHANTOM | 3,528 | 
| internal.0 | CT-verify-batch-ext | 2 | PHANTOM | 3,528 | 
| internal.0 | CT-verify-batch-ext | 3 | PHANTOM | 1,764 | 
| internal.0 | CT-verify-batch-reduce-fast | 0 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast | 1 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast | 2 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast | 3 | PHANTOM | 4,284 | 
| internal.0 | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast-setup | 1 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast-setup | 2 | PHANTOM | 8,568 | 
| internal.0 | CT-verify-batch-reduce-fast-setup | 3 | PHANTOM | 4,284 | 
| internal.0 | CT-verify-query | 0 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 1 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 2 | PHANTOM | 168 | 
| internal.0 | CT-verify-query | 3 | PHANTOM | 84 | 
| internal.0 | CastFV | 0 | ADD | 16 | 
| internal.0 | CastFV | 1 | ADD | 16 | 
| internal.0 | CastFV | 2 | ADD | 16 | 
| internal.0 | CastFV | 3 | ADD | 8 | 
| internal.0 | DivE | 0 | BBE4DIV | 13,088 | 
| internal.0 | DivE | 1 | BBE4DIV | 13,088 | 
| internal.0 | DivE | 2 | BBE4DIV | 13,088 | 
| internal.0 | DivE | 3 | BBE4DIV | 6,544 | 
| internal.0 | DivEIN | 0 | ADD | 1,496 | 
| internal.0 | DivEIN | 0 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 1 | ADD | 1,496 | 
| internal.0 | DivEIN | 1 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 2 | ADD | 1,496 | 
| internal.0 | DivEIN | 2 | BBE4DIV | 374 | 
| internal.0 | DivEIN | 3 | ADD | 748 | 
| internal.0 | DivEIN | 3 | BBE4DIV | 187 | 
| internal.0 | DivFIN | 0 | DIV | 782 | 
| internal.0 | DivFIN | 1 | DIV | 782 | 
| internal.0 | DivFIN | 2 | DIV | 782 | 
| internal.0 | DivFIN | 3 | DIV | 391 | 
| internal.0 | For | 0 | ADD | 338,200 | 
| internal.0 | For | 0 | BNE | 338,200 | 
| internal.0 | For | 0 | JAL | 25,967 | 
| internal.0 | For | 1 | ADD | 337,746 | 
| internal.0 | For | 1 | BNE | 337,746 | 
| internal.0 | For | 1 | JAL | 25,967 | 
| internal.0 | For | 2 | ADD | 337,467 | 
| internal.0 | For | 2 | BNE | 337,467 | 
| internal.0 | For | 2 | JAL | 25,967 | 
| internal.0 | For | 3 | ADD | 169,188 | 
| internal.0 | For | 3 | BNE | 169,188 | 
| internal.0 | For | 3 | JAL | 12,984 | 
| internal.0 | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 11,256 | 
| internal.0 | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 5,628 | 
| internal.0 | HintBitsF | 0 | PHANTOM | 86 | 
| internal.0 | HintBitsF | 1 | PHANTOM | 86 | 
| internal.0 | HintBitsF | 2 | PHANTOM | 86 | 
| internal.0 | HintBitsF | 3 | PHANTOM | 43 | 
| internal.0 | HintInputVec | 0 | PHANTOM | 45,973 | 
| internal.0 | HintInputVec | 1 | PHANTOM | 45,973 | 
| internal.0 | HintInputVec | 2 | PHANTOM | 45,973 | 
| internal.0 | HintInputVec | 3 | PHANTOM | 22,991 | 
| internal.0 | IfEq | 0 | BNE | 153,840 | 
| internal.0 | IfEq | 1 | BNE | 154,260 | 
| internal.0 | IfEq | 2 | BNE | 154,512 | 
| internal.0 | IfEq | 3 | BNE | 76,836 | 
| internal.0 | IfEqI | 0 | BNE | 201,916 | 
| internal.0 | IfEqI | 0 | JAL | 91,864 | 
| internal.0 | IfEqI | 1 | BNE | 201,496 | 
| internal.0 | IfEqI | 1 | JAL | 90,674 | 
| internal.0 | IfEqI | 2 | BNE | 201,244 | 
| internal.0 | IfEqI | 2 | JAL | 90,643 | 
| internal.0 | IfEqI | 3 | BNE | 101,042 | 
| internal.0 | IfEqI | 3 | JAL | 41,567 | 
| internal.0 | IfNe | 0 | BEQ | 33,306 | 
| internal.0 | IfNe | 0 | JAL | 6 | 
| internal.0 | IfNe | 1 | BEQ | 33,306 | 
| internal.0 | IfNe | 1 | JAL | 6 | 
| internal.0 | IfNe | 2 | BEQ | 33,306 | 
| internal.0 | IfNe | 2 | JAL | 6 | 
| internal.0 | IfNe | 3 | BEQ | 16,653 | 
| internal.0 | IfNe | 3 | JAL | 3 | 
| internal.0 | IfNeI | 0 | BEQ | 1,952 | 
| internal.0 | IfNeI | 1 | BEQ | 1,952 | 
| internal.0 | IfNeI | 2 | BEQ | 1,952 | 
| internal.0 | IfNeI | 3 | BEQ | 976 | 
| internal.0 | ImmE | 0 | ADD | 6,048 | 
| internal.0 | ImmE | 1 | ADD | 6,048 | 
| internal.0 | ImmE | 2 | ADD | 6,048 | 
| internal.0 | ImmE | 3 | ADD | 3,024 | 
| internal.0 | ImmF | 0 | ADD | 73,782 | 
| internal.0 | ImmF | 1 | ADD | 73,782 | 
| internal.0 | ImmF | 2 | ADD | 73,782 | 
| internal.0 | ImmF | 3 | ADD | 36,975 | 
| internal.0 | ImmV | 0 | ADD | 34,977 | 
| internal.0 | ImmV | 1 | ADD | 34,977 | 
| internal.0 | ImmV | 2 | ADD | 34,977 | 
| internal.0 | ImmV | 3 | ADD | 17,547 | 
| internal.0 | LoadE | 0 | ADD | 25,116 | 
| internal.0 | LoadE | 0 | LOADW4 | 46,506 | 
| internal.0 | LoadE | 0 | MUL | 25,116 | 
| internal.0 | LoadE | 1 | ADD | 25,116 | 
| internal.0 | LoadE | 1 | LOADW4 | 46,506 | 
| internal.0 | LoadE | 1 | MUL | 25,116 | 
| internal.0 | LoadE | 2 | ADD | 25,116 | 
| internal.0 | LoadE | 2 | LOADW4 | 46,506 | 
| internal.0 | LoadE | 2 | MUL | 25,116 | 
| internal.0 | LoadE | 3 | ADD | 12,558 | 
| internal.0 | LoadE | 3 | LOADW4 | 23,253 | 
| internal.0 | LoadE | 3 | MUL | 12,558 | 
| internal.0 | LoadF | 0 | ADD | 26,488 | 
| internal.0 | LoadF | 0 | LOADW | 206,084 | 
| internal.0 | LoadF | 0 | MUL | 17,962 | 
| internal.0 | LoadF | 1 | ADD | 26,488 | 
| internal.0 | LoadF | 1 | LOADW | 206,084 | 
| internal.0 | LoadF | 1 | MUL | 17,962 | 
| internal.0 | LoadF | 2 | ADD | 26,488 | 
| internal.0 | LoadF | 2 | LOADW | 206,084 | 
| internal.0 | LoadF | 2 | MUL | 17,962 | 
| internal.0 | LoadF | 3 | ADD | 13,244 | 
| internal.0 | LoadF | 3 | LOADW | 103,046 | 
| internal.0 | LoadF | 3 | MUL | 8,981 | 
| internal.0 | LoadHeapPtr | 0 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 1 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 2 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 3 | ADD | 1 | 
| internal.0 | LoadV | 0 | ADD | 102,222 | 
| internal.0 | LoadV | 0 | LOADW | 562,606 | 
| internal.0 | LoadV | 0 | MUL | 85,116 | 
| internal.0 | LoadV | 1 | ADD | 102,222 | 
| internal.0 | LoadV | 1 | LOADW | 562,606 | 
| internal.0 | LoadV | 1 | MUL | 85,116 | 
| internal.0 | LoadV | 2 | ADD | 102,222 | 
| internal.0 | LoadV | 2 | LOADW | 562,606 | 
| internal.0 | LoadV | 2 | MUL | 85,116 | 
| internal.0 | LoadV | 3 | ADD | 51,111 | 
| internal.0 | LoadV | 3 | LOADW | 281,308 | 
| internal.0 | LoadV | 3 | MUL | 42,558 | 
| internal.0 | MulE | 0 | BBE4MUL | 28,619 | 
| internal.0 | MulE | 1 | BBE4MUL | 28,589 | 
| internal.0 | MulE | 2 | BBE4MUL | 28,564 | 
| internal.0 | MulE | 3 | BBE4MUL | 14,312 | 
| internal.0 | MulEF | 0 | MUL | 10,048 | 
| internal.0 | MulEF | 1 | MUL | 10,048 | 
| internal.0 | MulEF | 2 | MUL | 10,048 | 
| internal.0 | MulEF | 3 | MUL | 5,024 | 
| internal.0 | MulEFI | 0 | MUL | 904 | 
| internal.0 | MulEFI | 1 | MUL | 904 | 
| internal.0 | MulEFI | 2 | MUL | 904 | 
| internal.0 | MulEFI | 3 | MUL | 452 | 
| internal.0 | MulEI | 0 | ADD | 9,240 | 
| internal.0 | MulEI | 0 | BBE4MUL | 2,310 | 
| internal.0 | MulEI | 1 | ADD | 9,240 | 
| internal.0 | MulEI | 1 | BBE4MUL | 2,310 | 
| internal.0 | MulEI | 2 | ADD | 9,240 | 
| internal.0 | MulEI | 2 | BBE4MUL | 2,310 | 
| internal.0 | MulEI | 3 | ADD | 4,620 | 
| internal.0 | MulEI | 3 | BBE4MUL | 1,155 | 
| internal.0 | MulF | 0 | MUL | 299,530 | 
| internal.0 | MulF | 1 | MUL | 298,690 | 
| internal.0 | MulF | 2 | MUL | 298,186 | 
| internal.0 | MulF | 3 | MUL | 149,933 | 
| internal.0 | MulFI | 0 | MUL | 2,700 | 
| internal.0 | MulFI | 1 | MUL | 2,700 | 
| internal.0 | MulFI | 2 | MUL | 2,700 | 
| internal.0 | MulFI | 3 | MUL | 1,350 | 
| internal.0 | MulVI | 0 | MUL | 61,595 | 
| internal.0 | MulVI | 1 | MUL | 61,595 | 
| internal.0 | MulVI | 2 | MUL | 61,595 | 
| internal.0 | MulVI | 3 | MUL | 30,798 | 
| internal.0 | NegE | 0 | MUL | 456 | 
| internal.0 | NegE | 1 | MUL | 456 | 
| internal.0 | NegE | 2 | MUL | 456 | 
| internal.0 | NegE | 3 | MUL | 228 | 
| internal.0 | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 32,508 | 
| internal.0 | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 32,508 | 
| internal.0 | Poseidon2CompressBabyBear | 2 | COMP_POS2 | 32,508 | 
| internal.0 | Poseidon2CompressBabyBear | 3 | COMP_POS2 | 16,254 | 
| internal.0 | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 16,106 | 
| internal.0 | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 16,190 | 
| internal.0 | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 16,232 | 
| internal.0 | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 7,990 | 
| internal.0 | Publish | 0 | PUBLISH | 52 | 
| internal.0 | Publish | 1 | PUBLISH | 52 | 
| internal.0 | Publish | 2 | PUBLISH | 52 | 
| internal.0 | Publish | 3 | PUBLISH | 52 | 
| internal.0 | StoreE | 0 | ADD | 18,060 | 
| internal.0 | StoreE | 0 | MUL | 18,060 | 
| internal.0 | StoreE | 0 | STOREW4 | 32,414 | 
| internal.0 | StoreE | 1 | ADD | 18,060 | 
| internal.0 | StoreE | 1 | MUL | 18,060 | 
| internal.0 | StoreE | 1 | STOREW4 | 32,414 | 
| internal.0 | StoreE | 2 | ADD | 18,060 | 
| internal.0 | StoreE | 2 | MUL | 18,060 | 
| internal.0 | StoreE | 2 | STOREW4 | 32,414 | 
| internal.0 | StoreE | 3 | ADD | 9,030 | 
| internal.0 | StoreE | 3 | MUL | 9,030 | 
| internal.0 | StoreE | 3 | STOREW4 | 16,207 | 
| internal.0 | StoreF | 0 | ADD | 15,612 | 
| internal.0 | StoreF | 0 | MUL | 752 | 
| internal.0 | StoreF | 0 | STOREW | 177,862 | 
| internal.0 | StoreF | 1 | ADD | 15,612 | 
| internal.0 | StoreF | 1 | MUL | 752 | 
| internal.0 | StoreF | 1 | STOREW | 177,862 | 
| internal.0 | StoreF | 2 | ADD | 15,612 | 
| internal.0 | StoreF | 2 | MUL | 752 | 
| internal.0 | StoreF | 2 | STOREW | 177,862 | 
| internal.0 | StoreF | 3 | ADD | 7,806 | 
| internal.0 | StoreF | 3 | MUL | 376 | 
| internal.0 | StoreF | 3 | STOREW | 89,015 | 
| internal.0 | StoreHeapPtr | 0 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 1 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 2 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 3 | ADD | 1 | 
| internal.0 | StoreHintWord | 0 | HINT_STOREW | 431,672 | 
| internal.0 | StoreHintWord | 1 | HINT_STOREW | 431,672 | 
| internal.0 | StoreHintWord | 2 | HINT_STOREW | 431,672 | 
| internal.0 | StoreHintWord | 3 | HINT_STOREW | 215,845 | 
| internal.0 | StoreV | 0 | ADD | 47,384 | 
| internal.0 | StoreV | 0 | MUL | 38,346 | 
| internal.0 | StoreV | 0 | STOREW | 137,670 | 
| internal.0 | StoreV | 1 | ADD | 47,384 | 
| internal.0 | StoreV | 1 | MUL | 38,346 | 
| internal.0 | StoreV | 1 | STOREW | 137,670 | 
| internal.0 | StoreV | 2 | ADD | 47,384 | 
| internal.0 | StoreV | 2 | MUL | 38,346 | 
| internal.0 | StoreV | 2 | STOREW | 137,670 | 
| internal.0 | StoreV | 3 | ADD | 23,692 | 
| internal.0 | StoreV | 3 | MUL | 19,173 | 
| internal.0 | StoreV | 3 | STOREW | 68,891 | 
| internal.0 | SubE | 0 | FE4SUB | 6,622 | 
| internal.0 | SubE | 1 | FE4SUB | 6,622 | 
| internal.0 | SubE | 2 | FE4SUB | 6,622 | 
| internal.0 | SubE | 3 | FE4SUB | 3,311 | 
| internal.0 | SubEF | 0 | ADD | 33,996 | 
| internal.0 | SubEF | 0 | SUB | 11,332 | 
| internal.0 | SubEF | 1 | ADD | 33,996 | 
| internal.0 | SubEF | 1 | SUB | 11,332 | 
| internal.0 | SubEF | 2 | ADD | 33,996 | 
| internal.0 | SubEF | 2 | SUB | 11,332 | 
| internal.0 | SubEF | 3 | ADD | 16,998 | 
| internal.0 | SubEF | 3 | SUB | 5,666 | 
| internal.0 | SubEFI | 0 | ADD | 528 | 
| internal.0 | SubEFI | 1 | ADD | 528 | 
| internal.0 | SubEFI | 2 | ADD | 528 | 
| internal.0 | SubEFI | 3 | ADD | 264 | 
| internal.0 | SubEI | 0 | ADD | 2,992 | 
| internal.0 | SubEI | 1 | ADD | 2,992 | 
| internal.0 | SubEI | 2 | ADD | 2,992 | 
| internal.0 | SubEI | 3 | ADD | 1,496 | 
| internal.0 | SubF | 0 | SUB | 16 | 
| internal.0 | SubF | 1 | SUB | 16 | 
| internal.0 | SubF | 2 | SUB | 16 | 
| internal.0 | SubF | 3 | SUB | 8 | 
| internal.0 | SubFI | 0 | SUB | 2,666 | 
| internal.0 | SubFI | 1 | SUB | 2,666 | 
| internal.0 | SubFI | 2 | SUB | 2,666 | 
| internal.0 | SubFI | 3 | SUB | 1,333 | 
| internal.0 | SubV | 0 | SUB | 29,434 | 
| internal.0 | SubV | 1 | SUB | 29,434 | 
| internal.0 | SubV | 2 | SUB | 29,434 | 
| internal.0 | SubV | 3 | SUB | 14,717 | 
| internal.0 | SubVI | 0 | SUB | 2,088 | 
| internal.0 | SubVI | 1 | SUB | 2,088 | 
| internal.0 | SubVI | 2 | SUB | 2,088 | 
| internal.0 | SubVI | 3 | SUB | 1,044 | 
| internal.0 | SubVIN | 0 | SUB | 1,764 | 
| internal.0 | SubVIN | 1 | SUB | 1,764 | 
| internal.0 | SubVIN | 2 | SUB | 1,764 | 
| internal.0 | SubVIN | 3 | SUB | 882 | 
| internal.0 | UnsafeCastVF | 0 | ADD | 34 | 
| internal.0 | UnsafeCastVF | 1 | ADD | 34 | 
| internal.0 | UnsafeCastVF | 2 | ADD | 34 | 
| internal.0 | UnsafeCastVF | 3 | ADD | 17 | 
| internal.0 | ZipFor | 0 | ADD | 686,663 | 
| internal.0 | ZipFor | 0 | BNE | 580,023 | 
| internal.0 | ZipFor | 0 | JAL | 64,296 | 
| internal.0 | ZipFor | 1 | ADD | 687,503 | 
| internal.0 | ZipFor | 1 | BNE | 580,443 | 
| internal.0 | ZipFor | 1 | JAL | 64,296 | 
| internal.0 | ZipFor | 2 | ADD | 688,007 | 
| internal.0 | ZipFor | 2 | BNE | 580,695 | 
| internal.0 | ZipFor | 2 | JAL | 64,296 | 
| internal.0 | ZipFor | 3 | ADD | 343,173 | 
| internal.0 | ZipFor | 3 | BNE | 289,937 | 
| internal.0 | ZipFor | 3 | JAL | 32,153 | 
| internal.1 |  | 4 | ADD | 2 | 
| internal.1 |  | 4 | JAL | 1 | 
| internal.1 |  | 5 | ADD | 2 | 
| internal.1 |  | 5 | JAL | 1 | 
| internal.1 | AddE | 4 | FE4ADD | 25,176 | 
| internal.1 | AddE | 5 | FE4ADD | 25,089 | 
| internal.1 | AddEFFI | 4 | ADD | 2,976 | 
| internal.1 | AddEFFI | 5 | ADD | 2,960 | 
| internal.1 | AddEFI | 4 | ADD | 1,448 | 
| internal.1 | AddEFI | 5 | ADD | 1,448 | 
| internal.1 | AddEI | 4 | ADD | 61,472 | 
| internal.1 | AddEI | 5 | ADD | 60,960 | 
| internal.1 | AddF | 4 | ADD | 2,666 | 
| internal.1 | AddF | 5 | ADD | 2,666 | 
| internal.1 | AddFI | 4 | ADD | 160,801 | 
| internal.1 | AddFI | 5 | ADD | 157,601 | 
| internal.1 | AddV | 4 | ADD | 134,214 | 
| internal.1 | AddV | 5 | ADD | 131,709 | 
| internal.1 | AddVI | 4 | ADD | 613,924 | 
| internal.1 | AddVI | 5 | ADD | 602,813 | 
| internal.1 | Alloc | 4 | ADD | 235,442 | 
| internal.1 | Alloc | 4 | MUL | 69,310 | 
| internal.1 | Alloc | 5 | ADD | 229,894 | 
| internal.1 | Alloc | 5 | MUL | 67,755 | 
| internal.1 | AssertEqE | 4 | BNE | 472 | 
| internal.1 | AssertEqE | 5 | BNE | 472 | 
| internal.1 | AssertEqEI | 4 | BNE | 8 | 
| internal.1 | AssertEqEI | 5 | BNE | 8 | 
| internal.1 | AssertEqF | 4 | BNE | 21,609 | 
| internal.1 | AssertEqF | 5 | BNE | 21,273 | 
| internal.1 | AssertEqFI | 4 | BNE | 7 | 
| internal.1 | AssertEqFI | 5 | BNE | 7 | 
| internal.1 | AssertEqV | 4 | BNE | 2,298 | 
| internal.1 | AssertEqV | 5 | BNE | 2,256 | 
| internal.1 | AssertEqVI | 4 | BNE | 472 | 
| internal.1 | AssertEqVI | 5 | BNE | 472 | 
| internal.1 | AssertNonZero | 4 | BEQ | 1 | 
| internal.1 | AssertNonZero | 5 | BEQ | 1 | 
| internal.1 | CT-InitializePcsConst | 4 | PHANTOM | 2 | 
| internal.1 | CT-InitializePcsConst | 5 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 4 | PHANTOM | 2 | 
| internal.1 | CT-ReadProofsFromInput | 5 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 4 | PHANTOM | 2 | 
| internal.1 | CT-VerifyProofs | 5 | PHANTOM | 2 | 
| internal.1 | CT-compute-reduced-opening | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-exp-reverse-bits-len | 4 | PHANTOM | 16,296 | 
| internal.1 | CT-exp-reverse-bits-len | 5 | PHANTOM | 16,296 | 
| internal.1 | CT-poseidon2-hash | 4 | PHANTOM | 5,040 | 
| internal.1 | CT-poseidon2-hash | 5 | PHANTOM | 5,040 | 
| internal.1 | CT-poseidon2-hash-ext | 4 | PHANTOM | 3,696 | 
| internal.1 | CT-poseidon2-hash-ext | 5 | PHANTOM | 3,612 | 
| internal.1 | CT-poseidon2-hash-setup | 4 | PHANTOM | 215,208 | 
| internal.1 | CT-poseidon2-hash-setup | 5 | PHANTOM | 215,208 | 
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
| internal.1 | CT-verify-batch-ext | 4 | PHANTOM | 3,696 | 
| internal.1 | CT-verify-batch-ext | 5 | PHANTOM | 3,612 | 
| internal.1 | CT-verify-batch-reduce-fast | 4 | PHANTOM | 8,736 | 
| internal.1 | CT-verify-batch-reduce-fast | 5 | PHANTOM | 8,652 | 
| internal.1 | CT-verify-batch-reduce-fast-setup | 4 | PHANTOM | 8,736 | 
| internal.1 | CT-verify-batch-reduce-fast-setup | 5 | PHANTOM | 8,652 | 
| internal.1 | CT-verify-query | 4 | PHANTOM | 168 | 
| internal.1 | CT-verify-query | 5 | PHANTOM | 168 | 
| internal.1 | CastFV | 4 | ADD | 16 | 
| internal.1 | CastFV | 5 | ADD | 16 | 
| internal.1 | DivE | 4 | BBE4DIV | 13,172 | 
| internal.1 | DivE | 5 | BBE4DIV | 13,130 | 
| internal.1 | DivEIN | 4 | ADD | 1,496 | 
| internal.1 | DivEIN | 4 | BBE4DIV | 374 | 
| internal.1 | DivEIN | 5 | ADD | 1,496 | 
| internal.1 | DivEIN | 5 | BBE4DIV | 374 | 
| internal.1 | DivFIN | 4 | DIV | 782 | 
| internal.1 | DivFIN | 5 | DIV | 782 | 
| internal.1 | For | 4 | ADD | 349,089 | 
| internal.1 | For | 4 | BNE | 349,089 | 
| internal.1 | For | 4 | JAL | 26,219 | 
| internal.1 | For | 5 | ADD | 344,306 | 
| internal.1 | For | 5 | BNE | 344,306 | 
| internal.1 | For | 5 | JAL | 26,093 | 
| internal.1 | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 11,256 | 
| internal.1 | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 11,256 | 
| internal.1 | HintBitsF | 4 | PHANTOM | 86 | 
| internal.1 | HintBitsF | 5 | PHANTOM | 86 | 
| internal.1 | HintInputVec | 4 | PHANTOM | 48,411 | 
| internal.1 | HintInputVec | 5 | PHANTOM | 47,192 | 
| internal.1 | IfEq | 4 | BNE | 153,512 | 
| internal.1 | IfEq | 5 | BNE | 153,088 | 
| internal.1 | IfEqI | 4 | BNE | 212,548 | 
| internal.1 | IfEqI | 4 | JAL | 97,784 | 
| internal.1 | IfEqI | 5 | BNE | 207,836 | 
| internal.1 | IfEqI | 5 | JAL | 94,174 | 
| internal.1 | IfNe | 4 | BEQ | 35,498 | 
| internal.1 | IfNe | 4 | JAL | 6 | 
| internal.1 | IfNe | 5 | BEQ | 34,402 | 
| internal.1 | IfNe | 5 | JAL | 6 | 
| internal.1 | IfNeI | 4 | BEQ | 2,036 | 
| internal.1 | IfNeI | 5 | BEQ | 1,994 | 
| internal.1 | ImmE | 4 | ADD | 6,176 | 
| internal.1 | ImmE | 5 | ADD | 6,176 | 
| internal.1 | ImmF | 4 | ADD | 74,454 | 
| internal.1 | ImmF | 5 | ADD | 74,118 | 
| internal.1 | ImmV | 4 | ADD | 36,037 | 
| internal.1 | ImmV | 5 | ADD | 35,519 | 
| internal.1 | LoadE | 4 | ADD | 25,536 | 
| internal.1 | LoadE | 4 | LOADW4 | 47,178 | 
| internal.1 | LoadE | 4 | MUL | 25,536 | 
| internal.1 | LoadE | 5 | ADD | 25,326 | 
| internal.1 | LoadE | 5 | LOADW4 | 46,842 | 
| internal.1 | LoadE | 5 | MUL | 25,326 | 
| internal.1 | LoadF | 4 | ADD | 26,496 | 
| internal.1 | LoadF | 4 | LOADW | 208,304 | 
| internal.1 | LoadF | 4 | MUL | 17,962 | 
| internal.1 | LoadF | 5 | ADD | 26,492 | 
| internal.1 | LoadF | 5 | LOADW | 207,242 | 
| internal.1 | LoadF | 5 | MUL | 17,962 | 
| internal.1 | LoadHeapPtr | 4 | ADD | 2 | 
| internal.1 | LoadHeapPtr | 5 | ADD | 2 | 
| internal.1 | LoadV | 4 | ADD | 103,146 | 
| internal.1 | LoadV | 4 | LOADW | 579,916 | 
| internal.1 | LoadV | 4 | MUL | 85,620 | 
| internal.1 | LoadV | 5 | ADD | 102,684 | 
| internal.1 | LoadV | 5 | LOADW | 571,261 | 
| internal.1 | LoadV | 5 | MUL | 85,368 | 
| internal.1 | MulE | 4 | BBE4MUL | 29,734 | 
| internal.1 | MulE | 5 | BBE4MUL | 29,261 | 
| internal.1 | MulEF | 4 | MUL | 10,384 | 
| internal.1 | MulEF | 5 | MUL | 10,216 | 
| internal.1 | MulEFI | 4 | MUL | 904 | 
| internal.1 | MulEFI | 5 | MUL | 904 | 
| internal.1 | MulEI | 4 | ADD | 9,272 | 
| internal.1 | MulEI | 4 | BBE4MUL | 2,318 | 
| internal.1 | MulEI | 5 | ADD | 9,256 | 
| internal.1 | MulEI | 5 | BBE4MUL | 2,314 | 
| internal.1 | MulF | 4 | MUL | 314,650 | 
| internal.1 | MulF | 5 | MUL | 308,266 | 
| internal.1 | MulFI | 4 | MUL | 2,700 | 
| internal.1 | MulFI | 5 | MUL | 2,700 | 
| internal.1 | MulVI | 4 | MUL | 64,115 | 
| internal.1 | MulVI | 5 | MUL | 62,855 | 
| internal.1 | NegE | 4 | MUL | 456 | 
| internal.1 | NegE | 5 | MUL | 456 | 
| internal.1 | Poseidon2CompressBabyBear | 4 | COMP_POS2 | 34,692 | 
| internal.1 | Poseidon2CompressBabyBear | 5 | COMP_POS2 | 33,600 | 
| internal.1 | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 16,070 | 
| internal.1 | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 16,027 | 
| internal.1 | Publish | 4 | PUBLISH | 52 | 
| internal.1 | Publish | 5 | PUBLISH | 52 | 
| internal.1 | StoreE | 4 | ADD | 18,144 | 
| internal.1 | StoreE | 4 | MUL | 18,144 | 
| internal.1 | StoreE | 4 | STOREW4 | 32,920 | 
| internal.1 | StoreE | 5 | ADD | 18,102 | 
| internal.1 | StoreE | 5 | MUL | 18,102 | 
| internal.1 | StoreE | 5 | STOREW4 | 32,667 | 
| internal.1 | StoreF | 4 | ADD | 16,332 | 
| internal.1 | StoreF | 4 | MUL | 752 | 
| internal.1 | StoreF | 4 | STOREW | 179,254 | 
| internal.1 | StoreF | 5 | ADD | 15,988 | 
| internal.1 | StoreF | 5 | MUL | 752 | 
| internal.1 | StoreF | 5 | STOREW | 178,574 | 
| internal.1 | StoreHeapPtr | 4 | ADD | 2 | 
| internal.1 | StoreHeapPtr | 5 | ADD | 2 | 
| internal.1 | StoreHintWord | 4 | HINT_STOREW | 452,134 | 
| internal.1 | StoreHintWord | 5 | HINT_STOREW | 441,919 | 
| internal.1 | StoreV | 4 | ADD | 47,468 | 
| internal.1 | StoreV | 4 | MUL | 38,514 | 
| internal.1 | StoreV | 4 | STOREW | 142,546 | 
| internal.1 | StoreV | 5 | ADD | 47,426 | 
| internal.1 | StoreV | 5 | MUL | 38,430 | 
| internal.1 | StoreV | 5 | STOREW | 140,108 | 
| internal.1 | SubE | 4 | FE4SUB | 6,874 | 
| internal.1 | SubE | 5 | FE4SUB | 6,748 | 
| internal.1 | SubEF | 4 | ADD | 33,996 | 
| internal.1 | SubEF | 4 | SUB | 11,332 | 
| internal.1 | SubEF | 5 | ADD | 33,996 | 
| internal.1 | SubEF | 5 | SUB | 11,332 | 
| internal.1 | SubEFI | 4 | ADD | 528 | 
| internal.1 | SubEFI | 5 | ADD | 528 | 
| internal.1 | SubEI | 4 | ADD | 2,992 | 
| internal.1 | SubEI | 5 | ADD | 2,992 | 
| internal.1 | SubF | 4 | SUB | 16 | 
| internal.1 | SubF | 5 | SUB | 16 | 
| internal.1 | SubFI | 4 | SUB | 2,666 | 
| internal.1 | SubFI | 5 | SUB | 2,666 | 
| internal.1 | SubV | 4 | SUB | 29,770 | 
| internal.1 | SubV | 5 | SUB | 29,602 | 
| internal.1 | SubVI | 4 | SUB | 2,180 | 
| internal.1 | SubVI | 5 | SUB | 2,134 | 
| internal.1 | SubVIN | 4 | SUB | 1,848 | 
| internal.1 | SubVIN | 5 | SUB | 1,806 | 
| internal.1 | UnsafeCastVF | 4 | ADD | 34 | 
| internal.1 | UnsafeCastVF | 5 | ADD | 34 | 
| internal.1 | ZipFor | 4 | ADD | 713,011 | 
| internal.1 | ZipFor | 4 | BNE | 604,605 | 
| internal.1 | ZipFor | 4 | JAL | 66,986 | 
| internal.1 | ZipFor | 5 | ADD | 698,677 | 
| internal.1 | ZipFor | 5 | BNE | 591,742 | 
| internal.1 | ZipFor | 5 | JAL | 65,641 | 
| internal.2 |  | 6 | ADD | 2 | 
| internal.2 |  | 6 | JAL | 1 | 
| internal.2 | AddE | 6 | FE4ADD | 25,176 | 
| internal.2 | AddEFFI | 6 | ADD | 2,976 | 
| internal.2 | AddEFI | 6 | ADD | 1,448 | 
| internal.2 | AddEI | 6 | ADD | 61,472 | 
| internal.2 | AddF | 6 | ADD | 2,666 | 
| internal.2 | AddFI | 6 | ADD | 160,801 | 
| internal.2 | AddV | 6 | ADD | 134,214 | 
| internal.2 | AddVI | 6 | ADD | 613,924 | 
| internal.2 | Alloc | 6 | ADD | 235,442 | 
| internal.2 | Alloc | 6 | MUL | 69,310 | 
| internal.2 | AssertEqE | 6 | BNE | 472 | 
| internal.2 | AssertEqEI | 6 | BNE | 8 | 
| internal.2 | AssertEqF | 6 | BNE | 21,609 | 
| internal.2 | AssertEqFI | 6 | BNE | 7 | 
| internal.2 | AssertEqV | 6 | BNE | 2,298 | 
| internal.2 | AssertEqVI | 6 | BNE | 472 | 
| internal.2 | AssertNonZero | 6 | BEQ | 1 | 
| internal.2 | CT-InitializePcsConst | 6 | PHANTOM | 2 | 
| internal.2 | CT-ReadProofsFromInput | 6 | PHANTOM | 2 | 
| internal.2 | CT-VerifyProofs | 6 | PHANTOM | 2 | 
| internal.2 | CT-compute-reduced-opening | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-exp-reverse-bits-len | 6 | PHANTOM | 16,296 | 
| internal.2 | CT-poseidon2-hash | 6 | PHANTOM | 5,040 | 
| internal.2 | CT-poseidon2-hash-ext | 6 | PHANTOM | 3,696 | 
| internal.2 | CT-poseidon2-hash-setup | 6 | PHANTOM | 215,208 | 
| internal.2 | CT-single-reduced-opening-eval | 6 | PHANTOM | 22,512 | 
| internal.2 | CT-stage-c-build-rounds | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verifier-verify | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verify-pcs | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-e-verify-constraints | 6 | PHANTOM | 4 | 
| internal.2 | CT-verify-batch | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-verify-batch-ext | 6 | PHANTOM | 3,696 | 
| internal.2 | CT-verify-batch-reduce-fast | 6 | PHANTOM | 8,736 | 
| internal.2 | CT-verify-batch-reduce-fast-setup | 6 | PHANTOM | 8,736 | 
| internal.2 | CT-verify-query | 6 | PHANTOM | 168 | 
| internal.2 | CastFV | 6 | ADD | 16 | 
| internal.2 | DivE | 6 | BBE4DIV | 13,172 | 
| internal.2 | DivEIN | 6 | ADD | 1,496 | 
| internal.2 | DivEIN | 6 | BBE4DIV | 374 | 
| internal.2 | DivFIN | 6 | DIV | 782 | 
| internal.2 | For | 6 | ADD | 349,089 | 
| internal.2 | For | 6 | BNE | 349,089 | 
| internal.2 | For | 6 | JAL | 26,219 | 
| internal.2 | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 11,256 | 
| internal.2 | HintBitsF | 6 | PHANTOM | 86 | 
| internal.2 | HintInputVec | 6 | PHANTOM | 48,411 | 
| internal.2 | IfEq | 6 | BNE | 153,512 | 
| internal.2 | IfEqI | 6 | BNE | 212,548 | 
| internal.2 | IfEqI | 6 | JAL | 97,298 | 
| internal.2 | IfNe | 6 | BEQ | 35,498 | 
| internal.2 | IfNe | 6 | JAL | 6 | 
| internal.2 | IfNeI | 6 | BEQ | 2,036 | 
| internal.2 | ImmE | 6 | ADD | 6,176 | 
| internal.2 | ImmF | 6 | ADD | 74,454 | 
| internal.2 | ImmV | 6 | ADD | 36,037 | 
| internal.2 | LoadE | 6 | ADD | 25,536 | 
| internal.2 | LoadE | 6 | LOADW4 | 47,178 | 
| internal.2 | LoadE | 6 | MUL | 25,536 | 
| internal.2 | LoadF | 6 | ADD | 26,496 | 
| internal.2 | LoadF | 6 | LOADW | 208,304 | 
| internal.2 | LoadF | 6 | MUL | 17,962 | 
| internal.2 | LoadHeapPtr | 6 | ADD | 2 | 
| internal.2 | LoadV | 6 | ADD | 103,146 | 
| internal.2 | LoadV | 6 | LOADW | 579,916 | 
| internal.2 | LoadV | 6 | MUL | 85,620 | 
| internal.2 | MulE | 6 | BBE4MUL | 29,734 | 
| internal.2 | MulEF | 6 | MUL | 10,384 | 
| internal.2 | MulEFI | 6 | MUL | 904 | 
| internal.2 | MulEI | 6 | ADD | 9,272 | 
| internal.2 | MulEI | 6 | BBE4MUL | 2,318 | 
| internal.2 | MulF | 6 | MUL | 314,650 | 
| internal.2 | MulFI | 6 | MUL | 2,700 | 
| internal.2 | MulVI | 6 | MUL | 64,115 | 
| internal.2 | NegE | 6 | MUL | 456 | 
| internal.2 | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 34,692 | 
| internal.2 | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 16,070 | 
| internal.2 | Publish | 6 | PUBLISH | 52 | 
| internal.2 | StoreE | 6 | ADD | 18,144 | 
| internal.2 | StoreE | 6 | MUL | 18,144 | 
| internal.2 | StoreE | 6 | STOREW4 | 32,920 | 
| internal.2 | StoreF | 6 | ADD | 16,332 | 
| internal.2 | StoreF | 6 | MUL | 752 | 
| internal.2 | StoreF | 6 | STOREW | 179,254 | 
| internal.2 | StoreHeapPtr | 6 | ADD | 2 | 
| internal.2 | StoreHintWord | 6 | HINT_STOREW | 452,134 | 
| internal.2 | StoreV | 6 | ADD | 47,468 | 
| internal.2 | StoreV | 6 | MUL | 38,514 | 
| internal.2 | StoreV | 6 | STOREW | 142,546 | 
| internal.2 | SubE | 6 | FE4SUB | 6,874 | 
| internal.2 | SubEF | 6 | ADD | 33,996 | 
| internal.2 | SubEF | 6 | SUB | 11,332 | 
| internal.2 | SubEFI | 6 | ADD | 528 | 
| internal.2 | SubEI | 6 | ADD | 2,992 | 
| internal.2 | SubF | 6 | SUB | 16 | 
| internal.2 | SubFI | 6 | SUB | 2,666 | 
| internal.2 | SubV | 6 | SUB | 29,770 | 
| internal.2 | SubVI | 6 | SUB | 2,180 | 
| internal.2 | SubVIN | 6 | SUB | 1,848 | 
| internal.2 | UnsafeCastVF | 6 | ADD | 34 | 
| internal.2 | ZipFor | 6 | ADD | 713,011 | 
| internal.2 | ZipFor | 6 | BNE | 604,605 | 
| internal.2 | ZipFor | 6 | JAL | 66,986 | 
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
| leaf | AddE | 0 | FE4ADD | 12,433 | 
| leaf | AddE | 1 | FE4ADD | 8,885 | 
| leaf | AddE | 2 | FE4ADD | 8,885 | 
| leaf | AddE | 3 | FE4ADD | 8,885 | 
| leaf | AddE | 4 | FE4ADD | 8,885 | 
| leaf | AddE | 5 | FE4ADD | 8,885 | 
| leaf | AddE | 6 | FE4ADD | 10,063 | 
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
| leaf | AddEI | 0 | ADD | 27,652 | 
| leaf | AddEI | 1 | ADD | 22,048 | 
| leaf | AddEI | 2 | ADD | 22,048 | 
| leaf | AddEI | 3 | ADD | 22,048 | 
| leaf | AddEI | 4 | ADD | 22,048 | 
| leaf | AddEI | 5 | ADD | 22,048 | 
| leaf | AddEI | 6 | ADD | 23,948 | 
| leaf | AddF | 0 | ADD | 1,333 | 
| leaf | AddF | 1 | ADD | 1,333 | 
| leaf | AddF | 2 | ADD | 1,333 | 
| leaf | AddF | 3 | ADD | 1,333 | 
| leaf | AddF | 4 | ADD | 1,333 | 
| leaf | AddF | 5 | ADD | 1,333 | 
| leaf | AddF | 6 | ADD | 1,333 | 
| leaf | AddFI | 0 | ADD | 43,092 | 
| leaf | AddFI | 1 | ADD | 35,826 | 
| leaf | AddFI | 2 | ADD | 35,826 | 
| leaf | AddFI | 3 | ADD | 35,826 | 
| leaf | AddFI | 4 | ADD | 35,826 | 
| leaf | AddFI | 5 | ADD | 35,826 | 
| leaf | AddFI | 6 | ADD | 37,864 | 
| leaf | AddV | 0 | ADD | 63,840 | 
| leaf | AddV | 1 | ADD | 57,298 | 
| leaf | AddV | 2 | ADD | 57,298 | 
| leaf | AddV | 3 | ADD | 57,298 | 
| leaf | AddV | 4 | ADD | 57,298 | 
| leaf | AddV | 5 | ADD | 57,298 | 
| leaf | AddV | 6 | ADD | 59,487 | 
| leaf | AddVI | 0 | ADD | 273,294 | 
| leaf | AddVI | 1 | ADD | 232,884 | 
| leaf | AddVI | 2 | ADD | 232,884 | 
| leaf | AddVI | 3 | ADD | 232,884 | 
| leaf | AddVI | 4 | ADD | 232,884 | 
| leaf | AddVI | 5 | ADD | 232,884 | 
| leaf | AddVI | 6 | ADD | 246,606 | 
| leaf | Alloc | 0 | ADD | 112,732 | 
| leaf | Alloc | 0 | MUL | 33,587 | 
| leaf | Alloc | 1 | ADD | 107,560 | 
| leaf | Alloc | 1 | MUL | 32,123 | 
| leaf | Alloc | 2 | ADD | 107,560 | 
| leaf | Alloc | 2 | MUL | 32,123 | 
| leaf | Alloc | 3 | ADD | 107,560 | 
| leaf | Alloc | 3 | MUL | 32,123 | 
| leaf | Alloc | 4 | ADD | 107,560 | 
| leaf | Alloc | 4 | MUL | 32,123 | 
| leaf | Alloc | 5 | ADD | 107,560 | 
| leaf | Alloc | 5 | MUL | 32,123 | 
| leaf | Alloc | 6 | ADD | 110,856 | 
| leaf | Alloc | 6 | MUL | 32,964 | 
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
| leaf | AssertEqF | 0 | BNE | 10,784 | 
| leaf | AssertEqF | 1 | BNE | 10,784 | 
| leaf | AssertEqF | 2 | BNE | 10,784 | 
| leaf | AssertEqF | 3 | BNE | 10,784 | 
| leaf | AssertEqF | 4 | BNE | 10,784 | 
| leaf | AssertEqF | 5 | BNE | 10,784 | 
| leaf | AssertEqF | 6 | BNE | 10,792 | 
| leaf | AssertEqV | 0 | BNE | 1,073 | 
| leaf | AssertEqV | 1 | BNE | 1,007 | 
| leaf | AssertEqV | 2 | BNE | 1,007 | 
| leaf | AssertEqV | 3 | BNE | 1,007 | 
| leaf | AssertEqV | 4 | BNE | 1,007 | 
| leaf | AssertEqV | 5 | BNE | 1,007 | 
| leaf | AssertEqV | 6 | BNE | 1,029 | 
| leaf | AssertEqVI | 0 | BNE | 240 | 
| leaf | AssertEqVI | 1 | BNE | 174 | 
| leaf | AssertEqVI | 2 | BNE | 174 | 
| leaf | AssertEqVI | 3 | BNE | 174 | 
| leaf | AssertEqVI | 4 | BNE | 174 | 
| leaf | AssertEqVI | 5 | BNE | 174 | 
| leaf | AssertEqVI | 6 | BNE | 197 | 
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
| leaf | CT-poseidon2-hash | 0 | PHANTOM | 3,696 | 
| leaf | CT-poseidon2-hash | 1 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash | 2 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash | 3 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash | 4 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash | 5 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash | 6 | PHANTOM | 3,192 | 
| leaf | CT-poseidon2-hash-ext | 0 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 1 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 2 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 3 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 4 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 5 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-ext | 6 | PHANTOM | 1,680 | 
| leaf | CT-poseidon2-hash-setup | 0 | PHANTOM | 124,152 | 
| leaf | CT-poseidon2-hash-setup | 1 | PHANTOM | 79,800 | 
| leaf | CT-poseidon2-hash-setup | 2 | PHANTOM | 79,800 | 
| leaf | CT-poseidon2-hash-setup | 3 | PHANTOM | 79,800 | 
| leaf | CT-poseidon2-hash-setup | 4 | PHANTOM | 79,800 | 
| leaf | CT-poseidon2-hash-setup | 5 | PHANTOM | 79,800 | 
| leaf | CT-poseidon2-hash-setup | 6 | PHANTOM | 95,928 | 
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
| leaf | CT-verify-batch-reduce-fast | 0 | PHANTOM | 5,376 | 
| leaf | CT-verify-batch-reduce-fast | 1 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast | 2 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast | 3 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast | 4 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast | 5 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast | 6 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 5,376 | 
| leaf | CT-verify-batch-reduce-fast-setup | 1 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 2 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 3 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 4 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 5 | PHANTOM | 4,872 | 
| leaf | CT-verify-batch-reduce-fast-setup | 6 | PHANTOM | 4,872 | 
| leaf | CT-verify-query | 0 | PHANTOM | 84 | 
| leaf | CT-verify-query | 1 | PHANTOM | 84 | 
| leaf | CT-verify-query | 2 | PHANTOM | 84 | 
| leaf | CT-verify-query | 3 | PHANTOM | 84 | 
| leaf | CT-verify-query | 4 | PHANTOM | 84 | 
| leaf | CT-verify-query | 5 | PHANTOM | 84 | 
| leaf | CT-verify-query | 6 | PHANTOM | 84 | 
| leaf | CastFV | 0 | ADD | 1 | 
| leaf | CastFV | 1 | ADD | 1 | 
| leaf | CastFV | 2 | ADD | 1 | 
| leaf | CastFV | 3 | ADD | 1 | 
| leaf | CastFV | 4 | ADD | 1 | 
| leaf | CastFV | 5 | ADD | 1 | 
| leaf | CastFV | 6 | ADD | 1 | 
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
| leaf | DivFIN | 0 | DIV | 128 | 
| leaf | DivFIN | 1 | DIV | 86 | 
| leaf | DivFIN | 2 | DIV | 86 | 
| leaf | DivFIN | 3 | DIV | 86 | 
| leaf | DivFIN | 4 | DIV | 86 | 
| leaf | DivFIN | 5 | DIV | 86 | 
| leaf | DivFIN | 6 | DIV | 100 | 
| leaf | For | 0 | ADD | 136,206 | 
| leaf | For | 0 | BNE | 136,206 | 
| leaf | For | 0 | JAL | 12,031 | 
| leaf | For | 1 | ADD | 103,127 | 
| leaf | For | 1 | BNE | 103,127 | 
| leaf | For | 1 | JAL | 9,673 | 
| leaf | For | 2 | ADD | 103,127 | 
| leaf | For | 2 | BNE | 103,127 | 
| leaf | For | 2 | JAL | 9,673 | 
| leaf | For | 3 | ADD | 103,127 | 
| leaf | For | 3 | BNE | 103,127 | 
| leaf | For | 3 | JAL | 9,673 | 
| leaf | For | 4 | ADD | 103,127 | 
| leaf | For | 4 | BNE | 103,127 | 
| leaf | For | 4 | JAL | 9,673 | 
| leaf | For | 5 | ADD | 103,127 | 
| leaf | For | 5 | BNE | 103,127 | 
| leaf | For | 5 | JAL | 9,673 | 
| leaf | For | 6 | ADD | 114,387 | 
| leaf | For | 6 | BNE | 114,387 | 
| leaf | For | 6 | JAL | 10,376 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,334 | 
| leaf | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 3,822 | 
| leaf | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 4,326 | 
| leaf | HintBitsF | 0 | PHANTOM | 43 | 
| leaf | HintBitsF | 1 | PHANTOM | 43 | 
| leaf | HintBitsF | 2 | PHANTOM | 43 | 
| leaf | HintBitsF | 3 | PHANTOM | 43 | 
| leaf | HintBitsF | 4 | PHANTOM | 43 | 
| leaf | HintBitsF | 5 | PHANTOM | 43 | 
| leaf | HintBitsF | 6 | PHANTOM | 43 | 
| leaf | HintInputVec | 0 | PHANTOM | 22,779 | 
| leaf | HintInputVec | 1 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 2 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 3 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 4 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 5 | PHANTOM | 21,657 | 
| leaf | HintInputVec | 6 | PHANTOM | 22,464 | 
| leaf | IfEq | 0 | BNE | 89,896 | 
| leaf | IfEq | 1 | BNE | 56,589 | 
| leaf | IfEq | 2 | BNE | 56,589 | 
| leaf | IfEq | 3 | BNE | 56,589 | 
| leaf | IfEq | 4 | BNE | 56,589 | 
| leaf | IfEq | 5 | BNE | 56,589 | 
| leaf | IfEq | 6 | BNE | 67,509 | 
| leaf | IfEqI | 0 | BNE | 67,562 | 
| leaf | IfEqI | 0 | JAL | 28,533 | 
| leaf | IfEqI | 1 | BNE | 60,290 | 
| leaf | IfEqI | 1 | JAL | 24,526 | 
| leaf | IfEqI | 2 | BNE | 60,290 | 
| leaf | IfEqI | 2 | JAL | 26,594 | 
| leaf | IfEqI | 3 | BNE | 60,290 | 
| leaf | IfEqI | 3 | JAL | 24,798 | 
| leaf | IfEqI | 4 | BNE | 60,290 | 
| leaf | IfEqI | 4 | JAL | 25,403 | 
| leaf | IfEqI | 5 | BNE | 60,290 | 
| leaf | IfEqI | 5 | JAL | 24,947 | 
| leaf | IfEqI | 6 | BNE | 62,322 | 
| leaf | IfEqI | 6 | JAL | 25,954 | 
| leaf | IfNe | 0 | BEQ | 17,615 | 
| leaf | IfNe | 0 | JAL | 1 | 
| leaf | IfNe | 1 | BEQ | 17,363 | 
| leaf | IfNe | 1 | JAL | 2 | 
| leaf | IfNe | 2 | BEQ | 17,363 | 
| leaf | IfNe | 2 | JAL | 2 | 
| leaf | IfNe | 3 | BEQ | 17,363 | 
| leaf | IfNe | 3 | JAL | 2 | 
| leaf | IfNe | 4 | BEQ | 17,363 | 
| leaf | IfNe | 4 | JAL | 2 | 
| leaf | IfNe | 5 | BEQ | 17,363 | 
| leaf | IfNe | 5 | JAL | 2 | 
| leaf | IfNe | 6 | BEQ | 17,363 | 
| leaf | IfNe | 6 | JAL | 2 | 
| leaf | IfNeI | 0 | BEQ | 941 | 
| leaf | IfNeI | 1 | BEQ | 911 | 
| leaf | IfNeI | 2 | BEQ | 911 | 
| leaf | IfNeI | 3 | BEQ | 911 | 
| leaf | IfNeI | 4 | BEQ | 911 | 
| leaf | IfNeI | 5 | BEQ | 911 | 
| leaf | IfNeI | 6 | BEQ | 921 | 
| leaf | ImmE | 0 | ADD | 3,208 | 
| leaf | ImmE | 1 | ADD | 1,944 | 
| leaf | ImmE | 2 | ADD | 1,944 | 
| leaf | ImmE | 3 | ADD | 1,944 | 
| leaf | ImmE | 4 | ADD | 1,944 | 
| leaf | ImmE | 5 | ADD | 1,944 | 
| leaf | ImmE | 6 | ADD | 2,340 | 
| leaf | ImmF | 0 | ADD | 44,725 | 
| leaf | ImmF | 1 | ADD | 38,665 | 
| leaf | ImmF | 2 | ADD | 38,665 | 
| leaf | ImmF | 3 | ADD | 38,665 | 
| leaf | ImmF | 4 | ADD | 38,665 | 
| leaf | ImmF | 5 | ADD | 38,665 | 
| leaf | ImmF | 6 | ADD | 39,341 | 
| leaf | ImmV | 0 | ADD | 19,037 | 
| leaf | ImmV | 1 | ADD | 18,107 | 
| leaf | ImmV | 2 | ADD | 18,107 | 
| leaf | ImmV | 3 | ADD | 18,107 | 
| leaf | ImmV | 4 | ADD | 18,107 | 
| leaf | ImmV | 5 | ADD | 18,107 | 
| leaf | ImmV | 6 | ADD | 18,168 | 
| leaf | LoadE | 0 | ADD | 11,088 | 
| leaf | LoadE | 0 | LOADW4 | 21,858 | 
| leaf | LoadE | 0 | MUL | 11,088 | 
| leaf | LoadE | 1 | ADD | 9,072 | 
| leaf | LoadE | 1 | LOADW4 | 17,304 | 
| leaf | LoadE | 1 | MUL | 9,072 | 
| leaf | LoadE | 2 | ADD | 9,072 | 
| leaf | LoadE | 2 | LOADW4 | 17,304 | 
| leaf | LoadE | 2 | MUL | 9,072 | 
| leaf | LoadE | 3 | ADD | 9,072 | 
| leaf | LoadE | 3 | LOADW4 | 17,304 | 
| leaf | LoadE | 3 | MUL | 9,072 | 
| leaf | LoadE | 4 | ADD | 9,072 | 
| leaf | LoadE | 4 | LOADW4 | 17,304 | 
| leaf | LoadE | 4 | MUL | 9,072 | 
| leaf | LoadE | 5 | ADD | 9,072 | 
| leaf | LoadE | 5 | LOADW4 | 17,304 | 
| leaf | LoadE | 5 | MUL | 9,072 | 
| leaf | LoadE | 6 | ADD | 9,744 | 
| leaf | LoadE | 6 | LOADW4 | 18,854 | 
| leaf | LoadE | 6 | MUL | 9,744 | 
| leaf | LoadF | 0 | ADD | 11,512 | 
| leaf | LoadF | 0 | LOADW | 108,574 | 
| leaf | LoadF | 0 | MUL | 7,883 | 
| leaf | LoadF | 1 | ADD | 8,392 | 
| leaf | LoadF | 1 | LOADW | 81,238 | 
| leaf | LoadF | 1 | MUL | 5,771 | 
| leaf | LoadF | 2 | ADD | 8,392 | 
| leaf | LoadF | 2 | LOADW | 81,238 | 
| leaf | LoadF | 2 | MUL | 5,771 | 
| leaf | LoadF | 3 | ADD | 8,392 | 
| leaf | LoadF | 3 | LOADW | 81,238 | 
| leaf | LoadF | 3 | MUL | 5,771 | 
| leaf | LoadF | 4 | ADD | 8,392 | 
| leaf | LoadF | 4 | LOADW | 81,238 | 
| leaf | LoadF | 4 | MUL | 5,771 | 
| leaf | LoadF | 5 | ADD | 8,392 | 
| leaf | LoadF | 5 | LOADW | 81,238 | 
| leaf | LoadF | 5 | MUL | 5,771 | 
| leaf | LoadF | 6 | ADD | 9,432 | 
| leaf | LoadF | 6 | LOADW | 92,302 | 
| leaf | LoadF | 6 | MUL | 6,475 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadHeapPtr | 1 | ADD | 1 | 
| leaf | LoadHeapPtr | 2 | ADD | 1 | 
| leaf | LoadHeapPtr | 3 | ADD | 1 | 
| leaf | LoadHeapPtr | 4 | ADD | 1 | 
| leaf | LoadHeapPtr | 5 | ADD | 1 | 
| leaf | LoadHeapPtr | 6 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 45,949 | 
| leaf | LoadV | 0 | LOADW | 241,756 | 
| leaf | LoadV | 0 | MUL | 38,206 | 
| leaf | LoadV | 1 | ADD | 35,935 | 
| leaf | LoadV | 1 | LOADW | 197,770 | 
| leaf | LoadV | 1 | MUL | 29,086 | 
| leaf | LoadV | 2 | ADD | 35,935 | 
| leaf | LoadV | 2 | LOADW | 197,770 | 
| leaf | LoadV | 2 | MUL | 29,086 | 
| leaf | LoadV | 3 | ADD | 35,935 | 
| leaf | LoadV | 3 | LOADW | 197,770 | 
| leaf | LoadV | 3 | MUL | 29,086 | 
| leaf | LoadV | 4 | ADD | 35,935 | 
| leaf | LoadV | 4 | LOADW | 197,770 | 
| leaf | LoadV | 4 | MUL | 29,086 | 
| leaf | LoadV | 5 | ADD | 35,935 | 
| leaf | LoadV | 5 | LOADW | 197,770 | 
| leaf | LoadV | 5 | MUL | 29,086 | 
| leaf | LoadV | 6 | ADD | 39,315 | 
| leaf | LoadV | 6 | LOADW | 211,620 | 
| leaf | LoadV | 6 | MUL | 32,126 | 
| leaf | MulE | 0 | BBE4MUL | 8,530 | 
| leaf | MulE | 1 | BBE4MUL | 6,713 | 
| leaf | MulE | 2 | BBE4MUL | 6,713 | 
| leaf | MulE | 3 | BBE4MUL | 6,713 | 
| leaf | MulE | 4 | BBE4MUL | 6,713 | 
| leaf | MulE | 5 | BBE4MUL | 6,713 | 
| leaf | MulE | 6 | BBE4MUL | 7,251 | 
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
| leaf | MulF | 0 | MUL | 83,331 | 
| leaf | MulF | 1 | MUL | 69,831 | 
| leaf | MulF | 2 | MUL | 69,831 | 
| leaf | MulF | 3 | MUL | 69,831 | 
| leaf | MulF | 4 | MUL | 69,831 | 
| leaf | MulF | 5 | MUL | 69,831 | 
| leaf | MulF | 6 | MUL | 73,547 | 
| leaf | MulFI | 0 | MUL | 1,353 | 
| leaf | MulFI | 1 | MUL | 1,347 | 
| leaf | MulFI | 2 | MUL | 1,347 | 
| leaf | MulFI | 3 | MUL | 1,347 | 
| leaf | MulFI | 4 | MUL | 1,347 | 
| leaf | MulFI | 5 | MUL | 1,347 | 
| leaf | MulFI | 6 | MUL | 1,349 | 
| leaf | MulVI | 0 | MUL | 31,265 | 
| leaf | MulVI | 1 | MUL | 28,685 | 
| leaf | MulVI | 2 | MUL | 28,685 | 
| leaf | MulVI | 3 | MUL | 28,685 | 
| leaf | MulVI | 4 | MUL | 28,685 | 
| leaf | MulVI | 5 | MUL | 28,685 | 
| leaf | MulVI | 6 | MUL | 29,377 | 
| leaf | NegE | 0 | MUL | 172 | 
| leaf | NegE | 1 | MUL | 96 | 
| leaf | NegE | 2 | MUL | 96 | 
| leaf | NegE | 3 | MUL | 96 | 
| leaf | NegE | 4 | MUL | 96 | 
| leaf | NegE | 5 | MUL | 96 | 
| leaf | NegE | 6 | MUL | 124 | 
| leaf | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 17,136 | 
| leaf | Poseidon2CompressBabyBear | 1 | COMP_POS2 | 16,884 | 
| leaf | Poseidon2CompressBabyBear | 2 | COMP_POS2 | 16,884 | 
| leaf | Poseidon2CompressBabyBear | 3 | COMP_POS2 | 16,884 | 
| leaf | Poseidon2CompressBabyBear | 4 | COMP_POS2 | 16,884 | 
| leaf | Poseidon2CompressBabyBear | 5 | COMP_POS2 | 16,884 | 
| leaf | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 16,937 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 9,249 | 
| leaf | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 6,432 | 
| leaf | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 6,432 | 
| leaf | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 6,432 | 
| leaf | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 6,432 | 
| leaf | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 6,432 | 
| leaf | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 7,441 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | Publish | 1 | PUBLISH | 36 | 
| leaf | Publish | 2 | PUBLISH | 36 | 
| leaf | Publish | 3 | PUBLISH | 36 | 
| leaf | Publish | 4 | PUBLISH | 36 | 
| leaf | Publish | 5 | PUBLISH | 36 | 
| leaf | Publish | 6 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 7,728 | 
| leaf | StoreE | 0 | MUL | 7,728 | 
| leaf | StoreE | 0 | STOREW4 | 14,703 | 
| leaf | StoreE | 1 | ADD | 5,712 | 
| leaf | StoreE | 1 | MUL | 5,712 | 
| leaf | StoreE | 1 | STOREW4 | 12,669 | 
| leaf | StoreE | 2 | ADD | 5,712 | 
| leaf | StoreE | 2 | MUL | 5,712 | 
| leaf | StoreE | 2 | STOREW4 | 12,669 | 
| leaf | StoreE | 3 | ADD | 5,712 | 
| leaf | StoreE | 3 | MUL | 5,712 | 
| leaf | StoreE | 3 | STOREW4 | 12,669 | 
| leaf | StoreE | 4 | ADD | 5,712 | 
| leaf | StoreE | 4 | MUL | 5,712 | 
| leaf | StoreE | 4 | STOREW4 | 12,669 | 
| leaf | StoreE | 5 | ADD | 5,712 | 
| leaf | StoreE | 5 | MUL | 5,712 | 
| leaf | StoreE | 5 | STOREW4 | 12,669 | 
| leaf | StoreE | 6 | ADD | 6,384 | 
| leaf | StoreE | 6 | MUL | 6,384 | 
| leaf | StoreE | 6 | STOREW4 | 13,347 | 
| leaf | StoreF | 0 | ADD | 7,405 | 
| leaf | StoreF | 0 | MUL | 308 | 
| leaf | StoreF | 0 | STOREW | 105,896 | 
| leaf | StoreF | 1 | ADD | 7,279 | 
| leaf | StoreF | 1 | MUL | 212 | 
| leaf | StoreF | 1 | STOREW | 79,562 | 
| leaf | StoreF | 2 | ADD | 7,279 | 
| leaf | StoreF | 2 | MUL | 212 | 
| leaf | StoreF | 2 | STOREW | 79,562 | 
| leaf | StoreF | 3 | ADD | 7,279 | 
| leaf | StoreF | 3 | MUL | 212 | 
| leaf | StoreF | 3 | STOREW | 79,562 | 
| leaf | StoreF | 4 | ADD | 7,279 | 
| leaf | StoreF | 4 | MUL | 212 | 
| leaf | StoreF | 4 | STOREW | 79,562 | 
| leaf | StoreF | 5 | ADD | 7,279 | 
| leaf | StoreF | 5 | MUL | 212 | 
| leaf | StoreF | 5 | STOREW | 79,562 | 
| leaf | StoreF | 6 | ADD | 7,745 | 
| leaf | StoreF | 6 | MUL | 668 | 
| leaf | StoreF | 6 | STOREW | 88,940 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHeapPtr | 1 | ADD | 1 | 
| leaf | StoreHeapPtr | 2 | ADD | 1 | 
| leaf | StoreHeapPtr | 3 | ADD | 1 | 
| leaf | StoreHeapPtr | 4 | ADD | 1 | 
| leaf | StoreHeapPtr | 5 | ADD | 1 | 
| leaf | StoreHeapPtr | 6 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 228,778 | 
| leaf | StoreHintWord | 1 | HINT_STOREW | 201,352 | 
| leaf | StoreHintWord | 2 | HINT_STOREW | 201,352 | 
| leaf | StoreHintWord | 3 | HINT_STOREW | 201,352 | 
| leaf | StoreHintWord | 4 | HINT_STOREW | 201,352 | 
| leaf | StoreHintWord | 5 | HINT_STOREW | 201,352 | 
| leaf | StoreHintWord | 6 | HINT_STOREW | 212,160 | 
| leaf | StoreV | 0 | ADD | 20,416 | 
| leaf | StoreV | 0 | MUL | 16,502 | 
| leaf | StoreV | 0 | STOREW | 65,208 | 
| leaf | StoreV | 1 | ADD | 15,046 | 
| leaf | StoreV | 1 | MUL | 12,164 | 
| leaf | StoreV | 1 | STOREW | 57,600 | 
| leaf | StoreV | 2 | ADD | 15,046 | 
| leaf | StoreV | 2 | MUL | 12,164 | 
| leaf | StoreV | 2 | STOREW | 57,600 | 
| leaf | StoreV | 3 | ADD | 15,046 | 
| leaf | StoreV | 3 | MUL | 12,164 | 
| leaf | StoreV | 3 | STOREW | 57,600 | 
| leaf | StoreV | 4 | ADD | 15,046 | 
| leaf | StoreV | 4 | MUL | 12,164 | 
| leaf | StoreV | 4 | STOREW | 57,600 | 
| leaf | StoreV | 5 | ADD | 15,046 | 
| leaf | StoreV | 5 | MUL | 12,164 | 
| leaf | StoreV | 5 | STOREW | 57,600 | 
| leaf | StoreV | 6 | ADD | 16,836 | 
| leaf | StoreV | 6 | MUL | 13,610 | 
| leaf | StoreV | 6 | STOREW | 60,136 | 
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
| leaf | SubFI | 0 | SUB | 1,333 | 
| leaf | SubFI | 1 | SUB | 1,333 | 
| leaf | SubFI | 2 | SUB | 1,333 | 
| leaf | SubFI | 3 | SUB | 1,333 | 
| leaf | SubFI | 4 | SUB | 1,333 | 
| leaf | SubFI | 5 | SUB | 1,333 | 
| leaf | SubFI | 6 | SUB | 1,333 | 
| leaf | SubV | 0 | SUB | 14,636 | 
| leaf | SubV | 1 | SUB | 12,110 | 
| leaf | SubV | 2 | SUB | 12,110 | 
| leaf | SubV | 3 | SUB | 12,110 | 
| leaf | SubV | 4 | SUB | 12,110 | 
| leaf | SubV | 5 | SUB | 12,110 | 
| leaf | SubV | 6 | SUB | 12,784 | 
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
| leaf | UnsafeCastVF | 0 | ADD | 20 | 
| leaf | UnsafeCastVF | 1 | ADD | 14 | 
| leaf | UnsafeCastVF | 2 | ADD | 14 | 
| leaf | UnsafeCastVF | 3 | ADD | 14 | 
| leaf | UnsafeCastVF | 4 | ADD | 14 | 
| leaf | UnsafeCastVF | 5 | ADD | 14 | 
| leaf | UnsafeCastVF | 6 | ADD | 16 | 
| leaf | ZipFor | 0 | ADD | 365,007 | 
| leaf | ZipFor | 0 | BNE | 307,278 | 
| leaf | ZipFor | 0 | JAL | 32,115 | 
| leaf | ZipFor | 1 | ADD | 306,447 | 
| leaf | ZipFor | 1 | BNE | 263,754 | 
| leaf | ZipFor | 1 | JAL | 29,721 | 
| leaf | ZipFor | 2 | ADD | 306,447 | 
| leaf | ZipFor | 2 | BNE | 263,754 | 
| leaf | ZipFor | 2 | JAL | 29,721 | 
| leaf | ZipFor | 3 | ADD | 306,447 | 
| leaf | ZipFor | 3 | BNE | 263,754 | 
| leaf | ZipFor | 3 | JAL | 29,721 | 
| leaf | ZipFor | 4 | ADD | 306,447 | 
| leaf | ZipFor | 4 | BNE | 263,754 | 
| leaf | ZipFor | 4 | JAL | 29,721 | 
| leaf | ZipFor | 5 | ADD | 306,447 | 
| leaf | ZipFor | 5 | BNE | 263,754 | 
| leaf | ZipFor | 5 | JAL | 29,721 | 
| leaf | ZipFor | 6 | ADD | 325,673 | 
| leaf | ZipFor | 6 | BNE | 278,948 | 
| leaf | ZipFor | 6 | JAL | 30,868 | 
| root |  | 0 | ADD | 2 | 
| root |  | 0 | JAL | 1 | 
| root | AddE | 0 | FE4ADD | 12,588 | 
| root | AddEFFI | 0 | ADD | 1,488 | 
| root | AddEFI | 0 | ADD | 724 | 
| root | AddEI | 0 | ADD | 30,736 | 
| root | AddF | 0 | ADD | 1,333 | 
| root | AddFI | 0 | ADD | 80,413 | 
| root | AddV | 0 | ADD | 67,105 | 
| root | AddVI | 0 | ADD | 306,982 | 
| root | Alloc | 0 | ADD | 117,740 | 
| root | Alloc | 0 | MUL | 34,667 | 
| root | AssertEqE | 0 | BNE | 236 | 
| root | AssertEqEI | 0 | BNE | 4 | 
| root | AssertEqF | 0 | BNE | 10,800 | 
| root | AssertEqFI | 0 | BNE | 5 | 
| root | AssertEqV | 0 | BNE | 1,149 | 
| root | AssertEqVI | 0 | BNE | 237 | 
| root | AssertNonZero | 0 | BEQ | 1 | 
| root | CT-ExtractPublicValues | 0 | PHANTOM | 2 | 
| root | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| root | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| root | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| root | CT-compute-reduced-opening | 0 | PHANTOM | 504 | 
| root | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,148 | 
| root | CT-poseidon2-hash | 0 | PHANTOM | 2,520 | 
| root | CT-poseidon2-hash-ext | 0 | PHANTOM | 1,848 | 
| root | CT-poseidon2-hash-setup | 0 | PHANTOM | 107,604 | 
| root | CT-single-reduced-opening-eval | 0 | PHANTOM | 11,256 | 
| root | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| root | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| root | CT-verify-batch | 0 | PHANTOM | 504 | 
| root | CT-verify-batch-ext | 0 | PHANTOM | 1,848 | 
| root | CT-verify-batch-reduce-fast | 0 | PHANTOM | 4,368 | 
| root | CT-verify-batch-reduce-fast-setup | 0 | PHANTOM | 4,368 | 
| root | CT-verify-query | 0 | PHANTOM | 84 | 
| root | CastFV | 0 | ADD | 8 | 
| root | DivE | 0 | BBE4DIV | 6,586 | 
| root | DivEIN | 0 | ADD | 748 | 
| root | DivEIN | 0 | BBE4DIV | 187 | 
| root | DivFIN | 0 | DIV | 391 | 
| root | For | 0 | ADD | 174,545 | 
| root | For | 0 | BNE | 174,545 | 
| root | For | 0 | JAL | 13,110 | 
| root | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,628 | 
| root | HintBitsF | 0 | PHANTOM | 43 | 
| root | HintInputVec | 0 | PHANTOM | 24,203 | 
| root | IfEq | 0 | BNE | 76,756 | 
| root | IfEqI | 0 | BNE | 106,274 | 
| root | IfEqI | 0 | JAL | 47,269 | 
| root | IfNe | 0 | BEQ | 17,749 | 
| root | IfNe | 0 | JAL | 3 | 
| root | IfNeI | 0 | BEQ | 1,018 | 
| root | ImmE | 0 | ADD | 3,088 | 
| root | ImmF | 0 | ADD | 37,320 | 
| root | ImmV | 0 | ADD | 18,089 | 
| root | LoadE | 0 | ADD | 12,768 | 
| root | LoadE | 0 | LOADW4 | 23,589 | 
| root | LoadE | 0 | MUL | 12,768 | 
| root | LoadF | 0 | ADD | 13,248 | 
| root | LoadF | 0 | LOADW | 104,284 | 
| root | LoadF | 0 | MUL | 8,981 | 
| root | LoadHeapPtr | 0 | ADD | 1 | 
| root | LoadV | 0 | ADD | 51,573 | 
| root | LoadV | 0 | LOADW | 289,956 | 
| root | LoadV | 0 | MUL | 42,810 | 
| root | MulE | 0 | BBE4MUL | 14,867 | 
| root | MulEF | 0 | MUL | 5,192 | 
| root | MulEFI | 0 | MUL | 452 | 
| root | MulEI | 0 | ADD | 4,636 | 
| root | MulEI | 0 | BBE4MUL | 1,159 | 
| root | MulF | 0 | MUL | 157,325 | 
| root | MulFI | 0 | MUL | 1,350 | 
| root | MulVI | 0 | MUL | 32,058 | 
| root | NegE | 0 | MUL | 228 | 
| root | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 17,358 | 
| root | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 8,035 | 
| root | Publish | 0 | PUBLISH | 48 | 
| root | StoreE | 0 | ADD | 9,072 | 
| root | StoreE | 0 | MUL | 9,072 | 
| root | StoreE | 0 | STOREW4 | 16,460 | 
| root | StoreF | 0 | ADD | 8,166 | 
| root | StoreF | 0 | MUL | 376 | 
| root | StoreF | 0 | STOREW | 89,887 | 
| root | StoreHeapPtr | 0 | ADD | 1 | 
| root | StoreHintWord | 0 | HINT_STOREW | 226,093 | 
| root | StoreV | 0 | ADD | 23,734 | 
| root | StoreV | 0 | MUL | 19,257 | 
| root | StoreV | 0 | STOREW | 71,329 | 
| root | SubE | 0 | FE4SUB | 3,437 | 
| root | SubEF | 0 | ADD | 16,998 | 
| root | SubEF | 0 | SUB | 5,666 | 
| root | SubEFI | 0 | ADD | 264 | 
| root | SubEI | 0 | ADD | 1,496 | 
| root | SubF | 0 | SUB | 8 | 
| root | SubFI | 0 | SUB | 1,333 | 
| root | SubV | 0 | SUB | 14,885 | 
| root | SubVI | 0 | SUB | 1,090 | 
| root | SubVIN | 0 | SUB | 924 | 
| root | UnsafeCastVF | 0 | ADD | 17 | 
| root | ZipFor | 0 | ADD | 356,532 | 
| root | ZipFor | 0 | BNE | 302,329 | 
| root | ZipFor | 0 | JAL | 33,491 | 

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
| internal.0 | 0 | 3,989 | 49,528 | 6,839,770 | 609,094,104 | 21,837 | 4,816 | 3,634 | 4,142 | 4,762 | 4,038 | 226,780,243 | 443 | 23,702 | 
| internal.0 | 1 | 3,815 | 50,707 | 6,837,260 | 609,094,104 | 22,559 | 4,830 | 3,650 | 4,274 | 4,735 | 4,627 | 226,761,357 | 442 | 24,333 | 
| internal.0 | 2 | 3,821 | 50,606 | 6,836,392 | 609,094,104 | 22,527 | 4,796 | 3,679 | 4,242 | 4,745 | 4,618 | 226,750,660 | 444 | 24,258 | 
| internal.0 | 3 | 1,959 | 26,379 | 3,416,189 | 306,907,608 | 12,116 | 2,379 | 1,930 | 2,845 | 2,401 | 2,138 | 114,408,434 | 421 | 12,304 | 
| internal.1 | 4 | 4,132 | 51,564 | 7,064,557 | 609,094,104 | 22,130 | 4,841 | 3,691 | 4,281 | 4,767 | 4,103 | 234,181,114 | 444 | 25,302 | 
| internal.1 | 5 | 3,931 | 51,495 | 6,953,923 | 609,094,104 | 22,630 | 4,872 | 3,662 | 4,267 | 4,749 | 4,641 | 231,677,107 | 436 | 24,934 | 
| internal.2 | 6 | 3,925 | 51,929 | 7,064,071 | 609,094,104 | 22,465 | 4,838 | 3,722 | 4,089 | 4,740 | 4,634 | 234,176,254 | 439 | 25,539 | 
| leaf | 0 | 1,787 | 23,746 | 3,216,795 | 304,941,528 | 11,086 | 2,401 | 1,915 | 2,068 | 2,284 | 2,190 | 110,760,086 | 226 | 10,873 | 
| leaf | 1 | 1,541 | 21,538 | 2,649,626 | 274,467,288 | 10,939 | 2,415 | 1,831 | 1,997 | 2,237 | 2,235 | 91,424,498 | 220 | 9,058 | 
| leaf | 2 | 1,557 | 21,618 | 2,651,694 | 275,909,080 | 10,917 | 2,401 | 1,829 | 1,958 | 2,302 | 2,204 | 91,445,178 | 220 | 9,144 | 
| leaf | 3 | 1,588 | 21,640 | 2,649,898 | 274,467,288 | 10,827 | 2,400 | 1,798 | 1,996 | 2,244 | 2,177 | 91,427,218 | 207 | 9,225 | 
| leaf | 4 | 1,572 | 21,700 | 2,650,503 | 274,467,288 | 10,823 | 2,448 | 1,819 | 1,939 | 2,232 | 2,171 | 91,433,268 | 211 | 9,305 | 
| leaf | 5 | 1,575 | 21,721 | 2,650,047 | 274,467,288 | 10,805 | 2,401 | 1,803 | 1,962 | 2,243 | 2,180 | 91,428,708 | 214 | 9,341 | 
| leaf | 6 | 1,617 | 22,773 | 2,844,076 | 302,647,768 | 11,233 | 2,350 | 1,911 | 2,137 | 2,356 | 2,129 | 98,109,740 | 347 | 9,923 | 
| root | 0 | 2,752 | 81,172 | 3,531,442 | 306,907,608 | 65,548 | 2,746 | 14,835 | 20,538 | 7,270 | 19,927 | 118,184,203 | 229 | 12,872 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 962 | 11,789 | 1,747,603 | 197,440,542 | 5,048 | 830 | 526 | 1,664 | 1,031 | 798 | 59,825,673 | 197 | 5,779 | 
| fib_e2e | 1 | 908 | 11,420 | 1,747,502 | 197,423,810 | 4,818 | 823 | 441 | 1,782 | 1,020 | 617 | 59,798,110 | 132 | 5,694 | 
| fib_e2e | 2 | 991 | 11,587 | 1,747,502 | 197,423,810 | 4,840 | 815 | 433 | 1,835 | 987 | 634 | 59,798,101 | 133 | 5,756 | 
| fib_e2e | 3 | 876 | 11,542 | 1,747,502 | 197,423,810 | 4,920 | 812 | 491 | 1,879 | 1,009 | 609 | 59,798,420 | 118 | 5,746 | 
| fib_e2e | 4 | 868 | 11,393 | 1,747,502 | 197,423,810 | 4,777 | 725 | 368 | 1,931 | 1,033 | 601 | 59,798,719 | 117 | 5,748 | 
| fib_e2e | 5 | 864 | 11,411 | 1,747,502 | 197,423,810 | 4,798 | 842 | 413 | 1,782 | 1,057 | 585 | 59,798,709 | 117 | 5,749 | 
| fib_e2e | 6 | 760 | 10,654 | 1,515,024 | 197,432,594 | 4,897 | 807 | 420 | 1,818 | 996 | 670 | 52,001,176 | 184 | 4,997 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-fib_e2e.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-halo2_outer.cell_tracker_span.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-halo2_outer.cell_tracker_span.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-halo2_outer.cell_tracker_span.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-halo2_outer.cell_tracker_span.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.0.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.1.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-internal.2.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/c710bdc3f383aceed4ae546933cba5cbdb898e19/fib_e2e-c710bdc3f383aceed4ae546933cba5cbdb898e19-root.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/c710bdc3f383aceed4ae546933cba5cbdb898e19

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12851930243)
