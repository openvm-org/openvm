| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total |  474.72 |  288.45 |
| fib_e2e |  71.73 |  10.69 |
| leaf |  65.23 |  10.48 |
| internal.0 |  70.34 |  20.05 |
| internal.1 |  40.46 |  20.28 |
| internal.2 |  20.16 |  20.16 |
| root |  37.30 |  37.30 |
| halo2_outer |  122.46 |  122.46 |
| halo2_wrapper |  47.04 |  47.04 |


| fib_e2e |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  10,247.86 |  71,735 |  10,692 |  9,418 |
| `main_cells_used     ` |  58,670,158.29 |  410,691,108 |  59,803,503 |  51,985,396 |
| `total_cycles        ` |  1,515,083 |  1,515,083 |  1,515,083 |  1,515,083 |
| `execute_time_ms     ` |  5,577.43 |  39,042 |  5,704 |  4,921 |
| `trace_gen_time_ms   ` |  707 |  4,949 |  829 |  607 |
| `stark_prove_excluding_trace_time_ms` |  3,963.43 |  27,744 |  4,251 |  3,853 |
| `main_trace_commit_time_ms` |  719.43 |  5,036 |  860 |  682 |
| `generate_perm_trace_time_ms` |  134.14 |  939 |  163 |  122 |
| `perm_trace_commit_time_ms` |  802.71 |  5,619 |  851 |  750 |
| `quotient_poly_compute_time_ms` |  528 |  3,696 |  534 |  521 |
| `quotient_poly_commit_time_ms` |  704.57 |  4,932 |  788 |  676 |
| `pcs_opening_time_ms ` |  1,071.29 |  7,499 |  1,096 |  1,050 |

| leaf |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  9,319.29 |  65,235 |  10,480 |  9,011 |
| `main_cells_used     ` |  43,908,785.14 |  307,361,496 |  50,851,136 |  42,114,634 |
| `total_cycles        ` |  1,076,576 |  7,536,032 |  1,270,725 |  1,026,432 |
| `execute_time_ms     ` |  3,925.86 |  27,481 |  4,528 |  3,722 |
| `trace_gen_time_ms   ` |  790.14 |  5,531 |  892 |  754 |
| `stark_prove_excluding_trace_time_ms` |  4,603.29 |  32,223 |  5,060 |  4,504 |
| `main_trace_commit_time_ms` |  897.57 |  6,283 |  1,005 |  870 |
| `generate_perm_trace_time_ms` |  109.14 |  764 |  119 |  107 |
| `perm_trace_commit_time_ms` |  839 |  5,873 |  951 |  817 |
| `quotient_poly_compute_time_ms` |  616.71 |  4,317 |  680 |  601 |
| `quotient_poly_commit_time_ms` |  933.14 |  6,532 |  1,046 |  902 |
| `pcs_opening_time_ms ` |  1,203.71 |  8,426 |  1,255 |  1,181 |

| internal.0 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  17,585 |  70,340 |  20,049 |  10,826 |
| `main_cells_used     ` |  85,851,795.50 |  343,407,182 |  97,838,026 |  49,991,906 |
| `total_cycles        ` |  2,126,852.50 |  8,507,410 |  2,430,923 |  1,215,838 |
| `execute_time_ms     ` |  7,798.75 |  31,195 |  9,018 |  4,550 |
| `trace_gen_time_ms   ` |  1,422.75 |  5,691 |  1,637 |  1,006 |
| `stark_prove_excluding_trace_time_ms` |  8,363.50 |  33,454 |  9,411 |  5,270 |
| `main_trace_commit_time_ms` |  1,769.75 |  7,079 |  1,972 |  1,210 |
| `generate_perm_trace_time_ms` |  204.75 |  819 |  236 |  121 |
| `perm_trace_commit_time_ms` |  1,634 |  6,536 |  1,868 |  952 |
| `quotient_poly_compute_time_ms` |  1,120.75 |  4,483 |  1,271 |  696 |
| `quotient_poly_commit_time_ms` |  1,633.25 |  6,533 |  1,848 |  1,010 |
| `pcs_opening_time_ms ` |  1,996.75 |  7,987 |  2,239 |  1,277 |

| internal.1 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  20,231 |  40,462 |  20,284 |  20,178 |
| `main_cells_used     ` |  99,572,342 |  199,144,684 |  99,779,328 |  99,365,356 |
| `total_cycles        ` |  2,456,097.50 |  4,912,195 |  2,464,036 |  2,448,159 |
| `execute_time_ms     ` |  9,167.50 |  18,335 |  9,177 |  9,158 |
| `trace_gen_time_ms   ` |  1,607.50 |  3,215 |  1,645 |  1,570 |
| `stark_prove_excluding_trace_time_ms` |  9,456 |  18,912 |  9,481 |  9,431 |
| `main_trace_commit_time_ms` |  1,983.50 |  3,967 |  1,987 |  1,980 |
| `generate_perm_trace_time_ms` |  228 |  456 |  228 |  228 |
| `perm_trace_commit_time_ms` |  1,862 |  3,724 |  1,872 |  1,852 |
| `quotient_poly_compute_time_ms` |  1,271 |  2,542 |  1,271 |  1,271 |
| `quotient_poly_commit_time_ms` |  1,836 |  3,672 |  1,841 |  1,831 |
| `pcs_opening_time_ms ` |  2,270 |  4,540 |  2,297 |  2,243 |

| internal.2 |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  20,155 |  20,155 |  20,155 |  20,155 |
| `main_cells_used     ` |  99,779,733 |  99,779,733 |  99,779,733 |  99,779,733 |
| `total_cycles        ` |  2,464,081 |  2,464,081 |  2,464,081 |  2,464,081 |
| `execute_time_ms     ` |  9,117 |  9,117 |  9,117 |  9,117 |
| `trace_gen_time_ms   ` |  1,641 |  1,641 |  1,641 |  1,641 |
| `stark_prove_excluding_trace_time_ms` |  9,397 |  9,397 |  9,397 |  9,397 |
| `main_trace_commit_time_ms` |  1,956 |  1,956 |  1,956 |  1,956 |
| `generate_perm_trace_time_ms` |  229 |  229 |  229 |  229 |
| `perm_trace_commit_time_ms` |  1,858 |  1,858 |  1,858 |  1,858 |
| `quotient_poly_compute_time_ms` |  1,262 |  1,262 |  1,262 |  1,262 |
| `quotient_poly_commit_time_ms` |  1,840 |  1,840 |  1,840 |  1,840 |
| `pcs_opening_time_ms ` |  2,248 |  2,248 |  2,248 |  2,248 |

| root |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  37,296 |  37,296 |  37,296 |  37,296 |
| `main_cells_used     ` |  51,005,063 |  51,005,063 |  51,005,063 |  51,005,063 |
| `total_cycles        ` |  1,232,879 |  1,232,879 |  1,232,879 |  1,232,879 |
| `execute_time_ms     ` |  4,742 |  4,742 |  4,742 |  4,742 |
| `trace_gen_time_ms   ` |  815 |  815 |  815 |  815 |
| `stark_prove_excluding_trace_time_ms` |  31,739 |  31,739 |  31,739 |  31,739 |
| `main_trace_commit_time_ms` |  10,165 |  10,165 |  10,165 |  10,165 |
| `generate_perm_trace_time_ms` |  120 |  120 |  120 |  120 |
| `perm_trace_commit_time_ms` |  9,683 |  9,683 |  9,683 |  9,683 |
| `quotient_poly_compute_time_ms` |  682 |  682 |  682 |  682 |
| `quotient_poly_commit_time_ms` |  7,302 |  7,302 |  7,302 |  7,302 |
| `pcs_opening_time_ms ` |  3,776 |  3,776 |  3,776 |  3,776 |

| halo2_outer |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  122,456 |  122,456 |  122,456 |  122,456 |
| `main_cells_used     ` |  74,268,773 |  74,268,773 |  74,268,773 |  74,268,773 |

| halo2_wrapper |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` |  47,042 |  47,042 |  47,042 |  47,042 |



<details>
<summary>Detailed Metrics</summary>

|  | execute_time_ms |
| --- |
|  | 2,529 | 

| group | total_proof_time_ms | num_segments | main_cells_used |
| --- | --- | --- | --- |
| fib_e2e |  | 7 |  | 
| halo2_outer | 122,456 |  | 74,268,773 | 
| halo2_wrapper | 47,042 |  |  | 

| group | air_name | dsl_ir | idx | opcode | cells_used |
| --- | --- | --- | --- | --- | --- |
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 1 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 2 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 83,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 83,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 83,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 41,528 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 56,840 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 56,840 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 56,840 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 28,420 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 2,104,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 2,104,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 2,104,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 1,052,352 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 288,260 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 1 | ADD | 288,260 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 2 | ADD | 288,260 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 3 | ADD | 144,130 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 693,593 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 693,593 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 693,593 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 347,159 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 1,426,278 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 1 | ADD | 1,426,162 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 2 | ADD | 1,426,162 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 3 | ADD | 713,342 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 2,605,534 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 2,605,534 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 2,605,534 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 1,303,666 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 2,397,662 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 688,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 2,397,662 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 688,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 2,397,662 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 688,808 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 1,199,498 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 344,607 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 8,700 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 8,700 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 8,700 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 4,350 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 45,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 45,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 45,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 22,852 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 321,552 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 1 | DIV | 321,552 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 2 | DIV | 321,552 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 3 | DIV | 160,776 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 23,838 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 23,838 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 23,838 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 11,919 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 235,944 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 235,944 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 235,944 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 117,972 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 332,630 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 332,630 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 332,630 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 168,751 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 303,253 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 303,253 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 303,253 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 153,323 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 628,488 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 314,244 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 314,244 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 797,384 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 530,874 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 797,384 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 530,874 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 797,384 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 530,874 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 398,692 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 265,437 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 2,682,326 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 2,395,748 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 2,682,326 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 2,395,748 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 2,682,326 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 2,395,748 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 1,341,163 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 1,197,874 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 286,288 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 286,288 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 286,288 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 143,144 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 41,064 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 41,064 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 41,064 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 20,532 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 286,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 286,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 286,056 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 143,028 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,559,388 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 1 | MUL | 1,559,388 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 2 | MUL | 1,559,388 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 3 | MUL | 779,694 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 256,302 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 256,302 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 256,302 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 128,151 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 580,957 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 580,957 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 580,957 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 290,493 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 16,936 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 1 | MUL | 16,936 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 2 | MUL | 16,936 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 3 | MUL | 8,468 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 531,048 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 265,524 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 265,524 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 43,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 22,272 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 43,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 22,272 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 43,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 22,272 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 21,750 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 11,136 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 58 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 29 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 817,336 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 547,810 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 817,336 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 547,810 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 817,336 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 547,810 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 408,668 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 273,905 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 1,000,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 333,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 1,000,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 333,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 1,000,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 333,500 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 500,250 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 166,750 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 22,736 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 22,736 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 22,736 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 11,368 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 91,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 91,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 91,408 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 45,704 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 0 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 1 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 2 | SUB | 464 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 3 | SUB | 232 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 255,316 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 255,316 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 255,316 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 127,658 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 368,822 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 1 | SUB | 368,822 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 2 | SUB | 368,822 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 3 | SUB | 184,411 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 57,884 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 57,884 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 57,884 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 28,942 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 48,720 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 48,720 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 48,720 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 24,360 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 6,728 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 6,728 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 6,728 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 3,364 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 9,338,754 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 9,337,188 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 9,337,188 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 4,670,450 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 1 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 2 | BNE | 10,856 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 3 | BNE | 5,428 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 1 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 2 | BNE | 184 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 3 | BNE | 92 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 209,599 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 209,599 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 209,599 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 104,512 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 1 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 2 | BNE | 161 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 3 | BNE | 69 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 62,606 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 62,606 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 62,606 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 31,303 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 10,994 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 10,994 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 10,994 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 5,497 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 6,440 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 3,220 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 962,182 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 962,182 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 962,182 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 481,091 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 6,578 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 3,289 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 4,278 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 2,139 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 6,247,076 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 6,245,834 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 6,245,834 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 3,124,389 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 9 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 9 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 9 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 9 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 115,623 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 123,498 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 119,142 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 59,832 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 54 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 54 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 54 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 27 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 267,075 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 267,075 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 267,075 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 133,587 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 1,196 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 1,196 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 1,239,964 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 1,239,964 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 1,239,964 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 620,070 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 5,375,524 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 5,375,524 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 5,375,524 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 2,687,872 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 281,644 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 281,644 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 281,644 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 142,670 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 3,729,176 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 3,729,176 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 3,729,176 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 1,864,786 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 1,263,636 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 1,263,636 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 1,263,636 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 633,050 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 1,604,188 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 1,604,188 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 1,604,188 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 802,094 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 903,464 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 903,464 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 903,464 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 451,732 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 1,039,756 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 1,039,756 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 1,039,756 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 519,878 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 500,536 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 500,536 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 500,536 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 250,268 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 14,972 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 14,972 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 14,972 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 7,486 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 2,076,396 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 2,074,496 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 2,074,496 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 1,039,148 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 93,708 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 93,708 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 93,708 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 46,854 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 259,464 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 259,464 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 259,464 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 129,732 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,623,800 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 5,623,800 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 5,623,800 | 
| internal.0 | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 2,811,900 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-HintOpenedValues | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 6,072 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 1 | PHANTOM | 6,072 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 2 | PHANTOM | 6,072 | 
| internal.0 | PhantomAir | CT-HintOpeningProof | 3 | PHANTOM | 3,036 | 
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
| internal.0 | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-cache-generator-powers | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 1 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 2 | PHANTOM | 6,048 | 
| internal.0 | PhantomAir | CT-compute-reduced-opening | 3 | PHANTOM | 3,024 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 99,792 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 99,792 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 99,792 | 
| internal.0 | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 49,896 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 1 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 2 | PHANTOM | 24 | 
| internal.0 | PhantomAir | CT-pre-compute-alpha-pows | 3 | PHANTOM | 12 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 137,088 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 137,088 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 137,088 | 
| internal.0 | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 68,544 | 
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
| internal.0 | PhantomAir | HintBitsF | 0 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintBitsF | 1 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintBitsF | 2 | PHANTOM | 1,704 | 
| internal.0 | PhantomAir | HintBitsF | 3 | PHANTOM | 852 | 
| internal.0 | PhantomAir | HintInputVec | 0 | PHANTOM | 92,418 | 
| internal.0 | PhantomAir | HintInputVec | 1 | PHANTOM | 92,418 | 
| internal.0 | PhantomAir | HintInputVec | 2 | PHANTOM | 92,418 | 
| internal.0 | PhantomAir | HintInputVec | 3 | PHANTOM | 46,236 | 
| internal.0 | PhantomAir | HintLoad | 0 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | HintLoad | 1 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | HintLoad | 2 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | HintLoad | 3 | PHANTOM | 6,552 | 
| internal.0 | PhantomAir | PrintV | 0 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | PrintV | 1 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | PrintV | 2 | PHANTOM | 13,104 | 
| internal.0 | PhantomAir | PrintV | 3 | PHANTOM | 6,552 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 40,698 | 
| internal.0 | VerifyBatchAir | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 20,349 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 1 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 2 | VERIFY_BATCH | 9,049,320 | 
| internal.0 | VerifyBatchAir | VerifyBatchExt | 3 | VERIFY_BATCH | 4,524,660 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 10,440,234 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 10,389,960 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 10,389,960 | 
| internal.0 | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 5,194,980 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 4 | ADD | 29 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 5 | ADD | 29 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 87,696 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 87,232 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 56,840 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 56,840 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 2,138,112 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 2,123,264 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 4 | ADD | 288,260 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 5 | ADD | 288,260 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 724,681 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 709,833 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 4 | ADD | 1,491,412 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 5 | ADD | 1,458,961 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 2,659,880 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 2,633,635 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 2,446,614 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 703,482 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 2,422,138 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 696,145 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 8,700 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 8,700 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 45,704 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 45,704 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 4 | DIV | 336,168 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 5 | DIV | 328,860 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 23,838 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 23,838 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 239,656 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 239,656 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 332,630 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 332,630 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 312,069 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 308,009 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 635,796 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 635,796 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 632,142 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 632,142 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 797,616 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 530,874 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 797,500 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 530,874 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 2,694,506 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 2,405,492 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 2,688,416 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 2,400,620 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 296,032 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 291,160 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 41,064 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 41,064 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 286,984 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 286,520 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 4 | MUL | 1,617,852 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 5 | MUL | 1,588,620 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 256,302 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 256,302 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 583,393 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 582,175 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 4 | MUL | 16,936 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 5 | MUL | 16,936 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 533,484 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 533,484 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 532,266 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 532,266 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 44,892 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 22,272 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 44,660 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 22,272 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 58 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 814,900 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 547,810 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 816,118 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 547,810 | 
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
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 255,316 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 255,316 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 4 | SUB | 373,694 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 5 | SUB | 371,258 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 60,552 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 59,218 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 51,156 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 49,938 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 6,728 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 6,728 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 9,490,076 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 9,416,213 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 4 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 5 | BNE | 10,856 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 4 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 5 | BNE | 184 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 209,967 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 209,967 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 4 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 5 | BNE | 161 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 64,538 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 63,572 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 10,994 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 10,994 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 6,624 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 6,532 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 990,334 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 976,626 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 6,762 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 6,670 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 4,278 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 4,278 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 6,355,452 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 6,302,690 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 9 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 9 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 126,333 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 125,046 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 54 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 54 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 270,117 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 268,596 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 1,196 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 1,196 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 1,244,452 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 1,243,264 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 5,420,008 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 5,397,766 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 293,788 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 288,068 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 3,748,756 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 3,739,318 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 1,271,116 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 1,267,376 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 1,619,812 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 1,612,000 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 913,942 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 908,703 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 1,047,584 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 1,044,278 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 503,728 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 502,132 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 14,972 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 14,972 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 2,118,272 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 2,099,538 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 94,012 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 93,860 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 269,040 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 264,252 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 5,623,800 | 
| internal.1 | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 5,623,800 | 
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
| internal.1 | PhantomAir | CT-cache-generator-powers | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-cache-generator-powers | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 4 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-compute-reduced-opening | 5 | PHANTOM | 6,048 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 99,792 | 
| internal.1 | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 99,792 | 
| internal.1 | PhantomAir | CT-pre-compute-alpha-pows | 4 | PHANTOM | 24 | 
| internal.1 | PhantomAir | CT-pre-compute-alpha-pows | 5 | PHANTOM | 24 | 
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
| internal.1 | PhantomAir | HintBitsF | 4 | PHANTOM | 1,704 | 
| internal.1 | PhantomAir | HintBitsF | 5 | PHANTOM | 1,704 | 
| internal.1 | PhantomAir | HintInputVec | 4 | PHANTOM | 93,942 | 
| internal.1 | PhantomAir | HintInputVec | 5 | PHANTOM | 93,180 | 
| internal.1 | PhantomAir | HintLoad | 4 | PHANTOM | 13,608 | 
| internal.1 | PhantomAir | HintLoad | 5 | PHANTOM | 13,356 | 
| internal.1 | PhantomAir | PrintV | 4 | PHANTOM | 13,608 | 
| internal.1 | PhantomAir | PrintV | 5 | PHANTOM | 13,356 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 43,092 | 
| internal.1 | VerifyBatchAir | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 42,693 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 4 | VERIFY_BATCH | 9,853,704 | 
| internal.1 | VerifyBatchAir | VerifyBatchExt | 5 | VERIFY_BATCH | 9,451,512 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 10,591,056 | 
| internal.1 | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 10,540,782 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 6 | ADD | 29 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 87,696 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 56,840 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 2,138,112 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 6 | ADD | 288,260 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 724,681 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 6 | ADD | 1,491,412 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 2,659,880 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 2,446,614 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 703,482 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 8,700 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 45,704 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 6 | DIV | 336,168 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 23,838 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 239,656 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 332,630 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 312,069 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 635,796 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 635,796 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 797,616 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 530,874 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 58 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 2,694,506 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 2,405,492 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 296,032 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 41,064 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 286,984 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 6 | MUL | 1,617,852 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 256,302 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 583,393 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 6 | MUL | 16,936 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 533,484 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 533,484 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 44,892 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 22,272 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 58 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 814,900 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 547,810 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 1,000,500 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 333,500 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 22,736 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 91,408 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 6 | SUB | 464 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 255,316 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 6 | SUB | 373,694 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 60,552 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 51,156 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 6,728 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 9,490,076 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 6 | BNE | 10,856 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 6 | BNE | 184 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 209,967 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 6 | BNE | 161 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 64,538 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 10,994 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 6,624 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 990,334 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 6,762 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 4,278 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 6,355,452 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 9 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 126,738 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 54 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 270,117 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 1,196 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 1,244,452 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 5,420,008 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 293,788 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 3,748,756 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 1,271,116 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 1,619,812 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 913,942 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 1,047,584 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 503,728 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 14,972 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 2,118,272 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 94,012 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 269,040 | 
| internal.2 | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 5,623,800 | 
| internal.2 | PhantomAir | CT-HintOpenedValues | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-HintOpeningProof | 6 | PHANTOM | 6,072 | 
| internal.2 | PhantomAir | CT-HintOpeningValues | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-InitializePcsConst | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-ReadProofsFromInput | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-VerifyProofs | 6 | PHANTOM | 12 | 
| internal.2 | PhantomAir | CT-cache-generator-powers | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-compute-reduced-opening | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 99,792 | 
| internal.2 | PhantomAir | CT-pre-compute-alpha-pows | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 137,088 | 
| internal.2 | PhantomAir | CT-stage-c-build-rounds | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verifier-verify | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-d-verify-pcs | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-stage-e-verify-constraints | 6 | PHANTOM | 24 | 
| internal.2 | PhantomAir | CT-verify-batch | 6 | PHANTOM | 6,048 | 
| internal.2 | PhantomAir | CT-verify-batch-ext | 6 | PHANTOM | 21,168 | 
| internal.2 | PhantomAir | CT-verify-query | 6 | PHANTOM | 1,008 | 
| internal.2 | PhantomAir | HintBitsF | 6 | PHANTOM | 1,704 | 
| internal.2 | PhantomAir | HintInputVec | 6 | PHANTOM | 93,942 | 
| internal.2 | PhantomAir | HintLoad | 6 | PHANTOM | 13,608 | 
| internal.2 | PhantomAir | PrintV | 6 | PHANTOM | 13,608 | 
| internal.2 | VerifyBatchAir | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 43,092 | 
| internal.2 | VerifyBatchAir | VerifyBatchExt | 6 | VERIFY_BATCH | 9,853,704 | 
| internal.2 | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 10,591,056 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 36,424 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 1 | ADD | 29,464 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 2 | ADD | 29,464 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 3 | ADD | 29,464 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 4 | ADD | 29,464 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 5 | ADD | 29,464 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 6 | ADD | 32,248 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 19,604 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 1 | ADD | 15,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 2 | ADD | 15,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 3 | ADD | 15,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 4 | ADD | 15,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 5 | ADD | 15,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 6 | ADD | 17,516 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 872,320 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 1 | ADD | 702,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 2 | ADD | 702,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 3 | ADD | 702,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 4 | ADD | 702,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 5 | ADD | 702,612 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 6 | ADD | 781,608 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 1 | ADD | 120,785 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 2 | ADD | 120,785 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 3 | ADD | 120,785 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 4 | ADD | 120,785 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 5 | ADD | 120,785 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 6 | ADD | 132,965 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 455,996 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 1 | ADD | 454,111 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 2 | ADD | 454,111 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 3 | ADD | 454,111 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 4 | ADD | 454,111 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 5 | ADD | 454,111 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 6 | ADD | 455,097 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 744,749 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 1 | ADD | 664,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 2 | ADD | 664,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 3 | ADD | 664,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 4 | ADD | 664,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 5 | ADD | 664,912 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 6 | ADD | 703,279 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,489,150 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 1 | ADD | 1,303,405 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 2 | ADD | 1,303,405 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 3 | ADD | 1,303,405 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 4 | ADD | 1,303,405 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 5 | ADD | 1,303,405 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 6 | ADD | 1,384,257 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,283,482 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 369,837 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | ADD | 1,117,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 1 | MUL | 326,627 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | ADD | 1,117,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 2 | MUL | 326,627 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | ADD | 1,117,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 3 | MUL | 326,627 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | ADD | 1,117,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 4 | MUL | 326,627 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | ADD | 1,117,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 5 | MUL | 326,627 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | ADD | 1,210,112 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 6 | MUL | 350,552 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 4,350 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 1 | ADD | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 2 | ADD | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 3 | ADD | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 4 | ADD | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 5 | ADD | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 6 | ADD | 3,828 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,156 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 1 | ADD | 14,616 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 2 | ADD | 14,616 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 3 | ADD | 14,616 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 4 | ADD | 14,616 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 5 | ADD | 14,616 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 6 | ADD | 17,632 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 1 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 2 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 3 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 4 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 5 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 6 | DIV | 214,368 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,629 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 1 | DIV | 7,714 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 2 | DIV | 7,714 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 3 | DIV | 7,714 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 4 | DIV | 7,714 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 5 | DIV | 7,714 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 6 | DIV | 9,280 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 94,192 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 1 | ADD | 64,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 2 | ADD | 64,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 3 | ADD | 64,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 4 | ADD | 64,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 5 | ADD | 64,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 6 | ADD | 80,620 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 181,105 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 1 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 2 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 3 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 4 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 5 | ADD | 151,235 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 6 | ADD | 163,183 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 155,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 1 | ADD | 150,278 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 2 | ADD | 150,278 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 3 | ADD | 150,278 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 4 | ADD | 150,278 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 5 | ADD | 150,278 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 6 | ADD | 152,511 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 331,296 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 331,296 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | ADD | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 1 | MUL | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | ADD | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 2 | MUL | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | ADD | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 3 | MUL | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | ADD | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 4 | MUL | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | ADD | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 5 | MUL | 258,216 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | ADD | 287,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 6 | MUL | 287,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 432,448 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 288,231 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | ADD | 319,928 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 1 | MUL | 212,251 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | ADD | 319,928 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 2 | MUL | 212,251 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | ADD | 319,928 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 3 | MUL | 212,251 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | ADD | 319,928 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 4 | MUL | 212,251 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | ADD | 319,928 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 5 | MUL | 212,251 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | ADD | 364,936 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 6 | MUL | 242,643 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,441,445 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,291,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | ADD | 1,085,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 1 | MUL | 972,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | ADD | 1,085,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 2 | MUL | 972,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | ADD | 1,085,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 3 | MUL | 972,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | ADD | 1,085,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 4 | MUL | 972,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | ADD | 1,085,615 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 5 | MUL | 972,254 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | ADD | 1,227,947 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 6 | MUL | 1,099,854 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 141,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 1 | MUL | 126,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 2 | MUL | 126,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 3 | MUL | 126,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 4 | MUL | 126,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 5 | MUL | 126,672 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 6 | MUL | 132,704 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 14,848 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 1 | MUL | 6,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 2 | MUL | 6,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 3 | MUL | 6,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 4 | MUL | 6,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 5 | MUL | 6,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 6 | MUL | 11,020 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 114,956 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 1 | ADD | 77,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 2 | ADD | 77,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 3 | ADD | 77,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 4 | ADD | 77,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 5 | ADD | 77,720 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 6 | ADD | 93,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 1,000,471 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 1 | MUL | 970,601 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 2 | MUL | 970,601 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 3 | MUL | 970,601 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 4 | MUL | 970,601 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 5 | MUL | 970,601 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 6 | MUL | 982,549 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 134,502 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 1 | MUL | 107,387 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 2 | MUL | 107,387 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 3 | MUL | 107,387 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 4 | MUL | 107,387 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 5 | MUL | 107,387 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 6 | MUL | 118,233 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 313,084 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 1 | MUL | 238,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 2 | MUL | 238,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 3 | MUL | 238,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 4 | MUL | 238,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 5 | MUL | 238,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 6 | MUL | 268,192 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 1 | MUL | 2,668 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 2 | MUL | 2,668 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 3 | MUL | 2,668 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 4 | MUL | 2,668 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 5 | MUL | 2,668 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 6 | MUL | 3,480 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 282,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 282,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | ADD | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 1 | MUL | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | ADD | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 2 | MUL | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | ADD | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 3 | MUL | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | ADD | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 4 | MUL | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | ADD | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 5 | MUL | 209,496 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | ADD | 238,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 6 | MUL | 238,728 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 22,504 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 11,716 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | ADD | 18,299 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 1 | MUL | 8,236 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | ADD | 18,299 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 2 | MUL | 8,236 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | ADD | 18,299 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 3 | MUL | 8,236 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | ADD | 18,299 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 4 | MUL | 8,236 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | ADD | 18,299 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 5 | MUL | 8,236 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | ADD | 26,245 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 6 | MUL | 15,892 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 1 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 2 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 3 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 4 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 5 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 6 | ADD | 29 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 437,001 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 293,567 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | ADD | 316,796 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 1 | MUL | 210,772 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | ADD | 316,796 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 2 | MUL | 210,772 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | ADD | 316,796 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 3 | MUL | 210,772 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | ADD | 316,796 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 4 | MUL | 210,772 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | ADD | 316,796 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 5 | MUL | 210,772 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | ADD | 364,878 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 6 | MUL | 243,890 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 549,492 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 183,164 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | ADD | 402,462 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 1 | SUB | 134,154 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | ADD | 402,462 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 2 | SUB | 134,154 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | ADD | 402,462 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 3 | SUB | 134,154 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | ADD | 402,462 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 4 | SUB | 134,154 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | ADD | 402,462 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 5 | SUB | 134,154 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | ADD | 461,274 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 6 | SUB | 153,758 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 8,700 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 1 | ADD | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 2 | ADD | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 3 | ADD | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 4 | ADD | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 5 | ADD | 4,408 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 6 | ADD | 6,960 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 44,312 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 1 | ADD | 29,232 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 2 | ADD | 29,232 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 3 | ADD | 29,232 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 4 | ADD | 29,232 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 5 | ADD | 29,232 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 6 | ADD | 35,264 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 133,951 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 1 | SUB | 106,981 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 2 | SUB | 106,981 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 3 | SUB | 106,981 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 4 | SUB | 106,981 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 5 | SUB | 106,981 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 6 | SUB | 117,769 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 197,867 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 1 | SUB | 161,182 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 2 | SUB | 161,182 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 3 | SUB | 161,182 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 4 | SUB | 161,182 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 5 | SUB | 161,182 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 6 | SUB | 175,856 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 28,971 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 1 | SUB | 28,826 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 2 | SUB | 28,826 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 3 | SUB | 28,826 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 4 | SUB | 28,826 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 5 | SUB | 28,826 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 6 | SUB | 28,884 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 1 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 2 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 3 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 4 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 5 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 6 | SUB | 24,360 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,625 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 1 | ADD | 2,610 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 2 | ADD | 2,610 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 3 | ADD | 2,610 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 4 | ADD | 2,610 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 5 | ADD | 2,610 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 6 | ADD | 3,016 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 4,711,050 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 1 | ADD | 3,803,263 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 2 | ADD | 3,803,263 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 3 | ADD | 3,803,263 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 4 | ADD | 3,803,263 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 5 | ADD | 3,803,263 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 6 | ADD | 4,195,459 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,612 | 
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
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 109,664 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 1 | BNE | 87,584 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 2 | BNE | 87,584 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 3 | BNE | 87,584 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 4 | BNE | 87,584 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 5 | BNE | 87,584 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 6 | BNE | 96,600 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 33,764 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 1 | BNE | 32,269 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 2 | BNE | 32,269 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 3 | BNE | 32,269 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 4 | BNE | 32,269 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 5 | BNE | 32,269 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 6 | BNE | 32,867 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,934 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 1 | BNE | 4,439 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 2 | BNE | 4,439 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 3 | BNE | 4,439 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 4 | BNE | 4,439 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 5 | BNE | 4,439 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 6 | BNE | 5,060 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 1 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 2 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 3 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 4 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 5 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 6 | BEQ | 23 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,220 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 1 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 2 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 3 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 4 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 5 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 6 | BNE | 3,243 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 580,773 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 1 | BNE | 528,793 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 2 | BNE | 528,793 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 3 | BNE | 528,793 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 4 | BNE | 528,793 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 5 | BNE | 528,793 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 6 | BNE | 549,585 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 1 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 2 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 3 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 4 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 5 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 6 | BEQ | 3,289 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,185 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 1 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 2 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 3 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 4 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 5 | BEQ | 1,610 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 6 | BEQ | 1,840 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 3,071,328 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 1 | BNE | 2,477,629 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 2 | BNE | 2,477,629 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 3 | BNE | 2,477,629 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 4 | BNE | 2,477,629 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 5 | BNE | 2,477,629 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 6 | BNE | 2,738,173 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 1 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 2 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 3 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 4 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 5 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> |  | 6 | JAL | 9 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 80,604 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 1 | JAL | 79,569 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 2 | JAL | 78,660 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 3 | JAL | 79,497 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 4 | JAL | 78,381 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 5 | JAL | 76,914 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 6 | JAL | 79,146 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 27 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 1 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 2 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 3 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 4 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 5 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | IfNe | 6 | JAL | 18 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 144,567 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 1 | JAL | 119,187 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 2 | JAL | 119,187 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 3 | JAL | 119,187 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 4 | JAL | 119,187 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 5 | JAL | 119,187 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 6 | JAL | 131,373 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 1 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 2 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 3 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 4 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 5 | PUBLISH | 828 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 6 | PUBLISH | 828 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 664,576 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 1 | LOADW | 501,556 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 2 | LOADW | 501,556 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 3 | LOADW | 501,556 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 4 | LOADW | 501,556 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 5 | LOADW | 501,556 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 6 | LOADW | 581,196 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 2,967,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 1 | LOADW | 2,340,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 2 | LOADW | 2,340,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 3 | LOADW | 2,340,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 4 | LOADW | 2,340,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 5 | LOADW | 2,340,844 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 6 | LOADW | 2,596,616 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 182,490 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 1 | STOREW | 179,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 2 | STOREW | 179,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 3 | STOREW | 179,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 4 | STOREW | 179,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 5 | STOREW | 179,300 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 6 | STOREW | 194,832 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 1,749,330 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 1 | HINT_STOREW | 1,328,580 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 2 | HINT_STOREW | 1,328,580 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 3 | HINT_STOREW | 1,328,580 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 4 | HINT_STOREW | 1,328,580 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 5 | HINT_STOREW | 1,328,580 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 6 | HINT_STOREW | 1,519,540 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 675,840 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 1 | STOREW | 524,260 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 2 | STOREW | 524,260 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 3 | STOREW | 524,260 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 4 | STOREW | 524,260 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 5 | STOREW | 524,260 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 6 | STOREW | 584,892 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 838,488 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 1 | LOADW | 648,861 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 2 | LOADW | 648,861 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 3 | LOADW | 648,861 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 4 | LOADW | 648,861 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 5 | LOADW | 648,861 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 6 | LOADW | 725,493 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 470,363 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 1 | STOREW | 390,848 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 2 | STOREW | 390,848 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 3 | STOREW | 390,848 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 4 | STOREW | 390,848 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 5 | STOREW | 390,848 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 6 | STOREW | 422,654 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 488,072 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 1 | FE4ADD | 365,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 2 | FE4ADD | 365,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 3 | FE4ADD | 365,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 4 | FE4ADD | 365,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 5 | FE4ADD | 365,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 6 | FE4ADD | 418,798 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 271,168 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 1 | BBE4DIV | 206,948 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 2 | BBE4DIV | 206,948 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 3 | BBE4DIV | 206,948 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 4 | BBE4DIV | 206,948 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 5 | BBE4DIV | 206,948 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 6 | BBE4DIV | 232,636 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,258 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 1 | BBE4DIV | 4,788 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 2 | BBE4DIV | 4,788 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 3 | BBE4DIV | 4,788 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 4 | BBE4DIV | 4,788 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 5 | BBE4DIV | 4,788 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 6 | BBE4DIV | 5,776 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 914,204 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 1 | BBE4MUL | 704,900 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 2 | BBE4MUL | 704,900 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 3 | BBE4MUL | 704,900 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 4 | BBE4MUL | 704,900 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 5 | BBE4MUL | 704,900 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 6 | BBE4MUL | 791,768 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 37,658 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 1 | BBE4MUL | 25,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 2 | BBE4MUL | 25,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 3 | BBE4MUL | 25,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 4 | BBE4MUL | 25,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 5 | BBE4MUL | 25,460 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 6 | BBE4MUL | 30,704 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 120,574 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 1 | FE4SUB | 114,760 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 2 | FE4SUB | 114,760 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 3 | FE4SUB | 114,760 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 4 | FE4SUB | 114,760 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 5 | FE4SUB | 114,760 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 6 | FE4SUB | 117,420 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 2,574,600 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 1,858,500 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 1,858,500 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 1,858,500 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 1,858,500 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 1,858,500 | 
| leaf | FriReducedOpeningAir | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 2,169,300 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-ExtractPublicValuesCommit | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 1 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 2 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 3 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 4 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 5 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpenedValues | 6 | PHANTOM | 4,032 | 
| leaf | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 1 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 2 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 3 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 4 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 5 | PHANTOM | 4,044 | 
| leaf | PhantomAir | CT-HintOpeningProof | 6 | PHANTOM | 4,044 | 
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
| leaf | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 53,424 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 1 | PHANTOM | 38,304 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 2 | PHANTOM | 38,304 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 3 | PHANTOM | 38,304 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 4 | PHANTOM | 38,304 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 5 | PHANTOM | 38,304 | 
| leaf | PhantomAir | CT-exp-reverse-bits-len | 6 | PHANTOM | 44,352 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 1 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 2 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 3 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 4 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 5 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-pre-compute-alpha-pows | 6 | PHANTOM | 12 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 75,096 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 1 | PHANTOM | 54,936 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 2 | PHANTOM | 54,936 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 3 | PHANTOM | 54,936 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 4 | PHANTOM | 54,936 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 5 | PHANTOM | 54,936 | 
| leaf | PhantomAir | CT-single-reduced-opening-eval | 6 | PHANTOM | 63,000 | 
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
| leaf | PhantomAir | HintBitsF | 0 | PHANTOM | 894 | 
| leaf | PhantomAir | HintBitsF | 1 | PHANTOM | 714 | 
| leaf | PhantomAir | HintBitsF | 2 | PHANTOM | 714 | 
| leaf | PhantomAir | HintBitsF | 3 | PHANTOM | 714 | 
| leaf | PhantomAir | HintBitsF | 4 | PHANTOM | 714 | 
| leaf | PhantomAir | HintBitsF | 5 | PHANTOM | 714 | 
| leaf | PhantomAir | HintBitsF | 6 | PHANTOM | 786 | 
| leaf | PhantomAir | HintInputVec | 0 | PHANTOM | 49,200 | 
| leaf | PhantomAir | HintInputVec | 1 | PHANTOM | 40,950 | 
| leaf | PhantomAir | HintInputVec | 2 | PHANTOM | 40,950 | 
| leaf | PhantomAir | HintInputVec | 3 | PHANTOM | 40,950 | 
| leaf | PhantomAir | HintInputVec | 4 | PHANTOM | 40,950 | 
| leaf | PhantomAir | HintInputVec | 5 | PHANTOM | 40,950 | 
| leaf | PhantomAir | HintInputVec | 6 | PHANTOM | 45,600 | 
| leaf | PhantomAir | HintLoad | 0 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 1 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 2 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 3 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 4 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 5 | PHANTOM | 7,056 | 
| leaf | PhantomAir | HintLoad | 6 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 0 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 1 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 2 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 3 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 4 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 5 | PHANTOM | 7,056 | 
| leaf | PhantomAir | PrintV | 6 | PHANTOM | 7,056 | 
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
| leaf | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 5,831,784 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 1 | VERIFY_BATCH | 4,977,126 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 2 | VERIFY_BATCH | 4,977,126 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 3 | VERIFY_BATCH | 4,977,126 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 4 | VERIFY_BATCH | 4,977,126 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 5 | VERIFY_BATCH | 4,977,126 | 
| leaf | VerifyBatchAir | VerifyBatchFelt | 6 | VERIFY_BATCH | 5,362,560 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> |  | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFFI | 0 | ADD | 43,848 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEFI | 0 | ADD | 28,420 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddEI | 0 | ADD | 1,069,056 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddF | 0 | ADD | 144,130 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddFI | 0 | ADD | 362,703 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddV | 0 | ADD | 745,648 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | AddVI | 0 | ADD | 1,330,520 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | ADD | 1,223,858 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | Alloc | 0 | MUL | 352,089 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | CastFV | 0 | ADD | 4,350 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivEIN | 0 | ADD | 22,852 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivF | 0 | DIV | 168,084 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | DivFIN | 0 | DIV | 11,919 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmE | 0 | ADD | 119,828 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmF | 0 | ADD | 169,012 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ImmV | 0 | ADD | 158,079 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | ADD | 317,898 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadE | 0 | MUL | 317,898 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | ADD | 398,808 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadF | 0 | MUL | 265,437 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadHeapPtr | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | ADD | 1,347,253 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | LoadV | 0 | MUL | 1,202,746 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEF | 0 | MUL | 148,016 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEFI | 0 | MUL | 20,532 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulEI | 0 | ADD | 143,492 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulF | 0 | MUL | 808,926 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulFI | 0 | MUL | 128,151 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | MulVI | 0 | MUL | 291,711 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | NegE | 0 | MUL | 8,468 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | ADD | 266,742 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreE | 0 | MUL | 266,742 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | ADD | 22,446 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreF | 0 | MUL | 11,136 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreHeapPtr | 0 | ADD | 29 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | ADD | 407,450 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | StoreV | 0 | MUL | 273,905 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | ADD | 500,250 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEF | 0 | SUB | 166,750 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEFI | 0 | ADD | 11,368 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubEI | 0 | ADD | 45,704 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubF | 0 | SUB | 232 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubFI | 0 | SUB | 127,658 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubV | 0 | SUB | 186,847 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVI | 0 | SUB | 30,276 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | SubVIN | 0 | SUB | 25,578 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | UnsafeCastVF | 0 | ADD | 3,364 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | ZipFor | 0 | ADD | 4,745,821 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqE | 0 | BNE | 5,428 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqEI | 0 | BNE | 92 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqF | 0 | BNE | 104,880 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqFI | 0 | BNE | 115 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqV | 0 | BNE | 32,269 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertEqVI | 0 | BNE | 5,520 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | AssertNonZero | 0 | BEQ | 23 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEq | 0 | BNE | 3,312 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfEqI | 0 | BNE | 495,167 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNe | 0 | BEQ | 3,381 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | IfNeI | 0 | BEQ | 2,139 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | ZipFor | 0 | BNE | 3,178,347 | 
| root | <JalNativeAdapterAir,JalCoreAir> |  | 0 | JAL | 9 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfEqI | 0 | JAL | 63,837 | 
| root | <JalNativeAdapterAir,JalCoreAir> | IfNe | 0 | JAL | 27 | 
| root | <JalNativeAdapterAir,JalCoreAir> | ZipFor | 0 | JAL | 135,045 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | Publish | 0 | PUBLISH | 1,104 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadF | 0 | LOADW | 625,130 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | LoadV | 0 | LOADW | 2,709,960 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreF | 0 | STOREW | 152,614 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreHintWord | 0 | HINT_STOREW | 1,874,950 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | StoreV | 0 | STOREW | 636,790 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | LoadE | 0 | LOADW | 809,906 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | StoreE | 0 | STOREW | 456,971 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | AddE | 0 | FE4ADD | 523,792 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivE | 0 | BBE4DIV | 251,864 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | DivEIN | 0 | BBE4DIV | 7,486 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulE | 0 | BBE4MUL | 1,059,136 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | MulEI | 0 | BBE4MUL | 47,006 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | SubE | 0 | FE4SUB | 134,520 | 
| root | FriReducedOpeningAir | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 2,811,900 | 
| root | PhantomAir | CT-ExtractPublicValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-HintOpenedValues | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-HintOpeningProof | 0 | PHANTOM | 3,036 | 
| root | PhantomAir | CT-HintOpeningValues | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-InitializePcsConst | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-ReadProofsFromInput | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-VerifyProofs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-cache-generator-powers | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-compute-reduced-opening | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-exp-reverse-bits-len | 0 | PHANTOM | 49,896 | 
| root | PhantomAir | CT-pre-compute-alpha-pows | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-single-reduced-opening-eval | 0 | PHANTOM | 68,544 | 
| root | PhantomAir | CT-stage-c-build-rounds | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verifier-verify | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-d-verify-pcs | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-stage-e-verify-constraints | 0 | PHANTOM | 12 | 
| root | PhantomAir | CT-verify-batch | 0 | PHANTOM | 3,024 | 
| root | PhantomAir | CT-verify-batch-ext | 0 | PHANTOM | 10,584 | 
| root | PhantomAir | CT-verify-query | 0 | PHANTOM | 504 | 
| root | PhantomAir | HintBitsF | 0 | PHANTOM | 852 | 
| root | PhantomAir | HintInputVec | 0 | PHANTOM | 46,956 | 
| root | PhantomAir | HintLoad | 0 | PHANTOM | 6,804 | 
| root | PhantomAir | PrintV | 0 | PHANTOM | 6,804 | 
| root | VerifyBatchAir | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 4,788 | 
| root | VerifyBatchAir | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 21,546 | 
| root | VerifyBatchAir | VerifyBatchExt | 0 | VERIFY_BATCH | 4,926,852 | 
| root | VerifyBatchAir | VerifyBatchFelt | 0 | VERIFY_BATCH | 5,295,528 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 6 | 32,725,728 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 6 | 11,211,555 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 6 | 2,626,130 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 6 | 2,626,130 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 6 | 1,818,090 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 6 | 28 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 6 | 40 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 6 | 80 | 

| group | air_name | idx | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | AccessAdapterAir<2> | 0 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 1 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 2 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.0 | AccessAdapterAir<2> | 3 | 262,144 |  | 12 | 11 | 6,029,312 | 
| internal.0 | AccessAdapterAir<4> | 0 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 1 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 2 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.0 | AccessAdapterAir<4> | 3 | 131,072 |  | 12 | 13 | 3,276,800 | 
| internal.0 | AccessAdapterAir<8> | 0 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 1 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 2 | 512 |  | 12 | 17 | 14,848 | 
| internal.0 | AccessAdapterAir<8> | 3 | 256 |  | 12 | 17 | 7,424 | 
| internal.0 | FriReducedOpeningAir | 0 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 1 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 2 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.0 | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 25 | 7,995,392 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.0 | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 160 | 399 | 18,317,312 | 
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
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.0 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.0 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.0 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 32,768 |  | 12 | 9 | 688,128 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 524,288 |  | 24 | 22 | 24,117,248 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.0 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 24 | 31 | 3,604,480 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.0 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 38 | 3,801,088 | 
| internal.0 | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| internal.0 | VolatileBoundaryAir | 0 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 1 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 2 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.0 | VolatileBoundaryAir | 3 | 262,144 |  | 8 | 11 | 4,980,736 | 
| internal.1 | AccessAdapterAir<2> | 4 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<2> | 5 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.1 | AccessAdapterAir<4> | 4 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<4> | 5 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.1 | AccessAdapterAir<8> | 4 | 512 |  | 12 | 17 | 14,848 | 
| internal.1 | AccessAdapterAir<8> | 5 | 512 |  | 12 | 17 | 14,848 | 
| internal.1 | FriReducedOpeningAir | 4 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.1 | FriReducedOpeningAir | 5 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.1 | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.1 | PhantomAir | 4 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.1 | PhantomAir | 5 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.1 | ProgramAir | 4 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | ProgramAir | 5 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.1 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.1 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.1 | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| internal.1 | VolatileBoundaryAir | 4 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.1 | VolatileBoundaryAir | 5 | 524,288 |  | 8 | 11 | 9,961,472 | 
| internal.2 | AccessAdapterAir<2> | 6 | 524,288 |  | 12 | 11 | 12,058,624 | 
| internal.2 | AccessAdapterAir<4> | 6 | 262,144 |  | 12 | 13 | 6,553,600 | 
| internal.2 | AccessAdapterAir<8> | 6 | 512 |  | 12 | 17 | 14,848 | 
| internal.2 | FriReducedOpeningAir | 6 | 262,144 |  | 36 | 25 | 15,990,784 | 
| internal.2 | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 65,536 |  | 160 | 399 | 36,634,624 | 
| internal.2 | PhantomAir | 6 | 131,072 |  | 8 | 6 | 1,835,008 | 
| internal.2 | ProgramAir | 6 | 262,144 |  | 8 | 10 | 4,718,592 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| internal.2 | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 2,097,152 |  | 20 | 29 | 102,760,448 | 
| internal.2 | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 524,288 |  | 16 | 23 | 20,447,232 | 
| internal.2 | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 65,536 |  | 12 | 9 | 1,376,256 | 
| internal.2 | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 1,048,576 |  | 24 | 22 | 48,234,496 | 
| internal.2 | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 131,072 |  | 24 | 31 | 7,208,960 | 
| internal.2 | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 131,072 |  | 20 | 38 | 7,602,176 | 
| internal.2 | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| internal.2 | VolatileBoundaryAir | 6 | 524,288 |  | 8 | 11 | 9,961,472 | 
| leaf | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 1 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 2 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 3 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 4 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 5 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<2> | 6 | 262,144 |  | 12 | 11 | 6,029,312 | 
| leaf | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 1 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 2 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 3 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 4 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 5 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<4> | 6 | 131,072 |  | 12 | 13 | 3,276,800 | 
| leaf | AccessAdapterAir<8> | 0 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 1 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 2 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 3 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 4 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 5 | 256 |  | 12 | 17 | 7,424 | 
| leaf | AccessAdapterAir<8> | 6 | 512 |  | 12 | 17 | 14,848 | 
| leaf | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 1 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 2 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 3 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 4 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 5 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | FriReducedOpeningAir | 6 | 131,072 |  | 36 | 25 | 7,995,392 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 1 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 2 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 3 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 4 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 5 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | NativePoseidon2Air<BabyBearParameters>, 1> | 6 | 32,768 |  | 160 | 399 | 18,317,312 | 
| leaf | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| leaf | PhantomAir | 1 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 2 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 3 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 4 | 32,768 |  | 8 | 6 | 458,752 | 
| leaf | PhantomAir | 5 | 32,768 |  | 8 | 6 | 458,752 | 
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
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 1 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 2 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 3 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 4 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 5 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 6 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 1 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 2 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 3 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 4 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 5 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 6 | 262,144 |  | 16 | 23 | 10,223,616 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 1 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 2 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 3 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 4 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 5 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 6 | 32,768 |  | 12 | 9 | 688,128 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 1 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 2 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 3 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 4 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 5 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 6 | 64 |  | 16 | 23 | 2,496 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 1 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 2 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 3 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 4 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 5 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 6 | 262,144 |  | 24 | 22 | 12,058,624 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 1 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 2 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 3 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 4 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 5 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 6 | 65,536 |  | 24 | 31 | 3,604,480 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 1 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 2 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 3 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 4 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 5 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 6 | 65,536 |  | 20 | 38 | 3,801,088 | 
| leaf | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 
| leaf | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 1 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 2 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 3 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 4 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 5 | 262,144 |  | 8 | 11 | 4,980,736 | 
| leaf | VolatileBoundaryAir | 6 | 262,144 |  | 8 | 11 | 4,980,736 | 
| root | AccessAdapterAir<2> | 0 | 262,144 |  | 12 | 11 | 6,029,312 | 
| root | AccessAdapterAir<4> | 0 | 131,072 |  | 12 | 13 | 3,276,800 | 
| root | AccessAdapterAir<8> | 0 | 256 |  | 12 | 17 | 7,424 | 
| root | FriReducedOpeningAir | 0 | 131,072 |  | 36 | 25 | 7,995,392 | 
| root | NativePoseidon2Air<BabyBearParameters>, 1> | 0 | 32,768 |  | 160 | 399 | 18,317,312 | 
| root | PhantomAir | 0 | 65,536 |  | 8 | 6 | 917,504 | 
| root | ProgramAir | 0 | 262,144 |  | 8 | 10 | 4,718,592 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| root | VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir> | 0 | 1,048,576 |  | 20 | 29 | 51,380,224 | 
| root | VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1> | 0 | 262,144 |  | 16 | 23 | 10,223,616 | 
| root | VmAirWrapper<JalNativeAdapterAir, JalCoreAir> | 0 | 32,768 |  | 12 | 9 | 688,128 | 
| root | VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir> | 0 | 64 |  | 16 | 23 | 2,496 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<1>, NativeLoadStoreCoreAir<1> | 0 | 524,288 |  | 24 | 22 | 24,117,248 | 
| root | VmAirWrapper<NativeLoadStoreAdapterAir<4>, NativeLoadStoreCoreAir<4> | 0 | 65,536 |  | 24 | 31 | 3,604,480 | 
| root | VmAirWrapper<NativeVectorizedAdapterAir<4>, FieldExtensionCoreAir> | 0 | 65,536 |  | 20 | 38 | 3,801,088 | 
| root | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| root | VolatileBoundaryAir | 0 | 262,144 |  | 8 | 11 | 4,980,736 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | AccessAdapterAir<8> | 0 | 32 |  | 12 | 17 | 928 | 
| fib_e2e | AccessAdapterAir<8> | 1 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 2 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 3 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 4 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 5 | 16 |  | 12 | 17 | 464 | 
| fib_e2e | AccessAdapterAir<8> | 6 | 32 |  | 12 | 17 | 928 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fib_e2e | MemoryMerkleAir<8> | 0 | 256 |  | 12 | 32 | 11,264 | 
| fib_e2e | MemoryMerkleAir<8> | 1 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 2 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 3 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 4 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 5 | 128 |  | 12 | 32 | 5,632 | 
| fib_e2e | MemoryMerkleAir<8> | 6 | 256 |  | 12 | 32 | 11,264 | 
| fib_e2e | PersistentBoundaryAir<8> | 0 | 32 |  | 8 | 20 | 896 | 
| fib_e2e | PersistentBoundaryAir<8> | 1 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 2 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 3 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 4 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 5 | 16 |  | 8 | 20 | 448 | 
| fib_e2e | PersistentBoundaryAir<8> | 6 | 32 |  | 8 | 20 | 896 | 
| fib_e2e | PhantomAir | 0 | 2 |  | 8 | 6 | 28 | 
| fib_e2e | PhantomAir | 1 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 2 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 3 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 4 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 5 | 1 |  | 8 | 6 | 14 | 
| fib_e2e | PhantomAir | 6 | 1 |  | 8 | 6 | 14 | 
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
| fib_e2e | Rv32HintStoreAir | 0 | 4 |  | 24 | 32 | 224 | 
| fib_e2e | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 1 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 2 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 3 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 4 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 5 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VariableRangeCheckerAir | 6 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 1 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 3 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 4 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 5 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 6 | 1,048,576 |  | 28 | 36 | 67,108,864 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 1 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 3 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 4 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 5 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 6 | 524,288 |  | 24 | 37 | 31,981,568 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 1 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 3 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 4 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 5 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 6 | 262,144 |  | 16 | 26 | 11,010,048 | 
| fib_e2e | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 4 |  | 20 | 32 | 208 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 1 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 3 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 4 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 5 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 6 | 131,072 |  | 16 | 18 | 4,456,448 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 8 |  | 20 | 28 | 384 | 
| fib_e2e | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 6 | 1 |  | 20 | 28 | 48 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 16 |  | 28 | 40 | 1,088 | 
| fib_e2e | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 6 | 4 |  | 28 | 40 | 272 | 
| fib_e2e | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 8 |  | 16 | 21 | 296 | 
| fib_e2e | VmConnectorAir | 0 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 1 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 2 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 3 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 4 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 5 | 2 | 1 | 8 | 4 | 24 | 
| fib_e2e | VmConnectorAir | 6 | 2 | 1 | 8 | 4 | 24 | 

| group | cell_tracker_span | simple_advice_cells | lookup_advice_cells |
| --- | --- | --- | --- |
| halo2_outer | VerifierProgram | 763,177 | 207,155 | 
| halo2_outer | VerifierProgram;PoseidonCell | 20,120 |  | 
| halo2_outer | VerifierProgram;stage-c-build-rounds | 334,839 | 697 | 
| halo2_outer | VerifierProgram;stage-c-build-rounds;PoseidonCell | 47,785 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs | 161 | 40 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify | 561,377 | 2,728 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;PoseidonCell | 67,905 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;cache-generator-powers | 585,396 | 99,036 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;compute-reduced-opening;single-reduced-opening-eval | 11,889,822 | 434,532 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;pre-compute-alpha-pows | 76,596 | 11,168 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch | 119,028 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;PoseidonCell | 14,110,236 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-batch;verify-batch-reduce-fast;PoseidonCell | 11,910,192 | 334,404 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query | 1,468,404 | 238,980 | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext | 251,160 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;PoseidonCell | 24,401,160 |  | 
| halo2_outer | VerifierProgram;stage-d-verify-pcs;stage-d-verifier-verify;verify-query;verify-batch-ext;verify-batch-reduce-fast;PoseidonCell | 2,443,560 | 68,544 | 
| halo2_outer | VerifierProgram;stage-e-verify-constraints | 3,271,829 | 548,741 | 

| group | chip_name | idx | rows_used |
| --- | --- | --- | --- |
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 1,252,575 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 1 | 1,252,517 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 2 | 1,252,517 | 
| internal.0 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 3 | 626,549 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 326,999 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 326,945 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 326,945 | 
| internal.0 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 163,524 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 0 | 42,529 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 1 | 43,404 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 2 | 42,920 | 
| internal.0 | <JalNativeAdapterAir,JalCoreAir> | 3 | 21,495 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 52 | 
| internal.0 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 52 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 540,452 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 540,452 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 540,452 | 
| internal.0 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 270,384 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 80,892 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 80,892 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 80,892 | 
| internal.0 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 40,446 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 104,864 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 104,814 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 104,814 | 
| internal.0 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 52,457 | 
| internal.0 | AccessAdapter<2> | 0 | 353,876 | 
| internal.0 | AccessAdapter<2> | 1 | 353,876 | 
| internal.0 | AccessAdapter<2> | 2 | 353,876 | 
| internal.0 | AccessAdapter<2> | 3 | 196,060 | 
| internal.0 | AccessAdapter<4> | 0 | 166,946 | 
| internal.0 | AccessAdapter<4> | 1 | 166,946 | 
| internal.0 | AccessAdapter<4> | 2 | 166,946 | 
| internal.0 | AccessAdapter<4> | 3 | 93,034 | 
| internal.0 | AccessAdapter<8> | 0 | 312 | 
| internal.0 | AccessAdapter<8> | 1 | 312 | 
| internal.0 | AccessAdapter<8> | 2 | 312 | 
| internal.0 | AccessAdapter<8> | 3 | 156 | 
| internal.0 | Boundary | 0 | 290,513 | 
| internal.0 | Boundary | 1 | 290,513 | 
| internal.0 | Boundary | 2 | 290,513 | 
| internal.0 | Boundary | 3 | 195,119 | 
| internal.0 | FriReducedOpeningAir | 0 | 224,952 | 
| internal.0 | FriReducedOpeningAir | 1 | 224,952 | 
| internal.0 | FriReducedOpeningAir | 2 | 224,952 | 
| internal.0 | FriReducedOpeningAir | 3 | 112,476 | 
| internal.0 | PhantomAir | 0 | 68,137 | 
| internal.0 | PhantomAir | 1 | 68,137 | 
| internal.0 | PhantomAir | 2 | 68,137 | 
| internal.0 | PhantomAir | 3 | 34,076 | 
| internal.0 | ProgramChip | 0 | 137,082 | 
| internal.0 | ProgramChip | 1 | 137,082 | 
| internal.0 | ProgramChip | 2 | 137,082 | 
| internal.0 | ProgramChip | 3 | 137,082 | 
| internal.0 | VariableRangeCheckerAir | 0 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 1 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 2 | 262,144 | 
| internal.0 | VariableRangeCheckerAir | 3 | 262,144 | 
| internal.0 | VerifyBatchAir | 0 | 48,948 | 
| internal.0 | VerifyBatchAir | 1 | 48,822 | 
| internal.0 | VerifyBatchAir | 2 | 48,822 | 
| internal.0 | VerifyBatchAir | 3 | 24,411 | 
| internal.0 | VmConnectorAir | 0 | 2 | 
| internal.0 | VmConnectorAir | 1 | 2 | 
| internal.0 | VmConnectorAir | 2 | 2 | 
| internal.0 | VmConnectorAir | 3 | 2 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 4 | 1,271,639 | 
| internal.1 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 5 | 1,262,449 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 333,051 | 
| internal.1 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 330,111 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 4 | 44,057 | 
| internal.1 | <JalNativeAdapterAir,JalCoreAir> | 5 | 43,745 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 52 | 
| internal.1 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 52 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 544,460 | 
| internal.1 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 542,536 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 81,734 | 
| internal.1 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 81,313 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 106,516 | 
| internal.1 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 105,764 | 
| internal.1 | AccessAdapter<2> | 4 | 358,148 | 
| internal.1 | AccessAdapter<2> | 5 | 365,006 | 
| internal.1 | AccessAdapter<4> | 4 | 168,998 | 
| internal.1 | AccessAdapter<4> | 5 | 173,100 | 
| internal.1 | AccessAdapter<8> | 4 | 328 | 
| internal.1 | AccessAdapter<8> | 5 | 324 | 
| internal.1 | Boundary | 4 | 292,637 | 
| internal.1 | Boundary | 5 | 322,897 | 
| internal.1 | FriReducedOpeningAir | 4 | 224,952 | 
| internal.1 | FriReducedOpeningAir | 5 | 224,952 | 
| internal.1 | PhantomAir | 4 | 68,727 | 
| internal.1 | PhantomAir | 5 | 68,432 | 
| internal.1 | ProgramChip | 4 | 137,082 | 
| internal.1 | ProgramChip | 5 | 137,082 | 
| internal.1 | VariableRangeCheckerAir | 4 | 262,144 | 
| internal.1 | VariableRangeCheckerAir | 5 | 262,144 | 
| internal.1 | VerifyBatchAir | 4 | 51,348 | 
| internal.1 | VerifyBatchAir | 5 | 50,213 | 
| internal.1 | VmConnectorAir | 4 | 2 | 
| internal.1 | VmConnectorAir | 5 | 2 | 
| internal.2 | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 6 | 1,271,639 | 
| internal.2 | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 333,051 | 
| internal.2 | <JalNativeAdapterAir,JalCoreAir> | 6 | 44,102 | 
| internal.2 | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 52 | 
| internal.2 | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 544,460 | 
| internal.2 | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 81,734 | 
| internal.2 | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 106,516 | 
| internal.2 | AccessAdapter<2> | 6 | 358,148 | 
| internal.2 | AccessAdapter<4> | 6 | 168,998 | 
| internal.2 | AccessAdapter<8> | 6 | 328 | 
| internal.2 | Boundary | 6 | 292,637 | 
| internal.2 | FriReducedOpeningAir | 6 | 224,952 | 
| internal.2 | PhantomAir | 6 | 68,727 | 
| internal.2 | ProgramChip | 6 | 137,082 | 
| internal.2 | VariableRangeCheckerAir | 6 | 262,144 | 
| internal.2 | VerifyBatchAir | 6 | 51,348 | 
| internal.2 | VmConnectorAir | 6 | 2 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 660,752 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 1 | 539,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 2 | 539,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 3 | 539,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 4 | 539,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 5 | 539,576 | 
| leaf | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 6 | 591,735 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 165,908 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 1 | 136,701 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 2 | 136,701 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 3 | 136,701 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 4 | 136,701 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 5 | 136,701 | 
| leaf | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 6 | 149,396 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 0 | 25,023 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 1 | 22,087 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 2 | 21,986 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 3 | 22,079 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 4 | 21,955 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 5 | 21,792 | 
| leaf | <JalNativeAdapterAir,JalCoreAir> | 6 | 23,394 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 1 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 2 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 3 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 4 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 5 | 36 | 
| leaf | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 6 | 36 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 283,640 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 1 | 221,570 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 2 | 221,570 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 3 | 221,570 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 4 | 221,570 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 5 | 221,570 | 
| leaf | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 6 | 248,958 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 42,221 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 1 | 33,539 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 2 | 33,539 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 3 | 33,539 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 4 | 33,539 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 5 | 33,539 | 
| leaf | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 6 | 37,037 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 48,393 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 1 | 37,434 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 2 | 37,434 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 3 | 37,434 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 4 | 37,434 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 5 | 37,434 | 
| leaf | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 6 | 42,029 | 
| leaf | AccessAdapter<2> | 0 | 172,454 | 
| leaf | AccessAdapter<2> | 1 | 139,220 | 
| leaf | AccessAdapter<2> | 2 | 139,220 | 
| leaf | AccessAdapter<2> | 3 | 139,220 | 
| leaf | AccessAdapter<2> | 4 | 139,220 | 
| leaf | AccessAdapter<2> | 5 | 139,220 | 
| leaf | AccessAdapter<2> | 6 | 154,046 | 
| leaf | AccessAdapter<4> | 0 | 80,936 | 
| leaf | AccessAdapter<4> | 1 | 65,580 | 
| leaf | AccessAdapter<4> | 2 | 65,580 | 
| leaf | AccessAdapter<4> | 3 | 65,580 | 
| leaf | AccessAdapter<4> | 4 | 65,580 | 
| leaf | AccessAdapter<4> | 5 | 65,580 | 
| leaf | AccessAdapter<4> | 6 | 72,488 | 
| leaf | AccessAdapter<8> | 0 | 160 | 
| leaf | AccessAdapter<8> | 1 | 154 | 
| leaf | AccessAdapter<8> | 2 | 154 | 
| leaf | AccessAdapter<8> | 3 | 154 | 
| leaf | AccessAdapter<8> | 4 | 154 | 
| leaf | AccessAdapter<8> | 5 | 154 | 
| leaf | AccessAdapter<8> | 6 | 318 | 
| leaf | Boundary | 0 | 161,960 | 
| leaf | Boundary | 1 | 135,645 | 
| leaf | Boundary | 2 | 135,645 | 
| leaf | Boundary | 3 | 135,645 | 
| leaf | Boundary | 4 | 135,645 | 
| leaf | Boundary | 5 | 135,645 | 
| leaf | Boundary | 6 | 147,244 | 
| leaf | FriReducedOpeningAir | 0 | 102,984 | 
| leaf | FriReducedOpeningAir | 1 | 74,340 | 
| leaf | FriReducedOpeningAir | 2 | 74,340 | 
| leaf | FriReducedOpeningAir | 3 | 74,340 | 
| leaf | FriReducedOpeningAir | 4 | 74,340 | 
| leaf | FriReducedOpeningAir | 5 | 74,340 | 
| leaf | FriReducedOpeningAir | 6 | 86,772 | 
| leaf | PhantomAir | 0 | 37,267 | 
| leaf | PhantomAir | 1 | 29,982 | 
| leaf | PhantomAir | 2 | 29,982 | 
| leaf | PhantomAir | 3 | 29,982 | 
| leaf | PhantomAir | 4 | 29,982 | 
| leaf | PhantomAir | 5 | 29,982 | 
| leaf | PhantomAir | 6 | 33,121 | 
| leaf | ProgramChip | 0 | 88,955 | 
| leaf | ProgramChip | 1 | 88,955 | 
| leaf | ProgramChip | 2 | 88,955 | 
| leaf | ProgramChip | 3 | 88,955 | 
| leaf | ProgramChip | 4 | 88,955 | 
| leaf | ProgramChip | 5 | 88,955 | 
| leaf | ProgramChip | 6 | 88,955 | 
| leaf | VariableRangeCheckerAir | 0 | 262,144 | 
| leaf | VariableRangeCheckerAir | 1 | 262,144 | 
| leaf | VariableRangeCheckerAir | 2 | 262,144 | 
| leaf | VariableRangeCheckerAir | 3 | 262,144 | 
| leaf | VariableRangeCheckerAir | 4 | 262,144 | 
| leaf | VariableRangeCheckerAir | 5 | 262,144 | 
| leaf | VariableRangeCheckerAir | 6 | 262,144 | 
| leaf | VerifyBatchAir | 0 | 26,007 | 
| leaf | VerifyBatchAir | 1 | 23,862 | 
| leaf | VerifyBatchAir | 2 | 23,862 | 
| leaf | VerifyBatchAir | 3 | 23,862 | 
| leaf | VerifyBatchAir | 4 | 23,862 | 
| leaf | VerifyBatchAir | 5 | 23,862 | 
| leaf | VerifyBatchAir | 6 | 24,856 | 
| leaf | VmConnectorAir | 0 | 2 | 
| leaf | VmConnectorAir | 1 | 2 | 
| leaf | VmConnectorAir | 2 | 2 | 
| leaf | VmConnectorAir | 3 | 2 | 
| leaf | VmConnectorAir | 4 | 2 | 
| leaf | VmConnectorAir | 5 | 2 | 
| leaf | VmConnectorAir | 6 | 2 | 
| root | <AluNativeAdapterAir,FieldArithmeticCoreAir> | 0 | 636,073 | 
| root | <BranchNativeAdapterAir,BranchEqualCoreAir<1>> | 0 | 166,551 | 
| root | <JalNativeAdapterAir,JalCoreAir> | 0 | 22,102 | 
| root | <NativeAdapterAir<2, 0>,PublicValuesCoreAir> | 0 | 48 | 
| root | <NativeLoadStoreAdapterAir<1>,NativeLoadStoreCoreAir<1>> | 0 | 272,702 | 
| root | <NativeLoadStoreAdapterAir<4>,NativeLoadStoreCoreAir<4>> | 0 | 40,867 | 
| root | <NativeVectorizedAdapterAir<4>,FieldExtensionCoreAir> | 0 | 53,258 | 
| root | AccessAdapter<2> | 0 | 198,348 | 
| root | AccessAdapter<4> | 0 | 94,136 | 
| root | AccessAdapter<8> | 0 | 230 | 
| root | Boundary | 0 | 196,573 | 
| root | FriReducedOpeningAir | 0 | 112,476 | 
| root | PhantomAir | 0 | 34,366 | 
| root | ProgramChip | 0 | 137,355 | 
| root | VariableRangeCheckerAir | 0 | 262,144 | 
| root | VerifyBatchAir | 0 | 25,686 | 
| root | VmConnectorAir | 0 | 2 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 1,048,488 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 1 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 2 | 1,048,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 3 | 1,048,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 4 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 5 | 1,048,502 | 
| fib_e2e | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 6 | 909,048 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 349,485 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 1 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 2 | 349,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 3 | 349,501 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 4 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 5 | 349,500 | 
| fib_e2e | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 6 | 303,015 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 232,992 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 1 | 233,001 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 2 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 3 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 4 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 5 | 233,000 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 6 | 202,011 | 
| fib_e2e | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 4 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 116,501 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 1 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 2 | 116,501 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 3 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 4 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 5 | 116,500 | 
| fib_e2e | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 6 | 101,005 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 8 | 
| fib_e2e | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 6 | 1 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 12 | 
| fib_e2e | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 6 | 3 | 
| fib_e2e | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 7 | 
| fib_e2e | AccessAdapter<8> | 0 | 28 | 
| fib_e2e | AccessAdapter<8> | 1 | 14 | 
| fib_e2e | AccessAdapter<8> | 2 | 14 | 
| fib_e2e | AccessAdapter<8> | 3 | 14 | 
| fib_e2e | AccessAdapter<8> | 4 | 14 | 
| fib_e2e | AccessAdapter<8> | 5 | 14 | 
| fib_e2e | AccessAdapter<8> | 6 | 20 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 0 | 146 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 1 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 2 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 3 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 4 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 5 | 81 | 
| fib_e2e | Arc<BabyBearParameters>, 1> | 6 | 168 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 1 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 2 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 3 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 4 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 5 | 65,536 | 
| fib_e2e | BitwiseOperationLookupAir<8> | 6 | 65,536 | 
| fib_e2e | Boundary | 0 | 28 | 
| fib_e2e | Boundary | 1 | 14 | 
| fib_e2e | Boundary | 2 | 14 | 
| fib_e2e | Boundary | 3 | 14 | 
| fib_e2e | Boundary | 4 | 14 | 
| fib_e2e | Boundary | 5 | 14 | 
| fib_e2e | Boundary | 6 | 20 | 
| fib_e2e | Merkle | 0 | 172 | 
| fib_e2e | Merkle | 1 | 72 | 
| fib_e2e | Merkle | 2 | 72 | 
| fib_e2e | Merkle | 3 | 72 | 
| fib_e2e | Merkle | 4 | 72 | 
| fib_e2e | Merkle | 5 | 72 | 
| fib_e2e | Merkle | 6 | 178 | 
| fib_e2e | PhantomAir | 0 | 2 | 
| fib_e2e | ProgramChip | 0 | 3,241 | 
| fib_e2e | ProgramChip | 1 | 3,241 | 
| fib_e2e | ProgramChip | 2 | 3,241 | 
| fib_e2e | ProgramChip | 3 | 3,241 | 
| fib_e2e | ProgramChip | 4 | 3,241 | 
| fib_e2e | ProgramChip | 5 | 3,241 | 
| fib_e2e | ProgramChip | 6 | 3,241 | 
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
| internal.0 | AddE | 0 | FE4ADD | 27,362 | 
| internal.0 | AddE | 1 | FE4ADD | 27,362 | 
| internal.0 | AddE | 2 | FE4ADD | 27,362 | 
| internal.0 | AddE | 3 | FE4ADD | 13,681 | 
| internal.0 | AddEFFI | 0 | ADD | 2,864 | 
| internal.0 | AddEFFI | 1 | ADD | 2,864 | 
| internal.0 | AddEFFI | 2 | ADD | 2,864 | 
| internal.0 | AddEFFI | 3 | ADD | 1,432 | 
| internal.0 | AddEFI | 0 | ADD | 1,960 | 
| internal.0 | AddEFI | 1 | ADD | 1,960 | 
| internal.0 | AddEFI | 2 | ADD | 1,960 | 
| internal.0 | AddEFI | 3 | ADD | 980 | 
| internal.0 | AddEI | 0 | ADD | 72,576 | 
| internal.0 | AddEI | 1 | ADD | 72,576 | 
| internal.0 | AddEI | 2 | ADD | 72,576 | 
| internal.0 | AddEI | 3 | ADD | 36,288 | 
| internal.0 | AddF | 0 | ADD | 9,940 | 
| internal.0 | AddF | 1 | ADD | 9,940 | 
| internal.0 | AddF | 2 | ADD | 9,940 | 
| internal.0 | AddF | 3 | ADD | 4,970 | 
| internal.0 | AddFI | 0 | ADD | 23,917 | 
| internal.0 | AddFI | 1 | ADD | 23,917 | 
| internal.0 | AddFI | 2 | ADD | 23,917 | 
| internal.0 | AddFI | 3 | ADD | 11,971 | 
| internal.0 | AddV | 0 | ADD | 49,182 | 
| internal.0 | AddV | 1 | ADD | 49,178 | 
| internal.0 | AddV | 2 | ADD | 49,178 | 
| internal.0 | AddV | 3 | ADD | 24,598 | 
| internal.0 | AddVI | 0 | ADD | 89,846 | 
| internal.0 | AddVI | 1 | ADD | 89,846 | 
| internal.0 | AddVI | 2 | ADD | 89,846 | 
| internal.0 | AddVI | 3 | ADD | 44,954 | 
| internal.0 | Alloc | 0 | ADD | 82,678 | 
| internal.0 | Alloc | 0 | MUL | 23,752 | 
| internal.0 | Alloc | 1 | ADD | 82,678 | 
| internal.0 | Alloc | 1 | MUL | 23,752 | 
| internal.0 | Alloc | 2 | ADD | 82,678 | 
| internal.0 | Alloc | 2 | MUL | 23,752 | 
| internal.0 | Alloc | 3 | ADD | 41,362 | 
| internal.0 | Alloc | 3 | MUL | 11,883 | 
| internal.0 | AssertEqE | 0 | BNE | 472 | 
| internal.0 | AssertEqE | 1 | BNE | 472 | 
| internal.0 | AssertEqE | 2 | BNE | 472 | 
| internal.0 | AssertEqE | 3 | BNE | 236 | 
| internal.0 | AssertEqEI | 0 | BNE | 8 | 
| internal.0 | AssertEqEI | 1 | BNE | 8 | 
| internal.0 | AssertEqEI | 2 | BNE | 8 | 
| internal.0 | AssertEqEI | 3 | BNE | 4 | 
| internal.0 | AssertEqF | 0 | BNE | 9,113 | 
| internal.0 | AssertEqF | 1 | BNE | 9,113 | 
| internal.0 | AssertEqF | 2 | BNE | 9,113 | 
| internal.0 | AssertEqF | 3 | BNE | 4,544 | 
| internal.0 | AssertEqFI | 0 | BNE | 7 | 
| internal.0 | AssertEqFI | 1 | BNE | 7 | 
| internal.0 | AssertEqFI | 2 | BNE | 7 | 
| internal.0 | AssertEqFI | 3 | BNE | 3 | 
| internal.0 | AssertEqV | 0 | BNE | 2,722 | 
| internal.0 | AssertEqV | 1 | BNE | 2,722 | 
| internal.0 | AssertEqV | 2 | BNE | 2,722 | 
| internal.0 | AssertEqV | 3 | BNE | 1,361 | 
| internal.0 | AssertEqVI | 0 | BNE | 478 | 
| internal.0 | AssertEqVI | 1 | BNE | 478 | 
| internal.0 | AssertEqVI | 2 | BNE | 478 | 
| internal.0 | AssertEqVI | 3 | BNE | 239 | 
| internal.0 | AssertNonZero | 0 | BEQ | 1 | 
| internal.0 | AssertNonZero | 1 | BEQ | 1 | 
| internal.0 | AssertNonZero | 2 | BEQ | 1 | 
| internal.0 | AssertNonZero | 3 | BEQ | 1 | 
| internal.0 | CT-HintOpenedValues | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-HintOpenedValues | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-HintOpenedValues | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-HintOpenedValues | 3 | PHANTOM | 504 | 
| internal.0 | CT-HintOpeningProof | 0 | PHANTOM | 1,012 | 
| internal.0 | CT-HintOpeningProof | 1 | PHANTOM | 1,012 | 
| internal.0 | CT-HintOpeningProof | 2 | PHANTOM | 1,012 | 
| internal.0 | CT-HintOpeningProof | 3 | PHANTOM | 506 | 
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
| internal.0 | CT-cache-generator-powers | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-cache-generator-powers | 3 | PHANTOM | 504 | 
| internal.0 | CT-compute-reduced-opening | 0 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 1 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 2 | PHANTOM | 1,008 | 
| internal.0 | CT-compute-reduced-opening | 3 | PHANTOM | 504 | 
| internal.0 | CT-exp-reverse-bits-len | 0 | PHANTOM | 16,632 | 
| internal.0 | CT-exp-reverse-bits-len | 1 | PHANTOM | 16,632 | 
| internal.0 | CT-exp-reverse-bits-len | 2 | PHANTOM | 16,632 | 
| internal.0 | CT-exp-reverse-bits-len | 3 | PHANTOM | 8,316 | 
| internal.0 | CT-pre-compute-alpha-pows | 0 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 1 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 2 | PHANTOM | 4 | 
| internal.0 | CT-pre-compute-alpha-pows | 3 | PHANTOM | 2 | 
| internal.0 | CT-single-reduced-opening-eval | 0 | PHANTOM | 22,848 | 
| internal.0 | CT-single-reduced-opening-eval | 1 | PHANTOM | 22,848 | 
| internal.0 | CT-single-reduced-opening-eval | 2 | PHANTOM | 22,848 | 
| internal.0 | CT-single-reduced-opening-eval | 3 | PHANTOM | 11,424 | 
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
| internal.0 | CastFV | 0 | ADD | 300 | 
| internal.0 | CastFV | 1 | ADD | 300 | 
| internal.0 | CastFV | 2 | ADD | 300 | 
| internal.0 | CastFV | 3 | ADD | 150 | 
| internal.0 | DivE | 0 | BBE4DIV | 13,172 | 
| internal.0 | DivE | 1 | BBE4DIV | 13,172 | 
| internal.0 | DivE | 2 | BBE4DIV | 13,172 | 
| internal.0 | DivE | 3 | BBE4DIV | 6,586 | 
| internal.0 | DivEIN | 0 | ADD | 1,576 | 
| internal.0 | DivEIN | 0 | BBE4DIV | 394 | 
| internal.0 | DivEIN | 1 | ADD | 1,576 | 
| internal.0 | DivEIN | 1 | BBE4DIV | 394 | 
| internal.0 | DivEIN | 2 | ADD | 1,576 | 
| internal.0 | DivEIN | 2 | BBE4DIV | 394 | 
| internal.0 | DivEIN | 3 | ADD | 788 | 
| internal.0 | DivEIN | 3 | BBE4DIV | 197 | 
| internal.0 | DivF | 0 | DIV | 11,088 | 
| internal.0 | DivF | 1 | DIV | 11,088 | 
| internal.0 | DivF | 2 | DIV | 11,088 | 
| internal.0 | DivF | 3 | DIV | 5,544 | 
| internal.0 | DivFIN | 0 | DIV | 822 | 
| internal.0 | DivFIN | 1 | DIV | 822 | 
| internal.0 | DivFIN | 2 | DIV | 822 | 
| internal.0 | DivFIN | 3 | DIV | 411 | 
| internal.0 | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 11,424 | 
| internal.0 | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 11,424 | 
| internal.0 | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 11,424 | 
| internal.0 | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 5,712 | 
| internal.0 | HintBitsF | 0 | PHANTOM | 284 | 
| internal.0 | HintBitsF | 1 | PHANTOM | 284 | 
| internal.0 | HintBitsF | 2 | PHANTOM | 284 | 
| internal.0 | HintBitsF | 3 | PHANTOM | 142 | 
| internal.0 | HintInputVec | 0 | PHANTOM | 15,403 | 
| internal.0 | HintInputVec | 1 | PHANTOM | 15,403 | 
| internal.0 | HintInputVec | 2 | PHANTOM | 15,403 | 
| internal.0 | HintInputVec | 3 | PHANTOM | 7,706 | 
| internal.0 | HintLoad | 0 | PHANTOM | 2,184 | 
| internal.0 | HintLoad | 1 | PHANTOM | 2,184 | 
| internal.0 | HintLoad | 2 | PHANTOM | 2,184 | 
| internal.0 | HintLoad | 3 | PHANTOM | 1,092 | 
| internal.0 | IfEq | 0 | BNE | 280 | 
| internal.0 | IfEq | 1 | BNE | 280 | 
| internal.0 | IfEq | 2 | BNE | 280 | 
| internal.0 | IfEq | 3 | BNE | 140 | 
| internal.0 | IfEqI | 0 | BNE | 41,834 | 
| internal.0 | IfEqI | 0 | JAL | 12,847 | 
| internal.0 | IfEqI | 1 | BNE | 41,834 | 
| internal.0 | IfEqI | 1 | JAL | 13,722 | 
| internal.0 | IfEqI | 2 | BNE | 41,834 | 
| internal.0 | IfEqI | 2 | JAL | 13,238 | 
| internal.0 | IfEqI | 3 | BNE | 20,917 | 
| internal.0 | IfEqI | 3 | JAL | 6,648 | 
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
| internal.0 | ImmE | 0 | ADD | 8,136 | 
| internal.0 | ImmE | 1 | ADD | 8,136 | 
| internal.0 | ImmE | 2 | ADD | 8,136 | 
| internal.0 | ImmE | 3 | ADD | 4,068 | 
| internal.0 | ImmF | 0 | ADD | 11,470 | 
| internal.0 | ImmF | 1 | ADD | 11,470 | 
| internal.0 | ImmF | 2 | ADD | 11,470 | 
| internal.0 | ImmF | 3 | ADD | 5,819 | 
| internal.0 | ImmV | 0 | ADD | 10,457 | 
| internal.0 | ImmV | 1 | ADD | 10,457 | 
| internal.0 | ImmV | 2 | ADD | 10,457 | 
| internal.0 | ImmV | 3 | ADD | 5,287 | 
| internal.0 | LoadE | 0 | ADD | 21,672 | 
| internal.0 | LoadE | 0 | LOADW | 51,748 | 
| internal.0 | LoadE | 0 | MUL | 21,672 | 
| internal.0 | LoadE | 1 | ADD | 21,672 | 
| internal.0 | LoadE | 1 | LOADW | 51,748 | 
| internal.0 | LoadE | 1 | MUL | 21,672 | 
| internal.0 | LoadE | 2 | ADD | 21,672 | 
| internal.0 | LoadE | 2 | LOADW | 51,748 | 
| internal.0 | LoadE | 2 | MUL | 21,672 | 
| internal.0 | LoadE | 3 | ADD | 10,836 | 
| internal.0 | LoadE | 3 | LOADW | 25,874 | 
| internal.0 | LoadE | 3 | MUL | 10,836 | 
| internal.0 | LoadF | 0 | ADD | 27,496 | 
| internal.0 | LoadF | 0 | LOADW | 56,362 | 
| internal.0 | LoadF | 0 | MUL | 18,306 | 
| internal.0 | LoadF | 1 | ADD | 27,496 | 
| internal.0 | LoadF | 1 | LOADW | 56,362 | 
| internal.0 | LoadF | 1 | MUL | 18,306 | 
| internal.0 | LoadF | 2 | ADD | 27,496 | 
| internal.0 | LoadF | 2 | LOADW | 56,362 | 
| internal.0 | LoadF | 2 | MUL | 18,306 | 
| internal.0 | LoadF | 3 | ADD | 13,748 | 
| internal.0 | LoadF | 3 | LOADW | 28,185 | 
| internal.0 | LoadF | 3 | MUL | 9,153 | 
| internal.0 | LoadHeapPtr | 0 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 1 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 2 | ADD | 2 | 
| internal.0 | LoadHeapPtr | 3 | ADD | 1 | 
| internal.0 | LoadV | 0 | ADD | 92,494 | 
| internal.0 | LoadV | 0 | LOADW | 244,342 | 
| internal.0 | LoadV | 0 | MUL | 82,612 | 
| internal.0 | LoadV | 1 | ADD | 92,494 | 
| internal.0 | LoadV | 1 | LOADW | 244,342 | 
| internal.0 | LoadV | 1 | MUL | 82,612 | 
| internal.0 | LoadV | 2 | ADD | 92,494 | 
| internal.0 | LoadV | 2 | LOADW | 244,342 | 
| internal.0 | LoadV | 2 | MUL | 82,612 | 
| internal.0 | LoadV | 3 | ADD | 46,247 | 
| internal.0 | LoadV | 3 | LOADW | 122,176 | 
| internal.0 | LoadV | 3 | MUL | 41,306 | 
| internal.0 | MulE | 0 | BBE4MUL | 54,642 | 
| internal.0 | MulE | 1 | BBE4MUL | 54,592 | 
| internal.0 | MulE | 2 | BBE4MUL | 54,592 | 
| internal.0 | MulE | 3 | BBE4MUL | 27,346 | 
| internal.0 | MulEF | 0 | MUL | 9,872 | 
| internal.0 | MulEF | 1 | MUL | 9,872 | 
| internal.0 | MulEF | 2 | MUL | 9,872 | 
| internal.0 | MulEF | 3 | MUL | 4,936 | 
| internal.0 | MulEFI | 0 | MUL | 1,416 | 
| internal.0 | MulEFI | 1 | MUL | 1,416 | 
| internal.0 | MulEFI | 2 | MUL | 1,416 | 
| internal.0 | MulEFI | 3 | MUL | 708 | 
| internal.0 | MulEI | 0 | ADD | 9,864 | 
| internal.0 | MulEI | 0 | BBE4MUL | 2,466 | 
| internal.0 | MulEI | 1 | ADD | 9,864 | 
| internal.0 | MulEI | 1 | BBE4MUL | 2,466 | 
| internal.0 | MulEI | 2 | ADD | 9,864 | 
| internal.0 | MulEI | 2 | BBE4MUL | 2,466 | 
| internal.0 | MulEI | 3 | ADD | 4,932 | 
| internal.0 | MulEI | 3 | BBE4MUL | 1,233 | 
| internal.0 | MulF | 0 | MUL | 53,772 | 
| internal.0 | MulF | 1 | MUL | 53,772 | 
| internal.0 | MulF | 2 | MUL | 53,772 | 
| internal.0 | MulF | 3 | MUL | 26,886 | 
| internal.0 | MulFI | 0 | MUL | 8,838 | 
| internal.0 | MulFI | 1 | MUL | 8,838 | 
| internal.0 | MulFI | 2 | MUL | 8,838 | 
| internal.0 | MulFI | 3 | MUL | 4,419 | 
| internal.0 | MulVI | 0 | MUL | 20,033 | 
| internal.0 | MulVI | 1 | MUL | 20,033 | 
| internal.0 | MulVI | 2 | MUL | 20,033 | 
| internal.0 | MulVI | 3 | MUL | 10,017 | 
| internal.0 | NegE | 0 | MUL | 584 | 
| internal.0 | NegE | 1 | MUL | 584 | 
| internal.0 | NegE | 2 | MUL | 584 | 
| internal.0 | NegE | 3 | MUL | 292 | 
| internal.0 | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 102 | 
| internal.0 | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 51 | 
| internal.0 | PrintV | 0 | PHANTOM | 2,184 | 
| internal.0 | PrintV | 1 | PHANTOM | 2,184 | 
| internal.0 | PrintV | 2 | PHANTOM | 2,184 | 
| internal.0 | PrintV | 3 | PHANTOM | 1,092 | 
| internal.0 | Publish | 0 | PUBLISH | 52 | 
| internal.0 | Publish | 1 | PUBLISH | 52 | 
| internal.0 | Publish | 2 | PUBLISH | 52 | 
| internal.0 | Publish | 3 | PUBLISH | 52 | 
| internal.0 | StoreE | 0 | ADD | 18,312 | 
| internal.0 | StoreE | 0 | MUL | 18,312 | 
| internal.0 | StoreE | 0 | STOREW | 29,144 | 
| internal.0 | StoreE | 1 | ADD | 18,312 | 
| internal.0 | StoreE | 1 | MUL | 18,312 | 
| internal.0 | StoreE | 1 | STOREW | 29,144 | 
| internal.0 | StoreE | 2 | ADD | 18,312 | 
| internal.0 | StoreE | 2 | MUL | 18,312 | 
| internal.0 | StoreE | 2 | STOREW | 29,144 | 
| internal.0 | StoreE | 3 | ADD | 9,156 | 
| internal.0 | StoreE | 3 | MUL | 9,156 | 
| internal.0 | StoreE | 3 | STOREW | 14,572 | 
| internal.0 | StoreF | 0 | ADD | 1,500 | 
| internal.0 | StoreF | 0 | MUL | 768 | 
| internal.0 | StoreF | 0 | STOREW | 12,802 | 
| internal.0 | StoreF | 1 | ADD | 1,500 | 
| internal.0 | StoreF | 1 | MUL | 768 | 
| internal.0 | StoreF | 1 | STOREW | 12,802 | 
| internal.0 | StoreF | 2 | ADD | 1,500 | 
| internal.0 | StoreF | 2 | MUL | 768 | 
| internal.0 | StoreF | 2 | STOREW | 12,802 | 
| internal.0 | StoreF | 3 | ADD | 750 | 
| internal.0 | StoreF | 3 | MUL | 384 | 
| internal.0 | StoreF | 3 | STOREW | 6,485 | 
| internal.0 | StoreHeapPtr | 0 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 1 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 2 | ADD | 2 | 
| internal.0 | StoreHeapPtr | 3 | ADD | 1 | 
| internal.0 | StoreHintWord | 0 | HINT_STOREW | 169,508 | 
| internal.0 | StoreHintWord | 1 | HINT_STOREW | 169,508 | 
| internal.0 | StoreHintWord | 2 | HINT_STOREW | 169,508 | 
| internal.0 | StoreHintWord | 3 | HINT_STOREW | 84,763 | 
| internal.0 | StoreV | 0 | ADD | 28,184 | 
| internal.0 | StoreV | 0 | MUL | 18,890 | 
| internal.0 | StoreV | 0 | STOREW | 57,438 | 
| internal.0 | StoreV | 1 | ADD | 28,184 | 
| internal.0 | StoreV | 1 | MUL | 18,890 | 
| internal.0 | StoreV | 1 | STOREW | 57,438 | 
| internal.0 | StoreV | 2 | ADD | 28,184 | 
| internal.0 | StoreV | 2 | MUL | 18,890 | 
| internal.0 | StoreV | 2 | STOREW | 57,438 | 
| internal.0 | StoreV | 3 | ADD | 14,092 | 
| internal.0 | StoreV | 3 | MUL | 9,445 | 
| internal.0 | StoreV | 3 | STOREW | 28,775 | 
| internal.0 | SubE | 0 | FE4SUB | 6,828 | 
| internal.0 | SubE | 1 | FE4SUB | 6,828 | 
| internal.0 | SubE | 2 | FE4SUB | 6,828 | 
| internal.0 | SubE | 3 | FE4SUB | 3,414 | 
| internal.0 | SubEF | 0 | ADD | 34,500 | 
| internal.0 | SubEF | 0 | SUB | 11,500 | 
| internal.0 | SubEF | 1 | ADD | 34,500 | 
| internal.0 | SubEF | 1 | SUB | 11,500 | 
| internal.0 | SubEF | 2 | ADD | 34,500 | 
| internal.0 | SubEF | 2 | SUB | 11,500 | 
| internal.0 | SubEF | 3 | ADD | 17,250 | 
| internal.0 | SubEF | 3 | SUB | 5,750 | 
| internal.0 | SubEFI | 0 | ADD | 784 | 
| internal.0 | SubEFI | 1 | ADD | 784 | 
| internal.0 | SubEFI | 2 | ADD | 784 | 
| internal.0 | SubEFI | 3 | ADD | 392 | 
| internal.0 | SubEI | 0 | ADD | 3,152 | 
| internal.0 | SubEI | 1 | ADD | 3,152 | 
| internal.0 | SubEI | 2 | ADD | 3,152 | 
| internal.0 | SubEI | 3 | ADD | 1,576 | 
| internal.0 | SubF | 0 | SUB | 16 | 
| internal.0 | SubF | 1 | SUB | 16 | 
| internal.0 | SubF | 2 | SUB | 16 | 
| internal.0 | SubF | 3 | SUB | 8 | 
| internal.0 | SubFI | 0 | SUB | 8,804 | 
| internal.0 | SubFI | 1 | SUB | 8,804 | 
| internal.0 | SubFI | 2 | SUB | 8,804 | 
| internal.0 | SubFI | 3 | SUB | 4,402 | 
| internal.0 | SubV | 0 | SUB | 12,718 | 
| internal.0 | SubV | 1 | SUB | 12,718 | 
| internal.0 | SubV | 2 | SUB | 12,718 | 
| internal.0 | SubV | 3 | SUB | 6,359 | 
| internal.0 | SubVI | 0 | SUB | 1,996 | 
| internal.0 | SubVI | 1 | SUB | 1,996 | 
| internal.0 | SubVI | 2 | SUB | 1,996 | 
| internal.0 | SubVI | 3 | SUB | 998 | 
| internal.0 | SubVIN | 0 | SUB | 1,680 | 
| internal.0 | SubVIN | 1 | SUB | 1,680 | 
| internal.0 | SubVIN | 2 | SUB | 1,680 | 
| internal.0 | SubVIN | 3 | SUB | 840 | 
| internal.0 | UnsafeCastVF | 0 | ADD | 232 | 
| internal.0 | UnsafeCastVF | 1 | ADD | 232 | 
| internal.0 | UnsafeCastVF | 2 | ADD | 232 | 
| internal.0 | UnsafeCastVF | 3 | ADD | 116 | 
| internal.0 | VerifyBatchExt | 0 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 1 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 2 | VERIFY_BATCH | 1,680 | 
| internal.0 | VerifyBatchExt | 3 | VERIFY_BATCH | 840 | 
| internal.0 | VerifyBatchFelt | 0 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 1 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 2 | VERIFY_BATCH | 504 | 
| internal.0 | VerifyBatchFelt | 3 | VERIFY_BATCH | 252 | 
| internal.0 | ZipFor | 0 | ADD | 322,026 | 
| internal.0 | ZipFor | 0 | BNE | 271,612 | 
| internal.0 | ZipFor | 0 | JAL | 29,675 | 
| internal.0 | ZipFor | 1 | ADD | 321,972 | 
| internal.0 | ZipFor | 1 | BNE | 271,558 | 
| internal.0 | ZipFor | 1 | JAL | 29,675 | 
| internal.0 | ZipFor | 2 | ADD | 321,972 | 
| internal.0 | ZipFor | 2 | BNE | 271,558 | 
| internal.0 | ZipFor | 2 | JAL | 29,675 | 
| internal.0 | ZipFor | 3 | ADD | 161,050 | 
| internal.0 | ZipFor | 3 | BNE | 135,843 | 
| internal.0 | ZipFor | 3 | JAL | 14,843 | 
| internal.1 |  | 4 | ADD | 2 | 
| internal.1 |  | 4 | JAL | 1 | 
| internal.1 |  | 5 | ADD | 2 | 
| internal.1 |  | 5 | JAL | 1 | 
| internal.1 | AddE | 4 | FE4ADD | 27,568 | 
| internal.1 | AddE | 5 | FE4ADD | 27,481 | 
| internal.1 | AddEFFI | 4 | ADD | 3,024 | 
| internal.1 | AddEFFI | 5 | ADD | 3,008 | 
| internal.1 | AddEFI | 4 | ADD | 1,960 | 
| internal.1 | AddEFI | 5 | ADD | 1,960 | 
| internal.1 | AddEI | 4 | ADD | 73,728 | 
| internal.1 | AddEI | 5 | ADD | 73,216 | 
| internal.1 | AddF | 4 | ADD | 9,940 | 
| internal.1 | AddF | 5 | ADD | 9,940 | 
| internal.1 | AddFI | 4 | ADD | 24,989 | 
| internal.1 | AddFI | 5 | ADD | 24,477 | 
| internal.1 | AddV | 4 | ADD | 51,428 | 
| internal.1 | AddV | 5 | ADD | 50,309 | 
| internal.1 | AddVI | 4 | ADD | 91,720 | 
| internal.1 | AddVI | 5 | ADD | 90,815 | 
| internal.1 | Alloc | 4 | ADD | 84,366 | 
| internal.1 | Alloc | 4 | MUL | 24,258 | 
| internal.1 | Alloc | 5 | ADD | 83,522 | 
| internal.1 | Alloc | 5 | MUL | 24,005 | 
| internal.1 | AssertEqE | 4 | BNE | 472 | 
| internal.1 | AssertEqE | 5 | BNE | 472 | 
| internal.1 | AssertEqEI | 4 | BNE | 8 | 
| internal.1 | AssertEqEI | 5 | BNE | 8 | 
| internal.1 | AssertEqF | 4 | BNE | 9,129 | 
| internal.1 | AssertEqF | 5 | BNE | 9,129 | 
| internal.1 | AssertEqFI | 4 | BNE | 7 | 
| internal.1 | AssertEqFI | 5 | BNE | 7 | 
| internal.1 | AssertEqV | 4 | BNE | 2,806 | 
| internal.1 | AssertEqV | 5 | BNE | 2,764 | 
| internal.1 | AssertEqVI | 4 | BNE | 478 | 
| internal.1 | AssertEqVI | 5 | BNE | 478 | 
| internal.1 | AssertNonZero | 4 | BEQ | 1 | 
| internal.1 | AssertNonZero | 5 | BEQ | 1 | 
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
| internal.1 | CT-cache-generator-powers | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-cache-generator-powers | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 4 | PHANTOM | 1,008 | 
| internal.1 | CT-compute-reduced-opening | 5 | PHANTOM | 1,008 | 
| internal.1 | CT-exp-reverse-bits-len | 4 | PHANTOM | 16,632 | 
| internal.1 | CT-exp-reverse-bits-len | 5 | PHANTOM | 16,632 | 
| internal.1 | CT-pre-compute-alpha-pows | 4 | PHANTOM | 4 | 
| internal.1 | CT-pre-compute-alpha-pows | 5 | PHANTOM | 4 | 
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
| internal.1 | CastFV | 4 | ADD | 300 | 
| internal.1 | CastFV | 5 | ADD | 300 | 
| internal.1 | DivE | 4 | BBE4DIV | 13,256 | 
| internal.1 | DivE | 5 | BBE4DIV | 13,214 | 
| internal.1 | DivEIN | 4 | ADD | 1,576 | 
| internal.1 | DivEIN | 4 | BBE4DIV | 394 | 
| internal.1 | DivEIN | 5 | ADD | 1,576 | 
| internal.1 | DivEIN | 5 | BBE4DIV | 394 | 
| internal.1 | DivF | 4 | DIV | 11,592 | 
| internal.1 | DivF | 5 | DIV | 11,340 | 
| internal.1 | DivFIN | 4 | DIV | 822 | 
| internal.1 | DivFIN | 5 | DIV | 822 | 
| internal.1 | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 11,424 | 
| internal.1 | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 11,424 | 
| internal.1 | HintBitsF | 4 | PHANTOM | 284 | 
| internal.1 | HintBitsF | 5 | PHANTOM | 284 | 
| internal.1 | HintInputVec | 4 | PHANTOM | 15,657 | 
| internal.1 | HintInputVec | 5 | PHANTOM | 15,530 | 
| internal.1 | HintLoad | 4 | PHANTOM | 2,268 | 
| internal.1 | HintLoad | 5 | PHANTOM | 2,226 | 
| internal.1 | IfEq | 4 | BNE | 288 | 
| internal.1 | IfEq | 5 | BNE | 284 | 
| internal.1 | IfEqI | 4 | BNE | 43,058 | 
| internal.1 | IfEqI | 4 | JAL | 14,037 | 
| internal.1 | IfEqI | 5 | BNE | 42,462 | 
| internal.1 | IfEqI | 5 | JAL | 13,894 | 
| internal.1 | IfNe | 4 | BEQ | 294 | 
| internal.1 | IfNe | 4 | JAL | 6 | 
| internal.1 | IfNe | 5 | BEQ | 290 | 
| internal.1 | IfNe | 5 | JAL | 6 | 
| internal.1 | IfNeI | 4 | BEQ | 186 | 
| internal.1 | IfNeI | 5 | BEQ | 186 | 
| internal.1 | ImmE | 4 | ADD | 8,264 | 
| internal.1 | ImmE | 5 | ADD | 8,264 | 
| internal.1 | ImmF | 4 | ADD | 11,470 | 
| internal.1 | ImmF | 5 | ADD | 11,470 | 
| internal.1 | ImmV | 4 | ADD | 10,761 | 
| internal.1 | ImmV | 5 | ADD | 10,621 | 
| internal.1 | LoadE | 4 | ADD | 21,924 | 
| internal.1 | LoadE | 4 | LOADW | 52,252 | 
| internal.1 | LoadE | 4 | MUL | 21,924 | 
| internal.1 | LoadE | 5 | ADD | 21,798 | 
| internal.1 | LoadE | 5 | LOADW | 52,000 | 
| internal.1 | LoadE | 5 | MUL | 21,798 | 
| internal.1 | LoadF | 4 | ADD | 27,504 | 
| internal.1 | LoadF | 4 | LOADW | 56,566 | 
| internal.1 | LoadF | 4 | MUL | 18,306 | 
| internal.1 | LoadF | 5 | ADD | 27,500 | 
| internal.1 | LoadF | 5 | LOADW | 56,512 | 
| internal.1 | LoadF | 5 | MUL | 18,306 | 
| internal.1 | LoadHeapPtr | 4 | ADD | 2 | 
| internal.1 | LoadHeapPtr | 5 | ADD | 2 | 
| internal.1 | LoadV | 4 | ADD | 92,914 | 
| internal.1 | LoadV | 4 | LOADW | 246,364 | 
| internal.1 | LoadV | 4 | MUL | 82,948 | 
| internal.1 | LoadV | 5 | ADD | 92,704 | 
| internal.1 | LoadV | 5 | LOADW | 245,353 | 
| internal.1 | LoadV | 5 | MUL | 82,780 | 
| internal.1 | MulE | 4 | BBE4MUL | 55,744 | 
| internal.1 | MulE | 5 | BBE4MUL | 55,251 | 
| internal.1 | MulEF | 4 | MUL | 10,208 | 
| internal.1 | MulEF | 5 | MUL | 10,040 | 
| internal.1 | MulEFI | 4 | MUL | 1,416 | 
| internal.1 | MulEFI | 5 | MUL | 1,416 | 
| internal.1 | MulEI | 4 | ADD | 9,896 | 
| internal.1 | MulEI | 4 | BBE4MUL | 2,474 | 
| internal.1 | MulEI | 5 | ADD | 9,880 | 
| internal.1 | MulEI | 5 | BBE4MUL | 2,470 | 
| internal.1 | MulF | 4 | MUL | 55,788 | 
| internal.1 | MulF | 5 | MUL | 54,780 | 
| internal.1 | MulFI | 4 | MUL | 8,838 | 
| internal.1 | MulFI | 5 | MUL | 8,838 | 
| internal.1 | MulVI | 4 | MUL | 20,117 | 
| internal.1 | MulVI | 5 | MUL | 20,075 | 
| internal.1 | NegE | 4 | MUL | 584 | 
| internal.1 | NegE | 5 | MUL | 584 | 
| internal.1 | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 108 | 
| internal.1 | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 107 | 
| internal.1 | PrintV | 4 | PHANTOM | 2,268 | 
| internal.1 | PrintV | 5 | PHANTOM | 2,226 | 
| internal.1 | Publish | 4 | PUBLISH | 52 | 
| internal.1 | Publish | 5 | PUBLISH | 52 | 
| internal.1 | StoreE | 4 | ADD | 18,396 | 
| internal.1 | StoreE | 4 | MUL | 18,396 | 
| internal.1 | StoreE | 4 | STOREW | 29,482 | 
| internal.1 | StoreE | 5 | ADD | 18,354 | 
| internal.1 | StoreE | 5 | MUL | 18,354 | 
| internal.1 | StoreE | 5 | STOREW | 29,313 | 
| internal.1 | StoreF | 4 | ADD | 1,548 | 
| internal.1 | StoreF | 4 | MUL | 768 | 
| internal.1 | StoreF | 4 | STOREW | 13,354 | 
| internal.1 | StoreF | 5 | ADD | 1,540 | 
| internal.1 | StoreF | 5 | MUL | 768 | 
| internal.1 | StoreF | 5 | STOREW | 13,094 | 
| internal.1 | StoreHeapPtr | 4 | ADD | 2 | 
| internal.1 | StoreHeapPtr | 5 | ADD | 2 | 
| internal.1 | StoreHintWord | 4 | HINT_STOREW | 170,398 | 
| internal.1 | StoreHintWord | 5 | HINT_STOREW | 169,969 | 
| internal.1 | StoreV | 4 | ADD | 28,100 | 
| internal.1 | StoreV | 4 | MUL | 18,890 | 
| internal.1 | StoreV | 4 | STOREW | 57,778 | 
| internal.1 | StoreV | 5 | ADD | 28,142 | 
| internal.1 | StoreV | 5 | MUL | 18,890 | 
| internal.1 | StoreV | 5 | STOREW | 57,608 | 
| internal.1 | SubE | 4 | FE4SUB | 7,080 | 
| internal.1 | SubE | 5 | FE4SUB | 6,954 | 
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
| internal.1 | SubFI | 4 | SUB | 8,804 | 
| internal.1 | SubFI | 5 | SUB | 8,804 | 
| internal.1 | SubV | 4 | SUB | 12,886 | 
| internal.1 | SubV | 5 | SUB | 12,802 | 
| internal.1 | SubVI | 4 | SUB | 2,088 | 
| internal.1 | SubVI | 5 | SUB | 2,042 | 
| internal.1 | SubVIN | 4 | SUB | 1,764 | 
| internal.1 | SubVIN | 5 | SUB | 1,722 | 
| internal.1 | UnsafeCastVF | 4 | ADD | 232 | 
| internal.1 | UnsafeCastVF | 5 | ADD | 232 | 
| internal.1 | VerifyBatchExt | 4 | VERIFY_BATCH | 1,764 | 
| internal.1 | VerifyBatchExt | 5 | VERIFY_BATCH | 1,722 | 
| internal.1 | VerifyBatchFelt | 4 | VERIFY_BATCH | 504 | 
| internal.1 | VerifyBatchFelt | 5 | VERIFY_BATCH | 504 | 
| internal.1 | ZipFor | 4 | ADD | 327,244 | 
| internal.1 | ZipFor | 4 | BNE | 276,324 | 
| internal.1 | ZipFor | 4 | JAL | 30,013 | 
| internal.1 | ZipFor | 5 | ADD | 324,697 | 
| internal.1 | ZipFor | 5 | BNE | 274,030 | 
| internal.1 | ZipFor | 5 | JAL | 29,844 | 
| internal.2 |  | 6 | ADD | 2 | 
| internal.2 |  | 6 | JAL | 1 | 
| internal.2 | AddE | 6 | FE4ADD | 27,568 | 
| internal.2 | AddEFFI | 6 | ADD | 3,024 | 
| internal.2 | AddEFI | 6 | ADD | 1,960 | 
| internal.2 | AddEI | 6 | ADD | 73,728 | 
| internal.2 | AddF | 6 | ADD | 9,940 | 
| internal.2 | AddFI | 6 | ADD | 24,989 | 
| internal.2 | AddV | 6 | ADD | 51,428 | 
| internal.2 | AddVI | 6 | ADD | 91,720 | 
| internal.2 | Alloc | 6 | ADD | 84,366 | 
| internal.2 | Alloc | 6 | MUL | 24,258 | 
| internal.2 | AssertEqE | 6 | BNE | 472 | 
| internal.2 | AssertEqEI | 6 | BNE | 8 | 
| internal.2 | AssertEqF | 6 | BNE | 9,129 | 
| internal.2 | AssertEqFI | 6 | BNE | 7 | 
| internal.2 | AssertEqV | 6 | BNE | 2,806 | 
| internal.2 | AssertEqVI | 6 | BNE | 478 | 
| internal.2 | AssertNonZero | 6 | BEQ | 1 | 
| internal.2 | CT-HintOpenedValues | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-HintOpeningProof | 6 | PHANTOM | 1,012 | 
| internal.2 | CT-HintOpeningValues | 6 | PHANTOM | 4 | 
| internal.2 | CT-InitializePcsConst | 6 | PHANTOM | 2 | 
| internal.2 | CT-ReadProofsFromInput | 6 | PHANTOM | 2 | 
| internal.2 | CT-VerifyProofs | 6 | PHANTOM | 2 | 
| internal.2 | CT-cache-generator-powers | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-compute-reduced-opening | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-exp-reverse-bits-len | 6 | PHANTOM | 16,632 | 
| internal.2 | CT-pre-compute-alpha-pows | 6 | PHANTOM | 4 | 
| internal.2 | CT-single-reduced-opening-eval | 6 | PHANTOM | 22,848 | 
| internal.2 | CT-stage-c-build-rounds | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verifier-verify | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-d-verify-pcs | 6 | PHANTOM | 4 | 
| internal.2 | CT-stage-e-verify-constraints | 6 | PHANTOM | 4 | 
| internal.2 | CT-verify-batch | 6 | PHANTOM | 1,008 | 
| internal.2 | CT-verify-batch-ext | 6 | PHANTOM | 3,528 | 
| internal.2 | CT-verify-query | 6 | PHANTOM | 168 | 
| internal.2 | CastFV | 6 | ADD | 300 | 
| internal.2 | DivE | 6 | BBE4DIV | 13,256 | 
| internal.2 | DivEIN | 6 | ADD | 1,576 | 
| internal.2 | DivEIN | 6 | BBE4DIV | 394 | 
| internal.2 | DivF | 6 | DIV | 11,592 | 
| internal.2 | DivFIN | 6 | DIV | 822 | 
| internal.2 | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 11,424 | 
| internal.2 | HintBitsF | 6 | PHANTOM | 284 | 
| internal.2 | HintInputVec | 6 | PHANTOM | 15,657 | 
| internal.2 | HintLoad | 6 | PHANTOM | 2,268 | 
| internal.2 | IfEq | 6 | BNE | 288 | 
| internal.2 | IfEqI | 6 | BNE | 43,058 | 
| internal.2 | IfEqI | 6 | JAL | 14,082 | 
| internal.2 | IfNe | 6 | BEQ | 294 | 
| internal.2 | IfNe | 6 | JAL | 6 | 
| internal.2 | IfNeI | 6 | BEQ | 186 | 
| internal.2 | ImmE | 6 | ADD | 8,264 | 
| internal.2 | ImmF | 6 | ADD | 11,470 | 
| internal.2 | ImmV | 6 | ADD | 10,761 | 
| internal.2 | LoadE | 6 | ADD | 21,924 | 
| internal.2 | LoadE | 6 | LOADW | 52,252 | 
| internal.2 | LoadE | 6 | MUL | 21,924 | 
| internal.2 | LoadF | 6 | ADD | 27,504 | 
| internal.2 | LoadF | 6 | LOADW | 56,566 | 
| internal.2 | LoadF | 6 | MUL | 18,306 | 
| internal.2 | LoadHeapPtr | 6 | ADD | 2 | 
| internal.2 | LoadV | 6 | ADD | 92,914 | 
| internal.2 | LoadV | 6 | LOADW | 246,364 | 
| internal.2 | LoadV | 6 | MUL | 82,948 | 
| internal.2 | MulE | 6 | BBE4MUL | 55,744 | 
| internal.2 | MulEF | 6 | MUL | 10,208 | 
| internal.2 | MulEFI | 6 | MUL | 1,416 | 
| internal.2 | MulEI | 6 | ADD | 9,896 | 
| internal.2 | MulEI | 6 | BBE4MUL | 2,474 | 
| internal.2 | MulF | 6 | MUL | 55,788 | 
| internal.2 | MulFI | 6 | MUL | 8,838 | 
| internal.2 | MulVI | 6 | MUL | 20,117 | 
| internal.2 | NegE | 6 | MUL | 584 | 
| internal.2 | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 108 | 
| internal.2 | PrintV | 6 | PHANTOM | 2,268 | 
| internal.2 | Publish | 6 | PUBLISH | 52 | 
| internal.2 | StoreE | 6 | ADD | 18,396 | 
| internal.2 | StoreE | 6 | MUL | 18,396 | 
| internal.2 | StoreE | 6 | STOREW | 29,482 | 
| internal.2 | StoreF | 6 | ADD | 1,548 | 
| internal.2 | StoreF | 6 | MUL | 768 | 
| internal.2 | StoreF | 6 | STOREW | 13,354 | 
| internal.2 | StoreHeapPtr | 6 | ADD | 2 | 
| internal.2 | StoreHintWord | 6 | HINT_STOREW | 170,398 | 
| internal.2 | StoreV | 6 | ADD | 28,100 | 
| internal.2 | StoreV | 6 | MUL | 18,890 | 
| internal.2 | StoreV | 6 | STOREW | 57,778 | 
| internal.2 | SubE | 6 | FE4SUB | 7,080 | 
| internal.2 | SubEF | 6 | ADD | 34,500 | 
| internal.2 | SubEF | 6 | SUB | 11,500 | 
| internal.2 | SubEFI | 6 | ADD | 784 | 
| internal.2 | SubEI | 6 | ADD | 3,152 | 
| internal.2 | SubF | 6 | SUB | 16 | 
| internal.2 | SubFI | 6 | SUB | 8,804 | 
| internal.2 | SubV | 6 | SUB | 12,886 | 
| internal.2 | SubVI | 6 | SUB | 2,088 | 
| internal.2 | SubVIN | 6 | SUB | 1,764 | 
| internal.2 | UnsafeCastVF | 6 | ADD | 232 | 
| internal.2 | VerifyBatchExt | 6 | VERIFY_BATCH | 1,764 | 
| internal.2 | VerifyBatchFelt | 6 | VERIFY_BATCH | 504 | 
| internal.2 | ZipFor | 6 | ADD | 327,244 | 
| internal.2 | ZipFor | 6 | BNE | 276,324 | 
| internal.2 | ZipFor | 6 | JAL | 30,013 | 
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
| leaf | AddE | 0 | FE4ADD | 12,844 | 
| leaf | AddE | 1 | FE4ADD | 9,622 | 
| leaf | AddE | 2 | FE4ADD | 9,622 | 
| leaf | AddE | 3 | FE4ADD | 9,622 | 
| leaf | AddE | 4 | FE4ADD | 9,622 | 
| leaf | AddE | 5 | FE4ADD | 9,622 | 
| leaf | AddE | 6 | FE4ADD | 11,021 | 
| leaf | AddEFFI | 0 | ADD | 1,256 | 
| leaf | AddEFFI | 1 | ADD | 1,016 | 
| leaf | AddEFFI | 2 | ADD | 1,016 | 
| leaf | AddEFFI | 3 | ADD | 1,016 | 
| leaf | AddEFFI | 4 | ADD | 1,016 | 
| leaf | AddEFFI | 5 | ADD | 1,016 | 
| leaf | AddEFFI | 6 | ADD | 1,112 | 
| leaf | AddEFI | 0 | ADD | 676 | 
| leaf | AddEFI | 1 | ADD | 528 | 
| leaf | AddEFI | 2 | ADD | 528 | 
| leaf | AddEFI | 3 | ADD | 528 | 
| leaf | AddEFI | 4 | ADD | 528 | 
| leaf | AddEFI | 5 | ADD | 528 | 
| leaf | AddEFI | 6 | ADD | 604 | 
| leaf | AddEI | 0 | ADD | 30,080 | 
| leaf | AddEI | 1 | ADD | 24,228 | 
| leaf | AddEI | 2 | ADD | 24,228 | 
| leaf | AddEI | 3 | ADD | 24,228 | 
| leaf | AddEI | 4 | ADD | 24,228 | 
| leaf | AddEI | 5 | ADD | 24,228 | 
| leaf | AddEI | 6 | ADD | 26,952 | 
| leaf | AddF | 0 | ADD | 5,215 | 
| leaf | AddF | 1 | ADD | 4,165 | 
| leaf | AddF | 2 | ADD | 4,165 | 
| leaf | AddF | 3 | ADD | 4,165 | 
| leaf | AddF | 4 | ADD | 4,165 | 
| leaf | AddF | 5 | ADD | 4,165 | 
| leaf | AddF | 6 | ADD | 4,585 | 
| leaf | AddFI | 0 | ADD | 15,724 | 
| leaf | AddFI | 1 | ADD | 15,659 | 
| leaf | AddFI | 2 | ADD | 15,659 | 
| leaf | AddFI | 3 | ADD | 15,659 | 
| leaf | AddFI | 4 | ADD | 15,659 | 
| leaf | AddFI | 5 | ADD | 15,659 | 
| leaf | AddFI | 6 | ADD | 15,693 | 
| leaf | AddV | 0 | ADD | 25,681 | 
| leaf | AddV | 1 | ADD | 22,928 | 
| leaf | AddV | 2 | ADD | 22,928 | 
| leaf | AddV | 3 | ADD | 22,928 | 
| leaf | AddV | 4 | ADD | 22,928 | 
| leaf | AddV | 5 | ADD | 22,928 | 
| leaf | AddV | 6 | ADD | 24,251 | 
| leaf | AddVI | 0 | ADD | 51,350 | 
| leaf | AddVI | 1 | ADD | 44,945 | 
| leaf | AddVI | 2 | ADD | 44,945 | 
| leaf | AddVI | 3 | ADD | 44,945 | 
| leaf | AddVI | 4 | ADD | 44,945 | 
| leaf | AddVI | 5 | ADD | 44,945 | 
| leaf | AddVI | 6 | ADD | 47,733 | 
| leaf | Alloc | 0 | ADD | 44,258 | 
| leaf | Alloc | 0 | MUL | 12,753 | 
| leaf | Alloc | 1 | ADD | 38,528 | 
| leaf | Alloc | 1 | MUL | 11,263 | 
| leaf | Alloc | 2 | ADD | 38,528 | 
| leaf | Alloc | 2 | MUL | 11,263 | 
| leaf | Alloc | 3 | ADD | 38,528 | 
| leaf | Alloc | 3 | MUL | 11,263 | 
| leaf | Alloc | 4 | ADD | 38,528 | 
| leaf | Alloc | 4 | MUL | 11,263 | 
| leaf | Alloc | 5 | ADD | 38,528 | 
| leaf | Alloc | 5 | MUL | 11,263 | 
| leaf | Alloc | 6 | ADD | 41,728 | 
| leaf | Alloc | 6 | MUL | 12,088 | 
| leaf | AssertEqE | 0 | BNE | 244 | 
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
| leaf | AssertEqF | 0 | BNE | 4,768 | 
| leaf | AssertEqF | 1 | BNE | 3,808 | 
| leaf | AssertEqF | 2 | BNE | 3,808 | 
| leaf | AssertEqF | 3 | BNE | 3,808 | 
| leaf | AssertEqF | 4 | BNE | 3,808 | 
| leaf | AssertEqF | 5 | BNE | 3,808 | 
| leaf | AssertEqF | 6 | BNE | 4,200 | 
| leaf | AssertEqV | 0 | BNE | 1,468 | 
| leaf | AssertEqV | 1 | BNE | 1,403 | 
| leaf | AssertEqV | 2 | BNE | 1,403 | 
| leaf | AssertEqV | 3 | BNE | 1,403 | 
| leaf | AssertEqV | 4 | BNE | 1,403 | 
| leaf | AssertEqV | 5 | BNE | 1,403 | 
| leaf | AssertEqV | 6 | BNE | 1,429 | 
| leaf | AssertEqVI | 0 | BNE | 258 | 
| leaf | AssertEqVI | 1 | BNE | 193 | 
| leaf | AssertEqVI | 2 | BNE | 193 | 
| leaf | AssertEqVI | 3 | BNE | 193 | 
| leaf | AssertEqVI | 4 | BNE | 193 | 
| leaf | AssertEqVI | 5 | BNE | 193 | 
| leaf | AssertEqVI | 6 | BNE | 220 | 
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
| leaf | CT-HintOpenedValues | 0 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 1 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 2 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 3 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 4 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 5 | PHANTOM | 672 | 
| leaf | CT-HintOpenedValues | 6 | PHANTOM | 672 | 
| leaf | CT-HintOpeningProof | 0 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 1 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 2 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 3 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 4 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 5 | PHANTOM | 674 | 
| leaf | CT-HintOpeningProof | 6 | PHANTOM | 674 | 
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
| leaf | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,904 | 
| leaf | CT-exp-reverse-bits-len | 1 | PHANTOM | 6,384 | 
| leaf | CT-exp-reverse-bits-len | 2 | PHANTOM | 6,384 | 
| leaf | CT-exp-reverse-bits-len | 3 | PHANTOM | 6,384 | 
| leaf | CT-exp-reverse-bits-len | 4 | PHANTOM | 6,384 | 
| leaf | CT-exp-reverse-bits-len | 5 | PHANTOM | 6,384 | 
| leaf | CT-exp-reverse-bits-len | 6 | PHANTOM | 7,392 | 
| leaf | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 1 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 2 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 3 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 4 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 5 | PHANTOM | 2 | 
| leaf | CT-pre-compute-alpha-pows | 6 | PHANTOM | 2 | 
| leaf | CT-single-reduced-opening-eval | 0 | PHANTOM | 12,516 | 
| leaf | CT-single-reduced-opening-eval | 1 | PHANTOM | 9,156 | 
| leaf | CT-single-reduced-opening-eval | 2 | PHANTOM | 9,156 | 
| leaf | CT-single-reduced-opening-eval | 3 | PHANTOM | 9,156 | 
| leaf | CT-single-reduced-opening-eval | 4 | PHANTOM | 9,156 | 
| leaf | CT-single-reduced-opening-eval | 5 | PHANTOM | 9,156 | 
| leaf | CT-single-reduced-opening-eval | 6 | PHANTOM | 10,500 | 
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
| leaf | CastFV | 0 | ADD | 150 | 
| leaf | CastFV | 1 | ADD | 120 | 
| leaf | CastFV | 2 | ADD | 120 | 
| leaf | CastFV | 3 | ADD | 120 | 
| leaf | CastFV | 4 | ADD | 120 | 
| leaf | CastFV | 5 | ADD | 120 | 
| leaf | CastFV | 6 | ADD | 132 | 
| leaf | DivE | 0 | BBE4DIV | 7,136 | 
| leaf | DivE | 1 | BBE4DIV | 5,446 | 
| leaf | DivE | 2 | BBE4DIV | 5,446 | 
| leaf | DivE | 3 | BBE4DIV | 5,446 | 
| leaf | DivE | 4 | BBE4DIV | 5,446 | 
| leaf | DivE | 5 | BBE4DIV | 5,446 | 
| leaf | DivE | 6 | BBE4DIV | 6,122 | 
| leaf | DivEIN | 0 | ADD | 764 | 
| leaf | DivEIN | 0 | BBE4DIV | 191 | 
| leaf | DivEIN | 1 | ADD | 504 | 
| leaf | DivEIN | 1 | BBE4DIV | 126 | 
| leaf | DivEIN | 2 | ADD | 504 | 
| leaf | DivEIN | 2 | BBE4DIV | 126 | 
| leaf | DivEIN | 3 | ADD | 504 | 
| leaf | DivEIN | 3 | BBE4DIV | 126 | 
| leaf | DivEIN | 4 | ADD | 504 | 
| leaf | DivEIN | 4 | BBE4DIV | 126 | 
| leaf | DivEIN | 5 | ADD | 504 | 
| leaf | DivEIN | 5 | BBE4DIV | 126 | 
| leaf | DivEIN | 6 | ADD | 608 | 
| leaf | DivEIN | 6 | BBE4DIV | 152 | 
| leaf | DivF | 0 | DIV | 7,392 | 
| leaf | DivF | 1 | DIV | 7,392 | 
| leaf | DivF | 2 | DIV | 7,392 | 
| leaf | DivF | 3 | DIV | 7,392 | 
| leaf | DivF | 4 | DIV | 7,392 | 
| leaf | DivF | 5 | DIV | 7,392 | 
| leaf | DivF | 6 | DIV | 7,392 | 
| leaf | DivFIN | 0 | DIV | 401 | 
| leaf | DivFIN | 1 | DIV | 266 | 
| leaf | DivFIN | 2 | DIV | 266 | 
| leaf | DivFIN | 3 | DIV | 266 | 
| leaf | DivFIN | 4 | DIV | 266 | 
| leaf | DivFIN | 5 | DIV | 266 | 
| leaf | DivFIN | 6 | DIV | 320 | 
| leaf | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 6,258 | 
| leaf | FriReducedOpening | 1 | FRI_REDUCED_OPENING | 4,578 | 
| leaf | FriReducedOpening | 2 | FRI_REDUCED_OPENING | 4,578 | 
| leaf | FriReducedOpening | 3 | FRI_REDUCED_OPENING | 4,578 | 
| leaf | FriReducedOpening | 4 | FRI_REDUCED_OPENING | 4,578 | 
| leaf | FriReducedOpening | 5 | FRI_REDUCED_OPENING | 4,578 | 
| leaf | FriReducedOpening | 6 | FRI_REDUCED_OPENING | 5,250 | 
| leaf | HintBitsF | 0 | PHANTOM | 149 | 
| leaf | HintBitsF | 1 | PHANTOM | 119 | 
| leaf | HintBitsF | 2 | PHANTOM | 119 | 
| leaf | HintBitsF | 3 | PHANTOM | 119 | 
| leaf | HintBitsF | 4 | PHANTOM | 119 | 
| leaf | HintBitsF | 5 | PHANTOM | 119 | 
| leaf | HintBitsF | 6 | PHANTOM | 131 | 
| leaf | HintInputVec | 0 | PHANTOM | 8,200 | 
| leaf | HintInputVec | 1 | PHANTOM | 6,825 | 
| leaf | HintInputVec | 2 | PHANTOM | 6,825 | 
| leaf | HintInputVec | 3 | PHANTOM | 6,825 | 
| leaf | HintInputVec | 4 | PHANTOM | 6,825 | 
| leaf | HintInputVec | 5 | PHANTOM | 6,825 | 
| leaf | HintInputVec | 6 | PHANTOM | 7,600 | 
| leaf | HintLoad | 0 | PHANTOM | 1,176 | 
| leaf | HintLoad | 1 | PHANTOM | 1,176 | 
| leaf | HintLoad | 2 | PHANTOM | 1,176 | 
| leaf | HintLoad | 3 | PHANTOM | 1,176 | 
| leaf | HintLoad | 4 | PHANTOM | 1,176 | 
| leaf | HintLoad | 5 | PHANTOM | 1,176 | 
| leaf | HintLoad | 6 | PHANTOM | 1,176 | 
| leaf | IfEq | 0 | BNE | 140 | 
| leaf | IfEq | 1 | BNE | 141 | 
| leaf | IfEq | 2 | BNE | 141 | 
| leaf | IfEq | 3 | BNE | 141 | 
| leaf | IfEq | 4 | BNE | 141 | 
| leaf | IfEq | 5 | BNE | 141 | 
| leaf | IfEq | 6 | BNE | 141 | 
| leaf | IfEqI | 0 | BNE | 25,251 | 
| leaf | IfEqI | 0 | JAL | 8,956 | 
| leaf | IfEqI | 1 | BNE | 22,991 | 
| leaf | IfEqI | 1 | JAL | 8,841 | 
| leaf | IfEqI | 2 | BNE | 22,991 | 
| leaf | IfEqI | 2 | JAL | 8,740 | 
| leaf | IfEqI | 3 | BNE | 22,991 | 
| leaf | IfEqI | 3 | JAL | 8,833 | 
| leaf | IfEqI | 4 | BNE | 22,991 | 
| leaf | IfEqI | 4 | JAL | 8,709 | 
| leaf | IfEqI | 5 | BNE | 22,991 | 
| leaf | IfEqI | 5 | JAL | 8,546 | 
| leaf | IfEqI | 6 | BNE | 23,895 | 
| leaf | IfEqI | 6 | JAL | 8,794 | 
| leaf | IfNe | 0 | BEQ | 143 | 
| leaf | IfNe | 0 | JAL | 3 | 
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
| leaf | IfNeI | 0 | BEQ | 95 | 
| leaf | IfNeI | 1 | BEQ | 70 | 
| leaf | IfNeI | 2 | BEQ | 70 | 
| leaf | IfNeI | 3 | BEQ | 70 | 
| leaf | IfNeI | 4 | BEQ | 70 | 
| leaf | IfNeI | 5 | BEQ | 70 | 
| leaf | IfNeI | 6 | BEQ | 80 | 
| leaf | ImmE | 0 | ADD | 3,248 | 
| leaf | ImmE | 1 | ADD | 2,216 | 
| leaf | ImmE | 2 | ADD | 2,216 | 
| leaf | ImmE | 3 | ADD | 2,216 | 
| leaf | ImmE | 4 | ADD | 2,216 | 
| leaf | ImmE | 5 | ADD | 2,216 | 
| leaf | ImmE | 6 | ADD | 2,780 | 
| leaf | ImmF | 0 | ADD | 6,245 | 
| leaf | ImmF | 1 | ADD | 5,215 | 
| leaf | ImmF | 2 | ADD | 5,215 | 
| leaf | ImmF | 3 | ADD | 5,215 | 
| leaf | ImmF | 4 | ADD | 5,215 | 
| leaf | ImmF | 5 | ADD | 5,215 | 
| leaf | ImmF | 6 | ADD | 5,627 | 
| leaf | ImmV | 0 | ADD | 5,368 | 
| leaf | ImmV | 1 | ADD | 5,182 | 
| leaf | ImmV | 2 | ADD | 5,182 | 
| leaf | ImmV | 3 | ADD | 5,182 | 
| leaf | ImmV | 4 | ADD | 5,182 | 
| leaf | ImmV | 5 | ADD | 5,182 | 
| leaf | ImmV | 6 | ADD | 5,259 | 
| leaf | LoadE | 0 | ADD | 11,424 | 
| leaf | LoadE | 0 | LOADW | 27,048 | 
| leaf | LoadE | 0 | MUL | 11,424 | 
| leaf | LoadE | 1 | ADD | 8,904 | 
| leaf | LoadE | 1 | LOADW | 20,931 | 
| leaf | LoadE | 1 | MUL | 8,904 | 
| leaf | LoadE | 2 | ADD | 8,904 | 
| leaf | LoadE | 2 | LOADW | 20,931 | 
| leaf | LoadE | 2 | MUL | 8,904 | 
| leaf | LoadE | 3 | ADD | 8,904 | 
| leaf | LoadE | 3 | LOADW | 20,931 | 
| leaf | LoadE | 3 | MUL | 8,904 | 
| leaf | LoadE | 4 | ADD | 8,904 | 
| leaf | LoadE | 4 | LOADW | 20,931 | 
| leaf | LoadE | 4 | MUL | 8,904 | 
| leaf | LoadE | 5 | ADD | 8,904 | 
| leaf | LoadE | 5 | LOADW | 20,931 | 
| leaf | LoadE | 5 | MUL | 8,904 | 
| leaf | LoadE | 6 | ADD | 9,912 | 
| leaf | LoadE | 6 | LOADW | 23,403 | 
| leaf | LoadE | 6 | MUL | 9,912 | 
| leaf | LoadF | 0 | ADD | 14,912 | 
| leaf | LoadF | 0 | LOADW | 30,208 | 
| leaf | LoadF | 0 | MUL | 9,939 | 
| leaf | LoadF | 1 | ADD | 11,032 | 
| leaf | LoadF | 1 | LOADW | 22,798 | 
| leaf | LoadF | 1 | MUL | 7,319 | 
| leaf | LoadF | 2 | ADD | 11,032 | 
| leaf | LoadF | 2 | LOADW | 22,798 | 
| leaf | LoadF | 2 | MUL | 7,319 | 
| leaf | LoadF | 3 | ADD | 11,032 | 
| leaf | LoadF | 3 | LOADW | 22,798 | 
| leaf | LoadF | 3 | MUL | 7,319 | 
| leaf | LoadF | 4 | ADD | 11,032 | 
| leaf | LoadF | 4 | LOADW | 22,798 | 
| leaf | LoadF | 4 | MUL | 7,319 | 
| leaf | LoadF | 5 | ADD | 11,032 | 
| leaf | LoadF | 5 | LOADW | 22,798 | 
| leaf | LoadF | 5 | MUL | 7,319 | 
| leaf | LoadF | 6 | ADD | 12,584 | 
| leaf | LoadF | 6 | LOADW | 26,418 | 
| leaf | LoadF | 6 | MUL | 8,367 | 
| leaf | LoadHeapPtr | 0 | ADD | 1 | 
| leaf | LoadHeapPtr | 1 | ADD | 1 | 
| leaf | LoadHeapPtr | 2 | ADD | 1 | 
| leaf | LoadHeapPtr | 3 | ADD | 1 | 
| leaf | LoadHeapPtr | 4 | ADD | 1 | 
| leaf | LoadHeapPtr | 5 | ADD | 1 | 
| leaf | LoadHeapPtr | 6 | ADD | 1 | 
| leaf | LoadV | 0 | ADD | 49,705 | 
| leaf | LoadV | 0 | LOADW | 134,902 | 
| leaf | LoadV | 0 | MUL | 44,526 | 
| leaf | LoadV | 1 | ADD | 37,435 | 
| leaf | LoadV | 1 | LOADW | 106,402 | 
| leaf | LoadV | 1 | MUL | 33,526 | 
| leaf | LoadV | 2 | ADD | 37,435 | 
| leaf | LoadV | 2 | LOADW | 106,402 | 
| leaf | LoadV | 2 | MUL | 33,526 | 
| leaf | LoadV | 3 | ADD | 37,435 | 
| leaf | LoadV | 3 | LOADW | 106,402 | 
| leaf | LoadV | 3 | MUL | 33,526 | 
| leaf | LoadV | 4 | ADD | 37,435 | 
| leaf | LoadV | 4 | LOADW | 106,402 | 
| leaf | LoadV | 4 | MUL | 33,526 | 
| leaf | LoadV | 5 | ADD | 37,435 | 
| leaf | LoadV | 5 | LOADW | 106,402 | 
| leaf | LoadV | 5 | MUL | 33,526 | 
| leaf | LoadV | 6 | ADD | 42,343 | 
| leaf | LoadV | 6 | LOADW | 118,028 | 
| leaf | LoadV | 6 | MUL | 37,926 | 
| leaf | MulE | 0 | BBE4MUL | 24,058 | 
| leaf | MulE | 1 | BBE4MUL | 18,550 | 
| leaf | MulE | 2 | BBE4MUL | 18,550 | 
| leaf | MulE | 3 | BBE4MUL | 18,550 | 
| leaf | MulE | 4 | BBE4MUL | 18,550 | 
| leaf | MulE | 5 | BBE4MUL | 18,550 | 
| leaf | MulE | 6 | BBE4MUL | 20,836 | 
| leaf | MulEF | 0 | MUL | 4,888 | 
| leaf | MulEF | 1 | MUL | 4,368 | 
| leaf | MulEF | 2 | MUL | 4,368 | 
| leaf | MulEF | 3 | MUL | 4,368 | 
| leaf | MulEF | 4 | MUL | 4,368 | 
| leaf | MulEF | 5 | MUL | 4,368 | 
| leaf | MulEF | 6 | MUL | 4,576 | 
| leaf | MulEFI | 0 | MUL | 512 | 
| leaf | MulEFI | 1 | MUL | 232 | 
| leaf | MulEFI | 2 | MUL | 232 | 
| leaf | MulEFI | 3 | MUL | 232 | 
| leaf | MulEFI | 4 | MUL | 232 | 
| leaf | MulEFI | 5 | MUL | 232 | 
| leaf | MulEFI | 6 | MUL | 380 | 
| leaf | MulEI | 0 | ADD | 3,964 | 
| leaf | MulEI | 0 | BBE4MUL | 991 | 
| leaf | MulEI | 1 | ADD | 2,680 | 
| leaf | MulEI | 1 | BBE4MUL | 670 | 
| leaf | MulEI | 2 | ADD | 2,680 | 
| leaf | MulEI | 2 | BBE4MUL | 670 | 
| leaf | MulEI | 3 | ADD | 2,680 | 
| leaf | MulEI | 3 | BBE4MUL | 670 | 
| leaf | MulEI | 4 | ADD | 2,680 | 
| leaf | MulEI | 4 | BBE4MUL | 670 | 
| leaf | MulEI | 5 | ADD | 2,680 | 
| leaf | MulEI | 5 | BBE4MUL | 670 | 
| leaf | MulEI | 6 | ADD | 3,232 | 
| leaf | MulEI | 6 | BBE4MUL | 808 | 
| leaf | MulF | 0 | MUL | 34,499 | 
| leaf | MulF | 1 | MUL | 33,469 | 
| leaf | MulF | 2 | MUL | 33,469 | 
| leaf | MulF | 3 | MUL | 33,469 | 
| leaf | MulF | 4 | MUL | 33,469 | 
| leaf | MulF | 5 | MUL | 33,469 | 
| leaf | MulF | 6 | MUL | 33,881 | 
| leaf | MulFI | 0 | MUL | 4,638 | 
| leaf | MulFI | 1 | MUL | 3,703 | 
| leaf | MulFI | 2 | MUL | 3,703 | 
| leaf | MulFI | 3 | MUL | 3,703 | 
| leaf | MulFI | 4 | MUL | 3,703 | 
| leaf | MulFI | 5 | MUL | 3,703 | 
| leaf | MulFI | 6 | MUL | 4,077 | 
| leaf | MulVI | 0 | MUL | 10,796 | 
| leaf | MulVI | 1 | MUL | 8,216 | 
| leaf | MulVI | 2 | MUL | 8,216 | 
| leaf | MulVI | 3 | MUL | 8,216 | 
| leaf | MulVI | 4 | MUL | 8,216 | 
| leaf | MulVI | 5 | MUL | 8,216 | 
| leaf | MulVI | 6 | MUL | 9,248 | 
| leaf | NegE | 0 | MUL | 152 | 
| leaf | NegE | 1 | MUL | 92 | 
| leaf | NegE | 2 | MUL | 92 | 
| leaf | NegE | 3 | MUL | 92 | 
| leaf | NegE | 4 | MUL | 92 | 
| leaf | NegE | 5 | MUL | 92 | 
| leaf | NegE | 6 | MUL | 120 | 
| leaf | Poseidon2CompressBabyBear | 6 | COMP_POS2 | 27 | 
| leaf | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 51 | 
| leaf | Poseidon2PermuteBabyBear | 1 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 2 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 3 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 4 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 5 | PERM_POS2 | 48 | 
| leaf | Poseidon2PermuteBabyBear | 6 | PERM_POS2 | 49 | 
| leaf | PrintV | 0 | PHANTOM | 1,176 | 
| leaf | PrintV | 1 | PHANTOM | 1,176 | 
| leaf | PrintV | 2 | PHANTOM | 1,176 | 
| leaf | PrintV | 3 | PHANTOM | 1,176 | 
| leaf | PrintV | 4 | PHANTOM | 1,176 | 
| leaf | PrintV | 5 | PHANTOM | 1,176 | 
| leaf | PrintV | 6 | PHANTOM | 1,176 | 
| leaf | Publish | 0 | PUBLISH | 36 | 
| leaf | Publish | 1 | PUBLISH | 36 | 
| leaf | Publish | 2 | PUBLISH | 36 | 
| leaf | Publish | 3 | PUBLISH | 36 | 
| leaf | Publish | 4 | PUBLISH | 36 | 
| leaf | Publish | 5 | PUBLISH | 36 | 
| leaf | Publish | 6 | PUBLISH | 36 | 
| leaf | StoreE | 0 | ADD | 9,744 | 
| leaf | StoreE | 0 | MUL | 9,744 | 
| leaf | StoreE | 0 | STOREW | 15,173 | 
| leaf | StoreE | 1 | ADD | 7,224 | 
| leaf | StoreE | 1 | MUL | 7,224 | 
| leaf | StoreE | 1 | STOREW | 12,608 | 
| leaf | StoreE | 2 | ADD | 7,224 | 
| leaf | StoreE | 2 | MUL | 7,224 | 
| leaf | StoreE | 2 | STOREW | 12,608 | 
| leaf | StoreE | 3 | ADD | 7,224 | 
| leaf | StoreE | 3 | MUL | 7,224 | 
| leaf | StoreE | 3 | STOREW | 12,608 | 
| leaf | StoreE | 4 | ADD | 7,224 | 
| leaf | StoreE | 4 | MUL | 7,224 | 
| leaf | StoreE | 4 | STOREW | 12,608 | 
| leaf | StoreE | 5 | ADD | 7,224 | 
| leaf | StoreE | 5 | MUL | 7,224 | 
| leaf | StoreE | 5 | STOREW | 12,608 | 
| leaf | StoreE | 6 | ADD | 8,232 | 
| leaf | StoreE | 6 | MUL | 8,232 | 
| leaf | StoreE | 6 | STOREW | 13,634 | 
| leaf | StoreF | 0 | ADD | 776 | 
| leaf | StoreF | 0 | MUL | 404 | 
| leaf | StoreF | 0 | STOREW | 8,295 | 
| leaf | StoreF | 1 | ADD | 631 | 
| leaf | StoreF | 1 | MUL | 284 | 
| leaf | StoreF | 1 | STOREW | 8,150 | 
| leaf | StoreF | 2 | ADD | 631 | 
| leaf | StoreF | 2 | MUL | 284 | 
| leaf | StoreF | 2 | STOREW | 8,150 | 
| leaf | StoreF | 3 | ADD | 631 | 
| leaf | StoreF | 3 | MUL | 284 | 
| leaf | StoreF | 3 | STOREW | 8,150 | 
| leaf | StoreF | 4 | ADD | 631 | 
| leaf | StoreF | 4 | MUL | 284 | 
| leaf | StoreF | 4 | STOREW | 8,150 | 
| leaf | StoreF | 5 | ADD | 631 | 
| leaf | StoreF | 5 | MUL | 284 | 
| leaf | StoreF | 5 | STOREW | 8,150 | 
| leaf | StoreF | 6 | ADD | 905 | 
| leaf | StoreF | 6 | MUL | 548 | 
| leaf | StoreF | 6 | STOREW | 8,856 | 
| leaf | StoreHeapPtr | 0 | ADD | 1 | 
| leaf | StoreHeapPtr | 1 | ADD | 1 | 
| leaf | StoreHeapPtr | 2 | ADD | 1 | 
| leaf | StoreHeapPtr | 3 | ADD | 1 | 
| leaf | StoreHeapPtr | 4 | ADD | 1 | 
| leaf | StoreHeapPtr | 5 | ADD | 1 | 
| leaf | StoreHeapPtr | 6 | ADD | 1 | 
| leaf | StoreHintWord | 0 | HINT_STOREW | 79,515 | 
| leaf | StoreHintWord | 1 | HINT_STOREW | 60,390 | 
| leaf | StoreHintWord | 2 | HINT_STOREW | 60,390 | 
| leaf | StoreHintWord | 3 | HINT_STOREW | 60,390 | 
| leaf | StoreHintWord | 4 | HINT_STOREW | 60,390 | 
| leaf | StoreHintWord | 5 | HINT_STOREW | 60,390 | 
| leaf | StoreHintWord | 6 | HINT_STOREW | 69,070 | 
| leaf | StoreV | 0 | ADD | 15,069 | 
| leaf | StoreV | 0 | MUL | 10,123 | 
| leaf | StoreV | 0 | STOREW | 30,720 | 
| leaf | StoreV | 1 | ADD | 10,924 | 
| leaf | StoreV | 1 | MUL | 7,268 | 
| leaf | StoreV | 1 | STOREW | 23,830 | 
| leaf | StoreV | 2 | ADD | 10,924 | 
| leaf | StoreV | 2 | MUL | 7,268 | 
| leaf | StoreV | 2 | STOREW | 23,830 | 
| leaf | StoreV | 3 | ADD | 10,924 | 
| leaf | StoreV | 3 | MUL | 7,268 | 
| leaf | StoreV | 3 | STOREW | 23,830 | 
| leaf | StoreV | 4 | ADD | 10,924 | 
| leaf | StoreV | 4 | MUL | 7,268 | 
| leaf | StoreV | 4 | STOREW | 23,830 | 
| leaf | StoreV | 5 | ADD | 10,924 | 
| leaf | StoreV | 5 | MUL | 7,268 | 
| leaf | StoreV | 5 | STOREW | 23,830 | 
| leaf | StoreV | 6 | ADD | 12,582 | 
| leaf | StoreV | 6 | MUL | 8,410 | 
| leaf | StoreV | 6 | STOREW | 26,586 | 
| leaf | SubE | 0 | FE4SUB | 3,173 | 
| leaf | SubE | 1 | FE4SUB | 3,020 | 
| leaf | SubE | 2 | FE4SUB | 3,020 | 
| leaf | SubE | 3 | FE4SUB | 3,020 | 
| leaf | SubE | 4 | FE4SUB | 3,020 | 
| leaf | SubE | 5 | FE4SUB | 3,020 | 
| leaf | SubE | 6 | FE4SUB | 3,090 | 
| leaf | SubEF | 0 | ADD | 18,948 | 
| leaf | SubEF | 0 | SUB | 6,316 | 
| leaf | SubEF | 1 | ADD | 13,878 | 
| leaf | SubEF | 1 | SUB | 4,626 | 
| leaf | SubEF | 2 | ADD | 13,878 | 
| leaf | SubEF | 2 | SUB | 4,626 | 
| leaf | SubEF | 3 | ADD | 13,878 | 
| leaf | SubEF | 3 | SUB | 4,626 | 
| leaf | SubEF | 4 | ADD | 13,878 | 
| leaf | SubEF | 4 | SUB | 4,626 | 
| leaf | SubEF | 5 | ADD | 13,878 | 
| leaf | SubEF | 5 | SUB | 4,626 | 
| leaf | SubEF | 6 | ADD | 15,906 | 
| leaf | SubEF | 6 | SUB | 5,302 | 
| leaf | SubEFI | 0 | ADD | 300 | 
| leaf | SubEFI | 1 | ADD | 152 | 
| leaf | SubEFI | 2 | ADD | 152 | 
| leaf | SubEFI | 3 | ADD | 152 | 
| leaf | SubEFI | 4 | ADD | 152 | 
| leaf | SubEFI | 5 | ADD | 152 | 
| leaf | SubEFI | 6 | ADD | 240 | 
| leaf | SubEI | 0 | ADD | 1,528 | 
| leaf | SubEI | 1 | ADD | 1,008 | 
| leaf | SubEI | 2 | ADD | 1,008 | 
| leaf | SubEI | 3 | ADD | 1,008 | 
| leaf | SubEI | 4 | ADD | 1,008 | 
| leaf | SubEI | 5 | ADD | 1,008 | 
| leaf | SubEI | 6 | ADD | 1,216 | 
| leaf | SubFI | 0 | SUB | 4,619 | 
| leaf | SubFI | 1 | SUB | 3,689 | 
| leaf | SubFI | 2 | SUB | 3,689 | 
| leaf | SubFI | 3 | SUB | 3,689 | 
| leaf | SubFI | 4 | SUB | 3,689 | 
| leaf | SubFI | 5 | SUB | 3,689 | 
| leaf | SubFI | 6 | SUB | 4,061 | 
| leaf | SubV | 0 | SUB | 6,823 | 
| leaf | SubV | 1 | SUB | 5,558 | 
| leaf | SubV | 2 | SUB | 5,558 | 
| leaf | SubV | 3 | SUB | 5,558 | 
| leaf | SubV | 4 | SUB | 5,558 | 
| leaf | SubV | 5 | SUB | 5,558 | 
| leaf | SubV | 6 | SUB | 6,064 | 
| leaf | SubVI | 0 | SUB | 999 | 
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
| leaf | UnsafeCastVF | 0 | ADD | 125 | 
| leaf | UnsafeCastVF | 1 | ADD | 90 | 
| leaf | UnsafeCastVF | 2 | ADD | 90 | 
| leaf | UnsafeCastVF | 3 | ADD | 90 | 
| leaf | UnsafeCastVF | 4 | ADD | 90 | 
| leaf | UnsafeCastVF | 5 | ADD | 90 | 
| leaf | UnsafeCastVF | 6 | ADD | 104 | 
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
| leaf | ZipFor | 0 | ADD | 162,450 | 
| leaf | ZipFor | 0 | BNE | 133,536 | 
| leaf | ZipFor | 0 | JAL | 16,063 | 
| leaf | ZipFor | 1 | ADD | 131,147 | 
| leaf | ZipFor | 1 | BNE | 107,723 | 
| leaf | ZipFor | 1 | JAL | 13,243 | 
| leaf | ZipFor | 2 | ADD | 131,147 | 
| leaf | ZipFor | 2 | BNE | 107,723 | 
| leaf | ZipFor | 2 | JAL | 13,243 | 
| leaf | ZipFor | 3 | ADD | 131,147 | 
| leaf | ZipFor | 3 | BNE | 107,723 | 
| leaf | ZipFor | 3 | JAL | 13,243 | 
| leaf | ZipFor | 4 | ADD | 131,147 | 
| leaf | ZipFor | 4 | BNE | 107,723 | 
| leaf | ZipFor | 4 | JAL | 13,243 | 
| leaf | ZipFor | 5 | ADD | 131,147 | 
| leaf | ZipFor | 5 | BNE | 107,723 | 
| leaf | ZipFor | 5 | JAL | 13,243 | 
| leaf | ZipFor | 6 | ADD | 144,671 | 
| leaf | ZipFor | 6 | BNE | 119,051 | 
| leaf | ZipFor | 6 | JAL | 14,597 | 
| root |  | 0 | ADD | 2 | 
| root |  | 0 | JAL | 1 | 
| root | AddE | 0 | FE4ADD | 13,784 | 
| root | AddEFFI | 0 | ADD | 1,512 | 
| root | AddEFI | 0 | ADD | 980 | 
| root | AddEI | 0 | ADD | 36,864 | 
| root | AddF | 0 | ADD | 4,970 | 
| root | AddFI | 0 | ADD | 12,507 | 
| root | AddV | 0 | ADD | 25,712 | 
| root | AddVI | 0 | ADD | 45,880 | 
| root | Alloc | 0 | ADD | 42,202 | 
| root | Alloc | 0 | MUL | 12,141 | 
| root | AssertEqE | 0 | BNE | 236 | 
| root | AssertEqEI | 0 | BNE | 4 | 
| root | AssertEqF | 0 | BNE | 4,560 | 
| root | AssertEqFI | 0 | BNE | 5 | 
| root | AssertEqV | 0 | BNE | 1,403 | 
| root | AssertEqVI | 0 | BNE | 240 | 
| root | AssertNonZero | 0 | BEQ | 1 | 
| root | CT-ExtractPublicValues | 0 | PHANTOM | 2 | 
| root | CT-HintOpenedValues | 0 | PHANTOM | 504 | 
| root | CT-HintOpeningProof | 0 | PHANTOM | 506 | 
| root | CT-HintOpeningValues | 0 | PHANTOM | 2 | 
| root | CT-InitializePcsConst | 0 | PHANTOM | 2 | 
| root | CT-ReadProofsFromInput | 0 | PHANTOM | 2 | 
| root | CT-VerifyProofs | 0 | PHANTOM | 2 | 
| root | CT-cache-generator-powers | 0 | PHANTOM | 504 | 
| root | CT-compute-reduced-opening | 0 | PHANTOM | 504 | 
| root | CT-exp-reverse-bits-len | 0 | PHANTOM | 8,316 | 
| root | CT-pre-compute-alpha-pows | 0 | PHANTOM | 2 | 
| root | CT-single-reduced-opening-eval | 0 | PHANTOM | 11,424 | 
| root | CT-stage-c-build-rounds | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verifier-verify | 0 | PHANTOM | 2 | 
| root | CT-stage-d-verify-pcs | 0 | PHANTOM | 2 | 
| root | CT-stage-e-verify-constraints | 0 | PHANTOM | 2 | 
| root | CT-verify-batch | 0 | PHANTOM | 504 | 
| root | CT-verify-batch-ext | 0 | PHANTOM | 1,764 | 
| root | CT-verify-query | 0 | PHANTOM | 84 | 
| root | CastFV | 0 | ADD | 150 | 
| root | DivE | 0 | BBE4DIV | 6,628 | 
| root | DivEIN | 0 | ADD | 788 | 
| root | DivEIN | 0 | BBE4DIV | 197 | 
| root | DivF | 0 | DIV | 5,796 | 
| root | DivFIN | 0 | DIV | 411 | 
| root | FriReducedOpening | 0 | FRI_REDUCED_OPENING | 5,712 | 
| root | HintBitsF | 0 | PHANTOM | 142 | 
| root | HintInputVec | 0 | PHANTOM | 7,826 | 
| root | HintLoad | 0 | PHANTOM | 1,134 | 
| root | IfEq | 0 | BNE | 144 | 
| root | IfEqI | 0 | BNE | 21,529 | 
| root | IfEqI | 0 | JAL | 7,093 | 
| root | IfNe | 0 | BEQ | 147 | 
| root | IfNe | 0 | JAL | 3 | 
| root | IfNeI | 0 | BEQ | 93 | 
| root | ImmE | 0 | ADD | 4,132 | 
| root | ImmF | 0 | ADD | 5,828 | 
| root | ImmV | 0 | ADD | 5,451 | 
| root | LoadE | 0 | ADD | 10,962 | 
| root | LoadE | 0 | LOADW | 26,126 | 
| root | LoadE | 0 | MUL | 10,962 | 
| root | LoadF | 0 | ADD | 13,752 | 
| root | LoadF | 0 | LOADW | 28,415 | 
| root | LoadF | 0 | MUL | 9,153 | 
| root | LoadHeapPtr | 0 | ADD | 1 | 
| root | LoadV | 0 | ADD | 46,457 | 
| root | LoadV | 0 | LOADW | 123,180 | 
| root | LoadV | 0 | MUL | 41,474 | 
| root | MulE | 0 | BBE4MUL | 27,872 | 
| root | MulEF | 0 | MUL | 5,104 | 
| root | MulEFI | 0 | MUL | 708 | 
| root | MulEI | 0 | ADD | 4,948 | 
| root | MulEI | 0 | BBE4MUL | 1,237 | 
| root | MulF | 0 | MUL | 27,894 | 
| root | MulFI | 0 | MUL | 4,419 | 
| root | MulVI | 0 | MUL | 10,059 | 
| root | NegE | 0 | MUL | 292 | 
| root | Poseidon2CompressBabyBear | 0 | COMP_POS2 | 12 | 
| root | Poseidon2PermuteBabyBear | 0 | PERM_POS2 | 54 | 
| root | PrintV | 0 | PHANTOM | 1,134 | 
| root | Publish | 0 | PUBLISH | 48 | 
| root | StoreE | 0 | ADD | 9,198 | 
| root | StoreE | 0 | MUL | 9,198 | 
| root | StoreE | 0 | STOREW | 14,741 | 
| root | StoreF | 0 | ADD | 774 | 
| root | StoreF | 0 | MUL | 384 | 
| root | StoreF | 0 | STOREW | 6,937 | 
| root | StoreHeapPtr | 0 | ADD | 1 | 
| root | StoreHintWord | 0 | HINT_STOREW | 85,225 | 
| root | StoreV | 0 | ADD | 14,050 | 
| root | StoreV | 0 | MUL | 9,445 | 
| root | StoreV | 0 | STOREW | 28,945 | 
| root | SubE | 0 | FE4SUB | 3,540 | 
| root | SubEF | 0 | ADD | 17,250 | 
| root | SubEF | 0 | SUB | 5,750 | 
| root | SubEFI | 0 | ADD | 392 | 
| root | SubEI | 0 | ADD | 1,576 | 
| root | SubF | 0 | SUB | 8 | 
| root | SubFI | 0 | SUB | 4,402 | 
| root | SubV | 0 | SUB | 6,443 | 
| root | SubVI | 0 | SUB | 1,044 | 
| root | SubVIN | 0 | SUB | 882 | 
| root | UnsafeCastVF | 0 | ADD | 116 | 
| root | VerifyBatchExt | 0 | VERIFY_BATCH | 882 | 
| root | VerifyBatchFelt | 0 | VERIFY_BATCH | 252 | 
| root | ZipFor | 0 | ADD | 163,649 | 
| root | ZipFor | 0 | BNE | 138,189 | 
| root | ZipFor | 0 | JAL | 15,005 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fib_e2e |  | ADD | 6 | 909,048 | 
| fib_e2e |  | BEQ | 6 | 101,006 | 
| fib_e2e |  | BNE | 6 | 101,005 | 
| fib_e2e |  | JAL | 6 | 101,005 | 
| fib_e2e |  | JALR | 6 | 1 | 
| fib_e2e |  | LOADW | 6 | 1 | 
| fib_e2e |  | SLTU | 6 | 303,015 | 
| fib_e2e |  | STOREW | 6 | 2 | 

| group | idx | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal.0 | 0 | 1,425 | 19,615 | 2,430,210 | 277,758,936 | 9,411 | 1,257 | 1,848 | 1,861 | 2,232 | 1,972 | 97,838,026 | 236 | 8,779 | 
| internal.0 | 1 | 1,623 | 19,850 | 2,430,923 | 277,758,936 | 9,379 | 1,271 | 1,831 | 1,855 | 2,239 | 1,950 | 97,790,803 | 229 | 8,848 | 
| internal.0 | 2 | 1,637 | 20,049 | 2,430,439 | 277,758,936 | 9,394 | 1,259 | 1,844 | 1,868 | 2,239 | 1,947 | 97,786,447 | 233 | 9,018 | 
| internal.0 | 3 | 1,006 | 10,826 | 1,215,838 | 142,419,672 | 5,270 | 696 | 1,010 | 952 | 1,277 | 1,210 | 49,991,906 | 121 | 4,550 | 
| internal.1 | 4 | 1,570 | 20,178 | 2,464,036 | 277,758,936 | 9,431 | 1,271 | 1,831 | 1,872 | 2,243 | 1,980 | 99,779,328 | 228 | 9,177 | 
| internal.1 | 5 | 1,645 | 20,284 | 2,448,159 | 277,758,936 | 9,481 | 1,271 | 1,841 | 1,852 | 2,297 | 1,987 | 99,365,356 | 228 | 9,158 | 
| internal.2 | 6 | 1,641 | 20,155 | 2,464,081 | 277,758,936 | 9,397 | 1,262 | 1,840 | 1,858 | 2,248 | 1,956 | 99,779,733 | 229 | 9,117 | 
| leaf | 0 | 892 | 10,480 | 1,270,725 | 140,060,376 | 5,060 | 680 | 1,046 | 951 | 1,255 | 1,005 | 50,851,136 | 119 | 4,528 | 
| leaf | 1 | 771 | 9,011 | 1,026,727 | 127,543,000 | 4,518 | 605 | 923 | 819 | 1,181 | 879 | 42,117,289 | 107 | 3,722 | 
| leaf | 2 | 768 | 9,012 | 1,026,626 | 127,543,000 | 4,504 | 604 | 902 | 820 | 1,197 | 870 | 42,116,380 | 107 | 3,740 | 
| leaf | 3 | 762 | 9,036 | 1,026,719 | 127,543,000 | 4,544 | 608 | 908 | 828 | 1,204 | 884 | 42,117,217 | 109 | 3,730 | 
| leaf | 4 | 754 | 9,062 | 1,026,595 | 127,543,000 | 4,528 | 606 | 911 | 817 | 1,200 | 882 | 42,116,101 | 108 | 3,780 | 
| leaf | 5 | 774 | 9,081 | 1,026,432 | 127,543,000 | 4,527 | 601 | 917 | 817 | 1,203 | 877 | 42,114,634 | 107 | 3,780 | 
| leaf | 6 | 810 | 9,553 | 1,132,208 | 128,009,176 | 4,542 | 613 | 925 | 821 | 1,186 | 886 | 45,928,739 | 107 | 4,201 | 
| root | 0 | 815 | 37,296 | 1,232,879 | 142,419,672 | 31,739 | 682 | 7,302 | 9,683 | 3,776 | 10,165 | 51,005,063 | 120 | 4,742 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fib_e2e | 0 | 745 | 10,692 |  | 122,458,092 | 4,251 | 534 | 788 | 851 | 1,053 | 860 | 59,803,503 | 163 | 5,696 | 
| fib_e2e | 1 | 829 | 10,370 |  | 122,409,910 | 3,853 | 522 | 680 | 780 | 1,050 | 690 | 59,780,437 | 127 | 5,688 | 
| fib_e2e | 2 | 701 | 10,258 |  | 122,409,910 | 3,876 | 521 | 683 | 799 | 1,057 | 690 | 59,780,430 | 122 | 5,681 | 
| fib_e2e | 3 | 693 | 10,265 |  | 122,409,910 | 3,904 | 530 | 691 | 773 | 1,088 | 682 | 59,780,448 | 137 | 5,668 | 
| fib_e2e | 4 | 690 | 10,409 |  | 122,409,910 | 4,035 | 529 | 710 | 832 | 1,096 | 733 | 59,780,447 | 132 | 5,684 | 
| fib_e2e | 5 | 684 | 10,323 |  | 122,409,910 | 3,935 | 528 | 676 | 834 | 1,074 | 694 | 59,780,447 | 127 | 5,704 | 
| fib_e2e | 6 | 607 | 9,418 | 1,515,083 | 122,456,198 | 3,890 | 532 | 704 | 750 | 1,081 | 687 | 51,985,396 | 131 | 4,921 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-fib_e2e.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-halo2_outer.cell_tracker_span.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-halo2_outer.cell_tracker_span.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-halo2_outer.cell_tracker_span.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-halo2_outer.cell_tracker_span.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.0.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.1.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-internal.2.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-leaf.dsl_ir.opcode.frequency.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.air_name.cells_used.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.air_name.cells_used.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.frequency.reverse.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.frequency.reverse.svg)
[![](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.frequency.svg)](https://openvm-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64/fib_e2e-ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64-root.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/ad76dbaf9fc5afbedf3ba7affcd6ec0ec8358b64

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/13089911315)
