### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/base64_json-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,116,803 </div>  | <div style='text-align: right'> 217,347 </div>  | <span style='color: red'>(+30.0 [+1.5%])</span><div style='text-align: right'> 1,979.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-5,650 [-0.0%])</span><div style='text-align: right'> 294,980,869 </div>  | <span style='color: green'>(-579 [-0.0%])</span><div style='text-align: right'> 6,788,095 </div>  | <span style='color: red'>(+230.0 [+0.9%])</span><div style='text-align: right'> 26,009.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/ecrecover-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,567,808 </div>  | <div style='text-align: right'> 106,444 </div>  | <span style='color: green'>(-352.0 [-16.3%])</span><div style='text-align: right'> 1,808.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/fibonacci-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,612,244 </div>  | <div style='text-align: right'> 1,500,137 </div>  | <span style='color: red'>(+66.0 [+1.3%])</span><div style='text-align: right'> 5,184.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+100 [+0.0%])</span><div style='text-align: right'> 144,232,313 </div>  | <span style='color: red'>(+73 [+0.0%])</span><div style='text-align: right'> 3,520,052 </div>  | <span style='color: green'>(-163.0 [-1.2%])</span><div style='text-align: right'> 13,154.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/fibonacci-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,612,244 </div>  | <div style='text-align: right'> 1,500,137 </div>  | <span style='color: red'>(+66.0 [+1.3%])</span><div style='text-align: right'> 5,184.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+100 [+0.0%])</span><div style='text-align: right'> 144,232,313 </div>  | <span style='color: red'>(+73 [+0.0%])</span><div style='text-align: right'> 3,520,052 </div>  | <span style='color: green'>(-163.0 [-1.2%])</span><div style='text-align: right'> 13,154.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/regex-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,871,537 </div>  | <div style='text-align: right'> 4,190,904 </div>  | <span style='color: green'>(-333.0 [-2.0%])</span><div style='text-align: right'> 16,126.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-22,430 [-0.0%])</span><div style='text-align: right'> 315,429,347 </div>  | <span style='color: green'>(-2,131 [-0.0%])</span><div style='text-align: right'> 7,320,300 </div>  | <span style='color: red'>(+35.0 [+0.1%])</span><div style='text-align: right'> 26,163.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/997/individual/verify_fibair-54186a90898db0274e0e5143fd4f8a666c8c3b52.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-1,950 [-0.0%])</span><div style='text-align: right'> 48,126,337 </div>  | <span style='color: green'>(-83 [-0.0%])</span><div style='text-align: right'> 198,564 </div>  | <span style='color: red'>(+1.0 [+0.0%])</span><div style='text-align: right'> 2,918.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/54186a90898db0274e0e5143fd4f8a666c8c3b52

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12284726171)