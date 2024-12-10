### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,111,492 </div>  | <div style='text-align: right'> 217,347 </div>  | <span style='color: red'>(+45.0 [+1.7%])</span><div style='text-align: right'> 2,707.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/ecrecover-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 284,012,683 </div>  | <div style='text-align: right'> 5,163,177 </div>  | <span style='color: red'>(+25.0 [+0.1%])</span><div style='text-align: right'> 26,526.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,731 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-17.0 [-0.3%])</span><div style='text-align: right'> 6,634.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,721 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: red'>(+207.0 [+3.1%])</span><div style='text-align: right'> 6,894.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,973,705 </div>  | <div style='text-align: right'> 4,190,904 </div>  | <span style='color: green'>(-30.0 [-0.1%])</span><div style='text-align: right'> 27,303.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/952/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-850 [-0.0%])</span><div style='text-align: right'> 48,126,235 </div>  | <span style='color: green'>(-15 [-0.0%])</span><div style='text-align: right'> 198,565 </div>  | <span style='color: red'>(+23.0 [+0.4%])</span><div style='text-align: right'> 5,703.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/ccb4cde40d50bf52a726ebe4322d805bcb503549

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12187937243)