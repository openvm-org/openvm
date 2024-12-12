### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,111,492 </div>  | <div style='text-align: right'> 217,347 </div>  | <span style='color: red'>(+14.0 [+0.5%])</span><div style='text-align: right'> 2,658.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/ecrecover-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 284,010,873 </div>  | <div style='text-align: right'> 5,163,156 </div>  | <span style='color: red'>(+61.0 [+0.2%])</span><div style='text-align: right'> 26,567.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,731 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: red'>(+60.0 [+0.9%])</span><div style='text-align: right'> 6,639.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,721 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-132.0 [-1.8%])</span><div style='text-align: right'> 7,055.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,973,705 </div>  | <div style='text-align: right'> 4,190,904 </div>  | <span style='color: green'>(-64.0 [-0.2%])</span><div style='text-align: right'> 27,129.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/937/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-210 [-0.0%])</span><div style='text-align: right'> 48,126,275 </div>  | <span style='color: green'>(-7 [-0.0%])</span><div style='text-align: right'> 198,562 </div>  | <span style='color: green'>(-59.0 [-1.0%])</span><div style='text-align: right'> 5,593.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/6c4e94a84f3338cf1e32658d59a9eef20336374b

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12154402374)