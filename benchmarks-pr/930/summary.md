### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,111,492 </div>  | <div style='text-align: right'> 217,347 </div>  | <span style='color: red'>(+29.0 [+1.1%])</span><div style='text-align: right'> 2,666.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/ecrecover-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 284,012,683 </div>  | <div style='text-align: right'> 5,163,177 </div>  | <span style='color: green'>(-189.0 [-0.7%])</span><div style='text-align: right'> 26,349.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,731 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-23.0 [-0.3%])</span><div style='text-align: right'> 6,612.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,721 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-111.0 [-1.5%])</span><div style='text-align: right'> 7,126.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,973,705 </div>  | <div style='text-align: right'> 4,190,904 </div>  | <span style='color: red'>(+331.0 [+1.2%])</span><div style='text-align: right'> 27,268.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/930/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+730 [+0.0%])</span><div style='text-align: right'> 48,126,555 </div>  | <span style='color: red'>(+31 [+0.0%])</span><div style='text-align: right'> 198,576 </div>  | <span style='color: red'>(+41.0 [+0.7%])</span><div style='text-align: right'> 5,675.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/e8ee0ac93d438eccb4d1a47a8fc06494dfdc5b45

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12167115020)