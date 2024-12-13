### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,125,712 </div>  | <div style='text-align: right'> 217,353 </div>  | <span style='color: green'>(-5.0 [-0.2%])</span><div style='text-align: right'> 2,659.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/ecrecover-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 306,736,060 </div>  | <div style='text-align: right'> 5,786,891 </div>  | <span style='color: green'>(-130.0 [-0.3%])</span><div style='text-align: right'> 38,000.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,643,290 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-19.0 [-0.3%])</span><div style='text-align: right'> 6,590.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,645,664 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-97.0 [-1.4%])</span><div style='text-align: right'> 6,757.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,817,559 </div>  | <div style='text-align: right'> 4,181,198 </div>  | <span style='color: green'>(-20.0 [-0.1%])</span><div style='text-align: right'> 26,984.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+2,909 [+0.0%])</span><div style='text-align: right'> 8,428,736 </div>  | <span style='color: red'>(+148 [+0.1%])</span><div style='text-align: right'> 198,645 </div>  | <span style='color: green'>(-43.0 [-2.6%])</span><div style='text-align: right'> 1,632.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |

### E2E Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | root_log_blowup | root_total_cells_used | root_total_cycles | root_total_proof_time_ms | internal_log_blowup | internal_total_cells_used | internal_total_cycles | internal_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [ fibonacci_continuation_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/916/individual/fib_e2e-2-2-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 12,292,198 </div>  | <div style='text-align: right'> 12,000,219 </div>  | <div style='text-align: right'> 38,906.0 </div>  | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 144,144,192 </div>  | <div style='text-align: right'> 3,634,985 </div>  | <div style='text-align: right'> 73,838.0 </div>  | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 988,877,694 </div>  | <div style='text-align: right'> 24,150,903 </div>  | <div style='text-align: right'> 91,998.0 </div>  | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 859,554,552 </div>  | <div style='text-align: right'> 21,781,006 </div>  | <div style='text-align: right'> 82,135.0 </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/797ff538a0f968d37ad747d03437a7bdcca1ef13

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12128742304)