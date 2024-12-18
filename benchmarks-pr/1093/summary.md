### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ ecrecover_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1093/individual/ecrecover-d753e5c08f1fb9b8bfe101a7d5e34d1df81c6c82.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,251,804 </div>  | <div style='text-align: right'> 195,066 </div>  | <span style='color: red'>(+99.0 [+4.8%])</span><div style='text-align: right'> 2,145.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1093/individual/fibonacci-d753e5c08f1fb9b8bfe101a7d5e34d1df81c6c82.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,615,800 </div>  | <div style='text-align: right'> 3,000,274 </div>  | <span style='color: red'>(+13.0 [+0.2%])</span><div style='text-align: right'> 5,546.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1093/individual/regex-d753e5c08f1fb9b8bfe101a7d5e34d1df81c6c82.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,890,449 </div>  | <div style='text-align: right'> 8,381,808 </div>  | <span style='color: red'>(+97.0 [+0.6%])</span><div style='text-align: right'> 17,356.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1093/individual/verify_fibair-d753e5c08f1fb9b8bfe101a7d5e34d1df81c6c82.md) | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+1,260 [+0.0%])</span><div style='text-align: right'> 48,127,147 </div>  | <span style='color: red'>(+70 [+0.0%])</span><div style='text-align: right'> 397,164 </div>  | <span style='color: green'>(-7.0 [-0.2%])</span><div style='text-align: right'> 3,145.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/openvm-org/openvm/commit/d753e5c08f1fb9b8bfe101a7d5e34d1df81c6c82

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12349764011)