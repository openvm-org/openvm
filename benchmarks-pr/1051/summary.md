### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ ecrecover_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1051/individual/ecrecover-3e199f1573e08c65e32e5b3b287a1b9a33ac9e75.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,251,804 </div>  | <div style='text-align: right'> 195,066 </div>  | <span style='color: green'>(-6.0 [-0.3%])</span><div style='text-align: right'> 1,974.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1051/individual/fibonacci-3e199f1573e08c65e32e5b3b287a1b9a33ac9e75.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,615,800 </div>  | <div style='text-align: right'> 3,000,274 </div>  | <span style='color: green'>(-22.0 [-0.4%])</span><div style='text-align: right'> 5,508.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1051/individual/regex-3e199f1573e08c65e32e5b3b287a1b9a33ac9e75.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,890,449 </div>  | <div style='text-align: right'> 8,381,808 </div>  | <span style='color: green'>(-417.0 [-2.4%])</span><div style='text-align: right'> 17,075.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1051/individual/verify_fibair-3e199f1573e08c65e32e5b3b287a1b9a33ac9e75.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-1,100 [-0.0%])</span><div style='text-align: right'> 48,127,187 </div>  | <span style='color: green'>(-80 [-0.0%])</span><div style='text-align: right'> 397,214 </div>  | <span style='color: green'>(-24.0 [-0.8%])</span><div style='text-align: right'> 3,110.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/openvm-org/openvm/commit/3e199f1573e08c65e32e5b3b287a1b9a33ac9e75

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12344622151)