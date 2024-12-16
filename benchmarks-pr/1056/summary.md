### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ ecrecover_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1056/individual/ecrecover-68ea0831d91465505837b07c42245833a60ec6b5.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,251,804 </div>  | <div style='text-align: right'> 195,066 </div>  | <span style='color: green'>(-10.0 [-0.5%])</span><div style='text-align: right'> 2,039.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1056/individual/fibonacci-68ea0831d91465505837b07c42245833a60ec6b5.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,615,800 </div>  | <div style='text-align: right'> 3,000,274 </div>  | <span style='color: green'>(-238.0 [-4.3%])</span><div style='text-align: right'> 5,304.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1056/individual/regex-68ea0831d91465505837b07c42245833a60ec6b5.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,890,449 </div>  | <div style='text-align: right'> 8,381,808 </div>  | <span style='color: green'>(-268.0 [-1.5%])</span><div style='text-align: right'> 17,064.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1056/individual/verify_fibair-68ea0831d91465505837b07c42245833a60ec6b5.md) | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+2,400 [+0.0%])</span><div style='text-align: right'> 48,128,287 </div>  | <span style='color: red'>(+200 [+0.1%])</span><div style='text-align: right'> 397,294 </div>  | <span style='color: green'>(-49.0 [-1.6%])</span><div style='text-align: right'> 3,101.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/openvm-org/openvm/commit/68ea0831d91465505837b07c42245833a60ec6b5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12343781864)