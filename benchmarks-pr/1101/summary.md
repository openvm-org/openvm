### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ ecrecover_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1101/individual/ecrecover-529d365ff982c7d76e3375bbd5ea7ddef4231f69.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,251,804 </div>  | <div style='text-align: right'> 195,066 </div>  | <span style='color: green'>(-93.0 [-4.7%])</span><div style='text-align: right'> 1,896.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1101/individual/fibonacci-529d365ff982c7d76e3375bbd5ea7ddef4231f69.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,615,800 </div>  | <div style='text-align: right'> 3,000,274 </div>  | <div style='text-align: right'> 5,526.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+5,640 [+0.0%])</span><div style='text-align: right'> 144,225,163 </div>  | <span style='color: red'>(+1,170 [+0.0%])</span><div style='text-align: right'> 7,038,744 </div>  | <span style='color: red'>(+41.0 [+0.3%])</span><div style='text-align: right'> 14,463.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1101/individual/regex-529d365ff982c7d76e3375bbd5ea7ddef4231f69.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,890,449 </div>  | <div style='text-align: right'> 8,381,808 </div>  | <span style='color: green'>(-98.0 [-0.6%])</span><div style='text-align: right'> 17,195.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+33,620 [+0.0%])</span><div style='text-align: right'> 315,460,467 </div>  | <span style='color: red'>(+6,248 [+0.0%])</span><div style='text-align: right'> 14,646,418 </div>  | <span style='color: green'>(-80.0 [-0.3%])</span><div style='text-align: right'> 29,803.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1101/individual/verify_fibair-529d365ff982c7d76e3375bbd5ea7ddef4231f69.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 48,127,147 </div>  | <div style='text-align: right'> 397,164 </div>  | <span style='color: green'>(-68.0 [-2.1%])</span><div style='text-align: right'> 3,099.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/openvm-org/openvm/commit/529d365ff982c7d76e3375bbd5ea7ddef4231f69

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12361651366)