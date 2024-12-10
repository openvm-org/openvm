### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/881/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 14,987,538 </div>  | <div style='text-align: right'> 217,352 </div>  | <span style='color: red'>(+12.0 [+0.5%])</span><div style='text-align: right'> 2,515.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/881/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,516,960 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-32.0 [-0.5%])</span><div style='text-align: right'> 6,382.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/881/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,516,960 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-230.0 [-3.2%])</span><div style='text-align: right'> 6,860.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/881/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,661,728 </div>  | <div style='text-align: right'> 4,181,142 </div>  | <span style='color: red'>(+205.0 [+0.8%])</span><div style='text-align: right'> 26,912.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/881/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-1,150 [-0.0%])</span><div style='text-align: right'> 8,295,475 </div>  | <span style='color: green'>(-87 [-0.0%])</span><div style='text-align: right'> 198,527 </div>  | <span style='color: red'>(+4.0 [+0.3%])</span><div style='text-align: right'> 1,462.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/978912864ee22a032a01475d004b5267ba5ee616

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12075235410)