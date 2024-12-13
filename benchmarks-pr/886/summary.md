### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,118,610 </div>  | <div style='text-align: right'> 217,352 </div>  | <span style='color: red'>(+10.0 [+0.4%])</span><div style='text-align: right'> 2,654.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/ecrecover-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 306,788,650 </div>  | <div style='text-align: right'> 5,788,311 </div>  | <span style='color: green'>(-192.0 [-0.5%])</span><div style='text-align: right'> 37,968.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,648,032 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: red'>(+10.0 [+0.2%])</span><div style='text-align: right'> 6,431.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,648,032 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: green'>(-312.0 [-4.4%])</span><div style='text-align: right'> 6,744.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,699,490 </div>  | <div style='text-align: right'> 4,181,278 </div>  | <span style='color: green'>(-202.0 [-0.7%])</span><div style='text-align: right'> 26,927.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/886/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-2,080 [-0.0%])</span><div style='text-align: right'> 8,426,027 </div>  | <span style='color: green'>(-145 [-0.1%])</span><div style='text-align: right'> 198,524 </div>  | <span style='color: green'>(-45.0 [-2.7%])</span><div style='text-align: right'> 1,646.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/f331767a82f70e122320dca5fe09ba97064044b6

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12090957645)