### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/832/individual/base64_json-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 14,996,988 </div>  | <div style='text-align: right'> 217,352 </div>  | <span style='color: red'>(+63.0 [+2.5%])</span><div style='text-align: right'> 2,552.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/832/individual/fibonacci-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,512,218 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: red'>(+66.0 [+1.0%])</span><div style='text-align: right'> 6,460.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/832/individual/fibonacci-2-2-64cpu-linux-x64-jemalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,512,218 </div>  | <div style='text-align: right'> 1,500,219 </div>  | <span style='color: red'>(+85.0 [+1.3%])</span><div style='text-align: right'> 6,830.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-x64 | jemalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/832/individual/regex-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,565,138 </div>  | <div style='text-align: right'> 4,181,072 </div>  | <span style='color: red'>(+534.0 [+2.0%])</span><div style='text-align: right'> 27,366.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/832/individual/verify_fibair-2-2-64cpu-linux-arm64-mimalloc.md) | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+9,772 [+0.1%])</span><div style='text-align: right'> 8,306,247 </div>  | <span style='color: red'>(+554 [+0.3%])</span><div style='text-align: right'> 199,167 </div>  | <span style='color: red'>(+35.0 [+2.4%])</span><div style='text-align: right'> 1,470.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/508ee62c3634cad9f28d2bf52ff3434d8f89c659

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12041327054)