### Benchmarks
| group | app_log_blowup | app_total_cells_used | app_total_cycles | app_total_proof_time_ms | leaf_log_blowup | leaf_total_cells_used | leaf_total_cycles | leaf_total_proof_time_ms | max_segment_length | instance | alloc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [ base64_json_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/base64_json-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 15,116,803 </div>  | <div style='text-align: right'> 217,347 </div>  | <span style='color: red'>(+4.0 [+0.2%])</span><div style='text-align: right'> 1,953.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+1,720 [+0.0%])</span><div style='text-align: right'> 294,988,239 </div>  | <span style='color: red'>(+67 [+0.0%])</span><div style='text-align: right'> 6,788,741 </div>  | <span style='color: red'>(+152.0 [+0.6%])</span><div style='text-align: right'> 25,931.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ ecrecover_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/ecrecover-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 10,567,808 </div>  | <div style='text-align: right'> 106,444 </div>  | <span style='color: green'>(-357.0 [-16.5%])</span><div style='text-align: right'> 1,803.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/fibonacci-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,612,244 </div>  | <div style='text-align: right'> 1,500,137 </div>  | <span style='color: green'>(-24.0 [-0.5%])</span><div style='text-align: right'> 5,094.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-9,570 [-0.0%])</span><div style='text-align: right'> 144,222,643 </div>  | <span style='color: green'>(-824 [-0.0%])</span><div style='text-align: right'> 3,519,155 </div>  | <span style='color: green'>(-241.0 [-1.8%])</span><div style='text-align: right'> 13,076.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ fibonacci_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/fibonacci-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 51,612,244 </div>  | <div style='text-align: right'> 1,500,137 </div>  | <span style='color: green'>(-24.0 [-0.5%])</span><div style='text-align: right'> 5,094.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-9,570 [-0.0%])</span><div style='text-align: right'> 144,222,643 </div>  | <span style='color: green'>(-824 [-0.0%])</span><div style='text-align: right'> 3,519,155 </div>  | <span style='color: green'>(-241.0 [-1.8%])</span><div style='text-align: right'> 13,076.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ regex_program ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/regex-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <div style='text-align: right'> 238,871,537 </div>  | <div style='text-align: right'> 4,190,904 </div>  | <span style='color: green'>(-113.0 [-0.7%])</span><div style='text-align: right'> 16,346.0 </div>  | <div style='text-align: right'> 2 </div>  | <span style='color: red'>(+46,500 [+0.0%])</span><div style='text-align: right'> 315,498,277 </div>  | <span style='color: red'>(+4,398 [+0.1%])</span><div style='text-align: right'> 7,326,829 </div>  | <span style='color: red'>(+378.0 [+1.4%])</span><div style='text-align: right'> 26,506.0 </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |
| [ verify_fibair ](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/benchmarks-pr/989/individual/verify_fibair-0fe772cc46de634a121dedfd7631932dddec7947.md) | <div style='text-align: right'> 2 </div>  | <span style='color: green'>(-1,140 [-0.0%])</span><div style='text-align: right'> 48,127,147 </div>  | <span style='color: green'>(-65 [-0.0%])</span><div style='text-align: right'> 198,582 </div>  | <span style='color: green'>(-27.0 [-0.9%])</span><div style='text-align: right'> 2,890.0 </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | <div style='text-align: right'> - </div>  | 1048476 | 64cpu-linux-arm64 | mimalloc |


Commit: https://github.com/axiom-crypto/afs-prototype/commit/0fe772cc46de634a121dedfd7631932dddec7947

[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/12284155522)