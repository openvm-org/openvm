| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/fibonacci-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: green'>(-9 [-0.3%])</span> 3,083 |  12,000,265 | <span style='color: red'>(+7 [+1.0%])</span> 684 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/keccak-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: red'>(+271 [+1.6%])</span> 16,696 |  18,655,329 | <span style='color: red'>(+47 [+1.5%])</span> 3,093 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/sha2_bench-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: red'>(+10 [+0.1%])</span> 9,259 |  14,793,960 | <span style='color: red'>(+3 [+0.3%])</span> 1,131 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/regex-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: green'>(-13 [-1.1%])</span> 1,157 |  4,137,067 | <span style='color: green'>(-6 [-1.7%])</span> 349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/ecrecover-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: red'>(+3 [+0.5%])</span> 597 |  123,583 | <span style='color: red'>(+5 [+1.8%])</span> 283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/pairing-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: red'>(+9 [+1.0%])</span> 943 |  1,745,757 | <span style='color: red'>(+1 [+0.3%])</span> 306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2988/kitchen_sink-c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e.md) |<span style='color: red'>(+9 [+0.2%])</span> 4,139 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0e9f9d9402238ab05cfd8e7f0fd41f5bd23a90e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29038551600)
