| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/fibonacci-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: green'>(-14 [-0.8%])</span> 1,665 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/keccak-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: red'>(+19 [+0.2%])</span> 9,539 |  18,655,329 | <span style='color: red'>(+15 [+1.0%])</span> 1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/sha2_bench-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: red'>(+9 [+0.2%])</span> 5,329 |  14,793,960 | <span style='color: green'>(-1 [-0.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/regex-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: green'>(-10 [-1.4%])</span> 688 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/ecrecover-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: green'>(-10 [-2.3%])</span> 429 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/pairing-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: red'>(+2 [+0.3%])</span> 579 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/kitchen_sink-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) |<span style='color: green'>(-27 [-1.2%])</span> 2,295 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 494 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/fibonacci_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) | 1,773 |  12,000,265 |  362 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/regex_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) | 834 |  4,137,067 |  210 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/ecrecover_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) | 519 |  123,583 |  180 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/pairing_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) | 706 |  1,745,757 |  188 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3168/kitchen_sink_e2e-367f6b3bbd5e155db54af5e3b36562518adac4c7.md) | 2,545 |  2,579,903 |  488 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/367f6b3bbd5e155db54af5e3b36562518adac4c7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38793620937)
