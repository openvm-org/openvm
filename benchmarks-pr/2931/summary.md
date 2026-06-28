| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 1,030 |  4,000,051 |  401 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 15,713 |  14,365,133 |  3,016 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 8,213 |  11,167,961 |  1,009 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 1,184 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 439 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 589 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea.md) | 3,892 |  1,979,971 |  866 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fb178e1cbb246c4cbeb9ea76d40e8c47ffd075ea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28321471843)
