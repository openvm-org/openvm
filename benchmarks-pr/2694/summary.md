| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/fibonacci-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 3,833 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/keccak-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 18,733 |  18,655,329 |  3,357 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/regex-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 1,421 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/ecrecover-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 645 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/pairing-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 911 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/kitchen_sink-8de22c7fb43862593b2d2117e0e1cbb380eb58df.md) | 2,173 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8de22c7fb43862593b2d2117e0e1cbb380eb58df

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24253502796)
