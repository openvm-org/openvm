| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 408 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 8,368 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 3,930 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 574 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 217 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 275 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-369a1d4c7fd1050b007784dea60f47b70ab40e86.md) | 1,915 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/369a1d4c7fd1050b007784dea60f47b70ab40e86

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29458618444)
