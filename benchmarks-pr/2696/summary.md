| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/fibonacci-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 3,857 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/keccak-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 18,523 |  18,655,329 |  3,311 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/sha2_bench-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 9,840 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/regex-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 1,436 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/ecrecover-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 648 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/pairing-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 935 |  1,745,757 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/kitchen_sink-e580ac35a02ea67f0c3114d3baccc5c1433bd79f.md) | 2,141 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e580ac35a02ea67f0c3114d3baccc5c1433bd79f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24263947627)
