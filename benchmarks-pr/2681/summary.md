| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/fibonacci-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 3,812 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/keccak-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 18,233 |  18,655,329 |  3,276 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/regex-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 1,430 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/ecrecover-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 643 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/pairing-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 907 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/kitchen_sink-1263a2c9635c7d74a16f1be6ecd2303f6b613986.md) | 2,146 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1263a2c9635c7d74a16f1be6ecd2303f6b613986

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24214726948)
