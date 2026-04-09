| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/fibonacci-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 3,814 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/keccak-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 18,387 |  18,655,329 |  3,304 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/regex-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 1,410 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/ecrecover-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 647 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/pairing-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 906 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/kitchen_sink-76c929eb6ca73830bf197544b4ec6f1a1b6e9ece.md) | 2,153 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76c929eb6ca73830bf197544b4ec6f1a1b6e9ece

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24216730483)
