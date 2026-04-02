| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 3,796 |  12,000,265 |  930 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 18,510 |  18,655,329 |  3,287 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 1,414 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 647 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 898 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-15e303367ff349d2dea2a175e07dde9b311bfc1c.md) | 2,278 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15e303367ff349d2dea2a175e07dde9b311bfc1c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23925489217)
