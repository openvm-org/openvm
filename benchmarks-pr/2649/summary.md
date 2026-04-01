| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/fibonacci-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 3,798 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/keccak-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 15,649 |  1,235,218 |  2,191 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/regex-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 1,413 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/ecrecover-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 644 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/pairing-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 928 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/kitchen_sink-43ddf08f5d262cbf4e8c1b303923550160e236a1.md) | 2,379 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/43ddf08f5d262cbf4e8c1b303923550160e236a1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23868431315)
