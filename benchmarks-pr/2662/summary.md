| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 3,889 |  12,000,265 |  962 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 18,858 |  18,655,329 |  3,359 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/sha2_bench-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 9,959 |  14,793,960 |  1,413 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 1,429 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 649 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 902 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-ad81c01c6c1365f4ab8950f7b564741faf7e75db.md) | 2,157 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ad81c01c6c1365f4ab8950f7b564741faf7e75db

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24266668176)
