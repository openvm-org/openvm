| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 3,758 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 18,618 |  18,655,329 |  3,276 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 10,289 |  14,793,960 |  1,472 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 1,410 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 603 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 893 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-db2328ddf364b6be3ad4817631fd69b1fef5a6f7.md) | 1,906 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db2328ddf364b6be3ad4817631fd69b1fef5a6f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26978989753)
