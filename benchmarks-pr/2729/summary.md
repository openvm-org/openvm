| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/fibonacci-c8893566a502c3112982453530964dd57d8d1dee.md) | 3,850 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/keccak-c8893566a502c3112982453530964dd57d8d1dee.md) | 18,564 |  18,655,329 |  3,334 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/sha2_bench-c8893566a502c3112982453530964dd57d8d1dee.md) | 8,978 |  14,793,960 |  1,386 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/regex-c8893566a502c3112982453530964dd57d8d1dee.md) | 1,424 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/ecrecover-c8893566a502c3112982453530964dd57d8d1dee.md) | 638 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/pairing-c8893566a502c3112982453530964dd57d8d1dee.md) | 911 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2729/kitchen_sink-c8893566a502c3112982453530964dd57d8d1dee.md) | 2,101 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c8893566a502c3112982453530964dd57d8d1dee

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24731081376)
