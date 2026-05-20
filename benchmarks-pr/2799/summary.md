| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 3,803 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 18,630 |  18,655,329 |  3,293 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 10,274 |  14,793,960 |  1,467 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 1,393 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 596 |  123,583 |  244 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 886 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-4d3868f023acda5ee2c517a56c089b88d866a266.md) | 1,906 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4d3868f023acda5ee2c517a56c089b88d866a266

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26188806631)
