| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/fibonacci-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 3,738 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/keccak-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 18,579 |  18,655,329 |  3,378 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/sha2_bench-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 10,075 |  14,793,960 |  1,470 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/regex-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 1,411 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/ecrecover-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 600 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/pairing-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 887 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/kitchen_sink-8eeaad13b86ded49a55ffec899d20b685db2a300.md) | 1,874 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8eeaad13b86ded49a55ffec899d20b685db2a300

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26644542012)
