| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 3,823 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 19,169 |  18,655,329 |  3,362 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 9,111 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 1,432 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 640 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 908 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-cf9833e7e42b063bc70d2e76f3b7b8461417953c.md) | 2,035 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cf9833e7e42b063bc70d2e76f3b7b8461417953c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25764679644)
