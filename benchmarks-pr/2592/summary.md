| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 3,810 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 18,375 |  18,655,329 |  3,286 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 1,435 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 897 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-5e7f3020cbdbedc7537bb5e9acc3efa707e233f2.md) | 2,150 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5e7f3020cbdbedc7537bb5e9acc3efa707e233f2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24180188668)
