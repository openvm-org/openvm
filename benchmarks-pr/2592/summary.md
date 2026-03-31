| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 3,920 |  12,000,265 |  963 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 18,320 |  18,655,329 |  3,265 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 1,424 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 646 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 901 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-28d01d234d8b79ef6def2c0139ab9f6eb9f26336.md) | 2,275 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/28d01d234d8b79ef6def2c0139ab9f6eb9f26336

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23805703200)
