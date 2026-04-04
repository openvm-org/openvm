| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 3,845 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 18,908 |  18,655,329 |  3,388 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 1,424 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 647 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 906 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c.md) | 2,297 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/133ce7d4ed27ee20c8bf4a8e59bf76694cedfe5c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23968721364)
