| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 3,828 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 18,876 |  18,655,329 |  3,385 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 1,416 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 646 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 907 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-a41be85226f7cb6c8b985ba44c998be46845dc2c.md) | 2,152 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a41be85226f7cb6c8b985ba44c998be46845dc2c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24242667379)
