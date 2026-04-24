| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/fibonacci-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 3,833 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/keccak-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 15,793 |  1,235,218 |  2,215 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/regex-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 1,436 |  4,136,694 |  383 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/ecrecover-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 637 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/pairing-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 921 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2742/kitchen_sink-30fdbd84bc0138b0c95d7c32b7f1fa49272caea0.md) | 2,371 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/30fdbd84bc0138b0c95d7c32b7f1fa49272caea0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24845606920)
