| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 3,874 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 18,371 |  18,655,329 |  3,280 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 1,397 |  4,137,067 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 648 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 903 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-45182cf45ba6a76d937aa93949c95b0b538956a9.md) | 2,278 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/45182cf45ba6a76d937aa93949c95b0b538956a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23878548238)
