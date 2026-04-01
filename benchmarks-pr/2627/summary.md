| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 3,848 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 15,844 |  1,235,218 |  2,228 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 1,410 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 643 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 922 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-9abc53a15677fd8a3500a94e5a6ce81e810b94db.md) | 2,397 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9abc53a15677fd8a3500a94e5a6ce81e810b94db

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23857551504)
