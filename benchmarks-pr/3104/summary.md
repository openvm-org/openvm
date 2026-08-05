| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 894 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 8,531 |  14,365,133 |  1,509 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 4,227 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 736 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 501 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 477 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-c175e383636d38012a3fcd0cda097d84bd17c431.md) | 2,353 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c175e383636d38012a3fcd0cda097d84bd17c431

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31048057611)
