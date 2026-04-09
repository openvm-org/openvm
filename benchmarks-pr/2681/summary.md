| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/fibonacci-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 3,831 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/keccak-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 18,614 |  18,655,329 |  3,334 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/regex-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 1,412 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/ecrecover-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 647 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/pairing-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 909 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/kitchen_sink-ee9fcd21c3fffc307bec7a808be0aac5cc855fef.md) | 2,156 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee9fcd21c3fffc307bec7a808be0aac5cc855fef

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24215551746)
