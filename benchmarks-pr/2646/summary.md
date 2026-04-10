| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 3,858 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 18,598 |  18,655,329 |  3,335 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 1,426 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 729 |  317,792 |  354 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 912 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0.md) | 2,354 |  2,580,026 |  780 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc1f2e616c72a6ad76a0e8d1e9b52fd321d198b0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24244865811)
