| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 3,858 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 18,547 |  18,655,329 |  3,317 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 1,418 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 646 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 907 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-6a2696a6a0c59f4aa6422c69eab1250764b5e211.md) | 2,163 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6a2696a6a0c59f4aa6422c69eab1250764b5e211

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24258050128)
