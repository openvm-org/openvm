| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/fibonacci-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 3,804 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/keccak-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 18,473 |  18,655,329 |  3,324 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/regex-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 1,418 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/ecrecover-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 648 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/pairing-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 910 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2680/kitchen_sink-8ff6d42426453b18895d5832f5b622d543c618e2.md) | 2,150 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8ff6d42426453b18895d5832f5b622d543c618e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24180731005)
