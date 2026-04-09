| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 3,872 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 18,484 |  18,655,329 |  3,290 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 1,410 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 644 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 909 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca.md) | 2,154 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15dc11ddfff8a2cae0332e0f3facf7f7a1f98fca

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24185643681)
