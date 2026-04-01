| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/fibonacci-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 3,859 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/keccak-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 15,628 |  1,235,218 |  2,189 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/regex-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 1,442 |  4,136,694 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/ecrecover-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 634 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/pairing-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 919 |  1,745,757 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2649/kitchen_sink-4677959f0f04bac2a47d0fdcdb9014b90b8ce099.md) | 2,375 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4677959f0f04bac2a47d0fdcdb9014b90b8ce099

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23869682046)
