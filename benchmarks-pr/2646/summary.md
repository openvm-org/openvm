| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 3,891 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 18,489 |  18,655,329 |  3,311 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 1,420 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 736 |  317,792 |  352 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 903 |  1,745,757 |  314 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-70aaf4d60a7d039d46ceef671109938b4bacfd17.md) | 2,489 |  2,580,026 |  542 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/70aaf4d60a7d039d46ceef671109938b4bacfd17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23901775310)
