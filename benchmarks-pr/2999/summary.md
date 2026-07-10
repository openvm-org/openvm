| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/fibonacci-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 2,990 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/keccak-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 16,593 |  18,655,329 |  3,065 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/sha2_bench-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 9,489 |  14,793,960 |  1,139 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/regex-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 1,213 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/ecrecover-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 513 |  123,583 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/pairing-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 856 |  1,745,757 |  310 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/kitchen_sink-a0c933c91f4e58b6dabd78def3f6603c60a72fb6.md) | 4,527 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a0c933c91f4e58b6dabd78def3f6603c60a72fb6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29069294769)
