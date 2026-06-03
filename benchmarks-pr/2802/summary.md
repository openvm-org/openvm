| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 1,585 |  4,000,051 |  446 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 13,909 |  14,365,133 |  2,359 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 9,241 |  11,167,961 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 1,607 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 480 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 610 |  592,827 |  254 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-12c7f2d7331cea66b51eedbfbb2d0814fcb8389d.md) | 2,157 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/12c7f2d7331cea66b51eedbfbb2d0814fcb8389d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26878867194)
