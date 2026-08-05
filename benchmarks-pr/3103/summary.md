| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 478 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 7,366 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 4,098 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 662 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 222 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 238 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-17933c0f9a1fe721e9814e9e86e9e455521d439a.md) | 2,046 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/17933c0f9a1fe721e9814e9e86e9e455521d439a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31013266371)
