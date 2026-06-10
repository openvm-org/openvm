| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-395639ecf49663963d8c8e927caf497a65beb25c.md) | 5,354 |  4,000,051 |  526 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-395639ecf49663963d8c8e927caf497a65beb25c.md) | 20,611 |  14,365,133 |  3,066 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-395639ecf49663963d8c8e927caf497a65beb25c.md) | 14,115 |  11,167,961 |  1,933 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-395639ecf49663963d8c8e927caf497a65beb25c.md) | 3,781 |  4,090,656 |  436 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-395639ecf49663963d8c8e927caf497a65beb25c.md) | 1,988 |  112,210 |  317 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-395639ecf49663963d8c8e927caf497a65beb25c.md) | 2,090 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-395639ecf49663963d8c8e927caf497a65beb25c.md) | 5,632 |  1,979,971 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/395639ecf49663963d8c8e927caf497a65beb25c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27309884561)
