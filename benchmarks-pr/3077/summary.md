| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/fibonacci-c44f0634551311460e1b4852b9948256c07b772c.md) | 469 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/keccak-c44f0634551311460e1b4852b9948256c07b772c.md) | 7,293 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/sha2_bench-c44f0634551311460e1b4852b9948256c07b772c.md) | 4,746 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/regex-c44f0634551311460e1b4852b9948256c07b772c.md) | 667 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/ecrecover-c44f0634551311460e1b4852b9948256c07b772c.md) | 232 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/pairing-c44f0634551311460e1b4852b9948256c07b772c.md) | 307 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3077/kitchen_sink-c44f0634551311460e1b4852b9948256c07b772c.md) | 2,679 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c44f0634551311460e1b4852b9948256c07b772c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30310602104)
