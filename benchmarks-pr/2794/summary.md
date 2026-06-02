| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 1,557 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 14,227 |  14,365,133 |  2,413 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 9,378 |  11,167,961 |  1,423 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 1,634 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 491 |  112,210 |  262 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 606 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-00ad064874d7e5069dc939cd0fb6ee909dea4d24.md) | 1,850 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/00ad064874d7e5069dc939cd0fb6ee909dea4d24

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26853653737)
