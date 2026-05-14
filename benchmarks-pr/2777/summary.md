| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 1,889 |  4,000,051 |  466 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 14,256 |  14,365,133 |  2,270 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 8,300 |  11,167,961 |  920 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 1,601 |  4,090,656 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 645 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 766 |  592,827 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-e592b7583bbd8494cfd51a6cc37b94b6c954bef6.md) | 2,033 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e592b7583bbd8494cfd51a6cc37b94b6c954bef6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25885612206)
