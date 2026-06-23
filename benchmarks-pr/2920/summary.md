| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/fibonacci-faf65e19b255c6d4491452e3dddc03467829309a.md) | 1,043 |  4,000,051 |  405 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/keccak-faf65e19b255c6d4491452e3dddc03467829309a.md) | 16,486 |  14,365,133 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/sha2_bench-faf65e19b255c6d4491452e3dddc03467829309a.md) | 8,186 |  11,167,961 |  991 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/regex-faf65e19b255c6d4491452e3dddc03467829309a.md) | 1,212 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/ecrecover-faf65e19b255c6d4491452e3dddc03467829309a.md) | 436 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/pairing-faf65e19b255c6d4491452e3dddc03467829309a.md) | 598 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/kitchen_sink-faf65e19b255c6d4491452e3dddc03467829309a.md) | 3,890 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/faf65e19b255c6d4491452e3dddc03467829309a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28062675275)
