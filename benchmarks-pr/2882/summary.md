| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/fibonacci-2304127867f6b140f93810071deac7c95d602349.md) | 1,649 |  4,000,051 |  524 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/keccak-2304127867f6b140f93810071deac7c95d602349.md) | 16,420 |  14,365,133 |  3,061 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/sha2_bench-2304127867f6b140f93810071deac7c95d602349.md) | 10,484 |  11,167,961 |  1,944 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/regex-2304127867f6b140f93810071deac7c95d602349.md) | 1,554 |  4,090,656 |  434 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/ecrecover-2304127867f6b140f93810071deac7c95d602349.md) | 481 |  112,210 |  304 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/pairing-2304127867f6b140f93810071deac7c95d602349.md) | 628 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2882/kitchen_sink-2304127867f6b140f93810071deac7c95d602349.md) | 3,930 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2304127867f6b140f93810071deac7c95d602349

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27378771319)
