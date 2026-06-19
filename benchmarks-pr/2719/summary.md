| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 1,394 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 16,330 |  14,365,133 |  3,045 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 10,115 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 1,574 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 435 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 599 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 3,909 |  1,979,971 |  867 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 729 |  4,000,051 |  182 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 965 |  4,090,656 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 319 |  112,210 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 409 |  592,827 |  143 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-0b8d705e9d75b834e33ea98311d18879ee60e3ac.md) | 1,945 |  1,979,971 |  371 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0b8d705e9d75b834e33ea98311d18879ee60e3ac

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27820134533)
