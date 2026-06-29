| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 1,024 |  4,000,051 |  385 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 15,705 |  14,365,133 |  3,013 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 8,171 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 1,189 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 433 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 593 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71.md) | 3,876 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9eeb13f16b0ec82dc08e47a5eb8aa01e374f5e71

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28382952230)
