| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 447 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 7,335 |  14,365,133 |  1,630 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 4,047 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 729 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 203 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 245 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-0f0d30689e8cdf61818e1dfceaac23ee616d06fe.md) | 2,138 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0f0d30689e8cdf61818e1dfceaac23ee616d06fe

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33116926732)
