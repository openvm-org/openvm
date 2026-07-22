| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 475 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 7,317 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 4,735 |  11,167,961 |  543 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 235 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 319 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-fa3c86f546e43a39b834e0f69258d297d3538818.md) | 2,670 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fa3c86f546e43a39b834e0f69258d297d3538818

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29956101928)
