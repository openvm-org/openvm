| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 472 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 7,366 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 4,160 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 660 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 231 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-4300150aabc72f349a357f843e9bb00b15e75e97.md) | 2,025 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4300150aabc72f349a357f843e9bb00b15e75e97

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31040583257)
