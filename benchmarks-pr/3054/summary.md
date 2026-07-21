| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 475 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 7,263 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 4,700 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 659 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 228 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 311 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-ff36d6f3c346ae03c0c9ba77884bd2262c88f429.md) | 2,681 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ff36d6f3c346ae03c0c9ba77884bd2262c88f429

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29864631168)
