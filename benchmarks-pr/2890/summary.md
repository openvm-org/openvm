| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/fibonacci-b23d06351940aac0c6af0a65f717383182343ed3.md) | 1,644 |  4,000,051 |  521 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/keccak-b23d06351940aac0c6af0a65f717383182343ed3.md) | 16,086 |  14,365,133 |  2,987 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/sha2_bench-b23d06351940aac0c6af0a65f717383182343ed3.md) | 10,335 |  11,167,961 |  1,929 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/regex-b23d06351940aac0c6af0a65f717383182343ed3.md) | 1,536 |  4,090,656 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/ecrecover-b23d06351940aac0c6af0a65f717383182343ed3.md) | 480 |  112,210 |  314 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/pairing-b23d06351940aac0c6af0a65f717383182343ed3.md) | 625 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/kitchen_sink-b23d06351940aac0c6af0a65f717383182343ed3.md) | 4,001 |  1,979,971 |  877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b23d06351940aac0c6af0a65f717383182343ed3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27577638711)
