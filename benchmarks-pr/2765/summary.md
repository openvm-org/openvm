| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 1,872 |  4,000,051 |  513 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 13,637 |  14,365,133 |  2,235 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 9,421 |  11,167,961 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 1,573 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 601 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 730 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-c2fbeabb683502e54d0234915163087cb56ea47e.md) | 1,889 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c2fbeabb683502e54d0234915163087cb56ea47e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887966686)
