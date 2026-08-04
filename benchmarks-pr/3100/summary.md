| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/fibonacci-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 470 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/keccak-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 7,470 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/sha2_bench-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 4,157 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/regex-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 658 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/ecrecover-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 227 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/pairing-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 235 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3100/kitchen_sink-6325e91d90fc9d3c4ddb88ef54a1a25037822759.md) | 2,019 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6325e91d90fc9d3c4ddb88ef54a1a25037822759

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30936210110)
