| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 464 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 7,371 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 4,152 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 665 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 195 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 235 |  592,827 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-74c3590d3f7e5ef21ab07209fdd413343b778f4f.md) | 2,042 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/74c3590d3f7e5ef21ab07209fdd413343b778f4f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31626222343)
