| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 412 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 8,559 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 4,225 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 572 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 283 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e.md) | 1,932 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/68fed4d03751d7d3e8e08b0a5d1e4bc12fdcce9e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29846349273)
