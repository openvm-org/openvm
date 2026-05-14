| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 1,892 |  4,000,051 |  535 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 13,456 |  14,365,133 |  2,223 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 9,433 |  11,167,961 |  1,403 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 1,623 |  4,090,656 |  383 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 643 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 756 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-3e31d3a77f60a62dac4bbcbc793ee57d92a9f843.md) | 2,037 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3e31d3a77f60a62dac4bbcbc793ee57d92a9f843

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25855732030)
