| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 408 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 8,508 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 3,923 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 572 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 218 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 265 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-e9368df61e8284a6dec7402a086d17ae63aa72a3.md) | 1,881 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e9368df61e8284a6dec7402a086d17ae63aa72a3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29446037116)
