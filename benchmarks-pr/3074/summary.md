| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 497 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 7,437 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 4,179 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 697 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 230 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 245 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-3b68d8e86a8353583e178decc534da1ab54386a2.md) | 2,060 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b68d8e86a8353583e178decc534da1ab54386a2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30390525939)
