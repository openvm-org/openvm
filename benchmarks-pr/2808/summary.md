| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-2164e1e8f7519b2917477524e91199e959e82e22.md) | 1,572 |  4,000,051 |  432 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-2164e1e8f7519b2917477524e91199e959e82e22.md) | 13,819 |  14,365,133 |  2,369 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-2164e1e8f7519b2917477524e91199e959e82e22.md) | 9,141 |  11,167,961 |  1,394 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-2164e1e8f7519b2917477524e91199e959e82e22.md) | 1,474 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-2164e1e8f7519b2917477524e91199e959e82e22.md) | 476 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-2164e1e8f7519b2917477524e91199e959e82e22.md) | 604 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-2164e1e8f7519b2917477524e91199e959e82e22.md) | 1,812 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2164e1e8f7519b2917477524e91199e959e82e22

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26294279169)
