| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 476 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 7,725 |  14,365,133 |  1,672 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 4,326 |  11,167,961 |  535 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 773 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 208 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 249 |  592,827 |  169 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-a48d819c8288c8cb191558bd5b85f9ed96c9a868.md) | 2,246 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a48d819c8288c8cb191558bd5b85f9ed96c9a868

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33172034730)
