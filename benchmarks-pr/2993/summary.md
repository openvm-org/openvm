| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 471 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 8,788 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 4,037 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 562 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 218 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 278 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-22d76308bd121561bd9d7c798d5f0857760665d0.md) | 1,956 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/22d76308bd121561bd9d7c798d5f0857760665d0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29442894446)
