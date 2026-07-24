| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 475 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 7,303 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 4,756 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 676 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 229 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 331 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-175af8357901b346464fb6c2d5a33caa818da3e2.md) | 2,656 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/175af8357901b346464fb6c2d5a33caa818da3e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30114575348)
