| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 1,573 |  4,000,051 |  440 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 13,984 |  14,365,133 |  2,388 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 9,206 |  11,167,961 |  1,409 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 1,477 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 474 |  112,210 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 607 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-5d9fd3bbd36995cc5d924d3ad412b78457d5720b.md) | 1,810 |  1,979,971 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5d9fd3bbd36995cc5d924d3ad412b78457d5720b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26468323703)
