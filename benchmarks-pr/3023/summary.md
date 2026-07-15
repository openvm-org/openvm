| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 408 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 8,615 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 4,132 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 501 |  4,090,656 |  197 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 221 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 273 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-3cdc14dd3e021e96f3b8f0eacb02000da4669c1e.md) | 1,882 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3cdc14dd3e021e96f3b8f0eacb02000da4669c1e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29438095212)
