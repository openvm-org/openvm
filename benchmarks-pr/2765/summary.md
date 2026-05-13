| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 1,891 |  4,000,051 |  531 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 13,449 |  14,365,133 |  2,215 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 9,657 |  11,167,961 |  1,440 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 1,592 |  4,090,656 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 648 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 751 |  592,827 |  271 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-235f041d3d8ece127e682a4ca39b3ac053668de8.md) | 2,041 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/235f041d3d8ece127e682a4ca39b3ac053668de8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25825222103)
