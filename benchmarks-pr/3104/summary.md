| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 459 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 7,180 |  14,365,133 |  1,583 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 4,026 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 723 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 209 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 240 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-4ecf7c68d03a31c88b442dd15002ab8dc6c1564a.md) | 2,143 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4ecf7c68d03a31c88b442dd15002ab8dc6c1564a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32505627125)
