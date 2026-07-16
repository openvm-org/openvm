| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 420 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 8,660 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 4,250 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 568 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 221 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 291 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-64ae37f4c358446a29eed8bc41d918f78fb1eee0.md) | 1,925 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64ae37f4c358446a29eed8bc41d918f78fb1eee0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29496306334)
