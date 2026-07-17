| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 413 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 8,701 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 4,219 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 569 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 223 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 285 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-a31c0f0d33dc4a97926c23999ea264df0ec11c4a.md) | 2,015 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a31c0f0d33dc4a97926c23999ea264df0ec11c4a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29609136770)
