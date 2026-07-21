| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-2ba603fb7751a365bf20973013cf6361617736bb.md) | 412 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-2ba603fb7751a365bf20973013cf6361617736bb.md) | 8,647 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-2ba603fb7751a365bf20973013cf6361617736bb.md) | 4,213 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-2ba603fb7751a365bf20973013cf6361617736bb.md) | 561 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-2ba603fb7751a365bf20973013cf6361617736bb.md) | 224 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-2ba603fb7751a365bf20973013cf6361617736bb.md) | 297 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-2ba603fb7751a365bf20973013cf6361617736bb.md) | 1,919 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ba603fb7751a365bf20973013cf6361617736bb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29816591781)
