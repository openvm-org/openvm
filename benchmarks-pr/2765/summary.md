| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 1,888 |  4,000,051 |  537 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 13,575 |  14,365,133 |  2,247 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 9,472 |  11,167,961 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 1,592 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 644 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 757 |  592,827 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-a8a77ba0f707ce54281d23014a4be7fd2a902252.md) | 2,053 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8a77ba0f707ce54281d23014a4be7fd2a902252

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25818599658)
