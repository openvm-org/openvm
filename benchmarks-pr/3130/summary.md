| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/fibonacci-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 476 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/keccak-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 7,701 |  14,365,133 |  1,651 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/sha2_bench-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 4,413 |  11,167,961 |  538 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/regex-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 758 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/ecrecover-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 206 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/pairing-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 246 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3130/kitchen_sink-3f79c611f9c5a2d689249d13bdb419a2ff73f0ba.md) | 2,264 |  1,979,971 |  481 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f79c611f9c5a2d689249d13bdb419a2ff73f0ba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32528529397)
