| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 1,018 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 16,210 |  14,365,133 |  3,042 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 8,155 |  11,167,961 |  997 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 1,186 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 434 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 595 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-6f8dd748562c597c081710e7fd7eefe824126ed2.md) | 3,879 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6f8dd748562c597c081710e7fd7eefe824126ed2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27992048021)
