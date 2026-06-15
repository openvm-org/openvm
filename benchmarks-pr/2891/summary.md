| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/fibonacci-1052660ebf00014204513c7a0274725f1b92867c.md) | 1,661 |  4,000,051 |  528 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/keccak-1052660ebf00014204513c7a0274725f1b92867c.md) | 16,195 |  14,365,133 |  2,995 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/sha2_bench-1052660ebf00014204513c7a0274725f1b92867c.md) | 10,476 |  11,167,961 |  1,954 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/regex-1052660ebf00014204513c7a0274725f1b92867c.md) | 1,517 |  4,090,656 |  424 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/ecrecover-1052660ebf00014204513c7a0274725f1b92867c.md) | 481 |  112,210 |  308 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/pairing-1052660ebf00014204513c7a0274725f1b92867c.md) | 627 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2891/kitchen_sink-1052660ebf00014204513c7a0274725f1b92867c.md) | 3,880 |  1,979,971 |  846 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1052660ebf00014204513c7a0274725f1b92867c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27577973015)
