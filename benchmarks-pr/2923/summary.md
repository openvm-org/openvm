| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-d4891d7b5a1c6895f076030261807dd27b180230.md) | 1,020 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-d4891d7b5a1c6895f076030261807dd27b180230.md) | 15,715 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-d4891d7b5a1c6895f076030261807dd27b180230.md) | 8,141 |  11,167,961 |  997 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-d4891d7b5a1c6895f076030261807dd27b180230.md) | 1,164 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-d4891d7b5a1c6895f076030261807dd27b180230.md) | 439 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-d4891d7b5a1c6895f076030261807dd27b180230.md) | 594 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-d4891d7b5a1c6895f076030261807dd27b180230.md) | 3,889 |  1,979,971 |  865 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d4891d7b5a1c6895f076030261807dd27b180230

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28251831783)
