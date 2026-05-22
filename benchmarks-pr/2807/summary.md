| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 3,791 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 18,430 |  18,655,329 |  3,261 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 10,173 |  14,793,960 |  1,459 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 1,398 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 596 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 875 |  1,745,757 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-fee16cea2d6cc4705f6deccf5ff43f2f763a385e.md) | 1,898 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fee16cea2d6cc4705f6deccf5ff43f2f763a385e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26285920972)
