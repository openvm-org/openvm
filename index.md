| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 1,680 |  12,000,265 |  368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 9,610 |  18,655,329 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 5,350 |  14,793,960 |  594 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 689 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 439 |  123,583 |  193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 616 |  1,745,757 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-d1d9db0f3081948cfc1194233d7d2822164076b4.md) | 2,300 |  2,579,903 |  494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d1d9db0f3081948cfc1194233d7d2822164076b4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41873416338)
