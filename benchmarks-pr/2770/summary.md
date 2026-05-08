| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 3,864 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 18,526 |  18,655,329 |  3,346 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 8,997 |  14,793,960 |  1,393 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 1,409 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 633 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 892 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-48088a48b9eece132762f29fd5b6d43d1113700a.md) | 2,080 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/48088a48b9eece132762f29fd5b6d43d1113700a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25578529671)
