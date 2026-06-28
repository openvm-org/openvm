| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 1,048 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 16,010 |  14,365,133 |  3,072 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 8,161 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 1,164 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 435 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 586 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e.md) | 3,911 |  1,979,971 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a1cdbc86f2ed790f089e4bc9abc1ef0080eacb0e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28322393389)
