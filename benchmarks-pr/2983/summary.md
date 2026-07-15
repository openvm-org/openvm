| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 408 |  4,000,051 |  224 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 8,356 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 3,997 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 559 |  4,090,656 |  209 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 263 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-0dc7d7f7a74bd6c192ac656e24f074beb5182e48.md) | 1,924 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0dc7d7f7a74bd6c192ac656e24f074beb5182e48

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29418757847)
