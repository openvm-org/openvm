| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 861 |  4,000,051 |  400 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 15,391 |  14,365,133 |  3,020 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 8,059 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 1,032 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 303 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 455 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-19d9bfbb68788d92e3b62338a1ed2ce1163f2920.md) | 3,732 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19d9bfbb68788d92e3b62338a1ed2ce1163f2920

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28978154616)
