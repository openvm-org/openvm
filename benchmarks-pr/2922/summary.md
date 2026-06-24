| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 1,030 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 15,294 |  14,365,133 |  3,024 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 7,825 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 1,148 |  4,090,656 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 442 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 562 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616.md) | 3,796 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c58aa7a5311b3f7b6b92fb594d8e1675ec1a616

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28083458181)
