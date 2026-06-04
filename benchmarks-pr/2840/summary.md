| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 1,387 |  4,000,051 |  428 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 13,586 |  14,365,133 |  2,375 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 8,930 |  11,167,961 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 1,516 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 433 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 570 |  592,827 |  250 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-3f8fd410837aaaff9b305fe519f733aa37fc52b1.md) | 3,699 |  1,979,971 |  936 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f8fd410837aaaff9b305fe519f733aa37fc52b1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26959899807)
