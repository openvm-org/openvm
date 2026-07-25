| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 472 |  4,000,051 |  245 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 7,323 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 4,728 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 671 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 315 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-26958397701ade6fe99b8fe83438a5d9151f2442.md) | 2,664 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/26958397701ade6fe99b8fe83438a5d9151f2442

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30151483382)
