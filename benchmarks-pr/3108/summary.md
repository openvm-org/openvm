| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 463 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 7,236 |  14,365,133 |  1,603 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 4,035 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 740 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 205 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 246 |  592,827 |  170 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-bb7db866a69e01aad8b1d17d179085e9cc4406f7.md) | 2,138 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bb7db866a69e01aad8b1d17d179085e9cc4406f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33117940993)
