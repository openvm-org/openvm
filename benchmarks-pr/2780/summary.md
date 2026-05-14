| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/fibonacci-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 1,876 |  4,000,051 |  527 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/keccak-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 13,535 |  14,365,133 |  2,226 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/sha2_bench-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 9,484 |  11,167,961 |  1,410 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/regex-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 1,614 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/ecrecover-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 642 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/pairing-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 759 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/kitchen_sink-9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7.md) | 2,050 |  1,979,971 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c8ce6e816ab3a26750ae1bee15b3f9ad59f41b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25881994529)
