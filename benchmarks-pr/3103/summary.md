| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 466 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 7,364 |  14,365,133 |  1,513 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 4,159 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 669 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 197 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 236 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-53b4e3398249c5c054f3b92fa3a3b884e75dc677.md) | 2,029 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/53b4e3398249c5c054f3b92fa3a3b884e75dc677

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31719621718)
