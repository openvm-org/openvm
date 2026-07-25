| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 474 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 7,273 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 4,749 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 225 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 266 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93.md) | 2,734 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9ea12ad5b6c0beb4e3109fe5d17d9938bce50d93

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30136397786)
