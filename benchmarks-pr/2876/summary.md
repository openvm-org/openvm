| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 1,654 |  4,000,051 |  533 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 16,188 |  14,365,133 |  3,003 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 10,539 |  11,167,961 |  1,955 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 1,536 |  4,090,656 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 484 |  112,210 |  318 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 623 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-5026b7e11933f1328e66893a96bc547be64a2b30.md) | 3,946 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5026b7e11933f1328e66893a96bc547be64a2b30

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27374002374)
