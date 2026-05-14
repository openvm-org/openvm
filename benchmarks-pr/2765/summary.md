| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-4368df0765287de2fe5572070266dd2d20d89c04.md) | 1,905 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-4368df0765287de2fe5572070266dd2d20d89c04.md) | 13,524 |  14,365,133 |  2,237 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-4368df0765287de2fe5572070266dd2d20d89c04.md) | 9,530 |  11,167,961 |  1,432 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-4368df0765287de2fe5572070266dd2d20d89c04.md) | 1,606 |  4,090,656 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-4368df0765287de2fe5572070266dd2d20d89c04.md) | 639 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-4368df0765287de2fe5572070266dd2d20d89c04.md) | 751 |  592,827 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-4368df0765287de2fe5572070266dd2d20d89c04.md) | 2,032 |  1,979,971 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4368df0765287de2fe5572070266dd2d20d89c04

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25854903461)
