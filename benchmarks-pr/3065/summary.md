| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/fibonacci-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 471 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/keccak-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 7,311 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/sha2_bench-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 4,780 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/regex-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 670 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/ecrecover-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/pairing-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 317 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3065/kitchen_sink-e31e139c363fc6bafa366dcf53b063d5f55122e1.md) | 2,674 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e31e139c363fc6bafa366dcf53b063d5f55122e1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30010232867)
