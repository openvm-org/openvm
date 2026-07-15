| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 414 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 8,340 |  14,365,133 |  1,503 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 3,934 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 573 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 218 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 267 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5.md) | 1,898 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/61ea0b9f45bfe2aa92d1f70b26ca24abbf98d2b5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29453996825)
