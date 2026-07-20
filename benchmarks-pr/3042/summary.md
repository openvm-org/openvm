| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 413 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 8,698 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 4,238 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 583 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 221 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 292 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-4ffe89780ea5702d65e0cd6336cf4e0f2db73651.md) | 1,914 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4ffe89780ea5702d65e0cd6336cf4e0f2db73651

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29764747609)
