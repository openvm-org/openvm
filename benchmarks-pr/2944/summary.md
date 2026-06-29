| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 1,034 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 15,584 |  14,365,133 |  2,999 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 8,052 |  11,167,961 |  988 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 1,173 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 433 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 584 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-118745ab6eb6fb33168ff702d1f4b3eab00303a7.md) | 3,871 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/118745ab6eb6fb33168ff702d1f4b3eab00303a7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28384271673)
