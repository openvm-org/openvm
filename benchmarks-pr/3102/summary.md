| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 476 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 7,323 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 4,086 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 660 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 222 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 240 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-0c2498dceb75104a6281b2003a093d6a23e11ee8.md) | 2,064 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c2498dceb75104a6281b2003a093d6a23e11ee8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31036296482)
