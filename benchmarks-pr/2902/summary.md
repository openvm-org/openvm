| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/fibonacci-5614637008c0a050205855fd477b932d391007db.md) | 1,378 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/keccak-5614637008c0a050205855fd477b932d391007db.md) | 16,384 |  14,365,133 |  3,043 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/sha2_bench-5614637008c0a050205855fd477b932d391007db.md) | 8,720 |  11,167,961 |  1,141 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/regex-5614637008c0a050205855fd477b932d391007db.md) | 1,473 |  4,090,656 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/ecrecover-5614637008c0a050205855fd477b932d391007db.md) | 476 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/pairing-5614637008c0a050205855fd477b932d391007db.md) | 615 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2902/kitchen_sink-5614637008c0a050205855fd477b932d391007db.md) | 3,943 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5614637008c0a050205855fd477b932d391007db

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27727043152)
