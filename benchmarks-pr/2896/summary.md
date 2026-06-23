| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 1,032 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 16,067 |  14,365,133 |  3,003 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 8,262 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 1,193 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 434 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 599 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-7a71cdb160bf7034c022ac927555b4fa66d66edd.md) | 3,905 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a71cdb160bf7034c022ac927555b4fa66d66edd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28059239465)
