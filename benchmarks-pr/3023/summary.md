| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 410 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 8,600 |  14,365,133 |  1,550 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 4,106 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 504 |  4,090,656 |  195 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 222 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 271 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-0cf3291fa6da0b37fda964456a9050fb61f3364c.md) | 1,903 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0cf3291fa6da0b37fda964456a9050fb61f3364c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29454774721)
