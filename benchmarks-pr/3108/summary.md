| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 459 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 7,419 |  14,365,133 |  1,632 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 4,074 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 707 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 210 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 242 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-95be5f9007b90e7384982b1db3b3f55ef970cea1.md) | 2,132 |  1,979,971 |  453 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95be5f9007b90e7384982b1db3b3f55ef970cea1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32514386707)
