| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 1,034 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 15,676 |  14,365,133 |  3,031 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 8,306 |  11,167,961 |  1,021 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 1,147 |  4,090,656 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 433 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 588 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-9d8f55c6058b3e244da0e2a6923ec413d8d62a83.md) | 3,846 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9d8f55c6058b3e244da0e2a6923ec413d8d62a83

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28287758007)
