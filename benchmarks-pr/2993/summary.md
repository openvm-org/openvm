| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 861 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 15,554 |  14,365,133 |  3,007 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 7,723 |  11,167,961 |  988 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 1,024 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 297 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 434 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-9491e6cca57106cdf9bcc632a64e05fa3a01568f.md) | 3,707 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9491e6cca57106cdf9bcc632a64e05fa3a01568f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29278155862)
