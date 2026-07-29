| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/fibonacci-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 475 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/keccak-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 7,774 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/sha2_bench-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 4,758 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/regex-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 684 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/ecrecover-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 225 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/pairing-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 264 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3084/kitchen_sink-16cdf3e125e46032273f72f85d0a0fba7b5f92bc.md) | 2,715 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16cdf3e125e46032273f72f85d0a0fba7b5f92bc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30492508360)
