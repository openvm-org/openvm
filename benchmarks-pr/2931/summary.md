| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 1,036 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 15,918 |  14,365,133 |  3,067 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 8,098 |  11,167,961 |  990 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 1,177 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 435 |  112,210 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 586 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-19c0cca2213fefe6db86c2f63d0c2a0a31165461.md) | 3,858 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/19c0cca2213fefe6db86c2f63d0c2a0a31165461

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28270713001)
