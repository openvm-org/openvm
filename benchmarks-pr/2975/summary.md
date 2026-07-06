| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/fibonacci-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 873 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/keccak-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 15,481 |  14,365,133 |  3,054 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/sha2_bench-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 8,067 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/regex-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 1,032 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/ecrecover-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 300 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/pairing-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 451 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/kitchen_sink-d463085f7ed3d35e0875fbc7d5f350a582575610.md) | 3,758 |  1,979,971 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d463085f7ed3d35e0875fbc7d5f350a582575610

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28828538739)
