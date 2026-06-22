| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 1,038 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 16,320 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 8,284 |  11,167,961 |  1,015 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 1,179 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 440 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 602 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-e356e7a15d536b7e27fd4df816f5876063f44046.md) | 3,846 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e356e7a15d536b7e27fd4df816f5876063f44046

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27980806800)
