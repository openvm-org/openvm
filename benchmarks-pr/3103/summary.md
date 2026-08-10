| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 464 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 7,425 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 4,147 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 656 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 223 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 232 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23.md) | 2,046 |  1,979,971 |  533 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/207bfc7ebdc06de74cdf49b1e79ec5a8e875ad23

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31437192471)
