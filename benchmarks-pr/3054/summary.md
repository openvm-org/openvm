| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 464 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 7,303 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 4,720 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 668 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 226 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 321 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-266d9f5b7b998b91cfa6fe201f3b43ae3d503c63.md) | 2,667 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/266d9f5b7b998b91cfa6fe201f3b43ae3d503c63

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29934999071)
