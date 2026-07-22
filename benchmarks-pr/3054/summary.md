| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 464 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 7,237 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 4,725 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 674 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 329 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7.md) | 2,629 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a79cc60d7bab0cdd7abc7c853f3b5f08846acb7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29951472456)
