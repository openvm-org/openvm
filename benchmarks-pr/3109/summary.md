| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 471 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 7,316 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 4,116 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 661 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 227 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 230 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9.md) | 2,041 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c2f44c81e9f8a0ab0d8b560411d7b7552ec0a3c9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31040173723)
