| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/fibonacci-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 504 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/keccak-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 7,718 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/sha2_bench-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 4,560 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/regex-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 719 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/ecrecover-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 202 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/pairing-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 243 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/kitchen_sink-37f6cb736472805a94cf511f112ab198ad7654d9.md) | 2,125 |  1,979,971 |  478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/37f6cb736472805a94cf511f112ab198ad7654d9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32528461617)
