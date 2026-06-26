| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 3,061 |  12,000,265 |  674 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 16,335 |  18,655,329 |  3,032 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 9,187 |  14,793,960 |  1,120 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 1,176 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 600 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 933 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-7a9526cee8faf5e054d06a3ab7e01289030ccace.md) | 4,104 |  2,579,903 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a9526cee8faf5e054d06a3ab7e01289030ccace

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28265642117)
