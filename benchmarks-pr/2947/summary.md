| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/fibonacci-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 3,090 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/keccak-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 16,422 |  18,655,329 |  3,053 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/sha2_bench-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 9,112 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/regex-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 1,174 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/ecrecover-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 604 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/pairing-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 955 |  1,745,757 |  313 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/kitchen_sink-3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2.md) | 4,186 |  2,579,903 |  899 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3dc82cea52dc2ee14ef6d979520cadf0b7d6b7c2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28474416642)
