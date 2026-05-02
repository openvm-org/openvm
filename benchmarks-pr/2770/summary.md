| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 3,802 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 18,599 |  18,655,329 |  3,312 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 8,878 |  14,793,960 |  1,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 1,423 |  4,137,067 |  385 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 634 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 890 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-a7c85a407cf53ae401ac6a79d766f17265f367ff.md) | 2,103 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a7c85a407cf53ae401ac6a79d766f17265f367ff

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25239476274)
