| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/fibonacci-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 3,754 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/keccak-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 18,244 |  18,655,329 |  3,311 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/sha2_bench-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 10,129 |  14,793,960 |  1,473 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/regex-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 1,400 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/ecrecover-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 595 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/pairing-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 892 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/kitchen_sink-d2f1f572b44e239daa38d62b10936c7306f5e45c.md) | 3,878 |  2,579,903 |  957 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d2f1f572b44e239daa38d62b10936c7306f5e45c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27168128783)
