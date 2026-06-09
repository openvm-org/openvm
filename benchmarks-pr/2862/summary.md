| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/fibonacci-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 3,969 |  12,000,265 |  1,148 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/keccak-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 22,181 |  18,655,329 |  4,696 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/sha2_bench-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 9,545 |  14,793,960 |  1,829 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/regex-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 1,499 |  4,137,067 |  427 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/ecrecover-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 606 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/pairing-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 926 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/kitchen_sink-42b2af2d787b03d3fc0bf1269babbe9eda4056ca.md) | 4,154 |  2,579,903 |  891 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/42b2af2d787b03d3fc0bf1269babbe9eda4056ca

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27234135895)
