| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 3,068 |  12,000,265 |  675 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 16,360 |  18,655,329 |  3,036 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 9,089 |  14,793,960 |  1,110 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 1,167 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 605 |  123,583 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 934 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-b4af713c2a2c06a727778b82f7c560c2296ef209.md) | 4,162 |  2,579,903 |  893 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b4af713c2a2c06a727778b82f7c560c2296ef209

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28405026245)
