| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 3,266 |  12,000,265 |  699 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 16,351 |  18,655,329 |  3,006 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 9,327 |  14,793,960 |  1,138 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 1,151 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 600 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 931 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6.md) | 4,146 |  2,579,903 |  887 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f5c3a4553d95c5278e5d2c9be5fea99c5cb945a6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27791078383)
