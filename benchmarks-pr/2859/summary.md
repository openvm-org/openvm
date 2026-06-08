| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/fibonacci-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 3,738 |  12,000,265 |  924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/keccak-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 18,071 |  18,655,329 |  3,276 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/sha2_bench-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 9,983 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/regex-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 1,404 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/ecrecover-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 595 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/pairing-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 880 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/kitchen_sink-094e780c1ac157e5cd3af8ece725dcec137b06ec.md) | 3,879 |  2,579,903 |  955 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/094e780c1ac157e5cd3af8ece725dcec137b06ec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27169101120)
