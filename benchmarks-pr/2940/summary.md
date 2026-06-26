| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/fibonacci-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 3,103 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/keccak-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 16,285 |  18,655,329 |  3,022 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/sha2_bench-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 9,188 |  14,793,960 |  1,120 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/regex-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 1,170 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/ecrecover-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 605 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/pairing-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 951 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/kitchen_sink-ef9a208f799a9c152cbb49b5570c5b08dd83524e.md) | 4,105 |  2,579,903 |  877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ef9a208f799a9c152cbb49b5570c5b08dd83524e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28268505556)
