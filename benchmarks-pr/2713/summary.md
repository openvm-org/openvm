| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/fibonacci-43bdf848b67139c339046290606532f7fdd881c5.md) | 3,858 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/keccak-43bdf848b67139c339046290606532f7fdd881c5.md) | 18,482 |  18,655,329 |  3,304 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/sha2_bench-43bdf848b67139c339046290606532f7fdd881c5.md) | 9,134 |  14,793,960 |  1,412 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/regex-43bdf848b67139c339046290606532f7fdd881c5.md) | 1,418 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/ecrecover-43bdf848b67139c339046290606532f7fdd881c5.md) | 648 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/pairing-43bdf848b67139c339046290606532f7fdd881c5.md) | 911 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/kitchen_sink-43bdf848b67139c339046290606532f7fdd881c5.md) | 2,089 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/43bdf848b67139c339046290606532f7fdd881c5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24578286103)
