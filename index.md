| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-eb7c65e0f427926bde524c65435f9395156de676.md) | 3,002 |  12,000,265 |  659 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-eb7c65e0f427926bde524c65435f9395156de676.md) | 16,385 |  18,655,329 |  3,035 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-eb7c65e0f427926bde524c65435f9395156de676.md) | 9,244 |  14,793,960 |  1,127 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-eb7c65e0f427926bde524c65435f9395156de676.md) | 1,161 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-eb7c65e0f427926bde524c65435f9395156de676.md) | 594 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-eb7c65e0f427926bde524c65435f9395156de676.md) | 941 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-eb7c65e0f427926bde524c65435f9395156de676.md) | 4,195 |  2,579,903 |  904 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eb7c65e0f427926bde524c65435f9395156de676

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29068883699)
