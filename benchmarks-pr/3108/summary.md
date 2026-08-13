| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 451 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 7,113 |  14,365,133 |  1,594 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 4,139 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 727 |  4,090,656 |  229 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 218 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 236 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-983214f682bdd8418c6760a2c5f148acedb08fd9.md) | 2,169 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/983214f682bdd8418c6760a2c5f148acedb08fd9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31752015199)
