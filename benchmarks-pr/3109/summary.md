| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 477 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 7,404 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 4,116 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 663 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 231 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 231 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-99cb3adc6e33be9132d8d65f4df2f55600858df7.md) | 2,033 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/99cb3adc6e33be9132d8d65f4df2f55600858df7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31163012597)
