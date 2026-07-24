| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 470 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 7,325 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 4,686 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 669 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 231 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 329 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-7c7d951d304093d61fb0bfe0eb1c4516a8f66b58.md) | 2,675 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7c7d951d304093d61fb0bfe0eb1c4516a8f66b58

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30130572611)
