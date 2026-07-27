| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 463 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 7,273 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 4,756 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 666 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 227 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 266 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-2f918e1b7af9edd1c61c6cd26e67f49a07351c58.md) | 2,752 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2f918e1b7af9edd1c61c6cd26e67f49a07351c58

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30311880299)
