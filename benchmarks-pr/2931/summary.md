| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 1,015 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 16,060 |  14,365,133 |  2,999 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 8,047 |  11,167,961 |  992 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 1,173 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 433 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 590 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-1eef51c49277daf013846d83443e08dc0c61fcd9.md) | 3,905 |  1,979,971 |  872 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1eef51c49277daf013846d83443e08dc0c61fcd9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28160146828)
