| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 482 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 7,407 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 4,165 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 665 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 227 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 233 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-602d484c38e358fdd7b88b867fe45b612362b63c.md) | 2,044 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/602d484c38e358fdd7b88b867fe45b612362b63c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31433988720)
