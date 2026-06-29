| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-879111497b10677d46fb0df6f113b3878b195796.md) | 1,021 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-879111497b10677d46fb0df6f113b3878b195796.md) | 15,645 |  14,365,133 |  3,060 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-879111497b10677d46fb0df6f113b3878b195796.md) | 7,775 |  11,167,961 |  991 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-879111497b10677d46fb0df6f113b3878b195796.md) | 1,018 |  4,090,656 |  300 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-879111497b10677d46fb0df6f113b3878b195796.md) | 428 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-879111497b10677d46fb0df6f113b3878b195796.md) | 546 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-879111497b10677d46fb0df6f113b3878b195796.md) | 3,767 |  1,979,971 |  866 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/879111497b10677d46fb0df6f113b3878b195796

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28386026704)
