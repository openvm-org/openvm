| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 3,809 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 18,475 |  18,655,329 |  3,283 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 8,943 |  14,793,960 |  1,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 1,431 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 639 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 909 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-c64bdafab1c681b9b089b8809a03bf5f4e26f136.md) | 2,084 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c64bdafab1c681b9b089b8809a03bf5f4e26f136

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25694988051)
