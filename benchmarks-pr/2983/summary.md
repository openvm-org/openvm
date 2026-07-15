| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 408 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 8,459 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 3,962 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 566 |  4,090,656 |  209 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 273 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-24892c7231bd73730432ba96caa8e4740a69e6f5.md) | 1,905 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/24892c7231bd73730432ba96caa8e4740a69e6f5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29415946779)
