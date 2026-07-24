| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/fibonacci-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 472 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/keccak-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 7,257 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/sha2_bench-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 4,621 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/regex-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 671 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/ecrecover-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 285 |  78,475 |  225 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/pairing-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 323 |  592,827 |  200 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/kitchen_sink-39e7e1f919c03dd6fc86646a8db108646fc2b485.md) | 3,004 |  2,341,811 |  556 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/39e7e1f919c03dd6fc86646a8db108646fc2b485

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30120166716)
