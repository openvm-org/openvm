| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 411 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 8,770 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 4,261 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 580 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 217 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 296 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-cfa6027c60224ff0d532547507b5ef65ee9dfd1a.md) | 1,915 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cfa6027c60224ff0d532547507b5ef65ee9dfd1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29649615321)
