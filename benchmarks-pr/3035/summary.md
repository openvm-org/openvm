| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/fibonacci-42da049292f4050dd6d8e72d506808452714c0e0.md) | 481 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/keccak-42da049292f4050dd6d8e72d506808452714c0e0.md) | 7,346 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/sha2_bench-42da049292f4050dd6d8e72d506808452714c0e0.md) | 4,116 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/regex-42da049292f4050dd6d8e72d506808452714c0e0.md) | 647 |  4,090,656 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/ecrecover-42da049292f4050dd6d8e72d506808452714c0e0.md) | 250 |  78,475 |  224 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/pairing-42da049292f4050dd6d8e72d506808452714c0e0.md) | 228 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/kitchen_sink-42da049292f4050dd6d8e72d506808452714c0e0.md) | 2,345 |  2,341,811 |  561 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/42da049292f4050dd6d8e72d506808452714c0e0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31115178691)
