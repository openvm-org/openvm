| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/fibonacci-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 892 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/keccak-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 15,461 |  14,365,133 |  2,998 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/sha2_bench-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 8,227 |  11,167,961 |  1,019 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/regex-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 1,200 |  4,090,656 |  364 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/ecrecover-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 437 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/pairing-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 579 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/kitchen_sink-eb3f6fdc56150d781e874b4200f527a8d8de3537.md) | 3,808 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eb3f6fdc56150d781e874b4200f527a8d8de3537

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28822168106)
