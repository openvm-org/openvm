| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 408 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 8,673 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 4,168 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 575 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 294 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-71adc7aece96e30fd50d8712bf36b8ee8c620ce7.md) | 1,932 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/71adc7aece96e30fd50d8712bf36b8ee8c620ce7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29518680179)
