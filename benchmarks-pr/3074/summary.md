| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 472 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 7,300 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 4,153 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 655 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 231 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 239 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-a12277b5f1d69b3456d9b9c53ce38507f0040093.md) | 2,046 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a12277b5f1d69b3456d9b9c53ce38507f0040093

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30498147856)
