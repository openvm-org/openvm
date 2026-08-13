| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 468 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 7,368 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 4,177 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 650 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 197 |  112,210 |  200 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 235 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772.md) | 2,050 |  1,979,971 |  530 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f2f5f4e46c95dc3d4a55cca34bfc34f8d75b3772

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31742892035)
