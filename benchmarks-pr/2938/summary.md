| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/fibonacci-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 1,018 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/keccak-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 15,630 |  14,365,133 |  2,999 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/sha2_bench-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 8,115 |  11,167,961 |  996 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/regex-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 1,173 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/ecrecover-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 434 |  112,210 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/pairing-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 596 |  592,827 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2938/kitchen_sink-b3af48a198d26400ff5fbb69bc64f1e1451b8746.md) | 3,852 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b3af48a198d26400ff5fbb69bc64f1e1451b8746

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28261172158)
