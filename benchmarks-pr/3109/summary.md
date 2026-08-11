| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 479 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 7,343 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 4,189 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 674 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 222 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 231 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-fc889a05577e8ee727801c4d35512967b7f4875e.md) | 2,027 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fc889a05577e8ee727801c4d35512967b7f4875e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31541250912)
