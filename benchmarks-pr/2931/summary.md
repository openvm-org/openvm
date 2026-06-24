| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 1,031 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 16,215 |  14,365,133 |  3,032 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 8,169 |  11,167,961 |  996 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 1,179 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 431 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 580 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-96524f8c32665b99eb77a98c78167d9e88b339d0.md) | 3,873 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/96524f8c32665b99eb77a98c78167d9e88b339d0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28126162486)
