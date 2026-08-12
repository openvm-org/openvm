| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 469 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 7,339 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 4,242 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 688 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 198 |  112,210 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 232 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-2f3e2c848011633d03274b2f4ff2f8558ae3c7cf.md) | 2,036 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2f3e2c848011633d03274b2f4ff2f8558ae3c7cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31551431894)
