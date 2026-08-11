| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 474 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 7,349 |  14,365,133 |  1,505 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 4,210 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 670 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 200 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 233 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-c144dcbae4b38d6040ef069626cfa4c9aad72c41.md) | 2,037 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c144dcbae4b38d6040ef069626cfa4c9aad72c41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31533754191)
