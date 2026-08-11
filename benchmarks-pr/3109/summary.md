| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 480 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 7,357 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 4,176 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 671 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 224 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 231 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-3ea1dff6130f784195fc5f77a107e1dfc70f590f.md) | 2,046 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ea1dff6130f784195fc5f77a107e1dfc70f590f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31541722552)
