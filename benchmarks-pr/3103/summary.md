| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 461 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 7,312 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 4,164 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 650 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 197 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 240 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b.md) | 2,030 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f43b3c8a7c6354cbed0c7f131943b8c0a9606d4b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31715863426)
