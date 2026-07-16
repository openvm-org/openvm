| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-869217604d0b5f88ff5122b063fca550d35a0213.md) | 412 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-869217604d0b5f88ff5122b063fca550d35a0213.md) | 8,743 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-869217604d0b5f88ff5122b063fca550d35a0213.md) | 4,192 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-869217604d0b5f88ff5122b063fca550d35a0213.md) | 574 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-869217604d0b5f88ff5122b063fca550d35a0213.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-869217604d0b5f88ff5122b063fca550d35a0213.md) | 287 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-869217604d0b5f88ff5122b063fca550d35a0213.md) | 1,949 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/869217604d0b5f88ff5122b063fca550d35a0213

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29461432749)
