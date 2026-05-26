| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/fibonacci-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 1,887 |  4,000,051 |  516 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/keccak-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 13,591 |  14,365,133 |  2,223 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/sha2_bench-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 9,530 |  11,167,961 |  1,417 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/regex-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 1,552 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/ecrecover-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 603 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/pairing-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 746 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/kitchen_sink-323ca3228e1fcc654aa8cea2fb0eb8ca503fc279.md) | 1,870 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/323ca3228e1fcc654aa8cea2fb0eb8ca503fc279

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26477075142)
