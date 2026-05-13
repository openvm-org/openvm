| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 1,904 |  4,000,051 |  539 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 13,913 |  14,365,133 |  2,289 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 9,530 |  11,167,961 |  1,426 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 1,624 |  4,090,656 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 654 |  112,210 |  297 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 761 |  592,827 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-3ae8395d3080eebf45b41aff2e83f8aff2f408d8.md) | 2,020 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ae8395d3080eebf45b41aff2e83f8aff2f408d8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25822906867)
