| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 1,878 |  4,000,051 |  512 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 13,709 |  14,365,133 |  2,231 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 9,663 |  11,167,961 |  1,455 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 1,575 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 604 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 742 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-de73e22d80e2d5770e9a215858dd09cc6e930036.md) | 1,868 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/de73e22d80e2d5770e9a215858dd09cc6e930036

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26252448712)
