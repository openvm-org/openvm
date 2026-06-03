| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 1,582 |  4,000,051 |  438 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 14,036 |  14,365,133 |  2,377 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 9,212 |  11,167,961 |  1,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 1,617 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 482 |  112,210 |  263 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 609 |  592,827 |  247 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-03e874ed516696502064dc2017d04a5f91fea9c1.md) | 1,809 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/03e874ed516696502064dc2017d04a5f91fea9c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26855311134)
