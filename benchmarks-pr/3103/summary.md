| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 464 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 7,447 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 4,220 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 671 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 195 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 232 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79.md) | 2,012 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/df8c9c2071e88cf5e201cb9ddef3d7f37ed99c79

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31726260091)
