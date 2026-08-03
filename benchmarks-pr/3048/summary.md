| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-930c2d6acf48f81ada97974af845f3e84050b630.md) | 476 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-930c2d6acf48f81ada97974af845f3e84050b630.md) | 7,461 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-930c2d6acf48f81ada97974af845f3e84050b630.md) | 4,101 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-930c2d6acf48f81ada97974af845f3e84050b630.md) | 664 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-930c2d6acf48f81ada97974af845f3e84050b630.md) | 230 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-930c2d6acf48f81ada97974af845f3e84050b630.md) | 238 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-930c2d6acf48f81ada97974af845f3e84050b630.md) | 2,058 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/930c2d6acf48f81ada97974af845f3e84050b630

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30808263346)
