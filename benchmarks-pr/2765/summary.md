| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 1,902 |  4,000,051 |  537 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 13,653 |  14,365,133 |  2,249 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 9,527 |  11,167,961 |  1,421 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 1,590 |  4,090,656 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 640 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 754 |  592,827 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994.md) | 2,038 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6ff5dd39ac5b26b26fed64a62e1bcf5a7e692994

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25853620801)
