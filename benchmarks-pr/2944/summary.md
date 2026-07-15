| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 410 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 8,383 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 3,960 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 579 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 216 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 283 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000.md) | 1,890 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c7f14ba1e0d72ee88201f8d5e4e2a9c8f1e43000

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29459051902)
