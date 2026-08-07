| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 475 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 7,353 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 4,224 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 669 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 224 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 232 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-35e06fb1c391b8b46d26a701259e48fe4738ebaa.md) | 2,027 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/35e06fb1c391b8b46d26a701259e48fe4738ebaa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31208271427)
