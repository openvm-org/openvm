| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 495 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 7,417 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 4,172 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 685 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 230 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-f86059ec57f6231acc892bf1a4a9432b891651be.md) | 2,071 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f86059ec57f6231acc892bf1a4a9432b891651be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30302524867)
