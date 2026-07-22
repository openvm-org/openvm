| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 475 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 7,262 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 4,685 |  11,167,961 |  537 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 675 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 311 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-581cadfdc1443f57cbcf2100e7622cb738519033.md) | 2,678 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/581cadfdc1443f57cbcf2100e7622cb738519033

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29951462077)
