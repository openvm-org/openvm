| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/fibonacci-654402dc1168eac20f3078c5c25074c364634dd8.md) | 1,035 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/keccak-654402dc1168eac20f3078c5c25074c364634dd8.md) | 16,294 |  14,365,133 |  3,027 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/sha2_bench-654402dc1168eac20f3078c5c25074c364634dd8.md) | 8,152 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/regex-654402dc1168eac20f3078c5c25074c364634dd8.md) | 1,216 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/ecrecover-654402dc1168eac20f3078c5c25074c364634dd8.md) | 442 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/pairing-654402dc1168eac20f3078c5c25074c364634dd8.md) | 604 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/kitchen_sink-654402dc1168eac20f3078c5c25074c364634dd8.md) | 3,899 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/654402dc1168eac20f3078c5c25074c364634dd8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27946869577)
