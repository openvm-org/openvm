| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 1,412 |  4,000,051 |  443 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 13,686 |  14,365,133 |  2,381 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 8,835 |  11,167,961 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 1,508 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 442 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 572 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-c70a3434c6cb767f82a607653ef65b9ff9d37f4d.md) | 3,717 |  1,979,971 |  941 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c70a3434c6cb767f82a607653ef65b9ff9d37f4d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26959213867)
