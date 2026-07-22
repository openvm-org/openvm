| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 477 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 7,313 |  14,365,133 |  1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 4,666 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 674 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 229 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 315 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 2,663 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/070f14a6b800fe2c25329ede6eabf1b1dcc3d32d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29944291926)
