| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 468 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 7,264 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 4,721 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 673 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 231 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 319 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-070f14a6b800fe2c25329ede6eabf1b1dcc3d32d.md) | 2,670 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/070f14a6b800fe2c25329ede6eabf1b1dcc3d32d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30016719810)
