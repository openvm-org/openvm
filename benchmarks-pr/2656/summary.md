| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/fibonacci-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 3,874 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/keccak-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 15,642 |  1,235,218 |  2,188 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/regex-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 1,407 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/ecrecover-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 639 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/pairing-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 918 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/kitchen_sink-2c539ff9ada1b7d394996047030d5697f1a6f2d7.md) | 2,376 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c539ff9ada1b7d394996047030d5697f1a6f2d7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23922251039)
