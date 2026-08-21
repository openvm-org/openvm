| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 454 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 7,279 |  14,365,133 |  1,606 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 4,032 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 716 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 207 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 239 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-dc048937ec1a2d84ba648d1ed9a7acd1913df8aa.md) | 2,158 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc048937ec1a2d84ba648d1ed9a7acd1913df8aa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32493529033)
