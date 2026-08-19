| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 449 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 7,323 |  14,365,133 |  1,610 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 4,092 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 705 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 208 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 239 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-095fb12347cbdbbf307763b30bec1a12432d1eb0.md) | 2,175 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/095fb12347cbdbbf307763b30bec1a12432d1eb0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32315192398)
