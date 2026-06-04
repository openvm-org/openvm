| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/fibonacci-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 1,573 |  4,000,051 |  437 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/keccak-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 14,028 |  14,365,133 |  2,383 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/sha2_bench-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 9,051 |  11,167,961 |  1,392 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/regex-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 1,572 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/ecrecover-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 478 |  112,210 |  260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/pairing-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 610 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/kitchen_sink-0557b57166a4eb4473724d1cbc4d32cdcfa53ced.md) | 2,002 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0557b57166a4eb4473724d1cbc4d32cdcfa53ced

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26938871448)
