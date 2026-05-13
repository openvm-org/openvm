| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 1,852 |  4,000,051 |  455 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 13,992 |  14,365,133 |  2,231 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 8,239 |  11,167,961 |  916 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 1,586 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 638 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 755 |  592,827 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-d527075f8c4749dea3f01d315005f7e36f3f7c31.md) | 2,012 |  1,979,971 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d527075f8c4749dea3f01d315005f7e36f3f7c31

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25825356871)
